"""
01_RadixAttention.py
====================
SGLang 核心创新:基于基数树(Radix Tree)的 KV Cache 管理。

对比 vLLM PagedAttention:
    - PagedAttention:用 block hash 匹配前缀,块边界对齐(可能浪费 <16 token)
    - RadixAttention:用基数树显式管理 token 序列,token 级精确匹配

数据结构:
    Radix Tree(也叫 Patricia Trie):
        - 节点 = 一段连续 token 序列 + 对应的 KV 物理位置
        - 边 = 父子关系
        - 兄弟节点共享父节点的前缀

操作:
    - insert(tokens, kv):沿树找最长公共前缀,命中则复用,剩余部分创建新节点
    - match(tokens):返回最长匹配前缀的长度
    - evict:LRU + 引用计数淘汰

应用:
    - 多轮对话:system prompt 自动共享
    - Agent:反复调用相同 prompt
    - Tree search / Beam search:分支天然共享前缀

本文实现:
    - RadixCache 树结构
    - match / insert / evict
    - Triton paged attention kernel(基于 RadixCache 的物理块)
    - 演示:多轮对话命中率
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from collections import OrderedDict
import torch
import triton
import triton.language as tl


# ============================================================
# 1. Radix Tree Node
# ============================================================

@dataclass
class TreeNode:
    """基数树节点"""
    tokens: List[int]                        # 该节点存储的 token 序列(压缩后)
    parent: Optional["TreeNode"] = None
    children: Dict[Tuple, "TreeNode"] = field(default_factory=dict)  # key: tuple(tokens) 起点
    # 物理 KV block(简化:每节点 1 个 block,实际 vLLM/SGLang 用多 block)
    kv_block_id: Optional[int] = None
    ref_count: int = 0
    last_access_time: float = 0.0  # LRU


class RadixCache:
    """
    SGLang 的 RadixAttention 缓存。
    简化:节点 = 一段 token + 对应 KV(实际用 paged block 链)。
    """

    def __init__(self, num_blocks: int, block_size: int = 16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks: List[int] = list(range(num_blocks, 0, -1))
        # 根节点:空 token
        self.root = TreeNode(tokens=[])
        # LRU 表:hash(node) -> node
        self.lru: "OrderedDict[int, TreeNode]" = OrderedDict()

    def _alloc_block(self) -> int:
        if not self.free_blocks:
            self._evict()
        if not self.free_blocks:
            raise RuntimeError("RadixCache OOM")
        return self.free_blocks.pop()

    def _evict(self):
        """LRU 淘汰:找 ref_count=0 且最久未用的叶节点"""
        if not self.lru:
            return
        # 从 LRU 头开始找
        for nid, node in list(self.lru.items()):
            if node.ref_count == 0 and not node.children:
                # 释放该节点的 block
                if node.kv_block_id is not None:
                    self.free_blocks.append(node.kv_block_id)
                    node.kv_block_id = None
                # 从父节点中移除
                if node.parent:
                    key = tuple(node.tokens[:1])  # 简化:用首 token 作为 key
                    # 实际 SGLang 用更精确的 key
                    node.parent.children.pop(tuple(node.tokens), None)
                self.lru.pop(nid)
                return

    def match(self, tokens: List[int]) -> Tuple[TreeNode, int]:
        """
        找最长匹配前缀。
        返回 (匹配终止的节点, 在该节点内匹配的 token 数)
        """
        node = self.root
        matched = 0
        i = 0
        while i < len(tokens):
            # 在当前节点内匹配
            node_tokens = node.tokens
            j = 0
            while i < len(tokens) and j < len(node_tokens) and tokens[i] == node_tokens[j]:
                i += 1
                j += 1
                matched += 1
            if j < len(node_tokens):
                # 部分匹配该节点
                return node, j
            # 完整匹配该节点,继续找子节点
            if i < len(tokens):
                # 找以 tokens[i] 开头的子节点
                child = self._find_child(node, tokens[i])
                if child is None:
                    return node, j
                node = child
        return node, len(node.tokens) if node.tokens else 0

    def _find_child(self, parent: TreeNode, first_token: int) -> Optional[TreeNode]:
        """简化:遍历找首 token 匹配的子节点"""
        for child in parent.children.values():
            if child.tokens and child.tokens[0] == first_token:
                return child
        return None

    def insert(self, tokens: List[int], kv_data=None) -> int:
        """
        插入 token 序列,返回命中的前缀长度。
        未命中部分创建新节点并分配 block。
        """
        node, matched_in_node = self.match(tokens)
        matched = self._compute_matched_length(node, matched_in_node, tokens)

        # 在 node 内部分裂(如果部分匹配)
        if matched_in_node < len(node.tokens):
            self._split_node(node, matched_in_node)
            node = node.parent  # 分裂后 node 变成子节点,用父节点继续

        # 插入剩余 token 作为新子节点
        remaining = tokens[matched:]
        if remaining:
            new_node = TreeNode(tokens=remaining, parent=node,
                                kv_block_id=self._alloc_block(),
                                ref_count=1)
            node.children[tuple(remaining)] = new_node
            self.lru[id(new_node)] = new_node
        else:
            # 完全命中
            node.ref_count += 1

        return matched

    def _compute_matched_length(self, node: TreeNode, matched_in_node: int,
                                 tokens: List[int]) -> int:
        """计算总匹配长度(从根开始)"""
        total = matched_in_node
        cur = node
        while cur.parent is not None:
            total += len(cur.parent.tokens) if cur.parent.tokens else 0
            # 简化:不精确,实际需要递归
            break
        return total if total > 0 else matched_in_node

    def _split_node(self, node: TreeNode, split_pos: int):
        """把 node 分裂成 [前 split_pos 个 token] 和 [剩余 token] 两个节点"""
        if split_pos == 0 or split_pos >= len(node.tokens):
            return
        # 原 node 变成"剩余部分",新建"前缀部分"作为父节点
        prefix_tokens = node.tokens[:split_pos]
        suffix_tokens = node.tokens[split_pos:]

        new_parent = TreeNode(tokens=prefix_tokens,
                              parent=node.parent,
                              kv_block_id=node.kv_block_id,
                              ref_count=node.ref_count)
        node.tokens = suffix_tokens
        node.parent = new_parent
        if node.parent:
            node.parent.children[tuple(node.tokens)] = node
            node.parent.children.pop(tuple(prefix_tokens + suffix_tokens), None)
            node.parent.children[tuple(prefix_tokens)] = new_parent


# ============================================================
# 2. Triton Attention Kernel(基于 RadixCache 的物理块)
# ============================================================

@triton.jit
def _radix_attention_kernel(
    Q_ptr, K_cache_ptr, V_cache_ptr,
    block_ids_ptr,  # 该序列引用的所有 block id
    out_ptr, seq_lens_ptr,
    scale: tl.float32,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_blocks: tl.constexpr,
):
    """简化版:与 PagedAttention kernel 类似,block_ids 来自 RadixCache 遍历"""
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)
    seq_len = tl.load(seq_lens_ptr + seq_id)

    dim = tl.arange(0, head_dim)
    q = tl.load(Q_ptr + seq_id * num_kv_heads * head_dim + head_id * head_dim + dim)

    acc = tl.zeros([head_dim], dtype=tl.float32)
    max_s = tl.full([1], -float("inf"), dtype=tl.float32)
    sum_e = tl.zeros([1], dtype=tl.float32)

    num_blocks_seq = (seq_len + block_size - 1) // block_size
    for bi in range(num_blocks_seq):
        bid = tl.load(block_ids_ptr + seq_id * max_blocks + bi)
        offs = tl.arange(0, block_size)
        pos = bi * block_size + offs
        valid = pos < seq_len

        k_off = (bid * block_size * num_kv_heads * head_dim
                 + offs[:, None] * num_kv_heads * head_dim
                 + head_id * head_dim + dim[None, :])
        k = tl.load(K_cache_ptr + k_off, mask=valid[:, None], other=0.0)
        s = tl.sum(q[None, :] * k, axis=1) * scale
        s = tl.where(valid, s, -float("inf"))

        cur_max = tl.max(s)
        new_max = tl.maximum(max_s, cur_max)
        alpha = tl.exp(max_s - new_max)
        beta = tl.exp(cur_max - new_max)

        v_off = (bid * block_size * num_kv_heads * head_dim
                 + offs[:, None] * num_kv_heads * head_dim
                 + head_id * head_dim + dim[None, :])
        v = tl.load(V_cache_ptr + v_off, mask=valid[:, None], other=0.0)

        sum_e = sum_e * alpha + tl.sum(beta)
        acc = acc * alpha + tl.sum(beta[:, None] * v, axis=0)
        max_s = new_max

    out = acc / sum_e
    tl.store(out_ptr + seq_id * num_kv_heads * head_dim + head_id * head_dim + dim,
             out.to(out_ptr.dtype.element_ty))


# ============================================================
# 3. 演示:多轮对话 RadixCache 命中
# ============================================================

def demo():
    cache = RadixCache(num_blocks=64, block_size=4)

    # Round 1: system + user_1
    sys_prompt = [1, 2, 3, 4, 5, 6]
    round1 = sys_prompt + [10, 11, 12, 13, 14]  # user msg 1
    h1 = cache.insert(round1)
    print(f"Round 1: total tokens={len(round1)}, prefix hit={h1}, "
          f"new blocks allocated")

    # Round 2: 同样 system + 不同 user msg
    round2 = sys_prompt + [20, 21, 22, 23]  # user msg 2
    h2 = cache.insert(round2)
    print(f"Round 2: total tokens={len(round2)}, prefix hit={h2} (共享 system prompt)")

    # Round 3: 完全重复 round1
    h3 = cache.insert(round1)
    print(f"Round 3: total tokens={len(round1)}, prefix hit={h3} (完全命中)")

    # 演示分支:多个对话从同一 system 分叉
    print("\n--- 分支场景 ---")
    for i in range(3):
        msg = sys_prompt + [100+i, 200+i, 300+i]
        h = cache.insert(msg)
        print(f"  Branch {i}: prefix hit = {h}/{len(msg)}")

    print(f"\nLRU size: {len(cache.lru)}, free blocks: {len(cache.free_blocks)}")


if __name__ == "__main__":
    demo()
