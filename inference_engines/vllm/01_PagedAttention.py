"""
01_PagedAttention.py
====================
vLLM 核心创新:受 OS 虚拟内存启发的分页注意力机制。

核心思想:
    1. 将 KV Cache 划分为固定大小的 block(默认 16 token/block)
    2. 每个 sequence 维护一个 Block Table: logical_block_id -> physical_block_id
    3. Attention kernel 通过 block table 间接寻址,消除显存碎片

对比传统方案:
    传统: 每个序列申请连续 KV 显存 -> 内部碎片 + 外部碎片
    PagedAttention: block 级分配,显存利用率接近 100%

本文包含:
    - BlockSpaceManager: block 分配器
    - paged_attention_kernel: Triton 实现的分页注意力内核
    - 演示用例
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch
import triton
import triton.language as tl


# ============================================================
# 1. 数据结构:Block 与 Block Table
# ============================================================

@dataclass
class KVCacheBlock:
    """一个物理 KV block,固定大小 block_size 个 token 的 K/V"""
    block_id: int                    # 物理 block 编号
    ref_count: int = 0               # 引用计数(共享前缀时 >1)
    hash: Optional[int] = None       # block 内容 hash(用于 prefix caching)

    def acquire(self):
        self.ref_count += 1

    def release(self) -> bool:
        """返回 True 表示可回收"""
        self.ref_count -= 1
        return self.ref_count == 0


@dataclass
class Sequence:
    """一个推理序列(请求)"""
    seq_id: int
    token_ids: List[int]
    block_table: List[int] = field(default_factory=list)  # logical -> physical
    block_size: int = 16

    def num_tokens(self) -> int:
        return len(self.token_ids)

    def num_blocks(self) -> int:
        return (len(self.token_ids) + self.block_size - 1) // self.block_size

    def ensure_blocks(self, allocator: "BlockSpaceManager"):
        """按需扩容 block"""
        needed = self.num_blocks()
        while len(self.block_table) < needed:
            blk = allocator.alloc()
            self.block_table.append(blk.block_id)
            blk.acquire()


# ============================================================
# 2. BlockSpaceManager:block 分配器
# ============================================================

class BlockSpaceManager:
    """
    管理一个 GPU 上的物理 KV block 池。
    对应 vLLM 的 BlockSpaceManagerV1 / V2。
    """

    def __init__(self,
                 num_blocks: int,
                 block_size: int,
                 num_kv_heads: int,
                 head_dim: int,
                 dtype: torch.dtype = torch.float16,
                 device: str = "cuda"):
        self.block_size = block_size
        self.num_blocks = num_blocks

        # 物理显存池: [num_blocks, block_size, num_kv_heads, head_dim]
        # 所有 block 共享两个大 tensor,通过 index 寻址
        self.k_cache = torch.zeros(
            (num_blocks, block_size, num_kv_heads, head_dim),
            dtype=dtype, device=device
        )
        self.v_cache = torch.zeros_like(self.k_cache)

        # 空闲 block 栈(后进先出,便于 LRU)
        self.free_blocks: List[KVCacheBlock] = [
            KVCacheBlock(block_id=i) for i in range(num_blocks)
        ]
        self.allocated: Dict[int, KVCacheBlock] = {}

    def alloc(self) -> KVCacheBlock:
        if not self.free_blocks:
            raise RuntimeError("OOM: no free KV blocks")
        blk = self.free_blocks.pop()
        self.allocated[blk.block_id] = blk
        return blk

    def free(self, block_id: int):
        blk = self.allocated.pop(block_id)
        if blk.release():
            self.free_blocks.append(blk)

    def num_free_blocks(self) -> int:
        return len(self.free_blocks)


# ============================================================
# 3. PagedAttention Triton Kernel
# ============================================================

@triton.jit
def _paged_attention_kernel(
    Q_ptr,            # [num_seqs, num_heads, head_dim]
    K_cache_ptr,      # [num_blocks, block_size, num_kv_heads, head_dim]
    V_cache_ptr,      # [num_blocks, block_size, num_kv_heads, head_dim]
    block_table_ptr,  # [num_seqs, max_num_blocks_per_seq]
    out_ptr,          # [num_seqs, num_heads, head_dim]
    seq_lens_ptr,     # [num_seqs]
    scale: tl.float32,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_blocks_per_seq: tl.constexpr,
):
    """
    简化版 PagedAttention kernel:
        遍历该 sequence 的所有 block,对每个 block 内做 attention,累加输出。
    实际 vLLM 还做了 split-K 并行,这里为可读性省略。
    """
    seq_id = tl.program_id(0)
    head_id = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + seq_id)
    num_blocks_for_seq = (seq_len + block_size - 1) // block_size

    # Q 向量 [head_dim]
    dim_offsets = tl.arange(0, head_dim)
    q = tl.load(Q_ptr + seq_id * num_kv_heads * head_dim
                + head_id * head_dim + dim_offsets)  # [head_dim]

    # 累加器
    acc = tl.zeros([head_dim], dtype=tl.float32)
    max_score = tl.full([1], -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros([1], dtype=tl.float32)

    # 遍历所有 block
    for blk_idx in range(num_blocks_for_seq):
        # 通过 block table 找到物理 block
        phys_blk = tl.load(block_table_ptr + seq_id * max_blocks_per_seq + blk_idx)

        # block 内的 token offsets
        token_offsets = tl.arange(0, block_size)
        global_token_pos = blk_idx * block_size + token_offsets

        # mask 超出 seq_len 的位置
        valid_mask = global_token_pos < seq_len

        # 加载 K [block_size, head_dim]
        k_offsets = (phys_blk * block_size * num_kv_heads * head_dim
                     + token_offsets[:, None] * num_kv_heads * head_dim
                     + head_id * head_dim
                     + dim_offsets[None, :])
        k = tl.load(K_cache_ptr + k_offsets, mask=valid_mask[:, None], other=0.0)

        # attention score = Q @ K^T * scale
        scores = tl.sum(q[None, :] * k, axis=1) * scale  # [block_size]
        scores = tl.where(valid_mask, scores, -float("inf"))

        # online softmax (Numerical Stable)
        cur_max = tl.max(scores)
        new_max = tl.maximum(max_score, cur_max)
        alpha = tl.exp(max_score - new_max)
        beta = tl.exp(cur_max - new_max)

        # 加载 V 并加权累加
        v_offsets = (phys_blk * block_size * num_kv_heads * head_dim
                     + token_offsets[:, None] * num_kv_heads * head_dim
                     + head_id * head_dim
                     + dim_offsets[None, :])
        v = tl.load(V_cache_ptr + v_offsets, mask=valid_mask[:, None], other=0.0)

        sum_exp = sum_exp * alpha + tl.sum(beta)
        acc = acc * alpha + tl.sum(beta[:, None] * v, axis=0)
        max_score = new_max

    # 归一化
    out = acc / sum_exp
    tl.store(out_ptr + seq_id * num_kv_heads * head_dim
             + head_id * head_dim + dim_offsets, out.to(out_ptr.dtype.element_ty))


def paged_attention(q: torch.Tensor,
                    k_cache: torch.Tensor,
                    v_cache: torch.Tensor,
                    block_tables: torch.Tensor,
                    seq_lens: torch.Tensor,
                    scale: float) -> torch.Tensor:
    """
    q: [num_seqs, num_heads, head_dim]
    k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
    block_tables: [num_seqs, max_blocks_per_seq]  int32
    seq_lens: [num_seqs]  int32
    return: [num_seqs, num_heads, head_dim]
    """
    num_seqs, num_heads, head_dim = q.shape
    num_blocks, block_size, num_kv_heads, _ = k_cache.shape
    max_blocks_per_seq = block_tables.shape[1]

    out = torch.empty_like(q)
    grid = (num_seqs, num_heads)
    _paged_attention_kernel[grid](
        q, k_cache, v_cache, block_tables, out, seq_lens,
        scale=scale,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_blocks_per_seq=max_blocks_per_seq,
    )
    return out


# ============================================================
# 4. 演示用例
# ============================================================

def demo():
    torch.manual_seed(42)
    num_blocks, block_size = 128, 16
    num_kv_heads, head_dim = 8, 64
    dtype = torch.float16

    mgr = BlockSpaceManager(num_blocks, block_size, num_kv_heads, head_dim, dtype)

    # 两个序列
    seq0 = Sequence(seq_id=0, token_ids=list(range(40)))  # 40 token -> 3 block
    seq1 = Sequence(seq_id=1, token_ids=list(range(30)))
    seq0.ensure_blocks(mgr)
    seq1.ensure_blocks(mgr)

    print(f"Seq0 block_table: {seq0.block_table}")
    print(f"Seq1 block_table: {seq1.block_table}")
    print(f"Free blocks: {mgr.num_free_blocks()}")

    # 模拟 Q
    q = torch.randn(2, num_kv_heads, head_dim, dtype=dtype, device="cuda")
    block_tables = torch.tensor([seq0.block_table + [0]*8,
                                 seq1.block_table + [0]*8],
                                dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([40, 30], dtype=torch.int32, device="cuda")

    out = paged_attention(q, mgr.k_cache, mgr.v_cache, block_tables, seq_lens,
                          scale=1.0 / (head_dim ** 0.5))
    print(f"Output shape: {out.shape}")


if __name__ == "__main__":
    demo()
