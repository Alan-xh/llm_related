"""
04_PrefixCaching.py
===================
vLLM Automatic Prefix Caching (APC):基于 block hash 自动复用共享前缀的 KV。

核心机制:
    1. 每个 block 计算一个 hash:
         hash = sha256(token_ids_in_block, parent_block_hash)
       即 hash 链式依赖父 block,保证前缀相同则 hash 相同。
    2. 全局维护 {block_hash -> physical_block} 表(LRU 淘汰)
    3. prefill 时,若 block hash 已存在,直接复用物理 block(refcount++)
       跳过该 block 的 prefill 计算

适用场景:
    - 多轮对话(system + history 共享)
    - Few-shot 多样本
    - Agent 反复调用相同 prompt 前缀

本文实现:
    - BlockHash 计算
    - PrefixCache 查找与复用
    - 演示:多轮对话场景的命中率
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict
import hashlib


# ============================================================
# 1. Block Hash 计算
# ============================================================

def compute_block_hash(token_ids: List[int],
                       parent_hash: Optional[int]) -> int:
    """
    block hash = hash(parent_hash || token_ids)
    链式依赖父 block,保证相同前缀 -> 相同 hash。
    vLLM 实际用更轻量的 hash(基于 tuple hash),这里用 sha256 演示。
    """
    h = hashlib.sha256()
    if parent_hash is not None:
        h.update(parent_hash.to_bytes(8, "little", signed=False))
    for t in token_ids:
        h.update(t.to_bytes(4, "little", signed=False))
    return int.from_bytes(h.digest()[:8], "little")


# ============================================================
# 2. 物理块管理
# ============================================================

@dataclass
class CachedBlock:
    block_id: int
    hash: int
    ref_count: int = 0
    token_ids: List[int] = field(default_factory=list)


class PrefixCache:
    """
    hash -> physical_block 的 LRU 缓存。
    对应 vLLM 的 LRUCache + BlockAllocator。
    """

    def __init__(self, num_blocks: int, block_size: int = 16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_ids: List[int] = list(range(num_blocks, 0, -1))  # 栈
        # hash -> CachedBlock, OrderedDict 实现 LRU
        self.cache: "OrderedDict[int, CachedBlock]" = OrderedDict()

    def _evict_one(self):
        """LRU 淘汰一个 refcount=0 的 block"""
        for h, blk in self.cache.items():
            if blk.ref_count == 0:
                self.cache.pop(h)
                self.free_ids.append(blk.block_id)
                return

    def alloc_new(self, h: int, token_ids: List[int]) -> CachedBlock:
        if not self.free_ids:
            self._evict_one()
            if not self.free_ids:
                raise RuntimeError("OOM: prefix cache full")
        bid = self.free_ids.pop()
        blk = CachedBlock(block_id=bid, hash=h, token_ids=token_ids)
        self.cache[h] = blk
        self.cache.move_to_end(h)  # 标记为最近使用
        return blk

    def lookup(self, h: int) -> Optional[CachedBlock]:
        blk = self.cache.get(h)
        if blk is not None:
            self.cache.move_to_end(h)
        return blk

    def acquire(self, blk: CachedBlock):
        blk.ref_count += 1

    def release(self, blk: CachedBlock):
        blk.ref_count -= 1
        # 不立即释放,留给 LRU 淘汰


# ============================================================
# 3. Sequence + Prefix Cache 复用
# ============================================================

@dataclass
class Sequence:
    seq_id: int
    token_ids: List[int]
    block_size: int = 16
    block_table: List[int] = field(default_factory=list)
    block_hashes: List[int] = field(default_factory=list)
    num_computed: int = 0  # 已 prefill 的 token 数(已命中或已计算)

    def num_blocks(self) -> int:
        return (len(self.token_ids) + self.block_size - 1) // self.block_size


def find_prefix_hits(seq: Sequence, cache: PrefixCache) -> Tuple[List[int], int]:
    """
    找出 seq 在 cache 中命中的连续前缀 block。
    返回 (hit_block_ids, num_hit_tokens)
    vLLM 实际支持中间命中(后续切分),这里简化为前缀连续命中。
    """
    hits = []
    parent_hash = None
    num_hit_tokens = 0
    block_size = seq.block_size

    for blk_idx in range(seq.num_blocks()):
        start = blk_idx * block_size
        end = min(start + block_size, len(seq.token_ids))
        token_ids = seq.token_ids[start:end]
        h = compute_block_hash(token_ids, parent_hash)
        blk = cache.lookup(h)
        if blk is None:
            break
        hits.append(blk.block_id)
        parent_hash = h
        num_hit_tokens = end

    return hits, num_hit_tokens


def prefill_with_cache(seq: Sequence, cache: PrefixCache):
    """
    带 prefix cache 的 prefill:
        - 命中部分:复用 KV,不计计算量
        - 未命中部分:写入 cache,需实际计算
    """
    hits, num_hit = find_prefix_hits(seq, cache)
    # 复用命中部分
    for bid in hits:
        blk = next(b for b in cache.cache.values() if b.block_id == bid)
        cache.acquire(blk)
    seq.block_table.extend(hits)
    seq.block_hashes.extend([
        compute_block_hash(
            seq.token_ids[i*seq.block_size:(i+1)*seq.block_size],
            seq.block_hashes[-1] if seq.block_hashes else None
        )
        for i in range(len(hits))
    ])
    seq.num_computed = num_hit

    # 处理未命中部分
    parent_hash = seq.block_hashes[-1] if seq.block_hashes else None
    for blk_idx in range(len(hits), seq.num_blocks()):
        start = blk_idx * seq.block_size
        end = min(start + seq.block_size, len(seq.token_ids))
        token_ids = seq.token_ids[start:end]
        h = compute_block_hash(token_ids, parent_hash)
        blk = cache.lookup(h)
        if blk is None:
            blk = cache.alloc_new(h, token_ids)
        cache.acquire(blk)
        seq.block_table.append(blk.block_id)
        seq.block_hashes.append(h)
        parent_hash = h
    seq.num_computed = len(seq.token_ids)


# ============================================================
# 4. 演示:多轮对话共享前缀
# ============================================================

def demo():
    cache = PrefixCache(num_blocks=64, block_size=4)

    system_prompt = [1, 2, 3, 4, 5, 6, 7, 8]  # 2 个 block

    # ---- Round 1 ----
    seq1 = Sequence(seq_id=0,
                    token_ids=system_prompt + [10, 11, 12, 13])
    prefill_with_cache(seq1, cache)
    print(f"Round 1: blocks={seq1.block_table}, computed={seq1.num_computed}/{seq1.num_tokens() if hasattr(seq1,'num_tokens') else len(seq1.token_ids)}")

    # ---- Round 2:同样 system prompt + 不同 user msg ----
    seq2 = Sequence(seq_id=1,
                    token_ids=system_prompt + [20, 21, 22, 23])
    prefill_with_cache(seq2, cache)
    print(f"Round 2: blocks={seq2.block_table}, computed={seq2.num_computed}/{len(seq2.token_ids)}")
    print(f"  -> 前 2 个 block 命中,实际只需 prefill 1 个 block")

    # ---- Round 3:多个 fork 共享前缀 ----
    seq3 = Sequence(seq_id=2,
                    token_ids=system_prompt + [30, 31])
    prefill_with_cache(seq3, cache)
    print(f"Round 3: blocks={seq3.block_table}, computed={seq3.num_computed}/{len(seq3.token_ids)}")

    print(f"\nCache stats: {len(cache.cache)} unique blocks, "
          f"{cache.num_blocks - len(cache.free_ids)} allocated")


if __name__ == "__main__":
    demo()
