"""
11_FlashInferIntegration.py
===========================
SGLang 与 FlashInfer 集成:统一 prefill/decode/append 路径,Hopper 上性能优秀。

FlashInfer 是什么:
    - 由 UMich/MLSys 团队开发的高效 LLM attention 库
    - 统一 prefill/decode/append 三种 attention 模式
    - 支持 Paged KV Cache
    - Hopper H100 上用 TMA(Tensor Memory Accelerator)+ WGMMA
    - 提供 batch API:一次 forward 处理多个不同长度的 sequence

为什么 SGLang 集成 FlashInfer:
    - 自研 Triton kernel 灵活但性能不如 CUDA 库
    - FlashInfer 已经优化了 paged KV + GQA + MLA
    - 与 SGLang 的 RadixCache 配合:paged KV cache 直接传给 FlashInfer

API 模式:
    1. BatchPrefillWithPagedKVCache:多个 sequence 同时 prefill
    2. BatchDecodeWithPagedKVCache:多个 sequence 同时 decode
    3. BatchPrefillWithRaggedKVCache:无 paged 的 prefill
    4. SingleDecodeWithPagedKVCache:单序列 decode

本文实现:
    - FlashInferWrapper(简化接口)
    - 与 RadixCache 配合使用
    - Mock 实际计算(无 FlashInfer 依赖)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import torch
import torch.nn.functional as F


# ============================================================
# 1. Paged KV Cache 数据结构
# ============================================================

@dataclass
class PagedKVCache:
    """
    FlashInfer 风格的 paged KV cache。
    k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
    """
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_size: int

    @property
    def num_blocks(self):
        return self.k_cache.shape[0]

    @property
    def num_kv_heads(self):
        return self.k_cache.shape[2]

    @property
    def head_dim(self):
        return self.k_cache.shape[3]


@dataclass
class BatchInfo:
    """一批 sequence 的 metadata"""
    q: torch.Tensor                       # [total_q_tokens, num_heads, head_dim]
    q_offsets: torch.Tensor               # [batch+1] 每个 sequence 的 q 起始位置
    block_tables: torch.Tensor            # [batch, max_blocks]
    seq_lens: torch.Tensor                # [batch]
    request_indices: torch.Tensor         # [total_q_tokens] 每个 q 属于哪个 sequence


# ============================================================
# 2. FlashInfer Wrapper(简化)
# ============================================================

class FlashInferWrapper:
    """
    模拟 FlashInfer 的 batch API。
    实际 FlashInfer 用 C++/CUDA 实现,这里用 PyTorch 模拟逻辑。
    """

    def __init__(self, kv_cache: PagedKVCache, num_heads: int):
        self.kv_cache = kv_cache
        self.num_heads = num_heads
        self.group_size = num_heads // kv_cache.num_kv_heads
        self.scale = 1.0 / (kv_cache.head_dim ** 0.5)

    def batch_prefill(self, batch: BatchInfo) -> torch.Tensor:
        """
        批量 prefill:多个 sequence 一次 forward。
        每个序列的 query 可能很长(prefill 整个 prompt)。

        FlashInfer 实际用:
            - 沿 query 维度并行
            - 每个 query block 对所有 KV 做 attention
            - 用 paged KV 寻址
        """
        total_q, num_heads, head_dim = batch.q.shape
        out = torch.zeros(total_q, num_heads, head_dim,
                          dtype=batch.q.dtype, device=batch.q.device)

        # 简化:逐个 sequence 逐个 query 计算
        q_offsets = batch.q_offsets.tolist()
        seq_lens = batch.seq_lens.tolist()
        block_tables = batch.block_tables

        for seq_idx in range(len(seq_lens)):
            q_start, q_end = q_offsets[seq_idx], q_offsets[seq_idx+1]
            seq_q = batch.q[q_start:q_end]  # [seq_q_len, num_heads, head_dim]
            seq_len = seq_lens[seq_idx]
            block_table = block_tables[seq_idx]

            # 收集该 sequence 的 KV(从 paged cache)
            k_seq, v_seq = self._gather_kv(block_table, seq_len)
            # k_seq: [seq_len, num_kv_heads, head_dim]

            # 对每个 head 做 attention
            for h in range(num_heads):
                kv_h = h // self.group_size
                q_h = seq_q[:, h, :]  # [seq_q_len, head_dim]
                k_h = k_seq[:, kv_h, :]  # [seq_len, head_dim]
                v_h = v_seq[:, kv_h, :]
                # causal attention
                scores = q_h @ k_h.T * self.scale  # [seq_q_len, seq_len]
                # 应用 causal mask
                q_positions = torch.arange(q_start, q_end) - q_start
                k_positions = torch.arange(seq_len)
                # 简化:假设 q 起始位置就是 seq 末尾(prefill 末尾)
                causal = q_positions.unsqueeze(1) >= k_positions.unsqueeze(0)
                scores = scores.masked_fill(~causal, float("-inf"))
                attn = F.softmax(scores, dim=-1)
                out[q_start:q_end, h, :] = attn @ v_h

        return out

    def batch_decode(self, batch: BatchInfo) -> torch.Tensor:
        """
        批量 decode:每个 sequence 1 个 query token。
        FlashInfer 用 SIMD 友好的 layout,提升 GPU 利用率。
        """
        batch_size = batch.q.shape[0]
        out = torch.zeros_like(batch.q)

        for seq_idx in range(batch_size):
            q = batch.q[seq_idx]  # [num_heads, head_dim]
            seq_len = batch.seq_lens[seq_idx].item()
            block_table = batch.block_tables[seq_idx]

            k_seq, v_seq = self._gather_kv(block_table, seq_len)

            for h in range(self.num_heads):
                kv_h = h // self.group_size
                q_h = q[h]  # [head_dim]
                k_h = k_seq[:, kv_h, :]
                v_h = v_seq[:, kv_h, :]
                scores = q_h @ k_h.T * self.scale
                attn = F.softmax(scores, dim=-1)
                out[seq_idx, h, :] = attn @ v_h

        return out

    def _gather_kv(self, block_table: torch.Tensor,
                   seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """从 paged cache 收集一个 sequence 的 KV"""
        bs = self.kv_cache.block_size
        num_blocks_seq = (seq_len + bs - 1) // bs
        k_parts, v_parts = [], []
        for i in range(num_blocks_seq):
            bid = block_table[i].item()
            n = min(bs, seq_len - i*bs)
            k_parts.append(self.kv_cache.k_cache[bid, :n])
            v_parts.append(self.kv_cache.v_cache[bid, :n])
        return torch.cat(k_parts), torch.cat(v_parts)


# ============================================================
# 3. 与 RadixCache 配合使用
# ============================================================

class SGLangAttentionBackend:
    """
    SGLang 把 FlashInfer 作为 attention backend:
        - RadixCache 管理 KV block 分配
        - FlashInfer 负责高效 attention 计算
        - 两者通过 block_table 解耦
    """

    def __init__(self, num_blocks: int, block_size: int,
                 num_heads: int, num_kv_heads: int, head_dim: int):
        # Paged KV Cache
        k_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim,
                              dtype=torch.float16)
        v_cache = torch.zeros_like(k_cache)
        self.kv_cache = PagedKVCache(k_cache, v_cache, block_size)
        self.wrapper = FlashInferWrapper(self.kv_cache, num_heads)
        self.block_size = block_size

    def prefill(self, q: torch.Tensor, kv_data: List[torch.Tensor],
                block_tables: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        """
        q: [total_q_tokens, num_heads, head_dim]
        kv_data: 每个序列的新 KV(要写入 cache)
        """
        # 1. 把新 KV 写入 paged cache(简化)
        for i, (kv, bt, sl) in enumerate(zip(kv_data, block_tables, seq_lens)):
            self._write_kv_to_cache(kv, bt, sl.item())

        # 2. 调用 FlashInfer 做 attention
        q_offsets = torch.tensor([0, q.shape[0]], dtype=torch.int32)  # 简化:单序列
        batch = BatchInfo(q=q, q_offsets=q_offsets,
                          block_tables=block_tables.unsqueeze(0),
                          seq_lens=seq_lens.unsqueeze(0) if seq_lens.dim() == 1 else seq_lens,
                          request_indices=torch.zeros(q.shape[0], dtype=torch.int32))
        return self.wrapper.batch_prefill(batch)

    def decode(self, q: torch.Tensor, block_tables: torch.Tensor,
               seq_lens: torch.Tensor) -> torch.Tensor:
        batch = BatchInfo(q=q, q_offsets=torch.arange(q.shape[0]+1),
                          block_tables=block_tables, seq_lens=seq_lens,
                          request_indices=torch.arange(q.shape[0]))
        return self.wrapper.batch_decode(batch)

    def _write_kv_to_cache(self, kv: torch.Tensor, block_table: torch.Tensor,
                            seq_len: int):
        """把 KV 写入 paged cache(简化)"""
        bs = self.block_size
        for i in range((seq_len + bs - 1) // bs):
            bid = block_table[i].item()
            n = min(bs, seq_len - i*bs)
            # kv shape: [seq_len, num_kv_heads, head_dim]
            self.kv_cache.k_cache[bid, :n] = kv[i*bs:i*bs+n]
            self.kv_cache.v_cache[bid, :n] = kv[i*bs:i*bs+n]


# ============================================================
# 4. 演示
# ============================================================

def demo():
    torch.manual_seed(42)
    num_heads, num_kv_heads, head_dim = 8, 2, 32
    block_size, num_blocks = 4, 32

    backend = SGLangAttentionBackend(num_blocks, block_size,
                                     num_heads, num_kv_heads, head_dim)

    # ---- Prefill 演示 ----
    print("--- FlashInfer Batch Prefill ---")
    seq_len = 10
    q = torch.randn(seq_len, num_heads, head_dim, dtype=torch.float16)
    kv = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.float16)
    block_table = torch.tensor([0, 1, 2], dtype=torch.int32)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)

    out = backend.prefill(q, [kv], block_table.unsqueeze(0), seq_lens)
    print(f"Prefill output: {out.shape}")

    # ---- Decode 演示(批量) ----
    print("\n--- FlashInfer Batch Decode ---")
    batch_size = 3
    q_decode = torch.randn(batch_size, num_heads, head_dim, dtype=torch.float16)
    block_tables = torch.tensor([[0,1,2],[3,4,0],[5,6,7]], dtype=torch.int32)
    seq_lens_decode = torch.tensor([10, 8, 12], dtype=torch.int32)

    out = backend.decode(q_decode, block_tables, seq_lens_decode)
    print(f"Decode output: {out.shape}")
    print(f"\n注:FlashInfer 实际用 TMA + WGMMA,比模拟快 10-100x")
    print(f"   Hopper 上 decode attention 接近理论带宽上限")


if __name__ == "__main__":
    demo()
