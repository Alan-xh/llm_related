"""
10_TritonAttention.py
=====================
SGLang 自研 Triton Attention Kernel:支持 MHA/MQA/GQA/MLA,统一 prefill/decode 路径。

为什么自研:
    - FlashAttention 不支持所有变体(如 MLA、tree attention)
    - 自研 kernel 可针对 SGLang 的 RadixCache + paged KV 优化
    - Triton 比 CUDA C 更易维护,且性能接近

本文实现:
    1. Prefill Attention(长序列,大 batch 计算密集型)
    2. Decode Attention(每步 1 token,内存带宽密集型)
    3. Paged KV Cache 支持
    4. GQA 支持(多 query head 共享 KV head)

FlashAttention v2 算法核心:
    - 把 Q,K,V 切成 block,沿 K 维度循环
    - 用 online softmax 避免显式 NxN attention matrix
    - 减少显存读写:HBM O(N) 而非 O(N^2)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import triton
import triton.language as tl


# ============================================================
# 1. Prefill Attention Kernel(FlashAttention v2 风格)
# ============================================================

@triton.jit
def _flash_attn_prefill_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    seq_len, num_heads, head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    scale: tl.float32,
):
    """
    Prefill:Q 也是长序列,沿 (M=head, N=seq) 二维并行
    简化:单 sequence,batch=1
    """
    pid_m = tl.program_id(0)  # head id
    pid_n = tl.program_id(1)  # query block id

    # Q block: [BLOCK_M, head_dim]
    q_start = pid_n * BLOCK_M
    q_offs = q_start + tl.arange(0, BLOCK_M)
    dim_offs = tl.arange(0, head_dim)

    q_ptr = Q_ptr + pid_m * seq_len * head_dim + q_offs[:, None] * head_dim + dim_offs[None, :]
    q_mask = q_offs[:, None] < seq_len
    q = tl.load(q_ptr, mask=q_mask, other=0.0)

    # 累加器
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    # K/V 沿 seq 循环
    for k_start in range(0, seq_len, BLOCK_N):
        k_offs = k_start + tl.arange(0, BLOCK_N)
        kv_mask = k_offs < seq_len

        # Causal mask
        causal_mask = q_offs[:, None] >= k_offs[None, :]

        # K block: [BLOCK_N, head_dim]
        k_ptr = K_ptr + pid_m * seq_len * head_dim + k_offs[:, None] * head_dim + dim_offs[None, :]
        k = tl.load(k_ptr, mask=kv_mask[:, None], other=0.0)

        # Score: Q @ K^T
        s = tl.dot(q, k.T) * scale  # [BLOCK_M, BLOCK_N]
        s = tl.where(q_mask & kv_mask[None, :] & causal_mask, s, -float("inf"))

        # Online softmax
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_ij = tl.sum(p, axis=1)
        l_new = l_i * alpha + l_ij

        # V block: [BLOCK_N, head_dim]
        v_ptr = V_ptr + pid_m * seq_len * head_dim + k_offs[:, None] * head_dim + dim_offs[None, :]
        v = tl.load(v_ptr, mask=kv_mask[:, None], other=0.0)

        # 累加
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
        l_i = l_new

    # 写回
    out = acc / l_i[:, None]
    o_ptr = O_ptr + pid_m * seq_len * head_dim + q_offs[:, None] * head_dim + dim_offs[None, :]
    tl.store(o_ptr, out.to(O_ptr.dtype.element_ty), mask=q_mask)


def flash_attn_prefill(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       scale: float) -> torch.Tensor:
    """
    q,k,v: [num_heads, seq_len, head_dim]
    return: [num_heads, seq_len, head_dim]
    """
    num_heads, seq_len, head_dim = q.shape
    o = torch.empty_like(q)
    BLOCK_M, BLOCK_N = 64, 64
    grid = (num_heads, triton.cdiv(seq_len, BLOCK_M))
    _flash_attn_prefill_kernel[grid](
        q, k, v, o, seq_len, num_heads, head_dim,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        scale=scale,
    )
    return o


# ============================================================
# 2. Decode Attention Kernel(Paged KV)
# ============================================================

@triton.jit
def _paged_decode_attn_kernel(
    Q_ptr,           # [num_seqs, num_heads, head_dim]
    K_cache_ptr,     # [num_blocks, block_size, num_kv_heads, head_dim]
    V_cache_ptr,
    block_table_ptr, # [num_seqs, max_blocks]
    O_ptr,
    seq_lens_ptr,
    scale: tl.float32,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_blocks: tl.constexpr,
    group_size: tl.constexpr,  # num_heads // num_kv_heads (GQA)
):
    """
    Decode:每个 sequence 1 个 query token,遍历所有 KV。
    支持 GQA:group_size 个 query head 共享 1 个 KV head。
    """
    seq_id = tl.program_id(0)
    head_group_id = tl.program_id(1)  # 0 to num_kv_heads-1

    seq_len = tl.load(seq_lens_ptr + seq_id)
    num_blocks_seq = (seq_len + block_size - 1) // block_size

    # 该 group 内的 query head
    head_start = head_group_id * group_size
    dim_offs = tl.arange(0, head_dim)

    # 加载该 group 所有 query head
    q_offs = (seq_id * num_heads * head_dim
              + (head_start + tl.arange(0, group_size)[:, None]) * head_dim
              + dim_offs[None, :])
    q = tl.load(Q_ptr + q_offs)  # [group_size, head_dim]

    # 累加器(per query head)
    m_i = tl.full([group_size], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([group_size], dtype=tl.float32)
    acc = tl.zeros([group_size, head_dim], dtype=tl.float32)

    # 遍历 KV blocks
    for blk_idx in range(num_blocks_seq):
        phys_blk = tl.load(block_table_ptr + seq_id * max_blocks + blk_idx)
        token_offs = tl.arange(0, block_size)
        pos = blk_idx * block_size + token_offs
        valid = pos < seq_len

        # 加载 K: [block_size, head_dim]
        k_offs = (phys_blk * block_size * num_kv_heads * head_dim
                  + token_offs[:, None] * num_kv_heads * head_dim
                  + head_group_id * head_dim + dim_offs[None, :])
        k = tl.load(K_cache_ptr + k_offs, mask=valid[:, None], other=0.0)

        # Scores: [group_size, block_size]
        s = tl.dot(q, k.T.to(q.dtype)) * scale
        s = tl.where(valid[None, :], s, -float("inf"))

        # Online softmax
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_ij = tl.sum(p, axis=1)
        l_new = l_i * alpha + l_ij

        # V
        v_offs = (phys_blk * block_size * num_kv_heads * head_dim
                  + token_offs[:, None] * num_kv_heads * head_dim
                  + head_group_id * head_dim + dim_offs[None, :])
        v = tl.load(V_cache_ptr + v_offs, mask=valid[:, None], other=0.0)

        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
        l_i = l_new

    out = acc / l_i[:, None]
    o_offs = (seq_id * num_heads * head_dim
              + (head_start + tl.arange(0, group_size)[:, None]) * head_dim
              + dim_offs[None, :])
    tl.store(O_ptr + o_offs, out.to(O_ptr.dtype.element_ty))


def paged_decode_attn(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                       block_tables: torch.Tensor, seq_lens: torch.Tensor,
                       scale: float) -> torch.Tensor:
    """
    q: [num_seqs, num_heads, head_dim]
    k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
    return: [num_seqs, num_heads, head_dim]
    """
    num_seqs, num_heads, head_dim = q.shape
    num_blocks, block_size, num_kv_heads, _ = k_cache.shape
    max_blocks = block_tables.shape[1]
    group_size = num_heads // num_kv_heads

    o = torch.empty_like(q)
    grid = (num_seqs, num_kv_heads)
    _paged_decode_attn_kernel[grid](
        q, k_cache, v_cache, block_tables, o, seq_lens, scale,
        num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
        block_size=block_size, max_blocks=max_blocks, group_size=group_size,
    )
    return o


# ============================================================
# 3. 演示
# ============================================================

def demo():
    if not torch.cuda.is_available():
        print("CUDA required for Triton kernels")
        return

    torch.manual_seed(42)

    # ---- Prefill 演示 ----
    print("--- Prefill Attention ---")
    num_heads, seq_len, head_dim = 8, 512, 64
    q = torch.randn(num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    scale = 1.0 / (head_dim ** 0.5)

    o = flash_attn_prefill(q, k, v, scale)
    print(f"Prefill output: {o.shape}")

    # 与 PyTorch 参考实现对比
    q_ref = q.transpose(0, 1)  # [seq, heads, dim]
    k_ref = k.transpose(0, 1)
    v_ref = v.transpose(0, 1)
    attn = torch.einsum("shd,thd->hst", q_ref, k_ref) * scale
    # causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len, device="cuda", dtype=torch.bool), diagonal=1)
    attn = attn.masked_fill(mask, float("-inf"))
    attn = torch.softmax(attn, dim=-1)
    o_ref = torch.einsum("hst,thd->shd", attn, v_ref)
    print(f"Max diff vs PyTorch ref: {(o - o_ref).abs().max().item():.6f}")

    # ---- Decode 演示(GQA) ----
    print("\n--- Decode Attention (GQA) ---")
    num_seqs = 4
    num_heads, num_kv_heads = 8, 2  # GQA: 4 query heads per KV head
    block_size = 16
    num_blocks = 32
    max_blocks_per_seq = 8

    q = torch.randn(num_seqs, num_heads, head_dim, device="cuda", dtype=torch.float16)
    k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim,
                          device="cuda", dtype=torch.float16)
    v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim,
                          device="cuda", dtype=torch.float16)
    block_tables = torch.randint(0, num_blocks, (num_seqs, max_blocks_per_seq),
                                 device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([32, 16, 48, 24], device="cuda", dtype=torch.int32)

    o = paged_decode_attn(q, k_cache, v_cache, block_tables, seq_lens, scale)
    print(f"Decode output: {o.shape}  (GQA: {num_heads} q-heads, {num_kv_heads} kv-heads)")


if __name__ == "__main__":
    demo()
