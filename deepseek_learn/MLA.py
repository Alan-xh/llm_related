"""
任务定义: MLA (Multi-Head Latent Attention) 模块实现
领域分类: 序列到序列模型 / 大语言模型注意力机制
代表架构: DeepSeek-V2 / V3 MLA 机制
核心思想: 通过将 KV Cache 进行低秩压缩，极大地减少推理时的显存占用，同时保持注意力计算的表达能力。
        将 Query 和 Key 的向量拆分为非旋转（nope）部分和旋转位置编码（rope）部分，以实现高效的位置感知。
数学目标:
    Attention(Q, K, V) = Softmax( (Q_nope K_nope^T + Q_rope K_rope^T) / sqrt(d_k) ) * V
    MLA 通过矩阵分解 Q = W_q_b * Norm(W_q_a * x), KV_compressed = Norm(W_kv_a * x) 进行计算。

数据输入规范:
    Input: [B, Seq_Len, Dim]
    Output: [B, Seq_Len, Dim]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -----------------------------------------------------------------------------
# 1. 基础模块 (Sub-components)
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """
    RMSNorm (Root Mean Square Layer Normalization)
    作用: 对输入进行均方根归一化，去除均值先验，增强模型训练稳定性。
    公式: x' = (x / sqrt(mean(x^2) + eps)) * weight

    Inputs:
        hidden_states (Tensor): [B, N, C]
    Outputs:
        out (Tensor): [B, N, C]
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.float()


def rotate_half(x):
    """
    旋转位置编码辅助函数：将向量平分，交换位置并取反
    x: [..., dim]
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotate_pos_emb(q, k, cos, sin, unsqueeze_dim=2):
    """
    应用旋转位置编码 (RoPE)
    Q_rope_new = Q_rope * cos + rotate_half(Q_rope) * sin
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """
    旋转位置编码模块 (RoPE)
    """
    def __init__(self, dim, max_seq_len=1024):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float().unsqueeze(1)
        freqs = t @ inv_freq.unsqueeze(0)
        # 复制形成完整的旋转维度
        freqs = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def forward(self, q, k):
        cos = self.cos_cached[: q.shape[1], :].unsqueeze(0)
        sin = self.sin_cached[: q.shape[1], :].unsqueeze(0)
        return apply_rotate_pos_emb(q, k, cos, sin)

# -----------------------------------------------------------------------------
# 2. 核心 MLA 模块 (Top-level Architecture)
# -----------------------------------------------------------------------------

class MLA(nn.Module):
    """
    Multi-Head Latent Attention 模块
    通过低秩分解压缩 KV Cache。
    """
    def __init__(
        self, dim, n_heads, q_lora_rank, kv_lora_rank, 
        qk_nope_head_dim, qk_rope_head_dim, v_head_dim, 
        max_seq_len, max_batch_size, mode='none'
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mode = mode

        # Query 分支: 压缩 -> Norm -> 投影
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim)

        # KV 分支: 压缩 -> Norm -> 投影
        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))

        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim)
        self.rotary_emb = RotaryEmbedding(self.qk_rope_head_dim)

        # Cache 初始化
        if self.mode == 'naive':
            self.register_buffer('k_cache', torch.zeros(max_batch_size, max_seq_len, n_heads, self.qk_head_dim), persistent=False)
            self.register_buffer('v_cache', torch.zeros(max_batch_size, max_seq_len, n_heads, v_head_dim), persistent=False)
        else:
            self.register_buffer('kv_cache', torch.zeros(max_batch_size, max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer('pe_cache', torch.zeros(max_batch_size, max_seq_len, self.qk_rope_head_dim), persistent=False)

    def forward(self, x, mask=None):
        """
        Args:
            x (Tensor): [B, Seq_Len, Dim]
            mask (Tensor, optional): [B, Seq_Len, Seq_Len]
        Returns:
            out (Tensor): [B, Seq_Len, Dim]
        """
        bs, seq_len, _ = x.shape

        # Query 计算
        q = self.wq_b(self.q_norm(self.wq_a(x))) # [B, S, nH * qk_head_dim]
        q = q.view(bs, seq_len, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # KV 计算
        kv_raw = self.wkv_a(x)
        kv, k_pe = torch.split(kv_raw, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        
        # Apply RoPE
        k_pe = k_pe.unsqueeze(2) # [B, S, 1, qk_rope_head_dim]
        q_pe, k_pe = self.rotary_emb(q_pe, k_pe)
        k_pe = k_pe.squeeze(2)

        # Attention 计算
        if self.mode == 'naive':
            # 传统计算逻辑
            # ... 省略逻辑 ...
            pass
        else:
            # MLA 压缩计算逻辑
            kv = self.kv_norm(kv)
            self.kv_cache[:bs, :seq_len, :] = kv
            self.pe_cache[:bs, :seq_len, :] = k_pe
            
            # scores_nope: [B, S, nH, S]
            # 这里使用了 einsum 实现低秩注意力优化
            wkv_b_nope = self.wkv_b.weight.view(self.n_heads, -1, self.kv_lora_rank)[:, :self.qk_nope_head_dim, :]
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b_nope)
            scores_nope = torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bs, :seq_len, :])
            
            scores_pe = torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bs, :seq_len, :])
            
            scores = (scores_nope + scores_pe) / math.sqrt(self.qk_head_dim)
            if mask is not None: scores += mask.unsqueeze(2)
            scores = scores.softmax(dim=-1)

            # Output
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bs, :seq_len])
            wkv_b_v = self.wkv_b.weight.view(self.n_heads, -1, self.kv_lora_rank)[:, -self.v_head_dim:, :]
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b_v)

        x = x.contiguous().view(bs, seq_len, -1)
        return self.wo(x)

def main():
    # Example Pipeline
    B, S, D = 4, 100, 4096
    x = torch.randn(B, S, D)
    mla = MLA(D, 16, 128, 64, 256, 48, 256, 512, 16, mode='mla')
    out = mla(x)
    print(f"Input Shape: {x.shape}, Output Shape: {out.shape}")

if __name__ == '__main__':
    main()
