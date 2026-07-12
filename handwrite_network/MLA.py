'''
MLA(Multi-head Latent Attention, 多头潜在注意力) 

核心思想是通过低秩压缩（Low-Rank Compression）将Key和Value（KV）先投影到一个低维的潜在空间（Latent Space） 中，再进行后续的注意力计算。这样做的主要目的，是在保持模型性能的同时，显著压缩KV缓存（KV Cache） 的大小，从而降低大语言模型推理时的内存占用和带宽需求。KV 压缩到一个低维的潜在向量（Latent Vector）中，推理时再动态解压出来。解决长上下文推理问题。

MHA（Multi-Head Attention，标准多头注意力）: 每个查询头（Q）都对应独立的键（K）和值（V）头
MQA（Multi-Query Attention，多查询注意力）: 注意力头共享同一份键（K）和值（V），只有查询（Q）是多头的
GQA（Grouped-Query Attention，分组查询注意力）: 将Q头分成若干组，每组共享一份KV。代表作是Meta的Llama 2/3。它既比MHA省钱，又比MQA效果好，是目前开源大模型最主流的方案。

Sparse Attention（稀疏注意力）：不看全部token，只让每个token关注局部的、或步长固定的少数token。
FlashAttention（闪存注意力）：严格来说它不是改变注意力数学公式的变体，而是IO感知（IO-aware）的精确注意力实现。它不压缩参数，而是通过分块（tiling）和重计算，避免读写慢速的HBM（高带宽内存），大幅加速了MHA和GQA的训练速度。
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .RMSNorm import RMSNorm
from .RotaryEmbedding import RotaryEmbedding


class MLA(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        max_seq_len,
        max_batch_size,
        mode,
    ):
        '''
        NOPE：No Positional Embedding

        ROPE：Rotary Positional Embedding
        '''
        super(MLA).__init__()
        self.dim = dim  # 隐藏层纬度  4096 / n_heads
        self.n_heads = n_heads  # 注意力机制头数
        self.q_lora_rank = q_lora_rank  # query降秩纬度
        self.kv_loara_rank = kv_lora_rank  # kv降秩纬度
        self.qk_nope_head_dim = qk_nope_head_dim  # 不包含旋转位置编码的特征维度 4096
        self.qk_rope_head_dim = qk_rope_head_dim  # 包含旋转位置编码的特征维度 768
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim # 完整的 Query/Key 头维度
        self.v_head_dim = v_head_dim # 完整的 Value 头维度
        self.mode = mode # native or rope
        self.max_seq_len = max_seq_len # 缓存的最大序列长度
        self.max_batch_size = max_batch_size # 缓存的最大批量大小

        # 定义投影矩阵
        self.wq_a = nn.Linear(dim, q_lora_rank)
        self.q_norm = RMSNorm(hidden_size=self.q_lora_rank)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim)

        self.wkv_a = nn.Linear(dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(hidden_size=self.kv_lora_rank)
        self.wkv_b = nn.Linear(
            self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim)
        )

        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim)

        self.rotary_emb = RotaryEmbedding(dim=self.qk_rope_head_dim)

        # kv cache
        if self.mode == 'naive':
            self.register_buffer(
                'k_cache',
                torch.zeros(
                    self.max_batch_size,
                    self.max_seq_len,
                    self.n_heads,
                    self.qk_head_dim,
                ),
                persistent=False,  # 不做模块的一部分
            )

            self.register_buffer(
                'v_cache',
                torch.zeros(
                    self.max_batch_size, self.max_seq_len, self.n_heads, self.v_head_dim
                ),
                persistent=False,
            )

        else:
            # 只保留多头共享的 kv秩(qk_nope_head_dim和v_head_dim之前的)和 c_p
            self.register_buffer(
                'kv_cache',
                torch.zeros(
                    self.max_batch_size,
                    self.max_seq_len,
                    self.kv_loara_rank,
                ),
                persistent=False,
            )
            self.register_buffer(
                'pe_cache',
                torch.zeros(
                    self.max_batch_size, self.max_seq_len, self.qk_rope_head_dim
                ),
                persistent=False,
            )

    def forward(self, x, mask=None):
        bs, seq_len, dim = x.shape

        q = self.wq_a(x)  # [bs, seq_len, dim]
        q = self.q_norm(q)  # [bs, seq_len, q_lora_rank]
        q = self.wq_b(q)  # [bs, seq_len, n_heads * qk_head_dim]
        q = q.view(
            bs, seq_len, self.n_heads, self.qk_head_dim
        )  # [bs, seq_len, n_heads, qk_head_dim]
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )  # q_nope shape:[bs, seq_len, n_heads, qk_nope_head_dim] q_pe shape:[bs, seq_len, n_heads, qk_rope_head_dim]

        kv = self.wkv_a(x)  # [bs, seq_len, kv_lora_rank + qk_rope_head_dim]
        kv, k_pe = torch.split(
            kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )  # kv shape: [bs, seq_len, kv_lora_rank] kv_pe shape: [bs, seq_len, qk_rope_head_dim]

        # k_pe only one head, q_pe has multi heads
        k_pe = k_pe.unsqueeze(2)  # [bs, seq_len, 1, qk_rope_head_dim]
        q_pe, k_pe = self.rotary_emb(q_pe, k_pe)

        if self.mode == 'naive':
            '''register k_cache and v_cache'''
            q = torch.cat([q_nope, q_pe], dim=-1)  # [bs, seq_len, n_heads, qk_head_dim]

            # 在 rotary_emb后计算玩 k_nope再添加位置编码部分, qk_rope only one head
            kv = self.kv_norm(kv)  # [bs, seq_len, kv_lora_rank]
            kv = self.wkv_b(
                kv
            )  # [bs, seq_len, n_heads * (qk_nope_head_dim + v_head_dim)]
            kb = kv.view(
                bs, seq_len, self.n_heads, self.qk_nope_head_dim + self.v_head_dim
            )  # [bs, seq_len, n_heads, qk_nope_head_dim + v_head_dim]
            k_nope, v = torch.split(
                kb, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )  # k_nope shape:[bs, seq_len, n_heads, qk_nope_head_dim] v shape:[bs, seq_len, n_heads, v_head_dim]

            k = torch.cat(
                [k_nope, k_pe.extand(-1, -1, self.n_heads, -1)], dim=-1
            )  # [bs, seq_len, n_heads, qk_nope_head_dim + qk_rope_head_dim]

            # cache
            self.k_cache[:bs, :seq_len, :, :] = k
            self.v_cache[:bs, :seq_len, :, :] = v

            # calculation attention scores on einsum
            ''' 
            einsum; Einstein summation convention 爱因斯坦求和约定，简洁的字符串表示法，执行各种复杂的张量操作。
             [bs, q_seq_len, n_heads, qk_nope_head_dim + qk_rope_head_dim] * [bs, k_seq_len, n_heads, qk_nope_head_dim + qk_rope_head_dim] -> [bs, seq_len, q_seq_len, n_heads]
            '''
            # scores = torch.einsum(
            #     "bshd,buhd->bsuh", q, self.k_cache[:bs, :seq_len]
            # ) / math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim)

            # calculation attention scores on matmul
            scores = torch.matmul(
                q.transpose(1, 2),
                self.k_cache[:bs, :seq_len, :, :].transpose(1, 2).transpose(2, 3)
                / math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim),
            )

            scores = scores.transpose(1, 2)
        else:
            '''register kv_loara_rank and pe_cache'''
            k_pe = k_pe.unsqueeze(2)  # [bs, seq_len, 1, 1, qk_rope_head_dim]
            wkv_b = (
                self.wkv_b.weight
            )  # [n_heads * (qk_nope_dim + v_head_dim), kv_lora_rank] type: torch.nn.Parameter
            wkv_b = wkv_b.view(
                self.n_heads, -1, self.kv_lora_rank
            )  # [n_heads, qk_nope_dim + v_head_dim, kv_lora_rank]  note: view中的-1表示自动计算维度

            # q 直接点乘 kv的秩，相当于 q * ci, dim_c = kv_lora_rank
            q_nope = torch.einsum(
                "bshd, hdc-> bshc", q_nope, wkv_b[:, : self.qk_nope_head_dim]
            )  # q_nope shape:[bs, seq_len, n_heads, kv_lora_rank]

            # q * K(T) = x * wq * ( c * wkv_b[ : , :self.qk_nope_head_dim])(T))
            #          = x * wq * wkv_b[:, :self.qk_nope_head_dim](T) * c(T)
