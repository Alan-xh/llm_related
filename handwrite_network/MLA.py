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
        super(MLA).__init__()
        self.dim = dim  # 隐藏层纬度  4096 / n_heads
        self.n_heads = n_heads  # 注意力机制头数
        self.q_lora_rank = q_lora_rank  # query降秩纬度
        self.kv_loara_rank = kv_lora_rank  # kv降秩纬度
        self.qk_nope_head_dim = qk_nope_head_dim  # 4096
        self.qk_rope_head_dim = qk_rope_head_dim  # 768
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mode = mode
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size

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
            # 只保留多头共享的c_i和c_p
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

        k_pe = k_pe.unsqueeze(2)  # [bs, seq_len, 1, qk_rope_head_dim]
        q_pe, k_pe = self.rotary_emb(q_pe, k_pe)

        if self.mode == 'naive':
            q = torch.cat([q_nope, q_pe], dim=-1)  # [bs, seq_len, n_heads, qk_head_dim]

            kv = self.kv_norm(kv)  # [bs, seq_len, kv_lora_rank]
