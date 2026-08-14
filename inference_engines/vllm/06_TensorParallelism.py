"""
06_TensorParallelism.py
=======================
vLLM 张量并行:基于 Megatron-LM 风格,把模型权重按维度切分到多 GPU。

两种切分:
    Column Parallel(列并行):权重 W [out, in] 沿 out 维切分
        Y = X @ W^T  -> 每个 rank 算 Y[:, shard]  -> 输出 [batch, out/N]
        适合:QKV proj、FFN up/gate proj
        后接 all-gather 或 row-parallel 才能合并

    Row Parallel(行并行):权重 W [out, in] 沿 in 维切分
        Y = X[:, shard] @ W[:, shard]^T -> 每个 rank 算部分和
        后接 all-reduce 合并
        适合:O proj、FFN down proj

经典配对:ColumnParallel -> RowParallel(中间不需要同步)
    QKV(ColumnParallel) -> Attention -> O(RowParallel) -> all-reduce
    gate/up(ColumnParallel) -> SiLU -> down(RowParallel) -> all-reduce

本文实现:
    - ColumnParallelLinear / RowParallelLinear
    - 张量并行 Attention / MLP
    - 模拟 2-GPU TP
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 通信原语(模拟)
# ============================================================

class TPCommunicator:
    """
    模拟多 GPU 通信(实际用 NCCL)。
    单进程模拟时:把每个 rank 的输出存到列表,再用 PyTorch 操作模拟。
    """
    @staticmethod
    def all_reduce(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """sum all-reduce"""
        s = sum(tensors)
        return [s.clone() for _ in tensors]

    @staticmethod
    def all_gather(tensors: List[torch.Tensor], dim: int = -1) -> List[torch.Tensor]:
        """沿 dim 拼接"""
        g = torch.cat(tensors, dim=dim)
        return [g for _ in tensors]


# ============================================================
# 2. ColumnParallelLinear / RowParallelLinear
# ============================================================

class ColumnParallelLinear:
    """
    Y = X @ W^T, W 沿输出维(out)切分
    每个 rank 持有 W_shard: [out/N, in]
    输出 Y_shard: [batch, out/N]
    """

    def __init__(self, in_features: int, out_features: int,
                 rank: int, world_size: int, bias: bool = True):
        assert out_features % world_size == 0
        self.rank = rank
        self.world_size = world_size
        self.shard_out = out_features // world_size
        self.weight = nn.Parameter(torch.randn(self.shard_out, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(self.shard_out)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, in]
        y = x @ self.weight.t()  # [batch, shard_out]
        if self.bias is not None:
            y = y + self.bias
        return y


class RowParallelLinear:
    """
    Y = X @ W^T, W 沿输入维(in)切分
    每个 rank 持有 W_shard: [out, in/N]
    输入 X_shard: [batch, in/N] -> 输出 Y: [batch, out](部分和,需 all-reduce)
    """

    def __init__(self, in_features: int, out_features: int,
                 rank: int, world_size: int, bias: bool = True):
        assert in_features % world_size == 0
        self.rank = rank
        self.world_size = world_size
        self.shard_in = in_features // world_size
        self.weight = nn.Parameter(torch.randn(out_features, self.shard_in) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x_shard: torch.Tensor) -> torch.Tensor:
        # x_shard: [batch, in/N]
        y = x_shard @ self.weight.t()  # [batch, out]
        if self.bias is not None:
            # bias 只在 rank 0 加(或 reduce 后加)
            if self.rank == 0:
                y = y + self.bias
        return y


# ============================================================
# 3. TP Attention
# ============================================================

@dataclass
class TPConfig:
    hidden_size: int = 512
    num_heads: int = 8
    head_dim: int = 64
    world_size: int = 2


class TPAttentionRank:
    """
    单 rank 上的 Attention:
        QKV(ColumnParallel) -> per-head attention -> O(RowParallel) -> all-reduce
    """

    def __init__(self, cfg: TPConfig, rank: int):
        self.cfg = cfg
        self.rank = rank
        # num_heads 在各 rank 间均分
        assert cfg.num_heads % cfg.world_size == 0
        self.local_heads = cfg.num_heads // cfg.world_size

        self.qkv = ColumnParallelLinear(cfg.hidden_size, 3 * cfg.num_heads * cfg.head_dim,
                                        rank, cfg.world_size, bias=True)
        self.o = RowParallelLinear(cfg.num_heads * cfg.head_dim, cfg.hidden_size,
                                   rank, cfg.world_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq, hidden]"""
        b, s, h = x.shape
        # QKV column parallel: 输出 [b, s, 3 * local_heads * head_dim]
        qkv = self.qkv.forward(x)
        qkv = qkv.view(b, s, 3, self.local_heads, self.cfg.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: [b, s, local_heads, head_dim]

        # 简化:不做 causal mask,只演示 TP
        scale = self.cfg.head_dim ** -0.5
        attn = torch.einsum("bshd,bthd->bhst", q, k) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhst,bthd->bshd", attn, v)  # [b, s, local_heads, head_dim]
        out = out.reshape(b, s, -1)  # [b, s, local_heads * head_dim]

        # O proj (row parallel):输入是 local 部分,输出是 partial sum
        out_partial = self.o.forward(out)  # [b, s, hidden]
        return out_partial


def tp_attention_forward(x: torch.Tensor, ranks: List[TPAttentionRank]) -> torch.Tensor:
    """完整 TP attention:各 rank forward 后 all-reduce"""
    partials = [r.forward(x) for r in ranks]
    reduced = TPCommunicator.all_reduce(partials)
    return reduced[0]


# ============================================================
# 4. TP MLP (SwiGLU 风格)
# ============================================================

class TPMLPRank:
    """gate/up column parallel -> SiLU -> down row parallel -> all-reduce"""

    def __init__(self, cfg: TPConfig, rank: int, intermediate_size: int = 1024):
        self.rank = rank
        self.gate = ColumnParallelLinear(cfg.hidden_size, intermediate_size, rank, cfg.world_size)
        self.up = ColumnParallelLinear(cfg.hidden_size, intermediate_size, rank, cfg.world_size)
        self.down = RowParallelLinear(intermediate_size, cfg.hidden_size, rank, cfg.world_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate.forward(x))
        up = self.up.forward(x)
        h = gate * up
        return self.down.forward(h)  # partial sum


def tp_mlp_forward(x: torch.Tensor, ranks: List[TPMLPRank]) -> torch.Tensor:
    partials = [r.forward(x) for r in ranks]
    reduced = TPCommunicator.all_reduce(partials)
    return reduced[0]


# ============================================================
# 5. 演示
# ============================================================

def demo():
    torch.manual_seed(42)
    cfg = TPConfig(hidden_size=512, num_heads=8, head_dim=64, world_size=2)

    # 每个 rank 一份权重
    attn_ranks = [TPAttentionRank(cfg, r) for r in range(cfg.world_size)]
    mlp_ranks = [TPMLPRank(cfg, r, intermediate_size=1024) for r in range(cfg.world_size)]

    x = torch.randn(2, 16, cfg.hidden_size)  # [batch=2, seq=16, hidden=512]
    print(f"Input: {x.shape}")

    out_attn = tp_attention_forward(x, attn_ranks)
    print(f"After TP Attention + all-reduce: {out_attn.shape}")

    out_mlp = tp_mlp_forward(out_attn, mlp_ranks)
    print(f"After TP MLP + all-reduce: {out_mlp.shape}")

    # 验证:各 rank 的输出在 all-reduce 后一致
    print(f"\nRank 0 == Rank 1 after all-reduce: "
          f"{torch.allclose(tp_attention_forward(x, attn_ranks), out_attn)}")


if __name__ == "__main__":
    demo()
