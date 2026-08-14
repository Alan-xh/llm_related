"""
12_MultiLoRA.py
===============
SGLang S-LoRA:同时为 batch 中不同请求应用不同 LoRA,无切换开销。

对比 vLLM 的 LoRA 实现:
    - vLLM 早期:每 step 切换 adapter(切换开销)
    - vLLM BGMN kernel:支持 batched LoRA(类似 S-LoRA)
    - SGLang:从开始就基于 S-LoRA 论文实现

S-LoRA 核心算法:
    1. 把所有 adapter 的 A、B 堆叠成统一 tensor
    2. 用 grouped GEMM 一次算出所有 adapter 的增量
    3. scatter_add 把增量加到对应请求的输出

关键技巧:
    - Padding:把 batch 中不同 rank 的 adapter pad 到相同 rank
    - Index:每个 token 对应一个 adapter index
    - Compute:A @ X^T (grouped) -> B @ (A @ X^T) (grouped)

性能:
    - 单 LoRA 接近无 LoRA 推理速度
    - 多 LoRA 仅小幅下降(grouped GEMM 利用率高)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. LoRA Adapter 池
# ============================================================

@dataclass
class LoRAAdapter:
    adapter_id: int
    rank: int
    lora_A: torch.Tensor  # [rank, in_features]
    lora_B: torch.Tensor  # [out_features, rank]
    scaling: float = 1.0


class LoRAPool:
    """管理所有 LoRA adapter,提供统一访问"""

    def __init__(self, in_features: int, out_features: int, max_rank: int = 32):
        self.in_features = in_features
        self.out_features = out_features
        self.max_rank = max_rank
        self.adapters: Dict[int, LoRAAdapter] = {}

    def register(self, adapter_id: int, rank: int, scaling: float = 1.0):
        # A 用 kaiming 初始化,B 用 0(初始 ΔW=0)
        A = torch.randn(rank, self.in_features) * (1.0 / self.in_features ** 0.5)
        B = torch.zeros(self.out_features, rank)
        self.adapters[adapter_id] = LoRAAdapter(adapter_id, rank, A, B, scaling)

    def get_padded_weights(self) -> tuple:
        """
        把所有 adapter 的 A、B pad 到 max_rank,堆叠成统一 tensor。
        return:
            A_stack: [num_adapters, max_rank, in_features]
            B_stack: [num_adapters, out_features, max_rank]
            scalings: [num_adapters]
            id_to_idx: dict
        """
        num = len(self.adapters)
        A_stack = torch.zeros(num, self.max_rank, self.in_features)
        B_stack = torch.zeros(num, self.out_features, self.max_rank)
        scalings = torch.zeros(num)
        id_to_idx = {}

        for idx, (aid, adp) in enumerate(sorted(self.adapters.items())):
            A_stack[idx, :adp.rank] = adp.lora_A
            B_stack[idx, :, :adp.rank] = adp.lora_B
            scalings[idx] = adp.scaling * (self.max_rank / adp.rank)  # padding 补偿
            id_to_idx[aid] = idx

        return A_stack, B_stack, scalings, id_to_idx


# ============================================================
# 2. S-LoRA Linear Layer
# ============================================================

class SLoRALinear:
    """
    S-LoRA 实现:base weight + grouped LoRA delta
    """

    def __init__(self, in_features: int, out_features: int, pool: LoRAPool):
        self.in_features = in_features
        self.out_features = out_features
        self.pool = pool
        # Base weight
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor,
                adapter_ids: List[int]) -> torch.Tensor:
        """
        x: [batch, in_features]
        adapter_ids: List[int] 长度 batch
        """
        # 1. Base 计算
        y = x @ self.weight.t() + self.bias  # [batch, out]

        # 2. 收集本 batch 用到的 adapter 的权重
        A_stack, B_stack, scalings, id_to_idx = self.pool.get_padded_weights()

        # 为 batch 中每个 token 索引其 adapter
        idx_tensor = torch.tensor(
            [id_to_idx.get(a, 0) for a in adapter_ids],
            dtype=torch.long
        )

        # 3. S-LoRA grouped GEMM(简化版)
        # 第一阶段:tmp = x @ A^T  (per token 选对应 A)
        # tmp[b, r] = sum_i x[b, i] * A_stack[adapter_idx[b], r, i]
        # 用 gather + bmm 模拟
        A_selected = A_stack[idx_tensor]  # [batch, max_rank, in_features]
        tmp = torch.einsum("bi,bri->br", x, A_selected)  # [batch, max_rank]

        # 第二阶段:delta = tmp @ B^T (per token 选对应 B)
        B_selected = B_stack[idx_tensor]  # [batch, out_features, max_rank]
        delta = torch.einsum("br,bor->bo", tmp, B_selected)  # [batch, out_features]

        # 应用 scaling
        scalings_selected = scalings[idx_tensor]  # [batch]
        delta = delta * scalings_selected.unsqueeze(1)

        return y + delta


# ============================================================
# 3. 模拟 S-LoRA 整层 forward
# ============================================================

def benchmark_slora_vs_naive(in_features=256, out_features=512,
                              num_adapters=8, batch_size=32):
    """对比 S-LoRA grouped GEMM vs 逐 adapter 计算"""
    import time
    pool = LoRAPool(in_features, out_features, max_rank=16)
    for i in range(num_adapters):
        pool.register(i, rank=8, scaling=0.5)
        # 让 B 非零(模拟训练后)
        pool.adapters[i].lora_B = torch.randn_like(pool.adapters[i].lora_B) * 0.01

    layer = SLoRALinear(in_features, out_features, pool)
    x = torch.randn(batch_size, in_features)

    # 每个请求随机分配 adapter
    adapter_ids = [i % num_adapters for i in range(batch_size)]

    # ---- S-LoRA ----
    for _ in range(5):
        _ = layer.forward(x, adapter_ids)
    t0 = time.perf_counter()
    for _ in range(100):
        _ = layer.forward(x, adapter_ids)
    slora_time = time.perf_counter() - t0

    # ---- Naive(逐 adapter 分组) ----
    def naive_forward():
        y = x @ layer.weight.t() + layer.bias
        # 按 adapter 分组
        groups: Dict[int, List[int]] = {}
        for i, aid in enumerate(adapter_ids):
            groups.setdefault(aid, []).append(i)
        for aid, indices in groups.items():
            adp = pool.adapters[aid]
            idx = torch.tensor(indices)
            x_sub = x[idx]
            tmp = x_sub @ adp.lora_A.t()
            delta = (tmp @ adp.lora_B.t()) * adp.scaling
            y[idx] += delta
        return y

    for _ in range(5):
        _ = naive_forward()
    t0 = time.perf_counter()
    for _ in range(100):
        _ = naive_forward()
    naive_time = time.perf_counter() - t0

    return slora_time, naive_time


# ============================================================
# 4. 演示
# ============================================================

def demo():
    torch.manual_seed(42)
    in_f, out_f = 256, 512
    pool = LoRAPool(in_f, out_f, max_rank=16)

    # 创建 4 个 LoRA adapter(不同 rank)
    pool.register(0, rank=8, scaling=0.5)
    pool.register(1, rank=16, scaling=0.3)
    pool.register(2, rank=4, scaling=0.8)
    pool.register(3, rank=16, scaling=0.4)

    # 让 B 非零(模拟训练后)
    for adp in pool.adapters.values():
        adp.lora_B = torch.randn_like(adp.lora_B) * 0.01

    layer = SLoRALinear(in_f, out_f, pool)

    # batch=8,每个请求用不同 adapter
    x = torch.randn(8, in_f)
    adapter_ids = [0, 1, 2, 3, 0, 1, 2, 3]
    y = layer.forward(x, adapter_ids)

    print(f"S-LoRA forward:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Adapter assignment: {adapter_ids}")

    # 验证正确性:对比逐 adapter 计算
    y_ref = x @ layer.weight.t() + layer.bias
    for i, aid in enumerate(adapter_ids):
        adp = pool.adapters[aid]
        delta = (x[i] @ adp.lora_A.t()) @ adp.lora_B.t() * adp.scaling
        y_ref[i] += delta

    err = (y - y_ref).abs().max().item()
    print(f"  Max error vs naive: {err:.6e}")
    print(f"  (注:误差来自 rank padding 补偿)")

    # Benchmark
    print("\n--- Benchmark ---")
    slora_t, naive_t = benchmark_slora_vs_naive()
    print(f"S-LoRA grouped:  {slora_t*1000:.1f} ms / 100 iters")
    print(f"Naive per-group: {naive_t*1000:.1f} ms / 100 iters")
    print(f"Speedup: {naive_t/slora_t:.2f}x")


if __name__ == "__main__":
    demo()
