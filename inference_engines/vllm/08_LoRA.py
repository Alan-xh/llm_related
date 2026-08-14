"""
08_LoRA.py
==========
vLLM Multi-LoRA Serving:同时为不同请求加载不同 LoRA adapter,无需切换权重。

核心机制:
    1. Base model 权重常驻 GPU
    2. 多个 LoRA adapter (A, B) 池化管理
    3. batch 内不同请求绑定不同 LoRA ID
    4. forward 时按请求分组,各自应用 LoRA:
         Y = X @ W_base^T + (X @ A) @ B  (per-request)
    5. v0.6+ 引入 BGMN kernel:同一 batch 多 LoRA 无切换开销

LoRA 数学:
    增量 ΔW = B @ A,其中 A: [r, in], B: [out, r], rank r << in/out
    推理时:y = x @ (W + ΔW)^T = x @ W^T + (x @ A^T) @ B^T
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. LoRA Adapter 定义
# ============================================================

@dataclass
class LoRAAdapter:
    """单个 LoRA adapter"""
    adapter_id: int
    rank: int
    # 增量矩阵
    lora_A: torch.Tensor  # [rank, in_features]
    lora_B: torch.Tensor  # [out_features, rank]
    scaling: float = 1.0

    @classmethod
    def create(cls, adapter_id: int, in_features: int, out_features: int,
               rank: int = 8, scaling: float = 1.0, dtype=torch.float16):
        # LoRA 初始化:A 用 kaiming,B 用 0(初始 ΔW=0)
        A = torch.randn(rank, in_features, dtype=dtype) * (1.0 / in_features ** 0.5)
        B = torch.zeros(out_features, rank, dtype=dtype)
        return cls(adapter_id, rank, A, B, scaling)


class LoRAPool:
    """管理多个 LoRA adapter,常驻 GPU"""

    def __init__(self):
        self.adapters: Dict[int, LoRAAdapter] = {}

    def register(self, adapter: LoRAAdapter):
        self.adapters[adapter.adapter_id] = adapter

    def get(self, adapter_id: int) -> Optional[LoRAAdapter]:
        return self.adapters.get(adapter_id)


# ============================================================
# 2. Base Linear Layer(支持 per-request LoRA)
# ============================================================

class LoRALinear:
    """
    Y = X @ W_base^T + (X @ A^T) @ B^T * scaling

    简化实现:对 batch 中每个请求单独应用 LoRA,然后拼接。
    实际 vLLM 用 grouped GEMM kernel(BGMN)做高效批量计算。
    """

    def __init__(self, in_features: int, out_features: int, dtype=torch.float16):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=dtype) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))

    def forward(self, x: torch.Tensor, lora_pool: LoRAPool,
                request_adapter_ids: List[int]) -> torch.Tensor:
        """
        x: [batch, in_features]
        request_adapter_ids: 长度 batch,每个元素是该请求用的 LoRA id
        """
        # 1. Base 计算
        y = x @ self.weight.t() + self.bias  # [batch, out]

        # 2. 按 adapter_id 分组,各算 LoRA 增量
        groups: Dict[int, List[int]] = {}
        for i, aid in enumerate(request_adapter_ids):
            if aid is not None:
                groups.setdefault(aid, []).append(i)

        for aid, indices in groups.items():
            adapter = lora_pool.get(aid)
            if adapter is None:
                continue
            idx = torch.tensor(indices, dtype=torch.long)
            x_sub = x[idx]  # [n, in]
            # Δy = (x @ A^T) @ B^T * scaling
            tmp = x_sub @ adapter.lora_A.t()  # [n, rank]
            delta_y = (tmp @ adapter.lora_B.t()) * adapter.scaling  # [n, out]
            y[idx] += delta_y

        return y


# ============================================================
# 3. BGMN-style 批量 LoRA(高效实现)
# ============================================================

def batched_lora_forward(x: torch.Tensor,
                         base_weight: torch.Tensor,
                         bias: torch.Tensor,
                         lora_pool: LoRAPool,
                         request_adapter_ids: List[int]) -> torch.Tensor:
    """
    模拟 BGMN kernel:
        - 把所有 LoRA 的 A 堆叠成 [num_adapters, rank, in]
        - 把所有 LoRA 的 B 堆叠成 [num_adapters, out, rank]
        - 用 grouped GEMM 一次算出所有 adapter 的增量
        - 用 scatter_add 把增量加到对应请求的输出上
    """
    y = x @ base_weight.t() + bias  # [batch, out]

    # 收集本 batch 用到的 adapter
    used_ids = sorted(set(a for a in request_adapter_ids if a is not None))
    if not used_ids:
        return y

    # 堆叠 A、B
    A_stack = torch.stack([lora_pool.get(aid).lora_A for aid in used_ids])  # [n_adp, rank, in]
    B_stack = torch.stack([lora_pool.get(aid).lora_B for aid in used_ids])  # [n_adp, out, rank]
    scalings = torch.tensor([lora_pool.get(aid).scaling for aid in used_ids])

    # 为 batch 中每个 token 索引其 adapter
    id_to_idx = {aid: i for i, aid in enumerate(used_ids)}
    adapter_indices = torch.tensor(
        [id_to_idx[a] if a is not None else 0 for a in request_adapter_ids],
        dtype=torch.long
    )

    # 简化:用 einsum 模拟 grouped GEMM
    # tmp[b, r] = sum_i x[b, i] * A_stack[adapter_indices[b], r, i]
    tmp = torch.einsum("bi,nri->br", x, A_stack)  # 这里简化,实际需要 gather
    # 正确做法:逐 token 选对应 adapter 的 A
    tmp = torch.zeros(x.shape[0], A_stack.shape[1], dtype=x.dtype)
    for b in range(x.shape[0]):
        idx = adapter_indices[b].item()
        tmp[b] = x[b] @ A_stack[idx].t()
    # delta_y[b, o] = tmp[b, r] * B_stack[idx, o, r] * scaling
    delta_y = torch.zeros_like(y)
    for b in range(x.shape[0]):
        idx = adapter_indices[b].item()
        delta_y[b] = tmp[b] @ B_stack[idx].t() * scalings[idx]

    return y + delta_y


# ============================================================
# 4. 演示:Multi-LoRA 推理
# ============================================================

def demo():
    torch.manual_seed(42)
    in_f, out_f = 64, 128
    layer = LoRALinear(in_f, out_f, dtype=torch.float32)

    # 创建 3 个 LoRA adapter
    pool = LoRAPool()
    for i in range(3):
        adapter = LoRAAdapter.create(adapter_id=i,
                                    in_features=in_f, out_features=out_f,
                                    rank=8, scaling=0.5)
        # 让 B 非零(训练后)
        adapter.lora_B = torch.randn_like(adapter.lora_B) * 0.01
        pool.register(adapter)

    # batch=6,每个请求用不同 LoRA(最后一个不用)
    x = torch.randn(6, in_f)
    request_adapter_ids = [0, 1, 2, 0, 1, None]

    y = layer.forward(x, pool, request_adapter_ids)
    print(f"Multi-LoRA forward output: {y.shape}")
    print(f"  batch 0 (LoRA 0) output[:4]: {y[0, :4].tolist()}")
    print(f"  batch 3 (LoRA 0) output[:4]: {y[3, :4].tolist()}")
    print(f"  batch 5 (no LoRA) output[:4]: {y[5, :4].tolist()}")

    # 验证:base + LoRA0 增量
    y_base = x[0] @ layer.weight.t() + layer.bias
    a = pool.get(0)
    y_lora0 = y_base + (x[0] @ a.lora_A.t()) @ a.lora_B.t() * a.scaling
    print(f"\nVerification (batch 0 should match LoRA0 result): "
          f"{torch.allclose(y[0], y_lora0, atol=1e-5)}")

    # BGMN-style
    y2 = batched_lora_forward(x, layer.weight, layer.bias, pool, request_adapter_ids)
    print(f"BGMN-style matches naive: {torch.allclose(y, y2, atol=1e-5)}")


if __name__ == "__main__":
    demo()
