"""
07_Quantization.py
==================
vLLM 量化集成:GPTQ、AWQ、FP8、INT8(W8A8 SmoothQuant)。

核心思路:
    - 权重低精度存储(4bit / 8bit),节省显存
    - 计算时 on-the-fly 反量化,或直接用低精度 Tensor Core
    - 不同算法的"如何选 scale"是关键差异

本文实现(教学版):
    1. INT8 Per-channel 量化(基础)
    2. GPTQ 风格:基于 Hessian 的逐列量化误差最小化
    3. AWQ 风格:激活感知,保护重要权重(salient channels)
    4. FP8 (E4M3):Hopper 原生支持
    5. W8A8 SmoothQuant:激活平滑后用 INT8 GEMM
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import torch
import torch.nn.functional as F


# ============================================================
# 1. 基础 INT8 Per-Channel 量化
# ============================================================

def quantize_int8_per_channel(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    weight: [out, in]
    返回 q_weight (int8), scale [out]
    """
    max_abs = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = max_abs / 127.0
    q = (weight / scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale.squeeze(1)


def dequantize_int8(q_weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q_weight.to(torch.float32) * scale.unsqueeze(1)


def int8_matmul(x: torch.Tensor, q_weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Y = X @ W^T (W 已量化为 int8)
    实际 vLLM 用 Triton/CUTLASS kernel 做 int8 GEMM,这里反量化后做 fp 计算
    """
    w = dequantize_int8(q_weight, scale)
    return x @ w.t()


# ============================================================
# 2. GPTQ:基于 Hessian 的逐列量化
# ============================================================

def gptq_quantize(weight: torch.Tensor,
                  hessian: torch.Tensor,
                  group_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPTQ 核心思想:
        逐列量化,用 Hessian 信息把量化误差"分摊"到尚未量化的列上,
        使得整体输出误差最小。

    weight: [out, in]
    hessian: [in, in]  H = 2 * X^T X (X 是校准数据激活)
    """
    out_dim, in_dim = weight.shape
    q_weight = weight.clone().to(torch.float32)
    scales = torch.zeros(out_dim, dtype=torch.float32)

    # 对 Hessian 加对角线正则
    dead = torch.diag(hessian) == 0
    hessian[range(in_dim), range(in_dim)] += torch.where(dead, 0.1, 0.0)
    hessian_inv = hessian.inverse()

    # 逐列处理(GPTQ 是逐 column,group_size 控制多少列一组共用 scale)
    for col in range(in_dim):
        # 当前列权重
        w_col = q_weight[:, col]  # [out]

        # 计算 scale(本列)
        max_abs = w_col.abs().amax().clamp(min=1e-8)
        scale = max_abs / 127.0
        scales[col] = scale  # 简化:每列一个 scale

        # 量化
        q_col = (w_col / scale).round().clamp(-128, 127)
        deq = q_col * scale
        q_weight[:, col] = deq

        # 误差 = q_col * scale - w_col
        err = (deq - w_col) / hessian[col, col]

        # 把误差分摊到后续未量化列
        if col + 1 < in_dim:
            q_weight[:, col+1:] -= err.unsqueeze(1) * hessian_inv[col, col+1:].unsqueeze(0)

    # 转 int8
    q_int = (q_weight / scales.unsqueeze(0)).round().clamp(-128, 127).to(torch.int8)
    return q_int, scales


# ============================================================
# 3. AWQ:激活感知,保护重要权重
# ============================================================

def awq_quantize(weight: torch.Tensor,
                 activation_stats: torch.Tensor,
                 group_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    AWQ 核心思想:
        - 激活幅度大的 channel 是"重要"的
        - 给这些 channel 一个较大的 scale,让权重相对更小,量化误差更小
        - 通过 weight clipping 找最优 scale,使 mse 最小
    activation_stats: [in]  每个 input channel 的激活幅度(如 mean(|X|))
    """
    out_dim, in_dim = weight.shape

    # 1. 找激活幅度大的 channel(salient)
    s_scale = activation_stats.sqrt().clamp(min=1e-8)  # [in]
    # 简化:直接用激活均值作为 scale(实际 AWQ 用 grid search)

    # 2. 用 scale 缩放权重,使其与"激活/scale"配合后等价
    # 等价变换:Y = (X * s) @ (W / s)^T
    w_scaled = weight / s_scale.unsqueeze(0)  # [out, in]

    # 3. 按 group_size 分组量化(每组共用一个 zero-point 和 scale)
    assert in_dim % group_size == 0
    num_groups = in_dim // group_size
    q_weight = torch.zeros_like(w_scaled, dtype=torch.int8)
    group_scales = torch.zeros(out_dim, num_groups)

    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        w_g = w_scaled[:, start:end]
        max_abs = w_g.abs().amax().clamp(min=1e-8)
        scale_g = max_abs / 127.0
        group_scales[:, g] = scale_g
        q_weight[:, start:end] = (w_g / scale_g).round().clamp(-128, 127).to(torch.int8)

    return q_weight, group_scales, s_scale


def awq_dequantize(q_weight, group_scales, s_scale):
    """反量化时需要乘回 s_scale"""
    out_dim, in_dim = q_weight.shape
    group_size = 128
    w = torch.zeros_like(q_weight, dtype=torch.float32)
    for g in range(in_dim // group_size):
        s, e = g*group_size, (g+1)*group_size
        w[:, s:e] = q_weight[:, s:e].to(torch.float32) * group_scales[:, g:g+1]
    return w * s_scale.unsqueeze(0)


# ============================================================
# 4. FP8 (E4M3) 量化
# ============================================================

def quantize_fp8(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FP8 E4M3:1 sign + 4 exponent + 3 mantissa bits
    范围: ±448,最小 normal: 2^-6
    Hopper H100 有原生 FP8 Tensor Core

    PyTorch 2.1+ 支持 torch.float8_e4m3fn
    """
    if hasattr(torch, 'float8_e4m3fn'):
        # 真实路径
        max_val = weight.abs().amax().clamp(min=1e-8)
        scale = (max_val / 448.0).clamp(min=1e-12)
        q = (weight / scale).to(torch.float8_e4m3fn)
        return q, scale
    else:
        # 模拟:用 int8 近似(教学)
        max_val = weight.abs().amax().clamp(min=1e-8)
        scale = max_val / 127.0
        q = (weight / scale).round().clamp(-127, 127).to(torch.int8)
        return q, scale


# ============================================================
# 5. W8A8 SmoothQuant
# ============================================================

def smooth_quant_smooth(weight: torch.Tensor,
                        activation_max: torch.Tensor,
                        alpha: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SmoothQuant 核心思想:
        激活有 outlier channel(幅度极大),导致 INT8 量化困难
        把激活的"难度"转移到权重上:
            Y = (X / s) @ (W * s)^T
        其中 s = max(|X|)^alpha / max(|W|)^(1-alpha)
        使 X/s 和 W*s 都便于 INT8 量化
    """
    w_max = weight.abs().amax(dim=0)  # [in]
    x_max = activation_max.clamp(min=1e-8)  # [in]
    s = (x_max.pow(alpha) / w_max.pow(1 - alpha)).clamp(min=1e-8)
    w_smoothed = weight * s.unsqueeze(0)  # [out, in]
    return w_smoothed, s


# ============================================================
# 6. 演示
# ============================================================

def demo():
    torch.manual_seed(42)
    weight = torch.randn(64, 128) * 0.1
    x = torch.randn(8, 128)  # batch=8, in=128

    # ---- INT8 per-channel ----
    q, scale = quantize_int8_per_channel(weight)
    deq = dequantize_int8(q, scale)
    err = (weight - deq).abs().mean()
    print(f"[INT8 per-channel] quantization error: {err:.6f}")
    y_q = int8_matmul(x, q, scale)
    y_ref = x @ weight.t()
    print(f"  matmul output error: {(y_q - y_ref).abs().mean():.6f}")

    # ---- GPTQ ----
    hessian = 2 * x.t() @ x  # 近似
    q_gptq, scales_gptq = gptq_quantize(weight.clone(), hessian.clone())
    print(f"\n[GPTQ] q_weight shape={q_gptq.shape}, dtype={q_gptq.dtype}")

    # ---- AWQ ----
    act_stats = x.abs().mean(dim=0)
    q_awq, g_scales, s_scale = awq_quantize(weight, act_stats)
    w_rec = awq_dequantize(q_awq, g_scales, s_scale)
    print(f"[AWQ] reconstruction error: {(weight - w_rec).abs().mean():.6f}")

    # ---- FP8 ----
    q_fp8, s_fp8 = quantize_fp8(weight)
    print(f"\n[FP8] q dtype={q_fp8.dtype}, scale={s_fp8.item():.6f}")

    # ---- SmoothQuant ----
    act_max = x.abs().amax(dim=0)
    w_smoothed, s_smooth = smooth_quant_smooth(weight, act_max)
    print(f"\n[SmoothQuant] smoothed weight mean abs: "
          f"{w_smoothed.abs().mean():.4f} vs original {weight.abs().mean():.4f}")


if __name__ == "__main__":
    demo()
