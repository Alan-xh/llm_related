"""
09_FP8Kernel.py
===============
SGLang FP8 Kernel:Hopper H100 原生 FP8 Tensor Core 支持,2x 吞吐提升。

FP8 格式:
    - E4M3:1 sign + 4 exponent + 3 mantissa,范围 ±448,精度低
    - E5M2:1 sign + 5 exponent + 2 mantissa,范围 ±57344,精度更低
    推理一般用 E4M3(精度高),训练 E5M2(范围大,防溢出)

为什么需要:
    - Hopper H100 FP8 Tensor Core 算力 ~2000 TFLOPS,是 BF16 的 2x
    - 显存带宽也减半(8 bit vs 16 bit)
    - DeepSeek-V3 训练用 FP8,推理自然 FP8

关键技术:
    1. Per-tensor scaling:整个 tensor 一个 scale
    2. Per-channel scaling:每个输出 channel 一个 scale(更精确)
    3. Per-block scaling(NVIDIA transformer engine):128x128 block 一个 scale
    4. On-the-fly 反量化:GEMM 时直接用 FP8 输入

本文实现(教学版 Triton):
    - FP8 量化 / 反量化
    - FP8 GEMM Triton kernel
    - 与 BF16 GEMM 对比
"""

from __future__ import annotations
from typing import Tuple
import torch
import triton
import triton.language as tl


# ============================================================
# 1. FP8 量化 / 反量化
# ============================================================

def quantize_to_fp8(tensor: torch.Tensor,
                    scale_mode: str = "per_tensor"
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    把 fp16/bf16 tensor 量化为 FP8 E4M3
    return: (q_tensor fp8, scale)
    """
    if scale_mode == "per_tensor":
        # 整个 tensor 一个 scale
        max_val = tensor.abs().amax().clamp(min=1e-12)
        scale = (max_val / 448.0).clamp(min=1e-12)  # FP8 E4M3 max = 448
        q = (tensor / scale).to(torch.float8_e4m3fn)
        return q, scale

    elif scale_mode == "per_channel":
        # 每行(out channel)一个 scale
        max_val = tensor.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
        scale = (max_val / 448.0).clamp(min=1e-12)
        q = (tensor / scale).to(torch.float8_e4m3fn)
        return q, scale

    elif scale_mode == "per_block":
        # 每 block_size x block_size 一个 scale
        block_size = 128
        n_rows = (tensor.shape[0] + block_size - 1) // block_size
        n_cols = (tensor.shape[1] + block_size - 1) // block_size
        scales = torch.zeros(n_rows, n_cols, dtype=torch.float32, device=tensor.device)
        q = torch.zeros_like(tensor, dtype=torch.float8_e4m3fn)
        for i in range(n_rows):
            for j in range(n_cols):
                r0, r1 = i*block_size, min((i+1)*block_size, tensor.shape[0])
                c0, c1 = j*block_size, min((j+1)*block_size, tensor.shape[1])
                block = tensor[r0:r1, c0:c1]
                mv = block.abs().amax().clamp(min=1e-12)
                s = (mv / 448.0).clamp(min=1e-12)
                scales[i, j] = s
                q[r0:r1, c0:c1] = (block / s).to(torch.float8_e4m3fn)
        return q, scales


def dequantize_fp8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP8 -> fp16"""
    if scale.dim() == 0:
        return q.to(torch.float32) * scale
    elif scale.dim() == 1:
        return q.to(torch.float32) * scale.unsqueeze(1)
    else:
        # per-block 反量化(简化)
        return q.to(torch.float32)


# ============================================================
# 2. FP8 GEMM Triton Kernel
# ============================================================

@triton.jit
def _fp8_gemm_kernel(
    A_ptr,           # [M, K] FP8
    B_ptr,           # [K, N] FP8  (注意:转置存储)
    C_ptr,           # [M, N] FP16/BF16
    a_scale_ptr,     # [1] or [M] or [M//128, K//128]
    b_scale_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    FP8 GEMM:C = A @ B (with scaling)
    A: [M, K] FP8, B: [N, K] FP8 (转置), C: [M, N] FP16

    实际 Hopper 用 TMA + async copy,这里简化。
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 累加器(BF16/FP32)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # K 维度循环
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k

        # 加载 A 和 B 的 block(FP8 -> FP32 内部计算)
        a_block = tl.load(A_ptr + offs_m[:, None] * K + k_offs[None, :],
                          mask=(offs_m[:, None] < M) & (k_offs[None, :] < K),
                          other=0.0).to(tl.float32)
        b_block = tl.load(B_ptr + offs_n[:, None] * K + k_offs[None, :],
                          mask=(offs_n[:, None] < N) & (k_offs[None, :] < K),
                          other=0.0).to(tl.float32)

        # 加载 scale(简化:per-tensor)
        a_scale = tl.load(a_scale_ptr)
        b_scale = tl.load(b_scale_ptr)

        # 应用 scale 后做 GEMM
        acc += tl.dot(a_block * a_scale, b_block.T * b_scale)

    # 写回 C
    tl.store(C_ptr + offs_m[:, None] * N + offs_n[None, :],
             acc.to(C_ptr.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def fp8_matmul(a_fp8: torch.Tensor, b_fp8: torch.Tensor,
               a_scale: torch.Tensor, b_scale: torch.Tensor) -> torch.Tensor:
    """
    a_fp8: [M, K] FP8
    b_fp8: [N, K] FP8  (注意:已经转置)
    return: [M, N] fp16
    """
    M, K = a_fp8.shape
    N, _ = b_fp8.shape
    c = torch.empty(M, N, dtype=torch.float16, device=a_fp8.device)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _fp8_gemm_kernel[grid](
        a_fp8, b_fp8, c, a_scale, b_scale,
        M=M, N=N, K=K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


# ============================================================
# 3. 演示:对比 BF16 vs FP8
# ============================================================

def demo():
    if not torch.cuda.is_available():
        print("CUDA not available, running CPU simulation")
        return _demo_cpu()

    torch.manual_seed(42)
    M, N, K = 512, 512, 256

    # 原始 BF16 权重
    a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    b_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.1

    # BF16 GEMM 基准
    c_bf16 = a_bf16 @ b_bf16.T  # [M, N]
    print(f"BF16 GEMM: {c_bf16.shape}")

    # 量化为 FP8
    a_fp8, a_scale = quantize_to_fp8(a_bf16, "per_tensor")
    b_fp8, b_scale = quantize_to_fp8(b_bf16, "per_tensor")
    print(f"FP8 quantized: a={a_fp8.shape} {a_fp8.dtype}, scale={a_scale.item():.6f}")

    # FP8 GEMM
    c_fp8 = fp8_matmul(a_fp8, b_fp8, a_scale, b_scale)
    print(f"FP8 GEMM: {c_fp8.shape}")

    # 误差
    err = (c_bf16.to(torch.float32) - c_fp8.to(torch.float32)).abs().mean()
    print(f"Mean abs error: {err:.6f}")
    print(f"Relative error: {err / c_bf16.abs().mean():.2%}")

    # 显存对比
    bf16_mem = a_bf16.numel() * 2  # bytes
    fp8_mem = a_fp8.numel() * 1
    print(f"\n显存占用:")
    print(f"  BF16: {bf16_mem / 1024:.0f} KB")
    print(f"  FP8:  {fp8_mem / 1024:.0f} KB")
    print(f"  节省: {(1 - fp8_mem/bf16_mem)*100:.0f}%")


def _demo_cpu():
    """CPU 模拟(无 FP8 支持)"""
    torch.manual_seed(42)
    M, N, K = 64, 64, 32

    a = torch.randn(M, K, dtype=torch.float32) * 0.1
    b = torch.randn(N, K, dtype=torch.float32) * 0.1

    # 模拟 FP8 量化(用 int8 近似)
    a_scale = a.abs().amax() / 127.0
    b_scale = b.abs().amax() / 127.0
    a_q = (a / a_scale).round().clamp(-128, 127).to(torch.int8)
    b_q = (b / b_scale).round().clamp(-128, 127).to(torch.int8)

    # 反量化后做 GEMM
    a_deq = a_q.to(torch.float32) * a_scale
    b_deq = b_q.to(torch.float32) * b_scale
    c_fp8_sim = a_deq @ b_deq.T
    c_ref = a @ b.T

    err = (c_ref - c_fp8_sim).abs().mean()
    print(f"FP8 simulated GEMM (CPU, int8 proxy):")
    print(f"  Output shape: {c_fp8_sim.shape}")
    print(f"  Mean abs error: {err:.6f}")
    print(f"  Relative error: {err / c_ref.abs().mean():.2%}")
    print(f"\n注:真实 FP8 用 H100 FP8 Tensor Core,精度比 int8 高,速度比 BF16 快 2x")


if __name__ == "__main__":
    demo()
