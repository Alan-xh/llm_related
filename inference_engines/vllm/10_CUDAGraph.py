"""
10_CUDAGraph.py
===============
vLLM CUDA Graph:对 decode 阶段(每步只多 1 token)捕获固定形状的 kernel 序列,
避免每次 forward 都要 CPU launcher 开销。

为什么需要:
    - Decode 阶段每步 forward 的 GPU 计算量小(几个 GEMM)
    - 但每个 kernel 都要 CPU 端 launch,launch 开销可能 > 计算
    - CUDA Graph 把整段 kernel 序列录制成"图",一次 replay,避免反复 launch

核心 trick:
    1. 静态形状:输入 tensor 形状固定(占位符),每次只 copy 数据进去
    2. 多桶策略:为不同 batch size 各捕获一张图(bucket)
    3. 输入/输出 buffer 用 static tensor,通过 copy_ 替换数据

本文实现:
    - 简化版 CUDAGraphWrapper
    - 多 bucket 捕获与回放
    - 演示对比 eager vs graph 模式
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch
import torch.nn as nn


# ============================================================
# 1. 简单模型(模拟 vLLM decode step)
# ============================================================

class DecodeModel(nn.Module):
    """模拟一个 decode step:RMSNorm + Linear + RMSNorm + Linear"""

    def __init__(self, hidden=512, intermediate=1024, vocab=1000):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, intermediate)
        self.fc2 = nn.Linear(intermediate, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, x):
        # x: [batch, hidden]
        h = self.norm1(x)
        h = torch.relu(self.fc1(h))
        h = self.fc2(h)
        h = self.norm2(h + x)
        return self.head(h)


# ============================================================
# 2. CUDA Graph Wrapper
# ============================================================

class CUDAGraphRunner:
    """
    对固定 batch size 的 decode forward 捕获 CUDA Graph。
    """

    def __init__(self, model: nn.Module, batch_size: int, hidden: int):
        self.batch_size = batch_size
        self.hidden = hidden

        # 静态输入/输出 buffer(CUDA Graph 必须 static)
        self.static_input = torch.zeros(batch_size, hidden,
                                        device="cuda", dtype=torch.float32)
        self.static_output: Optional[torch.Tensor] = None

        # warmup(必须先做几步 eager forward,初始化 lazy buffer)
        for _ in range(3):
            with torch.no_grad():
                _ = model(self.static_input)

        # 捕获图
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            with torch.no_grad():
                self.static_output = model(self.static_input)

        self.model = model

    def replay(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """回放图:copy 输入到 static buffer,replay,读 static_output"""
        assert input_tensor.shape == self.static_input.shape
        self.static_input.copy_(input_tensor)
        self.graph.replay()
        return self.static_output


class CUDAGraphBatchedRunner:
    """
    多 bucket:CUDA Graph 不支持动态 batch,所以为不同 batch size 各捕获一张图。
    vLLM 实际实现:vllm/worker/worker.py:CUDAGraphWrappers
    """

    def __init__(self, model: nn.Module,
                 batch_sizes: List[int],
                 hidden: int):
        self.runners: Dict[int, CUDAGraphRunner] = {}
        for bs in batch_sizes:
            self.runners[bs] = CUDAGraphRunner(model, bs, hidden)
        self.batch_sizes = sorted(batch_sizes, reverse=True)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        bs = input_tensor.shape[0]
        # 找最小的能容纳 bs 的 bucket
        target_bs = None
        for cand in reversed(self.batch_sizes):
            if cand >= bs:
                target_bs = cand
                break
        if target_bs is None:
            # 超出最大 bucket,回退到 eager
            return self.runners[self.batch_sizes[0]].model(input_tensor)

        # padding 到 target_bs
        if target_bs > bs:
            pad = torch.zeros(target_bs - bs, input_tensor.shape[1],
                              device=input_tensor.device, dtype=input_tensor.dtype)
            padded = torch.cat([input_tensor, pad], dim=0)
        else:
            padded = input_tensor

        out = self.runners[target_bs].replay(padded)
        return out[:bs]  # 去 padding


# ============================================================
# 3. 演示:对比 eager vs graph
# ============================================================

def demo():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping demo")
        return

    torch.manual_seed(42)
    model = DecodeModel(hidden=512, intermediate=1024, vocab=1000).cuda().eval()
    hidden = 512

    # 测试不同 batch size
    batch_sizes = [1, 4, 8, 16]
    runner = CUDAGraphBatchedRunner(model, batch_sizes=[1, 4, 8, 16, 32],
                                    hidden=hidden)

    # ---- Eager 模式计时 ----
    import time
    x = torch.randn(8, hidden, device="cuda")
    # warmup
    for _ in range(10):
        _ = model(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(1000):
        _ = model(x)
    torch.cuda.synchronize()
    eager_time = (time.perf_counter() - t0) * 1000

    # ---- Graph 模式计时 ----
    for _ in range(10):
        _ = runner.forward(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(1000):
        _ = runner.forward(x)
    torch.cuda.synchronize()
    graph_time = (time.perf_counter() - t0) * 1000

    print(f"Eager mode: {eager_time:.2f} ms / 1000 steps")
    print(f"Graph mode: {graph_time:.2f} ms / 1000 steps")
    print(f"Speedup: {eager_time / graph_time:.2f}x")

    # 正确性验证
    y_eager = model(x)
    y_graph = runner.forward(x)
    print(f"Output match: {torch.allclose(y_eager, y_graph, atol=1e-4)}")


if __name__ == "__main__":
    demo()
