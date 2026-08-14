"""
03_OverlapScheduler.py
======================
SGLang Overlap Scheduler:CPU 调度与 GPU 计算重叠,默认开启。

为什么需要:
    传统流程(同步):
        [CPU 调度 N][GPU forward N][CPU 后处理 N][CPU 调度 N+1][GPU forward N+1]...
        GPU 在 CPU 工作期间空闲。

    重叠流程:
        [CPU 调度 N][GPU forward N] + 同时 [CPU 后处理 N-1 + 调度 N+1]
        通过双缓冲,CPU 准备下一步的 batch 时 GPU 在跑当前 step。

关键实现:
    1. 双 batch buffer:current_batch 和 next_batch
    2. CPU 线程做:sample、detokenize、调度、apply grammar mask、copy 输入到 GPU
    3. GPU 线程做:forward
    4. 两者通过 asyncio / 队列同步

SGLang 的实现:
    srt/managers/scheduler.py:Scheduler.event_loop_overlap()
    - 两个事件循环交替运行
    - 一个负责 GPU forward,一个负责 CPU 准备
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Deque, Tuple
from collections import deque
import asyncio
import time
import random


# ============================================================
# 1. 模拟组件
# ============================================================

@dataclass
class Request:
    request_id: int
    token_ids: List[int]
    max_tokens: int = 20
    output_token_ids: List[int] = field(default_factory=list)
    is_finished: bool = False

    def append(self, t: int, eos: int = 2):
        self.output_token_ids.append(t)
        if t == eos or len(self.output_token_ids) >= self.max_tokens:
            self.is_finished = True


class MockGPU:
    """模拟 GPU forward,带固定耗时"""
    def __init__(self, forward_time: float = 0.05):
        self.forward_time = forward_time

    async def forward(self, batch: List[Request]) -> List[int]:
        await asyncio.sleep(self.forward_time)
        return [random.randint(0, 100) for _ in batch]


class MockCPU:
    """模拟 CPU 后处理 + 调度,带固定耗时"""

    def __init__(self, schedule_time: float = 0.02, postprocess_time: float = 0.01):
        self.schedule_time = schedule_time
        self.postprocess_time = postprocess_time

    async def prepare_next_batch(self,
                                  waiting: Deque[Request],
                                  running: List[Request],
                                  max_seqs: int) -> List[Request]:
        """调度:决定下一步跑哪些请求"""
        await asyncio.sleep(self.schedule_time)
        # 移除完成的
        running[:] = [r for r in running if not r.is_finished]
        # 加入新请求
        while waiting and len(running) < max_seqs:
            running.append(waiting.popleft())
        return list(running)

    async def postprocess(self, batch: List[Request], tokens: List[int]):
        """后处理:采样、detokenize、检查完成"""
        await asyncio.sleep(self.postprocess_time)
        for r, t in zip(batch, tokens):
            r.append(t)


# ============================================================
# 2. 同步引擎(对比基线)
# ============================================================

class SyncEngine:
    """传统同步引擎:CPU 和 GPU 串行"""

    def __init__(self, gpu: MockGPU, cpu: MockCPU, max_seqs: int = 8):
        self.gpu = gpu
        self.cpu = cpu
        self.max_seqs = max_seqs
        self.waiting: Deque[Request] = deque()
        self.running: List[Request] = []

    def add_request(self, req: Request):
        self.waiting.append(req)

    async def run(self, num_steps: int = 20):
        for _ in range(num_steps):
            batch = await self.cpu.prepare_next_batch(
                self.waiting, self.running, self.max_seqs)
            if not batch:
                break
            tokens = await self.gpu.forward(batch)
            await self.cpu.postprocess(batch, tokens)


# ============================================================
# 3. Overlap 引擎(关键)
# ============================================================

class OverlapEngine:
    """
    CPU 和 GPU 重叠:
        step N: GPU 跑 batch_N,同时 CPU 准备 batch_N+1 + 后处理 batch_N-1
    通过两个 asyncio task 并行实现。
    """

    def __init__(self, gpu: MockGPU, cpu: MockCPU, max_seqs: int = 8):
        self.gpu = gpu
        self.cpu = cpu
        self.max_seqs = max_seqs
        self.waiting: Deque[Request] = deque()
        self.running: List[Request] = []

    def add_request(self, req: Request):
        self.waiting.append(req)

    async def run(self, num_steps: int = 20):
        # 初始:准备 batch_0
        batch_cur = await self.cpu.prepare_next_batch(
            self.waiting, self.running, self.max_seqs)

        for step in range(num_steps):
            if not batch_cur:
                break

            # 同时启动 GPU forward(当前 batch) 和 CPU prepare_next(下一 batch)
            gpu_task = asyncio.create_task(self.gpu.forward(batch_cur))
            cpu_task = asyncio.create_task(self.cpu.prepare_next_batch(
                self.waiting, self.running, self.max_seqs))

            # 等两者都完成
            tokens, batch_next = await asyncio.gather(gpu_task, cpu_task)

            # 后处理当前 batch(可以和下一个 GPU forward 重叠)
            # 这里简化为顺序执行,实际 SGLang 也是重叠的
            await self.cpu.postprocess(batch_cur, tokens)

            batch_cur = batch_next


# ============================================================
# 4. 演示:对比同步 vs 重叠
# ============================================================

async def demo_async():
    random.seed(42)

    def make_requests(n=10):
        return [Request(request_id=i, token_ids=list(range(5)),
                        max_tokens=8) for i in range(n)]

    # 同步
    sync = SyncEngine(MockGPU(0.05), MockCPU(0.02, 0.01), max_seqs=8)
    for r in make_requests():
        sync.add_request(r)
    t0 = time.perf_counter()
    await sync.run(num_steps=30)
    sync_time = time.perf_counter() - t0

    # 重叠
    overlap = OverlapEngine(MockGPU(0.05), MockCPU(0.02, 0.01), max_seqs=8)
    for r in make_requests():
        overlap.add_request(r)
    t0 = time.perf_counter()
    await overlap.run(num_steps=30)
    overlap_time = time.perf_counter() - t0

    print(f"Sync engine:   {sync_time*1000:.0f} ms")
    print(f"Overlap engine: {overlap_time*1000:.0f} ms")
    print(f"Speedup: {sync_time/overlap_time:.2f}x")
    print(f"\n注:每步 sync 耗时 = schedule(20) + gpu(50) + post(10) = 80ms")
    print(f"    每步 overlap 耗时 ≈ max(schedule(20), gpu(50)) + post(10) ≈ 60ms")
    print(f"    理论加速 ≈ 80/60 = 1.33x")


def demo():
    asyncio.run(demo_async())


if __name__ == "__main__":
    demo()
