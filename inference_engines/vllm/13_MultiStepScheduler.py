"""
13_MultiStepScheduler.py
========================
vLLM Multi-Step Scheduling:一次 CPU 调度规划多步 forward,减少 CPU-GPU 同步开销。

为什么需要:
    每步 forward 后都要 CPU 同步(取 output、更新状态、调度下一步),
    CPU 一次调度可能 ~1ms,如果 GPU forward 才 5ms,占比 20%。

解决方案:
    - 一次调度规划 K 步(比如 8 步)
    - 这 K 步用相同的 batch 配置(无新请求加入,无 prefix cache miss)
    - GPU 跑 K 步只在中断点同步一次 CPU
    - 配合 CUDA Graph 进一步加速

限制:
    - K 步内不能添加新请求(否则 batch 变化破坏 graph)
    - 不能处理 prefill(只适合纯 decode)
    - 需要预留足够 KV cache 空间(预分配 K 步的 block)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Deque
from collections import deque
import torch


# ============================================================
# 1. 简化 Request
# ============================================================

@dataclass
class Request:
    request_id: int
    prompt_token_ids: List[int]
    max_tokens: int = 32
    output_token_ids: List[int] = field(default_factory=list)
    is_finished: bool = False
    num_computed: int = 0

    def append_output(self, token: int, eos: int = 2):
        self.output_token_ids.append(token)
        if token == eos or len(self.output_token_ids) >= self.max_tokens:
            self.is_finished = True


# ============================================================
# 2. Multi-Step Scheduler
# ============================================================

class MultiStepScheduler:
    """
    一次调度 K 步,期间 batch 不变。

    策略:
        1. 在 step % K == 0 时执行一次完整调度(prefill + 新请求入队)
        2. 在 step % K != 0 时,跳过调度,直接复用上次的 batch
        3. K 步结束后,统一处理输出、检查完成、释放资源
    """

    def __init__(self,
                 max_num_seqs: int = 16,
                 max_new_tokens_per_step: int = 1,
                 num_steps: int = 8):
        self.max_num_seqs = max_num_seqs
        self.num_steps = num_steps  # K
        self.waiting: Deque[Request] = deque()
        self.running: List[Request] = []
        self.current_batch: List[Request] = []
        self.step_in_window = 0  # 当前在 K 步窗口中的位置

    def add_request(self, req: Request):
        self.waiting.append(req)

    def schedule(self) -> List[Request]:
        """
        返回本 step 要跑的 batch。
        在 K 步窗口起点:重新调度;其他步:返回上次的 batch。
        """
        if self.step_in_window == 0:
            # ---- 完整调度 ----
            # 1. 清理已完成的
            self.running = [r for r in self.running if not r.is_finished]

            # 2. 从 waiting 拉新请求(只有窗口起点允许加入)
            while (self.waiting and
                   len(self.running) < self.max_num_seqs):
                req = self.waiting.popleft()
                req.num_computed = len(req.prompt_token_ids)
                self.running.append(req)

            self.current_batch = list(self.running)
        else:
            # ---- 复用上次 batch,但移除已完成的 ----
            self.current_batch = [r for r in self.current_batch if not r.is_finished]

        self.step_in_window = (self.step_in_window + 1) % self.num_steps
        return self.current_batch


# ============================================================
# 3. Mock Executor
# ============================================================

class MockExecutor:
    def __init__(self, vocab_size: int = 1000, forward_time_ms: float = 5.0):
        self.vocab_size = vocab_size
        self.forward_time_ms = forward_time_ms
        self.rng = torch.Generator().manual_seed(0)
        self.scheduling_count = 0

    def forward(self, batch: List[Request]) -> List[int]:
        # 模拟 GPU forward
        self.scheduling_count += 1
        return torch.randint(0, self.vocab_size, (len(batch),),
                             generator=self.rng).tolist()


# ============================================================
# 4. 对比:单步调度 vs 多步调度
# ============================================================

def run_single_step(scheduler: MultiStepScheduler, executor: MockExecutor,
                    total_steps: int = 32) -> int:
    """单步模式:每步都做完整调度"""
    scheduler.num_steps = 1
    scheduler.step_in_window = 0
    executor.scheduling_count = 0
    for _ in range(total_steps):
        batch = scheduler.schedule()
        if not batch:
            break
        tokens = executor.forward(batch)
        for r, t in zip(batch, tokens):
            r.append_output(t)
    return executor.scheduling_count


def run_multi_step(scheduler: MultiStepScheduler, executor: MockExecutor,
                   total_steps: int = 32, K: int = 8) -> int:
    """多步模式:K 步只调度 1 次"""
    scheduler.num_steps = K
    scheduler.step_in_window = 0
    executor.scheduling_count = 0
    for _ in range(total_steps):
        batch = scheduler.schedule()
        if not batch:
            break
        tokens = executor.forward(batch)
        for r, t in zip(batch, tokens):
            r.append_output(t)
    return executor.scheduling_count


# ============================================================
# 5. 演示
# ============================================================

def demo():
    # 准备请求
    def make_requests():
        return [Request(request_id=i,
                        prompt_token_ids=list(range(10)),
                        max_tokens=20) for i in range(4)]

    sched1 = MultiStepScheduler(max_num_seqs=8, num_steps=1)
    for r in make_requests():
        sched1.add_request(r)
    exe1 = MockExecutor(forward_time_ms=5.0)

    sched2 = MultiStepScheduler(max_num_seqs=8, num_steps=8)
    for r in make_requests():
        sched2.add_request(r)
    exe2 = MockExecutor(forward_time_ms=5.0)

    total_steps = 32
    single_calls = run_single_step(sched1, exe1, total_steps)
    multi_calls = run_multi_step(sched2, exe2, total_steps, K=8)

    print(f"Total forward steps: {total_steps}")
    print(f"Single-step mode: {single_calls} scheduling iterations")
    print(f"Multi-step (K=8):  {multi_calls} scheduling iterations")
    print(f"CPU scheduling overhead reduction: "
          f"{(1 - multi_calls/single_calls)*100:.1f}%")
    print(f"\n注:实际收益取决于 CPU 调度开销占 forward 时间的比例。")
    print(f"   若 GPU forward 很短(decode 阶段),收益显著;若 prefill 长,收益小。")


if __name__ == "__main__":
    demo()
