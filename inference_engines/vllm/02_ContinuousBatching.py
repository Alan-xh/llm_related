"""
02_ContinuousBatching.py
========================
vLLM 连续批处理(Continuous Batching / Iteration-Level Scheduling)。

对比 Static Batching:
    Static: 整个 batch 必须全部完成才能结束 -> 短序列被长序列拖累,bubble 严重
    Continuous: 每个 forward step 后,完成的序列立即出队,新请求入队

核心组件:
    - Request: 一个推理请求(prompt + 已生成 token)
    - Scheduler: 调度器,管理 waiting / running / swapped 三个队列
    - Engine.step(): 一次调度 + 一次 forward

调度循环:
    while True:
        schedule = scheduler.schedule()       # 决定本 step 跑哪些序列
        output = executor.forward(schedule)    # GPU forward
        scheduler.update(output)               # 更新序列状态,移除已完成的
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Deque, Any
from collections import deque
import torch


# ============================================================
# 1. Request 数据结构
# ============================================================

@dataclass
class Request:
    """一个推理请求"""
    request_id: int
    prompt_token_ids: List[int]
    max_tokens: int = 256
    # 运行时状态
    output_token_ids: List[int] = field(default_factory=list)
    is_finished: bool = False
    # 调度状态
    num_computed_tokens: int = 0  # 已经 prefill 过的 token 数

    def all_tokens(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    def append_output(self, token: int, eos_token_id: int = 2):
        self.output_token_ids.append(token)
        if token == eos_token_id or len(self.output_token_ids) >= self.max_tokens:
            self.is_finished = True


# ============================================================
# 2. Scheduler
# ============================================================

class Scheduler:
    """
    vLLM 调度器简化版。
    三个队列:
        waiting:  还未开始 prefill 的新请求
        running:  正在 decode 的请求
        swapped:  因显存不足被换出到 CPU 的请求(本文简化省略换出逻辑)
    """

    def __init__(self,
                 max_num_seqs: int = 32,
                 max_num_batched_tokens: int = 8192):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens

        self.waiting: Deque[Request] = deque()
        self.running: List[Request] = []
        self.swapped: Deque[Request] = deque()

    def add_request(self, req: Request):
        self.waiting.append(req)

    @dataclass
    class Schedule:
        """一个 step 的调度结果"""
        prefill_reqs: List[Request]    # 本 step 要 prefill 的请求
        decode_reqs: List[Request]     # 本 step 要 decode 的请求
        # 简化:实际 vLLM 还会传 block_table、seq_lens 等

    def schedule(self) -> "Scheduler.Schedule":
        """
        调度策略(简化):
            1. running 队列优先保留(已 decode 的不中断)
            2. 有空闲 slot 时,从 waiting 队列拉新请求 prefill
            3. 受 max_num_seqs / max_num_batched_tokens 约束
        """
        # 1. running 队列:全部 decode
        decode_reqs = list(self.running)

        # 2. 限制总序列数
        spare_slots = self.max_num_seqs - len(decode_reqs)
        prefill_reqs: List[Request] = []

        # token budget: prefill 的 token 总数限制
        token_budget = self.max_num_batched_tokens - len(decode_reqs)  # decode 各占 1

        # 3. 从 waiting 拉 prefill
        while self.waiting and spare_slots > 0 and token_budget > 0:
            req = self.waiting[0]
            num_prefill_tokens = len(req.prompt_token_ids)
            if num_prefill_tokens > token_budget:
                # 超过 budget,留给 chunked prefill 处理(见 03 文件)
                break
            self.waiting.popleft()
            prefill_reqs.append(req)
            spare_slots -= 1
            token_budget -= num_prefill_tokens

        # 4. 把 prefill 完成后加入 running
        for req in prefill_reqs:
            req.num_computed_tokens = len(req.prompt_token_ids)
            self.running.append(req)

        return self.Schedule(prefill_reqs=prefill_reqs, decode_reqs=decode_reqs)


# ============================================================
# 3. Mock Executor(模拟 GPU forward)
# ============================================================

class MockExecutor:
    """模拟 LLM forward,返回每个序列下一个 token 的 logits"""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.rng = torch.Generator().manual_seed(0)

    def forward(self, schedule: Scheduler.Schedule) -> List[int]:
        num_tokens = sum(len(r.prompt_token_ids) for r in schedule.prefill_reqs)
        num_tokens += len(schedule.decode_reqs)
        # 模拟:每个 token 采样一个 next token
        next_tokens = torch.randint(0, self.vocab_size, (num_tokens,),
                                    generator=self.rng).tolist()
        return next_tokens


# ============================================================
# 4. Engine:调度 + forward 主循环
# ============================================================

class Engine:
    """vLLM Engine 简化版主循环"""

    def __init__(self):
        self.scheduler = Scheduler()
        self.executor = MockExecutor()

    def add_request(self, req: Request):
        self.scheduler.add_request(req)

    def has_unfinished(self) -> bool:
        return bool(self.scheduler.waiting or self.scheduler.running)

    def step(self) -> List[Request]:
        """一个 step = 一次调度 + 一次 forward + 更新状态"""
        schedule = self.scheduler.schedule()
        next_tokens = self.executor.forward(schedule)

        # 把 next_token 分配给请求
        idx = 0
        finished = []
        for req in schedule.prefill_reqs:
            token = next_tokens[idx]; idx += 1
            req.append_output(token)
            if req.is_finished:
                finished.append(req)
                self.scheduler.running.remove(req)
        for req in schedule.decode_reqs:
            token = next_tokens[idx]; idx += 1
            req.append_output(token)
            if req.is_finished:
                finished.append(req)
                self.scheduler.running.remove(req)
        return finished


# ============================================================
# 5. 演示
# ============================================================

def demo():
    engine = Engine()

    # 模拟 5 个不同长度的请求并发到达
    for i in range(5):
        req = Request(request_id=i,
                      prompt_token_ids=list(range(10 + i * 5)),  # 不同长度
                      max_tokens=8)
        engine.add_request(req)

    step = 0
    while engine.has_unfinished():
        finished = engine.step()
        step += 1
        running = len(engine.scheduler.running)
        waiting = len(engine.scheduler.waiting)
        print(f"step {step}: running={running}, waiting={waiting}, "
              f"finished={len(finished)}")
        if step > 100:
            break

    print("All requests finished.")


if __name__ == "__main__":
    demo()
