"""
03_ChunkedPrefill.py
====================
vLLM Chunked Prefill:把长 prompt 切分成多个 chunk,逐 step 与 decode 混合调度。

为什么需要:
    长 prompt(如 32K)一次 prefill 会占满 GPU 几百 ms,导致已 running 的
    decode 序列延迟飙升(TTFT 好但 ITL 差)。

解决方案:
    - 把 prefill 切成 max_chunk_tokens(默认 2048~8192)大小的 chunk
    - 每个 step 至多 1 个 prefill chunk + 多个 decode,混合在一个 batch
    - 同一序列的 prefill chunk 和它自己的 decode 不能同 step(KV 依赖)

调度策略:
    - 单 prefill chunk < token_budget:与 decode 混合
    - 单 prefill chunk >= token_budget:纯 prefill step,decode 让路
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Deque
from collections import deque
import torch


@dataclass
class ChunkedRequest:
    """支持 chunked prefill 的请求"""
    request_id: int
    prompt_token_ids: List[int]
    max_tokens: int = 256
    output_token_ids: List[int] = field(default_factory=list)
    is_finished: bool = False
    # prefill 进度
    num_prefilled: int = 0    # 已 prefill 的 token 数
    is_prefill_done: bool = False

    def next_prefill_chunk(self, chunk_size: int) -> List[int]:
        """取下一个 prefill chunk"""
        end = min(self.num_prefilled + chunk_size, len(self.prompt_token_ids))
        chunk = self.prompt_token_ids[self.num_prefilled:end]
        self.num_prefilled = end
        if end >= len(self.prompt_token_ids):
            self.is_prefill_done = True
        return chunk

    def append_output(self, token: int, eos: int = 2):
        self.output_token_ids.append(token)
        if token == eos or len(self.output_token_ids) >= self.max_tokens:
            self.is_finished = True


@dataclass
class ScheduledBatch:
    """一个 step 的 batch"""
    # (request, tokens_to_run) 列表
    items: List = field(default_factory=list)

    def total_tokens(self) -> int:
        return sum(len(toks) for _, toks in self.items)

    def is_empty(self) -> bool:
        return not self.items


class ChunkedPrefillScheduler:
    """
    简化版 chunked prefill 调度器。
    vLLM 实际策略见 scheduler.py:Scheduler._schedule_chunked_prefill
    """

    def __init__(self,
                 max_num_seqs: int = 32,
                 max_num_batched_tokens: int = 8192,
                 chunk_size: int = 2048,
                 eos_token_id: int = 2):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.chunk_size = chunk_size
        self.eos_token_id = eos_token_id

        self.waiting: Deque[ChunkedRequest] = deque()
        self.running: List[ChunkedRequest] = []  # 已 prefill 完,正在 decode

    def add_request(self, req: ChunkedRequest):
        self.waiting.append(req)

    def schedule(self) -> ScheduledBatch:
        batch = ScheduledBatch()
        token_budget = self.max_num_batched_tokens

        # ----- 1. 优先填 decode(running 队列) -----
        for req in self.running:
            if token_budget <= 0:
                break
            # decode 每次 1 token
            batch.items.append((req, [req.output_token_ids[-1] if req.output_token_ids
                                      else req.prompt_token_ids[-1]]))
            token_budget -= 1

        # ----- 2. 填 prefill chunk -----
        # vLLM 默认一个 step 只处理一个 prefill chunk(避免抢占 decode 太多)
        if self.waiting and token_budget > 0:
            req = self.waiting[0]
            chunk = req.next_prefill_chunk(min(self.chunk_size, token_budget))
            batch.items.append((req, chunk))
            token_budget -= len(chunk)

            # 如果该请求 prefill 完成,加入 running(下个 step 开始 decode)
            if req.is_prefill_done:
                self.waiting.popleft()
                self.running.append(req)

        return batch

    def update(self, batch: ScheduledBatch, next_tokens: List[int]):
        """forward 后更新状态"""
        idx = 0
        finished = []
        for req, tokens in batch.items:
            # decode 项(只 1 个 token)产出新 token
            if len(tokens) == 1 and req.is_prefill_done:
                req.append_output(next_tokens[idx], self.eos_token_id)
                idx += 1
                if req.is_finished:
                    finished.append(req)
                    self.running.remove(req)
            else:
                # prefill 项:本 step 不产出 token(下一个 step 才 decode 第一个)
                # 实际实现里 prefill 最后一个位置也会算 logits 产出首 token
                idx += len(tokens)
        return finished


class MockExecutor:
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.rng = torch.Generator().manual_seed(0)

    def forward(self, batch: ScheduledBatch) -> List[int]:
        total = batch.total_tokens()
        return torch.randint(0, self.vocab_size, (total,),
                             generator=self.rng).tolist()


# ============================================================
# 演示:对比 pure prefill vs chunked prefill
# ============================================================

def demo():
    sched = ChunkedPrefillScheduler(max_num_seqs=4,
                                    max_num_batched_tokens=4096,
                                    chunk_size=512)
    exe = MockExecutor()

    # 模拟:1 个长 prompt(3000 token)+ 3 个短请求正在 decode
    long_req = ChunkedRequest(request_id=0,
                              prompt_token_ids=list(range(3000)),
                              max_tokens=10)
    sched.add_request(long_req)

    # 把 3 个已经在 decode 的请求加入 running(模拟)
    for i in range(1, 4):
        r = ChunkedRequest(request_id=i,
                           prompt_token_ids=[100+i]*5,  # 已 prefill
                           max_tokens=6)
        r.num_prefilled = 5
        r.is_prefill_done = True
        r.output_token_ids = [200+i]  # 已生成 1 token
        sched.running.append(r)

    step = 0
    while sched.waiting or sched.running:
        step += 1
        batch = sched.schedule()
        if batch.is_empty():
            break
        toks = exe.forward(batch)
        finished = sched.update(batch, toks)
        prefill_tok = sum(len(t) for r, t in batch.items if not r.is_prefill_done)
        decode_tok = sum(len(t) for r, t in batch.items if r.is_prefill_done)
        print(f"step {step}: prefill_tokens={prefill_tok}, decode_tokens={decode_tok}, "
              f"running={len(sched.running)}, waiting={len(sched.waiting)}, "
              f"finished={len(finished)}")
        if step > 30:
            break


if __name__ == "__main__":
    demo()
