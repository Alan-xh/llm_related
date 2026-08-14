"""
02_CacheAwareScheduler.py
=========================
SGLang 缓存感知调度器:调度时优先选择 RadixCache 命中长的请求,最大化前缀复用。

对比 vLLM 调度:
    vLLM: 按到达时间 FIFO 调度,前缀命中是被动发生的
    SGLang: 主动查询每个请求的 RadixCache 命中长度,优先调度命中长的

核心思想:
    - 命中长的请求:节省 prefill 计算多 -> 优先调度,提升吞吐
    - 配合公平性:避免短任务饿死

调度算法:
    1. 对 waiting 队列每个请求,查询 RadixCache 命中长度
    2. 按 (-prefix_hit_len, arrival_time) 排序
    3. 在 batch 容量内,从队首开始填入

注意:
    - 命中查询必须高效(RadixCache.match 是 O(prefix_len))
    - 长尾请求(无前缀命中)不能饿死,用 starvation prevention
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Deque
from collections import deque
import time


# ============================================================
# 1. 简化 RadixCache(只暴露 match 接口)
# ============================================================

class SimpleRadixCache:
    """简化:只演示命中长度查询"""

    def __init__(self):
        self.known_prefixes: List[List[int]] = []

    def add_prefix(self, tokens: List[int]):
        self.known_prefixes.append(tokens)

    def match(self, tokens: List[int]) -> int:
        """返回最长前缀命中长度"""
        best = 0
        for pref in self.known_prefixes:
            i = 0
            while i < len(pref) and i < len(tokens) and pref[i] == tokens[i]:
                i += 1
            best = max(best, i)
        return best


# ============================================================
# 2. Request
# ============================================================

@dataclass
class Request:
    request_id: int
    token_ids: List[int]
    arrival_time: float
    max_tokens: int = 32
    output_token_ids: List[int] = field(default_factory=list)
    is_finished: bool = False
    prefix_hit_len: int = 0   # 调度时计算
    wait_time: float = 0.0    # 等待时间(用于饥饿防护)

    def append_output(self, token: int, eos: int = 2):
        self.output_token_ids.append(token)
        if token == eos or len(self.output_token_ids) >= self.max_tokens:
            self.is_finished = True


# ============================================================
# 3. Cache-Aware Scheduler
# ============================================================

class CacheAwareScheduler:
    """
    SGLang 风格调度器:
        1. 优先调度 prefix_hit_len 长的请求
        2. 饥饿防护:等待时间过长的请求提升优先级
        3. token budget 限制
    """

    def __init__(self,
                 max_num_seqs: int = 16,
                 max_num_batched_tokens: int = 8192,
                 starvation_threshold: float = 1.0):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.starvation_threshold = starvation_threshold
        self.cache = SimpleRadixCache()

        self.waiting: Deque[Request] = deque()
        self.running: List[Request] = []

    def add_request(self, req: Request):
        # 入队时计算 prefix hit
        req.prefix_hit_len = self.cache.match(req.token_ids)
        self.waiting.append(req)

    def schedule(self, current_time: float) -> List[Request]:
        """返回本 step 要调度的 batch"""
        # 1. running 队列:全部 decode
        decode_batch = list(self.running)

        # 2. 计算 waiting 的优先级
        for req in self.waiting:
            req.wait_time = current_time - req.arrival_time

        # 排序:饥饿请求优先 / 否则按 prefix_hit_len 降序
        def priority(req: Request):
            # 饥饿分:超过阈值则给予高优先级
            starvation_boost = 1e6 if req.wait_time > self.starvation_threshold else 0
            return (starvation_boost - req.prefix_hit_len, req.arrival_time)

        sorted_waiting = sorted(self.waiting, key=priority)

        # 3. 填充 prefill
        spare = self.max_num_seqs - len(decode_batch)
        token_budget = self.max_num_batched_tokens - len(decode_batch)
        to_admit: List[Request] = []
        for req in sorted_waiting:
            if spare <= 0 or token_budget <= 0:
                break
            prefill_len = len(req.token_ids) - req.prefix_hit_len
            if prefill_len > token_budget:
                continue
            to_admit.append(req)
            spare -= 1
            token_budget -= prefill_len

        # 从 waiting 移除,加入 running
        for req in to_admit:
            self.waiting.remove(req)
            # 把新 prefix 加入 cache
            self.cache.add_prefix(req.token_ids)
            self.running.append(req)

        return decode_batch + to_admit


# ============================================================
# 4. 演示
# ============================================================

def demo():
    sched = CacheAwareScheduler(max_num_seqs=8, max_num_batched_tokens=4096)

    # 预先填充一些共享前缀
    system_prompt = list(range(100))  # 100 token system prompt

    # 模拟 5 个请求,3 个共享 system prompt,2 个不共享
    base_time = time.time()
    for i in range(5):
        if i < 3:
            # 共享 system prompt
            tokens = system_prompt + [200+i, 201+i, 202+i]
        else:
            # 独立 prompt
            tokens = [500+i, 501+i, 502+i]
        req = Request(request_id=i, token_ids=tokens,
                      arrival_time=base_time + i * 0.1)
        sched.add_request(req)

    # 第一次调度
    print("--- Step 1 调度 ---")
    batch = sched.schedule(base_time + 0.6)
    for r in batch:
        print(f"  Req {r.request_id}: prefix_hit={r.prefix_hit_len}, "
              f"prefill_tokens={len(r.token_ids)-r.prefix_hit_len}")

    # 演示饥饿防护
    print("\n--- 饥饿场景 ---")
    sched2 = CacheAwareScheduler(max_num_seqs=2, max_num_batched_tokens=4096,
                                  starvation_threshold=0.5)
    # 一个无前缀命中的请求,早到
    req_starved = Request(request_id=0, token_ids=[1000, 1001], arrival_time=base_time)
    sched2.add_request(req_starved)

    # 多个有前缀命中的请求,后到
    sched2.cache.add_prefix([2000, 2001])
    for i in range(5):
        req = Request(request_id=i+1, token_ids=[2000, 2001, 100+i],
                      arrival_time=base_time + 0.1)
        sched2.add_request(req)

    # 短时间调度:有前缀命中的优先
    batch = sched2.schedule(base_time + 0.2)
    print(f"t=0.2: 调度的请求 = {[r.request_id for r in batch]} (期望:有命中优先)")

    # 长时间后:饥饿请求被提升
    batch = sched2.schedule(base_time + 1.0)
    print(f"t=1.0: 调度的请求 = {[r.request_id for r in batch]} "
          f"(期望:饥饿请求 0 提升优先级)")


if __name__ == "__main__":
    demo()
