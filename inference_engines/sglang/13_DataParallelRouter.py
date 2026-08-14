"""
13_DataParallelRouter.py
========================
SGLang 数据并行 + 缓存感知路由:多个 TP 副本,新请求路由到与前缀匹配度最高的副本。

为什么需要 DP:
    - TP 在单机内扩展到 8 GPU 后,跨机 TP 通信开销大
    - DP:每个节点一个完整副本,独立处理请求
    - 但简单 round-robin DP 会导致前缀命中率低(同一会话散落到不同副本)

SGLang 的 Cache-Aware Router:
    1. 维护每个 DP rank 的 RadixCache 摘要(前缀树的关键节点)
    2. 新请求来时,查询每个 rank 的命中长度
    3. 路由到命中长度最大的 rank
    4. 配合负载均衡(避免某个 rank 过载)

退化策略:
    - 命中长度都 0:round-robin 或 least-load
    - 某个 rank 队列过长:跳过该 rank

跨 rank KV 迁移:
    - 极端场景:某请求的 KV 在 rank A,但 A 过载
    - 通过 Mooncake/NIXL 把 KV 迁移到 rank B
    - 见 14_MooncakeKVTransfer.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque
import time


# ============================================================
# 1. DP Rank 状态
# ============================================================

@dataclass
class DPRank:
    """一个 DP 副本的状态"""
    rank: int
    known_prefixes: List[List[int]] = field(default_factory=list)  # 简化的 RadixCache 摘要
    queue_size: int = 0
    last_dispatch_time: float = 0.0

    def prefix_hit(self, tokens: List[int]) -> int:
        """返回最长前缀命中长度"""
        best = 0
        for pref in self.known_prefixes:
            i = 0
            while i < len(pref) and i < len(tokens) and pref[i] == tokens[i]:
                i += 1
            best = max(best, i)
        return best

    def add_prefix(self, tokens: List[int]):
        self.known_prefixes.append(tokens)
        # 限制摘要大小(实际用 RadixCache,空间高效)
        if len(self.known_prefixes) > 100:
            self.known_prefixes.pop(0)


# ============================================================
# 2. Cache-Aware Router
# ============================================================

class CacheAwareRouter:
    """
    SGLang 数据并行路由器:
        1. 优先路由到 prefix hit 最大的 rank
        2. 负载均衡:queue_size 超过阈值则跳过
        3. 饥饿防护:全 0 hit 时 round-robin
    """

    def __init__(self, num_ranks: int,
                 max_queue_size: int = 32,
                 min_hit_advantage: int = 10):
        self.ranks: List[DPRank] = [DPRank(rank=i) for i in range(num_ranks)]
        self.max_queue_size = max_queue_size
        self.min_hit_advantage = min_hit_advantage  # 命中优势阈值
        self.rr_counter = 0  # round-robin 计数器

    def route(self, tokens: List[int]) -> int:
        """返回路由到的 rank"""
        # 1. 计算每个 rank 的 prefix hit
        hits = [(r, r.prefix_hit(tokens), r.queue_size) for r in self.ranks]

        # 2. 排序:hit 降序 + queue 升序
        hits.sort(key=lambda x: (-x[1], x[2]))

        # 3. 找最佳 rank(同时考虑负载)
        for rank, hit, qsize in hits:
            if qsize >= self.max_queue_size:
                continue
            # 如果 hit 显著大于 0,优先选这个
            if hit >= self.min_hit_advantage:
                rank.queue_size += 1
                rank.add_prefix(tokens)
                rank.last_dispatch_time = time.time()
                return rank.rank

        # 4. 退化:round-robin(在不过载的 rank 中)
        for _ in range(len(self.ranks)):
            r = self.ranks[self.rr_counter % len(self.ranks)]
            self.rr_counter += 1
            if r.queue_size < self.max_queue_size:
                r.queue_size += 1
                r.add_prefix(tokens)
                r.last_dispatch_time = time.time()
                return r.rank

        # 全部过载:返回 least-loaded
        return min(self.ranks, key=lambda r: r.queue_size).rank

    def request_completed(self, rank: int):
        """请求完成时调用"""
        self.ranks[rank].queue_size = max(0, self.ranks[rank].queue_size - 1)


# ============================================================
# 3. 模拟多 rank 推理服务
# ============================================================

class MockDPRank:
    """模拟一个 DP rank 的处理"""

    def __init__(self, rank: int):
        self.rank = rank
        self.processed = 0
        self.total_prefix_saved = 0

    def process(self, tokens: List[int], prefix_hit: int):
        self.processed += 1
        self.total_prefix_saved += prefix_hit
        # 模拟处理时间
        return f"rank_{self.rank}_processed"


# ============================================================
# 4. 演示
# ============================================================

def demo():
    router = CacheAwareRouter(num_ranks=4, max_queue_size=8, min_hit_advantage=5)
    workers = [MockDPRank(i) for i in range(4)]

    # 模拟场景:多个用户对话,每个用户多轮
    system_prompt = list(range(50))  # 50 token system prompt

    # 用户 1 的多轮对话
    user1_rounds = [
        system_prompt + [100, 101, 102],  # round 1
        system_prompt + [100, 101, 102, 200, 201],  # round 2(扩展)
        system_prompt + [100, 101, 102, 200, 201, 300, 302],  # round 3
    ]

    # 用户 2 的多轮(不同 system)
    sys2 = list(range(50, 100))
    user2_rounds = [
        sys2 + [500, 501],
        sys2 + [500, 501, 600, 601],
    ]

    # 用户 3 用 user1 的 system prompt
    user3_rounds = [
        system_prompt + [700, 701],
    ]

    print("=== 路由决策(查看 prefix hit) ===\n")
    for i, tokens in enumerate(user1_rounds + user2_rounds + user3_rounds):
        # 计算每个 rank 的 hit
        hits = [(r.rank, r.prefix_hit(tokens)) for r in router.ranks]
        routed = router.route(tokens)
        workers[routed].process(tokens, max(h[1] for h in hits if h[0] == routed))

        user = "user1" if i < 3 else ("user2" if i < 5 else "user3")
        print(f"Request {i} ({user}, len={len(tokens)}): routed to rank {routed}")
        print(f"  Hits per rank: {hits}")
        print(f"  Worker {routed} queue={router.ranks[routed].queue_size}")

    # 统计
    print("\n=== 统计 ===")
    for w in workers:
        print(f"Rank {w.rank}: processed={w.processed}, "
              f"total_prefix_saved={w.total_prefix_saved} tokens")
    print(f"\n注:同一用户的多轮对话应路由到同一 rank,以复用 system prompt 前缀")


if __name__ == "__main__":
    demo()
