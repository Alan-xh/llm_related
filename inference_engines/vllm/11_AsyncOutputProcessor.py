"""
11_AsyncOutputProcessor.py
==========================
vLLM Async Output Processor:GPU forward 的同时,CPU 端流式吐 token 给客户端。

为什么需要:
    传统流程:
        GPU forward (50ms) -> CPU detokenize + send (5ms) -> GPU forward -> ...
    GPU 在 CPU 工作期间空闲,吞吐损失。

    异步流程:
        step N GPU forward -> 同时 CPU 处理 step N-1 的输出 -> GPU forward step N+1
    通过双缓冲 + asyncio 让 CPU 后处理与 GPU 计算重叠。

核心组件:
    - OutputProcessor:消费 GPU 输出,detokenize,调用 callback
    - AsyncLLMEngine:用 asyncio 把 forward 和 output 处理解耦
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Callable, AsyncIterator, Any
import asyncio
import time


# ============================================================
# 1. Mock 模型与 Tokenizer
# ============================================================

class MockTokenizer:
    """模拟 tokenizer:token_id -> 字符"""

    def __init__(self):
        self.id_to_token = {i: f"<tok{i}>" for i in range(100)}
        self.id_to_token[2] = "<EOS>"

    def decode(self, token_ids: List[int]) -> str:
        return " ".join(self.id_to_token.get(t, "<?>") for t in token_ids)


class MockModel:
    """模拟 GPU forward,带 sleep 模拟计算时间"""

    def __init__(self, vocab_size: int = 100, forward_time: float = 0.05):
        self.vocab_size = vocab_size
        self.forward_time = forward_time
        self._step = 0

    def forward(self, prompt: List[int]) -> int:
        """返回下一个 token"""
        time.sleep(self.forward_time)  # 模拟 GPU 计算
        self._step += 1
        if self._step > 5:
            return 2  # EOS
        return (self._step * 7) % self.vocab_size


# ============================================================
# 2. Request 与 Output Queue
# ============================================================

@dataclass
class Request:
    request_id: int
    prompt: List[int]
    output_tokens: List[int] = field(default_factory=list)
    is_finished: bool = False


class AsyncOutputProcessor:
    """
    异步输出处理器:
        - 接收 GPU 输出的 token_id
        - 后台 detokenize + 触发 callback
        - 不阻塞 GPU forward
    """

    def __init__(self, tokenizer: MockTokenizer,
                 callback: Optional[Callable[[int, str], None]] = None):
        self.tokenizer = tokenizer
        self.callback = callback
        self.queues: dict[int, asyncio.Queue] = {}

    def register_request(self, request_id: int):
        self.queues[request_id] = asyncio.Queue()

    async def put_token(self, request_id: int, token_id: int):
        await self.queues[request_id].put(token_id)

    async def processor_loop(self, request_id: int, max_tokens: int = 100):
        """后台循环:消费 token -> detokenize -> callback"""
        count = 0
        while count < max_tokens:
            token_id = await self.queues[request_id].get()
            if token_id == 2:  # EOS
                if self.callback:
                    self.callback(request_id, "<EOS>")
                break
            # detokenize(模拟 CPU 工作)
            text = self.tokenizer.decode([token_id])
            # 模拟 CPU 处理时间
            await asyncio.sleep(0.01)
            if self.callback:
                self.callback(request_id, text)
            count += 1


# ============================================================
# 3. AsyncLLMEngine:forward + output 并行
# ============================================================

class AsyncLLMEngine:
    """
    把 GPU forward(同步)和 output 处理(异步)解耦。
    forward 在 executor 线程跑,不阻塞 event loop。
    """

    def __init__(self, model: MockModel, tokenizer: MockTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = AsyncOutputProcessor(tokenizer)

    async def generate(self, request_id: int, prompt: List[int],
                       max_tokens: int = 20) -> List[int]:
        self.processor.register_request(request_id)
        outputs: List[str] = []
        self.processor.callback = lambda rid, txt: outputs.append(txt)

        # 启动 output processor 任务
        proc_task = asyncio.create_task(
            self.processor.processor_loop(request_id, max_tokens)
        )

        # GPU forward 循环(在线程池里跑,避免阻塞 event loop)
        loop = asyncio.get_event_loop()
        cur = list(prompt)
        while True:
            token_id = await loop.run_in_executor(None, self.model.forward, cur)
            await self.processor.put_token(request_id, token_id)
            if token_id == 2:
                break
            cur.append(token_id)

        await proc_task
        return cur[len(prompt):]


# ============================================================
# 4. 同步对比
# ============================================================

def sync_generate(model: MockModel, tokenizer: MockTokenizer,
                  prompt: List[int], max_tokens: int = 20) -> List[int]:
    """传统同步方式:forward -> detokenize -> forward -> ..."""
    cur = list(prompt)
    while True:
        token_id = model.forward(cur)  # GPU
        text = tokenizer.decode([token_id])  # CPU
        # 模拟发送给客户端
        time.sleep(0.01)
        if token_id == 2:
            break
        cur.append(token_id)
    return cur[len(prompt):]


# ============================================================
# 5. 演示
# ============================================================

async def demo_async():
    model = MockModel(forward_time=0.05)
    tokenizer = MockTokenizer()
    engine = AsyncLLMEngine(model, tokenizer)

    t0 = time.perf_counter()
    tokens = await engine.generate(request_id=0, prompt=[10, 20, 30], max_tokens=20)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[Async] generated {len(tokens)} tokens in {elapsed:.0f} ms")
    return elapsed


def demo_sync():
    model = MockModel(forward_time=0.05)
    tokenizer = MockTokenizer()

    t0 = time.perf_counter()
    tokens = sync_generate(model, tokenizer, [10, 20, 30], max_tokens=20)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[Sync]  generated {len(tokens)} tokens in {elapsed:.0f} ms")
    return elapsed


def demo():
    print("Comparing sync vs async output processing...")
    sync_time = demo_sync()
    async_time = asyncio.run(demo_async())
    print(f"\nSpeedup: {sync_time / async_time:.2f}x")
    print("(理想情况下 async 接近 1x,因为 GPU 与 CPU 重叠)")


if __name__ == "__main__":
    demo()
