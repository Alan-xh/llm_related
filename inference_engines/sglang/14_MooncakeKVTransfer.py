"""
14_MooncakeKVTransfer.py
========================
SGLang Mooncake KV Transfer Engine:跨节点 KV cache 传输,支持 P/D 分离。

为什么需要:
    Prefill/Decode 分离架构:
        - Prefill 节点:算力强,处理长 prompt
        - Decode 节点:显存大,跑 decode
        - Prefill 完成后,把 KV cache 传给 Decode 节点

    传统方案:Prefill 完成后,通过 TCP/NCCL 传 KV
        - KV 大(几 GB),传输慢
        - 阻塞 prefill 节点

Mooncake(月球)Transfer Engine:
    - 针对 KV cache 传输优化的 RDMA 库
    - 利用 KV cache 的局部性(只有最近 token 需要传)
    - 异步传输,不阻塞 prefill
    - 与 Paged KV Cache 集成(直接传 page)

核心机制:
    1. Prefill 节点:把 KV 写入 local GPU
    2. 同时把 KV 序列化到 CPU 内存(via P2P copy)
    3. RDMA WRITE 直接写到 Decode 节点的 CPU 内存
    4. Decode 节点 P2P copy 到 GPU
    5. 全程 GPU 不阻塞

本文实现:
    - 模拟 KV Transfer 流程
    - 与 Paged KV Cache 集成
    - 异步传输
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import asyncio
import time
import torch


# ============================================================
# 1. Paged KV Cache(可序列化)
# ============================================================

@dataclass
class PagedKVCache:
    """简化版 paged KV cache,支持序列化传输"""
    k_cache: torch.Tensor  # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: torch.Tensor
    block_size: int

    def serialize_blocks(self, block_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """把指定 block 序列化为连续 tensor(用于传输)"""
        return self.k_cache[block_ids].cpu(), self.v_cache[block_ids].cpu()

    def deserialize_and_write(self, k_data: torch.Tensor, v_data: torch.Tensor,
                               block_ids: List[int]):
        """把传输过来的数据写回 cache"""
        self.k_cache[block_ids] = k_data.to(self.k_cache.device)
        self.v_cache[block_ids] = v_data.to(self.v_cache.device)


# ============================================================
# 2. Mooncake Transfer Engine(模拟)
# ============================================================

class MooncakeTransferEngine:
    """
    模拟 Mooncake KV transfer engine。
    实际用 RDMA + GPU P2P,这里用 asyncio 模拟异步传输。
    """

    def __init__(self, node_name: str):
        self.node_name = node_name
        self.transfer_bandwidth_gbps = 100  # 100 GB/s RDMA
        self.pending_transfers: Dict[str, asyncio.Queue] = {}

    def register_remote(self, remote_node: str):
        self.pending_transfers[remote_node] = asyncio.Queue()

    async def send_kv(self, remote_node: str,
                       k_data: torch.Tensor, v_data: torch.Tensor,
                       session_id: str):
        """异步发送 KV 到远端"""
        # 计算传输时间(模拟)
        data_size_bytes = (k_data.numel() + v_data.numel()) * k_data.element_size()
        transfer_time = data_size_bytes / (self.transfer_bandwidth_gbps * 1e9)

        # 模拟 RDMA WRITE(异步)
        await asyncio.sleep(transfer_time)
        # 远端"接收"
        await self.pending_transfers[remote_node].put((session_id, k_data, v_data))

        return transfer_time

    async def recv_kv(self, remote_node: str,
                       timeout: Optional[float] = None) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """异步接收 KV"""
        return await self.pending_transfers[self.node_name].get()


# ============================================================
# 3. Prefill Worker + Decode Worker
# ============================================================

class PrefillWorker:
    """Prefill 节点:处理长 prompt,完成后把 KV 传给 Decode 节点"""

    def __init__(self, kv_cache: PagedKVCache,
                 transfer_engine: MooncakeTransferEngine):
        self.kv_cache = kv_cache
        self.transfer_engine = transfer_engine

    async def prefill_and_transfer(self,
                                    prompt_token_ids: List[int],
                                    decode_worker_node: str,
                                    session_id: str,
                                    prefill_time: float = 0.1) -> Tuple[int, float]:
        """
        1. Prefill(模拟)
        2. 异步传输 KV 到 Decode 节点
        return: (block_ids, transfer_time)
        """
        # 1. Prefill
        await asyncio.sleep(prefill_time)

        # 2. 分配 block 并写入 KV(模拟)
        num_tokens = len(prompt_token_ids)
        num_blocks = (num_tokens + self.kv_cache.block_size - 1) // self.kv_cache.block_size
        block_ids = list(range(num_blocks))

        # 模拟写入(实际由 LLM forward 完成)
        for bid in block_ids:
            self.kv_cache.k_cache[bid] = torch.randn_like(self.kv_cache.k_cache[bid])
            self.kv_cache.v_cache[bid] = torch.randn_like(self.kv_cache.v_cache[bid])

        # 3. 序列化 + 传输
        k_data, v_data = self.kv_cache.serialize_blocks(block_ids)
        transfer_time = await self.transfer_engine.send_kv(
            decode_worker_node, k_data, v_data, session_id
        )

        return block_ids, transfer_time


class DecodeWorker:
    """Decode 节点:接收 KV,然后开始 decode"""

    def __init__(self, kv_cache: PagedKVCache,
                 transfer_engine: MooncakeTransferEngine):
        self.kv_cache = kv_cache
        self.transfer_engine = transfer_engine
        self.sessions: Dict[str, List[int]] = {}  # session_id -> block_ids

    async def recv_and_start_decode(self,
                                     prefill_worker_node: str,
                                     session_id: str,
                                     num_blocks: int) -> float:
        """接收 KV,写入本地 cache"""
        # 等待 KV 传输完成
        sid, k_data, v_data = await self.transfer_engine.recv_kv(prefill_worker_node)

        # 分配本地 block
        block_ids = list(range(num_blocks))
        self.sessions[session_id] = block_ids

        # 写入 cache
        self.kv_cache.deserialize_and_write(k_data, v_data, block_ids)
        return 0.0


# ============================================================
# 4. 完整 P/D 分离流程
# ============================================================

class PDDisaggregatedEngine:
    """P/D 分离引擎:Prefill 和 Decode 在不同节点"""

    def __init__(self,
                 prefill_worker: PrefillWorker,
                 decode_worker: DecodeWorker,
                 prefill_node: str,
                 decode_node: str):
        self.prefill_worker = prefill_worker
        self.decode_worker = decode_worker
        self.prefill_node = prefill_node
        self.decode_node = decode_node

    async def generate(self, prompt: List[int], session_id: str,
                       max_decode_steps: int = 10) -> List[int]:
        """完整生成流程"""
        t0 = time.perf_counter()

        # 1. Prefill + 异步传输 KV
        block_ids, transfer_time = await self.prefill_worker.prefill_and_transfer(
            prompt, self.decode_node, session_id, prefill_time=0.1
        )
        prefill_done = time.perf_counter() - t0
        print(f"[{session_id}] Prefill + KV transfer done in {prefill_done*1000:.0f} ms "
              f"(transfer={transfer_time*1000:.0f} ms)")

        # 2. Decode 节点接收 KV
        await self.decode_worker.recv_and_start_decode(
            self.prefill_node, session_id, len(block_ids)
        )
        recv_done = time.perf_counter() - t0
        print(f"[{session_id}] Decode worker received KV at {recv_done*1000:.0f} ms")

        # 3. Decode(模拟)
        output = []
        for step in range(max_decode_steps):
            await asyncio.sleep(0.05)  # 模拟 decode step
            output.append(step * 7)
        total = time.perf_counter() - t0
        print(f"[{session_id}] Decode finished at {total*1000:.0f} ms "
              f"({len(output)} tokens)")

        return output


# ============================================================
# 5. 演示
# ============================================================

async def demo_async():
    # 创建两个节点
    prefill_engine = MooncakeTransferEngine("prefill_node")
    decode_engine = MooncakeTransferEngine("decode_node")
    prefill_engine.register_remote("decode_node")
    decode_engine.register_remote("prefill_node")

    # KV cache
    num_blocks, block_size = 64, 16
    num_kv_heads, head_dim = 8, 64
    prefill_kv = PagedKVCache(
        k_cache=torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, dtype=torch.float16),
        v_cache=torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, dtype=torch.float16),
        block_size=block_size,
    )
    decode_kv = PagedKVCache(
        k_cache=torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, dtype=torch.float16),
        v_cache=torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, dtype=torch.float16),
        block_size=block_size,
    )

    prefill_worker = PrefillWorker(prefill_kv, prefill_engine)
    decode_worker = DecodeWorker(decode_kv, decode_engine)
    engine = PDDisaggregatedEngine(prefill_worker, decode_worker,
                                    "prefill_node", "decode_node")

    # 单请求
    print("=== Single request ===")
    out = await engine.generate(list(range(100)), "req1", max_decode_steps=5)

    # 并发多请求(展示异步优势)
    print("\n=== Concurrent requests (overlap prefill/decode) ===")
    tasks = [
        engine.generate(list(range(100 + i*10)), f"req{i+2}", max_decode_steps=4)
        for i in range(3)
    ]
    t0 = time.perf_counter()
    results = await asyncio.gather(*tasks)
    total = time.perf_counter() - t0
    print(f"\n3 concurrent requests done in {total*1000:.0f} ms")
    print(f"(对比串行: 3 * 单请求时间)")


def demo():
    asyncio.run(demo_async())


if __name__ == "__main__":
    demo()
