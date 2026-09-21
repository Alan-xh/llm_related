"""Qwen2 教学版 Grouped-Query Attention decoder 模型定义。

任务定义:
    任务编号: QWEN2；领域: GQA causal language modeling。输入 shape 为
    [B, T]，输出 logits shape 为 [B, T, V]。

核心机制:
    query 使用 n_heads 个 head，而 key/value 使用 n_kv_heads 个 head；
    ``repeat_kv`` 将 KV 扩展到 query head 数后执行
    Attention(Q, K', V') = softmax(QK'ᵀ / sqrt(d) + mask)V'。
    推理时每层保存 [B, n_kv_heads, T_cache, d] 的 KV cache。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from architecture.qwen.common import ModelConfig, QwenCausalLM
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.qwen.common import ModelConfig, QwenCausalLM


@dataclass
class Qwen2Config(ModelConfig):
    """Qwen2 风格的 GQA 配置，默认 4 个 query heads、2 个 KV heads。"""

    hidden_size: int = 128
    intermediate_size: int = 352
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 2
    rope_theta: float = 1_000_000.0
    qkv_bias: bool = True


class Qwen2ForCausalLM(QwenCausalLM):
    """通过公共注意力模块启用 GQA 与逐层 KV cache 的 decoder。"""


def build_model() -> Qwen2ForCausalLM:
    """按默认 Qwen2 教学配置构造模型。"""
    return Qwen2ForCausalLM(Qwen2Config())


if __name__ == "__main__":
    model = build_model()
    # 打印参数量和 head 配置，便于观察 GQA 的结构差异。
    print(f"Qwen2 parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"GQA: {model.config.num_heads} query heads / {model.config.num_kv_heads} KV heads")
