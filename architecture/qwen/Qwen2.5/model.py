"""Qwen2.5 教学版长上下文 RoPE/GQA decoder 模型定义。

任务定义:
    任务编号: QWEN2.5；领域: 长上下文 causal language modeling。输入
    token ids shape [B, T]，输出 logits shape [B, T, V]。

核心机制:
    在 Qwen2 风格 GQA 上加入缩放 RoPE：``pos' = pos / rope_scaling``，
    再用 ``cos(pos' * inv_freq)`` 和 ``sin(pos' * inv_freq)`` 旋转 Q/K。
    该教学实现用于展示位置频率缩放接口，不等价于官方长上下文训练方案。
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
class Qwen25Config(ModelConfig):
    """Qwen2.5 风格的 GQA 配置，并启用可调 RoPE 缩放因子。"""

    hidden_size: int = 128
    intermediate_size: int = 384
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 2
    max_position_embeddings: int = 2048
    rope_theta: float = 1_000_000.0
    rope_scaling: float = 4.0
    qkv_bias: bool = True


class Qwen25ForCausalLM(QwenCausalLM):
    """使用 scaled RoPE 展示长上下文位置编码接口的 decoder。"""


def build_model() -> Qwen25ForCausalLM:
    """按默认 Qwen2.5 教学配置构造模型。"""
    return Qwen25ForCausalLM(Qwen25Config())


if __name__ == "__main__":
    model = build_model()
    # 打印 RoPE cache 覆盖长度和缩放因子。
    print(f"Qwen2.5 parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"RoPE cache: {model.config.max_position_embeddings} positions, scale={model.config.rope_scaling}")
