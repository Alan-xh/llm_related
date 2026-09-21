"""Qwen1 教学版 decoder-only causal language model。

任务定义:
    任务编号: QWEN1；领域: 自回归语言建模。输入 token ids shape 为
    [B, T]，经过 embedding、RMSNorm、RoPE 多头自注意力和 SwiGLU 后，
    输出 vocabulary logits shape [B, T, V]。

代表架构与核心公式:
    Qwen1 风格的 dense Transformer 使用 MHA（n_kv_heads = n_heads）。
    主要计算为 Attention(Q, K, V) = softmax(QKᵀ / sqrt(d) + mask)V，
    SwiGLU(x) = W_down(SiLU(W_gate x) * W_up x)。

说明:
    该文件只定义版本配置和模型构造器，具体网络实现复用
    ``architecture.qwen.common``，不兼容官方 Qwen1 checkpoint。
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
class Qwen1Config(ModelConfig):
    """Qwen1 风格的小型 dense decoder 配置。

    该版本使用相同数量的 query/KV heads，因此注意力路径为标准 MHA。
    """

    hidden_size: int = 128
    intermediate_size: int = 352
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 4
    rope_theta: float = 1_000_000.0
    qkv_bias: bool = True


class Qwen1ForCausalLM(QwenCausalLM):
    """Qwen1 风格的 dense decoder。

    数据流为 ``[B, T] -> [B, T, H] -> [B, T, H] -> [B, T, V]``，
    每层采用 RMSNorm、RoPE 注意力、残差连接和 SwiGLU。
    """


def build_model() -> Qwen1ForCausalLM:
    """按默认教学配置构造 Qwen1 模型。"""
    return Qwen1ForCausalLM(Qwen1Config())


if __name__ == "__main__":
    model = build_model()
    # 参数量用于快速确认配置是否按预期生效。
    print(f"Qwen1 parameters: {sum(p.numel() for p in model.parameters()):,}")
