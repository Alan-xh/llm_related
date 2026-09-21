"""GLM-4 教学版 GQA、长上下文缩放与视觉占位符模型定义。

任务定义:
    任务编号: GLM-4；领域: 支持 GQA 和多模态占位符的自回归语言建模。
    输入 token ids shape 为 [B, T]，输出 logits shape 为 [B, T, V]。

核心机制:
    4 个 query head 共享 2 个 KV head（GQA），RoPE 使用 scaling=2.0；
    ``<image>`` 仅是 byte-token 文本占位符，不包含视觉编码器。

数学映射:
    ``Attention(Q,K,V)=softmax(QK^T/sqrt(D)+mask)V``；
    ``RoPE(x,p)=x*cos(p/scaling)+rotate_half(x)*sin(p/scaling)``。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from architecture.glm.common import GLMConfig, GLMCausalLM
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.glm.common import GLMConfig, GLMCausalLM


@dataclass
class GLM4Config(GLMConfig):
    """GLM-4 风格的小型配置，使用 GQA、QK norm 和 RoPE 缩放。"""

    hidden_size: int = 128
    intermediate_size: int = 352
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 2
    max_position_embeddings: int = 1024
    rope_scaling: float = 2.0
    qkv_bias: bool = False
    use_qk_norm: bool = True


class GLM4ForCausalLM(GLMCausalLM):
    """将视觉输入抽象为 placeholder token 的 GLM-4 风格文本核心。"""


def build_model() -> GLM4ForCausalLM:
    """按默认 GLM-4 教学配置构造模型。"""
    return GLM4ForCausalLM(GLM4Config())


if __name__ == "__main__":
    model = build_model()
    # 打印 GQA 配置，便于核对 Q/KV head 数量和 RoPE 缩放。
    print(f"GLM-4 parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("attention: 4 query heads / 2 KV heads (GQA)")
    print("RoPE scaling: 2.0; image inputs: placeholder-token demo")
