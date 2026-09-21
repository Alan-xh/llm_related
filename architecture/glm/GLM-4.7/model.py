"""GLM-4.7 教学版 reasoning-before-acting agent 模型定义。

任务定义:
    任务编号: GLM-4.7；领域: 支持 thinking 和终端工具协议的自回归语言建模。
    输入 token ids shape 为 [B, T]，输出 logits shape 为 [B, T, V]。

核心机制:
    使用 MQA、QK norm、RoPE scaling=4.0 和 top-k MoE；模型通过结构化
    ``<think>``/``<tool_call>`` 文本表达先思考后行动，外部系统负责执行工具。

数学映射:
    ``p(e|x)=softmax(W_router x)``；
    ``y=sum_{e in TopK(p)} p_e*SwiGLU_e(x)``；
    ``L_total=L_lm+0.01*L_aux``。
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
class GLM47Config(GLMConfig):
    """GLM-4.7 风格的小型配置，使用 MQA、QK norm 和 top-k MoE。"""

    hidden_size: int = 128
    intermediate_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 1
    max_position_embeddings: int = 2048
    rope_scaling: float = 4.0
    qkv_bias: bool = False
    use_qk_norm: bool = True
    use_moe: bool = True
    num_experts: int = 4
    num_experts_per_tok: int = 2


class GLM47ForCausalLM(GLMCausalLM):
    """通过结构化 prompt 标记表达终端和工具动作的 agent decoder。"""


def build_model() -> GLM47ForCausalLM:
    """按默认 GLM-4.7 教学配置构造模型。"""
    return GLM47ForCausalLM(GLM47Config())


if __name__ == "__main__":
    model = build_model()
    # 打印 MQA 与 MoE 配置，便于核对推理缓存和稀疏路由设置。
    print(f"GLM-4.7 parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("attention: 4 query heads / 1 KV head (MQA)")
    print("MoE: 4 experts, top-2 routing; tool loop: prompt-level demo")
