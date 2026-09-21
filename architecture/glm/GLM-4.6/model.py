"""GLM-4.6 教学版长上下文 coding-agent 模型定义。

任务定义:
    任务编号: GLM-4.6；领域: 长上下文与代码 agent 接口的自回归语言建模。
    输入 token ids shape 为 [B, T]，输出 logits shape 为 [B, T, V]。

核心机制:
    使用 GQA、QK norm、RoPE scaling=4.0 和 top-k MoE；多步代码任务通过
    role/thinking prompt 表达，KV cache 让后续生成步骤只计算新 token。

数学映射:
    ``Attention(Q,K,V)=softmax(QK^T/sqrt(D)+mask)V``；
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
class GLM46Config(GLMConfig):
    """GLM-4.6 风格的小型长上下文配置，启用 GQA 与 top-k MoE。"""

    hidden_size: int = 128
    intermediate_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 2
    max_position_embeddings: int = 2048
    rope_scaling: float = 4.0
    qkv_bias: bool = False
    use_qk_norm: bool = True
    use_moe: bool = True
    num_experts: int = 4
    num_experts_per_tok: int = 2


class GLM46ForCausalLM(GLMCausalLM):
    """通过位置缩放和结构化 prompt 表达代码 agent 接口的 decoder。"""


def build_model() -> GLM46ForCausalLM:
    """按默认 GLM-4.6 教学配置构造模型。"""
    return GLM46ForCausalLM(GLM46Config())


if __name__ == "__main__":
    model = build_model()
    # 打印位置缩放与稀疏路由设置，便于核对教学配置。
    print(f"GLM-4.6 parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("RoPE scaling: 4.0; MoE: 4 experts, top-2 routing")
