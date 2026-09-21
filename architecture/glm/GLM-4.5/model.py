"""GLM-4.5 教学版 hybrid reasoning 与 agentic MoE 模型定义。

任务定义:
    任务编号: GLM-4.5；领域: 带稀疏专家和 thinking prompt 的自回归语言建模。
    输入 token ids shape 为 [B, T]，输出 logits shape 为 [B, T, V]。

核心机制:
    GQA/QK norm 负责注意力稳定性，top-k MoE 负责稀疏前馈；thinking/non-thinking
    由 prompt 模式控制，共享同一个网络，工具调用由外部应用解析和执行。

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
class GLM45Config(GLMConfig):
    """GLM-4.5 风格的小型配置，启用 GQA、QK norm 和 top-k MoE。"""

    hidden_size: int = 128
    intermediate_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 2
    max_position_embeddings: int = 1024
    rope_scaling: float = 2.0
    qkv_bias: bool = False
    use_qk_norm: bool = True
    use_moe: bool = True
    num_experts: int = 4
    num_experts_per_tok: int = 2


class GLM45ForCausalLM(GLMCausalLM):
    """通过 prompt 模式表达 hybrid thinking 的 MoE decoder。"""


def build_model() -> GLM45ForCausalLM:
    """按默认 GLM-4.5 教学配置构造模型。"""
    return GLM45ForCausalLM(GLM45Config())


if __name__ == "__main__":
    model = build_model()
    # 打印稀疏路由配置，便于核对每个 token 激活的 expert 数量。
    print(f"GLM-4.5 parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("MoE: 4 experts, top-2 routing; thinking/non-thinking: prompt mode")
