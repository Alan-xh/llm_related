"""Qwen3 教学版 GQA + QK normalization + MoE decoder 模型定义。

任务定义:
    任务编号: QWEN3；领域: 带稀疏专家的自回归语言建模。输入 shape 为
    [B, T]，输出 logits shape 为 [B, T, V]。

核心机制:
    GQA 先将 Q/K 分别做 head 级 RMSNorm，再应用 RoPE；前馈路径使用
    top-k MoE。router 公式为 ``p(e|x)=softmax(W_r x)``，输出为
    ``sum_{e in TopK(p)} p_e * SwiGLU_e(x)``。thinking/non-thinking
    仅由推理 prompt 格式控制，不会创建第二个网络。
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
class Qwen3Config(ModelConfig):
    """Qwen3 风格的小型配置，默认启用 GQA、QK norm 和 top-k MoE。"""

    hidden_size: int = 128
    intermediate_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 2
    rope_theta: float = 1_000_000.0
    qkv_bias: bool = False
    use_qk_norm: bool = True
    use_moe: bool = True
    num_experts: int = 4
    num_experts_per_tok: int = 2


class Qwen3ForCausalLM(QwenCausalLM):
    """带 QK normalization 和 MoE 前馈的 Qwen3 风格 decoder。"""


def build_model() -> Qwen3ForCausalLM:
    """按默认 Qwen3 教学配置构造模型。"""
    return Qwen3ForCausalLM(Qwen3Config())


if __name__ == "__main__":
    model = build_model()
    # 打印稀疏路由配置，便于核对每个 token 的激活 expert 数量。
    print(f"Qwen3 parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"MoE: {model.config.num_experts} experts, top-{model.config.num_experts_per_tok}")
