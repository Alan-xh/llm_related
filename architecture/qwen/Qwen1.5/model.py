"""Qwen1.5 教学版 dense/MoE decoder 模型定义。

任务定义:
    任务编号: QWEN1.5；领域: 自回归语言建模与稀疏专家前馈网络。
    输入 token ids shape [B, T]，输出 logits shape [B, T, V]。

核心机制:
    基础路径为 RMSNorm -> RoPE 注意力 -> 残差 -> RMSNorm ->
    SwiGLU/MoE -> 残差。启用 MoE 时，router 产生 [B*T, E] 分数，
    每个 token 选择 K 个 expert，并将 expert 输出恢复为 [B, T, H]。

核心公式:
    p(e|x) = softmax(W_router x)
    y = sum_{e in TopK(p)} p(e|x) * Expert_e(x)
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
class Qwen15Config(ModelConfig):
    """Qwen1.5 风格的 dense 配置，并保留可选 top-k MoE 所需参数。"""

    hidden_size: int = 128
    intermediate_size: int = 352
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 4
    rope_theta: float = 1_000_000.0
    qkv_bias: bool = True
    num_experts: int = 4
    num_experts_per_tok: int = 2


class Qwen15ForCausalLM(QwenCausalLM):
    """可在 dense SwiGLU 与 top-k MoE 前馈之间切换的 decoder。"""


def build_model(use_moe: bool = False) -> Qwen15ForCausalLM:
    """构造 Qwen1.5 教学模型。

    Args:
        use_moe: 为 True 时，每个 Transformer block 使用 MoE 前馈层。
    """
    return Qwen15ForCausalLM(Qwen15Config(use_moe=use_moe))


if __name__ == "__main__":
    for use_moe in (False, True):
        model = build_model(use_moe=use_moe)
        kind = "MoE" if use_moe else "dense"
        # 对比两种前馈实现的参数规模。
        print(f"Qwen1.5 {kind} parameters: {sum(p.numel() for p in model.parameters()):,}")
