"""Kimi Linear 风格的最小 KDA/MLA 混合 Decoder。

任务定义:
    构造一个 CPU 可运行的因果语言模型，用于展示 KDA 有限状态递归记忆、
    每 3 个 KDA 层插入 1 个 MLA 层，以及两类缓存的差异。

代表架构/算法:
    Kimi Delta Attention（KDA）、MLA、RoPE、SwiGLU、Top-k MoE。

输入输出:
    ``input_ids`` shape 为 ``[B, T]``，模型输出 ``logits`` shape 为
    ``[B, T, V]``；KDA 层缓存状态 shape 为 ``[B, heads, D, D]``，
    MLA 层缓存 K/V shape 为 ``[B, heads, T_cache, D]``。

说明:
    这是教学实现，不兼容官方 tokenizer、checkpoint 或 FLA kernel。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from architecture.kimi.common import KimiConfig, KimiForCausalLM
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.kimi.common import KimiConfig, KimiForCausalLM


@dataclass
class KimiLinearConfig(KimiConfig):
    """每 3 个 KDA 层插入 1 个 MLA 层的混合配置。"""

    attention_type: str = "hybrid"
    kda_ratio: int = 3
    use_moe: bool = True
    num_experts: int = 4
    num_experts_per_tok: int = 2
    shared_expert: bool = True


class KimiLinearForCausalLM(KimiForCausalLM):
    """包含有限状态 KDA 层的 Kimi Linear 风格教学模型。"""


def build_model() -> KimiLinearForCausalLM:
    """构造适合 CPU smoke test 的 Kimi Linear 教学模型。"""
    return KimiLinearForCausalLM(KimiLinearConfig())


if __name__ == "__main__":
    model = build_model()
    pattern = ", ".join(layer.attention_name for layer in model.layers)
    print(f"Kimi Linear 参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"注意力层模式: {pattern}（KDA:MLA = 3:1）")
    print("KDA 缓存: 有限递归状态 [B, heads, head_dim, head_dim]")
