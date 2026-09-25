"""Kimi K2 风格的最小 MLA + 共享专家 MoE Decoder。

任务定义:
    构造一个 CPU 可运行的因果语言模型，用于展示 Kimi K2 风格的
    Multi-head Latent Attention、Top-k MoE、共享专家和增量 KV cache。

代表架构/算法:
    MLA、RoPE、SwiGLU、Top-k MoE、Decoder-only causal language model。

输入输出:
    ``input_ids`` shape 为 ``[B, T]``，模型输出 ``logits`` shape 为
    ``[B, T, V]``；生成时每层 MLA 缓存 shape 为
    ``[B, heads, T_cache, head_dim]``。

说明:
    这是教学实现，不兼容官方 tokenizer、checkpoint 或推理 kernel。
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
class KimiK2Config(KimiConfig):
    """体现 Kimi K2 风格 MLA 与稀疏 MoE 结构的默认配置。"""

    attention_type: str = "mla"
    use_moe: bool = True
    num_experts: int = 8
    num_experts_per_tok: int = 2
    shared_expert: bool = True


class KimiK2ForCausalLM(KimiForCausalLM):
    """Kimi K2 风格教学模型，不兼容官方权重。"""


def build_model() -> KimiK2ForCausalLM:
    """构造适合 CPU smoke test 的 Kimi K2 教学模型。"""
    return KimiK2ForCausalLM(KimiK2Config())


if __name__ == "__main__":
    model = build_model()
    print(f"Kimi K2 参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("注意力: Multi-head Latent Attention（MLA）")
    print("前馈网络: 8 个专家、top-2 路由、1 个共享专家")
