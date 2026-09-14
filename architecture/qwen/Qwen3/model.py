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
    """Small Qwen3-inspired decoder with GQA, QK normalization and MoE FFNs."""

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
    """Thinking/non-thinking is a prompt-and-generation mode, not a second net."""


def build_model() -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM(Qwen3Config())


if __name__ == "__main__":
    model = build_model()
    print(f"Qwen3 parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"MoE: {model.config.num_experts} experts, top-{model.config.num_experts_per_tok}")
