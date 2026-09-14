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
    """A dense default with an optional small MoE feed-forward variant."""

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
    """Qwen1.5-style dense/MoE configurable decoder."""


def build_model(use_moe: bool = False) -> Qwen15ForCausalLM:
    return Qwen15ForCausalLM(Qwen15Config(use_moe=use_moe))


if __name__ == "__main__":
    for use_moe in (False, True):
        model = build_model(use_moe=use_moe)
        kind = "MoE" if use_moe else "dense"
        print(f"Qwen1.5 {kind} parameters: {sum(p.numel() for p in model.parameters()):,}")
