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
class Qwen25Config(ModelConfig):
    """Qwen2.5-style GQA decoder with a configurable long-context RoPE scale."""

    hidden_size: int = 128
    intermediate_size: int = 384
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 2
    max_position_embeddings: int = 2048
    rope_theta: float = 1_000_000.0
    rope_scaling: float = 4.0
    qkv_bias: bool = True


class Qwen25ForCausalLM(QwenCausalLM):
    """The long-context behavior here is a compact, scaled-RoPE teaching model."""


def build_model() -> Qwen25ForCausalLM:
    return Qwen25ForCausalLM(Qwen25Config())


if __name__ == "__main__":
    model = build_model()
    print(f"Qwen2.5 parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"RoPE cache: {model.config.max_position_embeddings} positions, scale={model.config.rope_scaling}")
