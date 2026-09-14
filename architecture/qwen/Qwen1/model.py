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
class Qwen1Config(ModelConfig):
    """Compact defaults reflecting the first Qwen decoder architecture."""

    hidden_size: int = 128
    intermediate_size: int = 352
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 4
    rope_theta: float = 1_000_000.0
    qkv_bias: bool = True


class Qwen1ForCausalLM(QwenCausalLM):
    """Qwen1-style dense decoder: RMSNorm + RoPE + SwiGLU + causal LM."""


def build_model() -> Qwen1ForCausalLM:
    return Qwen1ForCausalLM(Qwen1Config())


if __name__ == "__main__":
    model = build_model()
    print(f"Qwen1 parameters: {sum(p.numel() for p in model.parameters()):,}")
