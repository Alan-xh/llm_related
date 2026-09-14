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
class Qwen2Config(ModelConfig):
    """Qwen2-style defaults with grouped-query attention and cache support."""

    hidden_size: int = 128
    intermediate_size: int = 352
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 2
    rope_theta: float = 1_000_000.0
    qkv_bias: bool = True


class Qwen2ForCausalLM(QwenCausalLM):
    """GQA is implemented by the shared attention block and exposed by config."""


def build_model() -> Qwen2ForCausalLM:
    return Qwen2ForCausalLM(Qwen2Config())


if __name__ == "__main__":
    model = build_model()
    print(f"Qwen2 parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"GQA: {model.config.num_heads} query heads / {model.config.num_kv_heads} KV heads")
