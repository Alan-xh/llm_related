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
class GLM4Config(GLMConfig):
    """Compact GLM-4-style long-context and tool-use language core."""

    hidden_size: int = 128
    intermediate_size: int = 352
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 2
    max_position_embeddings: int = 1024
    rope_scaling: float = 2.0
    qkv_bias: bool = False
    use_qk_norm: bool = True


class GLM4ForCausalLM(GLMCausalLM):
    """Visual inputs are represented by placeholder tokens in this text core."""


def build_model() -> GLM4ForCausalLM:
    return GLM4ForCausalLM(GLM4Config())


if __name__ == "__main__":
    model = build_model()
    print(f"GLM-4 parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("attention: 4 query heads / 2 KV heads (GQA)")
    print("RoPE scaling: 2.0; image inputs: placeholder-token demo")

