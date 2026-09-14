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
class GLM46Config(GLMConfig):
    """Compact GLM-4.6-style long-context coding-agent language core."""

    hidden_size: int = 128
    intermediate_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 2
    max_position_embeddings: int = 2048
    rope_scaling: float = 4.0
    qkv_bias: bool = False
    use_qk_norm: bool = True
    use_moe: bool = True
    num_experts: int = 4
    num_experts_per_tok: int = 2


class GLM46ForCausalLM(GLMCausalLM):
    """Long context and coding-agent behavior are represented in the interface."""


def build_model() -> GLM46ForCausalLM:
    return GLM46ForCausalLM(GLM46Config())


if __name__ == "__main__":
    model = build_model()
    print(f"GLM-4.6 parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("RoPE scaling: 4.0; MoE: 4 experts, top-2 routing")

