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
class GLM45Config(GLMConfig):
    """Compact GLM-4.5-style hybrid reasoning and agentic MoE config."""

    hidden_size: int = 128
    intermediate_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 2
    max_position_embeddings: int = 1024
    rope_scaling: float = 2.0
    qkv_bias: bool = False
    use_qk_norm: bool = True
    use_moe: bool = True
    num_experts: int = 4
    num_experts_per_tok: int = 2


class GLM45ForCausalLM(GLMCausalLM):
    """Hybrid thinking is exposed as a prompt mode; the core remains one LM."""


def build_model() -> GLM45ForCausalLM:
    return GLM45ForCausalLM(GLM45Config())


if __name__ == "__main__":
    model = build_model()
    print(f"GLM-4.5 parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("MoE: 4 experts, top-2 routing; thinking/non-thinking: prompt mode")

