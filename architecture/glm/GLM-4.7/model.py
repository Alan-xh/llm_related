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
class GLM47Config(GLMConfig):
    """Compact GLM-4.7-style reasoning-before-acting agentic configuration."""

    hidden_size: int = 128
    intermediate_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 1
    max_position_embeddings: int = 2048
    rope_scaling: float = 4.0
    qkv_bias: bool = False
    use_qk_norm: bool = True
    use_moe: bool = True
    num_experts: int = 4
    num_experts_per_tok: int = 2


class GLM47ForCausalLM(GLMCausalLM):
    """Terminal/tool actions are expressed through structured prompt markers."""


def build_model() -> GLM47ForCausalLM:
    return GLM47ForCausalLM(GLM47Config())


if __name__ == "__main__":
    model = build_model()
    print(f"GLM-4.7 parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("attention: 4 query heads / 1 KV head (MQA)")
    print("MoE: 4 experts, top-2 routing; tool loop: prompt-level demo")

