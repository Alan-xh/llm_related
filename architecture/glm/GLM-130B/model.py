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
class GLM130BConfig(GLMConfig):
    """Compact GLM-130B-style config with 2-D positions and blank infilling."""

    hidden_size: int = 128
    intermediate_size: int = 352
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 4
    use_2d_position_ids: bool = True
    max_block_position_embeddings: int = 128
    rope_theta: float = 10_000.0


class GLM130BForCausalLM(GLMCausalLM):
    """A small educational GLM decoder, not a checkpoint-compatible replica."""


def build_model() -> GLM130BForCausalLM:
    return GLM130BForCausalLM(GLM130BConfig())


if __name__ == "__main__":
    model = build_model()
    print(f"GLM-130B parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("2-D position ids: enabled; blank-infilling training: enabled")

