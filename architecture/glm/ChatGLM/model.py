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
class ChatGLMConfig(GLMConfig):
    """Compact bilingual ChatGLM-style config built on the GLM objective."""

    hidden_size: int = 128
    intermediate_size: int = 352
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 4
    use_2d_position_ids: bool = True
    max_block_position_embeddings: int = 128
    qkv_bias: bool = True


class ChatGLMForCausalLM(GLMCausalLM):
    """The chat behavior is supplied by the role template at inference time."""


def build_model() -> ChatGLMForCausalLM:
    return ChatGLMForCausalLM(ChatGLMConfig())


if __name__ == "__main__":
    model = build_model()
    print(f"ChatGLM parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("chat template: [gMASK] + role markers")

