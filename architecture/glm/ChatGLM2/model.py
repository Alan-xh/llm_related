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
class ChatGLM2Config(GLMConfig):
    """ChatGLM2-style config with multi-query attention and RoPE."""

    hidden_size: int = 128
    intermediate_size: int = 352
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 1
    use_2d_position_ids: bool = True
    max_block_position_embeddings: int = 128
    rope_theta: float = 10_000.0
    qkv_bias: bool = False


class ChatGLM2ForCausalLM(GLMCausalLM):
    """The shared attention block exposes MQA through num_kv_heads=1."""


def build_model() -> ChatGLM2ForCausalLM:
    return ChatGLM2ForCausalLM(ChatGLM2Config())


if __name__ == "__main__":
    model = build_model()
    print(f"ChatGLM2 parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("attention: 4 query heads / 1 KV head (MQA)")

