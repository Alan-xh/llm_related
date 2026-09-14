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
class ChatGLM3Config(GLMConfig):
    """ChatGLM3-style MQA decoder for tool and code-interpreter templates."""

    hidden_size: int = 128
    intermediate_size: int = 352
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 1
    use_2d_position_ids: bool = True
    max_block_position_embeddings: int = 128
    max_position_embeddings: int = 1024
    qkv_bias: bool = False


class ChatGLM3ForCausalLM(GLMCausalLM):
    """Tool use is represented by structured role markers around the LM."""


def build_model() -> ChatGLM3ForCausalLM:
    return ChatGLM3ForCausalLM(ChatGLM3Config())


if __name__ == "__main__":
    model = build_model()
    print(f"ChatGLM3 parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("tool protocol: <|tools|> and <tool_call>{...}</tool_call>")

