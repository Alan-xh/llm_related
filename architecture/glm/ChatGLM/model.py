"""ChatGLM 教学版双语对话模型定义。

任务定义:
    任务编号: CHATGLM；领域: 双语对话式自回归语言建模。输入 token ids
    shape 为 [B, T]，输出 logits shape 为 [B, T, V]。

核心机制:
    使用 GLM 风格 decoder、二维 position ids、prefix-visible attention 和
    KV cache；对话角色、system prompt 与多轮消息由 ``format_chat`` 在
    推理时编码。该实现用 UTF-8 byte tokenizer 表达中英文共享词表。

数学映射:
    ``Attention(Q,K,V)=softmax(QK^T/sqrt(D)+mask)V``；
    ``L_lm=CrossEntropy(logits[:, :-1], labels[:, 1:])``。
"""

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
    """ChatGLM 风格的小型双语配置，启用二维位置和带 bias 的 QKV 投影。"""

    hidden_size: int = 128
    intermediate_size: int = 352
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 4
    use_2d_position_ids: bool = True
    max_block_position_embeddings: int = 128
    qkv_bias: bool = True


class ChatGLMForCausalLM(GLMCausalLM):
    """由角色模板提供对话行为的 GLM decoder，主干输入输出为 [B,T] -> [B,T,V]。"""


def build_model() -> ChatGLMForCausalLM:
    """按默认 ChatGLM 教学配置构造模型。"""
    return ChatGLMForCausalLM(ChatGLMConfig())


if __name__ == "__main__":
    model = build_model()
    # 参数量用于确认当前是 CPU 教学配置，而非官方大规模 checkpoint。
    print(f"ChatGLM parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("chat template: [gMASK] + role markers")
