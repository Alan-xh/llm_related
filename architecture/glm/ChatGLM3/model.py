"""ChatGLM3 教学版工具调用与代码模板模型定义。

任务定义:
    任务编号: CHATGLM3；领域: 支持结构化工具协议的对话式语言建模。
    输入 token ids shape 为 [B, T]，输出 logits shape 为 [B, T, V]。

核心机制:
    主干仍是 decoder-only Transformer；``format_chat`` 将 tools schema、
    role 和 thinking 标记串成 prompt，外部应用再通过 ``parse_tool_call``
    解析模型输出。工具执行不属于模型前向计算。

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
class ChatGLM3Config(GLMConfig):
    """ChatGLM3 风格的 MQA 配置，预留工具和代码解释器 prompt 接口。"""

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
    """通过结构化 role/tool 标记表达工具调用的 ChatGLM3 风格 decoder。"""


def build_model() -> ChatGLM3ForCausalLM:
    """按默认 ChatGLM3 教学配置构造模型。"""
    return ChatGLM3ForCausalLM(ChatGLM3Config())


if __name__ == "__main__":
    model = build_model()
    # 该示例只生成结构化文本，不负责执行工具。
    print(f"ChatGLM3 parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("tool protocol: <|tools|> and <tool_call>{...}</tool_call>")
