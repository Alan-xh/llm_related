"""ChatGLM2 教学版 MQA decoder 模型定义。

任务定义:
    任务编号: CHATGLM2；领域: 带多查询注意力的对话式自回归语言建模。
    输入 token ids shape 为 [B, T]，输出 logits shape 为 [B, T, V]。

核心机制:
    所有 query head 共享一组 K/V（MQA），并使用 RoPE 编码位置；生成阶段
    每层 cache 的 K/V shape 为 [B, 1, T_cache, D]，降低增量解码缓存开销。

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
class ChatGLM2Config(GLMConfig):
    """ChatGLM2 风格的小型配置，使用 4 个 query head 和 1 个 KV head。"""

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
    """通过 ``num_kv_heads=1`` 表达 MQA 的 ChatGLM2 风格 decoder。"""


def build_model() -> ChatGLM2ForCausalLM:
    """按默认 ChatGLM2 教学配置构造模型。"""
    return ChatGLM2ForCausalLM(ChatGLM2Config())


if __name__ == "__main__":
    model = build_model()
    # 打印 MQA 的 head 配置，便于核对 query 与 KV 的 Shape。
    print(f"ChatGLM2 parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("attention: 4 query heads / 1 KV head (MQA)")
