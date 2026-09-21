"""ChatGLM2 教学模型训练入口。

训练 batch 的 ``input_ids/labels`` shape 为 [B, T]；模型内部将 query 变为
[B, n_heads, T, D]，将共享 KV 变为 [B, 1, T, D]，并按 causal shift
计算 logits shape [B, T, V] 对应的语言模型损失。
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from .model import build_model
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from model import build_model

try:
    from architecture.glm.common import train_cli
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.glm.common import train_cli


def main() -> None:
    """使用 MQA 主题文本启动 ChatGLM2 教学训练。"""
    train_cli(build_model, "ChatGLM2 uses multi-query attention for efficient decoding. ")


if __name__ == "__main__":
    main()
