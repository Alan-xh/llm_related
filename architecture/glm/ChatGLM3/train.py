"""ChatGLM3 教学模型训练入口。

训练输入和标签 shape 为 [B, T]，公共 decoder 输出 logits shape 为
[B, T, V]。工具调用只作为结构化文本训练示例，不在训练脚本中执行。
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
    """使用工具调用主题文本启动 ChatGLM3 教学训练。"""
    train_cli(build_model, "ChatGLM3 can call tools and return structured results. ")


if __name__ == "__main__":
    main()
