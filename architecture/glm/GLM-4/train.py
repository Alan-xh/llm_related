"""GLM-4 教学模型训练入口。

训练输入/标签 shape 为 [B, T]，模型经过 GQA、RoPE 和 SwiGLU 后输出
logits shape [B, T, V]。视觉能力在本例中只由文本 placeholder 表达。
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
    """使用 GLM-4 主题文本启动教学训练。"""
    train_cli(build_model, "GLM-4 combines long context, tools, and visual placeholders. ")


if __name__ == "__main__":
    main()
