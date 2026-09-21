"""GLM-4.7 教学模型训练入口。

训练 batch 和标签 shape 为 [B, T]，MoE decoder 输出 logits shape 为
[B, T, V]，thinking-before-acting 与终端工具仅由文本协议表达。
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from .model import build_model
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import build_model

try:
    from architecture.glm.common import train_cli
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.glm.common import train_cli


def main() -> None:
    """使用 reasoning-before-acting 主题文本启动 GLM-4.7 教学训练。"""
    train_cli(
        build_model,
        "GLM-4.7 thinks before acting and can operate terminal tools. ",
    )


if __name__ == "__main__":
    main()
