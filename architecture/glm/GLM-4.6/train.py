"""GLM-4.6 教学模型训练入口。

训练输入/标签 shape 为 [B, T]，模型输出 logits shape 为 [B, T, V]；
coding-agent 的多步行为以 role/thinking 文本模板表达，不执行外部工具。
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
    """使用代码 agent 主题文本启动 GLM-4.6 教学训练。"""
    train_cli(
        build_model,
        "GLM-4.6 writes code and coordinates multi-step agent workflows. ",
    )


if __name__ == "__main__":
    main()
