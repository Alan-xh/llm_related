"""GLM-130B 教学模型的 blank-infilling 训练入口。

共享训练循环构造 ``[BOS] + prefix + <BLANK> + suffix + target``，输入和
标签 shape 为 [B, T]，context 标签为 ``-100``；模型输出 logits shape
为 [B, T, V]，loss 只监督 target 区域。
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
    """启动 GLM-130B 风格的 blank-infilling 教学训练。"""
    train_cli(
        build_model,
        "GLM learns to fill a blank from bidirectional context. ",
        blank_infilling=True,
    )


if __name__ == "__main__":
    main()
