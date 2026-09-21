"""GLM-4.5 教学模型训练入口。

训练 batch shape 为 [B, T]，模型输出 logits shape 为 [B, T, V]；启用 MoE
时，公共 forward 额外返回标量 ``aux_loss``，并按
``L_total = L_lm + 0.01 * L_aux`` 优化。
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
    """使用 reasoning/agent 主题文本启动 GLM-4.5 教学训练。"""
    train_cli(
        build_model,
        "GLM-4.5 reasons, plans, and uses tools for agentic coding. ",
    )


if __name__ == "__main__":
    main()
