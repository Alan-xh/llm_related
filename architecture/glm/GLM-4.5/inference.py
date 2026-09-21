"""GLM-4.5 教学模型推理入口。

thinking prompt 通过 ChatGLM 模板编码为 shape [1, T] 的 token ids；模型
共享同一 MoE 网络完成 reasoning/non-thinking 生成，不会执行工具。
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
    from architecture.glm.common import generation_cli
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.glm.common import generation_cli


def main() -> None:
    """生成一个 GLM-4.5 风格的计划回答。"""
    generation_cli(build_model, "Plan the steps for fixing a failing test.", chat=True)


if __name__ == "__main__":
    main()
