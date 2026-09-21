"""GLM-4.7 教学模型推理入口。

thinking/tool prompt 编码为 shape [1, T] 的 token ids；模型生成
``<think>`` 或 ``<tool_call>`` 文本，工具执行和权限校验由外部系统负责。
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
    """生成一个 GLM-4.7 风格的终端 agent 计划。"""
    generation_cli(
        build_model,
        "Inspect the repository and propose the next terminal action.",
        chat=True,
    )


if __name__ == "__main__":
    main()
