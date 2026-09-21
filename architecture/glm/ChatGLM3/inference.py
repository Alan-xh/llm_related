"""ChatGLM3 教学模型推理入口。

工具 schema/role prompt 编码为 ``input_ids`` shape [1, T]；模型只生成
结构化文本，外部应用可用 ``parse_tool_call`` 将输出解析为 JSON。
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
    from architecture.glm.common import generation_cli
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.glm.common import generation_cli


def main() -> None:
    """生成一个 ChatGLM3 风格的工具调用示例。"""
    generation_cli(build_model, "调用天气工具查询北京天气。", chat=True)


if __name__ == "__main__":
    main()
