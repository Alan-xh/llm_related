"""ChatGLM 教学模型推理入口。

文本经 ChatGLM role template 和 byte tokenizer 后形成 ``input_ids``，
shape 为 [1, T]；增量生成返回 [1, T + T_new]，再解码为双语文本。
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
    """生成一句 ChatGLM 风格的双语回答。"""
    generation_cli(build_model, "用一句话介绍 ChatGLM。", chat=True)


if __name__ == "__main__":
    main()
