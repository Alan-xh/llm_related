"""ChatGLM2 教学模型推理入口。

完整 prompt 首轮输入 shape 为 [1, T]；后续 token 通过公共生成器以
[1, 1] 增量输入，并复用每层 shape [1, 1, T_cache, D] 的 MQA KV cache。
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
    """生成一个解释 ChatGLM2 MQA 的回答。"""
    generation_cli(build_model, "解释 ChatGLM2 的 MQA。", chat=True)


if __name__ == "__main__":
    main()
