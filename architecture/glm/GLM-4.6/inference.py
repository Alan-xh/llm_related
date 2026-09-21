"""GLM-4.6 教学模型推理入口。

代码任务 prompt 编码为 shape [1, T]；后续生成步骤复用每层
shape [1, n_kv_heads, T_cache, D] 的 KV cache，以展示长上下文接口。
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
    """生成一个 GLM-4.6 风格的 Python 代码回答。"""
    generation_cli(build_model, "Write a small Python function and explain it.", chat=True)


if __name__ == "__main__":
    main()
