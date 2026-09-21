"""GLM-130B 教学模型推理入口。

文本 prompt 编码为 ``input_ids`` shape [1, T]，生成器使用 causal decoder
和逐层 KV cache 返回 shape [1, T + T_new] 的完整序列。
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
    """生成一个 GLM 风格的文本续写结果。"""
    generation_cli(build_model, "GLM combines understanding and generation.")


if __name__ == "__main__":
    main()
