"""GLM-4 教学模型推理入口。

``<image>`` 在本例中是 byte-token 文本占位符；prompt 的 token shape 为
[1, T]，模型输出 logits shape 为 [1, T_current, V]。
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
    """生成一个包含视觉 placeholder 的 GLM-4 风格回答。"""
    generation_cli(build_model, "Describe the image placeholder <image>.", chat=True)


if __name__ == "__main__":
    main()
