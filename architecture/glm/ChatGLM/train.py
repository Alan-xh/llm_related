"""ChatGLM 教学模型训练入口。

训练数据经 ``ByteTokenizer`` 编码为 ``input_ids/labels``，二者 shape 为
[B, T]；公共模型输出 logits shape 为 [B, T, V]，并使用 causal shift
计算语言模型损失。该脚本只负责参数解析和调用共享训练循环。
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
    from architecture.glm.common import train_cli
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.glm.common import train_cli


def main() -> None:
    """使用双语 ChatGLM 教学文本启动训练。"""
    train_cli(build_model, "ChatGLM answers bilingual user questions. ")


if __name__ == "__main__":
    main()
