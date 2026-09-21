"""Deformable-DETR 训练入口。

模型输入为 ``[B,3,H,W]``，backbone 产生多尺度
``[B,D,H_l,W_l]``，稀疏采样 decoder 最终输出 ``[B,Q,K+1]`` 与
``[B,Q,4]``；训练仍使用 Hungarian matching 集合损失。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from ..common import add_common_train_args, train_detector
    from .model import build_model
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.detr.common import add_common_train_args, train_detector
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import build_model


def main() -> None:
    """解析共享参数并启动多尺度 Deformable-DETR 训练。"""

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_train_args(parser, "deformable_detr_tiny.pt")
    args = parser.parse_args()
    # train_detector 负责合成数据、损失、AdamW 更新和 checkpoint 保存。
    train_detector(build_model, steps=args.steps, batch_size=args.batch_size,
                   image_size=args.image_size, num_classes=3, lr=args.lr,
                   device=args.device, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
