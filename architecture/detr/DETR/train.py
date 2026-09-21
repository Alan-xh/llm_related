"""DETR 训练入口。

任务定义:
    使用合成彩色矩形完成闭集目标检测训练。输入 batch shape 为
    ``[B,3,H,W]``，标签为长度 ``B`` 的字典列表，labels ``[M_i]``、
    boxes ``[M_i,4]``；模型输出 logits ``[B,Q,K+1]`` 和 boxes ``[B,Q,4]``。

训练目标:
    ``L = L_cls + 5 L_L1 + 2 L_GIoU``，匹配由公共 HungarianMatcher 完成。
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
    from architecture.detr.DETR.model import build_model


def main() -> None:
    """解析 CLI 参数并启动共享 DETR 训练循环。"""

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_train_args(parser, "detr_tiny.pt")
    args = parser.parse_args()
    # train_detector 内部负责 [B,3,H,W] 数据生成、前向、匹配、反向和 checkpoint。
    train_detector(
        build_model,
        steps=args.steps,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_classes=3,
        lr=args.lr,
        device=args.device,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
