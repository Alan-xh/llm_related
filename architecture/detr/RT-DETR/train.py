"""RT-DETR 训练入口。

多尺度输入特征被展平为 ``[B,S,D]``，encoder 按
``max_class_prob * predicted_iou`` 选择 ``[B,Q,D]`` query，最后输出
``[B,Q,K+1]``、``[B,Q,4]``；训练采用公共集合损失。
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
    """解析参数并启动 RT-DETR 合成数据训练。"""

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_train_args(parser, "rt_detr_tiny.pt")
    args = parser.parse_args()
    # 公共训练循环会保留 RT-DETR 的 encoder proposal 输出但只优化主损失。
    train_detector(build_model, steps=args.steps, batch_size=args.batch_size,
                   image_size=args.image_size, num_classes=3, lr=args.lr,
                   device=args.device, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
