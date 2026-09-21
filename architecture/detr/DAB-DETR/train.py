"""DAB-DETR 训练入口。

输入图像 shape 为 ``[B,3,H,W]``；动态 anchor query 在模型内部以
``[B,Q,4]`` 表示，并按 ``b^(l+1)=sigmoid(inv_sigmoid(b^l)+Δb^l)``
逐层 refinement。训练损失由公共 Hungarian matching 和集合损失提供。
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
    """解析共享训练参数并启动 DAB-DETR 训练。"""

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_train_args(parser, "dab_detr_tiny.pt")
    args = parser.parse_args()
    # 公共训练循环保持输入 [B,3,H,W]、输出 [B,Q,K+1]/[B,Q,4] 的契约。
    train_detector(build_model, steps=args.steps, batch_size=args.batch_size,
                   image_size=args.image_size, num_classes=3, lr=args.lr,
                   device=args.device, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
