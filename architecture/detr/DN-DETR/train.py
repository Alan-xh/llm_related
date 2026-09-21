"""DN-DETR 训练入口。

训练时模型接收图像 ``[B,3,H,W]`` 与目标列表，并额外生成
denoising query ``[B,G*M,D]``；正常输出为 ``[B,Q,K+1]``、``[B,Q,4]``，
去噪输出为 ``[B,G*M,K+1]``、``[B,G*M,4]``。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    from ..common import add_common_train_args, synthetic_detection_batch
    from .model import DenoisingCriterion, build_model
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.detr.common import add_common_train_args, synthetic_detection_batch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import DenoisingCriterion, build_model


def main() -> None:
    """执行带 denoising criterion 的 DN-DETR 训练循环。"""

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_train_args(parser, "dn_detr_tiny.pt")
    args = parser.parse_args()
    model = build_model().to(args.device)
    criterion = DenoisingCriterion(3).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for step in range(args.steps):
        images, targets = synthetic_detection_batch(args.batch_size, args.image_size, 3, args.device)
        # targets 触发 DN 分支；criterion 同时计算 normal set loss 和 L_DN。
        loss = criterion(model(images, targets=targets), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(f"step={step + 1}/{args.steps} loss={loss.item():.4f}")
    torch.save({"model": model.state_dict()}, args.checkpoint)
    print(f"saved checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
