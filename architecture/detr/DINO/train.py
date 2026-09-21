"""DINO 训练入口。

two-stage encoder proposal 先选择 ``[B,Q,D]`` query，训练时再拼接
``[B,G*M,D]`` contrastive denoising query；criterion 计算正常集合损失和
``CE + 5*L1`` 去噪损失。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    from ..common import add_common_train_args, synthetic_detection_batch
    from .model import DINOSetCriterion, build_model
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.detr.common import add_common_train_args, synthetic_detection_batch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import DINOSetCriterion, build_model


def main() -> None:
    """执行 DINO 的 two-stage query selection 与 DN 训练。"""

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_train_args(parser, "dino_tiny.pt")
    args = parser.parse_args()
    model, criterion = build_model().to(args.device), DINOSetCriterion(3).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for step in range(args.steps):
        images, targets = synthetic_detection_batch(args.batch_size, args.image_size, 3, args.device)
        # targets 触发 encoder proposal 之后的 denoising query 拼接。
        loss = criterion(model(images, targets=targets), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(f"step={step + 1}/{args.steps} loss={loss.item():.4f}")
    torch.save({"model": model.state_dict()}, args.checkpoint)
    print(f"saved checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
