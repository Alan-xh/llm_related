"""Grounding DINO 训练入口。

图像输入为 ``[B,3,H,W]``，文本 token 为 ``[B,T]`；模型输出
token logits ``[B,Q,T]``、框 ``[B,Q,4]`` 和 objectness ``[B,Q]``。
目标 labels 被转换为文本 token 索引，损失由 token、objectness、L1 和
GIoU 组成。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    from ..common import add_common_train_args, synthetic_detection_batch
    from .model import GroundingCriterion, build_model
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.detr.common import add_common_train_args, synthetic_detection_batch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import GroundingCriterion, build_model


def main() -> None:
    """构造合成文本条件并启动 Grounding DINO 训练。"""

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_train_args(parser, "grounding_dino_tiny.pt")
    args = parser.parse_args()
    model = build_model().to(args.device)
    criterion = GroundingCriterion().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for step in range(args.steps):
        images, targets = synthetic_detection_batch(args.batch_size, args.image_size, 3, args.device)
        text_tokens = torch.tensor(
            [[1, 2, 3, 4, 5]], device=args.device
        ).expand(args.batch_size, -1)  # [B,T]
        for target in targets:
            target["labels"] = target["labels"] + 1  # 将类别 id 映射为文本 token id
        loss = criterion(model(images, text_tokens), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(f"step={step + 1}/{args.steps} loss={loss.item():.4f}")
    torch.save({"model": model.state_dict()}, args.checkpoint)
    print(f"saved checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
