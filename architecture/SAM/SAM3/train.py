"""Train the compact SAM 3 concept path on synthetic rectangle prompts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    from ..common import mask_loss, synthetic_segmentation_batch
    from .model import build_model
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.SAM.common import mask_loss, synthetic_segmentation_batch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default="sam3_tiny.pt")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = build_model(args.image_size).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for step in range(args.steps):
        images, points, labels, target = synthetic_segmentation_batch(
            args.batch_size, args.image_size, device, step
        )
        output = model(
            images,
            text_prompts=["rectangle"] * args.batch_size,
            point_coords=points,
            point_labels=labels,
        )
        loss = mask_loss(output["masks"], target, output["iou_scores"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(f"step {step + 1:03d}/{args.steps}: loss={loss.item():.4f}")
    torch.save({"model": model.state_dict()}, args.checkpoint)
    print(f"saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
