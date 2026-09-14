"""Run text-prompt concept segmentation with the compact SAM 3 model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    from .model import build_model
    from ..common import load_checkpoint
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.SAM.common import load_checkpoint
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="rectangle")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = build_model(args.image_size).to(device).eval()
    load_checkpoint(model, args.checkpoint, device)
    with torch.no_grad():
        output = model(
            torch.rand(1, 3, args.image_size, args.image_size, device=device),
            text_prompts=[args.prompt],
        )
    print(
        "masks:",
        tuple(output["masks"].shape),
        "scores:",
        tuple(output["iou_scores"].shape),
        "presence:",
        tuple(output["presence_logits"].shape),
    )


if __name__ == "__main__":
    main()
