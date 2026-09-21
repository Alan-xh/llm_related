"""Training pipeline for the SAM 2 teaching model.

This demo trains the image path on synthetic prompts. Each batch uses images
``[B,3,H,W]`` and returns masks ``[B,K,H,W]`` plus scores ``[B,K]``; video
memory is exercised by the separate inference entry point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from ..common import train_segmenter
    from .model import build_model
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.SAM.common import train_segmenter
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import build_model


def main() -> None:
    """Parse CLI options and train the SAM 2 image path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default="sam2_tiny.pt")
    args = parser.parse_args()
    train_segmenter(
        build_model,
        steps=args.steps,
        batch_size=args.batch_size,
        image_size=args.image_size,
        lr=args.lr,
        device=args.device,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
