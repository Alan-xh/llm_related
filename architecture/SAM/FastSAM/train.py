"""Training pipeline for the FastSAM candidate generator.

The image encoder runs once per batch. It produces candidate masks
``[B,N,H,W]`` and scores ``[B,N]``; point prompts select top-K outputs
``[B,K,H,W]`` and ``[B,K]`` before the common segmentation loss is applied.
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
    """Parse CLI options and train the FastSAM candidate path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default="fast_sam_tiny.pt")
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
