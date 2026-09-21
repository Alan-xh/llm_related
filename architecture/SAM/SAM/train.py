"""Training pipeline for the SAM v1 teaching model.

Data contract:
    ``synthetic_segmentation_batch`` returns images ``[B,3,H,W]``, points
    ``[B,2,2]``, labels ``[B,2]``, and targets ``[B,1,H,W]``.
    The model returns masks ``[B,K,H,W]`` and scores ``[B,K]``.

Objective:
    ``L = 20*BCE + Dice + MSE(predicted_iou, detached_true_iou)``.
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
    from architecture.SAM.SAM.model import build_model


def main() -> None:
    """Parse CLI options and delegate the decoupled SAM training loop."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default="sam_tiny.pt")
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
