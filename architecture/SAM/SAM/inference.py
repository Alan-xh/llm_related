"""Inference pipeline for the SAM v1 teaching model.

Input:
    synthetic images ``[1,3,H,W]``, point coordinates ``[1,1,2]``, and labels
    ``[1,1]``. Output masks have shape ``[1,K,H,W]`` and scores ``[1,K]``;
    ``--single-mask`` changes ``K`` from 3 to 1.
"""

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
    from architecture.SAM.SAM.model import build_model
    from architecture.SAM.common import load_checkpoint


def main() -> None:
    """Load an optional checkpoint and run one point-prompt prediction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--single-mask", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = build_model(args.image_size).to(device).eval()
    load_checkpoint(model, args.checkpoint, device)
    point = torch.tensor(
        [[[args.image_size / 2, args.image_size / 2]]],
        device=device,
    )
    label = torch.ones(1, 1, dtype=torch.long, device=device)
    with torch.no_grad():
        masks, scores = model(
            torch.rand(1, 3, args.image_size, args.image_size, device=device),
            point,
            label,
            multimask_output=not args.single_mask,
        )
    # masks: [1,K,H,W]; scores: [1,K].
    print("masks:", tuple(masks.shape), "scores:", tuple(scores.shape))


if __name__ == "__main__":
    main()
