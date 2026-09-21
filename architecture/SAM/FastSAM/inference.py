"""Inference pipeline for FastSAM candidate generation and selection.

The all-candidate path returns masks ``[1,N,H,W]`` and scores ``[1,N]``.
The prompted path ranks them and returns top-K masks ``[1,K,H,W]`` and
scores ``[1,K]``.
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
    from architecture.SAM.common import load_checkpoint
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import build_model


def main() -> None:
    """Load a checkpoint and run either all-candidate or point-selected inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--all-candidates", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = build_model(args.image_size).to(device).eval()
    load_checkpoint(model, args.checkpoint, device)
    images = torch.rand(1, 3, args.image_size, args.image_size, device=device)
    with torch.no_grad():
        if args.all_candidates:
            masks, scores = model.predict_all(images)
        else:
            point = torch.tensor([[[args.image_size / 2, args.image_size / 2]]], device=device)
            label = torch.ones(1, 1, dtype=torch.long, device=device)
            masks, scores = model(images, point, label)
    # all-candidates: [1,N,H,W]/[1,N]; selected: [1,K,H,W]/[1,K].
    print("masks:", tuple(masks.shape), "scores:", tuple(scores.shape))


if __name__ == "__main__":
    main()
