"""Inference pipeline for SAM 3.1 object-token multiplexing.

One image ``[1,3,H,W]`` is encoded once for O text prompts. The decoder batch
is reshaped to masks ``[1,O,K,H,W]`` and scores ``[1,O,K]`; token lengths are
returned to preserve object identity.
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
    """Load a checkpoint and run the shared-image multi-prompt path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", nargs="+", default=["rectangle", "object", "shape"])
    parser.add_argument("--bucket-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = build_model(args.image_size).to(device).eval()
    load_checkpoint(model, args.checkpoint, device)
    with torch.no_grad():
        output = model.multiplex(
            torch.rand(1, 3, args.image_size, args.image_size, device=device),
            args.prompts,
            args.bucket_size,
        )
    # output["masks"]: [1,O,K,H,W]; output["iou_scores"]: [1,O,K].
    print(
        "masks:",
        tuple(output["masks"].shape),
        "scores:",
        tuple(output["iou_scores"].shape),
        "object_token_lengths:",
        output["object_token_lengths"].tolist(),
    )


if __name__ == "__main__":
    main()
