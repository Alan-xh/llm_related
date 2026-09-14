"""Propagate a point prompt through a short SAM 2.1 video."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    from .model import build_model
    from ..common import VideoMemory, load_checkpoint
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.SAM.common import VideoMemory, load_checkpoint
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = build_model(args.image_size).to(device).eval()
    load_checkpoint(model, args.checkpoint, device)
    state = VideoMemory(model.config.max_memory_frames)
    point = torch.tensor([[[args.image_size / 2, args.image_size / 2]]], device=device)
    label = torch.ones(1, 1, dtype=torch.long, device=device)
    for frame in range(args.frames):
        with torch.no_grad():
            masks, scores, state = model.predict_frame(
                torch.rand(1, 3, args.image_size, args.image_size, device=device),
                point if frame == 0 else None,
                label if frame == 0 else None,
                state,
                frame,
            )
        print(f"frame={frame} masks={tuple(masks.shape)} scores={tuple(scores.shape)}")


if __name__ == "__main__":
    main()
