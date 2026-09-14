"""Train the YOLOv3 teaching model on synthetic rectangles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from ..common import train_detector
    from .model import build_model
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.yolo.common import train_detector
    from architecture.yolo.YOLOv3.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default="yolov3_tiny.pt")
    args = parser.parse_args()
    train_detector(build_model, steps=args.steps, batch_size=args.batch_size, image_size=args.image_size, lr=args.lr, device=args.device, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()

