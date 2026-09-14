"""Run YOLOv8 teaching-model inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from ..common import infer_detector
    from .model import build_model
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.yolo.common import infer_detector
    from architecture.yolo.YOLOv8.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--checkpoint")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    detections = infer_detector(build_model, checkpoint=args.checkpoint, image_size=args.image_size, confidence=args.confidence, device=args.device)
    print(f"image=0 detections={tuple(detections[0].shape)}")


if __name__ == "__main__":
    main()

