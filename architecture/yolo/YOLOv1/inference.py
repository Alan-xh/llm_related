"""Decode predictions from the YOLOv1 teaching model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    from .model import build_model, decode_yolov1
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.yolo.YOLOv1.model import build_model, decode_yolov1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--checkpoint")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model = build_model().to(args.device).eval()
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        model.load_state_dict(state.get("model", state))
    output = model(torch.rand(1, 3, args.image_size, args.image_size, device=args.device))
    detections = decode_yolov1(output, (args.image_size, args.image_size), args.confidence)
    print(f"image=0 detections={tuple(detections[0].shape)}")


if __name__ == "__main__":
    main()

