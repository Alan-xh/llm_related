"""Train compact RT-DETR on synthetic rectangles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from ..common import add_common_train_args, train_detector
    from .model import build_model
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.detr.common import add_common_train_args, train_detector
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_train_args(parser, "rt_detr_tiny.pt")
    args = parser.parse_args()
    train_detector(build_model, steps=args.steps, batch_size=args.batch_size,
                   image_size=args.image_size, num_classes=3, lr=args.lr,
                   device=args.device, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
