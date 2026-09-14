"""Compact YOLO26-style detector with a DFL-free dual-head design.

YOLO26 is represented here as a current-generation end-to-end teaching
variant: a one-to-many branch supplies dense supervision, while a one-to-one
branch is decoded without NMS.  The implementation is intentionally tiny and
does not claim checkpoint or operator compatibility with an official release.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from ..common import DetectorConfig, DualHeadDetector
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.yolo.common import DetectorConfig, DualHeadDetector


@dataclass
class YOLO26Config(DetectorConfig):
    width: int = 16
    dfl_free: bool = True
    progressive_loss: bool = True


class YOLO26Detector(DualHeadDetector):
    def __init__(self, config: YOLO26Config | None = None) -> None:
        self.config = config or YOLO26Config()
        super().__init__(
            self.config,
            backbone_style="csp",
            progressive_loss=self.config.progressive_loss,
        )


def build_model() -> YOLO26Detector:
    return YOLO26Detector()


if __name__ == "__main__":
    import torch

    outputs = build_model()(torch.rand(1, 3, 64, 64))
    print({name: [tuple(item.shape) for item in values] for name, values in outputs.items()})

