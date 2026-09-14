"""Compact YOLOv8-style anchor-free detector with a DFL regression head."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from ..common import DetectorConfig, MultiScaleDetector
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.yolo.common import DetectorConfig, MultiScaleDetector


@dataclass
class YOLOv8Config(DetectorConfig):
    width: int = 16
    reg_max: int = 8


class YOLOv8Detector(MultiScaleDetector):
    def __init__(self, config: YOLOv8Config | None = None) -> None:
        self.config = config or YOLOv8Config()
        super().__init__(
            self.config,
            backbone_style="csp",
            use_neck=True,
            regression_bins=self.config.reg_max,
        )


def build_model() -> YOLOv8Detector:
    return YOLOv8Detector()


if __name__ == "__main__":
    import torch

    model = build_model()
    print([tuple(output.shape) for output in model(torch.rand(1, 3, 64, 64))])

