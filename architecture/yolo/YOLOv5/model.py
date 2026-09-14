"""Compact YOLOv5-style detector with C3 blocks, SPPF, and a PAN neck."""

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
class YOLOv5Config(DetectorConfig):
    width: int = 16
    depth_multiple: float = 0.33
    width_multiple: float = 0.50


class YOLOv5Detector(MultiScaleDetector):
    def __init__(self, config: YOLOv5Config | None = None) -> None:
        self.config = config or YOLOv5Config()
        super().__init__(self.config, backbone_style="sppf", use_neck=True)


def build_model() -> YOLOv5Detector:
    return YOLOv5Detector()


if __name__ == "__main__":
    import torch

    print([tuple(output.shape) for output in build_model()(torch.rand(1, 3, 64, 64))])

