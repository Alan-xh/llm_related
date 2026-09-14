"""Compact YOLOv3-style detector with a three-level feature pyramid."""

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
class YOLOv3Config(DetectorConfig):
    width: int = 16
    anchors: tuple[tuple[tuple[float, float], ...], ...] = (
        ((10, 13), (16, 30), (33, 23)),
        ((30, 61), (62, 45), (59, 119)),
        ((116, 90), (156, 198), (373, 326)),
    )


class YOLOv3Detector(MultiScaleDetector):
    """The compact head is anchor-free in code; anchors remain explicit for study."""

    def __init__(self, config: YOLOv3Config | None = None) -> None:
        self.config = config or YOLOv3Config()
        super().__init__(self.config, backbone_style="plain", use_neck=True)


def build_model() -> YOLOv3Detector:
    return YOLOv3Detector()


if __name__ == "__main__":
    import torch

    model = build_model()
    print([tuple(output.shape) for output in model(torch.rand(1, 3, 64, 64))])

