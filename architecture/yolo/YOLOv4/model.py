"""Compact YOLOv4-style detector with CSP blocks and a PAN-like neck."""

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
class YOLOv4Config(DetectorConfig):
    width: int = 16
    use_spp: bool = True
    use_pan: bool = True


class YOLOv4Detector(MultiScaleDetector):
    """CSP-style backbone and feature fusion expose the main YOLOv4 ideas."""

    def __init__(self, config: YOLOv4Config | None = None) -> None:
        self.config = config or YOLOv4Config()
        super().__init__(self.config, backbone_style="csp", use_neck=self.config.use_pan)


def build_model() -> YOLOv4Detector:
    return YOLOv4Detector()


if __name__ == "__main__":
    import torch

    print([tuple(output.shape) for output in build_model()(torch.rand(1, 3, 64, 64))])

