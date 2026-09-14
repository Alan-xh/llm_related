"""Compact YOLOv10-style detector with one-to-many and one-to-one heads."""

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
class YOLOv10Config(DetectorConfig):
    width: int = 16
    use_nms_free_head: bool = True


class YOLOv10Detector(DualHeadDetector):
    def __init__(self, config: YOLOv10Config | None = None) -> None:
        self.config = config or YOLOv10Config()
        super().__init__(self.config, backbone_style="csp", progressive_loss=False)


def build_model() -> YOLOv10Detector:
    return YOLOv10Detector()


if __name__ == "__main__":
    import torch

    outputs = build_model()(torch.rand(1, 3, 64, 64))
    print({name: [tuple(item.shape) for item in values] for name, values in outputs.items()})

