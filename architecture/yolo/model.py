"""A compact multi-scale YOLO-style detector kept for backwards compatibility."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from .common import (
        ConvBNAct,
        DetectorConfig,
        MultiScaleDetector,
        TinyBackbone,
        AnchorFreeHead,
        decode_predictions,
        yolo_loss,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from architecture.yolo.common import (
        ConvBNAct,
        DetectorConfig,
        MultiScaleDetector,
        TinyBackbone,
        AnchorFreeHead,
        decode_predictions,
        yolo_loss,
    )


@dataclass
class YoloConfig(DetectorConfig):
    """Small configuration matching the original root-level teaching API."""

    num_classes: int = 3
    width: int = 32
    strides: tuple[int, ...] = (8, 16, 32)


class TinyYoloBackbone(TinyBackbone):
    """Compatibility wrapper for the former root-level detector."""

    def __init__(self, config: YoloConfig | None = None) -> None:
        super().__init__((config or YoloConfig()).width, style="plain")


class YoloHead(AnchorFreeHead):
    """Compatibility wrapper for a single feature-pyramid prediction head."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__(in_channels, num_classes)


class YoloTinyDetector(MultiScaleDetector):
    def __init__(self, config: YoloConfig | None = None) -> None:
        super().__init__(config or YoloConfig(), backbone_style="plain", use_neck=True)


__all__ = [
    "ConvBNAct",
    "YoloConfig",
    "TinyYoloBackbone",
    "YoloHead",
    "YoloTinyDetector",
    "yolo_loss",
    "decode_predictions",
]


if __name__ == "__main__":
    import torch

    model = YoloTinyDetector()
    outputs = model(torch.rand(1, 3, 64, 64))
    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print("outputs:", [tuple(output.shape) for output in outputs])
