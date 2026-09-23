"""兼容旧入口的轻量多尺度 YOLO 风格检测器。

任务：目标检测。输入 [B, 3, H, W]，输出 stride 8/16/32 的预测张量列表，
每层 shape 为 [B, 5+C, H_i, W_i]，通道依次表示四边距离、目标度和类别 logits。
边界框以网格中心和非负距离解码；具体算子复用 architecture.yolo.common。
"""

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
    """根目录兼容 API 的轻量配置；默认 3 类，检测层步幅为 8/16/32。"""

    num_classes: int = 3
    width: int = 32
    strides: tuple[int, ...] = (8, 16, 32)


class TinyYoloBackbone(TinyBackbone):
    """旧版骨干名称兼容包装，输出 P3/P4/P5 多尺度特征。"""

    def __init__(self, config: YoloConfig | None = None) -> None:
        super().__init__((config or YoloConfig()).width, style="plain")


class YoloHead(AnchorFreeHead):
    """单层无锚框检测头兼容包装；输出 [B, 5+C, H, W]。"""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__(in_channels, num_classes)


class YoloTinyDetector(MultiScaleDetector):
    """兼容旧 API 的完整轻量检测器，包含骨干、特征融合颈部和多尺度预测头。"""

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
