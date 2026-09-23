"""YOLOv5 风格轻量多尺度检测教学模型。

输入 [B, 3, H, W]，骨干使用 CSP/C3 风格模块与 SPPF，之后进行特征融合；
输出 stride 8/16/32 的预测列表，每层 shape 为 [B, 5+C, H_i, W_i]。
depth_multiple、width_multiple 用于说明配置缩放概念，当前最小模型并未应用它们。
"""

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
    """YOLOv5 风格配置；深度和宽度倍率仅作配置概念示例。"""

    width: int = 16
    depth_multiple: float = 0.33
    width_multiple: float = 0.50


class YOLOv5Detector(MultiScaleDetector):
    """采用 SPPF 末端骨干和多尺度颈部的轻量检测器。"""

    def __init__(self, config: YOLOv5Config | None = None) -> None:
        self.config = config or YOLOv5Config()
        super().__init__(self.config, backbone_style="sppf", use_neck=True)


def build_model() -> YOLOv5Detector:
    return YOLOv5Detector()


if __name__ == "__main__":
    import torch

    print([tuple(output.shape) for output in build_model()(torch.rand(1, 3, 64, 64))])
