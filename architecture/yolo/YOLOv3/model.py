"""YOLOv3 风格的三级特征金字塔检测教学模型。

输入 [B, 3, H, W]，经骨干与 top-down 融合后在 stride 8/16/32 上预测，
输出各为 [B, 5+C, H_i, W_i]。配置保留经典 anchor 供学习参考，
但当前共享检测头实际采用无锚框距离回归，并未消费 anchors。
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
class YOLOv3Config(DetectorConfig):
    """YOLOv3 教学配置；anchors 为经典先验框记录，不参与当前简化头计算。"""

    width: int = 16
    anchors: tuple[tuple[tuple[float, float], ...], ...] = (
        ((10, 13), (16, 30), (33, 23)),
        ((30, 61), (62, 45), (59, 119)),
        ((116, 90), (156, 198), (373, 326)),
    )


class YOLOv3Detector(MultiScaleDetector):
    """普通卷积骨干加多尺度融合头；检测输出为 anchor-free 距离格式。"""

    def __init__(self, config: YOLOv3Config | None = None) -> None:
        self.config = config or YOLOv3Config()
        super().__init__(self.config, backbone_style="plain", use_neck=True)


def build_model() -> YOLOv3Detector:
    return YOLOv3Detector()


if __name__ == "__main__":
    import torch

    model = build_model()
    print([tuple(output.shape) for output in model(torch.rand(1, 3, 64, 64))])
