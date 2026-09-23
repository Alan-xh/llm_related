"""YOLOv8 风格无锚框检测教学模型。

输入 [B, 3, H, W]，CSP 骨干与多尺度颈部后接解耦检测头；
每层输出 [B, 4*reg_max+1+C, H_i, W_i]，分别编码四边离散距离、
目标置信度和类别 logits。DFL 解码为 d=sum_j softmax(z)_j*j。
本实现复用教学损失，不等价于完整官方正样本分配和 DFL/IoU 损失。
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
class YOLOv8Config(DetectorConfig):
    """YOLOv8 风格配置；reg_max 是每条边离散距离分布的 bin 数。"""

    width: int = 16
    reg_max: int = 8


class YOLOv8Detector(MultiScaleDetector):
    """使用 reg_max 通道分布表示四边距离的多尺度无锚框检测器。"""

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
