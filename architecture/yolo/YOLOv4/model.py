"""YOLOv4 风格轻量目标检测教学模型。

输入 [B, 3, H, W]，CSP 风格骨干提取三级特征，检测头输出 stride 8/16/32
上的 [B, 5+C, H_i, W_i] 张量。配置开关控制是否启用融合颈部；
本实现展示结构思路，不包含完整官方 SPP/PAN 与训练配方。
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
class YOLOv4Config(DetectorConfig):
    """YOLOv4 教学配置；use_pan 控制公共 top-down neck 是否启用。"""

    width: int = 16
    use_spp: bool = True
    use_pan: bool = True


class YOLOv4Detector(MultiScaleDetector):
    """通过 CSP 风格骨干和可选多尺度融合展示 YOLOv4 的主要结构概念。"""

    def __init__(self, config: YOLOv4Config | None = None) -> None:
        self.config = config or YOLOv4Config()
        super().__init__(self.config, backbone_style="csp", use_neck=self.config.use_pan)


def build_model() -> YOLOv4Detector:
    return YOLOv4Detector()


if __name__ == "__main__":
    import torch

    print([tuple(output.shape) for output in build_model()(torch.rand(1, 3, 64, 64))])
