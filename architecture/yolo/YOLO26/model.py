"""YOLO26 风格 DFL-free 双头检测教学模型。

输入 [B, 3, H, W] 经共享 CSP 骨干与颈部后，由 one-to-many 分支提供密集监督，
one-to-one 分支用于端到端候选预测；两分支均在 stride 8/16/32 输出
[B, 5+C, H_i, W_i]。框距离由 softplus 保证非负。
这是轻量教学抽象，不宣称与官方权重、算子或完整训练细节兼容。
"""

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
class YOLO26Config(DetectorConfig):
    """YOLO26 教学配置；progressive_loss 控制是否附加 one-to-one 辅助损失。"""

    width: int = 16
    dfl_free: bool = True
    progressive_loss: bool = True


class YOLO26Detector(DualHeadDetector):
    """构造 DFL-free 双头检测器，并按配置启用渐进式辅助监督。"""

    def __init__(self, config: YOLO26Config | None = None) -> None:
        self.config = config or YOLO26Config()
        super().__init__(
            self.config,
            backbone_style="csp",
            progressive_loss=self.config.progressive_loss,
        )


def build_model() -> YOLO26Detector:
    return YOLO26Detector()


if __name__ == "__main__":
    import torch

    outputs = build_model()(torch.rand(1, 3, 64, 64))
    print({name: [tuple(item.shape) for item in values] for name, values in outputs.items()})
