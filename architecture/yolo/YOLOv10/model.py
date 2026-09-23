"""YOLOv10 风格双头端到端检测教学模型。

输入 [B, 3, H, W]，共享 CSP 骨干和颈部输出多尺度特征；one_to_many 与
one_to_one 分支分别给出三层 [B, 5+C, H_i, W_i] 预测。
训练侧使用密集分支监督，推理侧可使用 one-to-one 候选避免常规 NMS；
当前仅抽象双头接口，并未复现官方一致双重分配等完整训练机制。
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
class YOLOv10Config(DetectorConfig):
    """YOLOv10 风格配置；use_nms_free_head 记录无 NMS 推理意图。"""

    width: int = 16
    use_nms_free_head: bool = True


class YOLOv10Detector(DualHeadDetector):
    """构造 one-to-many 与 one-to-one 两套预测头的共享特征检测器。"""

    def __init__(self, config: YOLOv10Config | None = None) -> None:
        self.config = config or YOLOv10Config()
        super().__init__(self.config, backbone_style="csp", progressive_loss=False)


def build_model() -> YOLOv10Detector:
    return YOLOv10Detector()


if __name__ == "__main__":
    import torch

    outputs = build_model()(torch.rand(1, 3, 64, 64))
    print({name: [tuple(item.shape) for item in values] for name, values in outputs.items()})
