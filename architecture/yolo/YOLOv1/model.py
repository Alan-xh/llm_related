"""YOLOv1 单阶段检测教学模型。

输入图像 [B, 3, H, W] 经 CNN 与自适应池化得到 SxS 网格，
输出 [B, S, S, C+10]：C 个类别 logits 加两个五维边界框预测。
推理框分数为 sigmoid(confidence) * sigmoid(class_logits)，再将网格偏移映射到像素坐标。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

try:
    from ..common import YoloV1Detector, yolo_v1_loss
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.yolo.common import YoloV1Detector, yolo_v1_loss


@dataclass
class YOLOv1Config:
    """YOLOv1 网格检测器配置，boxes_per_cell 对应每格预测框数。"""

    num_classes: int = 3
    grid_size: int = 7
    boxes_per_cell: int = 2
    width: int = 16


class YOLOv1Detector(YoloV1Detector):
    """将版本配置适配到公共 YOLOv1 骨干与单网格预测头。"""

    def __init__(self, config: YOLOv1Config | None = None) -> None:
        self.config = config or YOLOv1Config()
        super().__init__(self.config.num_classes, self.config.grid_size, self.config.width)


def build_model() -> YOLOv1Detector:
    return YOLOv1Detector()


@torch.no_grad()
def decode_yolov1(
    output: Tensor, image_size: tuple[int, int] = (64, 64), confidence: float = 0.25
) -> list[Tensor]:
    """解码每格两个预测框，返回逐图 [N_i, 6]，列为 xyxy、分数和类别编号。

    中心点：(网格坐标 + sigmoid(offset)) / grid_size * 图像尺寸；
    宽高：sigmoid(raw_wh) * 图像尺寸。
    """
    batch, grid, _, channels = output.shape
    classes = channels - 10
    boxes = output[..., classes:].reshape(batch, grid, grid, 2, 5)
    class_scores = output[..., :classes].sigmoid()
    results: list[Tensor] = []
    for image_index in range(batch):
        rows: list[Tensor] = []
        for y in range(grid):
            for x in range(grid):
                for box_index in range(2):
                    box = boxes[image_index, y, x, box_index]
                    score, class_id = (box[4].sigmoid() * class_scores[image_index, y, x]).max(0)
                    if score < confidence:
                        continue
                    cx = (x + box[0].sigmoid()) / grid * image_size[1]
                    cy = (y + box[1].sigmoid()) / grid * image_size[0]
                    width = box[2].sigmoid() * image_size[1]
                    height = box[3].sigmoid() * image_size[0]
                    rows.append(
                        torch.stack(
                            (
                                (cx - width / 2).clamp(0, image_size[1]),
                                (cy - height / 2).clamp(0, image_size[0]),
                                (cx + width / 2).clamp(0, image_size[1]),
                                (cy + height / 2).clamp(0, image_size[0]),
                                score,
                                class_id.float(),
                            )
                        )
                    )
        results.append(torch.stack(rows) if rows else output.new_zeros((0, 6)))
    return results


__all__ = ["YOLOv1Config", "YOLOv1Detector", "build_model", "decode_yolov1", "yolo_v1_loss"]


if __name__ == "__main__":
    model = build_model()
    output = model(torch.rand(1, 3, 64, 64))
    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print("output:", tuple(output.shape))
