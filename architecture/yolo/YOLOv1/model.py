"""Small YOLOv1-style single-stage detector."""

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
    num_classes: int = 3
    grid_size: int = 7
    boxes_per_cell: int = 2
    width: int = 16


class YOLOv1Detector(YoloV1Detector):
    def __init__(self, config: YOLOv1Config | None = None) -> None:
        self.config = config or YOLOv1Config()
        super().__init__(self.config.num_classes, self.config.grid_size, self.config.width)


def build_model() -> YOLOv1Detector:
    return YOLOv1Detector()


@torch.no_grad()
def decode_yolov1(
    output: Tensor, image_size: tuple[int, int] = (64, 64), confidence: float = 0.25
) -> list[Tensor]:
    """Decode the two boxes in each grid cell into [x1,y1,x2,y2,score,class]."""
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

