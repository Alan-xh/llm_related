"""MobileSAM-style student encoder with the SAM prompt/mask interface."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor

try:
    from ..common import (
        PromptableSAM,
        SAMConfig,
        TinyMobileImageEncoder,
        mask_loss,
        sam_loss,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.SAM.common import (
        PromptableSAM,
        SAMConfig,
        TinyMobileImageEncoder,
        mask_loss,
        sam_loss,
    )


class MobileSAMModel(PromptableSAM):
    """Use a depthwise-separable student while preserving SAM's decoder API."""

    def __init__(self, config: SAMConfig | None = None) -> None:
        config = config or SAMConfig()
        super().__init__(config, image_encoder=TinyMobileImageEncoder(config))

    def student_embedding(self, images: Tensor) -> Tensor:
        return self.encode_image(images)


def distillation_loss(student_embedding: Tensor, teacher_embedding: Tensor) -> Tensor:
    """Normalize both feature maps and regress the teacher embedding."""
    teacher_embedding = teacher_embedding.detach()
    student = torch.nn.functional.normalize(student_embedding.flatten(1), dim=-1)
    teacher = torch.nn.functional.normalize(teacher_embedding.flatten(1), dim=-1)
    return torch.nn.functional.mse_loss(student, teacher)


def build_model(image_size: int = 64) -> MobileSAMModel:
    return MobileSAMModel(SAMConfig(image_size=image_size, encoder_depth=1))


__all__ = ["MobileSAMModel", "build_model", "distillation_loss", "mask_loss", "sam_loss"]


if __name__ == "__main__":
    model = build_model()
    images = torch.rand(1, 3, 64, 64)
    point = torch.tensor([[[32.0, 32.0]]])
    label = torch.ones(1, 1, dtype=torch.long)
    masks, scores = model(images, point, label)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("masks:", tuple(masks.shape), "scores:", tuple(scores.shape))
