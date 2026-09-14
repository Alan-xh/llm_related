"""Small SAM v1-style image segmentation model."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from ..common import PromptableSAM, SAMConfig, mask_loss, sam_loss
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.SAM.common import PromptableSAM, SAMConfig, mask_loss, sam_loss


class SAMModel(PromptableSAM):
    """ViT image encoder + prompt encoder + two-way mask decoder."""


SegmentAnythingModel = SAMModel


def build_model(image_size: int = 64) -> SAMModel:
    return SAMModel(SAMConfig(image_size=image_size))


__all__ = ["SAMConfig", "SAMModel", "SegmentAnythingModel", "build_model", "mask_loss", "sam_loss"]


if __name__ == "__main__":
    import torch

    model = build_model()
    images = torch.rand(1, 3, 64, 64)
    points = torch.tensor([[[32.0, 32.0]]])
    labels = torch.ones(1, 1, dtype=torch.long)
    masks, scores = model(images, points, labels)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("masks:", tuple(masks.shape), "scores:", tuple(scores.shape))
