"""SAM-HQ-style decoder with an additional high-resolution feature path."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from ..common import PromptableSAM, SAMConfig, mask_loss, sam_loss
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.SAM.common import PromptableSAM, SAMConfig, mask_loss, sam_loss


class HQSAMModel(PromptableSAM):
    """Fuse projected image features into the dynamic mask embedding."""

    def __init__(self, config: SAMConfig | None = None) -> None:
        super().__init__(config or SAMConfig(), high_quality=True)


def build_model(image_size: int = 64) -> HQSAMModel:
    return HQSAMModel(SAMConfig(image_size=image_size))


__all__ = ["HQSAMModel", "build_model", "mask_loss", "sam_loss"]


if __name__ == "__main__":
    import torch

    model = build_model()
    images = torch.rand(1, 3, 64, 64)
    point = torch.tensor([[[32.0, 32.0]]])
    label = torch.ones(1, 1, dtype=torch.long)
    masks, scores = model(images, point, label)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("masks:", tuple(masks.shape), "scores:", tuple(scores.shape))
