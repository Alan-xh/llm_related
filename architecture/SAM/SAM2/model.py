"""SAM 2-style image/video model with bounded streaming memory."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from ..common import SAM2Model as _SAM2Model
    from ..common import SAMConfig, VideoMemory, mask_loss, sam_loss
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.SAM.common import (
        SAM2Model as _SAM2Model,
        SAMConfig,
        VideoMemory,
        mask_loss,
        sam_loss,
    )


class SAM2Model(_SAM2Model):
    """Use the same prompt API for an image and for successive video frames."""


def build_model(image_size: int = 64) -> SAM2Model:
    return SAM2Model(SAMConfig(image_size=image_size, max_memory_frames=4))


__all__ = ["SAM2Model", "SAMConfig", "VideoMemory", "build_model", "mask_loss", "sam_loss"]


if __name__ == "__main__":
    import torch

    model = build_model()
    images = torch.rand(1, 3, 64, 64)
    point = torch.tensor([[[32.0, 32.0]]])
    label = torch.ones(1, 1, dtype=torch.long)
    masks, scores, _ = model.predict_frame(images, point, label)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("masks:", tuple(masks.shape), "scores:", tuple(scores.shape))
