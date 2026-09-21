"""SAM 2.1 teaching wrapper with expanded streaming memory.

Task:
    Preserve the SAM 2 frame API with a six-frame memory window. Input frames
    are ``[B,3,H,W]`` and ``predict_frame`` returns masks ``[B,K,H,W]``,
    scores ``[B,K]``, and updated ``VideoMemory``.

The ``improved`` flag records the version distinction; this is not an official
SAM 2.1 checkpoint reproduction.
"""

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


class SAM21Model(_SAM2Model):
    """Keep the SAM 2 API while exposing an improved-model configuration flag."""

    def __init__(self, config: SAMConfig | None = None) -> None:
        super().__init__(config or SAMConfig(max_memory_frames=6), improved=True)


SAM2_1Model = SAM21Model


def build_model(image_size: int = 64) -> SAM21Model:
    """Build SAM 2.1 with ``max_memory_frames=6``."""
    return SAM21Model(SAMConfig(image_size=image_size, max_memory_frames=6))


__all__ = [
    "SAM21Model",
    "SAM2_1Model",
    "SAMConfig",
    "VideoMemory",
    "build_model",
    "mask_loss",
    "sam_loss",
]


if __name__ == "__main__":
    import torch

    model = build_model()
    images = torch.rand(1, 3, 64, 64)
    point = torch.tensor([[[32.0, 32.0]]])
    label = torch.ones(1, 1, dtype=torch.long)
    masks, scores, _ = model.predict_frame(images, point, label)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("masks:", tuple(masks.shape), "scores:", tuple(scores.shape))
