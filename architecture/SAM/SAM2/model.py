"""SAM 2 teaching model for promptable image and video segmentation.

Task:
    Segment a current frame and propagate it through video. Frames are
    ``[B,3,H,W]``; outputs are masks ``[B,K,H,W]`` and scores ``[B,K]`` plus
    bounded ``VideoMemory`` state.

Core formula:
    ``feature' = LayerNorm(feature + Attention(feature, memory, memory))``.
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


class SAM2Model(_SAM2Model):
    """Use the same prompt API for an image and successive video frames."""


def build_model(image_size: int = 64) -> SAM2Model:
    """Build SAM 2 with a four-frame streaming memory window."""
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
