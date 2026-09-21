"""SAM 3.1 teaching model with object-token multiplexing.

Task:
    Decode multiple text/object prompts against one image. The image is
    ``[1,3,H,W]``; each prompt is an object token ``[1,1,C]``; outputs restore
    an object axis as masks ``[1,O,K,H,W]`` and scores ``[1,O,K]``.

Core optimization:
    Encode the image once, expand features to ``[O,C,h,w]``, and batch the
    prompt-conditioned decoder while preserving object token lengths.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from ..common import ObjectMultiplex, SAM31Model as _SAM31Model
    from ..common import SAMConfig, mask_loss, sam_loss
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.SAM.common import ObjectMultiplex, SAM31Model as _SAM31Model
    from architecture.SAM.common import SAMConfig, mask_loss, sam_loss


class SAM31Model(_SAM31Model):
    """Decode many concept prompts after one shared image-encoder pass."""


SAM3_1Model = SAM31Model


def build_model(image_size: int = 64) -> SAM31Model:
    """Build the multiplexed SAM 3.1 teaching model."""
    return SAM31Model(SAMConfig(image_size=image_size))


__all__ = [
    "ObjectMultiplex",
    "SAM31Model",
    "SAM3_1Model",
    "build_model",
    "mask_loss",
    "sam_loss",
]


if __name__ == "__main__":
    import torch

    model = build_model()
    output = model.multiplex(
        torch.rand(1, 3, 64, 64),
        ["rectangle", "object"],
    )
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("masks:", tuple(output["masks"].shape), "scores:", tuple(output["iou_scores"].shape))
