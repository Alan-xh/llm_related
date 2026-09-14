"""SAM 3.1-style multiplexed concept segmentation."""

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
