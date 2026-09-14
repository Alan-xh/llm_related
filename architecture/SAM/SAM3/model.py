"""SAM 3-style concept segmentation with text and exemplar prompts."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from ..common import SAM3Model as _SAM3Model
    from ..common import SAMConfig, mask_loss, sam_loss
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.SAM.common import SAM3Model as _SAM3Model
    from architecture.SAM.common import SAMConfig, mask_loss, sam_loss


class SAM3Model(_SAM3Model):
    """Return masks, quality scores, presence logits, and image embeddings."""


ConceptSegmenter = SAM3Model


def build_model(image_size: int = 64) -> SAM3Model:
    return SAM3Model(SAMConfig(image_size=image_size))


__all__ = ["SAM3Model", "ConceptSegmenter", "build_model", "mask_loss", "sam_loss"]


if __name__ == "__main__":
    import torch

    model = build_model()
    output = model(torch.rand(1, 3, 64, 64), text_prompts=["rectangle"])
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("masks:", tuple(output["masks"].shape), "presence:", tuple(output["presence_logits"].shape))
