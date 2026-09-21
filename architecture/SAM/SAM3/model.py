"""SAM 3 teaching model for text-conditioned concept segmentation.

Task:
    Combine text, exemplar-box, and point prompts. Images are ``[B,3,H,W]``;
    text becomes concept tokens ``[B,1,C]``; outputs contain masks
    ``[B,K,H,W]``, IoU scores ``[B,K]``, presence logits ``[B,1]``, and image
    embeddings ``[B,C,H/8,W/8]``.

Core formula:
    ``presence = Linear(mean(image_feature))``; masks use the shared dynamic
    hypernetwork decoder.
"""

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
    """Build a compact text-conditioned SAM 3 model."""
    return SAM3Model(SAMConfig(image_size=image_size))


__all__ = ["SAM3Model", "ConceptSegmenter", "build_model", "mask_loss", "sam_loss"]


if __name__ == "__main__":
    import torch

    model = build_model()
    output = model(torch.rand(1, 3, 64, 64), text_prompts=["rectangle"])
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("masks:", tuple(output["masks"].shape), "presence:", tuple(output["presence_logits"].shape))
