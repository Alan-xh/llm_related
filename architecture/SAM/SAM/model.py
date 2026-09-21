"""SAM v1 teaching model: promptable image segmentation.

Task:
    Segment objects from point, box, or mask prompts. Input images use
    ``[B,3,H,W]``; point prompts use ``[B,N,2]`` and labels ``[B,N]``.

Architecture:
    Tiny ViT image encoder -> random Fourier prompt encoder -> Two-Way
    Transformer -> dynamic mask hypernetworks. Outputs are masks
    ``[B,K,H,W]`` and predicted IoU scores ``[B,K]``.

Core mapping:
    ``PE(x)=[sin(2*pi*B*x), cos(2*pi*B*x)]`` and
    ``mask_k=hypernet_k(mask_token_k) dot upscaled_image_feature``.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from ..common import PromptableSAM, SAMConfig, mask_loss, sam_loss
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.SAM.common import PromptableSAM, SAMConfig, mask_loss, sam_loss


class SAMModel(PromptableSAM):
    """ViT image encoder + prompt encoder + two-way mask decoder.

    The call contract is images ``[B,3,H,W]`` -> masks ``[B,K,H,W]`` and
    scores ``[B,K]``.
    """


SegmentAnythingModel = SAMModel


def build_model(image_size: int = 64) -> SAMModel:
    """Build a compact SAM v1 model for square images of ``image_size``."""
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
