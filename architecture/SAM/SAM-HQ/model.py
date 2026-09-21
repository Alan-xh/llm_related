"""SAM-HQ teaching model with a high-resolution mask feature path.

Task:
    Promptable segmentation with images ``[B,3,H,W]`` and point/box prompts.
    Outputs are masks ``[B,K,H,W]`` and IoU scores ``[B,K]``.

Core mapping:
    ``U_hq = U(image_feature) + Conv1x1(Interpolate(high_res_feature))``;
    dynamic hypernetworks compute ``mask_k = hypernet_k(token_k) dot U_hq``.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from ..common import PromptableSAM, SAMConfig, mask_loss, sam_loss
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.SAM.common import PromptableSAM, SAMConfig, mask_loss, sam_loss


class HQSAMModel(PromptableSAM):
    """Fuse projected high-resolution image features into dynamic masks."""

    def __init__(self, config: SAMConfig | None = None) -> None:
        super().__init__(config or SAMConfig(), high_quality=True)


def build_model(image_size: int = 64) -> HQSAMModel:
    """Build SAM-HQ with the high-quality decoder branch enabled."""
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
