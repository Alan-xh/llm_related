"""FastSAM teaching model: candidate generation followed by prompt ranking.

Task:
    Generate all candidate masks once, then select top candidates for a point
    prompt. Candidate masks are ``[B,N,H,W]`` and scores ``[B,N]``; selected
    outputs are ``[B,K,H,W]`` and ``[B,K]``.

Core formula:
    ``score = 0.5*model_score + 0.5*prompt_agreement``. This is a compact
    educational approximation, not the official detector/NMS implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from ..common import SAMConfig, TinyMobileImageEncoder, mask_loss, sam_loss
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.SAM.common import SAMConfig, TinyMobileImageEncoder, mask_loss, sam_loss


class FastSAMModel(nn.Module):
    """Generate a fixed candidate bank once, then rank it using a prompt."""

    def __init__(self, config: Optional[SAMConfig] = None, num_candidates: int = 8) -> None:
        super().__init__()
        self.config = config or SAMConfig()
        self.num_candidates = num_candidates
        self.image_encoder = TinyMobileImageEncoder(self.config)
        self.mask_head = nn.Conv2d(self.config.embed_dim, num_candidates, 1)
        self.score_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.config.embed_dim, num_candidates),
        )

    def generate_candidates(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Return all mask logits ``[B,N,H,W]`` and scores ``[B,N]``."""
        features = self.image_encoder(images)
        logits = self.mask_head(features)  # [B,C,h,w] -> [B,N,h,w].
        masks = F.interpolate(
            logits, size=images.shape[-2:], mode="bilinear", align_corners=False
        )
        # [B,N,h,w] -> [B,N,H,W].
        scores = self.score_head(features).sigmoid()
        return masks, scores

    def select_candidates(
        self,
        masks: Tensor,
        scores: Tensor,
        point_coords: Optional[Tensor],
        point_labels: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Rank candidates and return selected masks ``[B,K,H,W]``/scores ``[B,K]``.

        Inputs are masks ``[B,N,H,W]``, scores ``[B,N]``, points ``[B,P,2]``,
        and labels ``[B,P]``.
        """
        if point_coords is not None:
            if point_labels is None:
                raise ValueError("point_labels is required when point_coords is provided")
            height, width = masks.shape[-2:]
            x = point_coords[..., 0].round().long().clamp(0, width - 1)
            y = point_coords[..., 1].round().long().clamp(0, height - 1)
            flat_masks = masks.sigmoid().flatten(2)
            # [B,N,H,W] -> [B,N,H*W] for point sampling.
            point_indices = (y * width + x)[:, None, :]
            sampled = flat_masks.gather(
                2, point_indices.expand(-1, flat_masks.shape[1], -1)
            )
            positive = point_labels.eq(1).float()
            negative = point_labels.eq(0).float()
            prompt_score = (sampled * positive[:, None, :]).sum(-1)
            prompt_score = prompt_score + ((1.0 - sampled) * negative[:, None, :]).sum(-1)
            prompt_score = prompt_score / (positive + negative).sum(-1, keepdim=True).clamp_min(1.0)
            scores = 0.5 * scores + 0.5 * prompt_score
        count = min(self.config.num_multimask_outputs, self.num_candidates)
        indices = scores.topk(count, dim=1).indices
        selected_masks = masks.gather(
            1, indices[..., None, None].expand(-1, -1, masks.shape[-2], masks.shape[-1])
        )
        selected_scores = scores.gather(1, indices)
        return selected_masks, selected_scores

    def forward(
        self,
        images: Tensor,
        point_coords: Optional[Tensor] = None,
        point_labels: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Generate candidates and return the prompt-selected top-K outputs."""
        masks, scores = self.generate_candidates(images)
        return self.select_candidates(masks, scores, point_coords, point_labels)

    @torch.no_grad()
    def predict_all(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Return the complete candidate bank ``[B,N,H,W]`` and ``[B,N]``."""
        return self.generate_candidates(images)


def candidate_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute pairwise box IoU ``[N,4] x [M,4] -> [N,M]``.

    The formula is ``intersection / (area1 + area2 - intersection)``.
    """
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(-1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0).prod(-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0).prod(-1)
    return intersection / (area1[:, None] + area2[None, :] - intersection + 1e-6)


def build_model(image_size: int = 64) -> FastSAMModel:
    """Build a compact FastSAM candidate generator."""
    return FastSAMModel(SAMConfig(image_size=image_size))


__all__ = ["FastSAMModel", "build_model", "candidate_iou", "mask_loss", "sam_loss"]


if __name__ == "__main__":
    model = build_model()
    images = torch.rand(1, 3, 64, 64)
    point = torch.tensor([[[32.0, 32.0]]])
    label = torch.ones(1, 1, dtype=torch.long)
    masks, scores = model(images, point, label)
    print("masks:", tuple(masks.shape), "scores:", tuple(scores.shape))
