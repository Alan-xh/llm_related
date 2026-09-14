"""Compact DN-DETR with noisy label/box queries during training."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from ..common import (
        DETRConfig, MLP, SetCriterion, TinyDETR, box_sine_embedding,
        decode_detections,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.detr.common import (
        DETRConfig, MLP, SetCriterion, TinyDETR, box_sine_embedding,
        decode_detections,
    )


def make_denoising_queries(
    targets: list[dict[str, Tensor]], num_classes: int, groups: int, hidden_dim: int,
    label_embedding: nn.Embedding, device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    max_targets = max((target["labels"].numel() for target in targets), default=0)
    count = max_targets * groups
    if count == 0:
        empty = torch.zeros(len(targets), 0, hidden_dim, device=device)
        return empty, empty, torch.zeros(len(targets), 0, dtype=torch.long, device=device), empty[..., :4]
    noisy_labels = torch.full((len(targets), count), num_classes, dtype=torch.long, device=device)
    clean_labels = noisy_labels.clone()
    clean_boxes = torch.zeros(len(targets), count, 4, device=device)
    for batch_index, target in enumerate(targets):
        n = target["labels"].numel()
        for group in range(groups):
            start = group * max_targets
            noisy_labels[batch_index, start:start + n] = target["labels"]
            clean_labels[batch_index, start:start + n] = target["labels"]
            clean_boxes[batch_index, start:start + n] = target["boxes"]
    noise = torch.rand_like(noisy_labels.float()) < 0.2
    replacements = torch.randint(num_classes, noisy_labels.shape, device=device)
    noisy_labels = torch.where(noise, replacements, noisy_labels)
    noisy_boxes = (clean_boxes + torch.randn_like(clean_boxes) * 0.05).clamp(0.01, 0.99)
    content = label_embedding(noisy_labels)
    position = box_sine_embedding(noisy_boxes, hidden_dim)
    valid = clean_labels != num_classes
    return content, position, clean_labels, clean_boxes.masked_fill(~valid[..., None], 0)


class DNDETRModel(nn.Module):
    def __init__(self, config: DETRConfig | None = None, denoising_groups: int = 2):
        super().__init__()
        self.config = config or DETRConfig()
        self.denoising_groups = denoising_groups
        self.core = TinyDETR(self.config)
        self.label_embedding = nn.Embedding(self.config.num_classes + 1, self.config.hidden_dim)
        self.class_embed = nn.Linear(self.config.hidden_dim, self.config.num_classes + 1)
        self.bbox_embed = MLP(self.config.hidden_dim, self.config.hidden_dim, 4)

    def forward(self, images: Tensor, targets: list[dict[str, Tensor]] | None = None) -> dict[str, Tensor]:
        memory, position = self.core.encode(images)
        batch = images.shape[0]
        query_position = self.core.query_embed.weight[None].expand(batch, -1, -1)
        query_content = torch.zeros_like(query_position)
        if targets is None:
            hidden = self.core.decode_queries(
                memory, position, query_content.transpose(0, 1), query_position.transpose(0, 1)
            )
            return {"pred_logits": self.class_embed(hidden), "pred_boxes": self.bbox_embed(hidden).sigmoid()}
        content, dn_position, dn_labels, dn_boxes = make_denoising_queries(
            targets, self.config.num_classes, self.denoising_groups, self.config.hidden_dim,
            self.label_embedding, images.device,
        )
        normal_content = query_content
        all_content = torch.cat((normal_content, content), dim=1).transpose(0, 1)
        all_position = torch.cat((query_position, dn_position), dim=1).transpose(0, 1)
        hidden = self.core.decode_queries(memory, position, all_content, all_position)
        normal_hidden = hidden[:, :self.config.num_queries]
        dn_hidden = hidden[:, self.config.num_queries:]
        return {
            "pred_logits": self.class_embed(normal_hidden),
            "pred_boxes": self.bbox_embed(normal_hidden).sigmoid(),
            "dn_logits": self.class_embed(dn_hidden),
            "dn_boxes": self.bbox_embed(dn_hidden).sigmoid(),
            "dn_labels": dn_labels,
            "dn_target_boxes": dn_boxes,
        }


class DenoisingCriterion(SetCriterion):
    def forward(self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]]) -> Tensor:
        loss = super().forward(outputs, targets)
        labels, boxes, prediction, pred_boxes = (
            outputs["dn_labels"], outputs["dn_target_boxes"],
            outputs["dn_logits"], outputs["dn_boxes"],
        )
        valid = labels != self.num_classes
        if not valid.any():
            return loss
        target_labels = labels.masked_fill(~valid, self.num_classes)
        dn_ce = F.cross_entropy(prediction.transpose(1, 2), target_labels, self.empty_weight)
        dn_bbox = F.l1_loss(pred_boxes[valid], boxes[valid])
        return loss + dn_ce + 5 * dn_bbox


def build_model() -> DNDETRModel:
    return DNDETRModel()


__all__ = ["DNDETRModel", "DenoisingCriterion", "make_denoising_queries",
           "build_model", "decode_detections"]


if __name__ == "__main__":
    from architecture.detr.common import synthetic_detection_batch

    model = build_model()
    images, targets = synthetic_detection_batch(2, 64, 3, "cpu")
    outputs = model(images, targets=targets)
    print("normal:", tuple(outputs["pred_logits"].shape), "denoising:", tuple(outputs["dn_logits"].shape))
