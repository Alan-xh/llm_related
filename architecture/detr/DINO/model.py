"""Compact DINO-style DETR with two-stage query selection and contrastive DN."""

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


class DINOModel(nn.Module):
    def __init__(self, config: DETRConfig | None = None, denoising_groups: int = 2):
        super().__init__()
        self.config = config or DETRConfig()
        self.denoising_groups = denoising_groups
        c = self.config
        self.core = TinyDETR(c)
        self.encoder_class = nn.Linear(c.hidden_dim, c.num_classes + 1)
        self.encoder_box = MLP(c.hidden_dim, c.hidden_dim, 4)
        self.label_embedding = nn.Embedding(c.num_classes + 1, c.hidden_dim)
        self.class_embed = nn.Linear(c.hidden_dim, c.num_classes + 1)
        self.bbox_embed = MLP(c.hidden_dim, c.hidden_dim, 4)

    def _denoising(self, targets: list[dict[str, Tensor]], device: torch.device):
        max_targets = max((target["labels"].numel() for target in targets), default=0)
        count = max_targets * self.denoising_groups
        labels = torch.full((len(targets), count), self.config.num_classes,
                            dtype=torch.long, device=device)
        boxes = torch.zeros(len(targets), count, 4, device=device)
        for batch_index, target in enumerate(targets):
            for group in range(self.denoising_groups):
                start = group * max_targets
                n = target["labels"].numel()
                labels[batch_index, start:start + n] = target["labels"]
                boxes[batch_index, start:start + n] = target["boxes"]
        # Half of the denoising queries receive a wrong class, providing a
        # compact contrastive positive/negative training signal.
        noisy = labels.clone()
        replace = (torch.rand_like(noisy.float()) < 0.25) & (noisy != self.config.num_classes)
        noisy[replace] = torch.randint(self.config.num_classes, noisy.shape, device=device)[replace]
        noisy_boxes = (boxes + torch.randn_like(boxes) * 0.04).clamp(0.01, 0.99)
        content = self.label_embedding(noisy)
        position = box_sine_embedding(noisy_boxes, self.config.hidden_dim)
        return content, position, labels, boxes

    def forward(self, images: Tensor, targets: list[dict[str, Tensor]] | None = None) -> dict[str, Tensor]:
        memory, memory_pos = self.core.encode(images)
        memory_batch = memory.transpose(0, 1)
        encoder_scores = self.encoder_class(memory_batch).softmax(-1)[..., :-1].amax(-1)
        encoder_boxes = self.encoder_box(memory_batch).sigmoid()
        query_count = self.config.num_queries
        selected = encoder_scores.topk(min(query_count, memory_batch.shape[1]), dim=1).indices
        gather_index = selected[..., None].expand(-1, -1, self.config.hidden_dim)
        query_content = memory_batch.gather(1, gather_index)
        query_boxes = encoder_boxes.gather(1, selected[..., None].expand(-1, -1, 4))
        if query_content.shape[1] < query_count:
            pad = query_count - query_content.shape[1]
            query_content = torch.cat((query_content, query_content[:, :1].expand(-1, pad, -1)), dim=1)
            query_boxes = torch.cat((query_boxes, query_boxes[:, :1].expand(-1, pad, -1)), dim=1)
        query_position = box_sine_embedding(query_boxes.detach(), self.config.hidden_dim)
        dn_labels = dn_boxes = None
        if targets is not None:
            dn_content, dn_position, dn_labels, dn_boxes = self._denoising(targets, images.device)
            query_content = torch.cat((query_content, dn_content), dim=1)
            query_position = torch.cat((query_position, dn_position), dim=1)
        hidden = self.core.decode_queries(
            memory, memory_pos, query_content.transpose(0, 1), query_position.transpose(0, 1)
        )
        normal_hidden = hidden[:, :query_count]
        output = {
            "pred_logits": self.class_embed(normal_hidden),
            "pred_boxes": self.bbox_embed(normal_hidden).sigmoid(),
            "enc_logits": self.encoder_class(memory_batch),
            "enc_boxes": encoder_boxes,
        }
        if targets is not None:
            dn_hidden = hidden[:, query_count:]
            output.update({
                "dn_logits": self.class_embed(dn_hidden),
                "dn_boxes": self.bbox_embed(dn_hidden).sigmoid(),
                "dn_labels": dn_labels,
                "dn_target_boxes": dn_boxes,
            })
        return output


class DINOSetCriterion(SetCriterion):
    def forward(self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]]) -> Tensor:
        loss = super().forward(outputs, targets)
        if "dn_logits" not in outputs or outputs["dn_logits"].shape[1] == 0:
            return loss
        labels, boxes = outputs["dn_labels"], outputs["dn_target_boxes"]
        valid = labels != self.num_classes
        if not valid.any():
            return loss
        dn_ce = F.cross_entropy(
            outputs["dn_logits"].transpose(1, 2),
            labels.masked_fill(~valid, self.num_classes),
            self.empty_weight,
        )
        return loss + dn_ce + 5 * F.l1_loss(outputs["dn_boxes"][valid], boxes[valid])


def build_model() -> DINOModel:
    return DINOModel()


__all__ = ["DINOModel", "DINOSetCriterion", "build_model", "decode_detections"]


if __name__ == "__main__":
    from architecture.detr.common import synthetic_detection_batch

    model = build_model()
    images, targets = synthetic_detection_batch(2, 64, 3, "cpu")
    output = model(images, targets=targets)
    print("normal:", tuple(output["pred_logits"].shape), "denoising:", tuple(output["dn_logits"].shape))
