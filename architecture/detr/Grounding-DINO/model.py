"""Compact Grounding DINO with text-conditioned open-vocabulary detection."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from ..common import (
        DETRConfig, MLP, TinyDETR, box_cxcywh_to_xyxy, generalized_box_iou,
        load_checkpoint, _hungarian_rect,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.detr.common import (
        DETRConfig, MLP, TinyDETR, box_cxcywh_to_xyxy, generalized_box_iou,
        load_checkpoint, _hungarian_rect,
    )


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, max_tokens: int = 32, nheads: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.position = nn.Embedding(max_tokens, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, nheads, hidden_dim * 2, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1)

    def forward(self, tokens: Tensor) -> Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device)[None]
        return self.encoder(
            self.embedding(tokens) + self.position(positions),
            src_key_padding_mask=tokens.eq(0),
        )


class GroundingDINOModel(nn.Module):
    def __init__(self, config: DETRConfig | None = None, vocab_size: int = 64):
        super().__init__()
        self.config = config or DETRConfig()
        c = self.config
        self.visual = TinyDETR(c)
        self.text = TextEncoder(vocab_size, c.hidden_dim, nheads=c.nheads)
        self.visual_to_text = nn.MultiheadAttention(c.hidden_dim, c.nheads, batch_first=True)
        self.query_embed = nn.Embedding(c.num_queries, c.hidden_dim)
        self.bbox_embed = MLP(c.hidden_dim, c.hidden_dim, 4)
        self.objectness = nn.Linear(c.hidden_dim, 1)
        self.logit_scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, images: Tensor, text_tokens: Tensor) -> dict[str, Tensor]:
        memory, memory_pos = self.visual.encode(images)
        visual = memory.transpose(0, 1)
        text = self.text(text_tokens)
        cross, _ = self.visual_to_text(
            visual, text, text, key_padding_mask=text_tokens.eq(0)
        )
        visual = visual + cross + memory_pos.transpose(0, 1)
        query_pos = self.query_embed.weight[None].expand(images.shape[0], -1, -1)
        hidden = self.visual.decoder(
            query_pos.transpose(0, 1), visual.transpose(0, 1)
        ).transpose(0, 1)
        hidden_norm = F.normalize(hidden, dim=-1)
        text_norm = F.normalize(text, dim=-1)
        token_logits = self.logit_scale.exp().clamp(max=100) * torch.einsum(
            "bqc,btc->bqt", hidden_norm, text_norm
        )
        return {
            "pred_logits": token_logits,
            "pred_boxes": self.bbox_embed(hidden).sigmoid(),
            "pred_objectness": self.objectness(hidden).squeeze(-1),
            "text_features": text,
        }


class GroundingCriterion(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes

    @torch.no_grad()
    def _match(self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]]):
        matches = []
        for logits, boxes, target in zip(outputs["pred_logits"], outputs["pred_boxes"], targets):
            labels = target["labels"]
            if not labels.numel():
                matches.append((labels.new_empty(0), labels.new_empty(0)))
                continue
            cost = -logits.sigmoid()[:, labels].transpose(0, 1)
            cost = cost + 5 * torch.cdist(target["boxes"], boxes, p=1)
            cost = cost - 2 * generalized_box_iou(
                box_cxcywh_to_xyxy(target["boxes"]), box_cxcywh_to_xyxy(boxes)
            )
            # The compact matcher delegates the rectangular assignment to the
            # same implementation used by the closed-set models.
            rows, columns = _hungarian_rect(cost.cpu().tolist())
            matches.append((
                torch.tensor(rows, dtype=torch.long, device=labels.device),
                torch.tensor(columns, dtype=torch.long, device=labels.device),
            ))
        return matches

    def forward(self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]]) -> Tensor:
        matches = self._match(outputs, targets)
        object_target = torch.zeros_like(outputs["pred_objectness"])
        loss_token = outputs["pred_logits"].sum() * 0
        matched_pred, matched_true = [], []
        for batch_index, (target_index, query_index) in enumerate(matches):
            object_target[batch_index, query_index] = 1
            if query_index.numel():
                selected = outputs["pred_logits"][batch_index, query_index]
                labels = targets[batch_index]["labels"][target_index]
                loss_token = loss_token + F.binary_cross_entropy_with_logits(
                    selected.gather(1, labels[:, None]).squeeze(1),
                    torch.ones_like(labels, dtype=selected.dtype),
                )
                matched_pred.append(outputs["pred_boxes"][batch_index, query_index])
                matched_true.append(targets[batch_index]["boxes"][target_index])
        loss_object = F.binary_cross_entropy_with_logits(outputs["pred_objectness"], object_target)
        if not matched_pred:
            return loss_object + loss_token
        pred_boxes, true_boxes = torch.cat(matched_pred), torch.cat(matched_true)
        loss_box = F.l1_loss(pred_boxes, true_boxes)
        loss_giou = (1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(true_boxes)
        ))).mean()
        return loss_object + loss_token + 5 * loss_box + 2 * loss_giou


@torch.no_grad()
def decode_grounding(
    outputs: dict[str, Tensor], text_tokens: Tensor,
    image_size: tuple[int, int] = (64, 64), threshold: float = 0.25,
) -> list[Tensor]:
    token_scores = outputs["pred_logits"].sigmoid()
    objectness = outputs["pred_objectness"].sigmoid()
    scores, token_positions = token_scores.max(-1)
    scores = scores * objectness
    results = []
    for batch_index in range(scores.shape[0]):
        keep = scores[batch_index] >= threshold
        boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"][batch_index, keep])
        height, width = image_size
        boxes = boxes * boxes.new_tensor([width, height, width, height])
        token_ids = text_tokens[batch_index, token_positions[batch_index, keep]].float()
        results.append(torch.cat((boxes, scores[batch_index, keep, None], token_ids[:, None]), dim=-1))
    return results


def build_model() -> GroundingDINOModel:
    return GroundingDINOModel()


__all__ = ["GroundingDINOModel", "GroundingCriterion", "TextEncoder",
           "build_model", "decode_grounding", "load_checkpoint"]


if __name__ == "__main__":
    model = build_model().eval()
    output = model(torch.rand(1, 3, 64, 64), torch.tensor([[1, 2, 3, 4, 5]]))
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("token logits:", tuple(output["pred_logits"].shape), "boxes:", tuple(output["pred_boxes"].shape))
