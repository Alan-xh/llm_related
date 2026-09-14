"""Shared, CPU-friendly building blocks for the DETR teaching examples.

The implementations in this directory are intentionally small.  They keep
the important tensor contracts of the papers while using synthetic rectangles
and compact Transformer layers so every example can run on a laptop.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class DETRConfig:
    num_classes: int = 3
    num_queries: int = 16
    hidden_dim: int = 64
    nheads: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    dim_feedforward: int = 128
    backbone_channels: int = 32
    dropout: float = 0.0


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    x = x.clamp(min=eps, max=1 - eps)
    return torch.log(x / (1 - x))


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, width, height = boxes.unbind(-1)
    return torch.stack(
        (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2),
        dim=-1,
    )


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Pairwise IoU for ``[..., N, 4]`` and ``[..., M, 4]`` boxes."""
    area1 = ((boxes1[..., 2] - boxes1[..., 0]).clamp_min(0)
             * (boxes1[..., 3] - boxes1[..., 1]).clamp_min(0))
    area2 = ((boxes2[..., 2] - boxes2[..., 0]).clamp_min(0)
             * (boxes2[..., 3] - boxes2[..., 1]).clamp_min(0))
    left_top = torch.maximum(boxes1[..., :, None, :2], boxes2[..., None, :, :2])
    right_bottom = torch.minimum(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])
    inter = (right_bottom - left_top).clamp_min(0)
    inter = inter[..., 0] * inter[..., 1]
    union = area1[..., :, None] + area2[..., None, :] - inter
    return inter / union.clamp_min(1e-6)


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter = (rb - lt).clamp_min(0)
    inter = inter[:, :, 0] * inter[:, :, 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp_min(0)
             * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp_min(0)
             * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0))
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp_min(1e-6)
    c_lt = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    c_rb = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    c_area = ((c_rb - c_lt).clamp_min(0)[..., 0]
              * (c_rb - c_lt).clamp_min(0)[..., 1])
    return iou - (c_area - union) / c_area.clamp_min(1e-6)


def _hungarian_rect(cost: list[list[float]]) -> tuple[list[int], list[int]]:
    """Solve a rectangular linear assignment problem without SciPy."""
    rows, cols = len(cost), len(cost[0]) if cost else 0
    if not rows or not cols:
        return [], []
    if rows > cols:
        transposed = [list(column) for column in zip(*cost)]
        assigned_cols, assigned_rows = _hungarian_rect(transposed)
        return assigned_rows, assigned_cols
    u = [0.0] * (rows + 1)
    v = [0.0] * (cols + 1)
    p = [0] * (cols + 1)
    way = [0] * (cols + 1)
    for row in range(1, rows + 1):
        p[0] = row
        column0 = 0
        min_value = [float("inf")] * (cols + 1)
        used = [False] * (cols + 1)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = float("inf")
            column1 = 0
            for column in range(1, cols + 1):
                if used[column]:
                    continue
                current = cost[row0 - 1][column - 1] - u[row0] - v[column]
                if current < min_value[column]:
                    min_value[column] = current
                    way[column] = column0
                if min_value[column] < delta:
                    delta = min_value[column]
                    column1 = column
            for column in range(cols + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    min_value[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    row_indices = []
    column_indices = []
    for column in range(1, cols + 1):
        if p[column]:
            row_indices.append(p[column] - 1)
            column_indices.append(column - 1)
    order = sorted(range(len(row_indices)), key=row_indices.__getitem__)
    return [row_indices[i] for i in order], [column_indices[i] for i in order]


class HungarianMatcher(nn.Module):
    """One-to-one matcher used by DETR-style set losses."""

    def __init__(self, class_cost: float = 1.0, bbox_cost: float = 5.0, giou_cost: float = 2.0):
        super().__init__()
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost

    @torch.no_grad()
    def forward(self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]]):
        probabilities = outputs["pred_logits"].softmax(-1)
        boxes = outputs["pred_boxes"]
        result: list[tuple[Tensor, Tensor]] = []
        for probability, prediction, target in zip(probabilities, boxes, targets):
            labels, target_boxes = target["labels"], target["boxes"]
            if labels.numel() == 0:
                result.append((labels.new_empty((0,), dtype=torch.long),
                               labels.new_empty((0,), dtype=torch.long)))
                continue
            class_cost = -probability[:, labels].transpose(0, 1)
            bbox_cost = torch.cdist(target_boxes, prediction, p=1)
            giou_cost = -generalized_box_iou(
                box_cxcywh_to_xyxy(target_boxes),
                box_cxcywh_to_xyxy(prediction),
            )
            cost = (
                self.class_cost * class_cost
                + self.bbox_cost * bbox_cost
                + self.giou_cost * giou_cost
            )
            rows, columns = _hungarian_rect(cost.detach().cpu().tolist())
            result.append((
                torch.tensor(rows, dtype=torch.long, device=labels.device),
                torch.tensor(columns, dtype=torch.long, device=labels.device),
            ))
        return result


class SinePositionEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, temperature: int = 10000):
        super().__init__()
        if hidden_dim % 4:
            raise ValueError("hidden_dim must be divisible by 4 for 2D sine positions")
        self.num_pos_feats = hidden_dim // 2
        self.temperature = temperature

    def forward(self, feature: Tensor) -> Tensor:
        batch, _, height, width = feature.shape
        device = feature.device
        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=feature.dtype),
            torch.arange(width, device=device, dtype=feature.dtype),
            indexing="ij",
        )
        scale = 2 * math.pi
        y = (y + 0.5) / max(height, 1) * scale
        x = (x + 0.5) / max(width, 1) * scale
        dim_t = self.temperature ** (
            2 * torch.div(
                torch.arange(self.num_pos_feats, device=device),
                2,
                rounding_mode="floor",
            ).float()
            / self.num_pos_feats
        )
        pos_x = x[None, :, :, None] / dim_t
        pos_y = y[None, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        position = torch.cat((pos_y, pos_x), dim=-1).permute(0, 3, 1, 2)
        return position.expand(batch, -1, -1, -1)


class TinyBackbone(nn.Module):
    """Three-stride CNN producing an 1/8-resolution feature map."""

    def __init__(self, hidden_dim: int = 64, channels: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, channels, 3, stride=2, padding=1),
            nn.GroupNorm(4, channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, channels * 2),
            nn.ReLU(),
            nn.Conv2d(channels * 2, hidden_dim, 3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.body(images)


class TinyMultiScaleBackbone(nn.Module):
    def __init__(self, hidden_dim: int = 64, channels: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, stride=2, padding=1),
            nn.GroupNorm(4, channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, channels * 2),
            nn.ReLU(),
        )
        self.levels = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels * 2, hidden_dim, 3, stride=2, padding=1),
                          nn.GroupNorm(8, hidden_dim), nn.ReLU()),
            nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
                          nn.GroupNorm(8, hidden_dim), nn.ReLU()),
            nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
                          nn.GroupNorm(8, hidden_dim), nn.ReLU()),
        ])

    def forward(self, images: Tensor) -> list[Tensor]:
        feature = self.stem(images)
        outputs = []
        for level in self.levels:
            feature = level(feature)
            outputs.append(feature)
        return outputs


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layers: int = 3):
        super().__init__()
        widths = [input_dim] + [hidden_dim] * (layers - 1) + [output_dim]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(widths, widths[1:]))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


def box_sine_embedding(boxes: Tensor, hidden_dim: int) -> Tensor:
    """Encode normalized cxcywh boxes as a ``[B, Q, hidden_dim]`` position."""
    if hidden_dim % 8:
        raise ValueError("hidden_dim must be divisible by 8 for box positions")
    num_feats = hidden_dim // 8
    dim_t = 10000 ** (
        2 * torch.div(torch.arange(num_feats, device=boxes.device), 2,
                      rounding_mode="floor").float() / num_feats
    )
    scaled = boxes[..., None] * (2 * math.pi)
    encoded = scaled / dim_t
    encoded = torch.stack((encoded.sin(), encoded.cos()), dim=-1).flatten(-2)
    return encoded.flatten(-2)


class TinyDETR(nn.Module):
    """Compact original DETR with a single feature level."""

    def __init__(self, config: DETRConfig | None = None):
        super().__init__()
        self.config = config or DETRConfig()
        c = self.config
        self.backbone = TinyBackbone(c.hidden_dim, c.backbone_channels)
        self.position_embedding = SinePositionEmbedding(c.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            c.hidden_dim, c.nheads, c.dim_feedforward, c.dropout, batch_first=False
        )
        decoder_layer = nn.TransformerDecoderLayer(
            c.hidden_dim, c.nheads, c.dim_feedforward, c.dropout, batch_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, c.encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, c.decoder_layers)
        self.query_embed = nn.Embedding(c.num_queries, c.hidden_dim)
        self.class_embed = nn.Linear(c.hidden_dim, c.num_classes + 1)
        self.bbox_embed = MLP(c.hidden_dim, c.hidden_dim, 4)

    def encode(self, images: Tensor) -> tuple[Tensor, Tensor]:
        feature = self.backbone(images)
        position = self.position_embedding(feature)
        source = (feature + position).flatten(2).permute(2, 0, 1)
        position = position.flatten(2).permute(2, 0, 1)
        return self.encoder(source), position

    def decode_queries(
        self, memory: Tensor, position: Tensor, query_content: Tensor, query_position: Tensor
    ) -> Tensor:
        target = torch.zeros_like(query_content)
        return self.decoder(
            target + query_content + query_position,
            memory + position,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
        ).transpose(0, 1)

    def forward(self, images: Tensor, **_: object) -> dict[str, Tensor]:
        memory, position = self.encode(images)
        batch = images.shape[0]
        query_position = self.query_embed.weight[None].expand(batch, -1, -1)
        query_content = torch.zeros_like(query_position)
        hidden = self.decode_queries(
            memory, position, query_content.transpose(0, 1), query_position.transpose(0, 1)
        )
        return {
            "pred_logits": self.class_embed(hidden),
            "pred_boxes": self.bbox_embed(hidden).sigmoid(),
        }


class SetCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher | None = None,
        no_object_weight: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher or HungarianMatcher()
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = no_object_weight
        self.register_buffer("empty_weight", empty_weight)

    def forward(self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]]) -> Tensor:
        matches = self.matcher(outputs, targets)
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        target_classes = torch.full(
            logits.shape[:2], self.num_classes, dtype=torch.long, device=logits.device
        )
        matched_pred, matched_target = [], []
        for batch_index, (target_index, query_index) in enumerate(matches):
            target_classes[batch_index, query_index] = targets[batch_index]["labels"][target_index]
            if query_index.numel():
                matched_pred.append(boxes[batch_index, query_index])
                matched_target.append(targets[batch_index]["boxes"][target_index])
        loss_ce = F.cross_entropy(logits.transpose(1, 2), target_classes, self.empty_weight)
        if matched_pred:
            pred_boxes = torch.cat(matched_pred)
            true_boxes = torch.cat(matched_target)
            loss_bbox = F.l1_loss(pred_boxes, true_boxes)
            loss_giou = (1 - torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(pred_boxes),
                box_cxcywh_to_xyxy(true_boxes),
            ))).mean()
        else:
            loss_bbox = boxes.sum() * 0
            loss_giou = boxes.sum() * 0
        return loss_ce + 5 * loss_bbox + 2 * loss_giou


def synthetic_detection_batch(
    batch_size: int, image_size: int, num_classes: int, device: torch.device | str
) -> tuple[Tensor, list[dict[str, Tensor]]]:
    """Create colored rectangles and normalized cxcywh labels."""
    device = torch.device(device)
    images = torch.rand(batch_size, 3, image_size, image_size, device=device) * 0.08
    targets: list[dict[str, Tensor]] = []
    colors = torch.tensor(
        [[0.9, 0.15, 0.12], [0.12, 0.8, 0.2], [0.1, 0.3, 0.95]],
        device=device,
    )
    for batch_index in range(batch_size):
        count = random.randint(1, min(3, num_classes + 1))
        labels, boxes = [], []
        for _ in range(count):
            width = random.randint(max(6, image_size // 8), max(8, image_size // 3))
            height = random.randint(max(6, image_size // 8), max(8, image_size // 3))
            x0 = random.randint(0, max(0, image_size - width - 1))
            y0 = random.randint(0, max(0, image_size - height - 1))
            class_id = random.randrange(num_classes)
            images[batch_index, :, y0:y0 + height, x0:x0 + width] = colors[class_id % 3, :, None, None]
            labels.append(class_id)
            boxes.append([
                (x0 + width / 2) / image_size,
                (y0 + height / 2) / image_size,
                width / image_size,
                height / image_size,
            ])
        targets.append({
            "labels": torch.tensor(labels, dtype=torch.long, device=device),
            "boxes": torch.tensor(boxes, dtype=torch.float32, device=device),
        })
    return images, targets


@torch.no_grad()
def decode_detections(
    outputs: dict[str, Tensor], image_size: tuple[int, int] = (64, 64), threshold: float = 0.25
) -> list[Tensor]:
    logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
    probabilities = logits.softmax(-1)
    height, width = image_size
    results = []
    for probability, box in zip(probabilities, boxes):
        scores, labels = probability[..., :-1].max(-1)
        keep = scores >= threshold
        xyxy = box_cxcywh_to_xyxy(box[keep])
        scale = xyxy.new_tensor([width, height, width, height])
        xyxy = (xyxy * scale).clamp_min(0)
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp_max(width)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp_max(height)
        results.append(torch.cat((xyxy, scores[keep, None], labels[keep, None].float()), dim=-1))
    return results


def load_checkpoint(model: nn.Module, path: str | None, device: torch.device) -> None:
    if not path:
        return
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state.get("model", state))


def train_detector(
    build_model: Callable[[], nn.Module],
    *,
    steps: int,
    batch_size: int,
    image_size: int,
    num_classes: int,
    lr: float,
    device: str,
    checkpoint: str,
) -> None:
    torch.manual_seed(0)
    model = build_model().to(device)
    criterion = SetCriterion(num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for step in range(steps):
        images, targets = synthetic_detection_batch(batch_size, image_size, num_classes, device)
        loss = criterion(model(images, targets=targets), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(f"step={step + 1}/{steps} loss={loss.item():.4f}")
    torch.save({"model": model.state_dict()}, checkpoint)
    print(f"saved checkpoint: {checkpoint}")


def add_common_train_args(parser: argparse.ArgumentParser, default_checkpoint: str) -> None:
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default=default_checkpoint)


__all__ = [
    "DETRConfig", "HungarianMatcher", "SetCriterion", "TinyDETR", "TinyBackbone",
    "TinyMultiScaleBackbone", "MLP", "SinePositionEmbedding", "box_cxcywh_to_xyxy",
    "box_iou", "generalized_box_iou", "box_sine_embedding", "inverse_sigmoid",
    "synthetic_detection_batch", "decode_detections", "load_checkpoint",
    "train_detector", "add_common_train_args",
]
