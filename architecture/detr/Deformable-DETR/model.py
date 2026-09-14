"""Compact Deformable-DETR with sparse multi-scale grid sampling."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor, nn

try:
    from ..common import (
        DETRConfig, MLP, SetCriterion, SinePositionEmbedding, TinyMultiScaleBackbone,
        decode_detections,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.detr.common import (
        DETRConfig, MLP, SetCriterion, SinePositionEmbedding, TinyMultiScaleBackbone,
        decode_detections,
    )


class MultiScaleDeformableAttention(nn.Module):
    """Sample a few points around each reference point from every feature level."""

    def __init__(self, hidden_dim: int, nheads: int = 4, levels: int = 3, points: int = 4):
        super().__init__()
        if hidden_dim % nheads:
            raise ValueError("hidden_dim must be divisible by nheads")
        self.hidden_dim, self.nheads, self.levels, self.points = hidden_dim, nheads, levels, points
        self.head_dim = hidden_dim // nheads
        self.offsets = nn.Linear(hidden_dim, nheads * levels * points * 2)
        self.weights = nn.Linear(hidden_dim, nheads * levels * points)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query: Tensor, references: Tensor, features: list[Tensor]) -> Tensor:
        batch, length, _ = query.shape
        offsets = self.offsets(query).view(batch, length, self.nheads, self.levels, self.points, 2)
        weights = self.weights(query).view(batch, length, self.nheads, self.levels, self.points)
        weights = weights.flatten(2).softmax(-1).view_as(weights)
        output = query.new_zeros(batch, length, self.nheads, self.head_dim)
        for level, feature in enumerate(features):
            height, width = feature.shape[-2:]
            value = self.value(feature.flatten(2).transpose(1, 2))
            value = value.view(batch, height * width, self.nheads, self.head_dim)
            value = value.permute(0, 2, 3, 1).reshape(batch * self.nheads, self.head_dim, height, width)
            points = references[:, :, None, None, :] + offsets[:, :, :, level] / max(height, width)
            grid = points.permute(0, 2, 1, 3, 4).reshape(
                batch * self.nheads, length, self.points, 2
            )
            grid = grid.mul(2).sub(1)
            sampled = nn.functional.grid_sample(
                value, grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )
            sampled = sampled.view(batch, self.nheads, self.head_dim, length, self.points)
            sampled = sampled.permute(0, 3, 1, 4, 2)
            output = output + (sampled * weights[:, :, :, level, :, None]).sum(3)
        return self.output(output.flatten(2))


class DeformableDecoderLayer(nn.Module):
    def __init__(self, config: DETRConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.hidden_dim, config.nheads, batch_first=True)
        self.cross_attn = MultiScaleDeformableAttention(config.hidden_dim, config.nheads)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.dim_feedforward),
            nn.ReLU(),
            nn.Linear(config.dim_feedforward, config.hidden_dim),
        )
        self.norms = nn.ModuleList(nn.LayerNorm(config.hidden_dim) for _ in range(3))

    def forward(self, query: Tensor, references: Tensor, features: list[Tensor]) -> Tensor:
        attention, _ = self.self_attn(query, query, query)
        query = self.norms[0](query + attention)
        query = self.norms[1](query + self.cross_attn(query, references, features))
        return self.norms[2](query + self.ffn(query))


class DeformableDETRModel(nn.Module):
    def __init__(self, config: DETRConfig | None = None):
        super().__init__()
        self.config = config or DETRConfig()
        c = self.config
        self.backbone = TinyMultiScaleBackbone(c.hidden_dim, c.backbone_channels)
        self.positions = SinePositionEmbedding(c.hidden_dim)
        self.query_embed = nn.Embedding(c.num_queries, c.hidden_dim)
        self.reference_points = nn.Embedding(c.num_queries, 2)
        self.layers = nn.ModuleList(DeformableDecoderLayer(c) for _ in range(c.decoder_layers))
        self.class_embed = nn.Linear(c.hidden_dim, c.num_classes + 1)
        self.bbox_embed = MLP(c.hidden_dim, c.hidden_dim, 4)

    def forward(self, images: Tensor, **_: object) -> dict[str, Tensor]:
        features = self.backbone(images)
        features = [feature + self.positions(feature) for feature in features]
        query = self.query_embed.weight[None].expand(images.shape[0], -1, -1)
        references = self.reference_points.weight.sigmoid()[None].expand(images.shape[0], -1, -1)
        for layer in self.layers:
            query = layer(query, references, features)
        return {
            "pred_logits": self.class_embed(query),
            "pred_boxes": self.bbox_embed(query).sigmoid(),
        }


def build_model() -> DeformableDETRModel:
    return DeformableDETRModel()


__all__ = ["DeformableDETRModel", "MultiScaleDeformableAttention", "SetCriterion",
           "build_model", "decode_detections"]


if __name__ == "__main__":
    model = build_model().eval()
    output = model(torch.rand(1, 3, 64, 64))
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("logits:", tuple(output["pred_logits"].shape), "boxes:", tuple(output["pred_boxes"].shape))
