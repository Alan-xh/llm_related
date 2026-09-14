"""Compact DAB-DETR with dynamic anchor-box queries."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor, nn

try:
    from ..common import (
        DETRConfig, MLP, SetCriterion, TinyDETR,
        box_sine_embedding, decode_detections, inverse_sigmoid,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.detr.common import (
        DETRConfig, MLP, SetCriterion, TinyDETR,
        box_sine_embedding, decode_detections, inverse_sigmoid,
    )


class DABDecoderLayer(nn.Module):
    def __init__(self, config: DETRConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.hidden_dim, config.nheads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(config.hidden_dim, config.nheads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.dim_feedforward), nn.ReLU(),
            nn.Linear(config.dim_feedforward, config.hidden_dim),
        )
        self.norms = nn.ModuleList(nn.LayerNorm(config.hidden_dim) for _ in range(3))

    def forward(self, query: Tensor, memory: Tensor, memory_pos: Tensor, box_pos: Tensor) -> Tensor:
        query_pos = box_pos
        self_out, _ = self.self_attn(query + query_pos, query + query_pos, query)
        query = self.norms[0](query + self_out)
        cross_out, _ = self.cross_attn(
            query + query_pos, memory + memory_pos, memory
        )
        query = self.norms[1](query + cross_out)
        return self.norms[2](query + self.ffn(query))


class DABDETRModel(nn.Module):
    def __init__(self, config: DETRConfig | None = None):
        super().__init__()
        self.config = config or DETRConfig()
        c = self.config
        self.backbone = TinyDETR(c)
        self.query_anchor = nn.Embedding(c.num_queries, 4)
        self.layers = nn.ModuleList(DABDecoderLayer(c) for _ in range(c.decoder_layers))
        self.class_embed = nn.Linear(c.hidden_dim, c.num_classes + 1)
        self.bbox_heads = nn.ModuleList(MLP(c.hidden_dim, c.hidden_dim, 4) for _ in self.layers)

    def forward(self, images: Tensor, **_: object) -> dict[str, Tensor]:
        memory, memory_pos = self.backbone.encode(images)
        memory = memory.transpose(0, 1)
        memory_pos = memory_pos.transpose(0, 1)
        query = torch.zeros(images.shape[0], self.config.num_queries,
                            self.config.hidden_dim, device=images.device)
        boxes = self.query_anchor.weight.sigmoid()[None].expand(images.shape[0], -1, -1)
        for layer, box_head in zip(self.layers, self.bbox_heads):
            query = layer(query, memory, memory_pos, box_sine_embedding(boxes, self.config.hidden_dim))
            boxes = (inverse_sigmoid(boxes) + box_head(query)).sigmoid()
        return {"pred_logits": self.class_embed(query), "pred_boxes": boxes}


def build_model() -> DABDETRModel:
    return DABDETRModel()


__all__ = ["DABDETRModel", "DABDecoderLayer", "SetCriterion", "build_model", "decode_detections"]


if __name__ == "__main__":
    model = build_model().eval()
    output = model(torch.rand(1, 3, 64, 64))
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("logits:", tuple(output["pred_logits"].shape), "boxes:", tuple(output["pred_boxes"].shape))
