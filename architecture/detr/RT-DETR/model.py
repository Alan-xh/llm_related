"""Compact RT-DETR with a hybrid multi-scale encoder and IoU-aware queries."""

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


class HybridEncoder(nn.Module):
    def __init__(self, config: DETRConfig):
        super().__init__()
        self.position = SinePositionEmbedding(config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            config.hidden_dim, config.nheads, config.dim_feedforward, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, 1)
        self.fuse = nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, padding=1, groups=8)

    def forward(self, features: list[Tensor]) -> Tensor:
        encoded = []
        for feature in features:
            encoded.append((feature + self.position(feature) + self.fuse(feature)).flatten(2).transpose(1, 2))
        return self.encoder(torch.cat(encoded, dim=1))


class RTDETRModel(nn.Module):
    def __init__(self, config: DETRConfig | None = None):
        super().__init__()
        self.config = config or DETRConfig()
        c = self.config
        self.backbone = TinyMultiScaleBackbone(c.hidden_dim, c.backbone_channels)
        self.encoder = HybridEncoder(c)
        self.encoder_class = nn.Linear(c.hidden_dim, c.num_classes + 1)
        self.encoder_box = MLP(c.hidden_dim, c.hidden_dim, 4)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(c.hidden_dim, c.nheads, c.dim_feedforward, batch_first=False),
            c.decoder_layers,
        )
        self.query_position = nn.Embedding(c.num_queries, c.hidden_dim)
        self.class_embed = nn.Linear(c.hidden_dim, c.num_classes + 1)
        self.bbox_embed = MLP(c.hidden_dim, c.hidden_dim, 4)
        self.iou_head = MLP(c.hidden_dim, c.hidden_dim, 1)

    def forward(self, images: Tensor, **_: object) -> dict[str, Tensor]:
        memory_batch = self.encoder(self.backbone(images))
        encoder_logits = self.encoder_class(memory_batch)
        encoder_boxes = self.encoder_box(memory_batch).sigmoid()
        quality = encoder_logits.softmax(-1)[..., :-1].amax(-1) * self.iou_head(memory_batch).sigmoid().squeeze(-1)
        topk = quality.topk(min(self.config.num_queries, quality.shape[1]), dim=1).indices
        gather_c = topk[..., None].expand(-1, -1, self.config.hidden_dim)
        query = memory_batch.gather(1, gather_c)
        boxes = encoder_boxes.gather(1, topk[..., None].expand(-1, -1, 4))
        if query.shape[1] < self.config.num_queries:
            pad = self.config.num_queries - query.shape[1]
            query = torch.cat((query, query[:, :1].expand(-1, pad, -1)), dim=1)
            boxes = torch.cat((boxes, boxes[:, :1].expand(-1, pad, -1)), dim=1)
        query_pos = self.query_position.weight[None].expand(images.shape[0], -1, -1)
        hidden = self.decoder(
            query.transpose(0, 1) + query_pos.transpose(0, 1),
            memory_batch.transpose(0, 1),
        ).transpose(0, 1)
        return {
            "pred_logits": self.class_embed(hidden),
            "pred_boxes": self.bbox_embed(hidden).sigmoid(),
            "pred_iou": self.iou_head(hidden).sigmoid().squeeze(-1),
            "enc_logits": encoder_logits,
            "enc_boxes": encoder_boxes,
        }


def build_model() -> RTDETRModel:
    return RTDETRModel()


__all__ = ["RTDETRModel", "HybridEncoder", "SetCriterion", "build_model", "decode_detections"]


if __name__ == "__main__":
    model = build_model().eval()
    output = model(torch.rand(1, 3, 64, 64))
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("logits:", tuple(output["pred_logits"].shape), "boxes:", tuple(output["pred_boxes"].shape))
