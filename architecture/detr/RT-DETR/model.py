"""RT-DETR 教学模型：混合多尺度编码器与 IoU-aware query selection。

任务定义:
    任务编号: RT-DETR；领域: 实时端到端闭集目标检测。输入为
    ``[B,3,H,W]``，主输出为 ``[B,Q,K+1]`` 和 ``[B,Q,4]``，同时保留
    encoder proposal ``[B,S,K+1]``、``[B,S,4]`` 与质量分数。

代表架构与来源:
    DETRs Beat YOLOs on Real-time Object Detection。多尺度 CNN 特征先经
    hybrid encoder 融合，再用 ``quality_i = max_c p_i(c) * IoU_i`` 进行
    top-k query selection，减少 decoder 处理的候选数量。

核心公式:
    ``quality_i = max_{c<K} softmax(cls_i)_c * sigmoid(iou_i)``
    ``Q = TopK_i(quality_i)``
    主分支仍使用 ``L_set = L_cls + 5 L_L1 + 2 L_GIoU``。
"""

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
    """对多尺度特征加入位置编码、局部融合和轻量 Transformer 编码。

    Inputs:
        features: 三层特征列表，每层 shape ``[B,D,H_l,W_l]``。
    Outputs:
        Tensor: 拼接后的 memory，shape ``[B,S,D]``，
        ``S = Σ_l H_l W_l``。
    """

    def __init__(self, config: DETRConfig):
        super().__init__()
        self.position = SinePositionEmbedding(config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            config.hidden_dim, config.nheads, config.dim_feedforward, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, 1)
        self.fuse = nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, padding=1, groups=8)

    def forward(self, features: list[Tensor]) -> Tensor:
        """展平各尺度并沿序列维拼接后编码。"""

        encoded = []
        for feature in features:
            fused = feature + self.position(feature) + self.fuse(feature)
            encoded.append(fused.flatten(2).transpose(1, 2))  # [B,D,H_l,W_l] -> [B,H_lW_l,D]
        return self.encoder(torch.cat(encoded, dim=1))  # [B,ΣH_lW_l,D]


class RTDETRModel(nn.Module):
    """带 encoder proposal 和 IoU-aware top-k query 的 RT-DETR。"""

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
        """完成多尺度编码、质量排序、decoder 和检测头预测。

        Inputs:
            images: shape ``[B,3,H,W]``。
        Outputs:
            pred_logits/pred_boxes: ``[B,Q,K+1]``、``[B,Q,4]``。
            pred_iou: ``[B,Q]``；enc_logits/enc_boxes: ``[B,S,K+1]``、
                ``[B,S,4]``。
        """

        memory_batch = self.encoder(self.backbone(images))
        encoder_logits = self.encoder_class(memory_batch)  # [B,S,K+1]
        encoder_boxes = self.encoder_box(memory_batch).sigmoid()  # [B,S,4]
        quality = (
            encoder_logits.softmax(-1)[..., :-1].amax(-1)
            * self.iou_head(memory_batch).sigmoid().squeeze(-1)
        )  # [B,S]，对应 quality_i
        topk = quality.topk(
            min(self.config.num_queries, quality.shape[1]), dim=1
        ).indices  # [B,Q']
        gather_c = topk[..., None].expand(-1, -1, self.config.hidden_dim)
        query = memory_batch.gather(1, gather_c)  # [B,Q',D]
        boxes = encoder_boxes.gather(
            1, topk[..., None].expand(-1, -1, 4)
        )  # [B,Q',4]
        if query.shape[1] < self.config.num_queries:
            pad = self.config.num_queries - query.shape[1]
            query = torch.cat((query, query[:, :1].expand(-1, pad, -1)), dim=1)
            boxes = torch.cat((boxes, boxes[:, :1].expand(-1, pad, -1)), dim=1)
        query_pos = self.query_position.weight[None].expand(
            images.shape[0], -1, -1
        )  # [B,Q,D]
        hidden = self.decoder(
            query.transpose(0, 1) + query_pos.transpose(0, 1),
            memory_batch.transpose(0, 1),
        ).transpose(0, 1)  # [Q,B,D] -> [B,Q,D]
        return {
            "pred_logits": self.class_embed(hidden),
            "pred_boxes": self.bbox_embed(hidden).sigmoid(),
            "pred_iou": self.iou_head(hidden).sigmoid().squeeze(-1),
            "enc_logits": encoder_logits,
            "enc_boxes": encoder_boxes,
        }


def build_model() -> RTDETRModel:
    """按默认配置构造 RT-DETR。"""

    return RTDETRModel()


__all__ = ["RTDETRModel", "HybridEncoder", "SetCriterion", "build_model", "decode_detections"]


if __name__ == "__main__":
    model = build_model().eval()
    output = model(torch.rand(1, 3, 64, 64))
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("logits:", tuple(output["pred_logits"].shape), "boxes:", tuple(output["pred_boxes"].shape))
