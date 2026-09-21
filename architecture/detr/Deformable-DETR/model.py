"""Deformable-DETR 教学模型：多尺度稀疏采样与参考点注意力。

任务定义:
    任务编号: DEFORMABLE-DETR；领域: 多尺度端到端目标检测。输入图像为
    ``[B,3,H,W]``，输出 ``pred_logits=[B,Q,K+1]`` 与
    ``pred_boxes=[B,Q,4]``。

代表架构与来源:
    Deformable DETR: Deformable Transformers for End-to-End Object Detection。
    轻量 CNN 产生多个尺度 ``[B,D,H_l,W_l]`` 特征，decoder query 围绕参考点
    预测少量偏移，并通过 ``grid_sample`` 进行双线性采样。

核心公式:
    ``MSDeformAttn(q,p,x) = Σ_l Σ_k A_lk W x_l(phi_l(p)+Δp_lk)``
    代码中 ``offsets`` 对应 ``Δp``，``weights`` 对应归一化的 ``A``，
    ``grid_sample`` 对应 ``x_l`` 在采样位置的双线性插值。
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


class MultiScaleDeformableAttention(nn.Module):
    """从每个尺度的参考点邻域采样少量位置并聚合。

    Inputs:
        query: decoder query，shape ``[B,Q,D]``。
        references: 归一化参考点，shape ``[B,Q,2]``。
        features: 多尺度特征列表，第 ``l`` 项 shape 为 ``[B,D,H_l,W_l]``。
    Outputs:
        Tensor: 聚合后的 query，shape ``[B,Q,D]``。
    """

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
        """预测采样偏移/权重，并在各尺度上执行稀疏双线性采样。"""

        batch, length, _ = query.shape
        offsets = self.offsets(query).view(
            batch, length, self.nheads, self.levels, self.points, 2
        )  # [B,Q,Hd,L,K,2]
        weights = self.weights(query).view(
            batch, length, self.nheads, self.levels, self.points
        )  # [B,Q,Hd,L,K]
        weights = weights.flatten(2).softmax(-1).view_as(weights)  # Σ_(Hd,L,K) A = 1
        output = query.new_zeros(batch, length, self.nheads, self.head_dim)  # [B,Q,Hd,Dh]
        for level, feature in enumerate(features):
            height, width = feature.shape[-2:]
            value = self.value(feature.flatten(2).transpose(1, 2))  # [B,D,H_lW_l] -> [B,H_lW_l,D]
            value = value.view(batch, height * width, self.nheads, self.head_dim)
            value = value.permute(0, 2, 3, 1).reshape(
                batch * self.nheads, self.head_dim, height, width
            )  # [B,Hd,Dh,H_lW_l] -> [B*Hd,Dh,H_l,W_l]
            points = references[:, :, None, None, :] + offsets[:, :, :, level] / max(height, width)
            # references [B,Q,1,1,2] + offsets [B,Q,Hd,K,2] -> [B,Q,Hd,K,2]。
            grid = points.permute(0, 2, 1, 3, 4).reshape(
                batch * self.nheads, length, self.points, 2
            )
            grid = grid.mul(2).sub(1)  # [0,1] -> grid_sample 需要的 [-1,1]
            sampled = nn.functional.grid_sample(
                value, grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )
            sampled = sampled.view(batch, self.nheads, self.head_dim, length, self.points)
            sampled = sampled.permute(0, 3, 1, 4, 2)  # [B,Q,Hd,K,Dh]
            output = output + (sampled * weights[:, :, :, level, :, None]).sum(3)  # [B,Q,Hd,Dh]
        return self.output(output.flatten(2))  # [B,Q,Hd,Dh] -> [B,Q,D]


class DeformableDecoderLayer(nn.Module):
    """包含 self-attention、可变形 cross-attention 和 FFN 的 decoder 层。

    输入 query 为 ``[B,Q,D]``，每层保持相同 Shape；多尺度 feature 列表
    只在 cross-attention 内被读取。
    """

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
        """执行一层带残差与 LayerNorm 的可变形 decoder 更新。"""

        attention, _ = self.self_attn(query, query, query)
        query = self.norms[0](query + attention)
        query = self.norms[1](query + self.cross_attn(query, references, features))
        return self.norms[2](query + self.ffn(query))


class DeformableDETRModel(nn.Module):
    """多尺度 Deformable-DETR 顶层网络。

    视觉特征 Shape 为 ``[B,D,H/8,W/8]``、``[B,D,H/16,W/16]``、
    ``[B,D,H/32,W/32]``；固定 reference points 为 ``[B,Q,2]``，最终
    检测输出为 ``[B,Q,K+1]`` 和 ``[B,Q,4]``。
    """

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
        """完成多尺度特征编码、稀疏解码与检测头预测。"""

        features = self.backbone(images)
        features = [
            feature + self.positions(feature) for feature in features
        ]  # 每项 [B,D,H_l,W_l]
        query = self.query_embed.weight[None].expand(images.shape[0], -1, -1)  # [B,Q,D]
        references = self.reference_points.weight.sigmoid()[None].expand(
            images.shape[0], -1, -1
        )  # [B,Q,2]
        for layer in self.layers:
            query = layer(query, references, features)
        return {
            "pred_logits": self.class_embed(query),
            "pred_boxes": self.bbox_embed(query).sigmoid(),
        }


def build_model() -> DeformableDETRModel:
    """按默认配置构造 Deformable-DETR。"""

    return DeformableDETRModel()


__all__ = ["DeformableDETRModel", "MultiScaleDeformableAttention", "SetCriterion",
           "build_model", "decode_detections"]


if __name__ == "__main__":
    model = build_model().eval()
    output = model(torch.rand(1, 3, 64, 64))
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("logits:", tuple(output["pred_logits"].shape), "boxes:", tuple(output["pred_boxes"].shape))
