"""DAB-DETR 教学模型：动态 anchor-box query 与逐层框 refinement。

任务定义:
    任务编号: DAB-DETR；领域: 端到端闭集目标检测。输入为
    ``[B,3,H,W]``，输出分类 logits ``[B,Q,K+1]`` 和归一化框
    ``[B,Q,4]``。

代表架构与来源:
    DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR。每个 query
    同时包含内容向量和 ``cx,cy,w,h`` 动态 anchor，anchor 经过 box sine
    embedding 注入 decoder，并在每层被回归头迭代更新。

核心公式:
    ``q_pos = BoxSineEmbedding(b^l)``
    ``b^(l+1) = sigmoid(inv_sigmoid(b^l) + Δb^l)``
    ``L = L_cls + 5 L_L1 + 2 L_GIoU``。
    代码中的 ``boxes`` 是 ``b^l``，``box_head(query)`` 是 ``Δb^l``。
"""

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
    """使用动态框位置编码的 decoder 层。

    Inputs:
        query: query content，shape ``[B,Q,D]``。
        memory/memory_pos: encoder memory 与位置编码，shape ``[B,L,D]``。
        box_pos: 当前 anchor 的 box sine embedding，shape ``[B,Q,D]``。
    Outputs:
        Tensor: 更新后的 query，shape ``[B,Q,D]``。
    """

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
        """将 box positional query 注入 self/cross attention。"""

        query_pos = box_pos  # [B,Q,D]，对应公式 q_pos
        self_out, _ = self.self_attn(query + query_pos, query + query_pos, query)
        query = self.norms[0](query + self_out)
        cross_out, _ = self.cross_attn(
            query + query_pos, memory + memory_pos, memory
        )
        query = self.norms[1](query + cross_out)
        return self.norms[2](query + self.ffn(query))


class DABDETRModel(nn.Module):
    """带动态 anchor query 和 iterative box refinement 的 DAB-DETR。

    初始 ``query_anchor`` 为 ``[Q,4]``，扩展为 batch 后的 ``[B,Q,4]``；
    每个 decoder 层保持 query ``[B,Q,D]``，并更新同 shape 的 boxes。
    """

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
        """完成动态 anchor 注入、逐层 refinement 和检测头预测。"""

        memory, memory_pos = self.backbone.encode(images)
        memory = memory.transpose(0, 1)  # [L,B,D] -> [B,L,D]
        memory_pos = memory_pos.transpose(0, 1)  # [L,B,D] -> [B,L,D]
        query = torch.zeros(images.shape[0], self.config.num_queries,
                            self.config.hidden_dim, device=images.device)  # [B,Q,D]
        boxes = self.query_anchor.weight.sigmoid()[None].expand(
            images.shape[0], -1, -1
        )  # [B,Q,4]
        for layer, box_head in zip(self.layers, self.bbox_heads):
            box_pos = box_sine_embedding(boxes, self.config.hidden_dim)  # [B,Q,4] -> [B,Q,D]
            query = layer(query, memory, memory_pos, box_pos)
            boxes = (inverse_sigmoid(boxes) + box_head(query)).sigmoid()  # b^(l+1)
        return {"pred_logits": self.class_embed(query), "pred_boxes": boxes}


def build_model() -> DABDETRModel:
    """按默认配置构造 DAB-DETR。"""

    return DABDETRModel()


__all__ = ["DABDETRModel", "DABDecoderLayer", "SetCriterion", "build_model", "decode_detections"]


if __name__ == "__main__":
    model = build_model().eval()
    output = model(torch.rand(1, 3, 64, 64))
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("logits:", tuple(output["pred_logits"].shape), "boxes:", tuple(output["pred_boxes"].shape))
