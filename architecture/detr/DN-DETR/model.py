"""DN-DETR 教学模型：训练期 noisy label/box query 去噪。

任务定义:
    任务编号: DN-DETR；领域: 端到端闭集目标检测。正常输入为图像
    ``[B,3,H,W]``，检测输出为 ``[B,Q,K+1]`` 与 ``[B,Q,4]``；训练时
    额外拼接 denoising query，数量为 ``G*M``，其中 ``M`` 是 batch 内最大
    目标数，``G`` 是 denoising group 数。

代表架构与来源:
    DN-DETR: Accelerate DETR Training by Introducing Query DeNoising。
    将真实标签/框复制后加入类别替换与框扰动，作为 decoder 的训练提示；
    推理时 ``targets=None``，仅保留正常 query 分支。

核心公式:
    ``y_tilde = corrupt(y)``
    ``L = L_set(normal) + L_DN``
    ``L_DN = CE(cls_dn, cls) + 5 L1(box_dn, box)``。
    代码中的 ``noisy_labels/noisy_boxes`` 是扰动输入，``dn_labels`` 和
    ``dn_target_boxes`` 是去噪监督目标。
"""

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
    """构造训练期去噪 query 的内容、位置和监督标签。

    Inputs:
        targets: 长度 ``B`` 的目标列表，labels ``[M_i]``、boxes ``[M_i,4]``。
        num_classes: 类别数 ``K``；``K`` 同时作为 no-object sentinel。
        groups: 复制组数 ``G``。
        hidden_dim: query 内容维度 ``D``。
        label_embedding: ``K+1`` 到 ``D`` 的 embedding。
        device: 输出设备。
    Outputs:
        content: noisy label embedding，shape ``[B,G*M,D]``。
        position: noisy box embedding，shape ``[B,G*M,D]``。
        clean_labels: 监督标签，shape ``[B,G*M]``。
        clean_boxes: 监督框，shape ``[B,G*M,4]``。
    """

    max_targets = max((target["labels"].numel() for target in targets), default=0)
    count = max_targets * groups  # denoising query 总数 G*M
    if count == 0:
        empty = torch.zeros(len(targets), 0, hidden_dim, device=device)
        return empty, empty, torch.zeros(len(targets), 0, dtype=torch.long, device=device), empty[..., :4]
    noisy_labels = torch.full(
        (len(targets), count), num_classes, dtype=torch.long, device=device
    )  # [B,G*M]
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
    content = label_embedding(noisy_labels)  # [B,G*M] -> [B,G*M,D]
    position = box_sine_embedding(noisy_boxes, hidden_dim)  # [B,G*M,4] -> [B,G*M,D]
    valid = clean_labels != num_classes
    return content, position, clean_labels, clean_boxes.masked_fill(~valid[..., None], 0)


class DNDETRModel(nn.Module):
    """带训练期 denoising 分支的 DETR。

    正常输出为 ``pred_logits=[B,Q,K+1]``、``pred_boxes=[B,Q,4]``；
    提供 targets 时还返回 ``dn_logits=[B,G*M,K+1]``、
    ``dn_boxes=[B,G*M,4]`` 及对应监督张量。
    """

    def __init__(self, config: DETRConfig | None = None, denoising_groups: int = 2):
        super().__init__()
        self.config = config or DETRConfig()
        self.denoising_groups = denoising_groups
        self.core = TinyDETR(self.config)
        self.label_embedding = nn.Embedding(self.config.num_classes + 1, self.config.hidden_dim)
        self.class_embed = nn.Linear(self.config.hidden_dim, self.config.num_classes + 1)
        self.bbox_embed = MLP(self.config.hidden_dim, self.config.hidden_dim, 4)

    def forward(self, images: Tensor, targets: list[dict[str, Tensor]] | None = None) -> dict[str, Tensor]:
        """根据是否提供 targets 选择推理分支或去噪训练分支。

        Inputs:
            images: shape ``[B,3,H,W]``。
            targets: 可选目标列表；训练时 labels/boxes 分别为 ``[M_i]``、
                ``[M_i,4]``。
        Outputs:
            dict: 正常分支 shape 见类说明；训练分支额外包含 denoising 输出。
        """

        memory, position = self.core.encode(images)
        batch = images.shape[0]
        query_position = self.core.query_embed.weight[None].expand(batch, -1, -1)  # [B,Q,D]
        query_content = torch.zeros_like(query_position)  # [B,Q,D]
        if targets is None:
            hidden = self.core.decode_queries(
                memory, position, query_content.transpose(0, 1), query_position.transpose(0, 1)
            )
            return {
                "pred_logits": self.class_embed(hidden),  # [B,Q,K+1]
                "pred_boxes": self.bbox_embed(hidden).sigmoid(),  # [B,Q,4]
            }
        content, dn_position, dn_labels, dn_boxes = make_denoising_queries(
            targets, self.config.num_classes, self.denoising_groups, self.config.hidden_dim,
            self.label_embedding, images.device,
        )
        normal_content = query_content
        all_content = torch.cat((normal_content, content), dim=1).transpose(0, 1)
        all_position = torch.cat((query_position, dn_position), dim=1).transpose(0, 1)
        # [B,Q+G*M,D] -> Transformer 需要的 [Q+G*M,B,D]。
        hidden = self.core.decode_queries(memory, position, all_content, all_position)
        normal_hidden = hidden[:, :self.config.num_queries]  # [B,Q,D]
        dn_hidden = hidden[:, self.config.num_queries:]  # [B,G*M,D]
        return {
            "pred_logits": self.class_embed(normal_hidden),
            "pred_boxes": self.bbox_embed(normal_hidden).sigmoid(),
            "dn_logits": self.class_embed(dn_hidden),
            "dn_boxes": self.bbox_embed(dn_hidden).sigmoid(),
            "dn_labels": dn_labels,
            "dn_target_boxes": dn_boxes,
        }


class DenoisingCriterion(SetCriterion):
    """集合损失加 denoising 分类/L1 损失。

    正常 query 使用 Hungarian matching；denoising query 已按目标槽位对齐，
    因此直接在 ``[B,G*M]`` 位置上计算监督。
    """

    def forward(self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]]) -> Tensor:
        """返回 ``L_set + L_DN`` 标量。"""

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
    """按默认配置构造 DN-DETR。"""

    return DNDETRModel()


__all__ = ["DNDETRModel", "DenoisingCriterion", "make_denoising_queries",
           "build_model", "decode_detections"]


if __name__ == "__main__":
    from architecture.detr.common import synthetic_detection_batch

    model = build_model()
    images, targets = synthetic_detection_batch(2, 64, 3, "cpu")
    outputs = model(images, targets=targets)
    print("normal:", tuple(outputs["pred_logits"].shape), "denoising:", tuple(outputs["dn_logits"].shape))
