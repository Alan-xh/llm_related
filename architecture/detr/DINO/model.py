"""DINO 风格 DETR 教学模型：two-stage query selection 与 contrastive DN。

任务定义:
    任务编号: DINO；领域: 端到端闭集目标检测。输入图像 shape 为
    ``[B,3,H,W]``，正常输出为 ``pred_logits=[B,Q,K+1]``、
    ``pred_boxes=[B,Q,4]``；训练期附加 ``G*M`` 个 denoising query。

代表架构与来源:
    DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object
    Detection。先对 encoder memory 预测类别/框，再按 foreground score
    选出 top-k query，并将其框编码为 decoder query position；训练时加入
    带错误类别和框扰动的对比式 denoising query。

核心公式:
    ``score_i = max_{c<K} softmax(cls_i)_c``
    ``Q = TopK_i(score_i)``
    ``L = L_set + CE(cls_dn, cls) + 5 L1(box_dn, box)``。
    代码中的 ``encoder_scores``、``selected``、``_denoising`` 分别对应
    query selection 和 contrastive denoising。
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


class DINOModel(nn.Module):
    """包含 encoder proposal 选 query 和 denoising 分支的 DINO 模型。

    encoder memory 为 ``[B,L,D]``；选出的正常 query 为 ``[B,Q,D]``、
    参考框为 ``[B,Q,4]``；训练时拼接后 decoder 输入为
    ``[B,Q+G*M,D]``。
    """

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
        """复制目标并施加类别替换、框噪声，生成对比式 DN query。

        Returns:
            content: ``[B,G*M,D]``。
            position: ``[B,G*M,D]``。
            labels: clean labels，``[B,G*M]``。
            boxes: clean boxes，``[B,G*M,4]``。
        """

        max_targets = max((target["labels"].numel() for target in targets), default=0)
        count = max_targets * self.denoising_groups  # G*M
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
        content = self.label_embedding(noisy)  # [B,G*M] -> [B,G*M,D]
        position = box_sine_embedding(noisy_boxes, self.config.hidden_dim)  # [B,G*M,4] -> [B,G*M,D]
        return content, position, labels, boxes

    def forward(self, images: Tensor, targets: list[dict[str, Tensor]] | None = None) -> dict[str, Tensor]:
        """执行 two-stage query selection、decoder 和可选 denoising。

        Inputs:
            images: shape ``[B,3,H,W]``。
            targets: 可选长度 ``B`` 的目标列表，训练时启用 DN 分支。
        Outputs:
            pred_logits/pred_boxes: ``[B,Q,K+1]``、``[B,Q,4]``。
            enc_logits/enc_boxes: encoder proposal，``[B,L,K+1]``、``[B,L,4]``。
            targets 非空时额外返回 ``[B,G*M,K+1]``、``[B,G*M,4]``。
        """

        memory, memory_pos = self.core.encode(images)
        memory_batch = memory.transpose(0, 1)  # [L,B,D] -> [B,L,D]
        encoder_scores = self.encoder_class(memory_batch).softmax(-1)[..., :-1].amax(-1)  # [B,L]
        encoder_boxes = self.encoder_box(memory_batch).sigmoid()  # [B,L,4]
        query_count = self.config.num_queries
        selected = encoder_scores.topk(
            min(query_count, memory_batch.shape[1]), dim=1
        ).indices  # [B,Q']
        gather_index = selected[..., None].expand(-1, -1, self.config.hidden_dim)
        query_content = memory_batch.gather(1, gather_index)  # [B,Q',D]
        query_boxes = encoder_boxes.gather(
            1, selected[..., None].expand(-1, -1, 4)
        )  # [B,Q',4]
        if query_content.shape[1] < query_count:
            pad = query_count - query_content.shape[1]
            query_content = torch.cat((query_content, query_content[:, :1].expand(-1, pad, -1)), dim=1)
            query_boxes = torch.cat((query_boxes, query_boxes[:, :1].expand(-1, pad, -1)), dim=1)
        query_position = box_sine_embedding(
            query_boxes.detach(), self.config.hidden_dim
        )  # [B,Q,4] -> [B,Q,D]
        dn_labels = dn_boxes = None
        if targets is not None:
            dn_content, dn_position, dn_labels, dn_boxes = self._denoising(targets, images.device)
            query_content = torch.cat((query_content, dn_content), dim=1)
            query_position = torch.cat((query_position, dn_position), dim=1)
        hidden = self.core.decode_queries(
            memory, memory_pos, query_content.transpose(0, 1), query_position.transpose(0, 1)
        )  # [B,Q(+G*M),D]
        normal_hidden = hidden[:, :query_count]  # [B,Q,D]
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
    """DINO 的集合损失与 contrastive denoising 损失。"""

    def forward(self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]]) -> Tensor:
        """返回正常集合损失与 DN 分类/L1 损失之和。"""

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
    """按默认配置构造 DINO 教学模型。"""

    return DINOModel()


__all__ = ["DINOModel", "DINOSetCriterion", "build_model", "decode_detections"]


if __name__ == "__main__":
    from architecture.detr.common import synthetic_detection_batch

    model = build_model()
    images, targets = synthetic_detection_batch(2, 64, 3, "cpu")
    output = model(images, targets=targets)
    print("normal:", tuple(output["pred_logits"].shape), "denoising:", tuple(output["dn_logits"].shape))
