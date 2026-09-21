"""原始 DETR 教学模型：object query、Hungarian matching 与集合损失。

任务定义:
    任务编号: DETR；领域: 端到端闭集目标检测。输入图像 shape 为
    ``[B, 3, H, W]``，输出分类 logits ``[B, Q, K+1]`` 与归一化
    ``cxcywh`` 框 ``[B, Q, 4]``。

代表架构与来源:
    End-to-End Object Detection with Transformers（Carion et al.）。
    本文件通过 ``TinyDETR`` 复用单尺度 CNN、Transformer encoder-decoder
    和固定 object queries，保留原始 DETR 的集合预测路径。

核心机制与公式:
    ``sigma* = argmin_sigma Σ_i C(y_i, y_hat_sigma(i))``
    ``C = -p_hat(c_i) + λ_L1 ||b_i-b_hat||_1 - λ_GIoU GIoU``
    匹配后使用 ``L = L_cls + 5 L_L1 + 2 L_GIoU``，未匹配 query 的标签为
    no-object；推理阶段不需要 anchor 或 NMS。
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from ..common import DETRConfig, SetCriterion, TinyDETR, decode_detections
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.detr.common import DETRConfig, SetCriterion, TinyDETR, decode_detections


class DETRModel(TinyDETR):
    """原始 DETR 的可运行轻量封装。

    继承 ``TinyDETR`` 的前向接口：输入 ``[B,3,H,W]``，输出
    ``pred_logits=[B,Q,K+1]`` 和 ``pred_boxes=[B,Q,4]``。
    """


def build_model() -> DETRModel:
    """按默认 ``DETRConfig`` 构造原始 DETR 教学模型。"""

    return DETRModel(DETRConfig())


__all__ = ["DETRConfig", "DETRModel", "SetCriterion", "build_model", "decode_detections"]


if __name__ == "__main__":
    import torch

    model = build_model().eval()
    output = model(torch.rand(1, 3, 64, 64))
    # 示例输入 [1,3,64,64] -> logits [1,Q,K+1]、boxes [1,Q,4]。
    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print("logits:", tuple(output["pred_logits"].shape), "boxes:", tuple(output["pred_boxes"].shape))
