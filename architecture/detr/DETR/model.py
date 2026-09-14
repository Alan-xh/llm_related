"""Compact original DETR: object queries, Hungarian matching and set loss."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from ..common import DETRConfig, SetCriterion, TinyDETR, decode_detections
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.detr.common import DETRConfig, SetCriterion, TinyDETR, decode_detections


class DETRModel(TinyDETR):
    """The small runnable model used in the original DETR example."""


def build_model() -> DETRModel:
    return DETRModel(DETRConfig())


__all__ = ["DETRConfig", "DETRModel", "SetCriterion", "build_model", "decode_detections"]


if __name__ == "__main__":
    import torch

    model = build_model().eval()
    output = model(torch.rand(1, 3, 64, 64))
    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print("logits:", tuple(output["pred_logits"].shape), "boxes:", tuple(output["pred_boxes"].shape))
