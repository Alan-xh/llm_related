"""DN-DETR 推理入口。

推理不传入 targets，因此只执行正常 query ``[B,Q,D]``，输出
``pred_logits=[B,Q,K+1]``、``pred_boxes=[B,Q,4]``；denoising query
仅存在于训练路径。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    from ..common import decode_detections, load_checkpoint
    from .model import build_model
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.detr.common import decode_detections, load_checkpoint
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import build_model


def main() -> None:
    """加载 checkpoint 并执行不带 denoising query 的推理。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--checkpoint")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = build_model().to(device).eval()
    load_checkpoint(model, args.checkpoint, device)
    # 不传 targets，明确走 DNDETRModel 的 inference 分支。
    with torch.no_grad():
        outputs = model(torch.rand(1, 3, args.image_size, args.image_size, device=device))
    detections = decode_detections(outputs, (args.image_size, args.image_size), args.confidence)
    print("image=0 detections:", tuple(detections[0].shape))


if __name__ == "__main__":
    main()
