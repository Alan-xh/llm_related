"""Deformable-DETR 推理入口。

输入图像为 ``[1,3,H,W]``；模型内部在多个尺度上围绕 reference point
采样，输出 ``pred_logits=[1,Q,K+1]`` 和 ``pred_boxes=[1,Q,4]``。
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
    """执行无梯度多尺度检测并打印解码结果 Shape。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--checkpoint")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = build_model().to(device).eval()
    load_checkpoint(model, args.checkpoint, device)
    # 推理只保留正常 query，不生成训练期辅助监督。
    with torch.no_grad():
        outputs = model(torch.rand(1, 3, args.image_size, args.image_size, device=device))
    print("image=0 detections:", tuple(
        decode_detections(outputs, (args.image_size, args.image_size), args.confidence)[0].shape
    ))


if __name__ == "__main__":
    main()
