"""Grounding DINO 文本条件推理入口。

输入为图像 ``[1,3,H,W]`` 和 token ids ``[1,T]``；模型返回
``pred_logits=[1,Q,T]``、``pred_boxes=[1,Q,4]``、``pred_objectness=[1,Q]``，
再按最高 token 分数解码为 ``[N,6]``。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    from ..common import load_checkpoint
    from .model import build_model, decode_grounding
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.detr.common import load_checkpoint
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import build_model, decode_grounding


def main() -> None:
    """加载模型并执行 token-level grounding 解码。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--checkpoint")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = build_model().to(device).eval()
    load_checkpoint(model, args.checkpoint, device)
    text_tokens = torch.tensor([[1, 2, 3, 4, 5]], device=device)  # [1,T]
    with torch.no_grad():
        outputs = model(torch.rand(1, 3, args.image_size, args.image_size, device=device), text_tokens)
    detections = decode_grounding(outputs, text_tokens, (args.image_size, args.image_size), args.confidence)
    print("image=0 grounding detections:", tuple(detections[0].shape))


if __name__ == "__main__":
    main()
