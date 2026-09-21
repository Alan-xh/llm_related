"""Qwen2.5 教学模型训练入口。

本文件训练带 scaled RoPE 的 GQA decoder。``seq_len`` 只控制教学 batch
的当前长度；模型的 RoPE cache 会按实际位置自动扩展到所需长度。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    from .model import build_model
except ImportError:
    from model import build_model

try:
    from architecture.qwen.common import ByteTokenizer, build_training_batch
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.qwen.common import ByteTokenizer, build_training_batch


def main() -> None:
    """训练 Qwen2.5 教学模型并保存参数。"""
    parser = argparse.ArgumentParser(description="Train a tiny Qwen2.5-style long-context LM.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--text", default="Qwen2.5 improves long-context, code, math, and structured output. ")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default="qwen2_5_tiny.pt")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = ByteTokenizer()
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    for step in range(args.steps):
        # inputs/labels: [B, T]；scaled RoPE 在注意力内部作用于 [B, heads, T, D]。
        inputs, labels = build_training_batch(
            tokenizer, args.text, args.batch_size, args.seq_len, step, device
        )
        output = model(inputs, labels=labels)
        # output.loss 为 shifted cross entropy 标量。
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % max(1, args.steps // 5) == 0:
            print(f"step {step + 1:03d}/{args.steps}: loss={output.loss.item():.4f}")
    torch.save({"model": model.state_dict()}, args.checkpoint)
    print(f"saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
