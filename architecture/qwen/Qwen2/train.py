"""Qwen2 教学模型训练入口。

训练数据以 [B, T] token ids 输入 GQA decoder，得到 [B, T, V] logits。
GQA 和 KV cache 由公共模型实现；训练阶段通常不传入 past cache，以便
对整段 teacher-forcing 序列并行计算 causal attention。
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
    """训练 Qwen2 教学模型，并将权重保存到指定 checkpoint。"""
    parser = argparse.ArgumentParser(description="Train a tiny Qwen2-style GQA causal LM.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--text", default="Qwen2 uses grouped-query attention and a reusable KV cache. ")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default="qwen2_tiny.pt")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = ByteTokenizer()
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    for step in range(args.steps):
        # inputs/labels shape: [B, T]；loss 使用 labels[:, 1:] 对齐下一 token。
        inputs, labels = build_training_batch(
            tokenizer, args.text, args.batch_size, args.seq_len, step, device
        )
        output = model(inputs, labels=labels)
        # output.logits shape: [B, T, V]；output.loss shape: []。
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
