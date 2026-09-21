"""Qwen3 教学模型训练入口。

训练流程与其他版本一致，但默认启用 top-k MoE。模型返回的 ``aux_loss``
用于监控路由均衡情况，并由公共 forward 按公式
``L_total = L_lm + 0.01 * L_aux`` 加入训练目标。
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
    """训练 Qwen3 教学模型并打印语言模型与 MoE 辅助损失。"""
    parser = argparse.ArgumentParser(description="Train a tiny Qwen3-style MoE causal LM.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--text", default="Qwen3 can reason in thinking mode and answer directly in non-thinking mode. ")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default="qwen3_tiny.pt")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = ByteTokenizer()
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    for step in range(args.steps):
        # inputs/labels shape: [B, T]；MoE 内部临时展平为 [B*T, H] 路由。
        inputs, labels = build_training_batch(
            tokenizer, args.text, args.batch_size, args.seq_len, step, device
        )
        output = model(inputs, labels=labels)
        # output.logits: [B, T, V]；loss/aux_loss: []。
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        aux = 0.0 if output.aux_loss is None else output.aux_loss.item()
        if step == 0 or (step + 1) % max(1, args.steps // 5) == 0:
            print(f"step {step + 1:03d}/{args.steps}: loss={output.loss.item():.4f}, aux={aux:.4f}")
    torch.save({"model": model.state_dict()}, args.checkpoint)
    print(f"saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
