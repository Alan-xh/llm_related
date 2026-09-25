"""Kimi Linear 教学模型的 CPU smoke-training 入口。

任务定义:
    使用重复文本构造因果语言建模 batch，训练 KDA/MLA 混合模型并保存
    教学 checkpoint。

张量约定:
    inputs/labels: [B, T]；模型 logits: [B, T, V]；loss: 标量。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    from .model import build_model
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import build_model

try:
    from architecture.kimi.common import ByteTokenizer, build_training_batch
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.kimi.common import ByteTokenizer, build_training_batch


def main() -> None:
    """解析命令行参数并运行 Kimi Linear 的最小训练循环。"""
    parser = argparse.ArgumentParser(description="训练 Kimi Linear 风格的最小 KDA/MLA 混合模型。")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--text",
        default="Kimi Linear combines KDA finite-state memory with periodic global MLA attention. ",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default="kimi_linear_tiny.pt")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = ByteTokenizer()
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    for step in range(args.steps):
        inputs, labels = build_training_batch(
            tokenizer, args.text, args.batch_size, args.seq_len, step, device
        )
        output = model(inputs, labels=labels)
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        aux = 0.0 if output.aux_loss is None else output.aux_loss.item()
        print(f"步骤 {step + 1:03d}/{args.steps}: loss={output.loss.item():.4f}, aux={aux:.4f}")
    torch.save({"model": model.state_dict()}, args.checkpoint)
    print(f"checkpoint 已保存到 {args.checkpoint}")


if __name__ == "__main__":
    main()
