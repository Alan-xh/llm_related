"""Qwen1 教学模型训练入口。

训练流程:
    文本 -> ByteTokenizer -> input/label [B, T] -> Qwen1 logits [B, T, V]
    -> shifted cross entropy 标量 loss -> AdamW 更新。模型本身只接收
    tensor，不依赖命令行状态；该文件负责数据构造、优化器和 checkpoint。
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
    """解析训练参数并执行固定步数的 CPU/GPU causal LM 训练。"""
    parser = argparse.ArgumentParser(description="Train a tiny Qwen1-style causal LM.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--text", default="Qwen teaches decoder-only Transformer architecture. ")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default="qwen1_tiny.pt")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = ByteTokenizer()
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    for step in range(args.steps):
        # inputs/labels: [B, T]；QwenCausalLM 内部用 logits[:, :-1]
        # 预测 labels[:, 1:]，对应标准 teacher forcing 目标。
        inputs, labels = build_training_batch(
            tokenizer, args.text, args.batch_size, args.seq_len, step, device
        )
        output = model(inputs, labels=labels)
        # output.logits: [B, T, V]；output.loss: []。
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % max(1, args.steps // 5) == 0:
            print(f"step {step + 1:03d}/{args.steps}: loss={output.loss.item():.4f}")
    # 仅保存权重字典，推理入口会按同一默认配置重建网络。
    torch.save({"model": model.state_dict()}, args.checkpoint)
    print(f"saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
