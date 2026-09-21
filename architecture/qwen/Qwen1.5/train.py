"""Qwen1.5 教学模型训练入口。

该入口与 Qwen1 共用 causal LM 训练流程；``--moe`` 决定前馈分支是
dense SwiGLU 还是 top-k MoE。MoE 模型返回的辅助负载均衡损失已在
``QwenCausalLM.forward`` 中按 0.01 系数并入总 loss。
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
    """解析参数，训练 dense 或 MoE Qwen1.5 教学模型并保存权重。"""
    parser = argparse.ArgumentParser(description="Train a tiny Qwen1.5-style causal LM.")
    parser.add_argument("--moe", action="store_true", help="use the optional MoE feed-forward blocks")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--text", default="Qwen1.5 unifies dense and mixture-of-experts language models. ")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default="qwen1_5_tiny.pt")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = ByteTokenizer()
    model = build_model(use_moe=args.moe).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    for step in range(args.steps):
        # 两个 batch tensor 均为 [B, T]；模型输出 logits 为 [B, T, V]。
        inputs, labels = build_training_batch(
            tokenizer, args.text, args.batch_size, args.seq_len, step, device
        )
        output = model(inputs, labels=labels)
        # MoE 时 output.aux_loss 是标量，已参与 output.loss 的反向传播。
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % max(1, args.steps // 5) == 0:
            print(f"step {step + 1:03d}/{args.steps}: loss={output.loss.item():.4f}")
    # 将 use_moe 一并写入 checkpoint，便于调用方复现结构选择。
    torch.save({"model": model.state_dict(), "use_moe": args.moe}, args.checkpoint)
    print(f"saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
