"""Qwen1.5 教学模型 dense/MoE 自回归推理入口。

输入 prompt 编码为 [1, T_prompt]，``generate`` 首轮构建每层
[1, n_kv_heads, T_prompt, D] cache，之后每轮追加一个 token 的 KV。
``--moe`` 必须与训练 checkpoint 的网络结构保持一致。
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
    from architecture.qwen.common import ByteTokenizer
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.qwen.common import ByteTokenizer


def main() -> None:
    """按 dense/MoE 选项加载模型并生成文本。"""
    parser = argparse.ArgumentParser(description="Generate with a tiny Qwen1.5-style LM.")
    parser.add_argument("--moe", action="store_true")
    parser.add_argument("--prompt", default="Qwen1.5 is")
    parser.add_argument("--checkpoint")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = build_model(use_moe=args.moe).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state.get("model", state))
    tokenizer = ByteTokenizer()
    input_ids = torch.tensor([tokenizer.encode(args.prompt)], device=device)  # [1, T_prompt]
    output_ids = model.generate(input_ids, args.max_new_tokens, args.temperature, eos_token_id=2)
    # 输出包含原 prompt 和新生成 token，shape: [1, T_prompt + T_new]。
    print(tokenizer.decode(output_ids[0].tolist()))


if __name__ == "__main__":
    main()
