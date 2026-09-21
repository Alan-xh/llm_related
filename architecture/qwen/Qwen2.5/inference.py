"""Qwen2.5 教学模型长上下文自回归推理入口。

prompt 的 token ids shape 为 [1, T_prompt]。模型内部使用 scaled RoPE
处理 [1, heads, T, D] 的 Q/K，并在生成阶段复用逐层 KV cache。
这里的长上下文能力是教学接口，不代表官方推理上限。
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
    """加载 Qwen2.5 教学模型并生成文本。"""
    parser = argparse.ArgumentParser(description="Generate with a tiny Qwen2.5-style LM.")
    parser.add_argument("--prompt", default="Qwen2.5 is")
    parser.add_argument("--checkpoint")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = build_model().to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state.get("model", state))
    tokenizer = ByteTokenizer()
    input_ids = torch.tensor([tokenizer.encode(args.prompt)], device=device)  # [1, T_prompt]
    output_ids = model.generate(input_ids, args.max_new_tokens, args.temperature, eos_token_id=2)
    # 生成序列 shape: [1, T_prompt + T_new]。
    print(tokenizer.decode(output_ids[0].tolist()))


if __name__ == "__main__":
    main()
