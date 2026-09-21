"""Qwen1 教学模型自回归推理入口。

推理流程:
    prompt -> ByteTokenizer -> input_ids [1, T_prompt] -> generate()
    -> output_ids [1, T_prompt + T_new] -> UTF-8 文本。首次生成会计算
    整个 prompt，后续步骤通过逐层 past_key_values 只输入最后一个 token。
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
    """加载可选 checkpoint，并按采样参数生成文本。"""
    parser = argparse.ArgumentParser(description="Generate with a tiny Qwen1-style LM.")
    parser.add_argument("--prompt", default="Qwen is")
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
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        eos_token_id=tokenizer.eos_token_id,
    )
    # output_ids: [1, T_prompt + T_new]；只在展示时解码 batch 中第一个样本。
    print(tokenizer.decode(output_ids[0].tolist()))


if __name__ == "__main__":
    main()
