"""Qwen3 教学模型 thinking/non-thinking 自回归推理入口。

thinking 模式只改变 prompt 模板：在 assistant 区域追加 ``<think>``；
网络仍是同一个 GQA + QK normalization + MoE decoder。编码后的输入 shape
为 [1, T_prompt]，生成结果 shape 为 [1, T_prompt + T_new]。
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


def format_prompt(prompt: str, thinking: bool) -> str:
    """将用户文本包装成 thinking 或直接回答的最小对话模板。"""
    if thinking:
        return f"<|user|>\n{prompt}\n<|assistant|>\n<think>\n"
    return f"<|user|>\n{prompt}\n<|assistant|>\n"


def main() -> None:
    """加载 Qwen3 教学模型，按 prompt 模式生成回答。"""
    parser = argparse.ArgumentParser(description="Generate with a tiny Qwen3-style MoE LM.")
    parser.add_argument("--prompt", default="Explain grouped-query attention.")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = build_model().to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state.get("model", state))
    tokenizer = ByteTokenizer()
    prompt = format_prompt(args.prompt, args.thinking)
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)  # [1, T_prompt]
    output_ids = model.generate(input_ids, args.max_new_tokens, args.temperature, eos_token_id=2)
    # generate 使用逐层 KV cache；输出包含 prompt，shape: [1, T_prompt + T_new]。
    print(tokenizer.decode(output_ids[0].tolist()))


if __name__ == "__main__":
    main()
