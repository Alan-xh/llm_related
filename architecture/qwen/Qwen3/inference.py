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
    if thinking:
        return f"<|user|>\n{prompt}\n<|assistant|>\n<think>\n"
    return f"<|user|>\n{prompt}\n<|assistant|>\n"


def main() -> None:
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
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    output_ids = model.generate(input_ids, args.max_new_tokens, args.temperature, eos_token_id=2)
    print(tokenizer.decode(output_ids[0].tolist()))


if __name__ == "__main__":
    main()
