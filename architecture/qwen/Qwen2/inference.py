"""Qwen2 教学模型 GQA 自回归推理入口。

``generate`` 首轮处理完整 prompt [1, T_prompt]；后续循环仅将最后一个
token [1, 1] 输入模型，并复用每层 [1, n_kv_heads, T_cache, D] 的 KV cache。
这展示了 GQA 在推理时减少缓存 head 数量的接口。
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
    """加载可选 checkpoint，并使用 temperature/top-k/top-p 生成文本。"""
    parser = argparse.ArgumentParser(description="Generate with a tiny Qwen2-style GQA LM.")
    parser.add_argument("--prompt", default="Qwen2 is")
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
    # output_ids: [1, T_prompt + T_new]。
    print(tokenizer.decode(output_ids[0].tolist()))


if __name__ == "__main__":
    main()
