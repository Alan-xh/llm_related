"""Kimi Linear 风格 KDA/MLA 混合生成入口。

任务定义:
    加载可选教学 checkpoint，使用字节级 tokenizer 编码 prompt，并通过
    KDA 有限状态缓存和 MLA KV cache 执行增量生成。

张量约定:
    prompt token ids: [1, T_prompt]；生成结果: [1, T_prompt + T_new]。
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
    from architecture.kimi.common import ByteTokenizer, format_prompt
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.kimi.common import ByteTokenizer, format_prompt


def main() -> None:
    """解析命令行参数并运行 Kimi Linear 风格的缓存生成。"""
    parser = argparse.ArgumentParser(description="使用 Kimi Linear 风格的最小模型生成文本。")
    parser.add_argument("--prompt", default="Why can linear attention reduce decoding memory?")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--checkpoint")
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
    output = model.generate(
        input_ids,
        args.max_new_tokens,
        args.temperature,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
