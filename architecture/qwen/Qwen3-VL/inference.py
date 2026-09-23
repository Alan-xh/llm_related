"""Run image/video inference with an official Qwen3-VL checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .model import Qwen3VLModel
except ImportError:
    from model import Qwen3VLModel

try:
    from architecture.qwen.production.inference_utils import (
        add_inference_arguments,
        make_user_messages,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.qwen.production.inference_utils import (
        add_inference_arguments,
        make_user_messages,
    )


def main() -> None:
    parser = add_inference_arguments(
        argparse.ArgumentParser(description=__doc__),
        "Qwen/Qwen3-VL-8B-Instruct",
    )
    args = parser.parse_args()
    model = Qwen3VLModel.from_pretrained(
        args.model, device_map=args.device_map
    )
    print(model.generate(
        make_user_messages(args),
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    ))


if __name__ == "__main__":
    main()
