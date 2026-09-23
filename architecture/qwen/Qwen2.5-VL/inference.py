"""Run image/video inference with an official Qwen2.5-VL checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .model import Qwen2_5VLModel
except ImportError:
    from model import Qwen2_5VLModel

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
        "Qwen/Qwen2.5-VL-7B-Instruct",
    )
    args = parser.parse_args()
    model = Qwen2_5VLModel.from_pretrained(
        args.model, device_map=args.device_map
    )
    print(model.generate(
        make_user_messages(args),
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    ))


if __name__ == "__main__":
    main()
