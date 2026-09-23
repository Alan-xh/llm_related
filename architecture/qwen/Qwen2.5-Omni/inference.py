"""Run multimodal inference with an official Qwen2.5-Omni checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .model import Qwen2_5OmniModel
except ImportError:
    from model import Qwen2_5OmniModel

try:
    from architecture.qwen.production.inference_utils import (
        add_inference_arguments,
        make_user_messages,
        write_audio,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.qwen.production.inference_utils import (
        add_inference_arguments,
        make_user_messages,
        write_audio,
    )


def main() -> None:
    parser = add_inference_arguments(
        argparse.ArgumentParser(description=__doc__),
        "Qwen/Qwen2.5-Omni-7B",
        audio_input=True,
        audio_output=True,
    )
    args = parser.parse_args()
    model = Qwen2_5OmniModel.from_pretrained(
        args.model, device_map=args.device_map
    )
    result = model.generate(
        make_user_messages(args),
        max_new_tokens=args.max_new_tokens,
        return_audio=bool(args.output_audio),
    )
    print(result.text)
    if args.output_audio and result.audio is not None:
        write_audio(result.audio, args.output_audio)


if __name__ == "__main__":
    main()
