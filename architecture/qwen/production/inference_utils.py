"""Shared command-line helpers for checkpoint-backed multimodal inference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def add_inference_arguments(
    parser: argparse.ArgumentParser,
    default_model: str,
    *,
    audio_input: bool = False,
    audio_output: bool = False,
) -> argparse.ArgumentParser:
    parser.add_argument("--model", default=default_model, help="Hugging Face model id or local path.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", action="append", default=[], help="Image path; repeatable.")
    parser.add_argument("--video", action="append", default=[], help="Video path; repeatable.")
    if audio_input:
        parser.add_argument(
            "--audio", action="append", default=[], help="Audio path; repeatable."
        )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device-map", default="auto")
    if audio_output:
        parser.add_argument(
            "--output-audio", help="Write generated speech to this WAV file."
        )
    return parser


def make_user_messages(args: argparse.Namespace) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    content.extend({"type": "image", "image": path} for path in args.image)
    content.extend({"type": "video", "video": path} for path in args.video)
    content.extend({"type": "audio", "audio": path} for path in args.audio)
    content.append({"type": "text", "text": args.prompt})
    return [{"role": "user", "content": content}]


def write_audio(audio: Any, output_path: str) -> None:
    import soundfile as sf

    waveform = audio.reshape(-1).detach().cpu().numpy()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, waveform, 24000)
