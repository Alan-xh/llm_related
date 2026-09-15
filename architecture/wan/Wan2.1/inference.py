from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    from .model import build_model
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run tiny Wan2.1 text-to-video or image-to-video sampling."
    )
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=("t2v", "i2v"), default="t2v")
    parser.add_argument("--output", default="wan21_sample.pt")
    args = parser.parse_args()
    prompts = args.prompt or ["a blue moving shape"]
    device = torch.device(args.device)
    model = build_model().to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.denoiser.load_state_dict(state.get("denoiser", state))
    model.eval()
    if args.mode == "i2v":
        frame = torch.zeros(
            len(prompts), 3, model.config.height, model.config.width, device=device
        )
        frame[:, 0] = 1.0
        video = model.sample_i2v(prompts, frame, args.steps, device, args.seed)
    else:
        video = model.sample_t2v(prompts, args.steps, device, args.seed)
    torch.save(video.cpu(), args.output)
    print(f"saved video tensor {tuple(video.shape)} to {args.output}")


if __name__ == "__main__":
    main()
