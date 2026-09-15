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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run tiny Wan2.2 text-image-to-video sampling."
    )
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="wan22_sample.pt")
    args = parser.parse_args()
    prompts = args.prompt or ["a green object moving right"]
    device = torch.device(args.device)
    model = build_model().to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.denoiser.load_state_dict(state.get("denoiser", state))
    model.eval()
    first_frame = torch.zeros(
        len(prompts), 3, model.config.height, model.config.width, device=device
    )
    first_frame[:, 1] = 0.8
    video = model.sample_ti2v(prompts, first_frame, args.steps, device, args.seed)
    torch.save(video.cpu(), args.output)
    print(f"saved video tensor {tuple(video.shape)} to {args.output}")


if __name__ == "__main__":
    main()
