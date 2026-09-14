from __future__ import annotations

import argparse

import torch
from torchvision.utils import save_image

try:
    from .model import build_model
except ImportError:
    from model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from the tiny Stable Diffusion v2 teaching model.")
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="sd_v2_sample.png")
    args = parser.parse_args()

    prompts = args.prompt or ["a blue diagonal", "a yellow grid"]
    device = torch.device(args.device)
    model = build_model().to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.denoiser.load_state_dict(state.get("denoiser", state))
    model.eval()
    images = model.sample(prompts, args.steps, args.guidance_scale, device=device, seed=args.seed)
    save_image(images, args.output, nrow=len(prompts), normalize=False)
    print(f"saved {len(prompts)} image(s) to {args.output}")


if __name__ == "__main__":
    main()
