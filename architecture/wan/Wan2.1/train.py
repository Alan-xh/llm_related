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


def build_toy_videos(
    batch_size: int, frames: int, height: int, width: int, device: torch.device
) -> tuple[torch.Tensor, list[str]]:
    axis = torch.linspace(-1.0, 1.0, width, device=device)
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device), axis, indexing="ij"
    )
    videos = []
    prompts = []
    names = ("red motion", "green circle", "blue diagonal", "yellow grid")
    for index in range(batch_size):
        variant = index % 4
        values = []
        for frame in range(frames):
            phase = frame / max(frames - 1, 1)
            if variant == 0:
                image = torch.stack((xx * 0 + 0.8, yy * 0, yy * 0), dim=0)
            elif variant == 1:
                mask = (((xx - phase * 0.5 + 0.25) ** 2 + yy**2) < 0.35).float()
                image = torch.stack((mask * 0.1, mask * 0.8, mask * 0.2), dim=0)
            elif variant == 2:
                stripe = ((xx + yy + phase) > 0).float()
                image = torch.stack((stripe * 0.1, stripe * 0.3, stripe * 0.9), dim=0)
            else:
                stripe = (
                    ((xx * width).long() + (yy * height).long() + frame) % 2
                ).float()
                image = torch.stack((stripe * 0.8, stripe * 0.7, stripe * 0.1), dim=0)
            values.append(image * 2.0 - 1.0)
        videos.append(torch.stack(values, dim=1))
        prompts.append(names[variant])
    return torch.stack(videos), prompts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the tiny Wan2.1 flow-matching video model."
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default="wan21_tiny.pt")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.denoiser.parameters(), lr=args.lr)
    model.train()
    for step in range(args.steps):
        videos, prompts = build_toy_videos(
            args.batch_size, args.frames, args.height, args.width, device
        )
        loss = model.training_loss(videos, prompts)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), 1.0)
        optimizer.step()
        print(f"step {step + 1:03d}/{args.steps}: loss={loss.item():.4f}")
    torch.save({"denoiser": model.denoiser.state_dict()}, args.checkpoint)
    print(f"saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
