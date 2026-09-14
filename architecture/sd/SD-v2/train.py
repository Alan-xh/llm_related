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
    from ..common import build_toy_images
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.sd.common import build_toy_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the tiny Stable Diffusion v2 teaching model.")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default="sd_v2_tiny.pt")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.denoiser.parameters(), lr=args.lr)
    model.train()
    for step in range(args.steps):
        images, prompts = build_toy_images(args.batch_size, args.image_size, device, step)
        loss = model.training_loss(images, prompts)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), 1.0)
        optimizer.step()
        print(f"step {step + 1:03d}/{args.steps}: loss={loss.item():.4f}")
    torch.save({"denoiser": model.denoiser.state_dict()}, args.checkpoint)
    print(f"saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
