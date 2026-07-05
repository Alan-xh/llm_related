"""
Simple inference / reconstruction demo for the trained VAE.
"""

import argparse

import torch
import torchvision
from torchvision import transforms
from PIL import Image

from .model import VAE


@torch.no_grad()
def reconstruct(model, image_path, output_path, image_size=256, device="cuda"):
    device = torch.device(device)
    model = model.to(device).eval()

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    recon, posterior = model(x, sample_posterior=False)
    # concat input and reconstruction side-by-side
    grid = torch.cat([x, recon], dim=0)
    grid = (grid + 1.0) / 2.0  # to [0, 1]
    torchvision.utils.save_image(grid, output_path)
    print(f"Saved reconstruction to {output_path}")


@torch.no_grad()
def sample(model, output_path, num_samples=8, image_size=256, device="cuda"):
    device = torch.device(device)
    model = model.to(device).eval()

    # infer latent shape from a dummy forward pass
    dummy = torch.zeros(1, 3, image_size, image_size, device=device)
    posterior = model.encode(dummy)
    z = posterior.mode()
    shape = (num_samples,) + tuple(z.shape[1:])

    z = torch.randn(shape, device=device)
    samples = model.decode(z)
    samples = (samples + 1.0) / 2.0
    torchvision.utils.save_image(samples, output_path, nrow=4)
    print(f"Saved samples to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, default=None, help="Image to reconstruct")
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_samples", type=int, default=8)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt.get("config", {})

    model = VAE(
        ch=config.get("ch", 128),
        ch_mult=tuple(config.get("ch_mult", [1, 2, 4, 8])),
        num_res_blocks=config.get("num_res_blocks", 2),
        attn_resolutions=tuple(config.get("attn_resolutions", [])),
        dropout=config.get("dropout", 0.0),
        in_channels=3,
        resolution=args.image_size,
        z_channels=config.get("z_channels", 4),
        double_z=True,
        embed_dim=config.get("embed_dim", 4),
    )
    model.load_state_dict(ckpt["model_state_dict"])

    if args.image is not None:
        reconstruct(model, args.image, args.output, args.image_size, args.device)
    else:
        sample(model, args.output, args.num_samples, args.image_size, args.device)


if __name__ == "__main__":
    main()
