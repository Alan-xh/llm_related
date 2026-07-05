"""
Plain PyTorch training script for the VAE.
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from .model import VAE
from .loss import VAELoss


def get_dataloader(data_dir, batch_size, image_size, num_workers=4):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # to [-1, 1]
        ]
    )
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    total_rec = 0.0
    total_kl = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch_idx, (images, _) in enumerate(pbar):
        images = images.to(device)

        recon, posterior = model(images)
        loss, log_dict = criterion(images, recon, posterior)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += log_dict["loss"].item()
        total_rec += log_dict["rec_loss"].item()
        total_kl += log_dict["kl_loss"].item()

        pbar.set_postfix(
            loss=f"{log_dict['loss'].item():.4f}",
            rec=f"{log_dict['rec_loss'].item():.4f}",
            kl=f"{log_dict['kl_loss'].item():.4f}",
        )

    n = len(loader)
    return total_loss / n, total_rec / n, total_kl / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_rec = 0.0
    total_kl = 0.0
    for images, _ in loader:
        images = images.to(device)
        recon, posterior = model(images)
        loss, log_dict = criterion(images, recon, posterior)
        total_loss += log_dict["loss"].item()
        total_rec += log_dict["rec_loss"].item()
        total_kl += log_dict["kl_loss"].item()
    n = len(loader)
    return total_loss / n, total_rec / n, total_kl / n


def main():
    parser = argparse.ArgumentParser(description="Train a KL-VAE")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to image folder dataset")
    parser.add_argument("--output_dir", type=str, default="./vae_checkpoints", help="Checkpoint output dir")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=4.5e-6)
    parser.add_argument("--kl_weight", type=float, default=1e-6)
    parser.add_argument("--ch", type=int, default=128)
    parser.add_argument("--ch_mult", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--z_channels", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=4)
    parser.add_argument("--attn_resolutions", type=int, nargs="+", default=[])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)

    model = VAE(
        ch=args.ch,
        ch_mult=tuple(args.ch_mult),
        num_res_blocks=args.num_res_blocks,
        attn_resolutions=tuple(args.attn_resolutions),
        dropout=args.dropout,
        in_channels=3,
        resolution=args.image_size,
        z_channels=args.z_channels,
        double_z=True,
        embed_dim=args.embed_dim,
    ).to(device)

    criterion = VAELoss(rec_loss="l1", kl_weight=args.kl_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.9))

    train_loader = get_dataloader(args.data_dir, args.batch_size, args.image_size, args.num_workers)

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        loss, rec, kl = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        elapsed = time.time() - start
        print(f"Epoch {epoch}/{args.epochs} | loss: {loss:.4f} rec: {rec:.4f} kl: {kl:.4f} | {elapsed:.1f}s")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = Path(args.output_dir) / f"vae_epoch_{epoch:04d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "config": vars(args),
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
