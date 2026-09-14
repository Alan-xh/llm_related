"""
任务 6：无条件生成（Unconditional Generation）

代表模型：
    DDIM (Denoising Diffusion Implicit Model)
    去噪扩散隐式模型

DDIM 与 DDPM 使用相同的前向加噪过程和噪声预测训练目标，区别在于
反向采样时可以跳过大部分时间步。eta=0 时采样是确定性的，eta>0
时会注入可控的随机噪声。

默认使用 synthetic 数据集，便于在没有网络或 GPU 的环境中运行示例。
使用 --dataset mnist 或 --dataset fashion-mnist 可以切换到真实图像数据。
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image


def set_seed(seed: int) -> None:
    """固定随机种子，方便比较不同采样器的结果。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def linear_beta_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """生成线性 beta schedule。"""
    if timesteps < 2:
        raise ValueError("timesteps must be at least 2")
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


class DiffusionSchedule:
    """保存 DDPM/DDIM 共用的前向扩散系数。"""

    def __init__(self, timesteps: int, device: torch.device) -> None:
        betas = linear_beta_schedule(timesteps).to(device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.timesteps = timesteps
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """根据 q(x_t|x_0) 直接生成任意时间步的带噪样本。"""
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t].view(
            -1, 1, 1, 1
        )
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise


class TimeEmbedding(nn.Module):
    """将离散时间步编码为连续向量。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            -scale * torch.arange(half_dim, device=t.device, dtype=torch.float32)
        )
        angles = t.float()[:, None] * frequencies[None, :]
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if embedding.shape[-1] < self.dim:
            embedding = F.pad(embedding, (0, self.dim - embedding.shape[-1]))
        return embedding


class TinyUNet(nn.Module):
    """适合 28x28 或 32x32 图像的教学用小型 U-Net。"""

    def __init__(
        self,
        in_ch: int = 1,
        base: int = 64,
        time_dim: int = 128,
    ) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
        )
        self.mid = nn.Sequential(
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(
                base * 2,
                base,
                3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.GroupNorm(8, base),
            nn.SiLU(),
        )
        self.up1 = nn.Conv2d(base, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time = self.time_embed(t)
        hidden = self.down1(x)
        hidden = self.down2(hidden)
        hidden = hidden + time[:, :, None, None]
        hidden = self.mid(hidden)
        hidden = self.up2(hidden)
        return self.up1(hidden)


class DDIMSampler:
    """DDIM generalized sampler；eta=0 对应确定性 DDIM。"""

    def __init__(self, schedule: DiffusionSchedule) -> None:
        self.schedule = schedule

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        sampling_steps: int,
        eta: float = 0.0,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if not 1 <= sampling_steps <= self.schedule.timesteps:
            raise ValueError(
                "sampling_steps must be in [1, timesteps], "
                f"got {sampling_steps}"
            )
        if eta < 0:
            raise ValueError(f"eta must be non-negative, got {eta}")

        device = device or self.schedule.alpha_bars.device
        # 均匀抽取原始时间轴的子序列，并从高噪声端反向遍历。
        time_indices = torch.linspace(
            0,
            self.schedule.timesteps - 1,
            sampling_steps,
            device=device,
        ).round().long().flip(0)

        x = torch.randn(shape, device=device)
        model.eval()

        for index, timestep in enumerate(time_indices):
            t = torch.full(
                (shape[0],),
                timestep.item(),
                device=device,
                dtype=torch.long,
            )
            predicted_noise = model(x, t)

            alpha_bar_t = self.schedule.alpha_bars[timestep]
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
            predicted_x0 = (
                x - sqrt_one_minus_alpha_bar_t * predicted_noise
            ) / sqrt_alpha_bar_t

            if index + 1 < len(time_indices):
                previous_timestep = time_indices[index + 1]
                alpha_bar_previous = self.schedule.alpha_bars[previous_timestep]
            else:
                # t=0 的前一个状态就是干净图像，alpha_bar_previous=1。
                alpha_bar_previous = torch.ones((), device=device)

            sigma = eta * torch.sqrt(
                torch.clamp(
                    (1.0 - alpha_bar_previous)
                    / (1.0 - alpha_bar_t)
                    * (1.0 - alpha_bar_t / alpha_bar_previous),
                    min=0.0,
                )
            )
            direction_scale = torch.sqrt(
                torch.clamp(1.0 - alpha_bar_previous - sigma.square(), min=0.0)
            )
            random_noise = (
                torch.randn_like(x)
                if sigma.item() > 0
                else torch.zeros_like(x)
            )
            x = (
                torch.sqrt(alpha_bar_previous) * predicted_x0
                + direction_scale * predicted_noise
                + sigma * random_noise
            )

        return x


def build_dataset(
    name: str,
    data_dir: Path,
    synthetic_samples: int,
    synthetic_channels: int,
    synthetic_size: int,
):
    """创建数据集，并返回 (dataset, channels, image_size)。"""
    if name == "synthetic":
        samples = torch.randn(
            synthetic_samples,
            synthetic_channels,
            synthetic_size,
            synthetic_size,
        )
        return TensorDataset(samples), synthetic_channels, synthetic_size

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    if name == "mnist":
        dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )
    elif name == "fashion-mnist":
        dataset = datasets.FashionMNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )
    else:
        raise ValueError(f"unsupported dataset: {name}")
    return dataset, 1, 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small DDIM-compatible model")
    parser.add_argument(
        "--dataset",
        choices=("synthetic", "mnist", "fashion-mnist"),
        default="synthetic",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./ddim_outputs"))
    parser.add_argument("--synthetic-samples", type=int, default=1000)
    parser.add_argument("--synthetic-channels", type=int, default=3)
    parser.add_argument("--synthetic-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_samples < 1:
        raise ValueError("epochs, batch-size and num-samples must be positive")

    set_seed(args.seed)
    device = torch.device(args.device)
    dataset, channels, image_size = build_dataset(
        args.dataset,
        args.data_dir,
        args.synthetic_samples,
        args.synthetic_channels,
        args.synthetic_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    schedule = DiffusionSchedule(args.timesteps, device)
    model = TinyUNet(in_ch=channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            x0 = batch[0].to(device)
            t = torch.randint(
                0,
                args.timesteps,
                (x0.shape[0],),
                device=device,
            )
            noise = torch.randn_like(x0)
            xt = schedule.q_sample(x0, t, noise)
            predicted_noise = model(xt, t)
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / max(len(loader), 1)
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Noise MSE: {average_loss:.4f}"
        )

    checkpoint_path = args.output_dir / "ddim_model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "channels": channels,
            "image_size": image_size,
            "timesteps": args.timesteps,
            "seed": args.seed,
        },
        checkpoint_path,
    )

    sampler = DDIMSampler(schedule)
    samples = sampler.sample(
        model,
        (args.num_samples, channels, image_size, image_size),
        sampling_steps=args.sampling_steps,
        eta=args.eta,
        device=device,
    )
    sample_path = args.output_dir / "ddim_samples.png"
    save_image(
        torch.clamp((samples + 1.0) / 2.0, 0.0, 1.0),
        sample_path,
        nrow=max(1, int(math.sqrt(args.num_samples))),
    )
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved samples to {sample_path}")


if __name__ == "__main__":
    main()
