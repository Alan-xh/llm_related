from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from ..common import (
        DDIMScheduler,
        DualTextEncoder,
        ResBlock2D,
        SpatialTransformer,
        TinyVAE,
        classifier_free_guidance,
        timestep_embedding,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.sd.common import (
        DDIMScheduler,
        DualTextEncoder,
        ResBlock2D,
        SpatialTransformer,
        TinyVAE,
        classifier_free_guidance,
        timestep_embedding,
    )


@dataclass
class SDXLConfig:
    image_size: int = 32
    latent_channels: int = 4
    base_channels: int = 40
    text_dim: int = 144
    pooled_dim: int = 144
    time_dim: int = 128
    latent_scaling: float = 0.18215


class SDXLUNet(nn.Module):
    """Small SDXL-like U-Net with dual text and pooled/time-id conditioning."""

    def __init__(self, config: SDXLConfig) -> None:
        super().__init__()
        channels = config.base_channels
        self.config = config
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_dim, config.time_dim * 4),
            nn.SiLU(),
            nn.Linear(config.time_dim * 4, config.time_dim),
        )
        self.added_cond = nn.Sequential(
            nn.Linear(config.pooled_dim + 6, config.time_dim),
            nn.SiLU(),
            nn.Linear(config.time_dim, config.time_dim),
        )
        self.in_conv = nn.Conv2d(config.latent_channels, channels, 3, padding=1)
        self.down1 = ResBlock2D(channels, channels, config.time_dim)
        self.attn1 = SpatialTransformer(channels, config.text_dim)
        self.downsample = nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1)
        self.down2 = ResBlock2D(channels * 2, channels * 2, config.time_dim)
        self.attn2 = SpatialTransformer(channels * 2, config.text_dim)
        self.mid = ResBlock2D(channels * 2, channels * 2, config.time_dim)
        self.mid_attn = SpatialTransformer(channels * 2, config.text_dim)
        self.upsample = nn.ConvTranspose2d(channels * 2, channels, 4, stride=2, padding=1)
        self.up = ResBlock2D(channels * 2, channels, config.time_dim)
        self.up_attn = SpatialTransformer(channels, config.text_dim)
        self.out_norm = nn.GroupNorm(8, channels)
        self.out_conv = nn.Conv2d(channels, config.latent_channels, 3, padding=1)

    def forward(
        self,
        latents: Tensor,
        timesteps: Tensor,
        context: Tensor,
        pooled: Tensor,
        time_ids: Tensor,
    ) -> Tensor:
        if timesteps.ndim == 0:
            timesteps = timesteps[None].expand(latents.shape[0])
        time = self.time_mlp(timestep_embedding(timesteps, self.config.time_dim))
        time = time + self.added_cond(torch.cat((pooled, time_ids), dim=-1))
        skip = self.attn1(self.down1(self.in_conv(latents), time), context)
        hidden = self.downsample(skip)
        hidden = self.attn2(self.down2(hidden, time), context)
        hidden = self.mid_attn(self.mid(hidden, time), context)
        hidden = self.upsample(hidden)
        hidden = self.up(torch.cat((hidden, skip), dim=1), time)
        hidden = self.up_attn(hidden, context)
        return self.out_conv(F.silu(self.out_norm(hidden)))


class SDXLModel(nn.Module):
    def __init__(self, config: SDXLConfig | None = None) -> None:
        super().__init__()
        self.config = config or SDXLConfig()
        self.vae = TinyVAE(self.config.latent_channels)
        self.text_encoder = DualTextEncoder()
        self.base = SDXLUNet(self.config)
        self.refiner = SDXLUNet(self.config)
        self.scheduler = DDIMScheduler()
        self.refiner_scheduler = DDIMScheduler()
        for parameter in (*self.vae.parameters(), *self.text_encoder.parameters()):
            parameter.requires_grad_(False)

    def encode_prompts(
        self, prompts: Sequence[str], device: torch.device
    ) -> tuple[Tensor, Tensor]:
        return self.text_encoder.encode_prompts(prompts, device)

    def _time_ids(self, batch_size: int, device: torch.device) -> Tensor:
        size = float(self.config.image_size)
        return torch.tensor(
            [size, size, 0.0, 0.0, size, size],
            device=device,
        ).expand(batch_size, -1)

    def training_loss(self, images: Tensor, prompts: Sequence[str]) -> Tensor:
        with torch.no_grad():
            latents = self.vae.encode(images).mode() * self.config.latent_scaling
            context, pooled = self.encode_prompts(prompts, images.device)
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
        )
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        time_ids = self._time_ids(images.shape[0], images.device)
        prediction = self.base(noisy_latents, timesteps, context, pooled, time_ids)
        return F.mse_loss(prediction.float(), noise.float())

    @torch.no_grad()
    def sample(
        self,
        prompts: Sequence[str],
        num_inference_steps: int = 8,
        refiner_steps: int = 2,
        guidance_scale: float = 5.0,
        device: torch.device | None = None,
        seed: int = 0,
    ) -> Tensor:
        device = device or next(self.parameters()).device
        generator = torch.Generator(device=device).manual_seed(seed)
        context, pooled = self.encode_prompts(prompts, device)
        unconditional, unconditional_pooled = self.encode_prompts([""] * len(prompts), device)
        time_ids = self._time_ids(len(prompts), device)
        height = self.config.image_size // 8
        latents = torch.randn(
            (len(prompts), self.config.latent_channels, height, height),
            generator=generator,
            device=device,
        )
        for timestep in self.scheduler.set_timesteps(num_inference_steps, device):
            timesteps = timestep.expand(len(prompts))
            unconditional_prediction = self.base(
                latents, timesteps, unconditional, unconditional_pooled, time_ids
            )
            conditional_prediction = self.base(
                latents, timesteps, context, pooled, time_ids
            )
            latents = self.scheduler.step(
                classifier_free_guidance(
                    unconditional_prediction, conditional_prediction, guidance_scale
                ),
                timestep,
                latents,
            ).prev_sample
        for timestep in self.refiner_scheduler.set_timesteps(refiner_steps, device):
            timesteps = timestep.expand(len(prompts))
            unconditional_prediction = self.refiner(
                latents, timesteps, unconditional, unconditional_pooled, time_ids
            )
            conditional_prediction = self.refiner(
                latents, timesteps, context, pooled, time_ids
            )
            latents = self.refiner_scheduler.step(
                classifier_free_guidance(
                    unconditional_prediction, conditional_prediction, guidance_scale
                ),
                timestep,
                latents,
            ).prev_sample
        images = self.vae.decode(latents / self.config.latent_scaling)
        return (images.clamp(-1.0, 1.0) + 1.0) / 2.0


def build_model() -> SDXLModel:
    return SDXLModel()


if __name__ == "__main__":
    model = build_model()
    print(f"SDXL tiny parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("conditioning: concat(text_encoder_1, text_encoder_2) + pooled embedding + 6D time ids")
