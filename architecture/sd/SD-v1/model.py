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
        ResBlock2D,
        SpatialTransformer,
        TinyTextEncoder,
        TinyVAE,
        classifier_free_guidance,
        timestep_embedding,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.sd.common import (
        DDIMScheduler,
        ResBlock2D,
        SpatialTransformer,
        TinyTextEncoder,
        TinyVAE,
        classifier_free_guidance,
        timestep_embedding,
    )


@dataclass
class SDv1Config:
    image_size: int = 32
    latent_channels: int = 4
    base_channels: int = 32
    text_dim: int = 64
    time_dim: int = 128
    prediction_type: str = "epsilon"
    latent_scaling: float = 0.18215


class SDv1UNet(nn.Module):
    """Tiny latent U-Net with timestep embeddings and CLIP-like cross-attention."""

    def __init__(self, config: SDv1Config) -> None:
        super().__init__()
        channels = config.base_channels
        self.config = config
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_dim, config.time_dim * 4),
            nn.SiLU(),
            nn.Linear(config.time_dim * 4, config.time_dim),
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

    def forward(self, latents: Tensor, timesteps: Tensor, context: Tensor) -> Tensor:
        if timesteps.ndim == 0:
            timesteps = timesteps[None].expand(latents.shape[0])
        time = self.time_mlp(timestep_embedding(timesteps, self.config.time_dim))
        skip = self.attn1(self.down1(self.in_conv(latents), time), context)
        hidden = self.downsample(skip)
        hidden = self.attn2(self.down2(hidden, time), context)
        hidden = self.mid_attn(self.mid(hidden, time), context)
        hidden = self.upsample(hidden)
        hidden = torch.cat((hidden, skip), dim=1)
        hidden = self.up(hidden, time)
        hidden = self.up_attn(hidden, context)
        return self.out_conv(F.silu(self.out_norm(hidden)))


class SDv1Model(nn.Module):
    def __init__(self, config: SDv1Config | None = None) -> None:
        super().__init__()
        self.config = config or SDv1Config()
        self.vae = TinyVAE(self.config.latent_channels)
        self.text_encoder = TinyTextEncoder(self.config.text_dim)
        self.denoiser = SDv1UNet(self.config)
        self.scheduler = DDIMScheduler(prediction_type=self.config.prediction_type)
        for parameter in (*self.vae.parameters(), *self.text_encoder.parameters()):
            parameter.requires_grad_(False)

    def encode_prompts(self, prompts: Sequence[str], device: torch.device) -> Tensor:
        context, _ = self.text_encoder.encode_prompts(prompts, device)
        return context

    def training_loss(self, images: Tensor, prompts: Sequence[str]) -> Tensor:
        with torch.no_grad():
            latents = self.vae.encode(images).mode() * self.config.latent_scaling
            context = self.encode_prompts(prompts, images.device)
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
        )
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        prediction = self.denoiser(noisy_latents, timesteps, context)
        if self.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise
        return F.mse_loss(prediction.float(), target.float())

    @torch.no_grad()
    def sample(
        self,
        prompts: Sequence[str],
        num_inference_steps: int = 8,
        guidance_scale: float = 7.5,
        device: torch.device | None = None,
        seed: int = 0,
    ) -> Tensor:
        device = device or next(self.parameters()).device
        generator = torch.Generator(device=device).manual_seed(seed)
        context = self.encode_prompts(prompts, device)
        unconditional = self.encode_prompts([""] * len(prompts), device)
        height = self.config.image_size // 8
        latents = torch.randn(
            (len(prompts), self.config.latent_channels, height, height),
            generator=generator,
            device=device,
        )
        for timestep in self.scheduler.set_timesteps(num_inference_steps, device):
            timesteps = timestep.expand(len(prompts))
            unconditional_prediction = self.denoiser(latents, timesteps, unconditional)
            conditional_prediction = self.denoiser(latents, timesteps, context)
            prediction = classifier_free_guidance(
                unconditional_prediction, conditional_prediction, guidance_scale
            )
            latents = self.scheduler.step(prediction, timestep, latents).prev_sample
        images = self.vae.decode(latents / self.config.latent_scaling)
        return (images.clamp(-1.0, 1.0) + 1.0) / 2.0


def build_model() -> SDv1Model:
    return SDv1Model()


if __name__ == "__main__":
    model = build_model()
    print(f"SD-v1 tiny parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"latent shape for 32x32 image: [B, 4, {model.config.image_size // 8}, {model.config.image_size // 8}]")
