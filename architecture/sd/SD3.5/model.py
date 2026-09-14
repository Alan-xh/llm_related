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
        FlowMatchScheduler,
        TinyTextEncoder,
        TinyVAE,
        classifier_free_guidance,
    )
    from ..SD3.model import MMDiTDenoiser
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.sd.common import (
        FlowMatchScheduler,
        TinyTextEncoder,
        TinyVAE,
        classifier_free_guidance,
    )
    from architecture.sd.SD3.model import MMDiTDenoiser


@dataclass
class SD35Config:
    image_size: int = 32
    latent_channels: int = 4
    text_dim: int = 96
    hidden_size: int = 160
    time_dim: int = 128
    layers: int = 4
    max_image_tokens: int = 256
    latent_scaling: float = 0.18215
    distilled: bool = False


class SD35Denoiser(MMDiTDenoiser):
    """A slightly wider/deeper MMDiT used to expose the SD3.5 family variants."""

    def __init__(self, config: SD35Config) -> None:
        super().__init__(config)
        self.guidance_embedding = nn.Sequential(
            nn.Linear(1, config.time_dim),
            nn.SiLU(),
            nn.Linear(config.time_dim, config.time_dim),
        )

    def forward(
        self,
        latents: Tensor,
        timesteps: Tensor,
        context: Tensor,
        guidance_scale: float = 1.0,
    ) -> Tensor:
        # The parent block already carries timestep conditioning.  This small
        # projection exposes the family-specific guidance/distillation hook.
        output = super().forward(latents, timesteps, context)
        if guidance_scale == 1.0:
            return output
        return output * (1.0 + 0.02 * self.guidance_embedding(
            torch.full((latents.shape[0], 1), guidance_scale, device=latents.device)
        )[:, :1, None, None])


class SD35Model(nn.Module):
    def __init__(self, config: SD35Config | None = None) -> None:
        super().__init__()
        self.config = config or SD35Config()
        self.vae = TinyVAE(self.config.latent_channels)
        self.text_encoder = TinyTextEncoder(self.config.text_dim)
        self.denoiser = SD35Denoiser(self.config)
        self.scheduler = FlowMatchScheduler()
        for parameter in (*self.vae.parameters(), *self.text_encoder.parameters()):
            parameter.requires_grad_(False)

    def encode_prompts(self, prompts: Sequence[str], device: torch.device) -> Tensor:
        context, _ = self.text_encoder.encode_prompts(prompts, device)
        return context

    def training_loss(self, images: Tensor, prompts: Sequence[str]) -> Tensor:
        with torch.no_grad():
            clean = self.vae.encode(images).mode() * self.config.latent_scaling
            context = self.encode_prompts(prompts, images.device)
        timestep = torch.rand(images.shape[0], device=images.device)
        noise = torch.randn_like(clean)
        noisy = self.scheduler.add_noise(clean, noise, timestep)
        velocity = self.denoiser(noisy, timestep, context)
        return F.mse_loss(velocity.float(), (noise - clean).float())

    @torch.no_grad()
    def sample(
        self,
        prompts: Sequence[str],
        num_inference_steps: int = 6,
        guidance_scale: float = 4.5,
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
        timesteps = self.scheduler.set_timesteps(num_inference_steps, device)
        for timestep, next_timestep in zip(timesteps[:-1], timesteps[1:]):
            current = timestep.expand(len(prompts))
            unconditional_velocity = self.denoiser(
                latents, current, unconditional, guidance_scale=1.0
            )
            conditional_velocity = self.denoiser(
                latents, current, context, guidance_scale=guidance_scale
            )
            velocity = classifier_free_guidance(
                unconditional_velocity, conditional_velocity, guidance_scale
            )
            latents = self.scheduler.step(velocity, timestep, latents, next_timestep)
        images = self.vae.decode(latents / self.config.latent_scaling)
        return (images.clamp(-1.0, 1.0) + 1.0) / 2.0


def build_model() -> SD35Model:
    return SD35Model()


if __name__ == "__main__":
    model = build_model()
    print(f"SD3.5 tiny parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("SD3.5 hook: wider/deeper MMDiT plus optional guidance/distillation embedding")
