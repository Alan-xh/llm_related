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
        Attention,
        FlowMatchScheduler,
        TinyTextEncoder,
        TinyVAE,
        classifier_free_guidance,
        timestep_embedding,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.sd.common import (
        Attention,
        FlowMatchScheduler,
        TinyTextEncoder,
        TinyVAE,
        classifier_free_guidance,
        timestep_embedding,
    )


@dataclass
class SD3Config:
    image_size: int = 32
    latent_channels: int = 4
    text_dim: int = 80
    hidden_size: int = 128
    time_dim: int = 128
    layers: int = 3
    max_image_tokens: int = 256
    latent_scaling: float = 0.18215


class MMDiTBlock(nn.Module):
    """A compact joint block with separate text/image streams and shared attention."""

    def __init__(self, hidden_size: int, time_dim: int) -> None:
        super().__init__()
        self.image_norm = nn.LayerNorm(hidden_size)
        self.text_norm = nn.LayerNorm(hidden_size)
        self.joint_attention = Attention(hidden_size, heads=4, head_dim=32)
        self.image_time = nn.Linear(time_dim, hidden_size)
        self.text_time = nn.Linear(time_dim, hidden_size)
        self.image_ff = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size * 4), nn.GELU(), nn.Linear(hidden_size * 4, hidden_size))
        self.text_ff = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size * 4), nn.GELU(), nn.Linear(hidden_size * 4, hidden_size))

    def forward(self, image_tokens: Tensor, text_tokens: Tensor, time: Tensor) -> tuple[Tensor, Tensor]:
        image_tokens = image_tokens + self.image_time(time)[:, None]
        text_tokens = text_tokens + self.text_time(time)[:, None]
        joint = torch.cat((self.image_norm(image_tokens), self.text_norm(text_tokens)), dim=1)
        joint = joint + self.joint_attention(joint)
        image_length = image_tokens.shape[1]
        image_tokens = image_tokens + joint[:, :image_length]
        text_tokens = text_tokens + joint[:, image_length:]
        return image_tokens + self.image_ff(image_tokens), text_tokens + self.text_ff(text_tokens)


class MMDiTDenoiser(nn.Module):
    def __init__(self, config: SD3Config) -> None:
        super().__init__()
        self.config = config
        self.image_in = nn.Linear(config.latent_channels, config.hidden_size)
        self.text_in = nn.Linear(config.text_dim, config.hidden_size)
        self.time_in = nn.Sequential(
            nn.Linear(config.time_dim, config.time_dim * 4),
            nn.SiLU(),
            nn.Linear(config.time_dim * 4, config.time_dim),
        )
        self.image_pos = nn.Parameter(torch.zeros(1, config.max_image_tokens, config.hidden_size))
        self.blocks = nn.ModuleList(
            [MMDiTBlock(config.hidden_size, config.time_dim) for _ in range(config.layers)]
        )
        self.image_out = nn.Sequential(nn.LayerNorm(config.hidden_size), nn.Linear(config.hidden_size, config.latent_channels))
        nn.init.normal_(self.image_pos, std=0.02)

    def forward(self, latents: Tensor, timesteps: Tensor, context: Tensor) -> Tensor:
        if timesteps.ndim == 0:
            timesteps = timesteps[None].expand(latents.shape[0])
        batch, _, height, width = latents.shape
        image_tokens = latents.flatten(2).transpose(1, 2)
        image_tokens = self.image_in(image_tokens) + self.image_pos[:, : image_tokens.shape[1]]
        text_tokens = self.text_in(context)
        time = self.time_in(timestep_embedding(timesteps, self.config.time_dim))
        for block in self.blocks:
            image_tokens, text_tokens = block(image_tokens, text_tokens, time)
        output = self.image_out(image_tokens)
        return output.transpose(1, 2).reshape(batch, self.config.latent_channels, height, width)


class SD3Model(nn.Module):
    def __init__(self, config: SD3Config | None = None) -> None:
        super().__init__()
        self.config = config or SD3Config()
        self.vae = TinyVAE(self.config.latent_channels)
        self.text_encoder = TinyTextEncoder(self.config.text_dim)
        self.denoiser = MMDiTDenoiser(self.config)
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
        num_inference_steps: int = 8,
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
            unconditional_velocity = self.denoiser(latents, current, unconditional)
            conditional_velocity = self.denoiser(latents, current, context)
            velocity = classifier_free_guidance(
                unconditional_velocity, conditional_velocity, guidance_scale
            )
            latents = self.scheduler.step(velocity, timestep, latents, next_timestep)
        images = self.vae.decode(latents / self.config.latent_scaling)
        return (images.clamp(-1.0, 1.0) + 1.0) / 2.0


def build_model() -> SD3Model:
    return SD3Model()


if __name__ == "__main__":
    model = build_model()
    print(f"SD3 tiny parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("image stream and text stream are projected separately, then joined for attention")
