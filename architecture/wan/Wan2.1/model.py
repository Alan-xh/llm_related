from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class Wan21Config:
    frames: int = 4
    height: int = 32
    width: int = 32
    in_channels: int = 3
    latent_channels: int = 4
    spatial_downsample: int = 4
    temporal_downsample: int = 2
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 64
    hidden_size: int = 96
    heads: int = 4
    layers: int = 3
    max_text_length: int = 64
    max_tokens: int = 1024


class CausalVideoVAE(nn.Module):
    """Small causal-in-time VAE-like encoder/decoder for [B, C, T, H, W]."""

    def __init__(self, config: Wan21Config) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Conv3d(config.in_channels, 32, 3, stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(32, 64, 3, stride=(2, 2, 2), padding=1),
            nn.SiLU(),
            nn.Conv3d(64, config.latent_channels, 3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(config.latent_channels, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(64, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(32, 3, 3, padding=1),
            nn.Tanh(),
        )

    def encode(self, video: Tensor) -> Tensor:
        # Left-only temporal padding makes the first latent depend only on
        # frames at or before its receptive field.
        hidden = F.pad(video, (0, 0, 0, 0, 2, 0))
        return self.encoder(hidden)

    def decode(self, latents: Tensor, output_shape: tuple[int, int, int]) -> Tensor:
        hidden = self.decoder(latents)
        return F.interpolate(
            hidden, size=output_shape, mode="trilinear", align_corners=False
        )


class ByteTokenizer:
    vocab_size = 259
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def batch_encode(
        self, texts: Sequence[str], max_length: int, device: torch.device
    ) -> Tensor:
        result = torch.full(
            (len(texts), max_length), self.pad_token_id, dtype=torch.long, device=device
        )
        for row, text in enumerate(texts):
            ids = [self.bos_token_id]
            ids.extend(byte + 3 for byte in text.encode("utf-8")[: max_length - 2])
            ids.append(self.eos_token_id)
            result[row, : len(ids)] = torch.tensor(ids, device=device)
        return result


class TinyT5Encoder(nn.Module):
    """A dependency-free T5-like encoder used for text conditioning."""

    def __init__(self, config: Wan21Config) -> None:
        super().__init__()
        self.tokenizer = ByteTokenizer()
        self.embedding = nn.Embedding(self.tokenizer.vocab_size, config.text_dim)
        self.position = nn.Parameter(
            torch.zeros(1, config.max_text_length, config.text_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.text_dim,
            nhead=4,
            dim_feedforward=config.text_dim * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(config.text_dim)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, texts: Sequence[str], device: torch.device) -> Tensor:
        ids = self.tokenizer.batch_encode(texts, self.position.shape[1], device)
        hidden = self.embedding(ids) + self.position[:, : ids.shape[1]]
        padding = ids.eq(self.tokenizer.pad_token_id)
        return self.norm(self.encoder(hidden, src_key_padding_mask=padding))


def timestep_embedding(timesteps: Tensor, dim: int) -> Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half, 1)
    )
    angles = timesteps.float()[:, None] * frequencies[None]
    result = torch.cat((angles.cos(), angles.sin()), dim=-1)
    return F.pad(result, (0, dim - result.shape[-1]))


class VideoPatchifier:
    def __init__(self, patch_size: tuple[int, int, int]) -> None:
        self.patch_size = patch_size

    def patchify(self, video: Tensor) -> Tensor:
        batch, channels, frames, height, width = video.shape
        pt, ph, pw = self.patch_size
        if (frames % pt, height % ph, width % pw) != (0, 0, 0):
            raise ValueError("video dimensions must be divisible by patch_size")
        video = video.reshape(
            batch, channels, frames // pt, pt, height // ph, ph, width // pw, pw
        )
        video = video.permute(0, 2, 4, 6, 1, 3, 5, 7)
        return video.reshape(batch, -1, channels * pt * ph * pw)

    def unpatchify(
        self, tokens: Tensor, channels: int, shape: tuple[int, int, int]
    ) -> Tensor:
        frames, height, width = shape
        pt, ph, pw = self.patch_size
        grid = (frames // pt, height // ph, width // pw)
        video = tokens.reshape(tokens.shape[0], *grid, channels, pt, ph, pw)
        video = video.permute(0, 4, 1, 5, 2, 6, 3, 7)
        return video.reshape(tokens.shape[0], channels, frames, height, width)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, heads: int, text_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(hidden_size, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.text_norm = nn.LayerNorm(text_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, heads, batch_first=True, kdim=text_dim, vdim=text_dim
        )
        self.norm3 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, tokens: Tensor, context: Tensor) -> Tensor:
        query = self.norm1(tokens)
        tokens = tokens + self.self_attn(query, query, query, need_weights=False)[0]
        tokens = (
            tokens
            + self.cross_attn(
                self.norm2(tokens),
                self.text_norm(context),
                self.text_norm(context),
                need_weights=False,
            )[0]
        )
        return tokens + self.ff(self.norm3(tokens))


class FlowMatchingDiT(nn.Module):
    def __init__(self, config: Wan21Config) -> None:
        super().__init__()
        self.config = config
        self.patchifier = VideoPatchifier(config.patch_size)
        patch_dim = config.latent_channels * math.prod(config.patch_size)
        self.input = nn.Linear(patch_dim, config.hidden_size)
        self.position = nn.Parameter(
            torch.zeros(1, config.max_tokens, config.hidden_size)
        )
        self.time = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.SiLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(config.hidden_size, config.heads, config.text_dim)
                for _ in range(config.layers)
            ]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(config.hidden_size), nn.Linear(config.hidden_size, patch_dim)
        )
        nn.init.normal_(self.position, std=0.02)

    def forward(self, latents: Tensor, timesteps: Tensor, context: Tensor) -> Tensor:
        shape = latents.shape[2:]
        tokens = self.patchifier.patchify(latents)
        if tokens.shape[1] > self.position.shape[1]:
            raise ValueError("max_tokens is smaller than the video patch grid")
        tokens = self.input(tokens) + self.position[:, : tokens.shape[1]]
        tokens = (
            tokens
            + self.time(timestep_embedding(timesteps, self.config.hidden_size))[:, None]
        )
        for block in self.blocks:
            tokens = block(tokens, context)
        return self.patchifier.unpatchify(
            self.output(tokens), self.config.latent_channels, shape
        )


class Wan21Model(nn.Module):
    def __init__(self, config: Wan21Config | None = None) -> None:
        super().__init__()
        self.config = config or Wan21Config()
        self.vae = CausalVideoVAE(self.config)
        self.text_encoder = TinyT5Encoder(self.config)
        self.denoiser = FlowMatchingDiT(self.config)
        for parameter in (*self.vae.parameters(), *self.text_encoder.parameters()):
            parameter.requires_grad_(False)

    def encode_text(self, prompts: Sequence[str], device: torch.device) -> Tensor:
        return self.text_encoder(prompts, device)

    def training_loss(self, videos: Tensor, prompts: Sequence[str]) -> Tensor:
        with torch.no_grad():
            clean = self.vae.encode(videos)
            context = self.encode_text(prompts, videos.device)
        timesteps = torch.rand(videos.shape[0], device=videos.device)
        noise = torch.randn_like(clean)
        noisy = (1.0 - timesteps[:, None, None, None, None]) * clean + timesteps[
            :, None, None, None, None
        ] * noise
        prediction = self.denoiser(noisy, timesteps, context)
        return F.mse_loss(prediction.float(), (noise - clean).float())

    @torch.no_grad()
    def sample_t2v(
        self,
        prompts: Sequence[str],
        steps: int = 6,
        device: torch.device | None = None,
        seed: int = 0,
    ) -> Tensor:
        device = device or next(self.parameters()).device
        shape = (
            len(prompts),
            self.config.latent_channels,
            math.ceil(self.config.frames / self.config.temporal_downsample),
            math.ceil(self.config.height / self.config.spatial_downsample),
            math.ceil(self.config.width / self.config.spatial_downsample),
        )
        generator = torch.Generator(device=device).manual_seed(seed)
        latents = torch.randn(shape, generator=generator, device=device)
        context = self.encode_text(prompts, device)
        for current, next_time in zip(
            torch.linspace(1, 0, steps + 1, device=device)[:-1],
            torch.linspace(1, 0, steps + 1, device=device)[1:],
        ):
            velocity = self.denoiser(latents, current.expand(len(prompts)), context)
            latents = latents + (next_time - current) * velocity
        return self.vae.decode(
            latents, (self.config.frames, self.config.height, self.config.width)
        )

    @torch.no_grad()
    def sample_i2v(
        self,
        prompts: Sequence[str],
        first_frame: Tensor,
        steps: int = 6,
        device: torch.device | None = None,
        seed: int = 0,
    ) -> Tensor:
        if first_frame.shape[0] != len(prompts):
            raise ValueError("first_frame batch and prompts must have the same length")
        device = device or next(self.parameters()).device
        first_frame = first_frame.to(device)
        condition = self.vae.encode(first_frame[:, :, None])
        batch = len(prompts)
        latent_shape = (
            batch,
            self.config.latent_channels,
            math.ceil(self.config.frames / self.config.temporal_downsample),
            math.ceil(self.config.height / self.config.spatial_downsample),
            math.ceil(self.config.width / self.config.spatial_downsample),
        )
        generator = torch.Generator(device=device).manual_seed(seed)
        latents = torch.randn(latent_shape, generator=generator, device=device)
        context = self.encode_text(prompts, device)
        times = torch.linspace(1, 0, steps + 1, device=device)
        for current, next_time in zip(times[:-1], times[1:]):
            velocity = self.denoiser(latents, current.expand(batch), context)
            latents = latents + (next_time - current) * velocity
            latents[:, :, : condition.shape[2]] = (
                1.0 - next_time
            ) * condition + next_time * latents[:, :, : condition.shape[2]]
        output = self.vae.decode(
            latents, (self.config.frames, self.config.height, self.config.width)
        )
        output[:, :, :1] = first_frame
        return output


def build_model(config: Wan21Config | None = None) -> Wan21Model:
    return Wan21Model(config)


if __name__ == "__main__":
    model = build_model()
    print(f"Wan2.1 tiny parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("latent video layout: [B, 4, ceil(T/2), ceil(H/4), ceil(W/4)]")
