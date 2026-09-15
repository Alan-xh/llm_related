from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class Wan22Config:
    frames: int = 4
    height: int = 32
    width: int = 32
    latent_channels: int = 4
    temporal_downsample: int = 2
    spatial_downsample: int = 8
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 64
    hidden_size: int = 96
    heads: int = 4
    layers: int = 2
    max_text_length: int = 64
    max_tokens: int = 512
    noise_switch: float = 0.5


class HighCompressionVideoVAE(nn.Module):
    """Tiny VAE-like module with a higher spatial compression ratio."""

    def __init__(self, config: Wan22Config) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 32, 3, stride=(1, 2, 2), padding=1),
            nn.SiLU(),
            nn.Conv3d(32, 64, 3, stride=(2, 2, 2), padding=1),
            nn.SiLU(),
            nn.Conv3d(64, config.latent_channels, 3, stride=(1, 2, 2), padding=1),
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
        return self.encoder(video)

    def decode(self, latents: Tensor, output_shape: tuple[int, int, int]) -> Tensor:
        return F.interpolate(
            self.decoder(latents),
            size=output_shape,
            mode="trilinear",
            align_corners=False,
        )


class ByteTokenizer:
    vocab_size = 259
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def batch_encode(
        self, texts: Sequence[str], max_length: int, device: torch.device
    ) -> Tensor:
        ids = torch.full(
            (len(texts), max_length), self.pad_token_id, dtype=torch.long, device=device
        )
        for row, text in enumerate(texts):
            values = [self.bos_token_id]
            values.extend(byte + 3 for byte in text.encode("utf-8")[: max_length - 2])
            values.append(self.eos_token_id)
            ids[row, : len(values)] = torch.tensor(values, device=device)
        return ids


class TinyTextEncoder(nn.Module):
    def __init__(self, config: Wan22Config) -> None:
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

    def forward(self, prompts: Sequence[str], device: torch.device) -> Tensor:
        ids = self.tokenizer.batch_encode(prompts, self.position.shape[1], device)
        hidden = self.embedding(ids) + self.position[:, : ids.shape[1]]
        return self.norm(self.encoder(hidden, src_key_padding_mask=ids.eq(0)))


def timestep_embedding(timesteps: Tensor, dim: int) -> Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half, 1)
    )
    angles = timesteps.float()[:, None] * frequencies[None]
    values = torch.cat((angles.cos(), angles.sin()), dim=-1)
    return F.pad(values, (0, dim - values.shape[-1]))


class VideoPatchifier:
    def __init__(self, patch_size: tuple[int, int, int]) -> None:
        self.patch_size = patch_size

    def patchify(self, video: Tensor) -> Tensor:
        batch, channels, frames, height, width = video.shape
        pt, ph, pw = self.patch_size
        if frames % pt or height % ph or width % pw:
            raise ValueError("video dimensions must be divisible by patch_size")
        hidden = video.reshape(
            batch, channels, frames // pt, pt, height // ph, ph, width // pw, pw
        )
        hidden = hidden.permute(0, 2, 4, 6, 1, 3, 5, 7)
        return hidden.reshape(batch, -1, channels * pt * ph * pw)

    def unpatchify(
        self, tokens: Tensor, channels: int, shape: tuple[int, int, int]
    ) -> Tensor:
        frames, height, width = shape
        pt, ph, pw = self.patch_size
        grid = (frames // pt, height // ph, width // pw)
        hidden = tokens.reshape(tokens.shape[0], *grid, channels, pt, ph, pw)
        hidden = hidden.permute(0, 4, 1, 5, 2, 6, 3, 7)
        return hidden.reshape(tokens.shape[0], channels, frames, height, width)


class ExpertBlock(nn.Module):
    def __init__(self, hidden_size: int, heads: int, text_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(hidden_size, heads, batch_first=True)
        self.text_norm = nn.LayerNorm(text_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, heads, batch_first=True, kdim=text_dim, vdim=text_dim
        )
        self.ff_norm = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, tokens: Tensor, context: Tensor) -> Tensor:
        query = self.norm(tokens)
        tokens = tokens + self.self_attn(query, query, query, need_weights=False)[0]
        text = self.text_norm(context)
        tokens = (
            tokens
            + self.cross_attn(self.norm(tokens), text, text, need_weights=False)[0]
        )
        return tokens + self.ff(self.ff_norm(tokens))


class NoiseStageExpert(nn.Module):
    def __init__(self, config: Wan22Config) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ExpertBlock(config.hidden_size, config.heads, config.text_dim)
                for _ in range(config.layers)
            ]
        )

    def forward(self, tokens: Tensor, context: Tensor) -> Tensor:
        for block in self.blocks:
            tokens = block(tokens, context)
        return tokens


class MoEDiffusionTransformer(nn.Module):
    def __init__(self, config: Wan22Config) -> None:
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
        self.high_noise_expert = NoiseStageExpert(config)
        self.low_noise_expert = NoiseStageExpert(config)
        self.router = nn.Linear(config.hidden_size, 1)
        self.output = nn.Sequential(
            nn.LayerNorm(config.hidden_size), nn.Linear(config.hidden_size, patch_dim)
        )
        nn.init.normal_(self.position, std=0.02)

    def forward(
        self,
        latents: Tensor,
        timesteps: Tensor,
        context: Tensor,
        condition: Tensor | None = None,
    ) -> Tensor:
        shape = latents.shape[2:]
        if condition is not None:
            latents = latents + 0.5 * condition
        tokens = self.patchifier.patchify(latents)
        if tokens.shape[1] > self.position.shape[1]:
            raise ValueError("max_tokens is smaller than the video patch grid")
        time = self.time(timestep_embedding(timesteps, self.config.hidden_size))
        tokens = (
            self.input(tokens) + self.position[:, : tokens.shape[1]] + time[:, None]
        )
        high = self.high_noise_expert(tokens, context)
        low = self.low_noise_expert(tokens, context)
        learned_gate = self.router(time).sigmoid()[:, None, None]
        stage_gate = (timesteps >= self.config.noise_switch).float()[:, None, None]
        gate = 0.5 * stage_gate + 0.5 * learned_gate
        mixed = gate * high + (1.0 - gate) * low
        return self.patchifier.unpatchify(
            self.output(mixed), self.config.latent_channels, shape
        )


class Wan22Model(nn.Module):
    def __init__(self, config: Wan22Config | None = None) -> None:
        super().__init__()
        self.config = config or Wan22Config()
        self.vae = HighCompressionVideoVAE(self.config)
        self.text_encoder = TinyTextEncoder(self.config)
        self.denoiser = MoEDiffusionTransformer(self.config)
        for parameter in (*self.vae.parameters(), *self.text_encoder.parameters()):
            parameter.requires_grad_(False)

    def encode_text(self, prompts: Sequence[str], device: torch.device) -> Tensor:
        return self.text_encoder(prompts, device)

    def training_loss(
        self,
        videos: Tensor,
        prompts: Sequence[str],
        first_frame: Tensor | None = None,
    ) -> Tensor:
        with torch.no_grad():
            clean = self.vae.encode(videos)
            context = self.encode_text(prompts, videos.device)
            condition = None
            if first_frame is not None:
                condition = self.vae.encode(first_frame[:, :, None])
                condition = F.pad(
                    condition, (0, 0, 0, 0, 0, clean.shape[2] - condition.shape[2])
                )
        timesteps = torch.rand(videos.shape[0], device=videos.device)
        noise = torch.randn_like(clean)
        view = timesteps[:, None, None, None, None]
        noisy = (1.0 - view) * clean + view * noise
        prediction = self.denoiser(noisy, timesteps, context, condition)
        return F.mse_loss(prediction.float(), (noise - clean).float())

    @torch.no_grad()
    def sample_ti2v(
        self,
        prompts: Sequence[str],
        first_frame: Tensor,
        steps: int = 6,
        device: torch.device | None = None,
        seed: int = 0,
    ) -> Tensor:
        device = device or next(self.parameters()).device
        first_frame = first_frame.to(device)
        batch = len(prompts)
        latent_shape = (
            batch,
            self.config.latent_channels,
            math.ceil(self.config.frames / self.config.temporal_downsample),
            math.ceil(self.config.height / self.config.spatial_downsample),
            math.ceil(self.config.width / self.config.spatial_downsample),
        )
        condition_frame = self.vae.encode(first_frame[:, :, None])
        condition = F.pad(
            condition_frame, (0, 0, 0, 0, 0, latent_shape[2] - condition_frame.shape[2])
        )
        generator = torch.Generator(device=device).manual_seed(seed)
        latents = torch.randn(latent_shape, generator=generator, device=device)
        context = self.encode_text(prompts, device)
        times = torch.linspace(1, 0, steps + 1, device=device)
        for current, next_time in zip(times[:-1], times[1:]):
            velocity = self.denoiser(latents, current.expand(batch), context, condition)
            latents = latents + (next_time - current) * velocity
            latents[:, :, : condition_frame.shape[2]] = (
                1.0 - next_time
            ) * condition_frame + next_time * latents[:, :, : condition_frame.shape[2]]
        video = self.vae.decode(
            latents, (self.config.frames, self.config.height, self.config.width)
        )
        video[:, :, :1] = first_frame
        return video


def build_model(config: Wan22Config | None = None) -> Wan22Model:
    return Wan22Model(config)


if __name__ == "__main__":
    model = build_model()
    print(f"Wan2.2 tiny parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("MoE routing: high-noise expert for t >= 0.5, low-noise expert otherwise")
