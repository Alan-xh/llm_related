"""Wan2.2 miniature text-image-to-video model with noise-stage MoE routing.

Task:
    Video generation from text plus an optional first-frame condition (TI2V).
    This file is a compact educational implementation of a Wan2.2-style
    high-compression latent video diffusion architecture.

Architecture:
    HighCompressionVideoVAE -> TinyTextEncoder -> MoEDiffusionTransformer.
    The VAE maps [B, 3, T, H, W] to [B, C_latent, T_latent, H_latent, W_latent].
    Text is encoded as context [B, L_text, D_text].  The denoiser patchifies
    latent videos, adds sinusoidal time and learned position embeddings, then
    blends high-noise and low-noise transformer experts.

Core objective:
    For clean latent z0, Gaussian noise z1, and t ~ U(0, 1):
        zt = (1 - t) * z0 + t * z1
        v*(zt, t, c) = z1 - z0
        L = E[||v_theta(zt, t, c) - (z1 - z0)||^2].
    The MoE gate is
        g = 0.5 * 1[t >= noise_switch] + 0.5 * sigmoid(router(time)),
    followed by
        h = g * h_high + (1 - g) * h_low.

Input/output conventions:
    Videos use [B, C, T, H, W], first-frame conditions use [B, 3, H, W],
    text is a sequence of Python strings, and generated videos use
    [B, 3, T, H, W].  The default latent layout is
    [B, 4, ceil(T / 2), ceil(H / 8), ceil(W / 8)].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class Wan22Config:
    """Configuration for the miniature Wan2.2 model and its MoE router."""

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
    """VAE-like encoder/decoder with 8x spatial and 2x temporal compression.

    Shape:
        encode: [B, 3, T, H, W] -> [B, C_latent, ceil(T / 2),
            ceil(H / 8), ceil(W / 8)]
        decode: [B, C_latent, T_latent, H_latent, W_latent] ->
            [B, 3, T, H, W]
    """

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
        """Encode an RGB video into the compressed latent grid.

        Args:
            video: RGB video, shape [B, 3, T, H, W].

        Returns:
            Latent video, shape [B, C_latent, ceil(T / 2), ceil(H / 8),
            ceil(W / 8)].
        """
        # The three convolution strides yield temporal factor 2 and spatial
        # factor 2 * 2 * 2 = 8.
        return self.encoder(video)

    def decode(self, latents: Tensor, output_shape: tuple[int, int, int]) -> Tensor:
        """Decode latents and resize to a requested ``(T, H, W)`` shape.

        Args:
            latents: Latent video, shape [B, C_latent, T_latent, H_latent,
                W_latent].
            output_shape: Target temporal and spatial dimensions.

        Returns:
            Reconstructed RGB video, shape [B, 3, T, H, W].
        """
        # Decoder convolutions preserve the latent grid; interpolation restores
        # the requested video grid.
        return F.interpolate(
            self.decoder(latents),
            size=output_shape,
            mode="trilinear",
            align_corners=False,
        )


class ByteTokenizer:
    """Minimal UTF-8 byte tokenizer used by the local text encoder."""

    vocab_size = 259
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def batch_encode(
        self, texts: Sequence[str], max_length: int, device: torch.device
    ) -> Tensor:
        """Encode and pad a batch of prompts.

        Args:
            texts: Prompt strings.
            max_length: Maximum sequence length including BOS/EOS.
            device: Target device.

        Returns:
            Token IDs, shape [B, max_length].
        """
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
    """Dependency-free Transformer text encoder for cross-attention conditioning.

    Shape:
        token IDs [B, L_text] -> context [B, L_text, D_text].
    """

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
        """Encode prompts into text context.

        Args:
            prompts: Batch of prompt strings.
            device: Device for token IDs and activations.

        Returns:
            Context tensor, shape [B, L_text, D_text].
        """
        ids = self.tokenizer.batch_encode(prompts, self.position.shape[1], device)
        # Token embedding plus learned positional embedding:
        # [B, L_text] -> [B, L_text, D_text].
        hidden = self.embedding(ids) + self.position[:, : ids.shape[1]]
        # Padding mask has shape [B, L_text], True means ignored by attention.
        return self.norm(self.encoder(hidden, src_key_padding_mask=ids.eq(0)))


def timestep_embedding(timesteps: Tensor, dim: int) -> Tensor:
    """Create sinusoidal embeddings for continuous diffusion time.

    Formula:
        e(t) = [cos(t * f_i), sin(t * f_i)]_i
        f_i = exp(-log(10000) * i / floor(dim / 2))

    Args:
        timesteps: Diffusion times, shape [B].
        dim: Output embedding dimension.

    Returns:
        Time embedding, shape [B, dim].
    """
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half, 1)
    )
    # [B] x [half] -> [B, half] phase values.
    angles = timesteps.float()[:, None] * frequencies[None]
    # [B, half] + [B, half] -> [B, 2*half], then pad to [B, dim].
    values = torch.cat((angles.cos(), angles.sin()), dim=-1)
    return F.pad(values, (0, dim - values.shape[-1]))


class VideoPatchifier:
    """Pack a latent video into non-overlapping 3-D transformer patches.

    For patch size (pt, ph, pw):
        N = (T / pt) * (H / ph) * (W / pw)
        D_patch = C * pt * ph * pw.
    """

    def __init__(self, patch_size: tuple[int, int, int]) -> None:
        self.patch_size = patch_size

    def patchify(self, video: Tensor) -> Tensor:
        """Convert [B, C, T, H, W] into [B, N, D_patch] tokens."""
        batch, channels, frames, height, width = video.shape
        pt, ph, pw = self.patch_size
        if frames % pt or height % ph or width % pw:
            raise ValueError("video dimensions must be divisible by patch_size")
        # Split each video axis into grid and local patch coordinates:
        # [B, C, T, H, W] -> [B, C, T/pt, pt, H/ph, ph, W/pw, pw].
        hidden = video.reshape(
            batch, channels, frames // pt, pt, height // ph, ph, width // pw, pw
        )
        # Put grid coordinates before channels and local coordinates.
        hidden = hidden.permute(0, 2, 4, 6, 1, 3, 5, 7)
        # Flatten the grid and patch contents:
        # [B, T/pt, H/ph, W/pw, C, pt, ph, pw] -> [B, N, D_patch].
        return hidden.reshape(batch, -1, channels * pt * ph * pw)

    def unpatchify(
        self, tokens: Tensor, channels: int, shape: tuple[int, int, int]
    ) -> Tensor:
        """Convert [B, N, D_patch] tokens back to [B, C, T, H, W]."""
        frames, height, width = shape
        pt, ph, pw = self.patch_size
        grid = (frames // pt, height // ph, width // pw)
        # Recover grid/local axes before reversing the permutation.
        hidden = tokens.reshape(tokens.shape[0], *grid, channels, pt, ph, pw)
        hidden = hidden.permute(0, 4, 1, 5, 2, 6, 3, 7)
        # Merge temporal and spatial grid axes with their local patch axes.
        return hidden.reshape(tokens.shape[0], channels, frames, height, width)


class ExpertBlock(nn.Module):
    """Transformer block shared by each noise-stage expert.

    Inputs:
        tokens: Latent tokens, shape [B, N, D_hidden].
        context: Text context, shape [B, L_text, D_text].

    Outputs:
        Updated tokens, shape [B, N, D_hidden].

    The residual equations are:
        X <- X + SelfAttention(LN(X))
        X <- X + CrossAttention(LN(X), LN(C), LN(C))
        X <- X + FFN(LN(X)).
    """

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
        """Apply self-attention, text cross-attention, and FFN residuals."""
        query = self.norm(tokens)
        # Self-attention keeps the token shape [B, N, D_hidden].
        tokens = tokens + self.self_attn(query, query, query, need_weights=False)[0]
        text = self.text_norm(context)
        # Text keys/values [B, L_text, D_text] condition latent queries
        # [B, N, D_hidden]; cross-attention output is [B, N, D_hidden].
        tokens = (
            tokens
            + self.cross_attn(self.norm(tokens), text, text, need_weights=False)[0]
        )
        return tokens + self.ff(self.ff_norm(tokens))


class NoiseStageExpert(nn.Module):
    """Stack of transformer blocks specialized for one noise regime."""

    def __init__(self, config: Wan22Config) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ExpertBlock(config.hidden_size, config.heads, config.text_dim)
                for _ in range(config.layers)
            ]
        )

    def forward(self, tokens: Tensor, context: Tensor) -> Tensor:
        """Run all blocks while preserving [B, N, D_hidden]."""
        for block in self.blocks:
            tokens = block(tokens, context)
        return tokens


class MoEDiffusionTransformer(nn.Module):
    """Flow-matching denoiser with high-noise and low-noise experts.

    Inputs:
        latents: Noisy latent video, shape [B, C_latent, T_latent, H_latent,
            W_latent].
        timesteps: Continuous diffusion times, shape [B].
        context: Text context, shape [B, L_text, D_text].
        condition: Optional latent first-frame condition, broadcast/padded to
            the latent shape [B, C_latent, T_latent, H_latent, W_latent].

    Outputs:
        Predicted flow, shape equal to ``latents``.
    """

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
        """Predict flow and blend the two noise-stage experts.

        The deterministic stage gate and learned gate are combined as
            g = 0.5 * 1[t >= s] + 0.5 * sigmoid(router(time)),
        where ``s`` is ``noise_switch``.
        """
        shape = latents.shape[2:]
        if condition is not None:
            # Condition injection preserves [B, C_latent, T_latent, H_latent,
            # W_latent].
            latents = latents + 0.5 * condition
        # Latent grid -> patch tokens [B, C, T, H, W] -> [B, N, D_patch].
        tokens = self.patchifier.patchify(latents)
        if tokens.shape[1] > self.position.shape[1]:
            raise ValueError("max_tokens is smaller than the video patch grid")
        # [B] -> [B, D_hidden], then broadcast over N tokens.
        time = self.time(timestep_embedding(timesteps, self.config.hidden_size))
        tokens = (
            self.input(tokens) + self.position[:, : tokens.shape[1]] + time[:, None]
        )
        # Both experts preserve [B, N, D_hidden].
        high = self.high_noise_expert(tokens, context)
        low = self.low_noise_expert(tokens, context)
        # learned_gate and stage_gate both become [B, 1, 1] for token-wise
        # interpolation; g is the explicit MoE mixing coefficient.
        learned_gate = self.router(time).sigmoid()[:, None, None]
        stage_gate = (timesteps >= self.config.noise_switch).float()[:, None, None]
        gate = 0.5 * stage_gate + 0.5 * learned_gate
        mixed = gate * high + (1.0 - gate) * low
        # [B, N, D_hidden] -> [B, N, D_patch] -> latent video grid.
        return self.patchifier.unpatchify(
            self.output(mixed), self.config.latent_channels, shape
        )


class Wan22Model(nn.Module):
    """Top-level Wan2.2 model exposing training and TI2V sampling APIs.

    The VAE and text encoder are frozen; the MoE diffusion transformer remains
    trainable for the compact training setup represented here.
    """

    def __init__(self, config: Wan22Config | None = None) -> None:
        super().__init__()
        self.config = config or Wan22Config()
        self.vae = HighCompressionVideoVAE(self.config)
        self.text_encoder = TinyTextEncoder(self.config)
        self.denoiser = MoEDiffusionTransformer(self.config)
        for parameter in (*self.vae.parameters(), *self.text_encoder.parameters()):
            parameter.requires_grad_(False)

    def encode_text(self, prompts: Sequence[str], device: torch.device) -> Tensor:
        """Encode prompts to context with shape [B, L_text, D_text]."""
        return self.text_encoder(prompts, device)

    def training_loss(
        self,
        videos: Tensor,
        prompts: Sequence[str],
        first_frame: Tensor | None = None,
    ) -> Tensor:
        """Compute flow-matching loss with optional first-frame conditioning.

        Args:
            videos: Clean RGB videos, shape [B, 3, T, H, W].
            prompts: B text prompts.
            first_frame: Optional RGB frame, shape [B, 3, H, W].

        Returns:
            Scalar MSE between predicted flow and ``noise - clean``.
        """
        with torch.no_grad():
            # [B, 3, T, H, W] -> [B, C_latent, T_latent, H_latent, W_latent].
            clean = self.vae.encode(videos)
            context = self.encode_text(prompts, videos.device)
            condition = None
            if first_frame is not None:
                # [B, 3, H, W] -> [B, 3, 1, H, W] -> frame latent prefix.
                condition = self.vae.encode(first_frame[:, :, None])
                # Temporal pad aligns the condition with the full latent grid.
                condition = F.pad(
                    condition, (0, 0, 0, 0, 0, clean.shape[2] - condition.shape[2])
                )
        timesteps = torch.rand(videos.shape[0], device=videos.device)
        noise = torch.randn_like(clean)
        # Broadcast t from [B] to [B, 1, 1, 1, 1] over the latent video.
        view = timesteps[:, None, None, None, None]
        noisy = (1.0 - view) * clean + view * noise
        # Flow-matching target is v* = dzt/dt = noise - clean.
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
        """Generate a video from text and a provided first frame.

        Args:
            prompts: B text prompts.
            first_frame: Conditioning frame, shape [B, 3, H, W].
            steps: Number of Euler integration steps.
            device: Optional execution device.
            seed: Random seed for initial latent noise.

        Returns:
            Generated videos, shape [B, 3, T, H, W], with the first frame
            replaced by the exact input frame.
        """
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
        # Encode the first frame with a singleton time axis:
        # [B, 3, H, W] -> [B, 3, 1, H, W] -> [B, C_latent, 1, H_latent, W_latent].
        condition_frame = self.vae.encode(first_frame[:, :, None])
        # Pad the known prefix to [B, C_latent, T_latent, H_latent, W_latent].
        condition = F.pad(
            condition_frame, (0, 0, 0, 0, 0, latent_shape[2] - condition_frame.shape[2])
        )
        generator = torch.Generator(device=device).manual_seed(seed)
        latents = torch.randn(latent_shape, generator=generator, device=device)
        context = self.encode_text(prompts, device)
        times = torch.linspace(1, 0, steps + 1, device=device)
        for current, next_time in zip(times[:-1], times[1:]):
            # Euler step: z_next = z + (t_next - t) * v_theta(z, t, c).
            velocity = self.denoiser(latents, current.expand(batch), context, condition)
            latents = latents + (next_time - current) * velocity
            # Preserve the known latent prefix while denoising the remaining
            # temporal positions.
            latents[:, :, : condition_frame.shape[2]] = (
                1.0 - next_time
            ) * condition_frame + next_time * latents[:, :, : condition_frame.shape[2]]
        # Decode and restore the exact RGB conditioning frame.
        video = self.vae.decode(
            latents, (self.config.frames, self.config.height, self.config.width)
        )
        video[:, :, :1] = first_frame
        return video


def build_model(config: Wan22Config | None = None) -> Wan22Model:
    """Build a Wan2.2 model from an optional configuration."""
    return Wan22Model(config)


if __name__ == "__main__":
    model = build_model()
    print(f"Wan2.2 tiny parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("MoE routing: high-noise expert for t >= 0.5, low-noise expert otherwise")
