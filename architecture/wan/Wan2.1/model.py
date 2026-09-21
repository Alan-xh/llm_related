"""Wan2.1 miniature text-to-video and image-to-video model.

Task:
    Video generation from text (T2V) and from text plus a first frame (I2V).
    This file is a compact, dependency-free educational implementation of the
    Wan2.1-style latent video diffusion architecture.

Architecture:
    CausalVideoVAE -> TinyT5Encoder -> FlowMatchingDiT.  The VAE maps a video
    [B, 3, T, H, W] to latent video [B, C_latent, T_latent, H_latent, W_latent].
    The text encoder produces context [B, L_text, D_text].  The denoiser
    patchifies the latent video into [B, N, D_patch], injects time and position
    embeddings, applies self-attention and text cross-attention, and restores
    the latent video layout.

Core objective:
    With clean latent z0, Gaussian noise z1, and t ~ U(0, 1), construct
        zt = (1 - t) * z0 + t * z1
    and train the denoiser v_theta(zt, t, c) to predict the constant flow
        v*(zt, t, c) = z1 - z0
    using
        L = E[||v_theta(zt, t, c) - (z1 - z0)||^2].

Input/output conventions:
    Video tensors use [B, C, T, H, W], text is a sequence of Python strings,
    and generated videos are returned in the same [B, 3, T, H, W] layout.
    The default latent layout is
    [B, 4, ceil(T / 2), ceil(H / 4), ceil(W / 4)].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class Wan21Config:
    """Configuration for the miniature Wan2.1 model.

    The spatial and temporal downsample factors describe the expected latent
    grid used by sampling.  The actual VAE layers implement the same factors.
    """

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
    """Encode and decode latent videos with causal temporal preprocessing.

    The encoder uses two strided 3-D convolutions.  Before encoding, two
    frames of left-only temporal padding are added so the first output does
    not depend on future input frames.

    Shape:
        encode: [B, 3, T, H, W] -> [B, C_latent, ceil(T / 2),
            ceil(H / 4), ceil(W / 4)]
        decode: [B, C_latent, T_latent, H_latent, W_latent] ->
            [B, 3, T, H, W]
    """

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
        """Encode a video into the latent diffusion space.

        Args:
            video: RGB video, shape [B, 3, T, H, W].

        Returns:
            Latent video, shape [B, C_latent, ceil(T / 2), ceil(H / 4),
            ceil(W / 4)].
        """
        # Left-only temporal padding makes the first latent depend only on
        # frames at or before its receptive field: [B, 3, T, H, W] ->
        # [B, 3, T + 2, H, W].
        hidden = F.pad(video, (0, 0, 0, 0, 2, 0))
        # Strides (1, 2, 2) and (2, 2, 2) produce the latent downsample.
        return self.encoder(hidden)

    def decode(self, latents: Tensor, output_shape: tuple[int, int, int]) -> Tensor:
        """Decode latents and resize to the requested video dimensions.

        Args:
            latents: Latent video, shape [B, C_latent, T_latent, H_latent,
                W_latent].
            output_shape: Target ``(T, H, W)``.

        Returns:
            Reconstructed video, shape [B, 3, T, H, W], with values in
            approximately [-1, 1] because the decoder ends with ``Tanh``.
        """
        hidden = self.decoder(latents)
        # Trilinear interpolation maps the decoder grid to [T, H, W].
        return F.interpolate(
            hidden, size=output_shape, mode="trilinear", align_corners=False
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
        """Encode text strings as padded token IDs.

        Args:
            texts: Batch of prompts.
            max_length: Maximum number of tokens, including BOS and EOS.
            device: Device for the returned tensor.

        Returns:
            Token IDs with shape [B, max_length].
        """
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
    """Dependency-free T5-like text encoder used for cross-attention context.

    Shape:
        token IDs [B, L_text] -> embeddings/context [B, L_text, D_text].
    """

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
        """Tokenize and encode a batch of prompts.

        Args:
            texts: Prompt strings, batch size B.
            device: Device on which token IDs and activations are created.

        Returns:
            Text context, shape [B, L_text, D_text].
        """
        ids = self.tokenizer.batch_encode(texts, self.position.shape[1], device)
        # Embedding lookup plus learned positions: [B, L_text] ->
        # [B, L_text, D_text].
        hidden = self.embedding(ids) + self.position[:, : ids.shape[1]]
        padding = ids.eq(self.tokenizer.pad_token_id)
        return self.norm(self.encoder(hidden, src_key_padding_mask=padding))


def timestep_embedding(timesteps: Tensor, dim: int) -> Tensor:
    """Create sinusoidal time embeddings.

    The implementation follows
        e(t) = [cos(t * f_i), sin(t * f_i)]_i,
        f_i = exp(-log(10000) * i / floor(dim / 2)).

    Args:
        timesteps: Continuous diffusion times, shape [B].
        dim: Output embedding dimension.

    Returns:
        Time embeddings, shape [B, dim].
    """
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half, 1)
    )
    # [B] x [half] -> phase matrix [B, half].
    angles = timesteps.float()[:, None] * frequencies[None]
    # Concatenate cosine and sine bases: [B, half] + [B, half] -> [B, 2*half].
    result = torch.cat((angles.cos(), angles.sin()), dim=-1)
    return F.pad(result, (0, dim - result.shape[-1]))


class VideoPatchifier:
    """Convert a latent video grid to transformer tokens and back.

    For patch size (pt, ph, pw), the token count is
        N = (T / pt) * (H / ph) * (W / pw)
    and each token has dimension D_patch = C * pt * ph * pw.
    """

    def __init__(self, patch_size: tuple[int, int, int]) -> None:
        self.patch_size = patch_size

    def patchify(self, video: Tensor) -> Tensor:
        """Flatten non-overlapping 3-D video patches into tokens.

        Args:
            video: Latent video, shape [B, C, T, H, W].

        Returns:
            Patch tokens, shape [B, N, C * pt * ph * pw].
        """
        batch, channels, frames, height, width = video.shape
        pt, ph, pw = self.patch_size
        if (frames % pt, height % ph, width % pw) != (0, 0, 0):
            raise ValueError("video dimensions must be divisible by patch_size")
        # Split each axis into grid coordinates and within-patch coordinates:
        # [B, C, T, H, W] -> [B, C, T/pt, pt, H/ph, ph, W/pw, pw].
        video = video.reshape(
            batch, channels, frames // pt, pt, height // ph, ph, width // pw, pw
        )
        # Move grid axes before channel and local patch axes:
        # [B, C, T/pt, pt, H/ph, ph, W/pw, pw] ->
        # [B, T/pt, H/ph, W/pw, C, pt, ph, pw].
        video = video.permute(0, 2, 4, 6, 1, 3, 5, 7)
        # Flatten the 3-D patch grid: [B, T/pt, H/ph, W/pw, ...] ->
        # [B, N, C * pt * ph * pw].
        return video.reshape(batch, -1, channels * pt * ph * pw)

    def unpatchify(
        self, tokens: Tensor, channels: int, shape: tuple[int, int, int]
    ) -> Tensor:
        """Restore a [B, N, D_patch] token sequence to a video grid.

        Args:
            tokens: Patch tokens, shape [B, N, channels * pt * ph * pw].
            channels: Number of output channels.
            shape: Target ``(T, H, W)`` before patchification.

        Returns:
            Video tensor, shape [B, channels, T, H, W].
        """
        frames, height, width = shape
        pt, ph, pw = self.patch_size
        grid = (frames // pt, height // ph, width // pw)
        # Recover grid and local patch axes:
        # [B, N, D_patch] -> [B, T/pt, H/ph, W/pw, C, pt, ph, pw].
        video = tokens.reshape(tokens.shape[0], *grid, channels, pt, ph, pw)
        # Inverse of patchify's axis permutation.
        video = video.permute(0, 4, 1, 5, 2, 6, 3, 7)
        # Merge grid and local axes back to [B, C, T, H, W].
        return video.reshape(tokens.shape[0], channels, frames, height, width)


class DiTBlock(nn.Module):
    """Transformer block with latent self-attention and text cross-attention.

    The block implements the residual sequence
        X <- X + SelfAttention(LN(X))
        X <- X + CrossAttention(LN(X), LN(C), LN(C))
        X <- X + FFN(LN(X)).

    Inputs:
        tokens: Latent tokens, shape [B, N, D_hidden].
        context: Text context, shape [B, L_text, D_text].

    Outputs:
        Updated latent tokens, shape [B, N, D_hidden].
    """

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
        """Apply self-attention, text cross-attention, and the feed-forward net."""
        query = self.norm1(tokens)
        # Self-attention: Q, K, V all have shape [B, N, D_hidden].
        tokens = tokens + self.self_attn(query, query, query, need_weights=False)[0]
        # Cross-attention maps text keys/values [B, L_text, D_text] into
        # latent queries [B, N, D_hidden]; output remains [B, N, D_hidden].
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
    """Latent video transformer denoiser trained with flow matching.

    Shape:
        [B, C_latent, T_latent, H_latent, W_latent] ->
        [B, C_latent, T_latent, H_latent, W_latent].
        Internally, the latent grid becomes [B, N, D_hidden].
    """

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
        """Predict the clean-to-noise flow field for a noisy latent video.

        Args:
            latents: Noisy latent video, shape [B, C_latent, T_latent, H_latent,
                W_latent].
            timesteps: Continuous times, shape [B].
            context: Text context, shape [B, L_text, D_text].

        Returns:
            Predicted velocity, shape equal to ``latents``.
        """
        shape = latents.shape[2:]
        # [B, C_latent, T_latent, H_latent, W_latent] -> [B, N, D_patch].
        tokens = self.patchifier.patchify(latents)
        if tokens.shape[1] > self.position.shape[1]:
            raise ValueError("max_tokens is smaller than the video patch grid")
        # Linear patch projection plus positional embedding:
        # [B, N, D_patch] -> [B, N, D_hidden].
        tokens = self.input(tokens) + self.position[:, : tokens.shape[1]]
        # Broadcast time conditioning [B, D_hidden] over all N tokens.
        tokens = (
            tokens
            + self.time(timestep_embedding(timesteps, self.config.hidden_size))[:, None]
        )
        for block in self.blocks:
            tokens = block(tokens, context)
        # [B, N, D_hidden] -> [B, N, D_patch] -> latent video grid.
        return self.patchifier.unpatchify(
            self.output(tokens), self.config.latent_channels, shape
        )


class Wan21Model(nn.Module):
    """Top-level Wan2.1 model exposing training and sampling APIs.

    The VAE and text encoder are frozen in this compact implementation; the
    flow-matching DiT is the trainable denoiser.
    """

    def __init__(self, config: Wan21Config | None = None) -> None:
        super().__init__()
        self.config = config or Wan21Config()
        self.vae = CausalVideoVAE(self.config)
        self.text_encoder = TinyT5Encoder(self.config)
        self.denoiser = FlowMatchingDiT(self.config)
        for parameter in (*self.vae.parameters(), *self.text_encoder.parameters()):
            parameter.requires_grad_(False)

    def encode_text(self, prompts: Sequence[str], device: torch.device) -> Tensor:
        """Encode prompts to cross-attention context [B, L_text, D_text]."""
        return self.text_encoder(prompts, device)

    def training_loss(self, videos: Tensor, prompts: Sequence[str]) -> Tensor:
        """Compute the flow-matching objective for a video/prompt batch.

        Args:
            videos: Clean RGB videos, shape [B, 3, T, H, W].
            prompts: B text prompts.

        Returns:
            Scalar MSE loss for predicting ``noise - clean``.
        """
        with torch.no_grad():
            # [B, 3, T, H, W] -> [B, C_latent, T_latent, H_latent, W_latent].
            clean = self.vae.encode(videos)
            context = self.encode_text(prompts, videos.device)
        timesteps = torch.rand(videos.shape[0], device=videos.device)
        noise = torch.randn_like(clean)
        # t is reshaped to [B, 1, 1, 1, 1] for broadcasting over the latent grid.
        noisy = (1.0 - timesteps[:, None, None, None, None]) * clean + timesteps[
            :, None, None, None, None
        ] * noise
        # Flow-matching target: v* = dzt/dt = noise - clean.
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
        """Generate a video from text with Euler integration.

        Args:
            prompts: B text prompts.
            steps: Number of Euler steps from t=1 to t=0.
            device: Optional execution device.
            seed: Random seed for the initial latent noise.

        Returns:
            Generated videos, shape [B, 3, T, H, W].
        """
        device = device or next(self.parameters()).device
        shape = (
            len(prompts),
            self.config.latent_channels,
            math.ceil(self.config.frames / self.config.temporal_downsample),
            math.ceil(self.config.height / self.config.spatial_downsample),
            math.ceil(self.config.width / self.config.spatial_downsample),
        )
        generator = torch.Generator(device=device).manual_seed(seed)
        # Initial point z(1) ~ N(0, I), shape [B, C_latent, T_latent, H_latent, W_latent].
        latents = torch.randn(shape, generator=generator, device=device)
        context = self.encode_text(prompts, device)
        for current, next_time in zip(
            torch.linspace(1, 0, steps + 1, device=device)[:-1],
            torch.linspace(1, 0, steps + 1, device=device)[1:],
        ):
            # Euler update for dz/dt = v_theta(z, t, c):
            # z_next = z + (t_next - t) * v_theta(z, t, c).
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
        """Generate a video conditioned on a first RGB frame and text.

        Args:
            prompts: B text prompts.
            first_frame: First RGB frame, shape [B, 3, H, W].
            steps: Number of Euler steps.
            device: Optional execution device.
            seed: Random seed for the initial latent noise.

        Returns:
            Generated videos, shape [B, 3, T, H, W], with the first frame
            replaced by the provided conditioning frame.
        """
        if first_frame.shape[0] != len(prompts):
            raise ValueError("first_frame batch and prompts must have the same length")
        device = device or next(self.parameters()).device
        first_frame = first_frame.to(device)
        # Add a singleton time axis: [B, 3, H, W] -> [B, 3, 1, H, W].
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
        # Start from Gaussian latent noise and progressively impose the frame.
        latents = torch.randn(latent_shape, generator=generator, device=device)
        context = self.encode_text(prompts, device)
        times = torch.linspace(1, 0, steps + 1, device=device)
        for current, next_time in zip(times[:-1], times[1:]):
            velocity = self.denoiser(latents, current.expand(batch), context)
            latents = latents + (next_time - current) * velocity
            # Blend the known latent prefix with the current denoised estimate.
            latents[:, :, : condition.shape[2]] = (
                1.0 - next_time
            ) * condition + next_time * latents[:, :, : condition.shape[2]]
        # Decode the latent video and enforce the exact input frame at t=0.
        output = self.vae.decode(
            latents, (self.config.frames, self.config.height, self.config.width)
        )
        output[:, :, :1] = first_frame
        return output


def build_model(config: Wan21Config | None = None) -> Wan21Model:
    """Build a Wan2.1 model from an optional configuration."""
    return Wan21Model(config)


if __name__ == "__main__":
    model = build_model()
    print(f"Wan2.1 tiny parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("latent video layout: [B, 4, ceil(T/2), ceil(H/4), ceil(W/4)]")
