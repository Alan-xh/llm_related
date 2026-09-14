"""Small, readable building blocks for Stable Diffusion teaching models.

The modules in this file deliberately use tiny dimensions and synthetic data.
They are meant to make the tensor flow visible, not to load official
checkpoints or reproduce the production training recipe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
    """Return sinusoidal timestep features with shape ``[B, dim]``."""

    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half, 1)
    )
    angles = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def _group_count(channels: int, requested: int = 32) -> int:
    for groups in range(min(requested, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResBlock2D(nn.Module):
    """A time-conditioned residual block used by the tiny U-Nets."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, time_embedding_value: Tensor) -> Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time_proj(time_embedding_value)[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return self.skip(x) + hidden


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, multiplier: int = 4) -> None:
        super().__init__()
        intermediate = hidden_size * multiplier
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, intermediate),
            nn.GELU(),
            nn.Linear(intermediate, hidden_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Attention(nn.Module):
    """Multi-head attention accepting query and context sequences."""

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 4,
        head_dim: int = 32,
    ) -> None:
        super().__init__()
        context_dim = query_dim if context_dim is None else context_dim
        self.heads = heads
        self.head_dim = head_dim
        inner_dim = heads * head_dim
        self.scale = head_dim**-0.5
        self.norm = nn.LayerNorm(query_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim, bias=False)

    def forward(self, query: Tensor, context: Optional[Tensor] = None) -> Tensor:
        context = query if context is None else context
        query = self.norm(query)
        context = self.context_norm(context)
        batch, query_length, _ = query.shape
        key_length = context.shape[1]
        q = self.to_q(query).view(batch, query_length, self.heads, self.head_dim)
        k = self.to_k(context).view(batch, key_length, self.heads, self.head_dim)
        v = self.to_v(context).view(batch, key_length, self.heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = weights.softmax(dim=-1)
        output = torch.matmul(weights, v).transpose(1, 2).reshape(
            batch, query_length, self.heads * self.head_dim
        )
        return self.to_out(output)


class SpatialTransformer(nn.Module):
    """Self-attention followed by text cross-attention over image features."""

    def __init__(self, channels: int, context_dim: int, heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.proj_in = nn.Conv2d(channels, channels, 1)
        self.self_attn = Attention(channels, heads=heads, head_dim=max(8, channels // heads))
        self.cross_attn = Attention(
            channels,
            context_dim=context_dim,
            heads=heads,
            head_dim=max(8, channels // heads),
        )
        self.ff = FeedForward(channels)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        hidden = self.proj_in(self.norm(x))
        tokens = hidden.flatten(2).transpose(1, 2)
        tokens = tokens + self.self_attn(tokens)
        tokens = tokens + self.cross_attn(tokens, context)
        tokens = tokens + self.ff(tokens)
        hidden = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        return x + self.proj_out(hidden)


class DiagonalGaussianDistribution:
    def __init__(self, parameters: Tensor) -> None:
        self.mean, self.logvar = parameters.chunk(2, dim=1)
        self.logvar = self.logvar.clamp(-30.0, 20.0)

    def sample(self) -> Tensor:
        return self.mean + torch.exp(0.5 * self.logvar) * torch.randn_like(self.mean)

    def mode(self) -> Tensor:
        return self.mean


class TinyVAE(nn.Module):
    """A small convolutional VAE with an 8x spatial downsampling factor."""

    def __init__(self, latent_channels: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, latent_channels * 2, 3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, base_channels * 4, 3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, images: Tensor, sample: bool = False) -> Tensor | DiagonalGaussianDistribution:
        posterior = DiagonalGaussianDistribution(self.encoder(images))
        return posterior.sample() if sample else posterior

    def decode(self, latents: Tensor) -> Tensor:
        return self.decoder(latents)


class ByteTokenizer:
    """Dependency-free tokenizer that keeps the text conditioning inspectable."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    vocab_size = 259

    def encode(self, text: str, max_length: int = 64) -> list[int]:
        ids = [self.bos_token_id]
        ids.extend(byte + 3 for byte in text.encode("utf-8")[: max_length - 2])
        ids.append(self.eos_token_id)
        return ids

    def batch_encode(self, texts: Sequence[str], max_length: int = 64) -> tuple[Tensor, Tensor]:
        encoded = [self.encode(text, max_length) for text in texts]
        input_ids = torch.full(
            (len(encoded), max_length),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for index, ids in enumerate(encoded):
            input_ids[index, : len(ids)] = torch.tensor(ids)
            attention_mask[index, : len(ids)] = True
        return input_ids, attention_mask


class TinyTextEncoder(nn.Module):
    """A tiny CLIP/OpenCLIP-like contextual text encoder."""

    def __init__(
        self,
        hidden_size: int = 64,
        max_length: int = 64,
        layers: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.tokenizer = ByteTokenizer()
        self.token_embedding = nn.Embedding(self.tokenizer.vocab_size, hidden_size)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_length, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_size)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        hidden = self.token_embedding(input_ids) + self.position_embedding[:, : input_ids.shape[1]]
        padding_mask = None if attention_mask is None else ~attention_mask
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        hidden = self.norm(hidden)
        if attention_mask is None:
            pooled = hidden[:, 0]
        else:
            lengths = attention_mask.sum(dim=1).clamp_min(1) - 1
            pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
        return hidden, pooled

    def encode_prompts(
        self,
        prompts: Sequence[str],
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        input_ids, attention_mask = self.tokenizer.batch_encode(prompts, self.max_length)
        return self(input_ids.to(device), attention_mask.to(device))


class DualTextEncoder(nn.Module):
    """SDXL-style pair of text encoders with concatenated token features."""

    def __init__(self, first_dim: int = 64, second_dim: int = 80, max_length: int = 64) -> None:
        super().__init__()
        self.first = TinyTextEncoder(first_dim, max_length=max_length)
        self.second = TinyTextEncoder(second_dim, max_length=max_length)
        self.hidden_size = first_dim + second_dim

    def encode_prompts(
        self,
        prompts: Sequence[str],
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        first_tokens, first_pooled = self.first.encode_prompts(prompts, device)
        second_tokens, second_pooled = self.second.encode_prompts(prompts, device)
        return torch.cat((first_tokens, second_tokens), dim=-1), torch.cat(
            (first_pooled, second_pooled), dim=-1
        )


@dataclass
class DDIMStepOutput:
    prev_sample: Tensor
    pred_original_sample: Tensor


class DDIMScheduler:
    """DDIM scheduler supporting epsilon and v-prediction parameterizations."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        prediction_type: str = "epsilon",
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1)

    def set_timesteps(self, num_inference_steps: int, device: torch.device) -> Tensor:
        self.timesteps = torch.linspace(
            self.num_train_timesteps - 1,
            0,
            num_inference_steps,
            device=device,
        ).round().long()
        return self.timesteps

    def add_noise(self, sample: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        alpha = self.alphas_cumprod.to(sample.device)[timesteps].view(-1, 1, 1, 1)
        return alpha.sqrt() * sample + (1.0 - alpha).sqrt() * noise

    def get_velocity(self, sample: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        alpha = self.alphas_cumprod.to(sample.device)[timesteps].view(-1, 1, 1, 1)
        return alpha.sqrt() * noise - (1.0 - alpha).sqrt() * sample

    def step(self, model_output: Tensor, timestep: Tensor | int, sample: Tensor) -> DDIMStepOutput:
        timestep_value = int(timestep.item()) if isinstance(timestep, Tensor) else int(timestep)
        matches = (self.timesteps == timestep_value).nonzero(as_tuple=False)
        step_index = int(matches[0].item()) if matches.numel() else 0
        previous_timestep = (
            int(self.timesteps[step_index + 1].item())
            if step_index + 1 < len(self.timesteps)
            else -1
        )
        alpha_t = self.alphas_cumprod.to(sample.device)[timestep_value]
        alpha_prev = (
            self.alphas_cumprod.to(sample.device)[previous_timestep]
            if previous_timestep >= 0
            else torch.ones((), device=sample.device)
        )
        if self.prediction_type == "v_prediction":
            pred_original = alpha_t.sqrt() * sample - (1.0 - alpha_t).sqrt() * model_output
            epsilon = alpha_t.sqrt() * model_output + (1.0 - alpha_t).sqrt() * sample
        else:
            pred_original = (sample - (1.0 - alpha_t).sqrt() * model_output) / alpha_t.sqrt()
            epsilon = model_output
        direction = (1.0 - alpha_prev).clamp_min(0).sqrt() * epsilon
        previous = alpha_prev.sqrt() * pred_original + direction
        return DDIMStepOutput(previous, pred_original)


class FlowMatchScheduler:
    """Euler solver for the linear flow-matching path data -> Gaussian noise."""

    def set_timesteps(self, num_inference_steps: int, device: torch.device) -> Tensor:
        return torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)

    def add_noise(self, sample: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        t = timesteps.view(-1, 1, 1, 1)
        return (1.0 - t) * sample + t * noise

    def step(self, velocity: Tensor, timestep: Tensor, sample: Tensor, next_timestep: Tensor) -> Tensor:
        delta = next_timestep - timestep
        return sample + delta * velocity


def classifier_free_guidance(unconditional: Tensor, conditional: Tensor, scale: float) -> Tensor:
    return unconditional + scale * (conditional - unconditional)


def build_toy_images(
    batch_size: int,
    image_size: int,
    device: torch.device,
    step: int = 0,
) -> tuple[Tensor, list[str]]:
    """Create deterministic colored gradients so training needs no dataset."""

    axis = torch.linspace(-1.0, 1.0, image_size, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    images = []
    prompts = []
    names = ("red square", "green circle", "blue diagonal", "yellow grid")
    for index in range(batch_size):
        variant = (index + step) % 4
        if variant == 0:
            image = torch.stack((xx * 0 + 0.8, yy * 0, yy * 0), dim=0)
        elif variant == 1:
            mask = ((xx**2 + yy**2) < 0.5).float()
            image = torch.stack((mask * 0.1, mask * 0.8, mask * 0.2), dim=0)
        elif variant == 2:
            stripe = ((xx + yy) > 0).float()
            image = torch.stack((stripe * 0.1, stripe * 0.3, stripe * 0.9), dim=0)
        else:
            stripe = ((xx * image_size).long() + (yy * image_size).long()) % 2
            image = torch.stack((stripe.float() * 0.8, stripe.float() * 0.7, stripe.float() * 0.1), dim=0)
        images.append(image * 2.0 - 1.0)
        prompts.append(names[variant])
    return torch.stack(images), prompts


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
