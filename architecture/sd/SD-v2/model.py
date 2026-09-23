"""SD-v2 风格文本/深度条件潜空间扩散模型的精简教学实现。

任务定义：使用文本提示和可选深度图控制图像生成。
代表架构：Stable Diffusion v2 风格 U-Net，使用 v-prediction 参数化。
核心流程：VAE 将图像压缩为潜变量，深度特征与带噪潜变量按通道拼接，
U-Net 结合时间与文本条件预测速度，DDIM 采样后由 VAE 解码。
目标函数：L=MSE(v_theta(z_t,t,c,d),v)，其中
z_t=sqrt(alpha_bar_t)z0+sqrt(1-alpha_bar_t)epsilon，
v=sqrt(alpha_bar_t)epsilon-sqrt(1-alpha_bar_t)z0。
形状：图像/潜变量 [B,C,H,W]，深度 [B,depth_channels,h,w]，
文本上下文 [B,T,text_dim]，输出图像 [B,3,H,W]。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

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
class SDv2Config:
    """SD-v2 教学模型配置，包含深度条件通道和 v-prediction 设置。"""

    image_size: int = 32
    latent_channels: int = 4
    depth_channels: int = 4
    base_channels: int = 32
    text_dim: int = 80
    time_dim: int = 128
    prediction_type: str = "v_prediction"
    latent_scaling: float = 0.18215


class SDv2UNet(nn.Module):
    """支持文本交叉注意力与可选深度输入的 U-Net 去噪器。

    latents/depth 分别为 [B,C,h,w]/[B,D,h,w]，context 为 [B,T,text_dim]，
    timesteps 为 [B]；返回与 latents 同形的噪声或速度预测。
    """

    def __init__(self, config: SDv2Config) -> None:
        super().__init__()
        channels = config.base_channels
        self.config = config
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_dim, config.time_dim * 4),
            nn.SiLU(),
            nn.Linear(config.time_dim * 4, config.time_dim),
        )
        self.in_conv = nn.Conv2d(config.latent_channels + config.depth_channels, channels, 3, padding=1)
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
        depth: Optional[Tensor] = None,
    ) -> Tensor:
        """融合潜变量和深度条件，输入输出空间尺寸保持为 [B,C,h,w]。"""
        if depth is None:
            # 未提供深度时使用同设备、同精度的零条件，形状为 [B,D,h,w]。
            depth = torch.zeros(
                latents.shape[0],
                self.config.depth_channels,
                latents.shape[2],
                latents.shape[3],
                device=latents.device,
                dtype=latents.dtype,
            )
        if timesteps.ndim == 0:
            timesteps = timesteps[None].expand(latents.shape[0])
        time = self.time_mlp(timestep_embedding(timesteps, self.config.time_dim))
        skip = self.attn1(
            # 沿通道拼接潜变量与深度：[B,C+D,h,w]。
            self.down1(self.in_conv(torch.cat((latents, depth), dim=1)), time),
            context,
        )
        hidden = self.downsample(skip)  # [B,C,h,w] -> [B,2C,ceil(h/2),ceil(w/2)]
        hidden = self.attn2(self.down2(hidden, time), context)
        hidden = self.mid_attn(self.mid(hidden, time), context)
        hidden = self.upsample(hidden)  # 上采样回 skip 尺度：[B,2C,h/2,w/2] -> [B,C,h,w]
        hidden = self.up(torch.cat((hidden, skip), dim=1), time)
        hidden = self.up_attn(hidden, context)
        return self.out_conv(F.silu(self.out_norm(hidden)))


class SDv2Model(nn.Module):
    """封装 SD-v2 风格的 VAE、文本条件、深度条件去噪和 DDIM 采样。

    训练图像 [B,3,H,W]；可选深度 [B,D,H/8,W/8]；生成图像 [B,3,H,W]。
    """

    def __init__(self, config: SDv2Config | None = None) -> None:
        """初始化模型并冻结 VAE 与文本编码器。"""
        super().__init__()
        self.config = config or SDv2Config()
        self.vae = TinyVAE(self.config.latent_channels)
        self.text_encoder = TinyTextEncoder(self.config.text_dim)
        self.denoiser = SDv2UNet(self.config)
        self.scheduler = DDIMScheduler(prediction_type=self.config.prediction_type)
        for parameter in (*self.vae.parameters(), *self.text_encoder.parameters()):
            parameter.requires_grad_(False)

    def encode_prompts(self, prompts: Sequence[str], device: torch.device) -> Tensor:
        """把 B 条提示编码为 [B,T,text_dim] 上下文特征。"""
        context, _ = self.text_encoder.encode_prompts(prompts, device)
        return context

    def training_loss(
        self,
        images: Tensor,
        prompts: Sequence[str],
        depth: Optional[Tensor] = None,
    ) -> Tensor:
        """计算 v-prediction 训练损失；返回对批次取均值后的标量 MSE。"""
        with torch.no_grad():
            latents = self.vae.encode(images).mode() * self.config.latent_scaling  # [B,3,H,W] -> [B,C,H/8,W/8]
            context = self.encode_prompts(prompts, images.device)
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
        )
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        prediction = self.denoiser(noisy_latents, timesteps, context, depth)
        target = self.scheduler.get_velocity(latents, noise, timesteps)
        return F.mse_loss(prediction.float(), target.float())

    @torch.no_grad()
    def sample(
        self,
        prompts: Sequence[str],
        num_inference_steps: int = 8,
        guidance_scale: float = 5.0,
        depth: Optional[Tensor] = None,
        device: torch.device | None = None,
        seed: int = 0,
    ) -> Tensor:
        """执行带可选深度条件的 DDIM 采样，返回范围 [0,1] 的 [B,3,H,W] 图像。

        深度条件应与生成潜变量具有相同批次和空间尺寸。
        """
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
        if depth is not None:
            depth = depth.to(device)
        for timestep in self.scheduler.set_timesteps(num_inference_steps, device):
            timesteps = timestep.expand(len(prompts))
            unconditional_prediction = self.denoiser(latents, timesteps, unconditional, depth)
            conditional_prediction = self.denoiser(latents, timesteps, context, depth)
            prediction = classifier_free_guidance(
                unconditional_prediction, conditional_prediction, guidance_scale
            )
            latents = self.scheduler.step(prediction, timestep, latents).prev_sample
        images = self.vae.decode(latents / self.config.latent_scaling)
        return (images.clamp(-1.0, 1.0) + 1.0) / 2.0


def build_model() -> SDv2Model:
    """按默认配置构建 SD-v2 教学模型。"""
    return SDv2Model()


if __name__ == "__main__":
    model = build_model()
    print(f"SD-v2 tiny parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("prediction target: v = sqrt(alpha_bar) * noise - sqrt(1 - alpha_bar) * clean_latent")
