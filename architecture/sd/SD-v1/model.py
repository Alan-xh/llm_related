"""SD-v1 风格潜空间扩散模型的精简教学实现。

任务定义：以文本为条件，在 VAE 潜空间中进行图像去噪生成。
代表架构：Stable Diffusion v1，采用潜空间 U-Net 与文本交叉注意力。
核心流程：图像编码为 z0，按 DDPM 前向过程加噪，U-Net 结合时间步和文本上下文预测噪声，
再通过 DDIM 与 classifier-free guidance 逐步采样并解码。
目标函数：epsilon 模式下 L=MSE(epsilon_theta(z_t,t,c),epsilon)；
v 模式下 L=MSE(v_theta(z_t,t,c),sqrt(alpha_bar_t)epsilon-sqrt(1-alpha_bar_t)z0)。
数据形状：图像 [B,3,H,W]，潜变量/预测 [B,4,H/8,W/8]，文本上下文 [B,T,text_dim]。
"""

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
    """SD-v1 教学模型配置；图像尺寸需与 VAE 的 8 倍下采样约定相容。"""

    image_size: int = 32
    latent_channels: int = 4
    base_channels: int = 32
    text_dim: int = 64
    time_dim: int = 128
    prediction_type: str = "epsilon"
    latent_scaling: float = 0.18215


class SDv1UNet(nn.Module):
    """带时间嵌入和文本交叉注意力的潜空间 U-Net 去噪器。

    输入潜变量 [B,C,h,w]、时间步 [B] 或标量、文本上下文 [B,T,text_dim]；
    输出噪声/速度预测 [B,C,h,w]。编码器下采样并保存 skip，解码器拼接同尺度特征。
    """

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
        """输入 latents [B,C,h,w]、timesteps [B]、context [B,T,D]，输出 [B,C,h,w]。"""
        if timesteps.ndim == 0:
            timesteps = timesteps[None].expand(latents.shape[0])
        time = self.time_mlp(timestep_embedding(timesteps, self.config.time_dim))
        skip = self.attn1(self.down1(self.in_conv(latents), time), context)
        hidden = self.downsample(skip)  # [B,C,h,w] -> [B,2C,ceil(h/2),ceil(w/2)]
        hidden = self.attn2(self.down2(hidden, time), context)
        hidden = self.mid_attn(self.mid(hidden, time), context)
        hidden = self.upsample(hidden)  # 上采样回 skip 尺度：[B,2C,h/2,w/2] -> [B,C,h,w]
        hidden = torch.cat((hidden, skip), dim=1)  # skip 拼接：[B,2*base_channels,h,w]
        hidden = self.up(hidden, time)
        hidden = self.up_attn(hidden, context)
        return self.out_conv(F.silu(self.out_norm(hidden)))


class SDv1Model(nn.Module):
    """封装冻结的 VAE/文本编码器、可训练 U-Net、DDIM 训练目标与采样流程。

    图像输入 [B,3,H,W]；文本上下文 [B,T,text_dim]；生成图像 [B,3,H,W]，值域 [0,1]。
    潜变量按 latent_scaling 缩放后进入扩散过程，解码前执行对应逆缩放。
    """

    def __init__(self, config: SDv1Config | None = None) -> None:
        """创建模型；冻结 VAE 和文本编码器参数，仅训练去噪网络。"""
        super().__init__()
        self.config = config or SDv1Config()
        self.vae = TinyVAE(self.config.latent_channels)
        self.text_encoder = TinyTextEncoder(self.config.text_dim)
        self.denoiser = SDv1UNet(self.config)
        self.scheduler = DDIMScheduler(prediction_type=self.config.prediction_type)
        for parameter in (*self.vae.parameters(), *self.text_encoder.parameters()):
            parameter.requires_grad_(False)

    def encode_prompts(self, prompts: Sequence[str], device: torch.device) -> Tensor:
        """将 B 条文本编码为上下文特征 [B,T,text_dim]。"""
        context, _ = self.text_encoder.encode_prompts(prompts, device)
        return context

    def training_loss(self, images: Tensor, prompts: Sequence[str]) -> Tensor:
        """计算单批扩散去噪 MSE。

        images: [B,3,H,W]；潜变量及噪声: [B,C,H/8,W/8]；
        返回标量损失。根据 prediction_type 监督 epsilon 或 v。
        """
        with torch.no_grad():
            latents = self.vae.encode(images).mode() * self.config.latent_scaling  # [B,3,H,W] -> [B,C,H/8,W/8]
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
        """执行 DDIM 文本生图，返回 [B,3,H,W]、值域 [0,1] 的图像批次。

        从标准高斯潜变量 [B,C,H/8,W/8] 开始，每步组合空提示与条件提示预测。
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
        for timestep in self.scheduler.set_timesteps(num_inference_steps, device):
            timesteps = timestep.expand(len(prompts))
            unconditional_prediction = self.denoiser(latents, timesteps, unconditional)
            conditional_prediction = self.denoiser(latents, timesteps, context)
            prediction = classifier_free_guidance(
                unconditional_prediction, conditional_prediction, guidance_scale
            )
            # DDIM 根据引导后的噪声/速度预测将 z_t 更新为下一时间步潜变量。
            latents = self.scheduler.step(prediction, timestep, latents).prev_sample
        images = self.vae.decode(latents / self.config.latent_scaling)
        return (images.clamp(-1.0, 1.0) + 1.0) / 2.0


def build_model() -> SDv1Model:
    """按默认配置构建 SD-v1 教学模型。"""
    return SDv1Model()


if __name__ == "__main__":
    model = build_model()
    print(f"SD-v1 tiny parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"latent shape for 32x32 image: [B, 4, {model.config.image_size // 8}, {model.config.image_size // 8}]")
