"""SDXL 风格双文本条件、Base/Refiner 潜空间扩散教学实现。

任务定义：以双文本编码器特征和图像尺寸/裁剪元数据为条件生成图像。
代表架构：Stable Diffusion XL 的双文本编码器及 Base/Refiner 两阶段流程。
核心流程：双编码器 token 特征拼接后进入交叉注意力，pooled 特征与 6 维 time IDs
共同形成附加条件；Base 与 Refiner 分别执行 DDIM 去噪。
目标函数：L=MSE(epsilon_theta(z_t,t,c,p,time_ids),epsilon)，其中
z_t=sqrt(alpha_bar_t)z0+sqrt(1-alpha_bar_t)epsilon。
形状：潜变量 [B,C,h,w]；文本 token [B,T,D]、pooled [B,D]；
time IDs [B,6]；去噪输出与潜变量同形。
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
        DualTextEncoder,
        ResBlock2D,
        SpatialTransformer,
        TinyVAE,
        classifier_free_guidance,
        timestep_embedding,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from architecture.sd.common import (
        DDIMScheduler,
        DualTextEncoder,
        ResBlock2D,
        SpatialTransformer,
        TinyVAE,
        classifier_free_guidance,
        timestep_embedding,
    )


@dataclass
class SDXLConfig:
    """SDXL 教学配置；text_dim/pooled_dim 应与双文本编码器输出匹配。"""

    image_size: int = 32
    latent_channels: int = 4
    base_channels: int = 40
    text_dim: int = 144
    pooled_dim: int = 144
    time_dim: int = 128
    latent_scaling: float = 0.18215


class SDXLUNet(nn.Module):
    """融合 token 交叉注意力、pooled 文本和 time IDs 的 U-Net。

    输入潜变量 [B,C,h,w]、时间步 [B]、文本上下文 [B,T,D]、
    pooled 特征 [B,P]、time IDs [B,6]；输出 [B,C,h,w]。
    附加条件按 concat(pooled,time_ids) 投影后加到时间嵌入。
    """

    def __init__(self, config: SDXLConfig) -> None:
        super().__init__()
        channels = config.base_channels
        self.config = config
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_dim, config.time_dim * 4),
            nn.SiLU(),
            nn.Linear(config.time_dim * 4, config.time_dim),
        )
        self.added_cond = nn.Sequential(
            nn.Linear(config.pooled_dim + 6, config.time_dim),
            nn.SiLU(),
            nn.Linear(config.time_dim, config.time_dim),
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

    def forward(
        self,
        latents: Tensor,
        timesteps: Tensor,
        context: Tensor,
        pooled: Tensor,
        time_ids: Tensor,
    ) -> Tensor:
        """处理潜变量与文本/尺寸条件，输出同形去噪预测 [B,C,h,w]。"""
        if timesteps.ndim == 0:
            timesteps = timesteps[None].expand(latents.shape[0])
        time = self.time_mlp(timestep_embedding(timesteps, self.config.time_dim))
        # pooled [B,P] 与 time_ids [B,6] 拼接为 [B,P+6] 后投影至时间维。
        time = time + self.added_cond(torch.cat((pooled, time_ids), dim=-1))
        skip = self.attn1(self.down1(self.in_conv(latents), time), context)
        hidden = self.downsample(skip)  # [B,C,h,w] -> [B,2C,ceil(h/2),ceil(w/2)]
        hidden = self.attn2(self.down2(hidden, time), context)
        hidden = self.mid_attn(self.mid(hidden, time), context)
        hidden = self.upsample(hidden)  # 上采样回 skip 尺度：[B,2C,h/2,w/2] -> [B,C,h,w]
        hidden = self.up(torch.cat((hidden, skip), dim=1), time)  # skip 拼接后通道翻倍
        hidden = self.up_attn(hidden, context)
        return self.out_conv(F.silu(self.out_norm(hidden)))


class SDXLModel(nn.Module):
    """封装 SDXL 风格 Base/Refiner 两阶段生成管线。

    图像输入/输出 [B,3,H,W]；潜变量 [B,C,H/8,W/8]；
    双编码器上下文 [B,T,D]、pooled 向量 [B,P]、尺寸条件 [B,6]。
    """

    def __init__(self, config: SDXLConfig | None = None) -> None:
        """创建 Base 与 Refiner 去噪器，并冻结 VAE 和双文本编码器。"""
        super().__init__()
        self.config = config or SDXLConfig()
        self.vae = TinyVAE(self.config.latent_channels)
        self.text_encoder = DualTextEncoder()
        self.base = SDXLUNet(self.config)
        self.refiner = SDXLUNet(self.config)
        self.scheduler = DDIMScheduler()
        self.refiner_scheduler = DDIMScheduler()
        for parameter in (*self.vae.parameters(), *self.text_encoder.parameters()):
            parameter.requires_grad_(False)

    def encode_prompts(
        self, prompts: Sequence[str], device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """返回双编码器拼接后的 token 上下文 [B,T,D] 与 pooled 特征 [B,P]。"""
        return self.text_encoder.encode_prompts(prompts, device)

    def _time_ids(self, batch_size: int, device: torch.device) -> Tensor:
        """构造 [原高,原宽,裁剪上,裁剪左,目标高,目标宽]，返回 [B,6]。"""
        size = float(self.config.image_size)
        return torch.tensor(
            [size, size, 0.0, 0.0, size, size],
            device=device,
        ).expand(batch_size, -1)

    def training_loss(self, images: Tensor, prompts: Sequence[str]) -> Tensor:
        """计算 Base 网络的 epsilon 预测 MSE，输入图像 [B,3,H,W]，返回标量。"""
        with torch.no_grad():
            latents = self.vae.encode(images).mode() * self.config.latent_scaling  # [B,3,H,W] -> [B,C,H/8,W/8]
            context, pooled = self.encode_prompts(prompts, images.device)
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
        )
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        time_ids = self._time_ids(images.shape[0], images.device)
        prediction = self.base(noisy_latents, timesteps, context, pooled, time_ids)
        return F.mse_loss(prediction.float(), noise.float())

    @torch.no_grad()
    def sample(
        self,
        prompts: Sequence[str],
        num_inference_steps: int = 8,
        refiner_steps: int = 2,
        guidance_scale: float = 5.0,
        device: torch.device | None = None,
        seed: int = 0,
    ) -> Tensor:
        """先用 Base、再用 Refiner 执行 DDIM；输出 [B,3,H,W] 且值域为 [0,1]。"""
        device = device or next(self.parameters()).device
        generator = torch.Generator(device=device).manual_seed(seed)
        context, pooled = self.encode_prompts(prompts, device)
        unconditional, unconditional_pooled = self.encode_prompts([""] * len(prompts), device)
        time_ids = self._time_ids(len(prompts), device)
        height = self.config.image_size // 8
        latents = torch.randn(
            (len(prompts), self.config.latent_channels, height, height),
            generator=generator,
            device=device,
        )
        for timestep in self.scheduler.set_timesteps(num_inference_steps, device):
            timesteps = timestep.expand(len(prompts))
            unconditional_prediction = self.base(
                latents, timesteps, unconditional, unconditional_pooled, time_ids
            )
            conditional_prediction = self.base(
                latents, timesteps, context, pooled, time_ids
            )
            latents = self.scheduler.step(
                classifier_free_guidance(
                    unconditional_prediction, conditional_prediction, guidance_scale
                ),
                timestep,
                latents,
            ).prev_sample
        # Refiner 接收 Base 的最终潜变量，继续沿其独立时间网格细化。
        for timestep in self.refiner_scheduler.set_timesteps(refiner_steps, device):
            timesteps = timestep.expand(len(prompts))
            unconditional_prediction = self.refiner(
                latents, timesteps, unconditional, unconditional_pooled, time_ids
            )
            conditional_prediction = self.refiner(
                latents, timesteps, context, pooled, time_ids
            )
            latents = self.refiner_scheduler.step(
                classifier_free_guidance(
                    unconditional_prediction, conditional_prediction, guidance_scale
                ),
                timestep,
                latents,
            ).prev_sample
        images = self.vae.decode(latents / self.config.latent_scaling)
        return (images.clamp(-1.0, 1.0) + 1.0) / 2.0


def build_model() -> SDXLModel:
    """按默认配置构建 SDXL 教学模型。"""
    return SDXLModel()


if __name__ == "__main__":
    model = build_model()
    print(f"SDXL tiny parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("conditioning: concat(text_encoder_1, text_encoder_2) + pooled embedding + 6D time ids")
