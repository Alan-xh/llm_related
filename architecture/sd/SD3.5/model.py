"""SD3.5 风格多模态扩散 Transformer 的精简教学实现。

任务定义：通过文本条件在潜空间生成图像，并展示带 guidance/distillation 接口的变体。
代表架构：Stable Diffusion 3.5 系列 MMDiT；该文件复用 SD3 的联合图文去噪骨干。
核心流程：VAE 潜变量和文本上下文进入 MMDiT，分别计算无条件/有条件速度，
进行 classifier-free guidance，再由 flow-matching Euler 求解器更新潜变量。
目标函数：z_t=(1-t)z0+t*epsilon，v=epsilon-z0，
L=MSE(v_theta(z_t,t,c),v)；引导公式 v_g=v_u+s(v_c-v_u)。
形状：潜变量 [B,C,h,w]，文本上下文 [B,T,text_dim]，guidance 标量，
输出图像 [B,3,H,W]。
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
    """SD3.5 教学配置，包含 MMDiT 宽度/深度与蒸馏变体标记。"""

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
    """扩展 MMDiT 去噪器，并提供可选 guidance embedding 接口。

    输入/输出潜变量形状均为 [B,C,h,w]；时间步 [B]，文本上下文 [B,T,D]。
    guidance_scale=1 时直接返回父类预测，否则对预测施加轻量条件缩放。
    """

    def __init__(self, config: SD35Config) -> None:
        """复用 SD3 MMDiT 主体并建立 guidance 强度嵌入投影。"""
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
        """执行去噪，输入 latents [B,C,h,w]、context [B,T,D]，输出同形预测。

        父类已注入 timestep；此处将 guidance_scale 映射为批次条件，
        作为 SD3.5 教学版的 guidance/distillation 扩展接口。
        """
        # 父类已经处理时间条件；该投影仅示范家族变体的 guidance/distillation 接口。
        output = super().forward(latents, timesteps, context)
        if guidance_scale == 1.0:
            return output
        # [B,1] -> [B,time_dim] -> [B,1,1,1]，广播缩放整张预测特征图。
        return output * (1.0 + 0.02 * self.guidance_embedding(
            torch.full((latents.shape[0], 1), guidance_scale, device=latents.device)
        )[:, :1, None, None])


class SD35Model(nn.Module):
    """封装 SD3.5 风格 MMDiT、flow matching 训练目标及引导采样。

    输入/输出图像 [B,3,H,W]；潜变量 [B,C,H/8,W/8]；
    文本上下文 [B,T,text_dim]。
    """

    def __init__(self, config: SD35Config | None = None) -> None:
        """初始化模型并冻结 VAE 和文本编码器。"""
        super().__init__()
        self.config = config or SD35Config()
        self.vae = TinyVAE(self.config.latent_channels)
        self.text_encoder = TinyTextEncoder(self.config.text_dim)
        self.denoiser = SD35Denoiser(self.config)
        self.scheduler = FlowMatchScheduler()
        for parameter in (*self.vae.parameters(), *self.text_encoder.parameters()):
            parameter.requires_grad_(False)

    def encode_prompts(self, prompts: Sequence[str], device: torch.device) -> Tensor:
        """将 B 条提示编码为 [B,T,text_dim] 文本上下文。"""
        context, _ = self.text_encoder.encode_prompts(prompts, device)
        return context

    def training_loss(self, images: Tensor, prompts: Sequence[str]) -> Tensor:
        """监督 flow 速度 epsilon-clean，输入图像 [B,3,H,W] 并返回标量 MSE。"""
        with torch.no_grad():
            clean = self.vae.encode(images).mode() * self.config.latent_scaling  # [B,3,H,W] -> [B,C,H/8,W/8]
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
        """执行 SD3.5 flow-matching 采样，返回 [B,3,H,W]、值域 [0,1] 的图像。"""
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
            # v_g=v_uncond+s*(v_cond-v_uncond)，引导后的速度用于 Euler 更新。
            velocity = classifier_free_guidance(
                unconditional_velocity, conditional_velocity, guidance_scale
            )
            latents = self.scheduler.step(velocity, timestep, latents, next_timestep)
        images = self.vae.decode(latents / self.config.latent_scaling)
        return (images.clamp(-1.0, 1.0) + 1.0) / 2.0


def build_model() -> SD35Model:
    """按默认配置构建 SD3.5 教学模型。"""
    return SD35Model()


if __name__ == "__main__":
    model = build_model()
    print(f"SD3.5 tiny parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("SD3.5 hook: wider/deeper MMDiT plus optional guidance/distillation embedding")
