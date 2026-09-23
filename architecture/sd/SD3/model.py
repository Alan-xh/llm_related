"""SD3 风格多模态扩散 Transformer（MMDiT）教学实现。

任务定义：使用文本与图像 token 的联合注意力，在潜空间执行文本条件图像生成。
代表架构：Stable Diffusion 3 的 MMDiT 与 flow matching。
核心流程：潜变量展平为图像 token，文本特征分别投影；MMDiT 保留两路特征流，
拼接归一化后的 token 执行联合注意力，再分流并各自更新，最终恢复空间布局。
目标函数：z_t=(1-t)z0+t*epsilon，v=epsilon-z0，
L=MSE(v_theta(z_t,t,c),v)；采样采用 Euler 更新 z_next=z_t+(t_next-t)*v_theta。
形状：潜变量 [B,C,h,w] -> 图像 token [B,h*w,D]；
文本上下文 [B,T,text_dim] -> 文本 token [B,T,D]；
输出潜变量预测 [B,C,h,w]。
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
    """SD3 教学配置，定义潜变量、文本、联合隐藏维度及 MMDiT 深度。"""

    image_size: int = 32
    latent_channels: int = 4
    text_dim: int = 80
    hidden_size: int = 128
    time_dim: int = 128
    layers: int = 3
    max_image_tokens: int = 256
    latent_scaling: float = 0.18215


class MMDiTBlock(nn.Module):
    """具有独立图像/文本前馈流和共享联合注意力的 MMDiT 块。

    image_tokens: [B,N_i,D]，text_tokens: [B,N_t,D]，time: [B,time_dim]；
    输出保持两路输入形状。时间条件分别投影并广播到各自 token。
    """

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
        """联合处理图像与文本 token，返回 [B,N_i,D] 和 [B,N_t,D]。"""
        image_tokens = image_tokens + self.image_time(time)[:, None]
        text_tokens = text_tokens + self.text_time(time)[:, None]
        # 沿 token 轴合并两路：[B,N_i+N_t,D]，供共享注意力交互。
        joint = torch.cat((self.image_norm(image_tokens), self.text_norm(text_tokens)), dim=1)
        joint = joint + self.joint_attention(joint)
        image_length = image_tokens.shape[1]
        # 根据原图像 token 数切分注意力输出，分别回写两路残差流。
        image_tokens = image_tokens + joint[:, :image_length]
        text_tokens = text_tokens + joint[:, image_length:]
        return image_tokens + self.image_ff(image_tokens), text_tokens + self.text_ff(text_tokens)


class MMDiTDenoiser(nn.Module):
    """将二维潜变量与文本上下文映射到联合 token 空间并预测流速度。

    输入 latents [B,C,h,w]、timesteps [B]、context [B,T,text_dim]；
    返回速度场 [B,C,h,w]。位置嵌入仅加到图像 token。
    """

    def __init__(self, config: SD3Config) -> None:
        """依据配置建立图像/文本投影、时间嵌入、MMDiT 块与输出投影。"""
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
        """潜变量 [B,C,h,w] 经 token 化与联合处理后恢复为 [B,C,h,w]。"""
        if timesteps.ndim == 0:
            timesteps = timesteps[None].expand(latents.shape[0])
        batch, _, height, width = latents.shape
        image_tokens = latents.flatten(2).transpose(1, 2)  # [B,C,h,w] -> [B,h*w,C]
        image_tokens = self.image_in(image_tokens) + self.image_pos[:, : image_tokens.shape[1]]
        text_tokens = self.text_in(context)
        time = self.time_in(timestep_embedding(timesteps, self.config.time_dim))
        for block in self.blocks:
            image_tokens, text_tokens = block(image_tokens, text_tokens, time)
        output = self.image_out(image_tokens)
        return output.transpose(1, 2).reshape(batch, self.config.latent_channels, height, width)  # [B,h*w,C] -> [B,C,h,w]


class SD3Model(nn.Module):
    """封装 SD3 风格的 VAE、文本编码器、MMDiT 去噪器和 flow-matching 采样。

    图像 [B,3,H,W] 经 VAE 变为 [B,C,H/8,W/8]；文本上下文为 [B,T,D]；
    生成图像为 [B,3,H,W]，值域 [0,1]。
    """

    def __init__(self, config: SD3Config | None = None) -> None:
        """初始化模型并冻结 VAE 与文本编码器参数。"""
        super().__init__()
        self.config = config or SD3Config()
        self.vae = TinyVAE(self.config.latent_channels)
        self.text_encoder = TinyTextEncoder(self.config.text_dim)
        self.denoiser = MMDiTDenoiser(self.config)
        self.scheduler = FlowMatchScheduler()
        for parameter in (*self.vae.parameters(), *self.text_encoder.parameters()):
            parameter.requires_grad_(False)

    def encode_prompts(self, prompts: Sequence[str], device: torch.device) -> Tensor:
        """将 B 条提示编码为 [B,T,text_dim] 上下文。"""
        context, _ = self.text_encoder.encode_prompts(prompts, device)
        return context

    def training_loss(self, images: Tensor, prompts: Sequence[str]) -> Tensor:
        """构造 flow-matching 插值并监督速度 epsilon-clean，返回标量 MSE。"""
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
        num_inference_steps: int = 8,
        guidance_scale: float = 4.5,
        device: torch.device | None = None,
        seed: int = 0,
    ) -> Tensor:
        """以 Euler flow matching 采样，返回 [B,3,H,W]、范围 [0,1] 的图像。"""
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
            # 按 dt=t_next-t 更新 z_next=z_t+dt*v，velocity 与 latents 同形。
            latents = self.scheduler.step(velocity, timestep, latents, next_timestep)
        images = self.vae.decode(latents / self.config.latent_scaling)
        return (images.clamp(-1.0, 1.0) + 1.0) / 2.0


def build_model() -> SD3Model:
    """按默认配置构建 SD3 教学模型。"""
    return SD3Model()


if __name__ == "__main__":
    model = build_model()
    print(f"SD3 tiny parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("image stream and text stream are projected separately, then joined for attention")
