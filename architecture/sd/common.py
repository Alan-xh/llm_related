"""Stable Diffusion 教学模型的公共模块与张量约定。

任务定义：提供潜空间扩散、文本条件编码、注意力和采样器的可复用组件。
代表架构：Stable Diffusion、SDXL 与 Stable Diffusion 3 的简化教学实现。
核心数据流：图像经 VAE 编码为潜变量，文本编码为上下文特征，去噪网络预测噪声/
速度场，最后由采样器迭代更新潜变量并经 VAE 解码。
核心目标：DDPM 加噪 z_t = sqrt(alpha_bar_t) z_0 + sqrt(1-alpha_bar_t) eps；
流匹配路径 z_t = (1-t) z_0 + t eps；分类器自由引导
f_guided = f_uncond + s (f_cond - f_uncond)。
数据约定：图像/潜变量为 [B,C,H,W]，文本序列为 [B,T,C]，时间嵌入为 [B,D]。

这里使用小型随机初始化模块和合成数据以展示张量流，不加载官方权重，
也不复现官方模型规模或训练配方。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
    """将时间步编码为正弦位置特征。

    输入 timesteps: [B]；输出: [B, dim]。偶数维使用 sin/cos 频率对，
    奇数维在末尾补零。
    """

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
    """选择不超过 requested 且可整除通道数的最大 GroupNorm 组数。"""
    for groups in range(min(requested, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResBlock2D(nn.Module):
    """带扩散时间条件的二维残差块，供各版本 U-Net 复用。

    输入/输出 x: [B,C,H,W]；时间嵌入: [B,D]；输出: [B,C_out,H,W]。
    时间投影后广播为 [B,C_out,1,1] 并加到卷积特征上；
    主支路与 identity/1x1 卷积捷径相加。
    """

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
        """处理特征图 x [B,C_in,H,W] 和时间向量 [B,D]，返回 [B,C_out,H,W]。"""
        hidden = self.conv1(F.silu(self.norm1(x)))
        # [B,D] -> [B,C_out] -> [B,C_out,1,1]，按空间维广播注入时间条件。
        hidden = hidden + self.time_proj(time_embedding_value)[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return self.skip(x) + hidden


class FeedForward(nn.Module):
    """逐 token 前馈网络：在特征维扩张、经过 GELU 后投影回原维度。"""

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
        """输入/输出形状均为 [B,T,C]，仅变换最后的特征维。"""
        return self.net(x)


class Attention(nn.Module):
    """支持独立 query 与 context 序列的多头注意力。

    query: [B,N,Cq]，context: [B,M,Cctx]（省略时使用 query）；
    输出: [B,N,Cq]。核心公式为 softmax(QK^T / sqrt(d))V，
    其中 to_q/to_k/to_v 对应 Q/K/V，self.scale 对应 1/sqrt(d)。
    """

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
        """输入 query [B,N,Cq]、context [B,M,Cctx]，输出 [B,N,Cq]。"""
        context = query if context is None else context
        query = self.norm(query)
        context = self.context_norm(context)
        batch, query_length, _ = query.shape
        key_length = context.shape[1]
        q = self.to_q(query).view(batch, query_length, self.heads, self.head_dim)
        k = self.to_k(context).view(batch, key_length, self.heads, self.head_dim)
        v = self.to_v(context).view(batch, key_length, self.heads, self.head_dim)
        q = q.transpose(1, 2)  # [B,N,H,D] -> [B,H,N,D]
        k = k.transpose(1, 2)  # [B,M,H,D] -> [B,H,M,D]
        v = v.transpose(1, 2)  # [B,M,H,D] -> [B,H,M,D]
        # scores = QK^T / sqrt(d)，softmax 后乘 V 得到每个 query 的加权上下文。
        weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = weights.softmax(dim=-1)
        # [B,H,N,D] -> [B,N,H*D]，合并注意力头并映射回 query 特征维。
        output = torch.matmul(weights, v).transpose(1, 2).reshape(
            batch, query_length, self.heads * self.head_dim
        )
        return self.to_out(output)


class SpatialTransformer(nn.Module):
    """在二维图像特征上依次执行自注意力、文本交叉注意力和前馈变换。

    输入/输出特征: [B,C,H,W]；文本 context: [B,T,Cctx]。
    展平空间维后得到 [B,H*W,C] token 序列，处理后恢复原空间形状，
    并通过残差投影与输入相加。
    """

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
        """输入 x [B,C,H,W]、context [B,T,Cctx]，输出 [B,C,H,W]。"""
        batch, channels, height, width = x.shape
        hidden = self.proj_in(self.norm(x))
        tokens = hidden.flatten(2).transpose(1, 2)  # [B,C,H,W] -> [B,H*W,C]
        tokens = tokens + self.self_attn(tokens)
        tokens = tokens + self.cross_attn(tokens, context)
        tokens = tokens + self.ff(tokens)
        hidden = tokens.transpose(1, 2).reshape(batch, channels, height, width)  # [B,H*W,C] -> [B,C,H,W]
        return x + self.proj_out(hidden)


class DiagonalGaussianDistribution:
    """由 VAE 编码器输出参数化的对角高斯后验分布。

    parameters: [B,2C,H,W]，沿通道均分为均值与对数方差 [B,C,H,W]。
    """

    def __init__(self, parameters: Tensor) -> None:
        """拆分并限制 log-variance，避免指数运算数值溢出。"""
        self.mean, self.logvar = parameters.chunk(2, dim=1)
        self.logvar = self.logvar.clamp(-30.0, 20.0)

    def sample(self) -> Tensor:
        """按 z = mean + exp(0.5*logvar)*epsilon 重参数化采样，输出 [B,C,H,W]。"""
        return self.mean + torch.exp(0.5 * self.logvar) * torch.randn_like(self.mean)

    def mode(self) -> Tensor:
        """返回后验众数（均值），形状为 [B,C,H,W]。"""
        return self.mean


class TinyVAE(nn.Module):
    """用于教学示例的卷积 VAE，编码器空间下采样 8 倍。

    输入图像: [B,3,H,W]；后验参数: [B,2*latent_channels,H/8,W/8]；
    潜变量: [B,latent_channels,H/8,W/8]；解码输出: [B,3,H,W]。
    """

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
        """将图像 [B,3,H,W] 编码为后验分布，或按 sample 选择潜变量样本。"""
        posterior = DiagonalGaussianDistribution(self.encoder(images))
        return posterior.sample() if sample else posterior

    def decode(self, latents: Tensor) -> Tensor:
        """将潜变量 [B,latent_channels,h,w] 解码为图像 [B,3,8h,8w]。"""
        return self.decoder(latents)


class ByteTokenizer:
    """无外部依赖的 UTF-8 字节 tokenizer，便于检查文本条件的张量流。"""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    vocab_size = 259

    def encode(self, text: str, max_length: int = 64) -> list[int]:
        """将单条文本编码为 BOS、字节 token 与 EOS 的整数序列（最长 max_length）。"""
        ids = [self.bos_token_id]
        ids.extend(byte + 3 for byte in text.encode("utf-8")[: max_length - 2])
        ids.append(self.eos_token_id)
        return ids

    def batch_encode(self, texts: Sequence[str], max_length: int = 64) -> tuple[Tensor, Tensor]:
        """批量编码文本，返回 input_ids 与有效位掩码，形状均为 [B,max_length]。"""
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
    """小型 CLIP/OpenCLIP 风格文本编码器。

    token ID 与掩码: [B,T]；上下文输出: [B,T,D]；池化文本向量: [B,D]。
    """

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
        """输入 token IDs [B,T] 和可选掩码 [B,T]，返回 token 特征 [B,T,D] 与池化特征 [B,D]。"""
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
        """编码 B 条 prompt，返回上下文 [B,T,D] 和池化向量 [B,D]。"""
        input_ids, attention_mask = self.tokenizer.batch_encode(prompts, self.max_length)
        return self(input_ids.to(device), attention_mask.to(device))


class DualTextEncoder(nn.Module):
    """SDXL 风格的双文本编码器，在最后一维拼接 token 与池化特征。

    输入为 B 条文本；输出 token 特征 [B,T,D1+D2] 和池化向量 [B,D1+D2]。
    """

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
        """分别编码 prompt 并拼接两路特征，维度由两个编码器的隐藏维相加。"""
        first_tokens, first_pooled = self.first.encode_prompts(prompts, device)
        second_tokens, second_pooled = self.second.encode_prompts(prompts, device)
        return torch.cat((first_tokens, second_tokens), dim=-1), torch.cat(
            (first_pooled, second_pooled), dim=-1
        )


@dataclass
class DDIMStepOutput:
    """一次 DDIM 更新结果：前一时间步样本与预测的干净潜变量，形状均为 [B,C,H,W]。"""

    prev_sample: Tensor
    pred_original_sample: Tensor


class DDIMScheduler:
    """支持 epsilon 与 v-prediction 参数化的 DDIM 扩散调度器。

    加噪公式 z_t=sqrt(alpha_bar_t)z_0+sqrt(1-alpha_bar_t)epsilon；
    get_velocity 返回 v=sqrt(alpha_bar_t)epsilon-sqrt(1-alpha_bar_t)z_0。
    """

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
        """创建从训练末端到 0 的推理时间步，输出 [num_inference_steps]。"""
        self.timesteps = torch.linspace(
            self.num_train_timesteps - 1,
            0,
            num_inference_steps,
            device=device,
        ).round().long()
        return self.timesteps

    def add_noise(self, sample: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """按 DDPM 前向过程混合干净样本与噪声；输入/输出 [B,C,H,W]，时间步 [B]。"""
        alpha = self.alphas_cumprod.to(sample.device)[timesteps].view(-1, 1, 1, 1)
        return alpha.sqrt() * sample + (1.0 - alpha).sqrt() * noise

    def get_velocity(self, sample: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """计算 v-prediction 目标，sample、noise 与返回值均为 [B,C,H,W]。"""
        alpha = self.alphas_cumprod.to(sample.device)[timesteps].view(-1, 1, 1, 1)
        return alpha.sqrt() * noise - (1.0 - alpha).sqrt() * sample

    def step(self, model_output: Tensor, timestep: Tensor | int, sample: Tensor) -> DDIMStepOutput:
        """将当前样本 [B,C,H,W] 更新到前一时间步，并返回 x0 估计。"""
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
            # v 参数化反解 x0 与 epsilon；变量名分别对应预测干净样本与噪声。
            pred_original = alpha_t.sqrt() * sample - (1.0 - alpha_t).sqrt() * model_output
            epsilon = alpha_t.sqrt() * model_output + (1.0 - alpha_t).sqrt() * sample
        else:
            # epsilon 参数化：由 z_t 与预测噪声反解干净样本。
            pred_original = (sample - (1.0 - alpha_t).sqrt() * model_output) / alpha_t.sqrt()
            epsilon = model_output
        direction = (1.0 - alpha_prev).clamp_min(0).sqrt() * epsilon
        previous = alpha_prev.sqrt() * pred_original + direction
        return DDIMStepOutput(previous, pred_original)


class FlowMatchScheduler:
    """数据到高斯噪声线性路径的 flow-matching Euler 求解器。

    路径 z_t=(1-t)z_0+t*epsilon；模型目标速度 v=epsilon-z_0；
    推理按 z_next=z_t+(t_next-t)*v 更新潜变量。
    """

    def set_timesteps(self, num_inference_steps: int, device: torch.device) -> Tensor:
        """生成从 1 到 0 的含端点时间网格，输出 [num_inference_steps+1]。"""
        return torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)

    def add_noise(self, sample: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """沿线性 flow 路径插值；sample/noise 与输出为 [B,C,H,W]，时间步为 [B]。"""
        t = timesteps.view(-1, 1, 1, 1)
        return (1.0 - t) * sample + t * noise

    def step(self, velocity: Tensor, timestep: Tensor, sample: Tensor, next_timestep: Tensor) -> Tensor:
        """执行一次显式 Euler 更新，velocity/sample 形状均为 [B,C,H,W]。"""
        delta = next_timestep - timestep
        return sample + delta * velocity


def classifier_free_guidance(unconditional: Tensor, conditional: Tensor, scale: float) -> Tensor:
    """按 u+s(c-u) 合并无条件与有条件预测；三者形状均为 [B,C,H,W]。"""
    return unconditional + scale * (conditional - unconditional)


def build_toy_images(
    batch_size: int,
    image_size: int,
    device: torch.device,
    step: int = 0,
) -> tuple[Tensor, list[str]]:
    """生成无需数据集的确定性彩色玩具图像。

    返回图像 [B,3,H,W]（值域约为 [-1,1]）和长度为 B 的对应文本提示。
    """

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
    """统计模块中所有参数的标量总数。"""
    return sum(parameter.numel() for parameter in module.parameters())
