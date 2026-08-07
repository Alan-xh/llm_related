"""
任务 6：无条件生成（Unconditional Generation）

代表模型：
    DDPM (Denoising Diffusion Probabilistic Model)
    去噪扩散概率模型

核心思想：
    训练一个神经网络 εθ(x_t, t)，预测扩散过程中加入的高斯噪声。

训练目标：
    最小化预测噪声与真实噪声之间的 MSE：
        L = E_{x0,ε,t} [ || ε - εθ(x_t,t) ||² ]

其中：
    x0: 原始图像
    x_t: 第 t 步加入噪声后的图像
    ε: 从标准高斯分布采样的噪声 ε ~ N(0,I)
    εθ: U-Net 噪声预测网络

数据：
    使用随机生成的 32×32 RGB 图像训练一个小型 U-Net。
"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset



# ==============================
# 超参数
# ==============================

BATCH_SIZE = 64
EPOCHS = 10
LR = 2e-4

T = 1000 # 扩散总时间步: t ∈ {0,1,...,T-1}


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def linear_beta_schedule(
    timesteps,
    start=1e-4,
    end=0.02
):
    """
    生成 DDPM 噪声调度 beta_t。

    公式:
        β_t = linspace(β_start, β_end, T)

    Args:
        timesteps: 扩散步数 T。
        start: 初始噪声方差 β_1。
        end: 最终噪声方差 β_T。

    Returns:
        betas: Tensor, shape [T]，每个时间步的噪声强度 β_t。
    """
    return torch.linspace(start, end, timesteps)

betas = linear_beta_schedule(T) # β_t
alphas = 1.0 - betas # 当前步骤保留的信息比例: α_t = 1 - β_t
alphas_cumprod = torch.cumprod(alphas, dim=0) # 经过 t 次扩散后，原始图像 x0 仍然保留的信息比例: ᾱ_t = Π(i=1,t) α_i
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod) # sqrt(ᾱ_t)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod) # 累计加入噪声的比例: sqrt(1-ᾱ_t)


class TimeEmbedding(nn.Module):
    """
    DDPM 时间步编码器，将离散时间步 t ∈ [0,T), 映射为连续向量：t -> R^dim

    使用 Transformer 相同形式的 sinusoidal embedding:
        PE(t,2i)=sin(t/10000^(2i/d))
        PE(t,2i+1)=cos(t/10000^(2i/d))

    Input:
        t: Tensor, shape [B]，每个样本对应的扩散时间步。

    Output:
        embedding: Tensor, shape [B, dim]，时间特征。
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Args:
            t: Tensor[int], [B]
        Returns:
            Tensor, [B, dim]
        """

        half_dim = self.dim // 2

        emb = math.log(10000) / (half_dim - 1)

        emb = torch.exp(
            torch.arange(
                half_dim,
                device=t.device
            ) * -emb
        )

        t = t[:, None].float() * emb[None, :]

        emb = torch.cat([torch.sin(t), torch.cos(t)], dim=-1)

        return emb


class TinyUNet(nn.Module):
    """
    用于 DDPM 噪声预测的小型 U-Net。
    x_t -> εθ(x_t,t), 即预测加入到图片中的噪声。

    网络结构：

        Encoder:

            32×32×3 - Conv -> 32×32×64 - Downsample -> 16×16×128

        Middle:

            16×16×128

        Decoder:
            
            16×16×128 - Upsample -> 32×32×64 - Upsample -> 32×32×3


    时间 embedding:

        t -> TimeEmbedding -> Linear -> 加入 feature map

    Args:

        in_ch: int, 输入图像通道数, 例如, RGB 图像, in_ch=3

        base: int, 基础 feature channel。

        time_dim: int, 时间编码维度。

    Input:

        x: Tensor, [B,C,H,W], 例如：[64,3,32,32]
        t: Tensor, [B]

    Output:

        predicted_noise: Tensor, [B,C,H,W], 即 εθ(x_t,t)
    """

    def __init__(
        self,
        in_ch=3,
        base=64,
        time_dim=128
    ):
        super().__init__()

        self.time_embed = nn.Sequential(TimeEmbedding(time_dim), nn.Linear(time_dim,time_dim), nn.SiLU())

        self.down1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(),
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
        )

        self.mid = nn.Sequential(
            nn.Conv2d(base * 2, base * 2, 3,padding=1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base * 2, base, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(),
        )

        self.up1 = nn.Conv2d(base, in_ch, 3, padding=1)

    def forward(self, x, t):
        """
        U-Net forward

        Args:

            x: Tensor, noisy image x_t, [B,3,32,32]

            t:Tensor, diffusion timestep,[B]

        Returns:
            predicted noise: Tensor, εθ(x_t,t),[B,3,32,32]
        """

        t_emb = self.time_embed(t)

        h = self.down1(x)

        h = self.down2(h)

        h = h + t_emb[:, :, None, None] # 将时间信息加入 feature map, h: [B,128,16,16], t_emb: [B,128]

        h = self.mid(h)

        h = self.up2(h)

        return self.up1(h)


def q_sample(x0,t,noise=None):
    """
    DDPM 前向扩散过程。

    根据公式：

        q(x_t|x_0) =  N(sqrt(ᾱ_t)x_0, (1-ᾱ_t)I)

    可以直接采样：
        x_t = sqrt(ᾱ_t)x_0 + sqrt(1-ᾱ_t)ε

    其中：
        ε ~ N(0,I)

    Args:
        x0:Tensor, 原始图片, [B,C,H,W]

        t:Tensor, 时间步, [B]

        noise: Tensor, 随机噪声 ε, [B,C,H,W]

    Returns:

        x_t: Tensor, 加噪后的图片, [B,C,H,W]
    """

    if noise is None:
        noise = torch.randn_like(x0)

    sqrt_acp = sqrt_alphas_cumprod[t].view(
        -1,1,1,1
    ).to(x0.device)

    sqrt_omc = sqrt_one_minus_alphas_cumprod[t].view(
        -1,1,1,1
    ).to(x0.device)

    return sqrt_acp * x0 + sqrt_omc * noise


def get_synthetic_dataset(num_samples=1000):
    x0 = torch.randn(num_samples, 3, 32, 32)
    return TensorDataset(x0)


def main():
    train_loader = DataLoader(
        get_synthetic_dataset(), batch_size=BATCH_SIZE, shuffle=True
    )

    model = TinyUNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for (x0,) in train_loader:
            x0 = x0.to(DEVICE)
            t = torch.randint(0, T, (x0.size(0),), device=DEVICE)
            noise = torch.randn_like(x0)
            xt = q_sample(x0, t, noise)

            pred_noise = model(xt, t)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}]  Noise MSE: {avg_loss:.4f}")


if __name__ == "__main__":
    main()