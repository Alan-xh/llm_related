"""
任务 7：条件生成 (Conditional Generation)
代表架构：Class-Conditional Denoising Diffusion Probabilistic Model (DDPM)
领域分类：生成式模型 / 扩散模型 (Generative Models / Diffusion Models)

1. 核心思想与机制:
   条件 DDPM 在标准无条件 DDPM 的基础上引入类别标签 (Class Label) 或其他条件控制信号。
   前向过程 (Forward Process) 在真实图像 $x_0$ 上逐步添加高斯噪声，获得任意时刻 $t$ 的含噪图像 $x_t$。
   反向过程 (Reverse Process) 利用神经网络 $\epsilon_\theta(x_t, t, y)$ 在给定扩散步 $t$ 及条件类别 $y$ 
   的指导下，预测添加至图像中的高斯噪声 $\epsilon$，从而实现按指定类别生成特定图像。

2. 数学公式与目标函数:
   - 前向加噪边际分布 (Forward Marginal Distribution):
     q(x_t | x_0) = N(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)
     重参数化表示: x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon,  \epsilon \sim N(0, I)

   - 条件噪声预测损失函数 (Conditional Noise Prediction MSE Loss):
     L(\theta) = E_{t, x_0, y, \epsilon} [ || \epsilon - \epsilon_\theta(x_t, t, y) ||^2 ]

   - 正弦时间嵌入 (Sinusoidal Positional Embedding):
     PE_{(pos, 2i)}   = sin(pos / 10000^{2i / d_{model}})
     PE_{(pos, 2i+1)} = cos(pos / 10000^{2i / d_{model}})

3. 数据输入/输出规范:
   - 输入图像 x_0:   [B, C, H, W] = [Batch_Size, 3, 32, 32]
   - 扩散步长 t:     [B]          = [Batch_Size], 范围为 [0, T-1]
   - 类别条件 y:     [B]          = [Batch_Size], 范围为 [0, Num_Classes-1]
   - 预测噪声 output:[B, C, H, W] = [Batch_Size, 3, 32, 32]
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ==============================================================================
# 2. 超参数与全局配置 (Hyperparameters & Config)
# ==============================================================================
BATCH_SIZE: int = 64
EPOCHS: int = 10
LR: float = 2e-4
T: int = 1000
NUM_CLASSES: int = 10
IN_CHANNELS: int = 3
IMAGE_SIZE: int = 32
BASE_CHANNELS: int = 64
COND_DIM: int = 128
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# 3. 扩散过程调度器 (Diffusion Scheduler & Schedule Constants)
# ==============================================================================
def LinearBetaSchedule(timesteps: int = 1000, start: float = 1e-4, end: float = 0.02) -> torch.Tensor:
    """
    线性 Beta 调度计划。
    
    Args:
        timesteps (int): 总扩散步数 T。
        start (float): 初始噪声强度 \beta_1。
        end (float): 最终噪声强度 \beta_T。
        
    Outputs:
        betas (Tensor): 噪声系数序列，shape: [T]
    """
    return torch.linspace(start, end, timesteps)


# 预先计算前向过程所需的标量常数表
betas: torch.Tensor = LinearBetaSchedule(T, start=1e-4, end=0.02)
alphas: torch.Tensor = 1.0 - betas
alphas_cumprod: torch.Tensor = torch.cumprod(alphas, dim=0) # \bar{\alpha}_t
sqrt_alphas_cumprod: torch.Tensor = torch.sqrt(alphas_cumprod) # \sqrt{\bar{\alpha}_t}
sqrt_one_minus_alphas_cumprod: torch.Tensor = torch.sqrt(1.0 - alphas_cumprod) # \sqrt{1 - \bar{\alpha}_t}


def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    前向加噪过程 (Forward Process Sampling)
    根据公式 x_t = \sqrt{\bar{\alpha}_t} * x_0 + \sqrt{1 - \bar{\alpha}_t} * \epsilon 从 x_0 采样 x_t。

    Args:
        x0 (Tensor): 原始无噪图像，shape: [B, C, H, W]
        t (Tensor): 扩散步索引，shape: [B]
        noise (Tensor, optional): 标准高斯噪声 \epsilon ~ N(0, I)，shape: [B, C, H, W]

    Outputs:
        xt (Tensor): 扩散步 t 时刻的含噪图像，shape: [B, C, H, W]
    """
    if noise is None:
        noise = torch.randn_like(x0)
        
    # 从预计算张量中提取步长系数并扩展至 4D 张量以支持广播机制
    sqrt_acp = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1).to(x0.device)       # shape: [B, 1, 1, 1]
    sqrt_omc = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1).to(x0.device) # shape: [B, 1, 1, 1]
    
    # 核心映射：x_t = \sqrt{\bar{\alpha}_t} * x0 + \sqrt{1 - \bar{\alpha}_t} * noise
    xt = sqrt_acp * x0 + sqrt_omc * noise                                   # shape: [B, C, H, W]
    return xt


# ==============================================================================
# 4. 数据处理与 Dataset 管道 (Data Pipeline & Utils)
# ==============================================================================
def get_synthetic_dataset(num_samples: int = 1000) -> TensorDataset:
    """
    构建合成图像与类别标签数据集，用于 Pipeline 演示与测试。

    Args:
        num_samples (int): 生成的样本总数量。

    Outputs:
        dataset (TensorDataset): 包装了合成图像 [N, 3, 32, 32] 与类别 [N] 的数据集。
    """
    x0 = torch.randn(num_samples, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    y = torch.randint(0, NUM_CLASSES, (num_samples,))
    return TensorDataset(x0, y)


# ==============================================================================
# 5. 核心子模块 (Sub-components)
# ==============================================================================
class TimeEmbedding(nn.Module):
    """
    正弦时间位置编码模块 (Sinusoidal Positional Embedding)

    数学原理:
        PE(t, 2i)   = sin(t / 10000^{2i / d})
        PE(t, 2i+1) = cos(t / 10000^{2i / d})

    Args:
        dim (int): 时间嵌入输出维度 d。

    Inputs:
        t (Tensor): 扩散步长张量，shape: [B]

    Outputs:
        emb (Tensor): 时间特征编码，shape: [B, dim]
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        # 计算频差衰减因子: \log(10000) / (d/2 - 1)
        emb = math.log(10000) / (half_dim - 1)
        # 计算指数衰减指数项: \exp(-i * emb)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        # 广播相乘计算正弦/余弦自变量: t_i * \omega_j -> shape: [B, half_dim]
        emb = t[:, None].float() * emb[None, :]
        # 拼接 sin 与 cos 分量 -> shape: [B, dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


# ==============================================================================
# 6. 顶层模型 / Pipeline 主体 (Top-level Architecture)
# ==============================================================================
class ConditionalTinyUNet(nn.Module):
    """
    类别条件控制的微型 U-Net 噪声预测网络 (Conditional Tiny U-Net)

    架构路径:
        x -> Down1 -> Down2 -> Add(Cond) -> Mid -> Up2 -> Up1 -> Pred_Noise

    Args:
        in_ch (int): 输入图像通道数，默认 3。
        num_classes (int): 条件生成的类别总数，默认 10。
        base (int): 第一层卷积的基础通道数，默认 64。
        cond_dim (int): 时间与类别条件嵌入的联合维度，默认 128。

    Inputs:
        x (Tensor): 当前时刻含噪图像 x_t，shape: [B, in_ch, H, W]
        t (Tensor): 扩散时间步 t，shape: [B]
        y (Tensor): 类别标签 condition y，shape: [B]

    Outputs:
        pred_noise (Tensor): 预测的高斯噪声 \epsilon_\theta，shape: [B, in_ch, H, W]
    """
    def __init__(
        self, 
        in_ch: int = IN_CHANNELS, 
        num_classes: int = NUM_CLASSES, 
        base: int = BASE_CHANNELS, 
        cond_dim: int = COND_DIM
    ):
        super().__init__()
        
        # 1. 条件融合投影 (Time + Class Embedding)
        self.time_embed = nn.Sequential(
            TimeEmbedding(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
        )
        self.class_embed = nn.Embedding(num_classes, cond_dim)

        # 2. 编码器下采样路径 (Encoder / Downsampling Path)
        # Stage 1: [B, in_ch, H, W] -> [B, base, H, W]
        self.down1 = nn.Sequential(
            nn.Conv2d(in_ch, base, kernel_size=3, padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(),
        )
        # Stage 2: [B, base, H, W] -> [B, base*2, H/2, W/2]
        self.down2 = nn.Sequential(
            nn.Conv2d(base, base * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
        )

        # 3. 中间特征瓶颈层 (Bottleneck Stage)
        # Stage Mid: [B, base*2, H/2, W/2] -> [B, base*2, H/2, W/2]
        self.mid = nn.Sequential(
            nn.Conv2d(base * 2, base * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
        )

        # 4. 解码器上采样路径 (Decoder / Upsampling Path)
        # Stage Up2: [B, base*2, H/2, W/2] -> [B, base, H, W]
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base * 2, base, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(),
        )
        # Stage Up1: [B, base, H, W] -> [B, in_ch, H, W]
        self.up1 = nn.Conv2d(base, in_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # --- 条件融合嵌入 ---
        # time_emb: [B, cond_dim], class_emb: [B, cond_dim]
        t_emb = self.time_embed(t)                                      # shape: [B, cond_dim]
        c_emb = self.class_embed(y)                                     # shape: [B, cond_dim]
        cond = t_emb + c_emb                                            # shape: [B, cond_dim]

        # --- 下采样编码阶段 ---
        h1 = self.down1(x)                                              # shape: [B, 64, 32, 32]
        h2 = self.down2(h1)                                             # shape: [B, 128, 16, 16]

        # --- 注入条件特征 (广播空间维度) ---
        # cond[:, :, None, None] 扩展为 [B, 128, 1, 1]
        h_cond = h2 + cond[:, :, None, None]                             # shape: [B, 128, 16, 16]

        # --- 瓶颈与解码上采样阶段 ---
        h_mid = self.mid(h_cond)                                        # shape: [B, 128, 16, 16]
        h_up = self.up2(h_mid)                                          # shape: [B, 64, 32, 32]
        pred_noise = self.up1(h_up)                                     # shape: [B, 3, 32, 32]

        return pred_noise


# ==============================================================================
# 7. 训练/推理逻辑与入口 (Training Pipeline & Execution)
# ==============================================================================
def main():
    # 1. 实例构建
    train_dataset = get_synthetic_dataset(num_samples=1000)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ConditionalTinyUNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    print(f"开始训练条件扩散模型... 运行设备: {DEVICE}")

    # 2. 训练循环
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for x0, y in train_loader:
            x0 = x0.to(DEVICE)                                          # shape: [B, 3, 32, 32]
            y = y.to(DEVICE)                                            # shape: [B]

            # 随机采样时间步 t ~ Uniform(0, T-1)
            t = torch.randint(0, T, (x0.size(0),), device=DEVICE)       # shape: [B]
            
            # 采样随机高斯噪声 \epsilon ~ N(0, I)
            noise = torch.randn_like(x0)                                # shape: [B, 3, 32, 32]
            
            # 生成扩散加噪样本 x_t
            xt = q_sample(x0, t, noise)                                 # shape: [B, 3, 32, 32]

            # 模型预测噪声 \epsilon_\theta(x_t, t, y)
            pred_noise = model(xt, t, y)                                # shape: [B, 3, 32, 32]
            
            # 计算预测噪声与真实噪声间的 MSE Loss
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1:02d}/{EPOCHS:02d}] | Conditional Noise MSE Loss: {avg_loss:.4f}")

    print("条件 DDPM 模型训练完成。")


if __name__ == "__main__":
    main()