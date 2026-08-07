"""
任务定义:
    - 任务编号: TASK-GEN-FM-01
    - 任务名称: 条件流匹配图像生成 (Conditional Flow Matching Image Generation)
    - 领域分类: 生成式模型 (Generative Modeling / Continuous Normalizing Flows)

代表架构/算法:
    - Conditional Flow Matching (CFM) / Optimal Transport CFM (OT-CFM)
    - 主要论文: "Flow Matching for Generative Modeling" (Lipman et al., 2022)

核心思想与机制:
    Conditional Flow Matching (CFM) 是一种通过回归目标向量场来训练连续时间生成模型的方法。
    与传统 Diffusion 模型相比，CFM 直接模拟连续概率路径 $p_t(x)$，将先验分布 $p_0 = N(0, I)$ 映射到
    数据分布 $p_1 = q(x1)$。在条件生成场景下，通过引入类别标签 $y$ 的 Embedding 编码，模型学习条件向量场
    $v_t(x_t, t, y)$，并在推理时通过数值常微分方程 (ODE) 求解器（如 Euler 方法）沿着预测的向量场对样本进行推演积分。

数学公式/目标函数:
    1. 条件边缘概率路径:
       x_t = (1 - (1 - sigma_min) * t) * x0 + t * x1
       (本实现使用简洁的线性条件路径: x_t = (1 - t) * x0 + t * x1 + sigma * sqrt(t * (1 - t)) * epsilon)
    2. 目标条件向量场 (Target Vector Field):
       u_t(x_t | x0, x1) = x1 - x0 (或考虑 Gaussian Noise Perturbation 衍生项)
    3. CFM 目标损失函数 (Objective Loss):
       Loss_CFM = E_{t ~ U[0, 1], x0 ~ N(0, I), (x1, y) ~ q(x1, y)} [ || v_t(x_t, t, y) - (x1 - x0) ||^2 ]

数据输入规范:
    - 图像数据 (x1): [B, C, H, W] = [Batch_Size, 1, 28, 28] (例如 MNIST)
    - 时间步 (t): [B, 1] = [Batch_Size, 1]，取值区间 [0, 1]
    - 条件标签 (y): [B] = [Batch_Size]，类别索引 long tensor
"""

import os
import math
import argparse
from typing import Tuple, List, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm


# ==================== 2. 超参数与全局配置 (Hyperparameters & Config) ====================

class Config:
    """全局训练与模型配置类"""
    dataset_name: str = "mnist"
    image_size: int = 28
    in_channels: int = 1
    num_classes: int = 10
    
    # 架构配置
    time_dim: int = 128
    class_embed_dim: int = 64
    hidden_dims: List[int] = [512, 512, 512]
    
    # Flow Matching 参数
    sigma: float = 0.0  # 采用标准 Optimal Transport 路径 (sigma = 0.0)
    
    # 训练配置
    batch_size: int = 128
    epochs: int = 20
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "./checkpoints"
    sample_dir: str = "./samples"


# ==================== 3. 数据处理与 Dataset 管道 (Data Pipeline & Utils) ====================

def get_dataloader(config: Config) -> DataLoader:
    """
    构建与预处理图像数据管道
    
    Args:
        config (Config): 超参及全局配置实例
        
    Returns:
        DataLoader: PyTorch 数据加载器 [B, 1, 28, 28]
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 归一化至 [-1, 1] 区间
    ])
    
    if config.dataset_name.lower() == "mnist":
        dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    elif config.dataset_name.lower() == "fashion_mnist":
        dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    else:
        raise ValueError(f"不支持的数据集: {config.dataset_name}")
        
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, drop_last=True)


# ==================== 4. 核心子模块 / Encoder / Decoder (Sub-components) ====================

class TimestepEmbedding(nn.Module):
    """
    正弦位置编码/时间步 Embedding 模块
    
    数学原理 / 变换逻辑:
        Sinusoidal Positional Encoding:
        PE(t, 2i)   = sin(t / 10000^(2i / d_model))
        PE(t, 2i+1) = cos(t / 10000^(2i / d_model))

    Args:
        embed_dim (int): 时间步嵌入的目标特征维度
        
    Inputs:
        t (Tensor): 连续时间标量，shape: [B, 1]
        
    Outputs:
        t_embed (Tensor): 映射后的高维时间向量，shape: [B, embed_dim]
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t shape: [B, 1]
        half_dim = self.embed_dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb_factor) # [half_dim]
        emb = t * emb.unsqueeze(0)  # Shape 广播: [B, 1] * [1, half_dim] -> [B, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # Shape: [B, embed_dim]
        
        return self.mlp(emb)  # Shape: [B, embed_dim]


# ==================== 5. 顶层模型 / Pipeline 主体 (Top-level Architecture / Model) ====================

class ConditionalMLP(nn.Module):
    """
    条件 Vector Field 预测网络 (MLP 架构)

    数学原理 / 变换逻辑:
        学习函数 v_theta(x_t, t, y)，预测定义在连续概率路径上的条件向量场。
        输入特征由 Flatten 后的图像特征 x_t、时间嵌入 t_embed 以及类别嵌入 y_embed 拼接而成。

    Args:
        in_dim (int): 输入展平图像维度，如 28*28 = 784
        num_classes (int): 条件类别数量
        hidden_dims (List[int]): 隐层神经元维度列表
        time_dim (int): 时间 Embedding 维度
        class_embed_dim (int): 类别 Embedding 维度

    Inputs:
        x (Tensor): 当前时刻的状态张量，shape: [B, C*H*W] 或 [B, C, H, W]
        t (Tensor): 标量时间步，shape: [B, 1]
        y (Tensor): 类别条件标签，shape: [B]

    Outputs:
        v_pred (Tensor): 预测的目标向量场，shape 与输入 x 展平后保持一致 [B, C*H*W]
    """
    def __init__(
        self, 
        in_dim: int = 784, 
        num_classes: int = 10, 
        hidden_dims: List[int] = [512, 512, 512],
        time_dim: int = 128, 
        class_embed_dim: int = 64
    ):
        super().__init__()
        self.in_dim = in_dim
        
        # 1. 条件与时间嵌入组件
        self.time_embedder = TimestepEmbedding(embed_dim=time_dim)
        self.class_embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=class_embed_dim)
        
        # 2. 主体 MLP 映射网络
        layers = []
        prev_dim = in_dim + time_dim + class_embed_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.SiLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, in_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # 输入维度判定与 Shape 调整
        if x.dim() == 4:
            B, C, H, W = x.shape
            x_flat = x.view(B, -1)  # Shape: [B, C*H*W]
        else:
            x_flat = x  # Shape: [B, in_dim]
            
        t_flat = t.view(-1, 1) if t.dim() > 2 else t  # Shape 保障: [B, 1]
        
        # 特征编码计算
        t_embed = self.time_embedder(t_flat)       # Shape: [B, time_dim]
        y_embed = self.class_embedding(y)          # Shape: [B, class_embed_dim]
        
        # 特征拼接: [B, in_dim] + [B, time_dim] + [B, class_embed_dim] -> [B, in_dim + time_dim + class_embed_dim]
        h = torch.cat([x_flat, t_embed, y_embed], dim=-1)
        
        out = self.net(h)  # Shape: [B, in_dim]
        return out


class ConditionalConvFlowMatcher(nn.Module):
    """
    基于卷积 (Conv2d/ConvTranspose2d) 的条件 Vector Field 预测网络

    数学原理 / 变换逻辑:
        将条件 Embedding (t_embed + y_embed) 线性投影后，重塑为 Spatial Channels 广播加和至编码器中间特征，
        通过 UNet 式的对称编解码结构预测高维图像空间处的向量场。

    Args:
        in_channels (int): 输入图像通道数 (例如 单通道灰度图为 1)
        image_size (int): 图像边长 (28)
        num_classes (int): 条件类别数量
        time_dim (int): 时间 Embedding 维度
        class_embed_dim (int): 类别 Embedding 维度

    Inputs:
        x (Tensor): 输入图像张量，shape: [B, C, H, W]
        t (Tensor): 时间步张量，shape: [B, 1]
        y (Tensor): 条件标签，shape: [B]

    Outputs:
        v_pred (Tensor): 预测向量场，shape: [B, C, H, W]
    """
    def __init__(
        self, 
        in_channels: int = 1, 
        image_size: int = 28, 
        num_classes: int = 10, 
        time_dim: int = 128, 
        class_embed_dim: int = 64
    ):
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        
        # 条件编码器
        self.time_embedder = TimestepEmbedding(embed_dim=time_dim)
        self.class_embedding = nn.Embedding(num_classes, class_embed_dim)
        self.cond_proj = nn.Linear(time_dim + class_embed_dim, 256)

        # 编码模块 (Encoder)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),   # [B, 64, 28, 28]
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # [B, 128, 14, 14]
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# [B, 256, 7, 7]
            nn.SiLU(),
        )

        # 解码模块 (Decoder)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # [B, 128, 14, 14]
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 28, 28]
            nn.SiLU(),
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1)               # [B, in_channels, 28, 28]
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # 保障输入 Shape 正确: [B, C, H, W]
        if x.dim() == 2:
            x = x.view(-1, self.in_channels, self.image_size, self.image_size)

        t_flat = t.view(-1, 1) if t.dim() > 2 else t
        t_embed = self.time_embedder(t_flat)        # [B, time_dim]
        y_embed = self.class_embedding(y)           # [B, class_embed_dim]

        # 组合条件并进行空间维度广播
        cond = torch.cat([t_embed, y_embed], dim=-1)  # [B, time_dim + class_embed_dim]
        cond_feat = self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1)  # 扩展维度: [B, 256, 1, 1]

        # 编码特征提取
        h = self.encoder(x)  # [B, 256, 7, 7]
        h = h + cond_feat    # 条件信息融合 (Channel-wise Addition)

        # 解码重建向量场
        out = self.decoder(h)  # [B, in_channels, 28, 28]
        return out


# ==================== 6. 损失函数与 Flow Matching 算法逻辑 (Loss & Pipeline) ====================

class ConditionalFlowMatching:
    """
    条件 Flow Matching (CFM) 训练与求解框架

    数学原理:
        对于时刻 t in [0, 1]，采样先验噪声 x0 ~ N(0, I) 及真实数据点 x1 ~ q(x1)。
        定义的概率边际采样路径为:
            x_t = (1 - t) * x0 + t * x1 + sigma * sqrt(t*(1-t)) * epsilon
        对应的理论目标向量场为:
            u_t(x_t | x0, x1) = x1 - x0
        模型学习参数化向量场 v_theta(x_t, t, y) 以拟合目标 u_t。

    Args:
        sigma (float): 噪声扰动系数，当 sigma = 0.0 时即退化为精确的 Optimal Transport Flow Matching (OT-CFM)。
    """
    def __init__(self, sigma: float = 0.0):
        self.sigma = sigma

    def sample_times(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """采样均匀分布的时间步 t ~ Uniform(0, 1)"""
        return torch.rand(batch_size, 1, device=device)

    def get_conditional_probability_path(
        self, x1: torch.Tensor, x0: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        计算条件采样轨迹点 x_t

        Args:
            x1 (Tensor): 真实目标样本，shape: [B, D] 或 [B, C, H, W]
            x0 (Tensor): 初始标准高斯噪声，shape: 同 x1
            t (Tensor): 时间步，shape: [B, 1]

        Outputs:
            x_t (Tensor): 时刻 t 的轨迹点，shape: 同 x1
        """
        # 调整 t 的维度以便于广播匹配
        t_view = t
        while t_view.dim() < x1.dim():
            t_view = t_view.unsqueeze(-1)  # 调整维度匹配，例如 [B, 1, 1, 1]
            
        x_t = (1 - t_view) * x0 + t_view * x1
        if self.sigma > 0:
            noise = torch.randn_like(x1)
            x_t = x_t + self.sigma * torch.sqrt(t_view * (1 - t_view) + 1e-8) * noise
            
        return x_t

    def get_target_vector_field(self, x1: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """计算目标向量场 u_t(x_t | x0, x1) = x1 - x0"""
        return x1 - x0

    def compute_loss(
        self, model: nn.Module, x1: torch.Tensor, y: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算 Flow Matching 回归损失

        Args:
            model (nn.Module): 神经网络向量场预测器
            x1 (Tensor): 目标图像张量 [B, C, H, W] 或展平张量 [B, D]
            y (Tensor): 条件标签 [B]
            device (torch.device): 计算设备

        Outputs:
            loss (Tensor): 标量均方误差损失 MSE
            x_t (Tensor): 插值点轨迹
            predicted_vf (Tensor): 模型预测的向量场
        """
        batch_size = x1.shape[0]
        
        t = self.sample_times(batch_size, device)            # [B, 1]
        x0 = torch.randn_like(x1)                             # [B, ...], Standard Normal Noise
        x_t = self.get_conditional_probability_path(x1, x0, t) # [B, ...]
        target_vf = self.get_target_vector_field(x1, x0)      # [B, ...]

        # 预测向量场
        predicted_vf = model(x_t, t, y)

        # 保证预测维度与目标维度一致
        if predicted_vf.shape != target_vf.shape:
            target_vf = target_vf.view_as(predicted_vf)

        # 回归 Loss: || v_theta(x_t, t, y) - (x1 - x0) ||^2
        loss = F.mse_loss(predicted_vf, target_vf)
        return loss, x_t, predicted_vf


# ==================== 7. 训练/推理逻辑与入口 (Training/Inference Execution) ====================

@torch.no_grad()
def sample_conditional_flow_matching(
    model: nn.Module, 
    device: torch.device, 
    save_path: Optional[str] = None, 
    num_samples: int = 64, 
    steps: int = 50, 
    y: int = 0,
    image_shape: Tuple[int, int, int] = (1, 28, 28)
) -> torch.Tensor:
    """
    数值 ODE 积分采样逻辑 (Euler Method)

    数学原理:
        采样解常微分方程: dx = v_theta(x_t, t, y) dt
        采用 Euler 数值积分步骤:
        x_{t + dt} = x_t + v_theta(x_t, t, y) * dt ,  t 从 0 逐步递增至 1
    """
    model.eval()
    C, H, W = image_shape
    flat_dim = C * H * W
    
    # 1. 采样初始高斯噪声 x0 ~ N(0, I)
    x = torch.randn(num_samples, flat_dim, device=device)  # [B, D]
    y_tensor = torch.full((num_samples,), y, device=device, dtype=torch.long) # [B]
    
    dt = 1.0 / steps
    
    # 2. 沿着 predicted vector field 步进求解 ODE
    for i in range(steps):
        t_val = i / steps
        t = torch.full((num_samples, 1), t_val, device=device, dtype=torch.float32) # [B, 1]
        
        # 预测向量场 v(x_t, t, y)
        v = model(x, t, y_tensor)
        if v.dim() == 4:
            v = v.view(num_samples, -1)
            
        # 欧拉步进更新: x_{t + dt} = x_t + v * dt
        x = x + v * dt

    # 3. 后处理并恢复图像维度
    x = x.view(num_samples, C, H, W)
    x = (x + 1.0) / 2.0  # 反归一化至 [0, 1] 范围
    x = torch.clamp(x, 0.0, 1.0)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_image(x, save_path, nrow=int(math.sqrt(num_samples)))
        
    return x


def train(config: Config) -> None:
    """训练 Pipeline 主循环"""
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.sample_dir, exist_ok=True)
    device = torch.device(config.device)

    # 1. 初始化数据加载器与模型
    dataloader = get_dataloader(config)
    
    # 可切换选择 ConditionalMLP 或 ConditionalConvFlowMatcher
    model = ConditionalMLP(
        in_dim=config.image_size * config.image_size * config.in_channels,
        num_classes=config.num_classes,
        hidden_dims=config.hidden_dims,
        time_dim=config.time_dim,
        class_embed_dim=config.class_embed_dim
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    flow_matcher = ConditionalFlowMatching(sigma=config.sigma)

    print(f"网络构建成功，模型总可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 2. 训练主循环
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.epochs}")

        for batch_idx, (data, labels) in enumerate(progress_bar):
            data, labels = data.to(device), labels.to(device) # Shape: [B, 1, 28, 28], [B]
            
            optimizer.zero_grad()
            loss, x_t, pred_vf = flow_matcher.compute_loss(model, data, labels, device)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f"==> Epoch {epoch} 完成! 平均 Loss: {avg_loss:.6f}")

        # 3. 检查点保存与采样验证
        if epoch % 5 == 0 or epoch == config.epochs:
            ckpt_path = os.path.join(config.save_dir, f"cfm_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)
            
            # 条件采样展示 (针对数字 0 到 9 各生成一组样本)
            print(f"正在生成条件采样图像至 {config.sample_dir} ...")
            for c in range(min(5, config.num_classes)):  # 抽样前5个类别生成可视化
                sample_conditional_flow_matching(
                    model, device,
                    save_path=os.path.join(config.sample_dir, f"epoch_{epoch}_class_{c}.png"),
                    num_samples=16,
                    steps=50,
                    y=c
                )


def main():
    parser = argparse.ArgumentParser(description="Conditional Flow Matching (PyTorch Implementation)")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # 载入并覆盖默认配置
    config = Config()
    config.dataset_name = args.dataset
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.lr = args.lr
    config.device = args.device

    # 启动训练流程
    train(config)


if __name__ == "__main__":
    main()