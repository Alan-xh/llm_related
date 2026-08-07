"""
===================================================================================
任务定义: TASK-GEN-01 Conditional Flow Matching (CFM) 生成模型
代表架构: Continuous Normalizing Flows / Optimal Transport Conditional Flow Matching (OT-CFM)
核心思想: 学习一个依赖于时间 t 的向量场 v_t(x)，将先验噪声分布 p_0 = N(0, I) 连续推移至目标数据分布 p_1。
          相比于 Diffusion Model，Flow Matching 通过直线路径 (Straight Paths) 加速 ODE 采样过程。
数学公式:
  1. 条件概率路径 (Conditional Probability Path): 
     x_t = (1 - t) * x_0 + t * x_1 + sigma * sqrt(t * (1 - t)) * epsilon
  2. 目标向量场 (Conditional Vector Field): 
     u_t(x | x_0, x_1) = x_1 - x_0
  3. 连续流匹配损失函数 (CFM Loss): 
     L_CFM(theta) = E_{t ~ U(0,1), x_0 ~ N(0,I), x_1 ~ p_data} || v_theta(x_t, t) - (x_1 - x_0) ||^2
  4. 采样常微分方程 (ODE Sampling - Euler Method): 
     dx/dt = v_theta(x_t, t) => x_{t+dt} = x_t + v_theta(x_t, t) * dt
数据输入:
  - 训练阶段 x1: [B, C, H, W] 或 [B, Dim]
  - 时间步 t: [B, 1]
  - 预测向量场 v_t: 对应输入相同 Shape
===================================================================================
"""

import os
import math
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

# =================================================================================
# 3. 超参数与全局配置 (Hyperparameters & Config)
# =================================================================================
CONFIG = {
    "sigma": 0.1,             # 条件路径噪声扰动项标准差
    "time_dim": 128,          # 时间编码维度
    "hidden_dims": [512, 512, 512],
    "sample_steps": 100,      # Euler ODE 求解器采样步数
}

# =================================================================================
# 5. 核心子模块 / Encoder / Decoder (Sub-components)
# =================================================================================

class SimpleMLP(nn.Module):
    """
    多层感知机向量场预测网络 (MLP Vector Field Predictor)

    数学原理 / 变换 logic:
        将扁平化输入向量 x 与时间嵌入 t_embed 在特征维度拼接，
        通过多层带 SiLU 激活函数的全连接层预测速度矢量 v(x, t)。

    Args:
        dim (int):展平后的特征维度 (例如 28x28 = 784)。
        hidden_dims (list[int]): 隐藏层维度列表。
        time_dim (int): 时间嵌入维度。

    Inputs:
        x (Tensor): 状态张量, shape: [B, Dim]
        t (Tensor): 时间步张量, shape: [B, 1]

    Outputs:
        v_t (Tensor): 预测的向量场, shape: [B, Dim]
    """
    def __init__(self, dim=784, hidden_dims=[512, 512, 512], time_dim=128):
        super().__init__()
        
        # 时间步正弦/线性编码层
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )
        
        # 主干预测网络
        layers = []
        prev_dim = dim + time_dim  # [B, Dim + Time_Dim]
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.SiLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        # x: [B, Dim]
        # t: [B, 1]
        
        # 1. 计算时间嵌入
        t_embed = self.time_mlp(t)                   # Shape: [B, 1] -> [B, Time_Dim]
        
        # 2. 特征与时间融合
        h = torch.cat([x, t_embed], dim=-1)           # Shape: [B, Dim + Time_Dim]
        
        # 3. 预测速度场
        v_t = self.net(h)                            # Shape: [B, Dim]
        return v_t


class ConvFlowMatcher(nn.Module):
    """
    卷积神经网络向量场预测器 (CNN Vector Field Predictor)

    数学原理 / 变换 logic:
        用于处理 2D 图像张量。结合下采样 Encoder、时间条件融合（广播相加）以及上采样 Decoder 架构。

    Args:
        in_channels (int): 输入图像通道数。
        image_size (int): 图像分辨率大小 (H=W)。
        time_dim (int): 时间步高维投影维度。

    Inputs:
        x (Tensor): 噪声/中间图像张量, shape: [B, C, H, W]
        t (Tensor): 时间步张量, shape: [B, 1]

    Outputs:
        v_t (Tensor): 预测的 2D 向量场, shape: [B, C, H, W]
    """
    def __init__(self, in_channels=1, image_size=28, time_dim=128):
        super().__init__()
        
        self.image_size = image_size
        
        # 时间特征映射网络
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )
        
        # 图像特征编码器 (Encoder)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),            # -> [B, 64, 28, 28]
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),           # -> [B, 128, 14, 14]
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),          # -> [B, 256, 7, 7]
            nn.SiLU(),
        )
        
        # 时间嵌入与瓶颈层维度对齐投影
        self.time_proj = nn.Linear(time_dim, 256)
        
        # 图像特征解码器 (Decoder)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> [B, 128, 14, 14]
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> [B, 64, 28, 28]
            nn.SiLU(),
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1),             # -> [B, C, 28, 28]
        )

    def forward(self, x, t):
        # x: [B, C, H, W]
        # t: [B, 1]
        
        # 1. 计算时间嵌入并重构维数以匹配瓶颈层特征图
        t_embed = self.time_mlp(t)                                            # Shape: [B, Time_Dim]
        t_embed = self.time_proj(t_embed).unsqueeze(-1).unsqueeze(-1)        # Shape: [B, 256, 1, 1]
        
        # 2. 卷积编码
        h = self.encoder(x)                                                   # Shape: [B, 256, H/4, W/4]
        
        # 3. 加入时间物理信息 (Spatial Broadcast Addition)
        h = h + t_embed                                                       # Shape: [B, 256, H/4, W/4]
        
        # 4. 解码重构向量场
        out = self.decoder(h)                                                 # Shape: [B, C, H, W]
        return out

# =================================================================================
# 6. 顶层模型 / Pipeline 主体 (Top-level Architecture / Model)
# =================================================================================

class FlowMatching:
    """
    条件流匹配 (Conditional Flow Matching, CFM) 执行套件

    负责概率路径构建、目标向量场计算以及损失求解。

    Args:
        sigma (float): 路径小扰动噪声标准差，保证条件概率路径的数值稳定性。
    """
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def sample_times(self, batch_size, device):
        """
        在 [0, 1] 均匀分布中采样时间步 t ~ Uniform(0, 1)

        Args:
            batch_size (int): 批大小。
            device (torch.device): 计算设备。

        Outputs:
            t (Tensor): 时间步张量, shape: [B, 1]
        """
        return torch.rand(batch_size, 1, device=device)

    def get_conditional_probability(self, x1, x0, t):
        """
        根据线性插值与扰动构建条件概率路径点 x_t
        
        数学表达: 
            x_t = (1 - t) * x_0 + t * x_1 + \sigma * \sqrt{t(1-t)} * \epsilon
            其中 x_0 ~ N(0, I), x_1 ~ p_data, \epsilon ~ N(0, I)

        Args:
            x1 (Tensor): 真实数据点, shape: [B, Dim] 或 [B, C, H, W]
            x0 (Tensor): 初始标准高斯噪声点, shape: 同 x1
            t (Tensor): 时间步, shape: [B, 1]

        Outputs:
            x_t (Tensor): t 时刻的中间状态张量, shape: 同 x1
        """
        # 针对 4D 图像张量 [B, C, H, W] 或 2D 向量 [B, Dim] 进行维度广播适配
        view_shape = [t.shape[0]] + [1] * (x1.ndim - 1)
        t_reshaped = t.view(*view_shape)                                      # Shape: [B, 1, 1, 1] 或 [B, 1]
        
        # 概率路径构造: 从 x0 到 x1 线性插值加上高斯边际抖动
        eps = torch.randn_like(x1)                                            # Shape: 与 x1 一致
        x_t = (1 - t_reshaped) * x0 + t_reshaped * x1 + \
              self.sigma * torch.sqrt(t_reshaped * (1 - t_reshaped) + 1e-8) * eps
        return x_t

    def get_conditional_vector_field(self, x1, x0, t=None):
        """
        计算边缘条件目标向量场 u_t(x | x_0, x_1)
        
        数学表达: 
            u_t(x_t | x_0, x_1) = x_1 - x_0 (对应最优传输路径 Optimal Transport)

        Args:
            x1 (Tensor): 目标数据, shape: [B, ...]
            x0 (Tensor): 初始噪声, shape: [B, ...]
            t (Tensor, optional): 时间步, 默认不显式使用 (OT-CFM 下为常数梯度)

        Outputs:
            target_vf (Tensor): 物理真实切线速度场, shape: [B, ...]
        """
        # 梯度/速度向量: 从起点 x0 指向终点 x1
        target_vf = x1 - x0
        return target_vf

    def compute_loss(self, model, x1, device):
        """
        计算 Flow Matching 核心优化目标 MSE Loss

        Args:
            model (nn.Module): 神经网络速度场预测模型
            x1 (Tensor): 批次真实数据, shape: [B, Dim] 或 [B, C, H, W]
            device (torch.device): 运行设备

        Outputs:
            loss (Tensor): 标量 MSE 损失值
            x_t (Tensor): 当前时刻插值采样点
            predicted_vf (Tensor): 模型预测速度场
        """
        batch_size = x1.shape[0]
        
        # 1. 采样 t ~ Uniform(0, 1)
        t = self.sample_times(batch_size, device)                             # Shape: [B, 1]
        
        # 2. 采样 x0 ~ N(0, I)
        x0 = torch.randn_like(x1)                                             # Shape: [B, ...]
        
        # 3. 计算条件插值点 x_t
        x_t = self.get_conditional_probability(x1, x0, t)                     # Shape: [B, ...]
        
        # 4. 计算真实目标向量场 (OT Vector Field)
        target_vf = self.get_conditional_vector_field(x1, x0, t)             # Shape: [B, ...]
        
        # 5. 模型前向推理预测向量场
        predicted_vf = model(x_t, t)                                          # Shape: [B, ...]
        
        # 6. 计算均方误差损失 Loss = Mean(|| predicted_vf - target_vf ||^2)
        loss = torch.mean((predicted_vf - target_vf) ** 2)
        
        return loss, x_t, predicted_vf

# =================================================================================
# 7. 训练/推理逻辑与入口 (Training/Inference Execution)
# =================================================================================

def sample_flow_matching(model, device, save_path=None, num_samples=64, 
                         steps=100, image_shape=(1, 28, 28)):
    """
    基于欧拉数值积分 (Euler ODE Solver) 的 Flow Matching 推理采样过程

    数学原理:
        dx / dt = v_theta(x, t)
        x_{t + dt} = x_t + v_theta(x_t, t) * dt,  t 从 0 逐步递增至 1

    Args:
        model (nn.Module): 训练完成的速度场模型。
        device (torch.device): 设备。
        save_path (str): 采样结果保存路径。
        num_samples (int): 生成样本数量。
        steps (int): ODE 求解器积分步数 (Discretization Steps)。
        image_shape (tuple): 单张样本的维度结构 (C, H, W)。

    Outputs:
        x (Tensor): 生成的最终数据样本, shape: [num_samples, C, H, W]
    """
    model.eval()
    with torch.no_grad():
        # 1. 采样初始噪声 x_0 ~ N(0, I)
        if len(image_shape) == 1:
            x = torch.randn(num_samples, image_shape[0], device=device)       # Shape: [B, Dim]
        else:
            x = torch.randn(num_samples, *image_shape, device=device)         # Shape: [B, C, H, W]
            if isinstance(model, SimpleMLP):
                x = x.view(num_samples, -1)                                   # 展平 Shape: [B, C*H*W]
        
        dt = 1.0 / steps
        
        # 2. ODE 欧拉显式积分推进 (t: 0.0 -> 1.0)
        for i in range(steps):
            t_val = i / steps
            t = torch.ones(num_samples, 1, device=device) * t_val             # Shape: [B, 1]
            
            # 预测切线矢量
            v = model(x, t)                                                   # Shape: 同 x
            
            # 欧拉步进更新: x_{t+dt} = x_t + v * dt
            x = x + v * dt                                                    # Shape: 同 x
        
        # 3. 后处理与图像保存
        if save_path:
            if len(image_shape) == 3:
                x_img = x.view(num_samples, *image_shape)                     # 重构为图像 Shape: [B, C, H, W]
                x_img = (x_img + 1) / 2                                       # 反归一化 [-1, 1] -> [0, 1]
                x_img = torch.clamp(x_img, 0, 1)
                save_image(x_img, save_path, nrow=int(math.sqrt(num_samples)))
            else:
                torch.save(x, save_path.replace('.png', '.pt'))
        return x


def train_flow_matching(model, dataloader, epochs, lr=1e-3, device='cuda', 
                         save_dir='checkpoints', sample_dir='samples'):
    """
    Flow Matching 模型训练 Pipeline

    Args:
        model (nn.Module): 神经网络模型。
        dataloader (DataLoader): 数据加载器。
        epochs (int): 训练轮次。
        lr (float): 学习率。
        device (str/torch.device): 设备。
        save_dir (str): Checkpoint 保存文件夹。
        sample_dir (str): 采样图片保存文件夹。
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    flow_matcher = FlowMatching(sigma=CONFIG["sigma"])
    
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        model.train()
        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(device)
            
            # 如果使用 MLP 网络，需将图像展平 [B, C, H, W] -> [B, C*H*W]
            if isinstance(model, SimpleMLP) and len(data.shape) == 4:
                data = data.view(data.shape[0], -1)                           # Shape: [B, 784]
            
            optimizer.zero_grad()
            
            # 计算 Flow Matching 损失
            loss, x_t, predicted_vf = flow_matcher.compute_loss(model, data, device)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}')
        
        # 周期性保存权重与推理采样
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'{save_dir}/checkpoint_epoch_{epoch+1}.pt')
            
            sample_flow_matching(
                model=model, 
                device=device, 
                save_path=f'{sample_dir}/samples_epoch_{epoch+1}.png',
                steps=CONFIG["sample_steps"]
            )


def main():
    parser = argparse.ArgumentParser(description="Continuous Flow Matching PyTorch Implementation")
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist'])
    parser.add_argument('--model_type', type=str, default='mlp', choices=['mlp', 'conv'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    args = parser.parse_args()
    
    # 图像预处理: 映射至 [-1, 1] 空间
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    if args.dataset == 'mnist':
        dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    else:
        dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    # 根据类型实例化模型
    if args.model_type == 'mlp':
        model = SimpleMLP(dim=28*28, hidden_dims=CONFIG["hidden_dims"], time_dim=CONFIG["time_dim"]).to(args.device)
    else:
        model = ConvFlowMatcher(in_channels=1, image_size=28, time_dim=CONFIG["time_dim"]).to(args.device)
    
    print(f'Model initialized. Total parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # 启动训练
    train_flow_matching(
        model=model,
        dataloader=dataloader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir
    )

if __name__ == '__main__':
    main()