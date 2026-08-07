"""
任务定义: 任务 T-TS-01 - 连续时间序列/轨迹生成的扩散模型 (Diffusion Model for Trajectory & Time-Series)
代表架构: DDPM (Denoising Diffusion Probabilistic Models) + Transformer Backbones
核心思想: 
    1. 前向过程 (Forward Diffusion): 逐步将高斯噪声通过 Markov 链添加到原始连续序列 $x_0$ 中，得到 $x_t$。
    2. 反向过程 (Reverse Process): 借助 Transformer 编码器拟合网络 $\epsilon_\theta(x_t, t)$，从噪点中预测加性噪声。
    3. 漂移建模 (Drifting Dynamics): 将时间步嵌入与序列特征融合，自回归或全序列并行恢复时间序列中的漂移与动力学轨迹。

数学公式与目标函数映射:
    1. 前向加噪公式:
       x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
       - 代码映射: `xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise`
    2. 优化目标 (MSE Loss):
       L(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \right]
       - 代码映射: `loss = nn.MSELoss()(predicted_noise, noise)`
    3. 反向去噪均值公式:
       \mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)
       - 代码映射: `xt_prev = sqrt_recip_alpha_t * (xt - beta_t / sqrt_one_minus_alpha_bar_t * predicted_noise) + sigma_t * noise`

数据输入/输出规范:
    - Input x0: [Batch_Size, Seq_Len, Input_Dim] 连续序列数据
    - Timestep t: [Batch_Size] 随机采样的扩散步数
    - Output \epsilon_\theta: [Batch_Size, Seq_Len, Input_Dim] 预测的各维度噪声张量
"""

import os
import math
import logging
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt


# ===================================================================================
# 超参数与全局配置 (Hyperparameters & Config)
# ===================================================================================
class Config:
    """
    通用扩散与漂移模型运行配置参数类
    """
    # 数据参数
    data_path: str = "./data/train_data.npy"  # 训练数据路径
    seq_length: int = 50                      # 序列长度 (Seq_Len)
    input_dim: int = 10                       # 输入特征维度 (C_in)
    
    # 扩散模型参数
    T: int = 1000                             # 扩散步数 (Time Steps)
    beta_start: float = 1e-4                  # 初始 Beta 值
    beta_end: float = 0.02                    # 终止 Beta 值
    time_emb_dim: int = 256                   # 正弦位置编码维度
    
    # 模型架构
    model_dim: int = 128                      # Transformer 隐层维度 (d_model)
    num_heads: int = 4                        # 多头注意力机制头数
    num_layers: int = 4                       # Transformer 编码器层数
    dropout: float = 0.1                      # Dropout 率
    
    # 训练参数
    batch_size: int = 64                      # Batch Size
    lr: float = 1e-4                          # 初始学习率
    num_epochs: int = 200                     # 总 Epoch 数
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 保存和日志
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    save_interval: int = 10

config = Config()


# ===================================================================================
# 数据处理与 Dataset 管道 (Data Pipeline & Utils)
# ===================================================================================
class DriftingDataset(Dataset):
    """
    连续时间序列漂移数据集

    Args:
        data_path (str): .npy 数据文件路径
        seq_length (int): 截取的序列滑动窗口长度

    Inputs:
        无直接 forward 输入，通过 __getitem__ 提取样本

    Outputs:
        sequence (Tensor): 单个时间序列样本，shape: [Seq_Len, Input_Dim]
    """
    def __init__(self, data_path: str, seq_length: int):
        super().__init__()
        # 兼容无实际数据时的随机 Mock 初始化逻辑，保证工程无报错运行
        if os.path.exists(data_path):
            self.data = np.load(data_path)
        else:
            # 自动构建伪数据 [1000, Input_Dim]
            self.data = np.sin(np.linspace(0, 100, 1000)[:, None] + np.arange(config.input_dim))
        self.seq_length = seq_length
        
    def __len__(self) -> int:
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx: int) -> torch.FloatTensor:
        # [seq_length, input_dim]
        sequence = self.data[idx : idx + self.seq_length]
        return torch.FloatTensor(sequence)


# ===================================================================================
# 核心子模块 / Encoder / Decoder (Sub-components)
# ===================================================================================
class SinusoidalPosEmb(nn.Module):
    """
    正弦时间步嵌入模块 (Sinusoidal Timestep Embedding)

    数学原理 / 变换逻辑:
        PE(t, 2i)   = sin(t / 10000^(2i / d))
        PE(t, 2i+1) = cos(t / 10000^(2i / d))

    Args:
        dim (int): 目标编码维度

    Inputs:
        t (Tensor): 时间步一维张量，shape: [B]

    Outputs:
        emb (Tensor): 正弦嵌入特征张量，shape: [B, dim]
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t shape: [B]
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb) 
        # emb shape: [half_dim]
        
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0) 
        # emb shape: [B, half_dim]
        
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1) 
        # emb shape: [B, half_dim * 2]
        
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
            # emb shape: [B, dim] (当 dim 为奇数时补零)
        return emb


# ===================================================================================
# 顶层模型 / Pipeline 主体 (Top-level Architecture / Model)
# ===================================================================================
class DriftingModel(nn.Module):
    """
    漂移预测模型 - 基于 Transformer 架构的噪声预测网络 \epsilon_\theta(x_t, t)

    数学原理 / 变换逻辑:
        1. 将物理序列维度输入从 Input_Dim 线性映射至 Model_Dim。
        2. 计算扩散时间步 t 的 Sinusoidal 嵌入并经过 MLP 升维。
        3. 将时间步向量广播并融合至序列各特征点: h = Projection(x) + Time_Embedding。
        4. 通过多层 TransformerEncoder 提取全序列时序依赖关系。
        5. 通过 Projection 映射恢复出物理特征维度的噪声预测结果。

    Args:
        config (Config): 包含模型维度、头数、层数等配置参数的结构体

    Inputs:
        x (Tensor): 加噪序列张量，shape: [B, Seq_Len, Input_Dim]
        t (Tensor): 扩散步数张量，shape: [B]

    Outputs:
        out (Tensor): 预测的噪声张量，shape: [B, Seq_Len, Input_Dim]
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # 1. 时间步嵌入与 MLP
        self.time_emb_func = SinusoidalPosEmb(config.time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_emb_dim, config.model_dim * 4),
            nn.SiLU(),
            nn.Linear(config.model_dim * 4, config.model_dim)
        )
        
        # 2. 特征投影与 Transformer 主干
        self.input_proj = nn.Linear(config.input_dim, config.model_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.model_dim,
                nhead=config.num_heads,
                dim_feedforward=config.model_dim * 4,
                dropout=config.dropout,
                activation="silu",
                batch_first=True
            ),
            num_layers=config.num_layers
        )
        self.output_proj = nn.Linear(config.model_dim, config.input_dim)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 输入维度标记: x: [B, Seq_Len, Input_Dim], t: [B]
        
        # 1. 时间步编码变换
        t_emb = self.time_emb_func(t)                             # [B] -> [B, Time_Emb_Dim]
        t_emb = self.time_mlp(t_emb)                              # [B, Time_Emb_Dim] -> [B, Model_Dim]
        
        # 2. 输入特征映射
        h = self.input_proj(x)                                    # [B, Seq_Len, Input_Dim] -> [B, Seq_Len, Model_Dim]
        
        # 3. 融合时间特征 (广播至序列维度)
        h = h + t_emb.unsqueeze(1)                                # [B, Seq_Len, Model_Dim] + [B, 1, Model_Dim] -> [B, Seq_Len, Model_Dim]
        
        # 4. Transformer 序列特征抽取
        h = self.transformer(h)                                   # [B, Seq_Len, Model_Dim]
        
        # 5. 输出特征恢复
        out = self.output_proj(h)                                 # [B, Seq_Len, Model_Dim] -> [B, Seq_Len, Input_Dim]
        return out


# ===================================================================================
# 损失函数与扩散过程 (Loss & Diffusion Process)
# ===================================================================================
class DiffusionProcess:
    """
    DDPM 前向加噪 (q_sample) 与反向去噪采样 (p_sample) 管理模块

    Args:
        config (Config): 全局配置参数

    数学与代码映射:
        - \beta_t (Linear Schedule): `torch.linspace(beta_start, beta_end, T)`
        - \alpha_t = 1 - \beta_t
        - \bar{\alpha}_t = \prod_{s=1}^t \alpha_s: `alpha_bar = torch.cumprod(alpha, dim=0)`
    """
    def __init__(self, config: Config):
        self.config = config
        self.T = config.T
        
        # 预计算前向扩散参数链 (全为 1D Tensor: [T])
        self.beta = torch.linspace(config.beta_start, config.beta_end, config.T)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        
    def to(self, device: torch.device):
        """同步扩散参数至目标计算设备"""
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """
        前向加噪过程: q(x_t | x_0) = N(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)

        Inputs:
            x0 (Tensor): 原始真实数据, shape: [B, Seq_Len, Input_Dim]
            t (Tensor): 采样时间步, shape: [B]
            noise (Tensor, optional): 标准高斯噪声, shape: [B, Seq_Len, Input_Dim]

        Outputs:
            xt (Tensor): t 时刻的加噪序列, shape: [B, Seq_Len, Input_Dim]
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        # 重排维度以适应广播机制: [B] -> [B, 1, 1]
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].reshape(-1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].reshape(-1, 1, 1)
        
        # 计算加噪结果
        xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise  # [B, Seq_Len, Input_Dim]
        return xt
    
    @torch.no_grad()
    def p_sample(self, model: nn.Module, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        反向单步去噪过程: p_\theta(x_{t-1} | x_t)

        Inputs:
            model (nn.Module): 噪声预测模型 \epsilon_\theta
            xt (Tensor): t 时刻的序列张量, shape: [B, Seq_Len, Input_Dim]
            t (Tensor): 当前单步时间步 (全 Batch 相同), shape: [B]

        Outputs:
            xt_prev (Tensor): t-1 时刻的估计序列张量, shape: [B, Seq_Len, Input_Dim]
        """
        # 1. 预测噪声
        predicted_noise = model(xt, t)                           # [B, Seq_Len, Input_Dim]
        
        # 2. 提取并扩展参数维度: [B] -> [B, 1, 1]
        beta_t = self.beta[t].reshape(-1, 1, 1)
        alpha_t = self.alpha[t].reshape(-1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1, 1)
        sqrt_recip_alpha_t = 1.0 / torch.sqrt(alpha_t)
        
        # 3. 计算均值项与加性方差噪声
        if t[0] > 0:
            noise = torch.randn_like(xt)
            sigma_t = torch.sqrt(beta_t)
        else:
            noise = 0.0
            sigma_t = 0.0
            
        # 4. 执行反向推导更新: x_{t-1}
        # \mu_t = 1/\sqrt{\alpha_t} * (x_t - \beta_t / \sqrt{1 - \bar{\alpha}_t} * \epsilon_\theta)
        xt_prev = sqrt_recip_alpha_t * (xt - beta_t / torch.sqrt(1.0 - alpha_bar_t) * predicted_noise) + sigma_t * noise
        return xt_prev


# ===================================================================================
# 训练/推理逻辑与入口 (Training/Inference Execution)
# ===================================================================================
def train():
    """
    网络训练主逻辑函数
    """
    # 创建运行与保存路径
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # 建立 Logging 控制台与文件输出
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    writer = SummaryWriter(config.log_dir)
    
    logger.info("Initializing Data and Pipelines...")
    dataset = DriftingDataset(config.data_path, config.seq_length)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=0,  # 提升跨平台兼容性
        pin_memory=True
    )
    logger.info(f"Dataset Size: {len(dataset)} samples.")
    
    # 初始化模型与扩散模块
    model = DriftingModel(config).to(config.device)
    diffusion = DiffusionProcess(config).to(config.device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    criterion = nn.MSELoss()
    
    logger.info("Starting Training Loop...")
    best_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # batch shape: [B, Seq_Len, Input_Dim]
            batch = batch.to(config.device)
            batch_size = batch.shape[0]
            
            # 1. 均等采样时间步 t \sim Uniform({0, ..., T-1})
            t = torch.randint(0, config.T, (batch_size,), device=config.device)
            
            # 2. 采样加性高斯噪声 \epsilon
            noise = torch.randn_like(batch)                      # [B, Seq_Len, Input_Dim]
            
            # 3. 计算前向加噪张量 x_t
            xt = diffusion.q_sample(batch, t, noise)             # [B, Seq_Len, Input_Dim]
            
            # 4. 预测噪声 \epsilon_\theta(x_t, t)
            predicted_noise = model(xt, t)                       # [B, Seq_Len, Input_Dim]
            
            # 5. 计算损失
            loss = criterion(predicted_noise, noise)
            
            # 6. 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss.item():.6f}",
                'lr': f"{scheduler.get_last_lr()[0]:.6f}"
            })
            
            # Step 级日志
            if batch_idx % 100 == 0:
                global_step = epoch * len(dataloader) + batch_idx
                writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
                writer.add_scalar('Train/LearningRate', scheduler.get_last_lr()[0], global_step)
        
        # Epoch 总结与学习率更新
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step()
        
        logger.info(f"Epoch [{epoch+1}/{config.num_epochs}] Finished - Avg Loss: {avg_loss:.6f}")
        writer.add_scalar('Epoch/AvgLoss', avg_loss, epoch)
        
        # 保存最优模型 Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': config
            }, os.path.join(config.save_dir, 'best_model.pth'))
            logger.info(f"--> Best Checkpoint saved with Loss: {best_loss:.6f}")
        
        # 定期保存定期检查点
        if (epoch + 1) % config.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }, os.path.join(config.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            logger.info(f"--> Interval Checkpoint saved for epoch {epoch+1}")
            
    # 保存最终 Checkpoint
    torch.save({
        'epoch': config.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'config': config
    }, os.path.join(config.save_dir, 'final_model.pth'))
    
    logger.info("Training Execution Finished Successfully!")
    writer.close()


def visualize_training():
    """
    绘制并保存训练损失曲线
    """
    log_file = os.path.join(config.log_dir, 'training.log')
    if not os.path.exists(log_file):
        print("Warning: Log file not found for visual plotting.")
        return
    
    losses = []
    with open(log_file, 'r') as f:
        for line in f:
            if 'Finished - Avg Loss:' in line:
                loss_str = line.split('Finished - Avg Loss:')[-1].strip()
                losses.append(float(loss_str))
    
    if losses:
        plt.figure(figsize=(10, 6))
        plt.plot(losses, label='Training Avg Loss', color='b', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Drifting Diffusion Model Training Loss Curve')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        curve_path = os.path.join(config.log_dir, 'loss_curve.png')
        plt.savefig(curve_path)
        plt.close()
        print(f"Loss curve plot saved to: {curve_path}")


if __name__ == "__main__":
    train()
    visualize_training()