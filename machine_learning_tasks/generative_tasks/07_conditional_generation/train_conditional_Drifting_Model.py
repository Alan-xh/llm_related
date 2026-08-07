"""
任务定义: TASK-DIFF-01 - 基于 Transformer 的条件时间序列生成与漂移扩散 (Conditional Time-Series Diffusion)
领域分类: 深度生成模型 / 时间序列生成 (Generative Models / Time-Series Generation)
代表架构: Conditional Transformer Diffusion Model (AdaLN/Concatenation Hybrid Integration) + Classifier-Free Guidance (CFG)

核心思想与机制:
1. 前向过程 (Forward Diffusion): 逐步将高斯噪声添加到原始序列 x_0 中，构造马尔可夫扩散链 x_1, x_2, ..., x_T。
2. 反向过程 (Reverse Denoising): 训练 Transformer 骨干网络 ε_θ(x_t, t, c) 预测在时刻 t 添加的噪声 ε。
3. 条件控制 (Conditioning): 支持序列级或时间步级条件 c 的注入，并结合 Classifier-Free Guidance (CFG) 在训练中以概率 p_drop 随机丢弃条件，实现无条件/有条件灵活采样。

数学公式/目标函数:
1. 前向加噪采样 (q-sampling):
   x_t = sqrt(bar{α}_t) * x_0 + sqrt(1 - bar{α}_t) * ε,   ε ~ N(0, I)
2. 训练优化目标 (Noise Prediction MSE Loss):
   L_simple(θ) = E_{t, x_0, ε, c} [ || ε - ε_θ(x_t, t, c) ||^2 ]
3. 反向均值重构 (p-sampling Mean):
   μ_θ(x_t, t, c) = (1 / sqrt(α_t)) * (x_t - (β_t / sqrt(1 - bar{α}_t)) * ε_θ(x_t, t, c))
4. 正弦时间编码 (Sinusoidal Positional Embedding):
   PE(t, 2i) = sin(t / 10000^(2i/d_model)),  PE(t, 2i+1) = cos(t / 10000^(2i/d_model))

数据输入规范:
- 目标序列 (x_0): [Batch_Size, Seq_Len, Input_Dim]
- 条件序列 (c):   [Batch_Size, Seq_Len, Cond_Dim]
- 时间步 (t):     [Batch_Size]
- 输出预测 (ε_θ): [Batch_Size, Seq_Len, Input_Dim]
"""

import os
import math
import logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


# ==============================================================================
# 3. 超参数与全局配置 (Hyperparameters & Config)
# ==============================================================================
class Config:
    """模型配置与训练超参数类"""
    # 数据参数
    data_path: str = "./data/train_data.npy"
    cond_data_path: str = "./data/conditions.npy"
    seq_length: int = 50
    input_dim: int = 10
    cond_dim: int = 5
    
    # 扩散模型参数
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    time_emb_dim: int = 256
    
    # 模型架构超参数
    model_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1
    
    # 训练超参数
    batch_size: int = 64
    lr: float = 1e-4
    num_epochs: int = 200
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gradient_accumulation_steps: int = 1
    
    # 条件增强 (CFG)
    cond_drop_rate: float = 0.1  # 条件丢弃率 (Classifier-Free Guidance)
    
    # 路径与日志
    save_dir: str = "./checkpoints_conditional"
    log_dir: str = "./logs_conditional"
    save_interval: int = 10


config = Config()


# ==============================================================================
# 4. 数据处理与 Dataset 管道 (Data Pipeline & Utils)
# ==============================================================================
class ConditionalDriftingDataset(Dataset):
    """
    条件漂移时间序列数据集

    Args:
        data_path (str): 目标序列数据路径 (.npy)。
        cond_path (str): 条件序列数据路径 (.npy)。
        seq_length (int): 滑动窗口切分的序列长度。

    Inputs:
        idx (int): 索引值。

    Outputs:
        sequence (Tensor): 目标时间序列，shape: [Seq_Len, Input_Dim]
        condition (Tensor): 条件时间序列，shape: [Seq_Len, Cond_Dim]
    """
    def __init__(self, data_path: str, cond_path: str, seq_length: int):
        super().__init__()
        self.seq_length = seq_length
        
        # 兼容性构建：若文件不存在，构建 Fake 数据保障 Pipeline 直接运行
        if not (os.path.exists(data_path) and os.path.exists(cond_path)):
            logging.warning("数据文件不存在，生成随机 Dummy 数据用于流程测试。")
            self.data = np.random.randn(1000, config.input_dim).astype(np.float32)
            self.conditions = np.random.randn(1000, config.cond_dim).astype(np.float32)
        else:
            self.data = np.load(data_path).astype(np.float32)
            self.conditions = np.load(cond_path).astype(np.float32)
            
        assert len(self.data) == len(self.conditions), "数据与条件序列总长度不匹配！"
        
    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_length)
    
    def __getitem__(self, idx: int):
        # 切片提取序列片段 [Seq_Len, Dim]
        sequence = self.data[idx : idx + self.seq_length]
        condition = self.conditions[idx : idx + self.seq_length]
        return torch.from_numpy(sequence), torch.from_numpy(condition)


# ==============================================================================
# 5. 核心子模块 / Encoder / Decoder (Sub-components)
# ==============================================================================
class SinusoidalPosEmb(nn.Module):
    """
    正弦位置/时间步嵌入模块

    数学原理:
        PE(t, 2i)   = sin(t / 10000^(2i/d))
        PE(t, 2i+1) = cos(t / 10000^(2i/d))

    Args:
        dim (int): 嵌入的目标特征维度 (d)。

    Inputs:
        timesteps (Tensor): 一维离散时间步张量，shape: [B]

    Outputs:
        emb (Tensor): 时间编码张量，shape: [B, Dim]
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # timesteps: [B]
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb) # [half_dim]
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)                                      # [B, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)                                    # [B, half_dim * 2]
        if self.dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))                                                     # [B, dim]
        return emb


class TimeConditionEncoder(nn.Module):
    """
    时间步与条件投影集成编码器

    Args:
        time_dim (int): 输入时间步正弦编码维度。
        cond_dim (int): 输入条件特征维度。
        model_dim (int): 模型隐层特征维度。

    Inputs:
        t (Tensor): 离散时间步，shape: [B]
        cond (Tensor): 条件特征张量，shape: [B, Seq_Len, Cond_Dim]

    Outputs:
        t_emb (Tensor): 时间步特征表示，shape: [B, Model_Dim]
        cond_emb (Tensor): 条件特征表示，shape: [B, Seq_Len, Model_Dim]
    """
    def __init__(self, time_dim: int, cond_dim: int, model_dim: int):
        super().__init__()
        self.time_pos_emb = SinusoidalPosEmb(time_dim)
        
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, model_dim * 4),
            nn.SiLU(),
            nn.Linear(model_dim * 4, model_dim)
        )
        
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, model_dim // 2),
            nn.SiLU(),
            nn.Linear(model_dim // 2, model_dim)
        )

    def forward(self, t: torch.Tensor, cond: torch.Tensor):
        # t: [B], cond: [B, Seq_Len, Cond_Dim]
        t_pos = self.time_pos_emb(t)                     # [B, Time_Dim]
        t_emb = self.time_mlp(t_pos)                     # [B, Model_Dim]
        
        cond_emb = self.cond_encoder(cond)               # [B, Seq_Len, Model_Dim]
        return t_emb, cond_emb


# ==============================================================================
# 6. 顶层模型 / Pipeline 主体 (Top-level Architecture / Model)
# ==============================================================================
class ConditionalDriftingModel(nn.Module):
    """
    基于 Transformer 的条件时间序列扩散去噪 Backbone

    架构原理:
        接收加噪序列 x_t、扩散时间步 t 与控制条件 c，拼接/融合特征后进入
        Multi-Head Attention Layers 提取时序依赖，预测注入的噪声 ε。

    Args:
        config (Config): 全局配置对象。

    Inputs:
        x (Tensor): 加噪序列，shape: [B, Seq_Len, Input_Dim]
        t (Tensor): 时间步，shape: [B]
        cond (Tensor): 条件序列，shape: [B, Seq_Len, Cond_Dim]
        cond_mask (Tensor, optional): 掩码 (1: 保留条件, 0: 丢弃)，shape: [B]

    Outputs:
        predicted_noise (Tensor): 预测噪声，shape: [B, Seq_Len, Input_Dim]
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # 编码子模块
        self.time_cond_encoder = TimeConditionEncoder(
            time_dim=config.time_emb_dim,
            cond_dim=config.cond_dim,
            model_dim=config.model_dim
        )
        
        # 序列输入投影
        self.input_proj = nn.Linear(config.input_dim, config.model_dim)
        
        # Transformer 骨干网络
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.model_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # 输出重构投影
        self.output_proj = nn.Linear(config.model_dim, config.input_dim)
        self.cond_drop_rate = config.cond_drop_rate

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, cond_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape  # [B, Seq_Len, Input_Dim]
        
        # 1. 动态条件掩码处理 (Classifier-Free Guidance Training)
        if self.training and self.cond_drop_rate > 0:
            if cond_mask is None:
                # 按 (1 - cond_drop_rate) 概率保留条件
                keep_prob = 1.0 - self.cond_drop_rate
                cond_mask = torch.bernoulli(torch.full((batch_size, 1, 1), keep_prob, device=x.device))
            else:
                cond_mask = cond_mask.view(-1, 1, 1)
            cond_effective = cond * cond_mask  # [B, Seq_Len, Cond_Dim]
        else:
            cond_effective = cond
            
        # 2. 提取特征嵌入
        t_emb, cond_emb = self.time_cond_encoder(t, cond_effective) # t_emb: [B, Model_Dim], cond_emb: [B, Seq_Len, Model_Dim]
        
        # 3. 特征投影与条件融合 (Additive Conditioning Fusion)
        h = self.input_proj(x)                                      # [B, Seq_Len, Model_Dim]
        h = h + t_emb.unsqueeze(1) + cond_emb                       # [B, Seq_Len, Model_Dim]
        
        # 4. 时序 Self-Attention 逻辑处理
        h = self.transformer(h)                                     # [B, Seq_Len, Model_Dim]
        
        # 5. 噪声映射输出
        out = self.output_proj(h)                                   # [B, Seq_Len, Input_Dim]
        return out


# ==============================================================================
# 7. 损失函数与扩散过程 Pipeline (Loss & Diffusion Process)
# ==============================================================================
class ConditionalDiffusionProcess:
    """
    DDPM 条件扩散与采样 Pipeline 封装

    Args:
        config (Config): 包含扩散步数 T 及 Beta 调度的全局配置。
    """
    def __init__(self, config: Config):
        self.config = config
        self.T = config.T
        
        # 线性 Beta 调度方案 (Linear Beta Schedule)
        self.beta = torch.linspace(config.beta_start, config.beta_end, config.T, device=config.device)
        self.alpha = 1.0 - self.beta                                                      # α_t = 1 - β_t
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)                                 # bar{α}_t = \prod_{s=1}^t α_s
        
        # 预计算推导系数
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)                                 # sqrt(bar{α}_t)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)                  # sqrt(1 - bar{α}_t)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alpha)                             # 1 / sqrt(α_t)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """
        前向加噪采样过程 q(x_t | x_0)

        Args:
            x0 (Tensor): 原始无噪序列，shape: [B, Seq_Len, Input_Dim]
            t (Tensor): 采样时间步，shape: [B]
            noise (Tensor, optional): 高斯噪声，shape: [B, Seq_Len, Input_Dim]

        Outputs:
            xt (Tensor): 加噪后的序列，shape: [B, Seq_Len, Input_Dim]
        """
        if noise is None:
            noise = torch.randn_like(x0)
            
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].reshape(-1, 1, 1)                      # [B, 1, 1]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].reshape(-1, 1, 1)  # [B, 1, 1]
        
        # 公式: x_t = sqrt(bar{α}_t) * x_0 + sqrt(1 - bar{α}_t) * ε
        xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
        return xt

    def p_sample(self, model: nn.Module, xt: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, cfg_scale: float = 1.0) -> torch.Tensor:
        """
        反向单步去噪采样过程 p_θ(x_{t-1} | x_t)

        Args:
            model (nn.Module): 去噪 Backbone。
            xt (Tensor): t 时刻张量，shape: [B, Seq_Len, Input_Dim]
            t (Tensor): 时间步，shape: [B]
            cond (Tensor): 条件序列，shape: [B, Seq_Len, Cond_Dim]
            cfg_scale (float): Classifier-Free Guidance 引导强度。

        Outputs:
            xt_prev (Tensor): t-1 时刻去噪张量，shape: [B, Seq_Len, Input_Dim]
        """
        # 1. 无条件与有条件引导预测 (Classifier-Free Guidance Inference)
        if cfg_scale > 1.0:
            uncond = torch.zeros_like(cond)
            pred_noise_cond = model(xt, t, cond)
            pred_noise_uncond = model(xt, t, uncond)
            # ε_total = ε_uncond + w * (ε_cond - ε_uncond)
            predicted_noise = pred_noise_uncond + cfg_scale * (pred_noise_cond - pred_noise_uncond)
        else:
            predicted_noise = model(xt, t, cond)

        # 2. 提取参数
        beta_t = self.beta[t].reshape(-1, 1, 1)                                           # [B, 1, 1]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].reshape(-1, 1, 1)  # [B, 1, 1]
        sqrt_recip_alpha_t = self.sqrt_recip_alpha[t].reshape(-1, 1, 1)                  # [B, 1, 1]

        # 3. 重构均值 μ_t = (1 / sqrt(α_t)) * (x_t - (β_t / sqrt(1 - bar{α}_t)) * ε_θ)
        mean = sqrt_recip_alpha_t * (xt - (beta_t / sqrt_one_minus_alpha_bar_t) * predicted_noise)

        # 4. 注入随机高斯方差 z ~ N(0, I)
        if t[0] > 0:
            noise = torch.randn_like(xt)
            sigma_t = torch.sqrt(beta_t)
            xt_prev = mean + sigma_t * noise
        else:
            xt_prev = mean
            
        return xt_prev

    @torch.no_grad()
    def ddpm_sample(self, model: nn.Module, cond: torch.Tensor, shape: tuple, cfg_scale: float = 1.0) -> torch.Tensor:
        """
        DDPM 反向迭代采样全管道

        Args:
            model (nn.Module): 训练好的 BackBone。
            cond (Tensor): 生成控制条件，shape: [B, Seq_Len, Cond_Dim]
            shape (tuple): 目标生成 Shape (B, Seq_Len, Input_Dim)。
            cfg_scale (float): CFG 引导权重。

        Outputs:
            xt (Tensor): 最终生成的重构时间序列，shape: [B, Seq_Len, Input_Dim]
        """
        batch_size = shape[0]
        xt = torch.randn(shape, device=cond.device)                                       # 初始高斯噪声
        
        for t_idx in reversed(range(self.T)):
            t_tensor = torch.full((batch_size,), t_idx, device=cond.device, dtype=torch.long)
            xt = self.p_sample(model, xt, t_tensor, cond, cfg_scale=cfg_scale)
            
        return xt


# ==============================================================================
# 8. 训练/推理逻辑与入口 (Training/Inference Execution)
# ==============================================================================
def visualize_samples(model: nn.Module, diffusion: ConditionalDiffusionProcess, epoch: int, logger: logging.Logger):
    """验证集图像生成与采样保存可视化"""
    model.eval()
    with torch.no_grad():
        cond_sample = torch.randn(4, config.seq_length, config.cond_dim, device=config.device)
        shape = (4, config.seq_length, config.input_dim)
        samples = diffusion.ddpm_sample(model, cond_sample, shape)
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        for i, ax in enumerate(axes.flat):
            if i < samples.shape[0]:
                for dim in range(min(3, config.input_dim)):
                    ax.plot(samples[i, :, dim].cpu().numpy(), label=f'Dim {dim}')
                ax.set_title(f'Epoch {epoch} Sample {i+1}')
                ax.legend()
                ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        os.makedirs(config.log_dir, exist_ok=True)
        save_path = os.path.join(config.log_dir, f'samples_epoch_{epoch}.png')
        plt.savefig(save_path)
        plt.close()
        logger.info(f"采样图像已保存至: {save_path}")


def train():
    """主训练 Pipeline 流程"""
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("Trainer")
    writer = SummaryWriter(config.log_dir)
    
    logger.info("初始化数据集与 DataLoader...")
    dataset = ConditionalDriftingDataset(config.data_path, config.cond_data_path, config.seq_length)
    if len(dataset) == 0:
        logger.error("数据集长度为0，终止训练流程！")
        return

    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=0, 
        drop_last=True
    )
    
    # 实例化网络与扩散引擎
    model = ConditionalDriftingModel(config).to(config.device)
    diffusion = ConditionalDiffusionProcess(config)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    criterion = nn.MSELoss()
    
    logger.info("开始执行条件扩散模型训练...")
    best_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch_idx, (x0, cond) in enumerate(pbar):
            x0 = x0.to(config.device)       # [B, Seq_Len, Input_Dim]
            cond = cond.to(config.device)   # [B, Seq_Len, Cond_Dim]
            
            # 1. 随机采样扩散时刻 t ~ U(0, T-1)
            t = torch.randint(0, config.T, (x0.shape[0],), device=config.device)
            
            # 2. 采样标准正态噪声 ε
            noise = torch.randn_like(x0)
            
            # 3. 前向加噪得到 x_t
            xt = diffusion.q_sample(x0, t, noise)
            
            # 4. 反向网络预测噪声
            predicted_noise = model(xt, t, cond)
            
            # 5. MSE 损失计算与梯度累积
            loss = criterion(predicted_noise, noise) / config.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
            epoch_loss += loss.item() * config.gradient_accumulation_steps
            pbar.set_postfix({"Loss": loss.item() * config.gradient_accumulation_steps, "LR": scheduler.get_last_lr()[0]})
            
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step()
        
        writer.add_scalar('Loss/Train', avg_loss, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        logger.info(f"Epoch {epoch+1} 完成 - 均方误差 Loss: {avg_loss:.6f}")
        
        # 保存最佳模型权重
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(config.save_dir, 'best_model.pth'))
            logger.info(f"记录新最佳模型，Loss: {best_loss:.6f}")
            
        # 定期绘图与保存 Checkpoint
        if (epoch + 1) % config.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }, os.path.join(config.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            visualize_samples(model, diffusion, epoch + 1, logger)
            
    writer.close()
    logger.info("训练过程全部完成。")


if __name__ == "__main__":
    train()