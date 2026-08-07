"""
任务定义:
  - 任务编号: TASK-GEN-002
  - 任务名称: 条件时序变分自编码器 (Conditional Time-Series VAE / CVAE)
  - 领域分类: 生成式建模 / 时序数据生成与控制 (Generative Modeling / Time-Series Generation)

代表架构/算法:
  - Conditional Variational Autoencoder (CVAE) [Sohn et al., NIPS 2015]
  - Recurrent CVAE for Sequential Data [Bowman et al., EMNLP 2016]

核心思想与机制:
  1. 条件编解码机制 (Conditional Encoder-Decoder): 在已知条件变量 c 的前提下，学习数据 x 的分布 p(x|c)。
  2. 序列隐藏表示抽取 (Sequential Latent Extraction): 利用双向 LSTM Encoder 提取完整上下文序列表示，通过线性映射预测潜在高斯分布参数 (μ, log_σ²)。
  3. 重参数化技巧 (Reparameterization Trick): 引入标准正态噪声 ε ~ N(0, I)，将采样过程转化为可导的确定性变换 z = μ + ε ⊙ σ。
  4. 条件解码与重构 (Conditional Reconstruction): 将潜在向量 z 融合首帧条件 c_0 映射为 LSTM 解码器初始状态 (h_0, c_0)，并在每个时间步融合条件序列 c_t 逐步重构输入序列 x_hat。

数学公式/目标函数:
  - 变分下界/损失函数 (Variational Lower Bound / CVAE Loss):
      L(θ, φ; x, c) = E_{q_φ(z|x,c)}[log p_θ(x|z,c)] - β * D_KL(q_φ(z|x,c) || p(z))
  - 重构损失 (MSE Loss for Gaussian Likelihood):
      L_recon = (1 / N) * Σ ||x - x_hat||^2
  - KL 散度 (KL Divergence with Standard Normal Prior p(z) = N(0, I)):
      L_KL = -0.5 * (1 / N) * Σ_{j=1}^d (1 + log(σ_j^2) - μ_j^2 - σ_j^2)
  - 总体优化目标:
      L_total = L_recon + β * L_KL

数据输入规范:
  - 输入序列 (x): [Batch_Size, Seq_Len, Input_Dim]
  - 条件序列 (cond): [Batch_Size, Seq_Len, Cond_Dim]
  - 潜在特征 (z): [Batch_Size, Latent_Dim]
  - 重构输出 (recon_x): [Batch_Size, Seq_Len, Input_Dim]
"""

import os
import math
import logging
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# ==============================================================================
# 2. 超参数与全局配置 (Hyperparameters & Config)
# ==============================================================================
@dataclass
class CVAEConfig:
    """条件变分自编码器全局配置类"""
    # 数据参数
    data_path: str = "./data/train_data.npy"
    cond_data_path: str = "./data/conditions.npy"
    seq_length: int = 50
    input_dim: int = 10
    cond_dim: int = 5

    # CVAE 架构参数
    latent_dim: int = 32        # 潜在空间维度 (z_dim)
    hidden_dim: int = 256       # LSTM 隐藏层维度 (h_dim)
    num_layers: int = 2         # LSTM 层数
    dropout: float = 0.1        # 正则化 Dropout 率

    # 训练参数
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 200
    beta: float = 1.0           # KL 散度权重 (β-VAE)
    kl_annealing: bool = True   # 是否开启 KL 退火机制
    kl_annealing_steps: int = 50 # 退火完成所需的 Epoch 数

    # 硬件与保存路径
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "./checkpoints_cvae"
    log_dir: str = "./logs_cvae"
    save_interval: int = 10


# ==============================================================================
# 3. 数据处理与 Dataset 管道 (Data Pipeline & Utils)
# ==============================================================================
class SyntheticConditionalDataset(Dataset):
    """
    条件时序数据集 (Conditional Sequential Dataset)

    支持加载本地 `.npy` 文件或在文件不存在时动态生成高斯正弦波合成数据，
    用于验证 CVAE 的序列重建与条件控制能力。

    Args:
        data_path (str): 主序列数据路径。
        cond_path (str): 条件序列数据路径。
        seq_length (int): 切片滑动窗口序列长度。
        num_samples (int): 若文件不存在时生成的样本总条数。
        input_dim (int): 序列特征维度。
        cond_dim (int): 条件特征维度。
    """
    def __init__(
        self, 
        data_path: str, 
        cond_path: str, 
        seq_length: int = 50, 
        num_samples: int = 1000,
        input_dim: int = 10,
        cond_dim: int = 5
    ):
        super().__init__()
        self.seq_length = seq_length

        if os.path.exists(data_path) and os.path.exists(cond_path):
            self.data = np.load(data_path)
            self.conditions = np.load(cond_path)
        else:
            # 自动生成合成数据以确保代码开箱即用 (Out-of-the-box Execution)
            total_len = num_samples + seq_length
            t = np.linspace(0, 100, total_len)
            
            # 生成合成条件变量 (频率与振幅控制信号)
            cond_list = []
            for c in range(cond_dim):
                cond_list.append(np.sin((c + 1) * 0.05 * t))
            self.conditions = np.stack(cond_list, axis=-1).astype(np.float32)

            # 根据条件变量合成特征数据 (带有非线性相位调制与噪声)
            data_list = []
            for i in range(input_dim):
                signal = np.sin(0.1 * t + i) * self.conditions[:, i % cond_dim] + 0.05 * np.random.randn(total_len)
                data_list.append(signal)
            self.data = np.stack(data_list, axis=-1).astype(np.float32)

        assert len(self.data) == len(self.conditions), "Data and conditions length mismatch!"

    def __len__(self) -> int:
        return len(self.data) - self.seq_length

    def __getitem__(self, idx: int):
        """
        Inputs:
            idx (int): 数据索引。

        Outputs:
            sequence (Tensor): 主特征序列，shape: [Seq_Len, Input_Dim]
            condition (Tensor): 条件特征序列，shape: [Seq_Len, Cond_Dim]
        """
        sequence = self.data[idx : idx + self.seq_length]
        condition = self.conditions[idx : idx + self.seq_length]
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(condition, dtype=torch.float32)


# ==============================================================================
# 4. 核心子模块 / Encoder / Decoder (Sub-components)
# ==============================================================================
class BiLSTMEncoder(nn.Module):
    """
    条件双向 LSTM 编码器 (Conditional BiLSTM Encoder)

    结构说明:
        1. 将特征序列 x 与条件序列 c 在特征维度融合。
        2. 通过双向 LSTM 获取包含双向上下文信息的隐状态。
        3. 提取最后一个时间步的高阶表示，并行映射到潜在分布参数 (μ, log_σ²)。

    数学原理:
        - 输入拼接: x_concat = [x_t ; c_t] ∈ R^(d_x + d_c)
        - 隐状态演化: (h_t, c_t) = BiLSTM(x_concat_t, h_{t-1})
        - 均值映射: μ = W_μ * Act(W_h * h_last + b_h) + b_μ
        - 对数方差映射: log(σ²) = W_σ * Act(W_h * h_last + b_h) + b_σ

    Args:
        input_dim (int): 主特征维度 (d_x)。
        cond_dim (int): 条件特征维度 (d_c)。
        hidden_dim (int): LSTM 单向隐藏层维度。
        latent_dim (int): 潜在空间维度 (d_z)。
        num_layers (int): LSTM 循环层数。
        dropout (float): Dropout 概率。

    Inputs:
        x (Tensor): 主输入序列，shape: [B, Seq_Len, Input_Dim]
        cond (Tensor): 条件序列，shape: [B, Seq_Len, Cond_Dim]

    Outputs:
        mu (Tensor): 潜在高斯分布均值 μ，shape: [B, Latent_Dim]
        logvar (Tensor): 潜在高斯分布对数方差 log(σ²)，shape: [B, Latent_Dim]
    """
    def __init__(
        self, 
        input_dim: int, 
        cond_dim: int, 
        hidden_dim: int, 
        latent_dim: int, 
        num_layers: int = 2, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 输入拼接后的维度: d_x + d_c
        in_features = input_dim + cond_dim

        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 双向输出拼接维度为 hidden_dim * 2
        bi_hidden_dim = hidden_dim * 2

        # 均值 μ 提取分支
        self.fc_mu = nn.Sequential(
            nn.Linear(bi_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )

        # 对数方差 log(σ²) 提取分支
        self.fc_logvar = nn.Sequential(
            nn.Linear(bi_hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        # 1. 在特征维度拼接输入和条件: [B, Seq_Len, Input_Dim] + [B, Seq_Len, Cond_Dim] -> [B, Seq_Len, Input_Dim + Cond_Dim]
        combined = torch.cat([x, cond], dim=-1)

        # 2. 经过双向 LSTM: lstm_out shape: [B, Seq_Len, Hidden_Dim * 2]
        lstm_out, _ = self.lstm(combined)

        # 3. 提取序列最后一个时间步的表达: last_hidden shape: [B, Hidden_Dim * 2]
        last_hidden = lstm_out[:, -1, :]

        # 4. 预测参数
        mu = self.fc_mu(last_hidden)        # [B, Latent_Dim]
        logvar = self.fc_logvar(last_hidden)  # [B, Latent_Dim]

        return mu, logvar


class ConditionalLSTMDecoder(nn.Module):
    """
    条件单向 LSTM 解码器 (Conditional LSTM Decoder)

    结构说明:
        1. 将采样得到的潜在变量 z 与首帧条件 c_0 拼接，通过 FC 初始化 LSTM 的隐状态 (h_0, c_0)。
        2. 每个时间步以条件特征 c_t 作为输入，通过单向多层 LSTM 循环自回归生成。
        3. 投影层将 LSTM 隐藏状态映射为重构特征 x_hat。

    Args:
        latent_dim (int): 潜在空间维度 (d_z)。
        cond_dim (int): 条件特征维度 (d_c)。
        hidden_dim (int): LSTM 隐藏层维度。
        output_dim (int): 重构目标维度 (d_x)。
        num_layers (int): LSTM 层数。
        dropout (float): Dropout 概率。

    Inputs:
        z (Tensor): 潜在向量，shape: [B, Latent_Dim]
        cond (Tensor): 完整条件序列，shape: [B, Seq_Len, Cond_Dim]

    Outputs:
        recon_x (Tensor): 重构序列，shape: [B, Seq_Len, Output_Dim]
    """
    def __init__(
        self, 
        latent_dim: int, 
        cond_dim: int, 
        hidden_dim: int, 
        output_dim: int, 
        num_layers: int = 2, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 将潜在变量 z 与首帧条件 c_0 映射为 LSTM 的隐藏状态 (h_0, c_0)
        self.fc_init = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 每个时间步仅以条件 c_t 作为驱动输入
        self.lstm = nn.LSTM(
            input_size=cond_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 隐藏状态到输出特征的投影 Head
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = cond.shape

        # 1. 提取首帧条件 c_0: [B, Cond_Dim]
        c_0 = cond[:, 0, :]

        # 2. 拼接潜在向量 z 与 c_0: [B, Latent_Dim + Cond_Dim]
        z_cond = torch.cat([z, c_0], dim=-1)

        # 3. 计算初始隐藏状态投影: [B, Hidden_Dim * 2]
        init_state = self.fc_init(z_cond)

        # 4. 切分为 h0 和 c0 并扩展为多层格式: [Num_Layers, B, Hidden_Dim]
        h0 = init_state[:, :self.hidden_dim].unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = init_state[:, self.hidden_dim:].unsqueeze(0).repeat(self.num_layers, 1, 1)

        # 5. LSTM 序列解码: lstm_out shape: [B, Seq_Len, Hidden_Dim]
        lstm_out, _ = self.lstm(cond, (h0, c0))

        # 6. 映射为重构特征: recon_x shape: [B, Seq_Len, Output_Dim]
        recon_x = self.fc_out(lstm_out)

        return recon_x


# ==============================================================================
# 5. 顶层模型 / Pipeline 主体 (Top-level Architecture / Model)
# ==============================================================================
class TimeSeriesCVAE(nn.Module):
    """
    条件时序变分自编码器整体架构 (Time-Series CVAE Model)

    包含重参数化采样、前向重构、控制条件生成以及潜在空间插值等完整 API。

    Args:
        config (CVAEConfig): 配置对象。
    """
    def __init__(self, config: CVAEConfig):
        super().__init__()
        self.config = config

        self.encoder = BiLSTMEncoder(
            input_dim=config.input_dim,
            cond_dim=config.cond_dim,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            num_layers=config.num_layers,
            dropout=config.dropout
        )

        self.decoder = ConditionalLSTMDecoder(
            latent_dim=config.latent_dim,
            cond_dim=config.cond_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.input_dim,
            num_layers=config.num_layers,
            dropout=config.dropout
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧 (Reparameterization Trick)

        数学逻辑:
            z = μ + ε ⊙ σ,  其中 ε ~ N(0, I),  σ = exp(0.5 * log(σ²))

        Args:
            mu (Tensor): 均值 μ，shape: [B, Latent_Dim]
            logvar (Tensor): 对数方差 log(σ²)，shape: [B, Latent_Dim]

        Outputs:
            z (Tensor): 采样潜在向量，shape: [B, Latent_Dim]
        """
        std = torch.exp(0.5 * logvar)             # σ = exp(0.5 * logvar) -> [B, Latent_Dim]
        eps = torch.randn_like(std)               # ε ~ N(0, I) -> [B, Latent_Dim]
        z = mu + eps * std                       # [B, Latent_Dim]
        return z

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        前向传播 (Forward Pass)

        Inputs:
            x (Tensor): 原始序列，shape: [B, Seq_Len, Input_Dim]
            cond (Tensor): 条件序列，shape: [B, Seq_Len, Cond_Dim]

        Outputs:
            recon_x (Tensor): 重构序列，shape: [B, Seq_Len, Input_Dim]
            mu (Tensor): 潜在均值，shape: [B, Latent_Dim]
            logvar (Tensor): 潜在对数方差，shape: [B, Latent_Dim]
            z (Tensor): 采样潜在向量，shape: [B, Latent_Dim]
        """
        # 1. 编码推断潜在参数
        mu, logvar = self.encoder(x, cond)

        # 2. 重参数化采样
        z = self.reparameterize(mu, logvar)

        # 3. 条件重构解码
        recon_x = self.decoder(z, cond)

        return recon_x, mu, logvar, z

    @torch.no_grad()
    def generate(self, cond: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        条件控制生成 API (Conditional Synthesis)

        Args:
            cond (Tensor): 给定的控制条件序列，shape: [B, Seq_Len, Cond_Dim]
            num_samples (int): 每个条件重复采样生成的样本数。

        Outputs:
            generated (Tensor): 生成序列，shape: [B * num_samples, Seq_Len, Input_Dim]
        """
        self.eval()
        batch_size = cond.shape[0]
        total_samples = batch_size * num_samples

        # 从先验分布 p(z) = N(0, I) 采样潜在向量 z
        z = torch.randn(total_samples, self.config.latent_dim, device=cond.device)

        # 扩展条件序列以匹配采样数量
        cond_expanded = cond.repeat_interleave(num_samples, dim=0)

        # 条件解码
        generated = self.decoder(z, cond_expanded)
        return generated

    @torch.no_grad()
    def interpolate(self, cond1: torch.Tensor, cond2: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        """
        潜在空间与条件流形插值 API (Latent & Condition Interpolation)

        Args:
            cond1 (Tensor): 起始条件序列，shape: [1, Seq_Len, Cond_Dim]
            cond2 (Tensor): 终止条件序列，shape: [1, Seq_Len, Cond_Dim]
            num_steps (int): 插值步数。

        Outputs:
            interpolated (Tensor): 插值生成的序列，shape: [Num_Steps, Seq_Len, Input_Dim]
        """
        self.eval()
        z1 = torch.randn(1, self.config.latent_dim, device=cond1.device)
        z2 = torch.randn(1, self.config.latent_dim, device=cond1.device)

        alphas = torch.linspace(0, 1, num_steps, device=cond1.device)
        zs, conds = [], []

        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            cond_interp = (1 - alpha) * cond1 + alpha * cond2
            zs.append(z_interp)
            conds.append(cond_interp)

        zs_tensor = torch.cat(zs, dim=0)          # [Num_Steps, Latent_Dim]
        conds_tensor = torch.cat(conds, dim=0)    # [Num_Steps, Seq_Len, Cond_Dim]

        interpolated = self.decoder(zs_tensor, conds_tensor)
        return interpolated


# ==============================================================================
# 6. 损失函数与评估指标 (Loss & Metrics)
# ==============================================================================
class CVAELoss(nn.Module):
    """
    CVAE 联合损失函数 (MSE Recon Loss + β-Annealed KL Divergence)

    数学公式:
        L_recon = (1 / B) * Σ_{i=1}^B ||x_i - x_hat_i||_F^2
        L_KL = -0.5 * (1 / B) * Σ_{i=1}^B Σ_{j=1}^d (1 + log(σ_{ij}^2) - μ_{ij}^2 - σ_{ij}^2)
        L_total = L_recon + β * L_KL
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='sum')

    def forward(
        self, 
        recon_x: torch.Tensor, 
        x: torch.Tensor, 
        mu: torch.Tensor, 
        logvar: torch.Tensor, 
        beta: float = 1.0
    ):
        """
        Inputs:
            recon_x (Tensor): 重构序列，shape: [B, Seq_Len, Input_Dim]
            x (Tensor): 真实序列，shape: [B, Seq_Len, Input_Dim]
            mu (Tensor): 潜在均值，shape: [B, Latent_Dim]
            logvar (Tensor): 潜在对数方差，shape: [B, Latent_Dim]
            beta (float): KL 散度退火权重。

        Outputs:
            total_loss (Tensor): 标量总损失。
            recon_loss (Tensor): 标量重构损失。
            kl_loss (Tensor): 标量 KL 散度损失。
        """
        batch_size = x.shape[0]

        # 1. 重构损失 (按 Batch 平均)
        recon_loss = self.mse(recon_x, x) / batch_size

        # 2. KL 散度损失 (解析解公式)
        kl_element = 1.0 + logvar - mu.pow(2) - logvar.exp()
        kl_loss = -0.5 * torch.sum(kl_element) / batch_size

        # 3. 加权组合
        total_loss = recon_loss + beta * kl_loss

        return total_loss, recon_loss, kl_loss


# ==============================================================================
# 7. 训练/推理逻辑与可视化 (Training & Evaluation Utils)
# ==============================================================================
def visualize_samples(model: TimeSeriesCVAE, dataloader: DataLoader, epoch: int, config: CVAEConfig, logger: logging.Logger):
    """绘制原始序列、重构序列与生成序列的对比折线图"""
    model.eval()
    with torch.no_grad():
        x, cond = next(iter(dataloader))
        x = x[:4].to(config.device)
        cond = cond[:4].to(config.device)

        recon_x, _, _, _ = model(x, cond)
        generated = model.generate(cond, num_samples=1)

        fig, axes = plt.subplots(4, 3, figsize=(15, 10))
        for i in range(4):
            for dim in range(min(3, config.input_dim)):
                axes[i, 0].plot(x[i, :, dim].cpu().numpy(), label=f'Dim {dim}')
                axes[i, 1].plot(recon_x[i, :, dim].cpu().numpy(), label=f'Dim {dim}')
                axes[i, 2].plot(generated[i, :, dim].cpu().numpy(), label=f'Dim {dim}')

            axes[i, 0].set_title(f'Sample {i+1} Original')
            axes[i, 1].set_title(f'Sample {i+1} Reconstructed')
            axes[i, 2].set_title(f'Sample {i+1} Generated')
            for j in range(3):
                axes[i, j].legend(loc='upper right', fontsize=8)
                axes[i, j].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(config.log_dir, f'samples_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Sample visualization saved to: {save_path}")


def evaluate_model(config: CVAEConfig):
    """运行预训练模型推理评估与插值验证"""
    logger = logging.getLogger("CVAE")
    checkpoint_path = os.path.join(config.save_dir, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        logger.warning(f"No checkpoint found at {checkpoint_path}, skipping evaluation.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model = TimeSeriesCVAE(config).to(config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dataset = SyntheticConditionalDataset(config.data_path, config.cond_data_path, config.seq_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    x, cond = next(iter(dataloader))
    x, cond = x[:8].to(config.device), cond[:8].to(config.device)

    criterion = CVAELoss()
    with torch.no_grad():
        recon_x, mu, logvar, _ = model(x, cond)
        total_loss, recon_loss, kl_loss = criterion(recon_x, x, mu, logvar, config.beta)

        logger.info(f"=== Evaluation Results ===")
        logger.info(f"Reconstruction Loss: {recon_loss.item():.6f}")
        logger.info(f"KL Loss:             {kl_loss.item():.6f}")
        logger.info(f"Total Loss:          {total_loss.item():.6f}")

        # 进行潜在条件流形插值
        cond1, cond2 = cond[0:1], cond[1:2]
        interpolated = model.interpolate(cond1, cond2, num_steps=10)

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i in range(10):
            r, c = i // 5, i % 5
            for dim in range(min(3, config.input_dim)):
                axes[r, c].plot(interpolated[i, :, dim].cpu().numpy(), label=f'Dim {dim}')
            axes[r, c].set_title(f'Interp Step {i+1}', fontsize=10)
            axes[r, c].grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(config.log_dir, 'interpolation.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Interpolation visualization saved to: {save_path}")


# ==============================================================================
# 8. 训练 Execution 主入口 (Main Execution)
# ==============================================================================
def train(config: CVAEConfig):
    """全流程模型训练函数"""
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # 规范化 Logging 系统
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("CVAE")
    writer = SummaryWriter(config.log_dir)

    logger.info(f"Initializing Synthetic Conditional Dataset...")
    dataset = SyntheticConditionalDataset(
        data_path=config.data_path,
        cond_data_path=config.cond_data_path,
        seq_length=config.seq_length,
        input_dim=config.input_dim,
        cond_dim=config.cond_dim
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    logger.info(f"Dataset Size: {len(dataset)} samples | Total Batches: {len(dataloader)}")

    # 实例化模型与优化器
    model = TimeSeriesCVAE(config).to(config.device)
    criterion = CVAELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    # KL 散度线性退火策略
    def get_beta(epoch_idx: int) -> float:
        if not config.kl_annealing:
            return config.beta
        if epoch_idx < config.kl_annealing_steps:
            return config.beta * (epoch_idx + 1) / config.kl_annealing_steps
        return config.beta

    best_loss = float('inf')
    logger.info("Starting CVAE Model Training Loop...")

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss, epoch_recon, epoch_kl = 0.0, 0.0, 0.0
        current_beta = get_beta(epoch)

        pbar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{config.num_epochs}]", leave=False)
        for batch_idx, (x, cond) in enumerate(pbar):
            x = x.to(config.device)
            cond = cond.to(config.device)

            optimizer.zero_grad()
            recon_x, mu, logvar, _ = model(x, cond)

            loss, recon_l, kl_l = criterion(recon_x, x, mu, logvar, beta=current_beta)
            loss.backward()

            # 梯度裁剪防止 LSTM 梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_l.item()
            epoch_kl += kl_l.item()

            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Recon': f"{recon_l.item():.4f}", 'KL': f"{kl_l.item():.4f}"})

        scheduler.step()

        # 计算 Epoch 平均指标
        avg_loss = epoch_loss / len(dataloader)
        avg_recon = epoch_recon / len(dataloader)
        avg_kl = epoch_kl / len(dataloader)

        # 记录 TensorBoard
        writer.add_scalar('Loss/Total', avg_loss, epoch)
        writer.add_scalar('Loss/Reconstruction', avg_recon, epoch)
        writer.add_scalar('Loss/KL_Divergence', avg_kl, epoch)
        writer.add_scalar('Hyperparameters/Beta', current_beta, epoch)
        writer.add_scalar('Hyperparameters/LR', scheduler.get_last_lr()[0], epoch)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch [{epoch+1:03d}/{config.num_epochs}] | "
                f"Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | Beta: {current_beta:.3f}"
            )

        # 保存最优模型 Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }, os.path.join(config.save_dir, 'best_model.pth'))

        # 定期保存与可视化
        if (epoch + 1) % config.save_interval == 0:
            visualize_samples(model, dataloader, epoch + 1, config, logger)

    logger.info("Training pipeline finished successfully!")
    writer.close()


if __name__ == "__main__":
    exec_config = CVAEConfig(num_epochs=200, kl_annealing_steps=50, save_interval=20)
    train(exec_config)
    evaluate_model(exec_config)