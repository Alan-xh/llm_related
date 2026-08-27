"""
任务 ID       : SEQ-GEN-CGAN-001
任务名称     : 基于 CGAN / WGAN-GP 的条件时序序列生成
领域         : 时间序列 / 生成模型 / 序列分析
架构         : 带 WGAN-GP 梯度惩罚的条件生成对抗网络 (Conditional GAN)
参考文献     : - Mirza, M., & Osindero, S. (2014). Conditional Generative Adversarial Nets.
               - Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein GAN.
               - Gulrajani, I., et al. (2017). Improved Training of Wasserstein GANs.

核心概念与机制:
    本模块实现了一个专为多变量时间序列生成设计的条件生成对抗网络 (CGAN)。
    生成器以时间特征 (cond) 和全局隐空间噪声向量 (z) 为条件，合成逼真的时间特征序列。
    判别器在给定相同条件输入的情况下，评估序列的真实度。
    模型训练的稳定性通过带有梯度惩罚的 Wasserstein 距离 (WGAN-GP) 来保障。

数学公式表达:
    1. 带梯度惩罚的 Wasserstein GAN 目标函数 (WGAN-GP):
       min_G max_D  E_{x~P_r}[D(x|c)] - E_{\hat{x}~P_g}[D(\hat{x}|c)] - \lambda E_{\tilde{x}~P_{\tilde{x}}}[(||\nabla_{\tilde{x}} D(\tilde{x}|c)||_2 - 1)^2]
       其中 \tilde{x} = \epsilon x + (1 - \epsilon) \hat{x}，\epsilon ~ U(0, 1)。

    2. 生成器初始隐状态投影:
       h_0 = MLP([z, c_0]),  c_0 = 0
       LSTM 映射: h_t, c_t = LSTM(c_t, (h_{t-1}, c_{t-1}))
       输出投影: x_t = Tanh(MLP(h_t))

数据输入 / 输出规范:
    真实数据张量 (x)     : 形状 [B, L, D_in]   - 连续时间序列数据。
    条件张量 (c)         : 形状 [B, L, D_cond] - 外生时间驱动特征。
    隐空间噪声张量 (z)   : 形状 [B, D_z]      - 标准正态分布噪声向量 ~ N(0, I)。
    生成数据输出         : 形状 [B, L, D_in]   - 与条件对齐的合成序列。
    判别器评分输出       : 形状 [B, 1]        - 标量评分 (WGAN-GP 下的无界 Logit 值)。
"""

import os
import math
import logging
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# ==============================================================================
# 3. 超参数与全局配置
# ==============================================================================

class CGANConfig:
    """
    条件时序 GAN 的全局配置类。
    封装了数据、模型架构、训练循环以及日志记录的相关参数。
    """
    def __init__(self):
        # 数据参数
        self.data_path: str = "./data/train_data.npy"
        self.cond_data_path: str = "./data/conditions.npy"
        self.seq_length: int = 50       # 序列长度 L
        self.input_dim: int = 10        # 目标特征维度 D_in
        self.cond_dim: int = 5          # 条件特征维度 D_cond

        # 生成器架构参数
        self.latent_dim: int = 100      # 隐空间噪声维度 D_z
        self.gen_hidden_dim: int = 256  # 生成器 LSTM 隐层维度
        self.gen_num_layers: int = 3    # 生成器 LSTM 层数

        # 判别器架构参数
        self.dis_hidden_dim: int = 256  # 判别器 LSTM 隐层维度
        self.dis_num_layers: int = 3    # 判别器双向 LSTM 层数
        self.dis_dropout: float = 0.2   # 判别器 Dropout 比率

        # 训练超参数
        self.batch_size: int = 64
        self.lr_g: float = 2e-4         # 生成器学习率
        self.lr_d: float = 2e-4         # 判别器学习率
        self.beta1: float = 0.5         # Adam 优化器 beta1 超参数
        self.beta2: float = 0.9         # Adam 优化器 beta2 超参数
        self.weight_decay: float = 1e-5
        self.num_epochs: int = 200
        self.n_critic: int = 1          # 每次更新 G 前判别器 D 的更新次数
        self.gp_weight: float = 10.0    # WGAN-GP 梯度惩罚系数 lambda
        self.label_smoothing: float = 0.1 # 真实目标的标签平滑因子

        # 执行环境
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 日志与持久化
        self.save_dir: str = "./checkpoints_cgan"
        self.log_dir: str = "./logs_cgan"
        self.save_interval: int = 10


# ==============================================================================
# 4. 数据处理与数据集流水线
# ==============================================================================

class ConditionalDataset(Dataset):
    """
    针对条件序列数据的滑动窗口数据集。

    输入:
        data_path (str): 目标时间序列 numpy 数组路径 [N_samples, D_in]。
        cond_path (str): 条件时间序列 numpy 数组路径 [N_samples, D_cond]。
        seq_length (int): 滑动窗口时间深度 (L)。
    """
    def __init__(self, data_path: str, cond_path: str, seq_length: int = 50):
        super().__init__()
        self.seq_length = seq_length

        if os.path.exists(data_path) and os.path.exists(cond_path):
            self.data = np.load(data_path)
            self.conditions = np.load(cond_path)
        else:
            # 数据路径不存在时生成合成虚拟数据，以便独立运行演示
            logging.warning(f"未找到数据路径。将在运行时自动生成合成虚拟数据。")
            num_samples = 1000
            self.data = np.sin(np.linspace(0, 100, num_samples)[:, None] + np.arange(10)[None, :]).astype(np.float32)
            self.conditions = np.cos(np.linspace(0, 100, num_samples)[:, None] + np.arange(5)[None, :]).astype(np.float32)

        assert len(self.data) == len(self.conditions), (
            f"时间长度不匹配: 数据 ({len(self.data)}) vs 条件 ({len(self.conditions)})"
        )
        self.num_windows = len(self.data) - self.seq_length + 1

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提取长度为 `seq_length` 的序列窗口。

        输出:
            sequence (Tensor)  : 真实时间特征，形状: [L, D_in]
            condition (Tensor) : 时间条件特征，形状: [L, D_cond]
        """
        sequence = self.data[idx : idx + self.seq_length]
        condition = self.conditions[idx : idx + self.seq_length]
        return torch.from_numpy(sequence).float(), torch.from_numpy(condition).float()


# ==============================================================================
# 5. 核心子组件 / 编码器 / 解码器
# ==============================================================================

class ConditionalGenerator(nn.Module):
    """
    条件循环生成器模块。

    数学变换过程:
        1. 上下文向量初始化:
           h_0 = MLP_in([z, c_0])，其中 z ~ N(0, I)，c_0 = c[:, 0, :]
           映射形状: [B, D_z + D_cond] -> [B, D_gen_hidden]
           扩展为多层 LSTM 状态: [Num_Layers, B, D_gen_hidden]

        2. 条件序列上的循环展开 (Recurrent Unrolling):
           H, (h_n, c_n) = LSTM(C, (h_0, c_0_zeros))
           其中 C 形状: [B, L, D_cond]，H 形状: [B, L, D_gen_hidden]

        3. 每个时间步的特征投影:
           x_t = Tanh(MLP_out(H_t))
           其中 x_t 形状: [B, D_in]

    参数:
        config (CGANConfig): 全局模型配置实例。

    输入:
        z (Tensor): 隐空间噪声张量，形状: [B, D_z]
        cond (Tensor): 时间条件张量，形状: [B, L, D_cond]

    输出:
        output (Tensor): 合成的时间序列，形状: [B, L, D_in]
    """
    def __init__(self, config: CGANConfig):
        super().__init__()
        self.config = config

        # 将 z 和 c_0 映射到隐层维度的初始隐状态投影器
        self.fc_input = nn.Sequential(
            nn.Linear(config.latent_dim + config.cond_dim, config.gen_hidden_dim),
            nn.BatchNorm1d(config.gen_hidden_dim),
            nn.SiLU()
        )

        # 自回归序列驱动模块
        self.lstm = nn.LSTM(
            input_size=config.cond_dim,
            hidden_size=config.gen_hidden_dim,
            num_layers=config.gen_num_layers,
            batch_first=True,
            dropout=0.1 if config.gen_num_layers > 1 else 0.0
        )

        # 非线性输出特征投影头
        self.fc_out = nn.Sequential(
            nn.Linear(config.gen_hidden_dim, config.gen_hidden_dim * 2),
            nn.SiLU(),
            nn.BatchNorm1d(config.gen_hidden_dim * 2),
            nn.Linear(config.gen_hidden_dim * 2, config.input_dim),
            nn.Tanh()  # 将输出约束在 [-1, 1] 范围内
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # z 形状: [B, D_z]
        # cond 形状: [B, L, D_cond]
        batch_size, seq_len, _ = cond.shape

        # 提取初始条件时间步: [B, D_cond]
        cond_first = cond[:, 0, :]

        # 拼接隐空间噪声与初始条件: [B, D_z + D_cond]
        init_input = torch.cat([z, cond_first], dim=-1)

        # 计算初始隐状态表示: [B, D_gen_hidden]
        init_state = self.fc_input(init_input)

        # 将隐状态扩展到多层 LSTM 维度: [Num_Layers, B, D_gen_hidden]
        h0 = init_state.unsqueeze(0).repeat(self.config.gen_num_layers, 1, 1)
        c0 = torch.zeros_like(h0)  # 初始细胞状态 (Cell State) 设为 0

        # 将条件输入传入循环网络: lstm_out 形状 -> [B, L, D_gen_hidden]
        lstm_out, _ = self.lstm(cond, (h0, c0))

        # 展平序列与批次维度，以便进行并行 MLP 投影
        # [B, L, D_gen_hidden] -> [B * L, D_gen_hidden]
        flat_lstm_out = lstm_out.reshape(-1, self.config.gen_hidden_dim)

        # 投影到输入特征维度: [B * L, D_in]
        flat_output = self.fc_out(flat_lstm_out)

        # 重构形状还原为目标序列张量: [B, L, D_in]
        output = flat_output.view(batch_size, seq_len, self.config.input_dim)

        return output


class ConditionalDiscriminator(nn.Module):
    """
    带有时间上下文特征的双向循环判别器模块。

    数学变换过程:
        1. 特征-条件拼接:
           U_t = [X_t || C_t]，t 范围为 1..L
           组合张量形状: [B, L, D_in + D_cond]

        2. 双向序列编码:
           H_seq = BiLSTM(U)
           其中 H_seq 形状: [B, L, 2 * D_dis_hidden]

        3. 全局时间表示聚合:
           H_final = H_seq[:, -1, :]  # 提取最后一个时间步切片
           形状: [B, 2 * D_dis_hidden]

        4. 真实度评分投影:
           Score = LeakyReLU(Linear(Dropout(LeakyReLU(Linear(H_final)))))
           输出评分形状: [B, 1]

    参数:
        config (CGANConfig): 全局配置参数。

    输入:
        x (Tensor): 目标真实或合成的时间序列，形状: [B, L, D_in]
        cond (Tensor): 时间条件因子，形状: [B, L, D_cond]

    输出:
        score (Tensor): 真实度 Logits/评分，形状: [B, 1]
    """
    def __init__(self, config: CGANConfig):
        super().__init__()
        self.config = config

        input_size = config.input_dim + config.cond_dim

        # 用于长程时间建模的双向 LSTM 编码器
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.dis_hidden_dim,
            num_layers=config.dis_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dis_dropout if config.dis_num_layers > 1 else 0.0
        )

        # 高容量分类器头
        self.fc_out = nn.Sequential(
            nn.Linear(config.dis_hidden_dim * 2, config.dis_hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(config.dis_dropout),
            nn.Linear(config.dis_hidden_dim * 2, config.dis_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(config.dis_dropout),
            nn.Linear(config.dis_hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x 形状   : [B, L, D_in]
        # cond 形状: [B, L, D_cond]

        # 沿特征维度拼接: [B, L, D_in + D_cond]
        combined = torch.cat([x, cond], dim=-1)

        # 处理完整序列: lstm_out 形状 -> [B, L, 2 * D_dis_hidden]
        lstm_out, _ = self.lstm(combined)

        # 时间池化: 选取序列最后一个隐层向量: [B, 2 * D_dis_hidden]
        last_out = lstm_out[:, -1, :]

        # 计算真实度评分: score 形状 -> [B, 1]
        out = self.fc_out(last_out)

        return out


# ==============================================================================
# 6. 顶层模型 / 流水线封装
# ==============================================================================

class CGANPipeline:
    """
    高层流水线，封装了生成器、判别器、优化器以及 WGAN-GP 的训练步骤逻辑。
    """
    def __init__(self, config: CGANConfig):
        self.config = config
        self.generator = ConditionalGenerator(config).to(config.device)
        self.discriminator = ConditionalDiscriminator(config).to(config.device)

        # 带动量超参数调优的 AdamW 优化器
        self.optimizer_g = optim.AdamW(
            self.generator.parameters(),
            lr=config.lr_g,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
        self.optimizer_d = optim.AdamW(
            self.discriminator.parameters(),
            lr=config.lr_d,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        # 带 Logits 的二元交叉熵损失函数
        self.bce_criterion = nn.BCEWithLogitsLoss()

    def gradient_penalty(self, real_data: torch.Tensor, fake_data: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        计算 WGAN-GP 梯度惩罚项，以强制满足 1-Lipschitz 连续性约束。

        公式:
            GP = E [(|| \nabla_{\tilde{x}} D(\tilde{x}, cond) ||_2 - 1)^2]
            其中 \tilde{x} = \epsilon * real + (1 - \epsilon) * fake

        输入:
            real_data (Tensor): [B, L, D_in]
            fake_data (Tensor): [B, L, D_in]
            cond (Tensor)     : [B, L, D_cond]

        输出:
            penalty (Tensor)  : 标量梯度惩罚张量。
        """
        batch_size = real_data.size(0)

        # 均匀分布随机采样权重: [B, 1, 1]
        epsilon = torch.rand(batch_size, 1, 1, device=self.config.device)
        epsilon = epsilon.expand_as(real_data)

        # 凸组合插值: [B, L, D_in]
        interpolated = (epsilon * real_data + (1 - epsilon) * fake_data).requires_grad_(True)

        # 判别器对插值输入进行前向传播: [B, 1]
        disc_interpolated = self.discriminator(interpolated, cond)

        # 计算关于插值特征的精确梯度
        gradients = torch.autograd.grad(
            outputs=disc_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(disc_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # 展平每个样本的梯度: [B, L * D_in]
        gradients = gradients.view(batch_size, -1)

        # 沿特征维度计算 L2 范数: [B]
        gradient_norm = gradients.norm(2, dim=1)

        # 计算偏离 1 的平方距离惩罚
        penalty = ((gradient_norm - 1.0) ** 2).mean()

        return penalty

    def train_discriminator_step(
        self, real_data: torch.Tensor, cond: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        使用 WGAN-GP 执行判别器的单步优化。
        """
        self.optimizer_d.zero_grad()
        batch_size = real_data.size(0)

        # 采样隐空间噪声并生成假数据
        z = torch.randn(batch_size, self.config.latent_dim, device=self.config.device)
        fake_data = self.generator(z, cond).detach()

        # 计算判别器预测结果
        real_validity = self.discriminator(real_data, cond)  # [B, 1]
        fake_validity = self.discriminator(fake_data, cond)  # [B, 1]

        # WGAN  Critic 损失: 最大化 D(real) - D(fake) <=> 最小化 D(fake) - D(real)
        d_loss_wasserstein = fake_validity.mean() - real_validity.mean()

        # 计算梯度惩罚
        gp = self.gradient_penalty(real_data, fake_data, cond)

        # 带 Lambda 权重的总损失
        total_d_loss = d_loss_wasserstein + self.config.gp_weight * gp

        total_d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        self.optimizer_d.step()

        return total_d_loss, d_loss_wasserstein, real_validity.mean(), fake_validity.mean()

    def train_generator_step(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行生成器的单步优化。
        """
        self.optimizer_g.zero_grad()
        batch_size = cond.size(0)

        # 重新采样全新的隐空间噪声
        z = torch.randn(batch_size, self.config.latent_dim, device=self.config.device)

        # 生成器前向传播
        fake_data = self.generator(z, cond)  # [B, L, D_in]

        # 在判别器中评估合成数据
        fake_validity = self.discriminator(fake_data, cond)  # [B, 1]

        # WGAN 生成器损失: 最大化 D(G(z|c)) <=> 最小化 -D(G(z|c))
        g_loss = -fake_validity.mean()

        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        self.optimizer_g.step()

        return g_loss, fake_data


# ==============================================================================
# 7. 损失与指标计算
# ==============================================================================

def compute_sequence_metrics(real_seq: torch.Tensor, fake_seq: torch.Tensor) -> Dict[str, float]:
    """
    计算真实序列批次与生成序列批次之间的统计相似度指标。

    输入:
        real_seq (Tensor): [B, L, D_in]
        fake_seq (Tensor): [B, L, D_in]

    输出:
        metrics (Dict[str, float]): 计算出的 MSE 和 MAE 指标字典。
    """
    with torch.no_grad():
        mse = nn.functional.mse_loss(fake_seq, real_seq).item()
        mae = nn.functional.l1_loss(fake_seq, real_seq).item()
    return {"seq_mse": mse, "seq_mae": mae}


# ==============================================================================
# 8. 训练/推理执行与入口函数
# ==============================================================================

def main():
    """
    数据集加载、模型建立与训练循环的主执行入口。
    """
    config = CGANConfig()

    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(config.log_dir, "training.log")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("CGAN_Training")
    writer = SummaryWriter(config.log_dir)

    logger.info("正在初始化数据集与 DataLoader...")
    dataset = ConditionalDataset(config.data_path, config.cond_data_path, config.seq_length)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # 设为 0 以确保最佳跨平台兼容性
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True
    )
    logger.info(f"数据集创建成功。总序列窗口数: {len(dataset)}")

    logger.info("正在初始化 CGAN 流水线与子模块...")
    pipeline = CGANPipeline(config)

    logger.info(f"生成器参数量     : {sum(p.numel() for p in pipeline.generator.parameters()):,}")
    logger.info(f"判别器参数量     : {sum(p.numel() for p in pipeline.discriminator.parameters()):,}")

    logger.info("开始执行训练循环...")
    for epoch in range(config.num_epochs):
        pipeline.generator.train()
        pipeline.discriminator.train()

        running_d_loss = 0.0
        running_g_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{config.num_epochs}]")
        for batch_idx, (real_data, cond) in enumerate(pbar):
            real_data = real_data.to(config.device)
            cond = cond.to(config.device)

            # ------------------------------------------------------------------
            # 1. 更新判别器
            # ------------------------------------------------------------------
            d_total_loss, d_wasserstein, real_score, fake_score = pipeline.train_discriminator_step(real_data, cond)

            # ------------------------------------------------------------------
            # 2. 更新生成器 (每 n_critic 步更新一次)
            # ------------------------------------------------------------------
            if batch_idx % config.n_critic == 0:
                g_loss, fake_data = pipeline.train_generator_step(cond)
            else:
                g_loss = torch.tensor(0.0)

            running_d_loss += d_total_loss.item()
            running_g_loss += g_loss.item()

            pbar.set_postfix({
                "D_Loss": f"{d_total_loss.item():.4f}",
                "G_Loss": f"{g_loss.item():.4f}",
                "D(x)": f"{real_score.item():.2f}",
                "D(G(z))": f"{fake_score.item():.2f}"
            })

            global_step = epoch * len(dataloader) + batch_idx
            if batch_idx % 20 == 0:
                writer.add_scalar("Train/D_Total_Loss", d_total_loss.item(), global_step)
                writer.add_scalar("Train/D_Wasserstein_Distance", -d_wasserstein.item(), global_step)
                writer.add_scalar("Train/G_Loss", g_loss.item(), global_step)
                writer.add_scalar("Train/Real_Score_Mean", real_score.item(), global_step)
                writer.add_scalar("Train/Fake_Score_Mean", fake_score.item(), global_step)

        epoch_d_loss = running_d_loss / len(dataloader)
        epoch_g_loss = running_g_loss / len(dataloader)
        logger.info(f"Epoch [{epoch + 1}/{config.num_epochs}] 完成 -> 平均 D 损失: {epoch_d_loss:.6f}, 平均 G 损失: {epoch_g_loss:.6f}")

        # 模型检查点保存
        if (epoch + 1) % config.save_interval == 0:
            ckpt_path = os.path.join(config.save_dir, f"cgan_epoch_{epoch + 1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "generator_state_dict": pipeline.generator.state_dict(),
                "discriminator_state_dict": pipeline.discriminator.state_dict(),
                "optimizer_g_state_dict": pipeline.optimizer_g.state_dict(),
                "optimizer_d_state_dict": pipeline.optimizer_d.state_dict(),
            }, ckpt_path)
            logger.info(f"检查点已成功保存至 {ckpt_path}")

    writer.close()
    logger.info("训练流水线执行完毕。")


if __name__ == "__main__":
    main()