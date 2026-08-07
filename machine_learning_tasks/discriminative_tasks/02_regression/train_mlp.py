"""
任务定义：
    - 任务编号：Task-02
    - 任务名称：多维特征连续值回归 (Multivariate Feature Regression)
    - 领域分类：基础机器学习 / 深度学习 (Tabular & Vector Dynamics)

代表架构/算法：
    - 模型名称：多层感知机 (Multi-Layer Perceptron, MLP) / 前馈神经网络 (Feedforward Neural Network)
    - 理论来源：Universal Approximation Theorem (Hornik et al., 1989)

核心思想与机制：
    通过叠加非线性变换层，将高维输入的连续特征空间映射到连续的标量/向量目标空间。
    每个隐藏层使用现代高效激活函数 (SiLU / GELU) 与 LayerNorm 进行特征正规化与非线性表征提取。

数学公式/目标函数：
    1. 层前向传播：
       h^(1) = LayerNorm(SiLU(X · W_1 + b_1))
       h^(2) = LayerNorm(SiLU(h^(1) · W_2 + b_2))
       y_pred = h^(2) · W_3 + b_3

    2. 损失函数 (MSE Loss)：
       L(y, y_pred) = (1 / N) * ∑_{i=1}^{N} ||y_i - y_pred_i||^2_2

数据输入规范：
    - 输入 (X)：Tensor, shape: [B, D_in], dtype: float32 (D_in 为输入特征维度)
    - 输出 (Y)：Tensor, shape: [B, D_out], dtype: float32 (D_out 为目标连续值维度)
"""

# ==============================================================================
# 2. 依赖导入 (Imports)
# ==============================================================================
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset


# ==============================================================================
# 3. 超参数与全局配置 (Hyperparameters & Config)
# ==============================================================================
class Config:
    # 数据集超参数
    NUM_SAMPLES: int = 2000
    INPUT_DIM: int = 10
    OUTPUT_DIM: int = 1
    NOISE_STD: float = 0.1

    # 模型架构超参数
    HIDDEN_DIM: int = 64
    DROPOUT_RATE: float = 0.1

    # 训练超参数
    BATCH_SIZE: int = 64
    EPOCHS: int = 10
    LEARNING_RATE: float = 1e-3
    SEED: int = 42

    # 计算设备
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 设置随机种子以保证结果可复现
torch.manual_seed(Config.SEED)


# ==============================================================================
# 4. 数据处理与 Dataset 管道 (Data Pipeline & Utils)
# ==============================================================================
def get_synthetic_dataset(
    num_samples: int = Config.NUM_SAMPLES,
    input_dim: int = Config.INPUT_DIM,
    output_dim: int = Config.OUTPUT_DIM,
    noise_std: float = Config.NOISE_STD,
) -> Tuple[Dataset, torch.Tensor, torch.Tensor]:
    """
    合成连续值回归数据集生成器。

    数学原理:
        Y = X · W + b + ε,  其中 ε ~ N(0, noise_std^2)

    Args:
        num_samples (int): 样本数量 N，默认 2000。
        input_dim (int): 特征维度 D_in，默认 10。
        output_dim (int): 目标维度 D_out，默认 1。
        noise_std (float): 高斯噪声标准差，默认 0.1。

    Returns:
        dataset (Dataset): PyTorch TensorDataset 对象。
        true_w (Tensor): 生成数据的真实权重矩阵，shape: [D_in, D_out]。
        true_b (Tensor): 生成数据的真实偏置向量，shape: [D_out]。
    """
    # [N, D_in] 输入特征矩阵
    x = torch.randn(num_samples, input_dim)

    # [D_in, D_out] 真实权重与 [D_out] 真实偏置
    true_w = torch.randn(input_dim, output_dim)
    true_b = torch.randn(output_dim)

    # [N, D_out] 高斯噪声 ε
    noise = torch.randn(num_samples, output_dim) * noise_std

    # [N, D_out] 矩阵乘法计算目标值 y = x @ w + b + noise
    y = torch.matmul(x, true_w) + true_b + noise

    dataset = TensorDataset(x, y)
    return dataset, true_w, true_b


# ==============================================================================
# 5. 核心子模块 / Encoder / Decoder (Sub-components)
# ==============================================================================
class MLPBlock(nn.Module):
    """
    标准 MLP 基础构建块：包含全连接层、归一化、激活函数与正则化。

    数学原理 / 变换逻辑:
        y = Dropout(SiLU(LayerNorm(Linear(x))))
        - Linear: y_lin = x · W^T + b
        - LayerNorm: y_norm = (y_lin - μ) / √(σ^2 + ε) · γ + β
        - SiLU: y_act = y_norm · σ(y_norm)
        - Dropout: y_out = y_act ⊙ m / (1 - p)

    Args:
        in_features (int): 输入特征维度 C_in。
        out_features (int): 输出特征维度 C_out。
        dropout (float): Dropout 概率，默认 0.1。

    Inputs:
        x (Tensor): 输入张量，shape: [B, C_in]

    Outputs:
        out (Tensor): 输出张量，shape: [B, C_out]
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.act = nn.SiLU()  # 采用现代 SiLU (Swish) 激活函数
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C_in]
        x = self.fc(x)
        # x shape: [B, C_out]
        x = self.norm(x)
        # x shape: [B, C_out]
        x = self.act(x)
        # x shape: [B, C_out]
        x = self.dropout(x)
        # x shape: [B, C_out]
        return x


# ==============================================================================
# 6. 顶层模型 / Pipeline 主体 (Top-level Architecture / Model)
# ==============================================================================
class MLP(nn.Module):
    """
    多层感知机 (MLP) 回归主架构。

    架构组成：
        - Block 1: Input Layer -> Hidden Layer 1 (含 LayerNorm + SiLU + Dropout)
        - Block 2: Hidden Layer 1 -> Hidden Layer 2 (含 LayerNorm + SiLU + Dropout)
        - Head: Hidden Layer 2 -> Output Layer (线性投影，无激活)

    Args:
        input_dim (int): 输入特征维度 D_in。
        hidden_dim (int): 隐藏层特征维度 D_hidden。
        output_dim (int): 输出目标维度 D_out。
        dropout (float): Dropout 归一化概率。

    Inputs:
        x (Tensor): 批次特征输入张量，shape: [B, D_in]

    Outputs:
        out (Tensor): 批次预测连续值，shape: [B, D_out]
    """

    def __init__(
        self,
        input_dim: int = Config.INPUT_DIM,
        hidden_dim: int = Config.HIDDEN_DIM,
        output_dim: int = Config.OUTPUT_DIM,
        dropout: float = Config.DROPOUT_RATE,
    ):
        super().__init__()
        # 模块化构建隐藏层
        self.block1 = MLPBlock(
            in_features=input_dim, out_features=hidden_dim, dropout=dropout
        )
        self.block2 = MLPBlock(
            in_features=hidden_dim, out_features=hidden_dim, dropout=dropout
        )

        # 回归输出层（直接进行线性变换投影，不加非线性激活与归一化）
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入张量 shape: [B, D_in]
        x = self.block1(x)
        # 维度转换: [B, D_in] -> [B, D_hidden]

        x = self.block2(x)
        # 维度转换: [B, D_hidden] -> [B, D_hidden]

        out = self.head(x)
        # 维度转换: [B, D_hidden] -> [B, D_out]

        return out


# ==============================================================================
# 7. 损失函数与评估指标 (Loss & Metrics)
# ==============================================================================
class RegressionLossAndMetrics(nn.Module):
    """
    回归任务损失函数与评估指标封装。

    计算公式:
        - MSE Loss: (1/N) * ∑ (y_pred - y_true)^2
        - MAE Metric: (1/N) * ∑ |y_pred - y_true|
    """

    def __init__(self):
        super().__init__()
        self.mse_fn = nn.MSELoss()
        self.mae_fn = nn.L1Loss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Inputs:
            pred (Tensor): 模型预测值，shape: [B, D_out]
            target (Tensor): 真实目标值，shape: [B, D_out]

        Outputs:
            loss (Tensor): 可导的 MSE 损失对象，shape: []
            mae (float): 当前 Batch 的 MAE 指标标量值
        """
        loss = self.mse_fn(pred, target)
        with torch.no_grad():
            mae = self.mae_fn(pred, target).item()
        return loss, mae


# ==============================================================================
# 8. 训练/推理逻辑与入口 (Training/Inference Execution)
# ==============================================================================
def main():
    print(f"[*] 运行设备配置: {Config.DEVICE}")

    # 1. 准备数据管道
    dataset, true_w, true_b = get_synthetic_dataset()
    train_loader = DataLoader(
        dataset=dataset, batch_size=Config.BATCH_SIZE, shuffle=True
    )

    # 2. 实例化模型、损失函数与优化器
    model = MLP(
        input_dim=Config.INPUT_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        output_dim=Config.OUTPUT_DIM,
        dropout=Config.DROPOUT_RATE,
    ).to(Config.DEVICE)

    criterion = RegressionLossAndMetrics()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4
    )

    # 3. 执行训练循环
    print("[*] 开始训练流程...")
    model.train()
    for epoch in range(Config.EPOCHS):
        total_mse = 0.0
        total_mae = 0.0
        total_batches = len(train_loader)

        for batch_idx, (xb, yb) in enumerate(train_loader):
            # 传输张量至目标设备: [B, D_in], [B, D_out]
            xb = xb.to(Config.DEVICE)
            yb = yb.to(Config.DEVICE)

            # 前向传播
            optimizer.zero_grad()
            pred = model(xb)  # shape: [B, D_out]

            # 损失与评估计算
            loss, mae = criterion(pred, yb)

            # 反向传播与优化
            loss.backward()
            optimizer.step()

            total_mse += loss.item()
            total_mae += mae

        avg_mse = total_mse / total_batches
        avg_mae = total_mae / total_batches
        print(
            f"Epoch [{epoch + 1:02d}/{Config.EPOCHS:02d}] | "
            f"Train MSE Loss: {avg_mse:.6f} | "
            f"Train MAE: {avg_mae:.6f}"
        )

    # 4. 执行推理校验
    model.eval()
    with torch.no_grad():
        test_x = torch.randn(5, Config.INPUT_DIM).to(Config.DEVICE)
        test_pred = model(test_x)
        print("\n[*] 推理测试 (5 个示例)：")
        print(f"    输入 Shape:  {test_x.shape}")
        print(f"    预测输出 Shape: {test_pred.shape}")
        print(f"    预测输出样例:\n{test_pred.cpu().numpy()}")


if __name__ == "__main__":
    main()