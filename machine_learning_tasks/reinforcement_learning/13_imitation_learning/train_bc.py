"""
代表模型：行为克隆 (Behavior Cloning, BC) 深度策略网络
领域分类：强化学习 / 模仿学习 (Reinforcement Learning / Imitation Learning)
论文来源：Pomerleau, D. A. (1988). ALVINN: An autonomous land vehicle in a neural network.

核心思想与机制：
    行为克隆 (Behavior Cloning) 将模仿学习问题规约为监督学习任务。
    假设存在专家策略 pi_*，其在环境状态 s 下产生动作 a。行为克隆通过收集专家轨迹数据集 
    D = {(s_i, a_i)}，使用最大似然估计 (Maximum Likelihood Estimation, MLE) 或交叉熵损失
    优化策略网络 pi_theta(a|s)，使其预测的动作概率分布拟合专家的决策行为。

数学公式 / 目标函数：
    1. 策略网络输出动作 Logits: z = PolicyNetwork(s)
    2. 策略概率分布 (Softmax): pi_theta(a|s) = exp(z_a) / sum_k(exp(z_k))
    3. 行为克隆交叉熵目标函数:
       L(theta) = - E_{(s, a*) ~ D} [ log pi_theta(a* | s) ]
       其中 a* 为专家的离散动作目标。

数据输入规范：
    - 输入状态张量 (State Tensor):  [Batch_Size, State_Dim] = [B, S]
    - 输出动作 logits (Logits Tensor): [Batch_Size, Action_Dim] = [B, A]
    - 专家动作标签 (Action Target):  [Batch_Size] (离散类别索引 0 ~ A-1)
"""


import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset


# ------------------------------------------------------------------------------
# 3. 超参数与全局配置 (Hyperparameters & Config)
# ------------------------------------------------------------------------------
class Config:
    """行为克隆策略网络超参数配置与全局变量设置。"""
    # 架构参数
    STATE_DIM: int = 10       # 状态空间维度 (State Space Dimension, S)
    ACTION_DIM: int = 4       # 离散动作空间维度 (Action Space Dimension, A)
    HIDDEN_DIM: int = 64      # 隐藏层特征维度 (Hidden Layer Dimension, H)
    DROPOUT: float = 0.1      # 正则化 Dropout 概率

    # 训练参数
    BATCH_SIZE: int = 64      # 批次大小 (Batch Size, B)
    EPOCHS: int = 10          # 训练轮次
    LR: float = 1e-3          # 初始学习率 (Learning Rate, eta)
    NUM_SAMPLES: int = 5000   # 合成专家数据集样本数

    # 设备选择
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------
# 4. 数据处理与 Dataset 管道 (Data Pipeline & Utils)
# ------------------------------------------------------------------------------
def get_synthetic_expert_dataset(
    num_samples: int = Config.NUM_SAMPLES,
    state_dim: int = Config.STATE_DIM,
    action_dim: int = Config.ACTION_DIM,
) -> Dataset:
    """
    合成专家轨迹数据集生成器。

    数学变换逻辑：
        给定规则专家权重 W_expert ~ N(0, I)，生成专家评分:
        scores = s @ W_expert
        a* = floor(|scores * 100|) mod action_dim

    Args:
        num_samples (int): 采样轨迹状态数量，默认 5000。
        state_dim (int): 状态特征维度，默认 10。
        action_dim (int): 离散动作类别数，默认 4。

    Outputs:
        dataset (TensorDataset): 包含状态与专家动作标签的数据集。
            - states: [num_samples, state_dim]
            - expert_actions: [num_samples]
    """
    # 状态采样: [N, S]
    states = torch.randn(num_samples, state_dim)

    # 模拟专家策略的决定性决策映射
    # W_expert: [S]
    weights = torch.randn(state_dim)
    # 矩阵向量乘法 (矩阵乘法规则映射): scores = states @ weights -> [N]
    scores = states @ weights

    # 产生专家动作标签 (映射至 0 ~ action_dim - 1 的离散动作): [N]
    expert_actions = torch.remainder((scores * 100.0).long().abs(), action_dim)

    return TensorDataset(states, expert_actions)


# ------------------------------------------------------------------------------
# 5. 核心子模块 / Encoder / Decoder (Sub-components)
# ------------------------------------------------------------------------------
class MLPBlock(nn.Module):
    """
    模块化多层感知机基础块 (Linear + LayerNorm + GELU + Dropout)。

    数学原理:
        y = Dropout(Activation(LayerNorm(W * x + b)))

    Args:
        in_features (int): 输入特征维度。
        out_features (int): 输出特征维度。
        dropout (float): Dropout 正则化概率。

    Inputs:
        x (Tensor): [B, in_features]

    Outputs:
        out (Tensor): [B, out_features]
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.act = nn.GELU()  # 使用现代高效激活函数 GELU
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, in_features]
        x = self.fc(x)         # [B, in_features] -> [B, out_features]
        x = self.norm(x)       # [B, out_features] -> [B, out_features]
        x = self.act(x)        # [B, out_features] -> [B, out_features]
        x = self.drop(x)       # [B, out_features] -> [B, out_features]
        return x


# ------------------------------------------------------------------------------
# 6. 顶层模型 / Pipeline 主体 (Top-level Architecture / Model)
# ------------------------------------------------------------------------------
class PolicyNetwork(nn.Module):
    """
    行为克隆策略网络 (Behavior Cloning Policy Network)。

    架构说明：
        输入环境状态 s，经过多层深度神经模块特征提取，预测离散动作的非归一化对数概率 (Logits)。

    数学原理 / 变换逻辑：
        h_1 = GELU(LN(W_1 * s + b_1))
        h_2 = GELU(LN(W_2 * h_1 + b_2))
        logits = W_3 * h_2 + b_3

    Args:
        state_dim (int): 状态空间维度 S。
        action_dim (int): 动作空间维度 A。
        hidden_dim (int): 隐层特征维度 H，默认 64。
        dropout (float): Dropout 正则化概率，默认 0.1。

    Inputs:
        state (Tensor): 输入状态张量，shape: [B, S]

    Outputs:
        logits (Tensor): 动作对数概率得分，shape: [B, A]
    """

    def __init__(
        self,
        state_dim: int = Config.STATE_DIM,
        action_dim: int = Config.ACTION_DIM,
        hidden_dim: int = Config.HIDDEN_DIM,
        dropout: float = Config.DROPOUT,
    ):
        super().__init__()
        # 深度 MLP 特征提取主干
        self.block1 = MLPBlock(state_dim, hidden_dim, dropout=dropout)
        self.block2 = MLPBlock(hidden_dim, hidden_dim, dropout=dropout)

        # 策略头 (Policy Head): 直接输出离散动作 logits
        self.policy_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # 输入维度: state [B, S]
        h1 = self.block1(state)        # [B, S] -> [B, H]
        h2 = self.block2(h1)           # [B, H] -> [B, H]
        logits = self.policy_head(h2)  # [B, H] -> [B, A]
        return logits


# ------------------------------------------------------------------------------
# 7. 损失函数与评估指标 (Loss & Metrics)
# ------------------------------------------------------------------------------
class BehaviorCloningLoss(nn.Module):
    """
    行为克隆损失函数 (基于离散动作的最大似然交叉熵)。

    数学公式映射:
        L(logits, target) = - log( exp(logits[target]) / sum_j exp(logits[j]) )
        映射关系:
            logits <-> z_a
            target <-> a*
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            logits (Tensor): 策略网络输出，shape: [B, A]
            targets (Tensor): 专家动作索引，shape: [B]
        Outputs:
            loss (Tensor): 标量 Loss，shape: []
        """
        return self.criterion(logits, targets)


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    计算策略预测动作与专家动作的一致性匹配准确率 (Action Match Accuracy)。

    Inputs:
        logits (Tensor): [B, A]
        targets (Tensor): [B]
    Outputs:
        acc (float): 动作精度占比
    """
    # 极大似然动作选择: a_pred = argmax_a (logits) -> [B]
    preds = torch.argmax(logits, dim=1)
    # 统计预测一致的数量
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total if total > 0 else 0.0


# ------------------------------------------------------------------------------
# 8. 训练/推理逻辑与入口 (Training/Inference Execution)
# ------------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """单轮次训练 Pipeline。"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for states, actions in dataloader:
        # 张量搬移至训练设备
        states = states.to(device)    # [B, S]
        actions = actions.to(device)  # [B]

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播 (Forward pass)
        logits = model(states)        # [B, S] -> [B, A]

        # 计算损失
        loss = criterion(logits, actions)

        # 反向传播与参数更新 (Backward pass & Optimization)
        loss.backward()
        optimizer.step()

        # 累计统计指标
        batch_size = states.size(0)
        total_loss += loss.item() * batch_size
        preds = torch.argmax(logits, dim=1)  # [B]
        total_correct += (preds == actions).sum().item()
        total_samples += batch_size

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples
    return epoch_loss, epoch_acc


def main():
    print(f"=== Starting Behavior Cloning (Task 13) Training on Device: {Config.DEVICE} ===")

    # 1. 构建数据集与数据加载器
    dataset = get_synthetic_expert_dataset(
        num_samples=Config.NUM_SAMPLES,
        state_dim=Config.STATE_DIM,
        action_dim=Config.ACTION_DIM,
    )
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # 2. 实例化模型、损失函数与优化器
    policy_net = PolicyNetwork(
        state_dim=Config.STATE_DIM,
        action_dim=Config.ACTION_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        dropout=Config.DROPOUT,
    ).to(Config.DEVICE)

    criterion = BehaviorCloningLoss()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=Config.LR)

    # 3. 训练循环 Pipeline
    for epoch in range(Config.EPOCHS):
        avg_loss, avg_acc = train_one_epoch(
            model=policy_net,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=Config.DEVICE,
        )

        print(
            f"Epoch [{epoch + 1:02d}/{Config.EPOCHS:02d}] | "
            f"Training Loss: {avg_loss:.4f} | "
            f"Action Match Acc: {avg_acc * 100.0:.2f}%"
        )

    # 4. 推理验证示例 (Inference Routine)
    policy_net.eval()
    with torch.no_grad():
        sample_state = torch.randn(1, Config.STATE_DIM).to(Config.DEVICE)  # [1, S]
        action_logits = policy_net(sample_state)                             # [1, A]
        action_probs = F.softmax(action_logits, dim=-1)                      # [1, A]
        selected_action = torch.argmax(action_probs, dim=-1).item()          # Scalar

        print("\n=== Inference Test Routine ===")
        print(f"Input State Shape : {list(sample_state.shape)}")
        print(f"Action Logits     : {action_logits.cpu().numpy().round(4)}")
        print(f"Action Probabilities: {action_probs.cpu().numpy().round(4)}")
        print(f"Predicted Action  : {selected_action}")


if __name__ == "__main__":
    main()