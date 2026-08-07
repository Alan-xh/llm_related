"""
任务定义：
    - 任务编号：Task-02
    - 任务名称：选择性状态空间模型回归 (Selective State Space Model Regression / Mamba)
    - 领域分类：深度学习 (Sequence & Tabular Dynamics / State Space Models)

代表架构/算法：
    - 模型名称：Mamba (Selective State Space Model, S6)
    - 理论来源：Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)

核心思想与机制：
    通过将连续时间状态空间模型 (SSM) 离散化，并引入输入驱动的选择性机制 (Selective Mechanism，即使 B, C, Δ 成为输入的函数)，
    打破传统 SSM 的时不变限制。利用并行关联扫描 (Parallel Associative Scan) 算法实现高效的前向传播计算，
    在保持线性时间复杂度的同时捕捉长距离与复杂非线性特征表征。

数学公式/目标函数：
    1. 连续状态空间方程：
       h'(t) = A · h(t) + B · x(t)
       y(t)   = C · h(t) + D · x(t)

    2. 零阶保持 (ZOH) 离散化：
       Ā = exp(Δ · A)
       B̄ = (Δ · A)^{-1} · (exp(Δ · A) - I) · Δ · B ≈ Δ · B

    3. 选择性参数计算 (Selective Mechanism)：
       Δ = Softplus(Linear_Δ(x))
       B = Linear_B(x)
       C = Linear_C(x)

    4. 损失函数 (MSE Loss)：
       L(y, y_pred) = (1 / N) * ∑_{i=1}^{N} ||y_i - y_pred_i||^2_2

数据输入规范：
    - 输入 (X)：Tensor, shape: [B, D_in], dtype: float32
    - 输出 (Y)：Tensor, shape: [B, D_out], dtype: float32
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
    D_MODEL: int = 64        # 隐层通道维度
    D_STATE: int = 16        # SSM 状态维度 N
    D_CONV: int = 4          # 局部一维卷积核大小
    EXPAND: int = 2          # 内部特征维度拓展倍数
    NUM_LAYERS: int = 2      # Mamba Block 堆叠层数
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
    x = torch.randn(num_samples, input_dim)
    true_w = torch.randn(input_dim, output_dim)
    true_b = torch.randn(output_dim)
    noise = torch.randn(num_samples, output_dim) * noise_std

    y = torch.matmul(x, true_w) + true_b + noise

    dataset = TensorDataset(x, y)
    return dataset, true_w, true_b


# ==============================================================================
# 5. 核心子模块 / Encoder / Decoder (Sub-components)
# ==============================================================================
class S6Kernel(nn.Module):
    """
    S6 (Selective State Space) 核心计算内核 (Pure PyTorch 实现)。

    实现选择性离散化与并行扫描算子:
        1. 依赖输入的参数映射: Δ, B, C = Linear(x)
        2. 离散化: Ā = exp(Δ ⊗ A), B̄ = Δ ⊗ B
        3. 状态递推 (Parallel Scan 算法模拟): h_t = Ā_t · h_{t-1} + B̄_t · x_t
        4. 输出投影: y_t = C_t · h_t

    Args:
        d_inner (int): 扩展后的内部维度 (D_model * Expand)。
        d_state (int): SSM 状态维度 N，默认 16。
        dt_rank (int): Δ 投影的秩，默认 ceil(d_inner / 16)。
    """

    def __init__(self, d_inner: int, d_state: int = 16, dt_rank: int = None):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d_inner / 16)

        # A 参数初始化 (HiPPO 矩阵思想：对角元素初始化)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # 存 log(A) 保证训练稳定性

        # D 跳跃连接参数 (Skip Connection)
        self.D = nn.Parameter(torch.ones(d_inner))

        # 选择性机制投影层
        self.x_proj = nn.Linear(d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)

        # 初始化 dt_proj 偏置，使 initial dt 处于合理区间
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            x (Tensor): 输入序列特征，shape: [B, L, D_inner]
        Outputs:
            y (Tensor): SSM 输出特征，shape: [B, L, D_inner]
        """
        batch_size, seq_len, d_inner = x.shape
        A = -torch.exp(self.A_log.float())  # shape: [D_inner, D_state]

        # 1. 动态生成选择性参数 Δ, B, C
        x_dbl = self.x_proj(x)  # [B, L, dt_rank + 2 * d_state]
        delta_rank, B_param, C_param = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # delta 计算与 Softplus 激活
        delta = F.softplus(self.dt_proj(delta_rank))  # [B, L, D_inner]

        # 2. 离散化: Ā = exp(Δ * A), B̄ = Δ * B
        # delta: [B, L, D_inner, 1], A: [1, 1, D_inner, D_state]
        delta_expanded = delta.unsqueeze(-1)
        A_expanded = A.unsqueeze(0).unsqueeze(0)
        delta_A = torch.exp(delta_expanded * A_expanded)  # [B, L, D_inner, D_state]

        # B_param: [B, L, 1, D_state], delta: [B, L, D_inner, 1]
        delta_B = delta_expanded * B_param.unsqueeze(-2)   # [B, L, D_inner, D_state]

        # 3. 顺序状态更新 (Scan Loop over sequence length L)
        # 针对表格维度小序列进行优化，也可使用关联并行扫描
        x_expanded = x.unsqueeze(-1)                       # [B, L, D_inner, 1]
        dB_x = delta_B * x_expanded                        # [B, L, D_inner, D_state]

        h = torch.zeros(
            batch_size, d_inner, self.d_state, device=x.device, dtype=x.dtype
        )
        ys = []

        for t in range(seq_len):
            h = delta_A[:, t] * h + dB_x[:, t]             # [B, D_inner, D_state]
            # C_param[:, t]: [B, D_state] -> [B, 1, D_state]
            c_t = C_param[:, t].unsqueeze(1)
            y_t = torch.sum(h * c_t, dim=-1)               # [B, D_inner]
            ys.append(y_t)

        y = torch.stack(ys, dim=1)                         # [B, L, D_inner]

        # 4. 加入 D 残差直接映射 D * x
        y = y + x * self.D
        return y


class MambaBlock(nn.Module):
    """
    单层 Mamba 模块 (包含分支选择、一维卷积、SiLU门控与 S6 内核)。

    架构设计:
              ┌───> Linear ───> Conv1d ───> SiLU ───> S6 ───┐
        x ────┤                                            ⊗ ───> Linear ───> Output
              └───> Linear ───────────────────> SiLU ──────┘

    Args:
        d_model (int): 输入与输出特征维度。
        d_state (int): SSM 隐藏状态维度。
        d_conv (int): 1D 卷积核宽度。
        expand (int): 内部通道维度扩展倍数。
    """

    def __init__(
        self,
        d_model: int = Config.D_MODEL,
        d_state: int = Config.D_STATE,
        d_conv: int = Config.D_CONV,
        expand: int = Config.EXPAND,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand

        # 升维输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 局部 1D 深度可分离卷积
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        # 选择性 SSM 内核
        self.s6_kernel = S6Kernel(d_inner=self.d_inner, d_state=d_state)

        # 降维输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            x (Tensor): [B, L, D_model]
        Outputs:
            out (Tensor): [B, L, D_model]
        """
        batch, seq_len, _ = x.shape

        # 1. 输入并行分流投影
        x_and_res = self.in_proj(x)                        # [B, L, 2 * D_inner]
        x_branch, res_branch = torch.chunk(x_and_res, 2, dim=-1)

        # 2. 主分支卷积与激活
        x_branch = x_branch.transpose(1, 2)                # [B, D_inner, L]
        x_branch = self.conv1d(x_branch)[:, :, :seq_len]   # 因果 Padding 截断
        x_branch = x_branch.transpose(1, 2)                # [B, L, D_inner]
        x_branch = F.silu(x_branch)

        # 3. 选择性 SSM 核心计算
        ssm_out = self.s6_kernel(x_branch)                 # [B, L, D_inner]

        # 4. 门控乘法融合 (Gated Linear Unit)
        gated_out = ssm_out * F.silu(res_branch)           # [B, L, D_inner]

        # 5. 最终投影输出
        out = self.out_proj(gated_out)                     # [B, L, D_model]
        return out


# ==============================================================================
# 6. 顶层模型 / Pipeline 主体 (Top-level Architecture / Model)
# ==============================================================================
class MambaRegressionModel(nn.Module):
    """
    基于 Mamba (S6) 的连续值特征回归主架构。

    架构组成：
        - Input Projector: [B, D_in] -> 特征拓展/伪序列重塑 -> [B, L=D_in, D_model]
        - Mamba Backbone: 堆叠多层 MambaBlock (含 LayerNorm 与 Skip Connections)
        - Pooling & Regression Head: 全局池化映射至目标空间 [B, D_out]
    """

    def __init__(
        self,
        input_dim: int = Config.INPUT_DIM,
        d_model: int = Config.D_MODEL,
        output_dim: int = Config.OUTPUT_DIM,
        num_layers: int = Config.NUM_LAYERS,
        dropout: float = Config.DROPOUT_RATE,
    ):
        super().__init__()
        # 1. 特征序列化映射层 (将表格特征转化为 1D 虚拟序列，或多维度特征表达)
        self.input_layer = nn.Linear(1, d_model)

        # 2. 堆叠 Mamba 骨干层
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(d_model),
                "mamba": MambaBlock(d_model=d_model),
                "dropout": nn.Dropout(dropout)
            })
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # 3. 回归输出 Head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入张量 shape: [B, D_in]
        # 重塑为序列维度：将每个输入特征看作序列中的一个 Token -> [B, D_in, 1]
        x_seq = x.unsqueeze(-1)
        
        # 投影至模型维度: [B, D_in, D_model]
        h = self.input_layer(x_seq)

        # 经过 Mamba Backbone
        for layer in self.layers:
            # Pre-LN 结构与残差连接
            residual = h
            h = layer["norm"](h)
            h = layer["mamba"](h)
            h = layer["dropout"](h)
            h = h + residual

        h = self.final_norm(h)                             # [B, D_in, D_model]

        # 4. 时序/特征维度池化 (Mean Pooling)
        h_pooled = h.mean(dim=1)                           # [B, D_model]

        # 5. 回归预测输出
        out = self.head(h_pooled)                          # [B, D_out]
        return out


# ==============================================================================
# 7. 损失函数与评估指标 (Loss & Metrics)
# ==============================================================================
class RegressionLossAndMetrics(nn.Module):
    """
    回归任务损失函数与评估指标封装。
    """

    def __init__(self):
        super().__init__()
        self.mse_fn = nn.MSELoss()
        self.mae_fn = nn.L1Loss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
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

    # 2. 实例化 Mamba 回归模型、损失函数与优化器
    model = MambaRegressionModel(
        input_dim=Config.INPUT_DIM,
        d_model=Config.D_MODEL,
        output_dim=Config.OUTPUT_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT_RATE,
    ).to(Config.DEVICE)

    criterion = RegressionLossAndMetrics()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4
    )

    # 3. 执行训练循环
    print("[*] 开始 Mamba Regression 模型训练流程...")
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
            pred = model(xb)                               # shape: [B, D_out]

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