"""
任务定义: 任务 02 - 意图识别文本分类 (Intent Recognition / Text Classification)
代表架构: Mamba (Selective State Space Models, Gu & Dao, 2023)
核心思想: 引入根据输入动态变化的选择性扫描机制 (Selective Scan), 解决传统 SSM 无法根据输入过滤信息的弊端。
          通过将连续状态空间模型 (A, B, C) 进行离散化 (Δ)，实现 O(N) 线性时间复杂度的序列建模。
数学公式:
    1. 连续 SSM 系统: h'(t) = A * h(t) + B * x(t), y(t) = C * h(t)
    2. 离散化 (Zero-Order Hold, ZOH):
       Ā = exp(Δ * A)
       B̄ = (Δ * A)^(-1) * (exp(Δ * A) - I) * Δ * B ≈ Δ * B
    3. 选择性机制 (Selective Mechanism):
       B, C, Δ 均为输入 x 的函数: B = Linear_B(x), C = Linear_C(x), Δ = Softplus(Linear_Δ(x))
数据输入规范:
    Input:  Tensor shape [B, L] (Token ID 序列)
    Output: Tensor shape [B, Num_Classes] (未经过 Softmax 的意图分类 Logits)
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

# ==============================================================================
# 3. 超参数与全局配置 (Hyperparameters & Config)
# ==============================================================================
BATCH_SIZE: int = 32
EPOCHS: int = 5
LEARNING_RATE: float = 1e-3
NUM_CLASSES: int = 8          # 意图类别数 (如: 查天气、播放音乐、订餐等)
VOCAB_SIZE: int = 1000        # 词表大小
MAX_SEQ_LEN: int = 32         # 文本最大序列长度
D_MODEL: int = 128            # 隐藏层维度
D_STATE: int = 16             # SSM 状态维度 (N)
D_CONV: int = 4               # 1D 卷积核大小
EXPAND: int = 2               # 内部特征维度扩展倍数 (E)
NUM_SAMPLES: int = 1000
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# 4. 数据处理与 Dataset 管道 (Data Pipeline & Utils)
# ==============================================================================
def get_synthetic_intent_dataset(
    num_samples: int = NUM_SAMPLES,
    vocab_size: int = VOCAB_SIZE,
    seq_len: int = MAX_SEQ_LEN,
    num_classes: int = NUM_CLASSES,
) -> Dataset:
    """
    生成合成的文本 Token 序列与意图标签，用于测试和演示 Pipeline。

    Returns:
        Dataset: 包含 Token ID 张量 [N, L] 与意图标签张量 [N] 的 PyTorch Dataset。
    """
    # 模拟文本 Token ID 序列: x ~ Uniform(1, vocab_size-1), shape: [N, L]
    x = torch.randint(1, vocab_size, (num_samples, seq_len), dtype=torch.long)
    # 模拟意图类别标签: y ~ Uniform(0, num_classes-1), shape: [N]
    y = torch.randint(0, num_classes, (num_samples,), dtype=torch.long)
    return TensorDataset(x, y)


# ==============================================================================
# 5. 核心子模块 (Mamba Block & Selective SSM)
# ==============================================================================
class MambaBlock(nn.Module):
    """
    纯 PyTorch 实现的手写单层 Mamba 模块 (Selective State Space Model)。

    结构路线:
        x -> Linear Expansion -> Split (x_proj, z)
        x_proj -> 1D Depthwise Conv -> SiLU
        -> Param Projections (B, C, Δ) -> Selective Scan (SSM)
        -> Multiplied by Gate z -> Out Linear

    Args:
        d_model (int): 输入/输出维度。
        d_state (int): SSM 隐藏状态维度 (N)。
        d_conv (int): 一维卷积核大小。
        expand (int): 内部展开倍率 E。
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        d_state: int = D_STATE,
        d_conv: int = D_CONV,
        expand: int = EXPAND,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)  # D = E * d_model
        self.dt_rank = math.ceil(self.d_model / 16)

        # 1. 输入维度扩展投影 (升维 2 * d_inner，用于分支分流和门控)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 2. 一维深度卷积 (Depthwise Conv1d)，提取局部上下文
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        # 3. 选择性参数投影层 (将输入投影至 B, C, Δ)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # 4. SSM 系统矩阵 A 与 D 初始化
        # A 矩阵采用 S4 的 HiPPO 衰减矩阵初始化策略: A_in = log(1..N)
        A = torch.repeat_interleave(
            torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0),
            repeats=self.d_inner,
            dim=0,
        )
        self.A_log = nn.Parameter(torch.log(A))  # 保持 log(A) 保证 A 恒为负
        self.D = nn.Parameter(torch.ones(self.d_inner))  # 跳跃连接参数

        # 5. 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            x: [B, L, D_model]
        Outputs:
            out: [B, L, D_model]
        """
        batch, seq_len, _ = x.shape

        # 1. 投影并拆分为主路径 x_branch 与门控分支 z
        in_projected = self.in_proj(x)  # [B, L, 2 * D_inner]
        x_branch, z = in_projected.chunk(2, dim=-1)  # 各 [B, L, D_inner]

        # 2. 因果一维卷积 (Conv1d 需要通道在前 [B, D_inner, L])
        x_conv = x_branch.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # 截断 Padding 保证因果性
        x_conv = x_conv.transpose(1, 2)
        x_active = F.silu(x_conv)  # [B, L, D_inner]

        # 3. 计算选择性参数 B, C, Δ
        # x_proj_res: [B, L, dt_rank + 2 * d_state]
        x_proj_res = self.x_proj(x_active)
        dt, B_t, C_t = torch.split(
            x_proj_res, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # Δ 离散化步长变换: [B, L, dt_rank] -> [B, L, D_inner]
        dt = F.softplus(self.dt_proj(dt))

        # 4. 离散化 SSM 参数 A, B
        A = -torch.exp(self.A_log)  # [D_inner, D_state]
        
        # 离散化 A: Ā = exp(Δ * A), shape: [B, L, D_inner, D_state]
        deltaA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        
        # 离散化 B: B̄ ≈ Δ * B, shape: [B, L, D_inner, D_state]
        deltaB_x = (dt.unsqueeze(-1) * B_t.unsqueeze(2)) * x_active.unsqueeze(-1)

        # 5. Selective Scan (选择性扫描计算循环 / 循环递推)
        # 依次更新隐状态 h_t = Ā_t * h_{t-1} + B̄_t * x_t
        h = torch.zeros(
            batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype
        )
        y_ssm = []

        for t in range(seq_len):
            h = deltaA[:, t] * h + deltaB_x[:, t]  # [B, D_inner, D_state]
            # y_t = C_t * h_t
            y_t = torch.einsum("bdn,bn->bd", h, C_t[:, t])  # [B, D_inner]
            y_ssm.append(y_t)

        y = torch.stack(y_ssm, dim=1)  # [B, L, D_inner]

        # 6. 添加残差 D 项并结合门控分支 z
        y = y + x_active * self.D
        y = y * F.silu(z)  # 乘门控激活信号

        # 7. 最终线性投影
        out = self.out_proj(y)  # [B, L, D_model]
        return out


# ==============================================================================
# 6. 顶层模型 (Top-level Architecture for Text Intent Classification)
# ==============================================================================
class MambaForIntentClassification(nn.Module):
    """
    基于 Mamba 骨干网络的意图识别文本分类模型。

    结构路线:
        Embedding -> RMSNorm -> [MambaBlock + Residual + RMSNorm] x Num_Layers
        -> Global Mean Pooling -> Classifier Linear -> Logits
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        num_classes: int = NUM_CLASSES,
        d_model: int = D_MODEL,
        d_state: int = D_STATE,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # 构建多层 Mamba 堆叠结构
        self.layers = nn.ModuleList([])
        self.norms = nn.ModuleList([])

        for _ in range(num_layers):
            self.layers.append(
                MambaBlock(d_model=d_model, d_state=d_state)
            )
            # 使用 LayerNorm 规范化层
            self.norms.append(nn.LayerNorm(d_model))

        self.final_norm = nn.LayerNorm(d_model)
        # 分类头 Header
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, L]
        x = self.embedding(input_ids)  # [B, L] -> [B, L, D_model]

        # 逐层经过 Mamba Block (带 Pre-LN 残差连接)
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(norm(x))

        x = self.final_norm(x)  # [B, L, D_model]

        # 全局平均池化 (Global Average Pooling 聚合成句向量)
        pooled = torch.mean(x, dim=1)  # [B, L, D_model] -> [B, D_model]

        # 获取分类 Logits
        logits = self.classifier(pooled)  # [B, D_model] -> [B, Num_Classes]
        return logits


# ==============================================================================
# 7. 损失函数与评估指标 (Loss & Metrics)
# ==============================================================================
def compute_accuracy(
    logits: torch.Tensor, targets: torch.Tensor
) -> Tuple[float, int]:
    """计算意图分类 Accuracy 指标。"""
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == targets).sum().item()
    acc = correct / targets.size(0)
    return acc, correct


# ==============================================================================
# 8. 训练/推理逻辑与入口 (Training/Inference Execution)
# ==============================================================================
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """训练单回合的核心逻辑。"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for input_ids, labels in dataloader:
        input_ids = input_ids.to(device)  # shape: [B, L]
        labels = labels.to(device)        # shape: [B]

        # 前向传播
        optimizer.zero_grad()
        logits = model(input_ids)         # shape: [B, Num_Classes]
        loss = criterion(logits, labels)

        # 反向传播与优化
        loss.backward()
        optimizer.step()

        # 统计指标
        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size
        _, correct = compute_accuracy(logits, labels)
        total_correct += correct
        total_samples += batch_size

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples
    return epoch_loss, epoch_acc


def main() -> None:
    print(f"[*] 使用设备: {DEVICE}")

    # 1. 构建合成文本意图数据集
    dataset = get_synthetic_intent_dataset(num_samples=NUM_SAMPLES)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. 实例化 Mamba 意图分类模型
    model = MambaForIntentClassification(
        vocab_size=VOCAB_SIZE,
        num_classes=NUM_CLASSES,
        d_model=D_MODEL,
        d_state=D_STATE,
        num_layers=2,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    print(f"[*] Pure PyTorch Mamba 意图识别模型构建完成，总参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("[*] 开始训练流程...\n" + "-" * 50)

    # 3. 执行训练循环
    for epoch in range(1, EPOCHS + 1):
        loss, acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        print(
            f"Epoch [{epoch:02d}/{EPOCHS:02d}] | "
            f"Train Loss: {loss:.4f} | "
            f"Train Acc: {acc * 100:.2f}%"
        )

    print("-" * 50 + "\n[*] 训练完成！")


if __name__ == "__main__":
    main()