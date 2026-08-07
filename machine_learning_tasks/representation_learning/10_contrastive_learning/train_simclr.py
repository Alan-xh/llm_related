"""
任务 10: 对比学习 (Contrastive Learning / 判别式自监督学习)
代表架构: SimCLR (A Simple Framework for Contrastive Learning of Visual Representations)
论文来源: Chen et al., ICML 2020 (https://arxiv.org/abs/2002.05709)

核心思想与机制:
    1. 数据增强 Pipeline: 对同一张输入图像 x 施加两种不同的随机数据增强变换 (t, t')，
       生成一对正样本视图 (x_i, x_j)。
    2. 编码器 (Encoder): 使用 ResNet18 骨干提取图像表征 h_i = f(x_i), h_j = f(x_j)。
    3. 投影头 (Projection Head): 使用多层感知机 (MLP) 将高维表征映射至低维对比空间 z_i = g(h_i), z_j = g(h_j)。
    4. 对比损失 (NT-Xent): 在 L2 归一化的特征空间中计算余弦相似度，最大化正样本对之间的相似度，
       同时最小化与 Mini-batch 中所有其他 2(N-1) 个负样本的相似度。

数学公式与优化目标:
    - Cosine Similarity:
        sim(u, v) = (u^T * v) / (||u||_2 * ||v||_2)
    - NT-Xent Loss (Normalized Temperature-scaled Cross Entropy Loss):
        ℓ_{i,j} = - log [ exp(sim(z_i, z_j) / τ) / Σ_{k=1, k≠i}^{2N} exp(sim(z_i, z_k) / τ) ]
      其中 τ (tau) 为温度参数 (Temperature Parameter)。

数据输入规范:
    - 输入 (x): Raw Image Batch, Shape: [B, C, H, W] = [32, 3, 64, 64]
    - 增强视图 (v1, v2): Augmented Views, Shape: [B, C, H, W]
    - 编码表征 (h): Base Representation, Shape: [B, 512]
    - 投影表示 (z): L2-Normalized Projection Vector, Shape: [B, D_proj] = [32, 128]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ==============================================================================
# 3. 超参数与全局配置 (Hyperparameters & Config)
# ==============================================================================
class Config:
    BATCH_SIZE: int = 32
    EPOCHS: int = 5
    LEARNING_RATE: float = 3e-4
    TEMPERATURE: float = 0.5  # NT-Xent 中的温度系数 τ
    PROJ_DIM: int = 128       # 投影头输出特征维度 D_proj
    FEATURE_DIM: int = 512    # ResNet18 编码后向量维度 D_feat
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# 4. 数据处理与 Dataset 管道 (Data Pipeline & Utils)
# ==============================================================================
class SyntheticImageDataset(Dataset):
    """
    合成图像数据集，生成标准 RGB 图像张量用于模拟真实训练。

    Args:
        num_samples (int): 样本总数。
        height (int): 图像高度。
        width (int): 图像宽度。
    """
    def __init__(self, num_samples: int = 1000, height: int = 64, width: int = 64):
        super().__init__()
        self.num_samples = num_samples
        self.height = height
        self.width = width
        # 生成 [N, C, H, W] 的标准正态分布随机图像
        self.data = torch.randn(num_samples, 3, height, width)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns:
            x (Tensor): 单张图片张量，Shape: [3, H, W]
        """
        return self.data[idx]


class ContrastiveAugmentation:
    """
    SimCLR 专属数据增强变换器，针对单张图片生成两个独立随机增强视图。

    Inputs:
        x (Tensor): 原始输入图像批次，Shape: [B, C, H, W]

    Outputs:
        v1 (Tensor): 增强视图 1，Shape: [B, C, H, W]
        v2 (Tensor): 增强视图 2，Shape: [B, C, H, W]
    """
    def __init__(self, noise_std: float = 0.05):
        self.noise_std = noise_std

    def _apply_transform(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 随机水平翻转 (Random Horizontal Flip)
        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[-1])
        # 2. 高斯噪声/扰动注入 (Noise Injection)
        noise = torch.randn_like(x) * self.noise_std
        x = x + noise
        return x

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v1 = self._apply_transform(x.clone())
        v2 = self._apply_transform(x.clone())
        return v1, v2


# ==============================================================================
# 5. 核心子模块 / Encoder (Sub-components)
# ==============================================================================
class BasicBlock(nn.Module):
    """
    ResNet18 的标准基础残差块 (Residual Block)。

    数学原理:
        y = σ( BatchNorm( Conv2d( σ( BatchNorm( Conv2d(x) ) ) ) ) + Shortcut(x) )
        其中 σ 代表 GELU 激活函数。

    Args:
        in_ch (int): 输入通道数。
        out_ch (int): 输出通道数。
        stride (int): 卷积步长，用于控制空间下采样。默认 1。

    Inputs:
        x (Tensor): 输入特征图，Shape: [B, C_in, H_in, W_in]

    Outputs:
        out (Tensor): 输出特征图，Shape: [B, C_out, H_out, W_out]
    """
    expansion: int = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        # 第一层 3x3 卷积 (可能包含下采样)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.GELU()  # 遵循 1.4 规范：采用现代 GELU 激活函数

        # 第二层 3x3 卷积
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.GELU()

        # 残差连接匹配维度 (Shortcut Connection)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x Shape: [B, C_in, H_in, W_in]
        identity = self.shortcut(x)  # Shape: [B, C_out, H_out, W_out]

        out = self.conv1(x)          # Shape: [B, C_out, H_out, W_out]
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)        # Shape: [B, C_out, H_out, W_out]
        out = self.bn2(out)

        out = out + identity         # 残差相加，维度保持不变
        out = self.act2(out)
        return out


class ResNet18Backbone(nn.Module):
    """
    手写 ResNet18 图像表征提取骨干网络 (剔除默认分类全连接层)。

    Args:
        in_channels (int): 输入图像通道数，默认 3 (RGB)。

    Inputs:
        x (Tensor): 输入图像批次，Shape: [B, C_in, H, W]

    Outputs:
        h (Tensor): 经过池化与 Flatten 后的全局特征向量，Shape: [B, 512]
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        # Stem Layer: 7x7 大卷积下采样
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.GELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual Stages
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        # Global Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, in_ch: int, out_ch: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [B, 3, 64, 64]
        x = self.conv1(x)            # Shape: [B, 64, 32, 32]
        x = self.bn1(x)              # Shape: [B, 64, 32, 32]
        x = self.act1(x)             # Shape: [B, 64, 32, 32]
        x = self.maxpool(x)          # Shape: [B, 64, 16, 16]

        x = self.layer1(x)           # Shape: [B, 64, 16, 16]
        x = self.layer2(x)           # Shape: [B, 128, 8, 8]
        x = self.layer3(x)           # Shape: [B, 256, 4, 4]
        x = self.layer4(x)           # Shape: [B, 512, 2, 2]

        x = self.avgpool(x)          # Global Avg Pooling: [B, 512, 1, 1]
        h = torch.flatten(x, 1)      # Reshape Flatten: [B, 512]
        return h


# ==============================================================================
# 6. 顶层模型 / Pipeline 主体 (Top-level Architecture)
# ==============================================================================
class SimCLR(nn.Module):
    """
    SimCLR 整体架构：结合 ResNet 骨干编码器与 MLP 非线性投影头 (Projection Head)。

    Args:
        feature_dim (int): 编码器输出特征维度 D_feat (如 512)。
        proj_dim (int): 投影空间特征维度 D_proj (如 128)。

    Inputs:
        x (Tensor): 增强图像批次 (包含 2*B 个视图拼接)，Shape: [2*B, C, H, W]

    Outputs:
        z (Tensor): 投影投影向量，Shape: [2*B, D_proj]
    """
    def __init__(self, feature_dim: int = Config.FEATURE_DIM, proj_dim: int = Config.PROJ_DIM):
        super().__init__()
        self.backbone = ResNet18Backbone(in_channels=3)
        # 非线性 MLP 投影头 (Projection Head)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, proj_dim, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input Shape: [2*B, 3, 64, 64]
        h = self.backbone(x)         # Shape: [2*B, 512]
        z = self.projector(h)        # Shape: [2*B, 128]
        return z


# ==============================================================================
# 7. 损失函数与评估指标 (Loss & Metrics)
# ==============================================================================
class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent 损失函数)。

    数学映射:
        - 相似度矩阵: S_{i,j} = (z_i · z_j^T) / (||z_i|| * ||z_j|| * τ)
        - 变量映射:
            `z`                  <-> z_i, z_j 拼接向量
            `sim_matrix`         <-> S_{i,j}
            `positive_indices`   <-> 正样本对映射索引 i -> j
            `temperature`        <-> τ

    Args:
        temperature (float): 温度控制参数 τ，默认 0.5。

    Inputs:
        z (Tensor): 经过投影头输出的特征矩阵，Shape: [2*B, D_proj]

    Outputs:
        loss (Tensor): 标量 Loss 值，Shape: []
    """
    def __init__(self, temperature: float = Config.TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        total_samples = z.size(0)  # N_total = 2 * B
        batch_size = total_samples // 2

        # 1. 特征向量 L2 归一化: z_norm = z / ||z||_2
        z_norm = F.normalize(z, p=2, dim=1)  # Shape: [2*B, D_proj]

        # 2. 计算全对全余弦相似度矩阵并在维度上除以温度系数 τ
        # sim_matrix_{i,j} = cos(z_i, z_j) / τ
        sim_matrix = torch.matmul(z_norm, z_norm.t()) / self.temperature  # Shape: [2*B, 2*B]

        # 3. 屏蔽对角线 (self-contrast) 的自身相似度
        mask_self = torch.eye(total_samples, dtype=torch.bool, device=z.device)
        sim_matrix = sim_matrix.masked_fill(mask_self, -1e9)  # 将对角线填充极小负数

        # 4. 构建正样本标签索引 (Positive Pair Target)
        # 前 half (v1) 的正样本在后 half (v2)，反之亦然
        # 索引变换：0 -> B, 1 -> B+1 ... B-1 -> 2B-1; B -> 0, B+1 -> 1 ... 2B-1 -> B-1
        pos_targets = torch.cat([
            torch.arange(batch_size, total_samples, device=z.device),
            torch.arange(0, batch_size, device=z.device)
        ], dim=0)  # Shape: [2*B]

        # 5. 使用交叉熵损失计算 Log-Softmax 概率分布
        loss = F.cross_entropy(sim_matrix, pos_targets)
        return loss


# ==============================================================================
# 8. 训练/推理逻辑与入口 (Training/Inference Execution)
# ==============================================================================
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    augmentor: ContrastiveAugmentation,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """
    单 Epoch 训练 Pipeline 逻辑。
    """
    model.train()
    running_loss = 0.0

    for step, x_batch in enumerate(dataloader):
        x_batch = x_batch.to(device)  # Input Shape: [B, 3, 64, 64]

        # 1. 为每个图像生成两路视图 v1 与 v2
        v1, v2 = augmentor(x_batch)   # v1, v2 Shape: [B, 3, 64, 64]

        # 2. 拼接两视图送入 Batch -> 构造成 [2*B, 3, 64, 64]
        x_concat = torch.cat([v1, v2], dim=0)  # Shape: [2*B, C, H, W]

        # 3. 前向传播计算投影向量 z
        z = model(x_concat)                    # Shape: [2*B, D_proj]

        # 4. 计算 NT-Xent 对比损失
        loss = criterion(z)

        # 5. 反向传播与优化器更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def main():
    print(f"[INFO] 运行设备: {Config.DEVICE}")

    # 数据集与数据加载器构建
    dataset = SyntheticImageDataset(num_samples=640, height=64, width=64)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)

    # 管道模块实例化
    augmentor = ContrastiveAugmentation(noise_std=0.05)
    model = SimCLR(feature_dim=Config.FEATURE_DIM, proj_dim=Config.PROJ_DIM).to(Config.DEVICE)
    criterion = NTXentLoss(temperature=Config.TEMPERATURE).to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)

    # 启动训练循环
    print("开始 SimCLR 对比学习自监督训练...")
    for epoch in range(Config.EPOCHS):
        avg_loss = train_one_epoch(model, dataloader, augmentor, criterion, optimizer, Config.DEVICE)
        print(f"Epoch [{epoch + 1:02d}/{Config.EPOCHS:02d}] | NT-Xent Loss: {avg_loss:.4f}")

    print("[SUCCESS] 训练完成！骨干网络特征表征 h 已就绪，可用于下游迁移学习或分类任务。")


if __name__ == "__main__":
    main()