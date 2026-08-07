"""
任务定义: 任务 01 - 图像分类 (Image Classification)
代表架构: ResNet18 (Deep Residual Learning for Image Recognition, He et al., CVPR 2016)
核心思想: 通过引入残差跳跃连接 (Shortcut Connection) 解决深层神经网络中的梯度消失与退化问题，
          使网络学习残差映射 F(x) = H(x) - x，而非直接拟合目标映射 H(x)。
数学公式:
    1. 残差单元计算: y = F(x, {W_i}) + W_s * x
    2. 交叉熵损失函数 (CrossEntropy Loss):
       L = - (1 / N) * sum_i [ log( exp(z_{i, y_i}) / sum_j exp(z_{i, j}) ) ]
       (其中 z 表示未归一化的 Logits，y_i 为真实类别标签索引)
数据输入规范:
    Input:  Tensor shape [B, C, H, W] = [B, 3, 64, 64] (RGB 图像)
    Output: Tensor shape [B, Num_Classes] = [B, 10] (未经过 Softmax 的类别概率分布 Logits)
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

# ==============================================================================
# 3. 超参数与全局配置 (Hyperparameters & Config)
# ==============================================================================
BATCH_SIZE: int = 64
EPOCHS: int = 5
LEARNING_RATE: float = 1e-3
NUM_CLASSES: int = 10
INPUT_CHANNELS: int = 3
IMAGE_HEIGHT: int = 64
IMAGE_WIDTH: int = 64
NUM_SAMPLES: int = 1000
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# 4. 数据处理与 Dataset 管道 (Data Pipeline & Utils)
# ==============================================================================
def get_synthetic_dataset(
    num_samples: int = NUM_SAMPLES,
    channels: int = INPUT_CHANNELS,
    height: int = IMAGE_HEIGHT,
    width: int = IMAGE_WIDTH,
    num_classes: int = NUM_CLASSES,
) -> Dataset:
    """
    生成合成的 RGB 图像数据与对应的多类别标签，用于测试和演示 Pipeline。

    Args:
        num_samples (int): 生成的样本总数量。
        channels (int): 图像通道数 C。
        height (int): 图像高度 H。
        width (int): 图像宽度 W。
        num_classes (int): 类别总数 K。

    Returns:
        Dataset: 包含图像张量 [N, C, H, W] 与标签张量 [N] 的 PyTorch Dataset。
    """
    # 模拟合成图像: x ~ N(0, 1), shape: [N, C, H, W] = [1000, 3, 64, 64]
    x = torch.randn(num_samples, channels, height, width, dtype=torch.float32)
    # 模拟类别标签: y ~ Uniform(0, K-1), shape: [N] = [1000]
    y = torch.randint(0, num_classes, (num_samples,), dtype=torch.long)
    return TensorDataset(x, y)


# ==============================================================================
# 5. 核心子模块 (Sub-components)
# ==============================================================================
class BasicBlock(nn.Module):
    """
    ResNet18 基础残差块 (BasicBlock)。

    结构组成:
        Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+) Shortcut -> ReLU

    数学原理 / 变换逻辑:
        F(x) = BN2(Conv2(ReLU(BN1(Conv1(x)))))
        y = ReLU(F(x) + Shortcut(x))

    Args:
        in_ch (int): 输入通道数 C_in。
        out_ch (int): 输出通道数 C_out。
        stride (int): 第一个卷积层的步长 (用于下采样)，默认 1。

    Inputs:
        x (Tensor): 输入特征图，shape: [B, C_in, H, W]

    Outputs:
        out (Tensor): 经过残差累加后的特征图，shape: [B, C_out, H', W']
                      其中 H' = H // stride, W' = W // stride
    """

    expansion: int = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()

        # 主路径第一层卷积，可能通过 stride > 1 进行空间下采样
        self.conv1 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        # 主路径第二层卷积，保持特征空间维度
        self.conv2 = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_ch)

        # 侧边 Shortcut 连接: 当维度不匹配或发生下采样时进行 1x1 卷积投影
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入: x shape [B, in_ch, H, W]
        identity = self.shortcut(x)  # shape: [B, out_ch, H//stride, W//stride]

        out = self.conv1(x)  # shape: [B, out_ch, H//stride, W//stride]
        out = self.bn1(out)  # shape: [B, out_ch, H//stride, W//stride]
        out = self.relu(out)  # shape: [B, out_ch, H//stride, W//stride]

        out = self.conv2(out)  # shape: [B, out_ch, H//stride, W//stride]
        out = self.bn2(out)  # shape: [B, out_ch, H//stride, W//stride]

        # 映射公式代码: y = F(x) + W_s * x
        out = out + identity  # 逐元素相加, shape 不变: [B, out_ch, H//stride, W//stride]
        out = self.relu(out)  # shape: [B, out_ch, H//stride, W//stride]

        return out


# ==============================================================================
# 6. 顶层模型 (Top-level Architecture)
# ==============================================================================
class ResNet18(nn.Module):
    """
    手写标准 ResNet18 完整模型实现。

    结构路线:
        Conv7x7(stride=2) -> BN -> ReLU -> MaxPool3x3(stride=2)
        -> Layer1 (BasicBlock x 2, stride=1)
        -> Layer2 (BasicBlock x 2, stride=2)
        -> Layer3 (BasicBlock x 2, stride=2)
        -> Layer4 (BasicBlock x 2, stride=2)
        -> AdaptiveAvgPool2d -> Flatten -> Linear(out_features=num_classes)

    Args:
        in_channels (int): 输入图像通道数，默认 3 (RGB)。
        num_classes (int): 分类任务的类别总数，默认 10。

    Inputs:
        x (Tensor): 批次图像数据，shape: [B, in_channels, H, W]

    Outputs:
        logits (Tensor): 分类 Logits，shape: [B, num_classes]
    """

    def __init__(
        self, in_channels: int = INPUT_CHANNELS, num_classes: int = NUM_CLASSES
    ) -> None:
        super().__init__()

        self.in_channels = 64

        # 茎干网络 (Stem Layer): 7x7 大卷积核迅速降低空间分辨率并提取基础特征
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差主体 (4 个 Stages)
        self.layer1 = self._make_layer(
            out_ch=64, blocks=2, stride=1
        )  # Stage 1: [B, 64, 16, 16]
        self.layer2 = self._make_layer(
            out_ch=128, blocks=2, stride=2
        )  # Stage 2: [B, 128, 8, 8]
        self.layer3 = self._make_layer(
            out_ch=256, blocks=2, stride=2
        )  # Stage 3: [B, 256, 4, 4]
        self.layer4 = self._make_layer(
            out_ch=512, blocks=2, stride=2
        )  # Stage 4: [B, 512, 2, 2]

        # 特征池化与分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应池化到 1x1
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, out_ch: int, blocks: int, stride: int) -> nn.Sequential:
        """构建由多个 BasicBlock 组成的 Layer Stage。"""
        layers = []
        # 首个 Block 负责下采样与通道维度提升
        layers.append(BasicBlock(self.in_channels, out_ch, stride))
        self.in_channels = out_ch * BasicBlock.expansion

        # 后续 Block 保持通道与维度不变
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_ch, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 shape: [B, 3, 64, 64]
        x = self.conv1(x)  # [B, 3, 64, 64] -> [B, 64, 32, 32]
        x = self.bn1(x)  # [B, 64, 32, 32] -> [B, 64, 32, 32]
        x = self.relu(x)  # [B, 64, 32, 32] -> [B, 64, 32, 32]
        x = self.maxpool(x)  # [B, 64, 32, 32] -> [B, 64, 16, 16]

        x = self.layer1(x)  # [B, 64, 16, 16] -> [B, 64, 16, 16]
        x = self.layer2(x)  # [B, 64, 16, 16] -> [B, 128, 8, 8]
        x = self.layer3(x)  # [B, 128, 8, 8]   -> [B, 256, 4, 4]
        x = self.layer4(x)  # [B, 256, 4, 4]   -> [B, 512, 2, 2]

        x = self.avgpool(x)  # [B, 512, 2, 2]   -> [B, 512, 1, 1]
        x = torch.flatten(x, start_dim=1)  # [B, 512, 1, 1]   -> [B, 512]
        logits = self.fc(x)  # [B, 512]         -> [B, 10]

        return logits


# ==============================================================================
# 7. 损失函数与评估指标 (Loss & Metrics)
# ==============================================================================
def compute_accuracy(
    logits: torch.Tensor, targets: torch.Tensor
) -> Tuple[float, int]:
    """
    计算分类 Accuracy 指标。

    Args:
        logits (Tensor): 模型未归一化的预测分值, shape: [B, NUM_CLASSES]
        targets (Tensor): 真实标签分类索引, shape: [B]

    Returns:
        Tuple[float, int]: (当前 Batch 准确率, 预测正确的样本数)
    """
    preds = torch.argmax(logits, dim=-1)  # [B, NUM_CLASSES] -> [B]
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
    """训练单回合 (Epoch) 的核心逻辑。"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)  # shape: [B, 3, 64, 64]
        labels = labels.to(device)  # shape: [B]

        # 前向传播
        optimizer.zero_grad()
        logits = model(images)  # shape: [B, 10]
        loss = criterion(logits, labels)

        # 反向传播与优化
        loss.backward()
        optimizer.step()

        # 统计指标
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        _, correct = compute_accuracy(logits, labels)
        total_correct += correct
        total_samples += batch_size

    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples
    return epoch_loss, epoch_acc


def main() -> None:
    print(f"[*] 使用设备: {DEVICE}")

    # 1. 构建数据加载器
    dataset = get_synthetic_dataset(num_samples=NUM_SAMPLES)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. 实例化模型与优化器
    model = ResNet18(in_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"[*] ResNet18 模型构建完成，总参数量: {sum(p.numel() for p in model.parameters()):,}")
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