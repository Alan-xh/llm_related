"""
================================================================================
任务与理论 Header (Task & Theory Header)
================================================================================
任务定义：
    - 任务编号：Task 004
    - 任务名称：语义分割 (Semantic Segmentation)
    - 领域分类：计算机视觉 (Computer Vision)

代表架构/算法：
    - 模型名称：U-Net (Encoder-Decoder with Skip Connections)
    - 论文来源：Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
      Image Segmentation", MICCAI 2015.

核心思想与机制：
    1. Encoder (Contracting Path)：通过连续的卷积层与最大池化层下采样，提取高层抽象语义特征，同时降低空间分辨率。
    2. Bottleneck：底层高通道数密集特征表达，捕捉图像全局/大范围上下文。
    3. Decoder (Expanding Path)：通过转置卷积 (Transposed Conv) 上采样逐步恢复空间分辨率。
    4. Skip Connections (特征跨层拼接)：将 Encoder 对应层的低级高分辨率纹理特征与 Decoder 的上采样特征在通道维度拼接 (Concat)，
       缓解下采样导致的位置信息丢失，提升边界细节分割精度。

数学公式 / 目标函数与代码映射：
    1. 交叉熵损失 (Cross Entropy Loss):
       L_CE = - (1 / N) * \sum_{i=1}^{N} \sum_{c=0}^{C-1} y_{i,c} * log(p_{i,c})
       代码映射: torch.nn.CrossEntropyLoss()

    2. 多分类 Dice 损失 (Multi-class Dice Loss):
       L_Dice = 1 - (1 / C) * \sum_{c=0}^{C-1} [ (2 * \sum_p p_{p,c} * y_{p,c} + \epsilon) / (\sum_p p_{p,c} + \sum_p y_{p,c} + \epsilon) ]
       - p_{p,c}: 像素 p 属于类别 c 的 Softmax 概率，对应代码 `pred`
       - y_{p,c}: 像素 p 的 One-Hot 标签，对应代码 `oh`
       - \epsilon: 平滑项 (smooth)，预防分母为 0，对应代码 `smooth`
       代码映射: dice_loss() 函数

    3. 综合损失函数 (Combined Loss):
       L_total = L_CE + L_Dice

数据输入输出规范：
    - 输入 (Input Tensor):  [B, C_in, H, W] = [Batch_Size, 3, 128, 128], 类型 float32
    - 目标 (Target Tensor): [B, H, W] = [Batch_Size, 128, 128], 类型 int64 (类索引 0 ~ C-1)
    - 输出 (Output Logits): [B, Num_Classes, H, W] = [Batch_Size, 4, 128, 128], 类型 float32
================================================================================
"""

# ==============================================================================
# 2. 依赖导入 (Imports)
# ==============================================================================
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ==============================================================================
# 3. 超参数与全局配置 (Hyperparameters & Config)
# ==============================================================================
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-3
NUM_CLASSES = 4
IMAGE_SIZE = 128
IN_CHANNELS = 3
BASE_CHANNELS = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# 4. 数据处理与 Dataset 管道 (Data Pipeline & Utils)
# ==============================================================================
def get_synthetic_dataset(num_samples=200, size=IMAGE_SIZE, num_classes=NUM_CLASSES):
    """
    生成合成的 2D 图像与像素级分割掩码数据集。

    Args:
        num_samples (int): 样本数量，默认 200。
        size (int): 图像高宽，默认 128。
        num_classes (int): 分割类别数，默认 4。

    Outputs:
        TensorDataset: 包含图像张量与对应掩码张量的数据集。
            - images: [num_samples, 3, size, size], Float32
            - masks:  [num_samples, size, size], Long
    """
    x = torch.randn(num_samples, 3, size, size, dtype=torch.float32)
    y = torch.randint(0, num_classes, (num_samples, size, size), dtype=torch.long)
    return TensorDataset(x, y)


# ==============================================================================
# 5. 核心子模块 / Encoder / Decoder (Sub-components)
# ==============================================================================
class DoubleConvBlock(nn.Module):
    """
    U-Net 基础卷积块：包含两个标准的 (3x3 卷积 -> BatchNorm -> ReLU) 结构，保持空间分辨率 H, W 不变。

    变换逻辑:
        x -> Conv2d(3x3, pad=1) -> BatchNorm -> ReLU -> Conv2d(3x3, pad=1) -> BatchNorm -> ReLU -> out

    Args:
        in_ch (int): 输入通道数 C_in。
        out_ch (int): 输出通道数 C_out。

    Inputs:
        x (Tensor): 输入特征图，shape: [B, in_ch, H, W]

    Outputs:
        out (Tensor): 输出特征图，shape: [B, out_ch, H, W]
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_ch, H, W]
        out = self.conv(x)
        # out: [B, out_ch, H, W]
        return out


# ==============================================================================
# 6. 顶层模型 / Pipeline 主体 (Top-level Architecture / Model)
# ==============================================================================
class UNet(nn.Module):
    """
    U-Net 神经网络模型主体。

    架构特点：
        - 编码器 (Encoder)：2 层下采样 (DoubleConv + MaxPool2d)
        - 瓶颈层 (Bottleneck)：最底层高维特征提取 (DoubleConv)
        - 解码器 (Decoder)：2 层上采样 (ConvTranspose2d + Cat + DoubleConv)
        - 跨层连接 (Skip Connection)：拼接对应编码层的浅层高分辨率特征

    Args:
        in_ch (int): 输入通道数，默认 3。
        num_classes (int): 分割类别数，默认 4。
        base (int): 基础特征通道数，默认 32。

    Inputs:
        x (Tensor): 原始输入图像张量，shape: [B, in_ch, H, W]

    Outputs:
        logits (Tensor): 像素级分类 未归一化 Logits，shape: [B, num_classes, H, W]
    """

    def __init__(self, in_ch: int = IN_CHANNELS, num_classes: int = NUM_CLASSES, base: int = BASE_CHANNELS):
        super().__init__()

        # --- Encoder Path ---
        # Stage 1: [B, in_ch, H, W] -> [B, base, H, W]
        self.enc1 = DoubleConvBlock(in_ch, base)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # H, W -> H/2, W/2

        # Stage 2: [B, base, H/2, W/2] -> [B, base*2, H/2, W/2]
        self.enc2 = DoubleConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # H/2, W/2 -> H/4, W/4

        # --- Bottleneck ---
        # Stage Bottleneck: [B, base*2, H/4, W/4] -> [B, base*4, H/4, W/4]
        self.bottleneck = DoubleConvBlock(base * 2, base * 4)

        # --- Decoder Path ---
        # Up 2: 上采样与解码 Stage 2
        # Transposed Conv: [B, base*4, H/4, W/4] -> [B, base*2, H/2, W/2]
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        # Dec 2: Concat 后通道数 (base*2 + base*2) -> base*4，解码输出 base*2
        self.dec2 = DoubleConvBlock(base * 4, base * 2)

        # Up 1: 上采样与解码 Stage 1
        # Transposed Conv: [B, base*2, H/2, W/2] -> [B, base, H, W]
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        # Dec 1: Concat 后通道数 (base + base) -> base*2，解码输出 base
        self.dec1 = DoubleConvBlock(base * 2, base)

        # --- Out Head ---
        # 1x1 Conv: 映射特征通道至类别数 [B, base, H, W] -> [B, num_classes, H, W]
        self.out_conv = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [B, 3, H, W]  (例: [8, 3, 128, 128])

        # ---------------- Encoder ----------------
        e1 = self.enc1(x)               # -> [B, 32, H, W]       ([8, 32, 128, 128])
        p1 = self.pool1(e1)             # -> [B, 32, H/2, W/2]   ([8, 32, 64, 64])

        e2 = self.enc2(p1)              # -> [B, 64, H/2, W/2]   ([8, 64, 64, 64])
        p2 = self.pool2(e2)             # -> [B, 64, H/4, W/4]   ([8, 64, 32, 32])

        # --------------- Bottleneck --------------
        b = self.bottleneck(p2)         # -> [B, 128, H/4, W/4]  ([8, 128, 32, 32])

        # ---------------- Decoder ----------------
        up_b = self.up2(b)              # -> [B, 64, H/2, W/2]   ([8, 64, 64, 64])
        cat2 = torch.cat([up_b, e2], dim=1)  # -> [B, 128, H/2, W/2] ([8, 128, 64, 64])
        d2 = self.dec2(cat2)            # -> [B, 64, H/2, W/2]   ([8, 64, 64, 64])

        up_d2 = self.up1(d2)            # -> [B, 32, H, W]       ([8, 32, 128, 128])
        cat1 = torch.cat([up_d2, e1], dim=1) # -> [B, 64, H, W]      ([8, 64, 128, 128])
        d1 = self.dec1(cat1)            # -> [B, 32, H, W]       ([8, 32, 128, 128])

        # ---------------- Output -----------------
        logits = self.out_conv(d1)      # -> [B, Num_Classes, H, W] ([8, 4, 128, 128])

        return logits


# ==============================================================================
# 7. 损失函数与评估指标 (Loss & Metrics)
# ==============================================================================
def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    多分类 Dice Loss 计算函数。关注图像区域级重叠度，有效应对背景与前景类别不平衡问题。

    数学原理:
        Dice = (2 * |P \cap Y| + \epsilon) / (|P| + |Y| + \epsilon)
        Loss = 1 - Mean(Dice)

    Args:
        pred (Tensor): 模型未归一化输出 Logits，shape: [B, C, H, W]
        target (Tensor): 真实目标类别索引 Mask，shape: [B, H, W]，取值范围 [0, C-1]
        smooth (float): 平滑系数 \epsilon，防止分母为 0，默认 1.0。

    Inputs:
        pred: [B, C, H, W]
        target: [B, H, W]

    Outputs:
        loss (Tensor): 标量 Dice Loss 损失值 (Scalar)
    """
    num_classes = pred.shape[1]  # 取类别数 C

    # 1. 软最大化计算概率值 p_{p,c}
    probs = F.softmax(pred, dim=1)  # [B, C, H, W]

    # 2. 转换真实 Target 为 One-Hot 编码 y_{p,c}
    target_one_hot = F.one_hot(target, num_classes=num_classes)  # [B, H, W] -> [B, H, W, C]
    target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # -> [B, C, H, W]

    # 3. 维度展开与交集/并集求和计算 (|P \cap Y| 与 |P| + |Y|)
    intersection = torch.sum(probs * target_one_hot, dim=(2, 3))  # [B, C]
    cardinality = torch.sum(probs, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))  # [B, C]

    # 4. 计算每个 Channel/Class 的 Dice 分数并求均值损失
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth)  # [B, C]
    loss = 1.0 - torch.mean(dice_score)  # Scalar

    return loss


# ==============================================================================
# 8. 训练/推理逻辑与入口 (Training/Inference Execution)
# ==============================================================================
def main():
    print(f"[*] 使用设备: {DEVICE}")

    # 1. 初始化构建数据加载器 (DataLoader)
    dataset = get_synthetic_dataset(num_samples=200, size=IMAGE_SIZE, num_classes=NUM_CLASSES)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. 实例化模型、优化器与损失函数
    model = UNet(in_ch=IN_CHANNELS, num_classes=NUM_CLASSES, base=BASE_CHANNELS).to(DEVICE)
    ce_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 3. 训练循环
    model.train()
    print("[*] 开始训练...")
    for epoch in range(EPOCHS):
        running_loss = 0.0
        running_ce = 0.0
        running_dice = 0.0

        for images, masks in train_loader:
            images = images.to(DEVICE)  # [B, 3, H, W]
            masks = masks.to(DEVICE)    # [B, H, W]

            optimizer.zero_grad()

            # 前向传播
            logits = model(images)      # [B, NUM_CLASSES, H, W]

            # 计算复合损失: CE + Dice
            loss_ce = ce_criterion(logits, masks)
            loss_dice = dice_loss(logits, masks)
            total_loss = loss_ce + loss_dice

            # 反向传播与优化
            total_loss.backward()
            optimizer.step()

            # 统计 Loss
            running_loss += total_loss.item()
            running_ce += loss_ce.item()
            running_dice += loss_dice.item()

        num_batches = len(train_loader)
        avg_loss = running_loss / num_batches
        avg_ce = running_ce / num_batches
        avg_dice = running_dice / num_batches

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] | "
            f"Total Loss: {avg_loss:.4f} (CE Loss: {avg_ce:.4f}, Dice Loss: {avg_dice:.4f})"
        )

    print("[*] 训练完成！")


if __name__ == "__main__":
    main()