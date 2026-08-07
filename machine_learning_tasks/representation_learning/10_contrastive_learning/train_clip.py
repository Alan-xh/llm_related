"""
任务定义: 任务 10 - 对比学习（Contrastive Learning / 判别式自监督）
代表架构: CLIP (Contrastive Language-Image Pre-training) 双塔编码器模型
核心思想: 建立图像塔与文本塔，将多模态数据映射到同一共享的高维嵌入空间。通过对称 InfoNCE 损失，
         极大化对齐的图像-文本对的余弦相似度，同时极小化未对齐（负样本）对的相似度。
数学公式:
    1. 余弦相似度 Logits:
       S_{i,j} = \tau \cdot \frac{z_i^I}{\|z_i^I\|_2} \cdot \left(\frac{z_j^T}{\|z_j^T\|_2}\right)^T
       其中 \tau = \exp(\text{logit\_scale}) 为可学习温度倒数。

    2. 对称 InfoNCE 损失函数:
       L_{I \to T} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp(S_{i,i})}{\sum_{j=1}^B \exp(S_{i,j})}
       L_{T \to I} = -\frac{1}{B} \sum_{j=1}^B \log \frac{\exp(S_{j,j})}{\sum_{i=1}^B \exp(S_{i,j})}
       L_{total}   = \frac{1}{2} (L_{I \to T} + L_{T \to I})

数据输入规范:
    - 图像张量 (Images): [B, C, H, W] = [B, 3, 64, 64]
    - 文本张量 (Texts):  [B, Seq_Len] = [B, 32]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ======================================================================================
# 3. 超参数与全局配置 (Hyperparameters & Config)
# ======================================================================================
BATCH_SIZE = 32
EPOCHS = 5
LR = 3e-4
TEMPERATURE = 0.07
EMBED_DIM = 128
VOCAB_SIZE = 1000
SEQ_LEN = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================================
# 4. 数据处理与 Dataset 管道 (Data Pipeline & Utils)
# ======================================================================================
def get_synthetic_dataset(num_samples: int = 1000) -> TensorDataset:
    """
    生成随机图像-文本对的数据集，正样本映射通过相同的批次索引隐式定义。

    Args:
        num_samples (int): 合成样本数量，默认 1000。

    Outputs:
        dataset (TensorDataset): 包含 images [N, 3, 64, 64] 和 texts [N, 32] 的数据集。
    """
    images = torch.randn(num_samples, 3, 64, 64)
    texts = torch.randint(1, VOCAB_SIZE, (num_samples, SEQ_LEN))
    return TensorDataset(images, texts)


# ======================================================================================
# 5. 核心子模块 / Encoder / Decoder (Sub-components)
# ======================================================================================
class ImageEncoder(nn.Module):
    """
    手写图像编码器：小型 CNN 骨干网络 + 投影与 L2 归一化。

    数学原理 / 变换逻辑:
        x_feat = ConvNet(x)                          # 特征提取
        x_pooled = AdaptiveAvgPool2d(x_feat)          # 全局池化 [B, 512, 1, 1]
        z_I = Linear(Flatten(x_pooled))              # 投影到共享嵌入空间
        z_I_norm = z_I / ||z_I||_2                   # L2 规范化

    Args:
        embed_dim (int): 共享高维嵌入空间维度，默认 EMBED_DIM。

    Inputs:
        x (Tensor): 输入图像张量，shape: [B, C, H, W] = [B, 3, 64, 64]

    Outputs:
        out (Tensor): L2 归一化后的图像嵌入向量，shape: [B, embed_dim]
    """

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.features = nn.Sequential(
            # Layer 1: [B, 3, 64, 64] -> [B, 64, 32, 32]
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            # Layer 2: [B, 64, 32, 32] -> [B, 128, 16, 16]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            # Layer 3: [B, 128, 16, 16] -> [B, 256, 8, 8]
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            # Layer 4: [B, 256, 8, 8] -> [B, 512, 4, 4]
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # [B, 512, 4, 4] -> [B, 512, 1, 1]
        self.proj = nn.Linear(512, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, 3, 64, 64]
        feat = self.features(x)  # shape: [B, 512, 4, 4]
        pooled = self.pool(feat)  # shape: [B, 512, 1, 1]
        flat = torch.flatten(pooled, 1)  # shape: [B, 512]
        projected = self.proj(flat)  # shape: [B, embed_dim]
        # L2 归一化使得点积等价于余弦相似度
        return F.normalize(projected, p=2, dim=-1)  # shape: [B, embed_dim]


class TextEncoder(nn.Module):
    """
    手写文本编码器：Embedding + 双向 GRU + 线性投影与 L2 归一化。

    数学原理 / 变换逻辑:
        E = Embedding(X_text)                       # [B, Seq_Len, Hidden_Dim]
        H, _ = BiGRU(E)                             # [B, Seq_Len, Hidden_Dim * 2]
        z_T = Linear(H[:, -1, :])                   # 抽取末尾时间步并投影
        z_T_norm = z_T / ||z_T||_2                   # L2 规范化

    Args:
        vocab_size (int): 词表大小，默认 VOCAB_SIZE。
        embed_dim (int): 投影到的目标嵌入空间维度，默认 EMBED_DIM。
        hidden_dim (int): GRU 隐层特征维度，默认 256。

    Inputs:
        x (Tensor): 输入文本 Index 张量，shape: [B, Seq_Len]

    Outputs:
        out (Tensor): L2 归一化后的文本嵌入向量，shape: [B, embed_dim]
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.gru = nn.GRU(
            hidden_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.proj = nn.Linear(hidden_dim * 2, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, Seq_Len]
        embedded = self.embedding(x)  # shape: [B, Seq_Len, hidden_dim]
        output, _ = self.gru(
            embedded
        )  # output shape: [B, Seq_Len, hidden_dim * 2]
        last_hidden = output[:, -1, :]  # shape: [B, hidden_dim * 2]
        projected = self.proj(last_hidden)  # shape: [B, embed_dim]
        # L2 归一化
        return F.normalize(projected, p=2, dim=-1)  # shape: [B, embed_dim]


# ======================================================================================
# 6. 顶层模型 / Pipeline 主体 (Top-level Architecture / Model)
# ======================================================================================
class CLIP(nn.Module):
    """
    CLIP 双塔模型：包含图像编码器、文本编码器与可学习温度系数。

    数学原理 / 变换逻辑:
        1. z_I = ImageEncoder(images)                 # [B, embed_dim]
        2. z_T = TextEncoder(texts)                   # [B, embed_dim]
        3. scale = exp(logit_scale)                   # \tau (温度倒数)
        4. Logits_I2T = scale * (z_I @ z_T^T)         # [B, B] 图像->文本相似度矩阵
        5. Logits_T2I = Logits_I2T^T                  # [B, B] 文本->图像相似度矩阵

    Args:
        embed_dim (int): 共享多模态嵌入空间维度，默认 EMBED_DIM。
        init_temperature (float): 温度参数初始值，默认 TEMPERATURE。

    Inputs:
        images (Tensor): 输入图像张量，shape: [B, 3, H, W]
        texts (Tensor): 输入文本 Token 张量，shape: [B, Seq_Len]

    Outputs:
        logits_per_image (Tensor): 图像到文本的相似度矩阵，shape: [B, B]
        logits_per_text (Tensor): 文本到图像的相似度矩阵，shape: [B, B]
    """

    def __init__(
        self, embed_dim: int = EMBED_DIM, init_temperature: float = TEMPERATURE
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim=embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)

        # 参数化 logit_scale 以防数值不稳定 (logit_scale = ln(1 / \tau))
        self.logit_scale = nn.Parameter(
            torch.ones([]) * math.log(1.0 / init_temperature)
        )

    def forward(
        self, images: torch.Tensor, texts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # images: [B, 3, 64, 64], texts: [B, 32]
        image_features = self.image_encoder(images)  # shape: [B, embed_dim]
        text_features = self.text_encoder(texts)  # shape: [B, embed_dim]

        # 计算指数倍率: logit_scale = log(1 / tau) -> exp(logit_scale) = 1 / tau
        logit_scale = self.logit_scale.exp()  # 标量 Scalar

        # 矩阵乘法计算余弦相似度: [B, embed_dim] @ [embed_dim, B] -> [B, B]
        logits_per_image = (
            logit_scale * image_features @ text_features.t()
        )  # shape: [B, B]
        logits_per_text = logits_per_image.t()  # shape: [B, B]

        return logits_per_image, logits_per_text


# ======================================================================================
# 7. 损失函数与评估指标 (Loss & Metrics)
# ======================================================================================
class ContrastiveLoss(nn.Module):
    """
    对称 InfoNCE 交叉熵损失函数。

    数学公式:
        L_{total} = 0.5 * (CrossEntropy(logits_I2T, labels) + CrossEntropy(logits_T2I, labels))
        其中 labels 为对角线索引 [0, 1, ..., B-1]。
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, logits_per_image: torch.Tensor, logits_per_text: torch.Tensor
    ) -> torch.Tensor:
        # logits_per_image: [B, B], logits_per_text: [B, B]
        batch_size = logits_per_image.size(0)
        labels = torch.arange(batch_size, device=logits_per_image.device)

        # 交叉熵损失计算（包含 Softmax 与 NLLLoss）
        loss_i = F.cross_entropy(logits_per_image, labels)  # 图像到文本分类损失
        loss_t = F.cross_entropy(logits_per_text, labels)  # 文本到图像分类损失

        return (loss_i + loss_t) / 2.0


# ======================================================================================
# 8. 训练/推理逻辑与入口 (Training/Inference Execution)
# ======================================================================================
def main():
    # 1. 初始化数据管道
    dataset = get_synthetic_dataset(num_samples=1000)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. 实例化模型、损失函数与优化器
    model = CLIP(embed_dim=EMBED_DIM, init_temperature=TEMPERATURE).to(DEVICE)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print(f"=== CLIP 对比学习模型配置完成 | 运行设备: {DEVICE} ===")
    print(f"输入图像尺寸: [B, 3, 64, 64] | 文本序列长度: [B, {SEQ_LEN}]")

    # 3. 训练循环
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for step, (images, texts) in enumerate(loader):
            images = images.to(DEVICE)  # shape: [B, 3, 64, 64]
            texts = texts.to(DEVICE)  # shape: [B, 32]

            # 前向传播
            logits_per_image, logits_per_text = model(images, texts)
            loss = criterion(logits_per_image, logits_per_text)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        current_temp = (1.0 / model.logit_scale.exp()).item()
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] | "
            f"Contrastive Loss: {avg_loss:.4f} | "
            f"Learned Temp (\u03c4): {current_temp:.4f}"
        )


if __name__ == "__main__":
    main()