"""
任务 10：对比学习（Contrastive Learning / 判别式自监督）
代表模型：CLIP（手写双塔编码器，不调用 torchvision 或预训练模型）
损失函数：对称 InfoNCE（图像→文本 + 文本→图像）
使用合成图像-文本对训练一个双塔对比模型。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# 超参数
BATCH_SIZE = 32
EPOCHS = 5
LR = 3e-4
TEMPERATURE = 0.07
EMBED_DIM = 128
VOCAB_SIZE = 1000
SEQ_LEN = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageEncoder(nn.Module):
    """手写图像编码器：小型 CNN 骨干 + 投影到嵌入空间。"""

    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # 自适应池化成 1*1
        self.proj = nn.Linear(512, embed_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return F.normalize(self.proj(x), dim=-1)


class TextEncoder(nn.Module):
    """手写文本编码器：Embedding + BiGRU + 投影到嵌入空间。"""

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, embed_dim)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)
        # 使用最后一个时间步的输出作为序列表示
        output, _ = self.gru(embedded)
        last_hidden = output[:, -1, :]
        return F.normalize(self.proj(last_hidden), dim=-1)


class CLIP(nn.Module):
    """CLIP 双塔模型：图像塔 + 文本塔 + 可学习温度系数。"""

    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)
        # 可学习温度系数的对数，训练时通过 exp 保证 temperature > 0
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / TEMPERATURE)))

    def forward(self, images, texts):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(texts)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


def contrastive_loss(logits_per_image, logits_per_text):
    """对称 InfoNCE 损失：同时优化图像分类文本和文本分类图像。"""
    batch_size = logits_per_image.size(0)
    labels = torch.arange(batch_size, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2


def get_synthetic_dataset(num_samples=1000):
    """生成随机图像-文本对；配对关系由相同索引隐含。"""
    images = torch.randn(num_samples, 3, 64, 64)
    texts = torch.randint(1, VOCAB_SIZE, (num_samples, SEQ_LEN))
    return TensorDataset(images, texts)


def main():
    dataset = get_synthetic_dataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = CLIP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for images, texts in loader:
            images = images.to(DEVICE)
            texts = texts.to(DEVICE)

            logits_per_image, logits_per_text = model(images, texts)
            loss = contrastive_loss(logits_per_image, logits_per_text)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}]  Contrastive Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
