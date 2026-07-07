"""
任务 4：语义分割（Semantic Segmentation）
代表模型：U-Net
损失函数：交叉熵损失 + Dice Loss
使用合成 128x128 图像和像素级掩码演示训练流程。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# 超参数
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-3
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        ''' [b, in_ch] -> [in_ch, out_ch] + bn + relu  + [out_ch, out_ch] + bn + relu  -> [b, out_ch]'''
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=NUM_CLASSES, base=32):
        super().__init__()
        # 上采样 [b, in_chanel] -> [b, ]
        self.enc1 = ConvBlock(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        # 瓶颈
        self.bottleneck = ConvBlock(base * 2, base * 4)

        # 转置卷积,根据 stride 间隔插 0,然后 padding,然后四周补上 kernal_size - 1 - padding 的 0，然后将原始卷积核上下左右翻转然后进行卷积计算
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def dice_loss(pred, target, smooth=1.0):
    """多分类 Dice Loss"""
    pred = F.softmax(pred, dim=1)
    oh = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (pred * oh).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + oh.sum(dim=(2, 3))
    return 1.0 - ((2.0 * intersection + smooth) / (union + smooth)).mean()


def get_synthetic_dataset(num_samples=200, size=128):
    x = torch.randn(num_samples, 3, size, size)
    y = torch.randint(0, NUM_CLASSES, (num_samples, size, size))
    return TensorDataset(x, y)


def main():
    train_loader = DataLoader(
        get_synthetic_dataset(), batch_size=BATCH_SIZE, shuffle=True
    )

    model = UNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}]  Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
