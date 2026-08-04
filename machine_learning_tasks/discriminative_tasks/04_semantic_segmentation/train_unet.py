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
        ''' 
        卷积块，h 和 w 不变，通道数变化,  [b, in_ch] -> [in_ch, out_ch] + bn + relu  + [out_ch, out_ch] + bn + relu  -> [b, out_ch]
        '''
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
        '''
        U-Net 模型

        下采样: 随着卷积层堆叠和池化（MaxPool2d），特征图的高和宽不断减半, 网络最底层（瓶颈）的每一个像素，对应原始输入图像中的范围越来越大,瓶颈层能够看到“全局”或“大范围”的上下文信息。
               浅层（enc1）通道少，负责提取低级特征（如边缘、颜色、纹理）；深层（bottleneck）通道多，负责提取高级语义特征（如“狗的头”、“汽车的轮子”）。
               bottleneck 通道数最多，每张图片最小，称为颈部

        Args:
            in_ch (int): 输入通道数
            num_classes (int): 类别数
            base (int): 基础通道数
        '''
        super().__init__()
        # 下采样 [b, in_ch] -> [b, base]
        self.enc1 = ConvBlock(in_ch, base)
        # 池化,特征图的高和宽减半
        self.pool1 = nn.MaxPool2d(2)
        # 下采样 [b, base] -> [b, base * 2]
        self.enc2 = ConvBlock(base, base * 2)
        # 池化
        self.pool2 = nn.MaxPool2d(2)

        # 瓶颈
        self.bottleneck = ConvBlock(base * 2, base * 4)

        # 转置卷积,根据 stride 间隔插 0,然后 padding,然后四周补上 kernal_size - 1 - padding 的 0，然后将原始卷积核上下左右翻转然后进行卷积计算
        # 转置卷积将图片变大，通道数变小，h_out = (h_in −1) × s − 2p + d × (k − 1)+ op + 1
        # 将转置卷积 + input 拼接起来送入 ConvBlock 解码
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2) # [b, base * 4, h, w] -> [b, base * 2, h, w]
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2) # [b, base * 2, h/2, w/2] -> [b, base, h, w]
        self.dec1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x):
        # 下采样
        e1 = self.enc1(x) # [b, 3, h, w] -> [b, base, h, w]
        e2 = self.enc2(self.pool1(e1)) # [b, base * 2, h/2, w/2]
        b = self.bottleneck(self.pool2(e2)) # [b, base * 4, h/4, w/4]
        # 上采样
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1)) # [b, base * 2, h/2, w/2]
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1)) # [b, base, h, w]
        return self.out(d1)


def dice_loss(pred, target, smooth=1.0):
    """多分类 Dice Loss, 区域级别损失，关注整体重叠度, 对类别不平衡不敏感， 能有效处理小目标。
    
    Args:
        pred (Tensor): 预测结果，[b, c, h, w]，为模型输出的 logits（未经过 softmax）
        target (Tensor): 目标结果，[b, h, w]，每个像素值为类别索引（0, 1, ..., c-1）
        smooth (float): 平滑系数，防止分母为0，默认为1.0
        
    Returns:
        Tensor: 标量损失值
        
    Note:
        - 内部会自动对 pred 进行 softmax 操作，因此输入应为 logits
        - target 会被自动转换为 one-hot 编码
        - 计算每个类别的 Dice 系数后取平均
    """
    pred = F.softmax(pred, dim=1)
    oh = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float() # one_hot 会 target 在最后加一维度 [8, 128, 128, 4]
    intersection = (pred * oh).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + oh.sum(dim=(2, 3)) # [8, 4]
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
