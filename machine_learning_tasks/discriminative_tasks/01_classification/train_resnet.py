"""
任务 1：分类（Classification）
代表模型：ResNet18（手写实现，不调用 torchvision.models）
损失函数：交叉熵损失
使用合成 64x64 图像数据演示训练流程。
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 超参数
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicBlock(nn.Module):
    """ResNet18 的基础残差块"""

    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        ''' conv -> bn -> relu - > conv - > bn -> add -> relu '''
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch) # 批归一，在 [b, h, w] 维度进行归一化， 同一批图片中，所有像素点在同一个通道上的整体表现
        self.relu = nn.ReLU(inplace=True) # 激活函数
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    """手写 ResNet18。"""

    def __init__(self, num_classes=NUM_CLASSES):
        ''' conv -> bn -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc '''
        super().__init__()
        # 输入卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 个残差块
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 任何尺寸自适应转成 1:1 大小
        # 全连接层，映射多类别概率分布
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def get_synthetic_dataset(num_samples=1000):
    """生成随机 64x64 RGB 图像和对应类别标签。"""
    x = torch.randn(num_samples, 3, 64, 64)
    y = torch.randint(0, NUM_CLASSES, (num_samples,))
    return TensorDataset(x, y)


def main():
    train_loader = DataLoader(
        get_synthetic_dataset(), batch_size=BATCH_SIZE, shuffle=True
    )

    # 手写 ResNet18，最后一层输出 NUM_CLASSES
    model = ResNet18(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss() # 损失函数: 交叉熵
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}]  CrossEntropy Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
