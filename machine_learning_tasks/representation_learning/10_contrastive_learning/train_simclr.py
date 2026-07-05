"""
任务 10：对比学习（Contrastive Learning / 判别式自监督）
代表模型：SimCLR（手写 ResNet18 骨干，不调用 torchvision.models）
损失函数：InfoNCE / NT-Xent
使用合成图像训练 ResNet 骨干网络和投影头。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# 超参数
BATCH_SIZE = 32
EPOCHS = 5
LR = 3e-4
TEMPERATURE = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicBlock(nn.Module):
    """ResNet18 的基础残差块。"""

    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
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


class ResNet18Backbone(nn.Module):
    """手写 ResNet18 骨干，去掉最后的全连接层。"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

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
        return x


class SimCLR(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = ResNet18Backbone()
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, proj_dim),
        )

    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        return z


def nt_xent_loss(z, temperature=TEMPERATURE):
    """NT-Xent 损失，假设输入 z 为 2N 个视图（每两个来自同一样本）。"""
    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.t()) / temperature
    n = z.size(0)

    # 屏蔽自身相似度
    mask = torch.eye(n, device=z.device).bool()
    sim = sim.masked_fill(mask, -9e15)

    # 正样本对：偶数 i 与 i+1，奇数 i 与 i-1
    positives = torch.cat([
        torch.arange(1, n, 2, device=z.device),
        torch.arange(0, n, 2, device=z.device),
    ])
    return F.cross_entropy(sim, positives)


def random_augment(x):
    """简单数据增强：随机翻转、裁剪、颜色抖动。"""
    x = torch.flip(x, dims=[-1]) if torch.rand(1).item() > 0.5 else x
    x = x + torch.randn_like(x) * 0.05
    return x


def get_synthetic_dataset(num_samples=1000):
    x = torch.randn(num_samples, 3, 64, 64)
    return TensorDataset(x)


def main():
    dataset = get_synthetic_dataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SimCLR().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for (x,) in loader:
            x = x.to(DEVICE)
            # 每个样本生成两个视图
            v1 = random_augment(x)
            v2 = random_augment(x)
            z = model(torch.cat([v1, v2], dim=0))

            optimizer.zero_grad()
            loss = nt_xent_loss(z)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}]  NT-Xent Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
