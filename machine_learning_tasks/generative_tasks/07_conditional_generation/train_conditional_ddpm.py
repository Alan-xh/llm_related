"""
任务 7：条件生成（Conditional Generation）
代表模型：条件扩散模型（Class-conditional DDPM）
损失函数：MSE（预测噪声）
在无条件 DDPM 基础上加入类别条件嵌入，演示条件生成训练。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# 超参数
BATCH_SIZE = 64
EPOCHS = 10
LR = 2e-4
T = 1000
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


betas = torch.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        t = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(t), torch.cos(t)], dim=-1)


class ConditionalTinyUNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=NUM_CLASSES, base=64, cond_dim=128):
        super().__init__()
        self.time_embed = nn.Sequential(
            TimeEmbedding(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
        )
        self.class_embed = nn.Embedding(num_classes, cond_dim)

        self.down1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
        )
        self.mid = nn.Sequential(
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base * 2, base, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(),
        )
        self.up1 = nn.Conv2d(base, in_ch, 3, padding=1)

    def forward(self, x, t, y):
        cond = self.time_embed(t) + self.class_embed(y)
        h = self.down1(x)
        h = self.down2(h)
        h = h + cond[:, :, None, None]
        h = self.mid(h)
        h = self.up2(h)
        return self.up1(h)


def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_acp = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1).to(x0.device)
    sqrt_omc = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1).to(x0.device)
    return sqrt_acp * x0 + sqrt_omc * noise


def get_synthetic_dataset(num_samples=1000):
    x0 = torch.randn(num_samples, 3, 32, 32)
    y = torch.randint(0, NUM_CLASSES, (num_samples,))
    return TensorDataset(x0, y)


def main():
    train_loader = DataLoader(
        get_synthetic_dataset(), batch_size=BATCH_SIZE, shuffle=True
    )

    model = ConditionalTinyUNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for x0, y in train_loader:
            x0 = x0.to(DEVICE)
            y = y.to(DEVICE)
            t = torch.randint(0, T, (x0.size(0),), device=DEVICE)
            noise = torch.randn_like(x0)
            xt = q_sample(x0, t, noise)

            pred_noise = model(xt, t, y)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}]  Conditional Noise MSE: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
