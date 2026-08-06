''' 变分自编码器 '''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt

# -------------------- 超参数 --------------------
EPOCHS = 50
BATCH_SIZE = 128
LATENT_DIM = 20
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data"
RESULTS_DIR = "./vae_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------- 数据加载 --------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 将像素值归一化到 [-1, 1]
])

train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------- VAE 模型定义 --------------------
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(200, latent_dim)
        self.logvar_layer = nn.Linear(200, latent_dim)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Tanh()  # 输出范围 [-1, 1]，与归一化匹配
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# -------------------- 损失函数 --------------------
def vae_loss(recon_x, x, mu, logvar):
    # 重建损失（二元交叉熵或MSE，此处用MSE因为Tanh输出）
    recon_loss = nn.functional.mse_loss(recon_x, x.view(-1, 784), reduction='sum')
    # KL散度
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss

# -------------------- 训练函数 --------------------
def train():
    model = VAE(LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"使用设备: {DEVICE}")
    print(f"训练集大小: {len(train_dataset)} 张图片")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        num_batches = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(DEVICE)
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data)
            loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1

        avg_loss = total_loss / (num_batches * BATCH_SIZE)
        avg_recon = total_recon / (num_batches * BATCH_SIZE)
        avg_kl = total_kl / (num_batches * BATCH_SIZE)

        print(f"Epoch [{epoch}/{EPOCHS}] - 平均损失: {avg_loss:.4f}, 重建: {avg_recon:.4f}, KL: {avg_kl:.4f}")

        # 每5个epoch保存生成的图像样本
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                # 从标准正态分布采样生成图像
                sample_z = torch.randn(64, LATENT_DIM).to(DEVICE)
                sample = model.decode(sample_z).cpu()
                save_image(sample.view(64, 1, 28, 28),
                           f"{RESULTS_DIR}/sample_epoch_{epoch}.png", nrow=8, normalize=True)

                # 重建一些测试样本（使用训练集前64张）
                test_data, _ = next(iter(train_loader))
                test_data = test_data.to(DEVICE)
                recon, _, _ = model(test_data)
                comparison = torch.cat([test_data[:64], recon.view(-1, 1, 28, 28)[:64]])
                save_image(comparison.cpu(),
                           f"{RESULTS_DIR}/recon_epoch_{epoch}.png", nrow=8, normalize=True)

    # 保存模型权重
    torch.save(model.state_dict(), f"{RESULTS_DIR}/vae_model.pth")
    print("训练完成！模型已保存。")

# -------------------- 可视化潜在空间（可选） --------------------
def visualize_latent_space():
    model = VAE(LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(f"{RESULTS_DIR}/vae_model.pth", map_location=DEVICE))
    model.eval()

    # 获取部分数据的潜在表示
    data_iter = iter(train_loader)
    data, labels = next(data_iter)
    data = data.to(DEVICE)
    mu, _ = model.encode(data.view(-1, 784))
    mu = mu.cpu().detach().numpy()
    labels = labels.numpy()

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(mu[:, 0], mu[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title("潜在空间二维可视化 (前两个维度)")
    plt.xlabel("z0")
    plt.ylabel("z1")
    plt.savefig(f"{RESULTS_DIR}/latent_space.png")
    plt.show()

if __name__ == "__main__":
    train()
    # 取消注释下面一行以可视化潜在空间（需要matplotlib）
    # visualize_latent_space()