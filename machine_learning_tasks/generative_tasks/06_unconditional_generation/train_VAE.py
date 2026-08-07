"""
任务定义:
    - 任务名称: 变分自编码器 (Variational Autoencoder, VAE) 图像重建与无监督生成
    - 领域分类: 生成式模型 (Generative Modeling) / 无监督学习 (Unsupervised Learning)

代表架构/算法:
    - VAE (Auto-Encoding Variational Bayes)
    - 论文来源: Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014.

核心思想与机制:
    - 通过概率图模型对数据分布 p_theta(x) 进行建模，将高维数据 x 映射到连续的低维隐空间(多组均值和对数方差) z ~ q_phi(z|x)。
    - 为解决采样过程的不可导问题，采用重参数化技巧 (Reparameterization Trick)：
      z = \mu(x) + \sigma(x) \odot \epsilon, \epsilon ~ N(0, I)。
    - 解码器 p_theta(x|z) 从隐变量 z 重新构建原始图像。

数学公式与目标函数 (ELBO Optimization):
    - 证据下界 (Evidence Lower Bound, ELBO):
      ELBO(\theta, \phi; x) = E_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))
    - 损失函数 (Loss Function):
      Loss = L_{Reconstruction} + L_{KL}
      1. L_{Reconstruction} (MSE Loss):
         ||x - \hat{x}||^2 = \sum_{i=1}^{D} (x_i - \hat{x}_i)^2
      2. L_{KL} (Analytic KL Divergence for Gaussian):
         D_{KL}(N(\mu, \sigma^2) || N(0, I)) = -0.5 * \sum_{j=1}^{d} (1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2)

数据输入规范:
    - 输入 (Input Tensor):  [B, C, H, W] = [Batch_Size, 1, 28, 28] (MNIST 图像)
    - 隐变量 (Latent Tensor): [B, Latent_Dim] = [Batch_Size, 20]
    - 输出 (Output Tensor): [B, 1, 28, 28] (重建图像)
"""

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt


# -------------------- 3. 超参数与全局配置 (Hyperparameters & Config) --------------------
class Config:
    """全局配置类，统一定义训练、模型结构与路径超参数"""
    # 硬件与数据配置
    EPOCHS: int = 50
    BATCH_SIZE: int = 128
    LEARNING_RATE: float = 1e-3
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 架构配置
    INPUT_DIM: int = 784  # 1 * 28 * 28
    HIDDEN_DIM_1: int = 400
    HIDDEN_DIM_2: int = 200
    LATENT_DIM: int = 20
    
    # 路径配置
    DATA_DIR: str = "./data"
    RESULTS_DIR: str = "./vae_results"


# -------------------- 4. 数据处理与 Dataset 管道 (Data Pipeline) --------------------
def get_data_loaders(config: Config):
    """
    加载 MNIST 数据集并构建 DataLoader 管道。

    Args:
        config (Config): 超参数配置对象

    Returns:
        train_loader (DataLoader): 训练集数据加载器
        train_dataset (Dataset): 训练集对象
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 归一化像素值至 [-1, 1], arg1 表示均值， arg2 表示标准差, (0.5) 表示单通道, RGB 图像为 (0.5, 0.5, 0.5)
    ])

    train_dataset = datasets.MNIST(
        root=config.DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True, # 训练时打乱数据
        num_workers=2,
        pin_memory=True if config.DEVICE.type == "cuda" else False
    )

    return train_loader, train_dataset


# -------------------- 5. 核心子模块 / Encoder / Decoder --------------------
class Encoder(nn.Module):
    """
    VAE 编码器网络 (Variational Encoder)
    将高维图像特征映射为高斯隐分布的均值 \mu 与对数方差 \log(\sigma^2)。

    - 均值
        模型通过反向传播学习如何将相似语义的输入数据, 映射到隐空间中相近的区域中心。
    - 方差
        代表了输入数据在隐空间中每个维度上的不确定性（分散程度）,对数方差主要是为了数值计算的稳定性，并防止方差出现负数。
        模型学习对每个特征维度的确定性有多大。如果某个特征对重建输入至关重要且确定，方差会变小；如果存在模糊性或多种可能，方差会变大。

    Args:
        input_dim (int): 输入展平后的特征维度，例如 784
        hidden_dim1 (int): 第一层隐藏层神经元数量，默认 400
        hidden_dim2 (int): 第二层隐藏层神经元数量，默认 200
        latent_dim (int): 隐空间维度，默认 20

    Inputs:
        x (Tensor): 输入图像张量，shape: [B, C, H, W] 或 [B, input_dim]

    Outputs:
        mu (Tensor): 隐分布均值 \mu，shape: [B, latent_dim]
        logvar (Tensor): 隐分布对数方差 \log(\sigma^2)，shape: [B, latent_dim]
    """
    def __init__(self, input_dim: int = 784, hidden_dim1: int = 400, hidden_dim2: int = 200, latent_dim: int = 20):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.SiLU(),  # 使用现代高效率激活函数 SiLU (Swish)
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.SiLU()
        )
        
        # 变分参数预测层
        self.mu_layer = nn.Linear(hidden_dim2, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim2, latent_dim)

    def forward(self, x: torch.Tensor):
        # 将图片展平, [B, C, H, W] -> [B, input_dim]
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        # 两层感知机完成特征提取, [B, input_dim] -> [B, hidden_dim2]
        h = self.feature_extractor(x)
        
        # 全连接层计算均值和方差, [B, hidden_dim2] -> [B, latent_dim]
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        
        return mu, logvar


class Decoder(nn.Module):
    """
    VAE 解码器网络 (Variational Decoder)
    将低维隐变量 z 重建为高维原始图像数据。

    Args:
        latent_dim (int): 隐空间维度，默认 20
        hidden_dim2 (int): 第一层隐藏层神经元数量，默认 200
        hidden_dim1 (int): 第二层隐藏层神经元数量，默认 400
        output_dim (int): 重建图像展平特征维度，默认 784

    Inputs:
        z (Tensor): 隐变量张量，shape: [B, latent_dim]

    Outputs:
        recon_x (Tensor): 重建的图像张量，shape: [B, output_dim]
    """
    def __init__(self, latent_dim: int = 20, hidden_dim2: int = 200, hidden_dim1: int = 400, output_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.SiLU(),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.SiLU(),
            nn.Linear(hidden_dim1, output_dim),
            nn.Tanh()  # 与归一化 [-1, 1] 的数据匹配
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Shape: [B, latent_dim] -> [B, output_dim]
        recon_x = self.net(z)
        return recon_x


# -------------------- 6. 顶层模型 / Pipeline 主体 --------------------
class VAE(nn.Module):
    """
    变分自编码器 (Variational Autoencoder) 完整架构

    数学原理 / 变换逻辑:
        1. Encode: (mu, logvar) = Encoder(x)
        2. Reparameterize: z = mu + exp(0.5 * logvar) * eps, eps ~ N(0, I)
        3. Decode: recon_x = Decoder(z)

    Args:
        config (Config): 模型配置超参数

    Inputs:
        x (Tensor): 输入图像张量，shape: [B, 1, 28, 28] 或 [B, 784]

    Outputs:
        recon_x (Tensor): 标量重建结果，shape: [B, 784]
        mu (Tensor): 隐空间均值，shape: [B, latent_dim]
        logvar (Tensor): 隐空间对数方差，shape: [B, latent_dim]
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(
            input_dim=config.INPUT_DIM,
            hidden_dim1=config.HIDDEN_DIM_1,
            hidden_dim2=config.HIDDEN_DIM_2,
            latent_dim=config.LATENT_DIM
        )
        self.decoder = Decoder(
            latent_dim=config.LATENT_DIM,
            hidden_dim2=config.HIDDEN_DIM_2,
            hidden_dim1=config.HIDDEN_DIM_1,
            output_dim=config.INPUT_DIM
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧 (Reparameterization Trick), 据对数方差计算标准差, 生成和标准差形状的高斯分布, 计算隐变量 z
        公式映射: z = \mu + \sigma \odot \epsilon, \sigma = \exp(0.5 * \log(\sigma^2))

        Args:
            mu (Tensor): [B, latent_dim]
            logvar (Tensor): [B, latent_dim]

        Returns:
            z (Tensor): 采样隐变量, [B, latent_dim]
        """
        if self.training:
            # std = exp(0.5 * logvar)
            std = torch.exp(0.5 * logvar)  # Shape: [B, latent_dim]
            # eps ~ N(0, I)
            eps = torch.randn_like(std)    # Shape: [B, latent_dim]
            # z = mu + std * eps
            return mu + eps * std          # Shape: [B, latent_dim]
        else_ = None
        # 测试阶段直接取均值作为隐变量
        return mu

    def forward(self, x: torch.Tensor):
        '''
        前向层
        
        将输入展平编码, 得到均值和方差, 通过重参数化采样得到隐变量 z， 返回采样隐变量 z, 均值和方差
        '''
        # 展平输入 Shape: [B, 1, 28, 28] -> [B, 784]
        x_flat = x.view(x.size(0), -1)
        
        # 编码阶段
        mu, logvar = self.encoder(x_flat)  # Shape: [B, LATENT_DIM]
        
        # 重参数化采样
        z = self.reparameterize(mu, logvar)  # Shape: [B, LATENT_DIM]
        
        # 解码重建
        recon_x = self.decoder(z)          # Shape: [B, 784]
        
        return recon_x, mu, logvar


# -------------------- 7. 损失函数与评估指标 --------------------
class VAELoss(nn.Module):
    """
    VAE 联合损失 = 重建 MSE 损失 + KL(Kullback-Leibler, 库尔巴克-莱布勒) 散度正则项
        MSE 目的：最小化重建误差，使模型尽量拟合训练数据
        KL 散度：最小化隐变量分布与标准正态分布的差异

    公式与代码映射:
        Total Loss = L_{recon} + L_{kl}
        L_{recon} = \sum ||x - \hat{x}||^2
        L_{kl} = -0.5 * \sum (1 + \log(\sigma^2) - \mu^2 - \exp(\log(\sigma^2)))
    """
    def __init__(self):
        super().__init__()

    def forward(self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
        """
        Args:
            recon_x (Tensor): 解码重建输出，shape: [B, 784]
            x (Tensor): 原始图像输入，shape: [B, 1, 28, 28] 或 [B, 784]
            mu (Tensor): 隐变量均值，shape: [B, latent_dim]
            logvar (Tensor): 隐变量对数方差，shape: [B, latent_dim]

        Outputs:
            total_loss (Tensor): 标量损失总量
            recon_loss (Tensor): 重建 MSE 损失值
            kl_loss (Tensor): KL 散度正则项损失值
        """
        x_flat = x.view(x.size(0), -1)
        
        # 1. 重建损失 (MSE Loss over Batch sum)
        recon_loss = nn.functional.mse_loss(recon_x, x_flat, reduction='sum')
        
        # 2. KL 散度 (Closed-form KL Divergence for Gaussian)
        # kl_loss = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        total_loss = recon_loss + kl_loss
        return total_loss, recon_loss, kl_loss


# -------------------- 8. 训练/推理逻辑与入口 --------------------
def train_pipeline(config: Config):
    """模型训练与验证 Pipeline"""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # 初始化数据、网络与优化器
    train_loader, train_dataset = get_data_loaders(config)
    model = VAE(config).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = VAELoss()

    print(f"==================================================")
    print(f"运行设备: {config.DEVICE}")
    print(f"训练样本数: {len(train_dataset)} | Batch 大小: {config.BATCH_SIZE}")
    print(f"隐空间维度: {config.LATENT_DIM}")
    print(f"==================================================")

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_loss_epoch = 0.0
        total_recon_epoch = 0.0
        total_kl_epoch = 0.0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(config.DEVICE)  # Shape: [B, 1, 28, 28]
            optimizer.zero_grad()

            # 前向传播
            recon_batch, mu, logvar = model(data)
            
            # 计算损失
            loss, recon_loss, kl_loss = criterion(recon_batch, data, mu, logvar)

            # 反向传播与优化
            loss.backward()
            optimizer.step()

            total_loss_epoch += loss.item()
            total_recon_epoch += recon_loss.item()
            total_kl_epoch += kl_loss.item()

        # 计算平均损失
        num_samples = len(train_loader.dataset)
        avg_loss = total_loss_epoch / num_samples
        avg_recon = total_recon_epoch / num_samples
        avg_kl = total_kl_epoch / num_samples

        print(f"Epoch [{epoch:02d}/{config.EPOCHS}] - Total Loss: {avg_loss:.4f} | Recon Loss: {avg_recon:.4f} | KL Loss: {avg_kl:.4f}")

        # 周期性采样保存
        if epoch % 5 == 0 or epoch == 1:
            save_samples_and_reconstructions(model, config, train_loader, epoch)

    # 保存训练终点权重
    model_save_path = os.path.join(config.RESULTS_DIR, "vae_mnist_final.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"\n训练结束！模型参数成功保存至: {model_save_path}")


def save_samples_and_reconstructions(model: nn.Module, config: Config, train_loader: DataLoader, epoch: int):
    """导出从标准高斯生成的样本图像以及对比原图的重建图像"""
    model.eval()
    with torch.no_grad():
        # 1. 从标准正态分布采样 z ~ N(0, I) 图像生成
        sample_z = torch.randn(64, config.LATENT_DIM).to(config.DEVICE)  # Shape: [64, LATENT_DIM]
        generated_samples = model.decoder(sample_z).cpu()                # Shape: [64, 784]
        
        save_image(
            generated_samples.view(64, 1, 28, 28),
            f"{config.RESULTS_DIR}/generated_epoch_{epoch:02d}.png",
            nrow=8,
            normalize=True
        )

        # 2. 真实图像与重建图像对比
        test_real_data, _ = next(iter(train_loader))
        test_real_data = test_real_data[:32].to(config.DEVICE)          # Shape: [32, 1, 28, 28]
        recon_data, _, _ = model(test_real_data)                         # Shape: [32, 784]
        
        # 上下拼接对比: 前 32 张为原图，后 32 张为重建图
        comparison = torch.cat([
            test_real_data,
            recon_data.view(-1, 1, 28, 28)
        ], dim=0)                                                        # Shape: [64, 1, 28, 28]
        
        save_image(
            comparison.cpu(),
            f"{config.RESULTS_DIR}/reconstruction_epoch_{epoch:02d}.png",
            nrow=8,
            normalize=True
        )


def visualize_latent_space(config: Config):
    """潜在空间二维散点图绘制 (分析隐空间结构分布)"""
    model = VAE(config).to(config.DEVICE)
    model_path = os.path.join(config.RESULTS_DIR, "vae_mnist_final.pth")
    
    if not os.path.exists(model_path):
        print("未找到权重文件，请先运行训练阶段。")
        return

    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()

    train_loader, _ = get_data_loaders(config)
    
    with torch.no_grad():
        data, labels = next(iter(train_loader))
        data = data.to(config.DEVICE)
        
        # 编码获取 \mu 均值作为隐特征
        mu, _ = model.encoder(data.view(data.size(0), -1))
        mu = mu.cpu().numpy()
        labels = labels.numpy()

    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(mu[:, 0], mu[:, 1], c=labels, cmap='tab10', alpha=0.7, edgecolors='none')
    plt.colorbar(scatter, label="Digit Label (0-9)")
    plt.title("VAE Latent Space Distribution (First 2 Dimensions)")
    plt.xlabel("Latent Dimension z[0]")
    plt.ylabel("Latent Dimension z[1]")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    save_fig_path = os.path.join(config.RESULTS_DIR, "latent_space_visualization.png")
    plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"隐空间二维散点图已成功生成至: {save_fig_path}")


if __name__ == "__main__":
    # 执行主配置与 Pipeline
    global_config = Config()
    train_pipeline(global_config)
    visualize_latent_space(global_config)