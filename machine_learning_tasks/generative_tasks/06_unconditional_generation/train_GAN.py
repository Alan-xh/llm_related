"""
任务定义:
    - 任务编号: CV-GAN-001
    - 任务名称: Vanilla Generative Adversarial Network (标准生成对抗网络) 图像生成
    - 领域分类: 生成式模型 (Generative Modeling / Computer Vision)

代表架构/算法:
    - 架构名称: Deep Learning GAN (Vanilla GAN)
    - 论文来源: Goodfellow et al., "Generative Adversarial Nets", NIPS 2014

核心思想与机制:
    - 采用极小化极大博弈 (Minimax Game) 机制。
    - 生成器 G (Generator): 接收随机高斯噪声 z ~ p_z(z)，学习从隐空间到真实图像数据分布 p_data 的映射 G(z)。
    - 判别器 D (Discriminator): 接收图像 x，输出标量 D(x) ∈ [0, 1]，表示图像来自于真实数据的概率。
    - 对抗训练 (Adversarial Training): D 努力区分真假图像，G 努力欺骗 D，两者在博弈中达成纳什均衡 (Nash Equilibrium)。

数学公式/目标函数:
    - 极小化极大目标函数 (Minimax Objective):
        min_G max_D V(D, G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]
    - 判别器损失 (Discriminator Loss):
        L_D = - E_{x~p_data}[log D(x)] - E_{z~p_z}[log(1 - D(G(z)))]
        在 Binary Cross Entropy (BCE) 代码实现中:
        L_D = 1/2 * (BCE(D(x), 1) + BCE(D(G(z)), 0))
    - 生成器损失 (Generator Loss - Non-saturating formulation):
        L_G = - E_{z~p_z}[log D(G(z))]
        在 BCE 代码实现中:
        L_G = BCE(D(G(z)), 1)

数据输入规范:
    - 输入噪声 z: FloatTensor, shape [B, Latent_Dim] ~ N(0, I)
    - 真实图像 x: FloatTensor, shape [B, C, H, W], 数值范围 [-1, 1]
    - 输出图像 G(z): FloatTensor, shape [B, C, H, W], 数值范围 [-1, 1]
    - 判别标量 D(x): FloatTensor, shape [B, 1], 数值范围 [0, 1]
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# --------------------------------------------------------------------------------------------------
# 2. 依赖与全局配置 (Config & Reproducibility)
# --------------------------------------------------------------------------------------------------

# 设置随机种子以保证可复现性
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Device] Using device: {device}")


class Config:
    """
    全局配置类，包含模型结构参数、训练超参数以及保存路径等。
    """
    def __init__(self):
        # 训练超参数
        self.epochs = 200            # 总训练轮数
        self.batch_size = 128        # 批次大小 (B)
        self.latent_dim = 100        # 随机隐变量 z 的维度 (Latent_Dim)
        self.lr = 0.0002             # 初始学习率 (Adam)
        self.beta1 = 0.5             # Adam 优化器的一阶矩估计指数衰减率
        self.beta2 = 0.999           # Adam 优化器的二阶矩估计指数衰减率
        
        # 图像特征参数
        self.image_size = 64         # 图像高度与宽度 (H = W = 64)
        self.channels = 1            # 通道数 (C = 1 表示灰度图，C = 3 表示彩色图)
        
        # 监控与保存设置
        self.save_interval = 10      # 每多少个 epoch 保存一次模型与采样图像
        self.sample_interval = 100   # 每多少个 batch 打印一次训练 Loss
        
        # 文件系统路径
        self.data_dir = './data'
        self.save_dir = './gan_results'
        
        # 创建目录结构
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'models'), exist_ok=True)


config = Config()

# --------------------------------------------------------------------------------------------------
# 3. 数据处理管道 (Data Pipeline)
# --------------------------------------------------------------------------------------------------

def load_data():
    """
    加载并预处理 MNIST 数据集。

    Returns:
        dataloader (DataLoader): PyTorch 数据加载器。
    """
    # 图像预处理流水线
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)), # [1, 28, 28] -> [1, 64, 64]
        transforms.ToTensor(),                                    # 转换为 Tensor，范围 [0.0, 1.0]
        transforms.Normalize([0.5], [0.5])                         # 归一化到 [-1.0, 1.0]，匹配 Generator 的 Tanh 输出
    ])
    
    dataset = torchvision.datasets.MNIST(
        root=config.data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=2,
        drop_last=True
    )
    
    return dataloader

# --------------------------------------------------------------------------------------------------
# 4. 核心子模块与权重初始化 (Sub-components & Initialization)
# --------------------------------------------------------------------------------------------------

def weights_init_normal(m):
    """
    自定义权重初始化函数，使用正态分布初始化 Linear 与 BatchNorm 层。
    
    Args:
        m (nn.Module): 当前神经网络模块。
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # 权重遵循 N(0, 0.02)
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        # 权重 (Gamma) 遵循 N(1.0, 0.02)，偏置 (Beta) 置 0
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


# --------------------------------------------------------------------------------------------------
# 5. 顶层模型架构 (Top-level Architecture)
# --------------------------------------------------------------------------------------------------

class Generator(nn.Module):
    """
    生成器 (Generator) 模块：全连接 MLP 网络。
    将从正态分布采样的随机噪声 z 映射为高维图像张量。

    数学原理:
        G: R^{Latent_Dim} -> R^{C x H x W}
        img = Tanh(Linear(...(LeakyReLU(BatchNorm(Linear(z))))))

    Args:
        latent_dim (int): 隐变量 z 的特征维度。
        img_shape (tuple): 生成目标图像的形状 (C, H, W)。

    Inputs:
        z (Tensor): 隐空间随机噪声张量，shape: [B, Latent_Dim]

    Outputs:
        img (Tensor): 生成的假图像张量，shape: [B, C, H, W]
    """
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        
        def block(in_feat, out_feat, normalize=True):
            """全连接线性块: Linear -> [BatchNorm] -> LeakyReLU"""
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            # Stage 1: [B, Latent_Dim] -> [B, 128]
            *block(latent_dim, 128, normalize=False),
            # Stage 2: [B, 128] -> [B, 256]
            *block(128, 256),
            # Stage 3: [B, 256] -> [B, 512]
            *block(256, 512),
            # Stage 4: [B, 512] -> [B, 1024]
            *block(512, 1024),
            # Output Stage: [B, 1024] -> [B, C * H * W]
            nn.Linear(1024, int(np.prod(img_shape))),
            # 激活函数限制输出至 [-1, 1] 范围
            nn.Tanh()
        )
    
    def forward(self, z):
        # z: [B, Latent_Dim]
        img_flat = self.model(z)                          # Shape: [B, C * H * W]
        img = img_flat.view(img_flat.size(0), *self.img_shape) # Shape: [B, C, H, W]
        return img


class Discriminator(nn.Module):
    """
    判别器 (Discriminator) 模块：全连接 MLP 分类器。
    判断输入的图像是真实数据 (Real) 还是由生成器伪造的数据 (Fake)。

    数学原理:
        D: R^{C x H x W} -> [0, 1]
        validity = Sigmoid(Linear(...(LeakyReLU(Linear(x_flat)))))

    Args:
        img_shape (tuple): 输入图像的形状 (C, H, W)。

    Inputs:
        img (Tensor): 输入图像张量 (真或假)，shape: [B, C, H, W]

    Outputs:
        validity (Tensor): 图像为真实数据的概率预测，shape: [B, 1]
    """
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        
        self.model = nn.Sequential(
            # Stage 1: [B, C * H * W] -> [B, 512]
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            # Stage 2: [B, 512] -> [B, 256]
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            # Output Stage: [B, 256] -> [B, 1]
            nn.Linear(256, 1),
            # 输出概率 [0, 1]
            nn.Sigmoid()
        )
    
    def forward(self, img):
        # img: [B, C, H, W]
        img_flat = img.view(img.size(0), -1)              # Shape: [B, C * H * W]
        validity = self.model(img_flat)                   # Shape: [B, 1]
        return validity


# --------------------------------------------------------------------------------------------------
# 6. 模型初始化与入口 (Model Initialization)
# --------------------------------------------------------------------------------------------------

def initialize_models():
    """
    实例化 Generator 和 Discriminator 并挂载至对应计算设备，绑定权重初始化。

    Returns:
        generator (nn.Module): 已初始化的生成器
        discriminator (nn.Module): 已初始化的判别器
    """
    img_shape = (config.channels, config.image_size, config.image_size)
    
    # 实例化网络
    generator = Generator(config.latent_dim, img_shape).to(device)
    discriminator = Discriminator(img_shape).to(device)
    
    # 应用权重初始化策略
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    
    return generator, discriminator

# --------------------------------------------------------------------------------------------------
# 7. 训练/推理逻辑 (Training Execution & Helper Functions)
# --------------------------------------------------------------------------------------------------

def save_sample_images(generator, epoch):
    """
    在推理模式下生成 25 张图片并保存拼接网格图。
    
    Args:
        generator (nn.Module): 生成器模型
        epoch (int): 当前 Epoch 序号
    """
    generator.eval()
    with torch.no_grad():
        # 1. 从标准正态分布采样固定噪声
        z = torch.randn(25, config.latent_dim).to(device) # Shape: [25, Latent_Dim]
        
        # 2. 前向推导生成伪造图像
        gen_imgs = generator(z).cpu()                      # Shape: [25, C, H, W]
        
        # 3. 反归一化：从 [-1, 1] 映射到 [0, 1] 以便于显示
        gen_imgs = (gen_imgs + 1.0) / 2.0                  
        
        # 4. 拼贴成 5x5 的图片网格
        grid = torchvision.utils.make_grid(gen_imgs, nrow=5, normalize=False) # Shape: [C, Grid_H, Grid_W]
        
        # 5. 保存绘制结果
        plt.figure(figsize=(8, 8))
        plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)), cmap='gray' if config.channels == 1 else None)
        plt.axis('off')
        plt.title(f'Generated Images - Epoch {epoch}')
        plt.savefig(os.path.join(config.save_dir, 'images', f'epoch_{epoch:03d}.png'))
        plt.close()
        
    generator.train()


def save_model(generator, discriminator, epoch):
    """保存 G 和 D 的 state_dict 到本地 checkpoint 文件"""
    torch.save(generator.state_dict(), os.path.join(config.save_dir, 'models', f'generator_epoch_{epoch:03d}.pth'))
    torch.save(discriminator.state_dict(), os.path.join(config.save_dir, 'models', f'discriminator_epoch_{epoch:03d}.pth'))


def plot_losses(g_losses, d_losses):
    """绘制并保存 G 和 D 的 Losses 变化趋势图"""
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss (L_G)', alpha=0.7)
    plt.plot(d_losses, label='Discriminator Loss (L_D)', alpha=0.7)
    plt.xlabel('Iteration Step')
    plt.ylabel('Binary Cross Entropy Loss')
    plt.legend()
    plt.title('GAN Training Loss Curves')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(config.save_dir, 'losses.png'))
    plt.close()


def train_gan(generator, discriminator, dataloader):
    """
    GAN 对抗训练主循环。
    交替优化 Discriminator 和 Generator。
    """
    # 交叉熵损失函数: BCE = - [y * log(y_hat) + (1 - y) * log(1 - y_hat)]
    adversarial_loss = nn.BCELoss()
    
    # 优化器设置
    optimizer_G = optim.Adam(generator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    
    g_losses = []
    d_losses = []
    
    print("[Training] Starting Training Loop...")
    
    for epoch in range(config.epochs):
        for i, (imgs, _) in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}/{config.epochs}')):
            batch_size = imgs.size(0)
            
            # 构建对抗标签 (Ground Truth Labels)
            valid = torch.ones(batch_size, 1, device=device)   # 真图像标签 y = 1.0, Shape: [B, 1]
            fake = torch.zeros(batch_size, 1, device=device)   # 假图像标签 y = 0.0, Shape: [B, 1]
            
            real_imgs = imgs.to(device)                         # 真实图片 x, Shape: [B, C, H, W]
            
            # --------------------------------------------------------------------------------------
            # (1) 训练判别器 D: max log(D(x)) + log(1 - D(G(z)))
            # --------------------------------------------------------------------------------------
            optimizer_D.zero_grad()
            
            # 计算真实图像的判别损失 L_D_real = BCE(D(x), 1)
            validity_real = discriminator(real_imgs)           # Shape: [B, 1]
            d_real_loss = adversarial_loss(validity_real, valid)
            
            # 采样随机隐变量噪声 z ~ N(0, I)
            z = torch.randn(batch_size, config.latent_dim, device=device) # Shape: [B, Latent_Dim]
            gen_imgs = generator(z)                              # Shape: [B, C, H, W]
            
            # 计算伪造图像的判别损失 L_D_fake = BCE(D(G(z).detach()), 0)
            # 使用 .detach() 截断梯度流，防止梯度回传更新 Generator 权重
            validity_fake = discriminator(gen_imgs.detach())    # Shape: [B, 1]
            d_fake_loss = adversarial_loss(validity_fake, fake)
            
            # 判别器总损失 (取平均以平衡尺度)
            d_loss = (d_real_loss + d_fake_loss) / 2.0
            
            d_loss.backward()
            optimizer_D.step()
            
            # --------------------------------------------------------------------------------------
            # (2) 训练生成器 G: max log(D(G(z))) <=> min BCE(D(G(z)), 1)
            # --------------------------------------------------------------------------------------
            optimizer_G.zero_grad()
            
            # 重新生成或利用已生成的假图片求梯度 (此处重新采样 z 保持计算独立性)
            z = torch.randn(batch_size, config.latent_dim, device=device) # Shape: [B, Latent_Dim]
            gen_imgs = generator(z)                              # Shape: [B, C, H, W]
            
            # 评估假图像 (不使用 detach，允许梯度流动至 Generator)
            validity = discriminator(gen_imgs)                  # Shape: [B, 1]
            
            # 计算生成器欺骗损失: 期望 D(G(z)) 被判定为 1 (Real)
            g_loss = adversarial_loss(validity, valid)
            
            g_loss.backward()
            optimizer_G.step()
            
            # 记录步骤 Losses
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            
            # 定期打印 Logs
            if i % config.sample_interval == 0:
                tqdm.write(f'[Epoch {epoch+1}/{config.epochs}] [Batch {i}/{len(dataloader)}] '
                           f'[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]')
        
        # 轮次结束评估与保存
        if epoch % config.save_interval == 0:
            save_sample_images(generator, epoch)
            save_model(generator, discriminator, epoch)
            
    return g_losses, d_losses


def main():
    """程序的主入口函数 (Pipeline Runner)"""
    generator, discriminator = initialize_models()
    
    dataloader = load_data()
    
    g_losses, d_losses = train_gan(generator, discriminator, dataloader)
    
    plot_losses(g_losses, d_losses)
    


if __name__ == "__main__":
    main()