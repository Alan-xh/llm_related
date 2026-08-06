'''
生成器（Generator）：全连接网络，将随机噪声（latent vector）转换为图像
判别器（Discriminator）：全连接网络，判断图像是真实还是生成的
训练循环：交替训练生成器和判别器
损失函数：使用二元交叉熵损失（BCELoss）
优化器：使用Adam优化器
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# 设置随机种子以保证可复现性
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 超参数设置
class Config:
    def __init__(self):
        self.epochs = 200
        self.batch_size = 128
        self.latent_dim = 100
        self.lr = 0.0002
        self.beta1 = 0.5  # Adam优化器的beta1参数
        self.image_size = 64
        self.channels = 1  # 1表示灰度图，3表示彩色图
        self.save_interval = 10  # 每10个epoch保存一次生成图像
        self.sample_interval = 100  # 每100个batch打印一次损失
        
        # 数据集路径
        self.data_dir = './data'
        self.save_dir = './gan_results'
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'models'), exist_ok=True)

config = Config()

# 定义生成器
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# 初始化模型
def initialize_models():
    img_shape = (config.channels, config.image_size, config.image_size)
    
    # 初始化生成器和判别器
    generator = Generator(config.latent_dim, img_shape).to(device)
    discriminator = Discriminator(img_shape).to(device)
    
    # 初始化权重
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    
    return generator, discriminator

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

# 数据加载
def load_data():
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 归一化到[-1, 1]
    ])
    
    # 使用MNIST数据集作为示例
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
        num_workers=2
    )
    
    return dataloader

# 训练函数
def train_gan(generator, discriminator, dataloader):
    # 损失函数
    adversarial_loss = nn.BCELoss()
    
    # 优化器
    optimizer_G = optim.Adam(generator.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    
    # 记录损失
    g_losses = []
    d_losses = []
    
    print("Starting training...")
    
    for epoch in range(config.epochs):
        for i, (imgs, _) in enumerate(tqdm(dataloader, desc=f'Epoch {epoch+1}/{config.epochs}')):
            # 准备数据
            valid = torch.ones(imgs.size(0), 1).to(device)
            fake = torch.zeros(imgs.size(0), 1).to(device)
            
            real_imgs = imgs.to(device)
            
            # ---------------------
            # 训练判别器
            # ---------------------
            optimizer_D.zero_grad()
            
            # 真实图像的损失
            validity_real = discriminator(real_imgs)
            d_real_loss = adversarial_loss(validity_real, valid)
            
            # 生成假图像
            z = torch.randn(imgs.size(0), config.latent_dim).to(device)
            gen_imgs = generator(z)
            
            # 假图像的损失
            validity_fake = discriminator(gen_imgs.detach())
            d_fake_loss = adversarial_loss(validity_fake, fake)
            
            # 总判别器损失
            d_loss = (d_real_loss + d_fake_loss) / 2
            
            d_loss.backward()
            optimizer_D.step()
            
            # ---------------------
            # 训练生成器
            # ---------------------
            optimizer_G.zero_grad()
            
            # 生成假图像并计算损失
            z = torch.randn(imgs.size(0), config.latent_dim).to(device)
            gen_imgs = generator(z)
            validity = discriminator(gen_imgs)
            g_loss = adversarial_loss(validity, valid)
            
            g_loss.backward()
            optimizer_G.step()
            
            # 记录损失
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            
            # 打印进度
            if i % config.sample_interval == 0:
                print(f'[Epoch {epoch+1}/{config.epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]')
        
        # 每个epoch结束后保存样本图像
        if epoch % config.save_interval == 0:
            save_sample_images(generator, epoch)
            save_model(generator, discriminator, epoch)
    
    return g_losses, d_losses

# 保存样本图像
def save_sample_images(generator, epoch):
    generator.eval()
    with torch.no_grad():
        # 生成样本图像
        z = torch.randn(25, config.latent_dim).to(device)
        gen_imgs = generator(z).cpu()
        gen_imgs = (gen_imgs + 1) / 2  # 从[-1, 1]转换到[0, 1]
        
        # 创建网格
        grid = torchvision.utils.make_grid(gen_imgs, nrow=5, normalize=False)
        
        # 保存图像
        plt.figure(figsize=(10, 10))
        plt.imshow(np.transpose(grid, (1, 2, 0)), cmap='gray' if config.channels == 1 else None)
        plt.axis('off')
        plt.title(f'Epoch {epoch}')
        plt.savefig(os.path.join(config.save_dir, 'images', f'epoch_{epoch:03d}.png'))
        plt.close()
    generator.train()

# 保存模型
def save_model(generator, discriminator, epoch):
    torch.save(generator.state_dict(), os.path.join(config.save_dir, 'models', f'generator_epoch_{epoch:03d}.pth'))
    torch.save(discriminator.state_dict(), os.path.join(config.save_dir, 'models', f'discriminator_epoch_{epoch:03d}.pth'))

# 绘制损失曲线
def plot_losses(g_losses, d_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss', alpha=0.7)
    plt.plot(d_losses, label='Discriminator Loss', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Losses')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(config.save_dir, 'losses.png'))
    plt.show()

# 主函数
def main():
    print("Initializing models...")
    generator, discriminator = initialize_models()
    
    print("Loading data...")
    dataloader = load_data()
    
    print("Training GAN...")
    g_losses, d_losses = train_gan(generator, discriminator, dataloader)
    
    print("Plotting losses...")
    plot_losses(g_losses, d_losses)
    
    print(f"Training complete! Results saved to {config.save_dir}")

if __name__ == "__main__":
    main()