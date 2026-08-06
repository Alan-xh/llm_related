''' Flow Matching 流匹配模型 '''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import numpy as np
from tqdm import tqdm
import argparse

# ==================== Flow Matching 核心组件 ====================

class FlowMatching:
    """Flow Matching 训练框架"""
    def __init__(self, sigma=0.1):
        """
        Args:
            sigma: 条件概率路径的标准差
        """
        self.sigma = sigma
    
    def sample_times(self, batch_size, device):
        """采样时间步 t ~ Uniform(0, 1)"""
        return torch.rand(batch_size, 1, device=device)
    
    def get_conditional_probability(self, x1, x0, t):
        """
        计算条件概率路径 p_t(x | x1)
        x_t = (1 - t) * x0 + t * x1 + sigma * sqrt(t*(1-t)) * epsilon
        """
        # x0 通常从标准正态分布采样
        # x1 是目标数据
        # t: (batch_size, 1)
        t = t.unsqueeze(-1)  # 扩展维度以适应数据形状
        x_t = (1 - t) * x0 + t * x1 + self.sigma * torch.sqrt(t * (1 - t) + 1e-8) * torch.randn_like(x1)
        return x_t
    
    def get_conditional_vector_field(self, x1, x0, t):
        """
        条件向量场 u_t(x|x1) = x1 - x0
        实际上这里是 (x1 - x_t) / (1 - t) 的简化版本
        用于 CFM 损失计算
        """
        return x1 - x0
    
    def compute_loss(self, model, x1, device):
        """
        计算 Flow Matching 损失
        Loss = E_{t, x0, epsilon} || v_t(x_t) - (x1 - x0) ||^2
        """
        batch_size = x1.shape[0]
        
        # 采样时间步
        t = self.sample_times(batch_size, device)
        
        # 采样 x0 ~ N(0, I)
        x0 = torch.randn_like(x1)
        
        # 采样条件概率路径
        x_t = self.get_conditional_probability(x1, x0, t)
        
        # 计算真实向量场
        target_vf = self.get_conditional_vector_field(x1, x0, t)
        
        # 模型预测
        t_expanded = t.unsqueeze(-1) if len(x1.shape) > 2 else t
        predicted_vf = model(x_t, t_expanded)
        
        # MSE 损失
        loss = torch.mean((predicted_vf - target_vf) ** 2)
        
        return loss, x_t, predicted_vf

# ==================== 神经网络模型 ====================

class SimpleMLP(nn.Module):
    """简单的 MLP 模型用于 Flow Matching"""
    def __init__(self, dim=784, hidden_dims=[512, 512, 512], time_dim=128):
        super().__init__()
        
        # 时间步编码
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )
        
        # 主网络
        layers = []
        prev_dim = dim + time_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.SiLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x, t):
        # t: (batch_size, 1)
        t_embed = self.time_mlp(t)
        h = torch.cat([x, t_embed], dim=-1)
        return self.net(h)

class ConvFlowMatcher(nn.Module):
    """卷积模型用于图像数据"""
    def __init__(self, in_channels=1, image_size=28, time_dim=128):
        super().__init__()
        
        self.image_size = image_size
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels + 1, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.SiLU(),
        )
        
        # 处理时间嵌入
        self.time_proj = nn.Linear(time_dim, 256)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, in_channels, 3, padding=1),
        )
    
    def forward(self, x, t):
        # x: (batch, in_channels, H, W)
        # t: (batch, 1)
        
        # 时间嵌入
        t_embed = self.time_mlp(t)  # (batch, time_dim)
        t_embed = self.time_proj(t_embed).unsqueeze(-1).unsqueeze(-1)  # (batch, 256, 1, 1)
        
        # 编码
        h = self.encoder(x)
        
        # 加入时间信息
        h = h + t_embed
        
        # 解码
        out = self.decoder(h)
        
        return out

# ==================== 训练函数 ====================

def train_flow_matching(model, dataloader, epochs, lr=1e-3, device='cuda', 
                         save_dir='checkpoints', sample_dir='samples'):
    """
    训练 Flow Matching 模型
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    flow_matcher = FlowMatching(sigma=0.1)
    
    global_step = 0
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, (data, _) in enumerate(progress_bar):
            data = data.to(device)
            if len(data.shape) == 4:  # 图像数据
                data = data.view(data.shape[0], -1)
            
            model.train()
            optimizer.zero_grad()
            
            loss, x_t, predicted_vf = flow_matcher.compute_loss(model, data, device)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}')
        
        # 保存检查点和采样
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'{save_dir}/checkpoint_epoch_{epoch+1}.pt')
            
            # 生成样本
            sample_flow_matching(model, device, f'{sample_dir}/samples_epoch_{epoch+1}.png')

def sample_flow_matching(model, device, save_path=None, num_samples=64, 
                         steps=100, image_shape=(1, 28, 28)):
    """
    使用 Flow Matching 生成样本
    """
    model.eval()
    with torch.no_grad():
        # 从标准正态分布采样初始点
        if len(image_shape) == 1:
            x = torch.randn(num_samples, image_shape[0], device=device)
        else:
            x = torch.randn(num_samples, *image_shape, device=device)
            x = x.view(num_samples, -1)
        
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.ones(num_samples, 1, device=device) * (i / steps)
            t = t.unsqueeze(-1) if len(image_shape) > 1 else t
            
            # 预测向量场
            v = model(x, t)
            
            # 欧拉更新
            x = x + v * dt
            
            # 可选：添加一些噪声（模拟随机微分方程）
            # x = x + 0.01 * torch.randn_like(x) * np.sqrt(dt)
        
        if save_path:
            if len(image_shape) == 3:
                # 图像数据
                x = x.view(num_samples, *image_shape)
                x = (x + 1) / 2  # 从 [-1, 1] 到 [0, 1]
                x = torch.clamp(x, 0, 1)
                save_image(x, save_path, nrow=8)
            else:
                # 其他数据直接保存
                torch.save(x, save_path.replace('.png', '.pt'))
        return x

# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    args = parser.parse_args()
    
    # 加载数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
    ])
    
    if args.dataset == 'mnist':
        dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    else:
        dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # 创建模型
    model = SimpleMLP(dim=28*28, hidden_dims=[512, 512, 512]).to(args.device)
    # 或者使用卷积模型：
    # model = ConvFlowMatcher(in_channels=1, image_size=28).to(args.device)
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # 训练
    train_flow_matching(
        model=model,
        dataloader=dataloader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir
    )

if __name__ == '__main__':
    main()