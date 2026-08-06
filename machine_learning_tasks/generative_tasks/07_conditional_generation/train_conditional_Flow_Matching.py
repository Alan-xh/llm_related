''' 条件流匹配模型 '''

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

# ==================== 条件 Flow Matching 核心组件 ====================

class ConditionalFlowMatching:
    """条件 Flow Matching 训练框架"""
    def __init__(self, sigma=0.1):
        self.sigma = sigma
    
    def sample_times(self, batch_size, device):
        return torch.rand(batch_size, 1, device=device)
    
    def get_conditional_probability(self, x1, x0, t, y=None):
        """
        条件概率路径 p_t(x | x1, y)
        x_t = (1 - t) * x0 + t * x1 + sigma * sqrt(t*(1-t)) * epsilon
        """
        t = t.unsqueeze(-1)
        x_t = (1 - t) * x0 + t * x1 + self.sigma * torch.sqrt(t * (1 - t) + 1e-8) * torch.randn_like(x1)
        return x_t
    
    def get_conditional_vector_field(self, x1, x0, t):
        return x1 - x0
    
    def compute_loss(self, model, x1, y, device):
        """
        计算条件 Flow Matching 损失
        Loss = E_{t, x0, epsilon} || v_t(x_t, y) - (x1 - x0) ||^2
        """
        batch_size = x1.shape[0]
        
        t = self.sample_times(batch_size, device)
        x0 = torch.randn_like(x1)
        x_t = self.get_conditional_probability(x1, x0, t)
        target_vf = self.get_conditional_vector_field(x1, x0, t)
        
        t_expanded = t.unsqueeze(-1) if len(x1.shape) > 2 else t
        predicted_vf = model(x_t, t_expanded, y)
        
        loss = torch.mean((predicted_vf - target_vf) ** 2)
        
        return loss, x_t, predicted_vf

# ==================== 条件神经网络模型 ====================

class ConditionalMLP(nn.Module):
    """条件 MLP 模型"""
    def __init__(self, dim=784, num_classes=10, hidden_dims=[512, 512, 512], 
                 time_dim=128, class_embed_dim=64):
        super().__init__()
        
        # 类别嵌入
        self.class_embedding = nn.Embedding(num_classes, class_embed_dim)
        
        # 时间步编码
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )
        
        # 主网络
        layers = []
        prev_dim = dim + time_dim + class_embed_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.SiLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x, t, y):
        # t: (batch_size, 1)
        # y: (batch_size,) 类别标签
        t_embed = self.time_mlp(t)
        y_embed = self.class_embedding(y)
        h = torch.cat([x, t_embed, y_embed], dim=-1)
        return self.net(h)

class ConditionalConvFlowMatcher(nn.Module):
    """条件卷积模型用于图像数据"""
    def __init__(self, in_channels=1, image_size=28, num_classes=10, 
                 time_dim=128, class_embed_dim=64):
        super().__init__()
        
        self.image_size = image_size
        
        # 类别嵌入
        self.class_embedding = nn.Embedding(num_classes, class_embed_dim)
        
        # 时间步编码
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
        
        # 条件嵌入投影
        self.cond_proj = nn.Linear(time_dim + class_embed_dim, 256)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, in_channels, 3, padding=1),
        )
    
    def forward(self, x, t, y):
        # x: (batch, in_channels, H, W)
        # t: (batch, 1)
        # y: (batch,) 类别标签
        
        # 时间嵌入
        t_embed = self.time_mlp(t)  # (batch, time_dim)
        y_embed = self.class_embedding(y)  # (batch, class_embed_dim)
        
        # 组合条件
        cond = torch.cat([t_embed, y_embed], dim=-1)
        cond = self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1)  # (batch, 256, 1, 1)
        
        # 编码
        h = self.encoder(x)
        
        # 加入条件信息
        h = h + cond
        
        # 解码
        out = self.decoder(h)
        
        return out

# ==================== 训练函数 ====================

def train_conditional_flow_matching(model, dataloader, epochs, lr=1e-3, 
                                     device='cuda', save_dir='checkpoints', 
                                     sample_dir='samples'):
    """训练条件 Flow Matching 模型"""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    flow_matcher = ConditionalFlowMatching(sigma=0.1)
    
    global_step = 0
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, (data, labels) in enumerate(progress_bar):
            data = data.to(device)
            labels = labels.to(device)
            
            if len(data.shape) == 4:  # 图像数据
                data = data.view(data.shape[0], -1)
            
            model.train()
            optimizer.zero_grad()
            
            loss, x_t, predicted_vf = flow_matcher.compute_loss(model, data, labels, device)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}')
        
        # 保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'{save_dir}/conditional_checkpoint_epoch_{epoch+1}.pt')
            
            # 为每个类别生成样本
            for class_id in range(10):
                sample_conditional_flow_matching(
                    model, device, 
                    save_path=f'{sample_dir}/samples_class_{class_id}_epoch_{epoch+1}.png',
                    num_samples=64, y=class_id
                )

def sample_conditional_flow_matching(model, device, save_path=None, 
                                     num_samples=64, steps=100, y=0,
                                     image_shape=(1, 28, 28)):
    """
    使用条件 Flow Matching 生成特定类别的样本
    """
    model.eval()
    with torch.no_grad():
        # 从标准正态分布采样初始点
        if len(image_shape) == 1:
            x = torch.randn(num_samples, image_shape[0], device=device)
        else:
            x = torch.randn(num_samples, *image_shape, device=device)
            x = x.view(num_samples, -1)
        
        # 条件标签
        y_tensor = torch.full((num_samples,), y, device=device, dtype=torch.long)
        
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.ones(num_samples, 1, device=device) * (i / steps)
            t = t.unsqueeze(-1) if len(image_shape) > 1 else t
            
            # 预测向量场
            v = model(x, t, y_tensor)
            
            # 欧拉更新
            x = x + v * dt
        
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
    
    # 创建条件模型
    model = ConditionalMLP(
        dim=28*28, 
        num_classes=10, 
        hidden_dims=[512, 512, 512]
    ).to(args.device)
    # 或者使用卷积模型：
    # model = ConditionalConvFlowMatcher(in_channels=1, image_size=28, num_classes=10).to(args.device)
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # 训练
    train_conditional_flow_matching(
        model=model,
        dataloader=dataloader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir
    )

if __name__ == '__main__':
    main()