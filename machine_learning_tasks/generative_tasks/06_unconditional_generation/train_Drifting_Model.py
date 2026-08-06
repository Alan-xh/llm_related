'''
基于扩散模型（Diffusion Model）架构，用于训练漂移模型（Drifting Model）。

假设这是一个用于时间序列预测或轨迹生成的扩散模型
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from torch.utils.tensorboard import SummaryWriter

# ==================== 配置参数 ====================
class Config:
    # 数据参数
    data_path = "./data/train_data.npy"  # 训练数据路径
    seq_length = 50  # 序列长度
    input_dim = 10   # 输入特征维度
    
    # 扩散模型参数
    T = 1000  # 扩散步数
    beta_start = 1e-4
    beta_end = 0.02
    time_emb_dim = 256
    
    # 模型架构
    model_dim = 128
    num_heads = 4
    num_layers = 4
    dropout = 0.1
    
    # 训练参数
    batch_size = 64
    lr = 1e-4
    num_epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 保存和日志
    save_dir = "./checkpoints"
    log_dir = "./logs"
    save_interval = 10

config = Config()

# ==================== 数据集 ====================
class DriftingDataset(Dataset):
    def __init__(self, data_path, seq_length):
        self.data = np.load(data_path)
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        # 返回连续的序列作为训练样本
        sequence = self.data[idx:idx + self.seq_length]
        return torch.FloatTensor(sequence)

# ==================== 扩散模型核心组件 ====================
class DriftingModel(nn.Module):
    """漂移模型 - 基于Transformer的扩散模型"""
    def __init__(self, config):
        super(DriftingModel, self).__init__()
        self.config = config
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, model_dim * 4),
            nn.SiLU(),
            nn.Linear(model_dim * 4, model_dim)
        )
        
        # 主网络 - 使用Transformer
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=model_dim * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.output_proj = nn.Linear(model_dim, input_dim)
        
    def forward(self, x, t):
        # x: [batch, seq_len, input_dim]
        # t: [batch] 时间步
        
        # 时间嵌入
        t_emb = self.get_timestep_embedding(t, time_emb_dim)
        t_emb = self.time_mlp(t_emb)  # [batch, model_dim]
        
        # 投影输入
        h = self.input_proj(x)  # [batch, seq_len, model_dim]
        
        # 添加时间嵌入到每个时间步
        h = h + t_emb.unsqueeze(1)
        
        # Transformer编码
        h = self.transformer(h)
        
        # 输出
        out = self.output_proj(h)
        return out
    
    def get_timestep_embedding(self, t, dim):
        """正弦位置编码用于时间步"""
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb

# ==================== 扩散过程 ====================
class DiffusionProcess:
    def __init__(self, config):
        self.config = config
        self.T = config.T
        
        # 预计算beta和alpha
        self.beta = torch.linspace(config.beta_start, config.beta_end, config.T)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)
        
    def q_sample(self, x0, t, noise=None):
        """前向扩散过程：加噪声"""
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].reshape(-1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].reshape(-1, 1, 1)
        
        xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
        return xt
    
    def p_sample(self, model, xt, t):
        """反向去噪过程：单步去噪"""
        # 预测噪声
        predicted_noise = model(xt, t)
        
        # 计算均值和方差
        beta_t = self.beta[t].reshape(-1, 1, 1)
        alpha_t = self.alpha[t].reshape(-1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1, 1)
        sqrt_recip_alpha_t = 1 / torch.sqrt(alpha_t)
        
        # 预测x0
        x0_pred = (xt - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        
        # 重新参数化
        if t[0] > 0:
            noise = torch.randn_like(xt)
            sigma_t = torch.sqrt(beta_t)
        else:
            noise = 0
            sigma_t = 0
            
        xt_prev = sqrt_recip_alpha_t * (xt - beta_t / torch.sqrt(1 - alpha_bar_t) * predicted_noise) + sigma_t * noise
        return xt_prev

# ==================== 训练函数 ====================
def train():
    # 创建保存目录
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    writer = SummaryWriter(config.log_dir)
    
    # 数据加载
    logger.info("Loading data...")
    dataset = DriftingDataset(config.data_path, config.seq_length)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    logger.info(f"Dataset size: {len(dataset)}")
    
    # 模型初始化
    model = DriftingModel(config).to(config.device)
    diffusion = DiffusionProcess(config)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.num_epochs)
    
    # 训练循环
    logger.info("Starting training...")
    best_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 将数据移动到设备
            batch = batch.to(config.device)  # [batch, seq_len, input_dim]
            
            # 采样时间步
            t = torch.randint(0, config.T, (batch.shape[0],), device=config.device)
            
            # 采样噪声
            noise = torch.randn_like(batch)
            
            # 前向扩散
            xt = diffusion.q_sample(batch, t, noise)
            
            # 预测噪声
            predicted_noise = model(xt, t)
            
            # 计算损失 (MSE)
            loss = nn.MSELoss()(predicted_noise, noise)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': loss.item(),
                'lr': scheduler.get_last_lr()[0]
            })
            
            # TensorBoard日志
            if batch_idx % 100 == 0:
                writer.add_scalar('Train/Loss', loss.item(), epoch * len(dataloader) + batch_idx)
                writer.add_scalar('Train/LR', scheduler.get_last_lr()[0], epoch * len(dataloader) + batch_idx)
        
        # Epoch结束
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step()
        
        logger.info(f"Epoch {epoch+1}/{config.num_epochs} - Average Loss: {avg_loss:.6f}")
        writer.add_scalar('Epoch/Average_Loss', avg_loss, epoch)
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }, os.path.join(config.save_dir, 'best_model.pth'))
            logger.info(f"Best model saved with loss: {best_loss:.6f}")
        
        # 定期保存检查点
        if (epoch + 1) % config.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }, os.path.join(config.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            logger.info(f"Checkpoint saved at epoch {epoch+1}")
    
    # 保存最终模型
    torch.save({
        'epoch': config.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'config': config
    }, os.path.join(config.save_dir, 'final_model.pth'))
    
    logger.info("Training completed!")
    writer.close()

# ==================== 可视化训练过程 ====================
def visualize_training():
    """可视化训练损失曲线"""
    log_file = os.path.join(config.log_dir, 'training.log')
    if not os.path.exists(log_file):
        print("No training log found.")
        return
    
    # 解析日志文件
    losses = []
    with open(log_file, 'r') as f:
        for line in f:
            if 'Average Loss:' in line:
                loss = float(line.split('Average Loss:')[-1].strip())
                losses.append(loss)
    
    if losses:
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('Training Loss Curve')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(config.log_dir, 'loss_curve.png'))
        plt.show()
        print(f"Loss curve saved to {config.log_dir}/loss_curve.png")

# ==================== 主程序 ====================
if __name__ == "__main__":
    train()
    visualize_training()