import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal

# ==================== 配置参数 ====================
class Config:
    # 数据参数
    data_path = "./data/train_data.npy"
    cond_data_path = "./data/conditions.npy"
    seq_length = 50
    input_dim = 10
    cond_dim = 5
    
    # CVAE架构参数
    latent_dim = 32  # 潜在空间维度
    hidden_dim = 256
    num_layers = 2
    dropout = 0.1
    
    # 训练参数
    batch_size = 64
    lr = 1e-3
    num_epochs = 200
    beta = 1.0  # KL散度权重（beta-VAE）
    kl_annealing = True  # KL退火
    kl_annealing_steps = 50  # 退火步数
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 保存和日志
    save_dir = "./checkpoints_cvae"
    log_dir = "./logs_cvae"
    save_interval = 10

config = Config()

# ==================== 数据集 ====================
class ConditionalDataset(Dataset):
    def __init__(self, data_path, cond_path, seq_length):
        self.data = np.load(data_path)
        self.conditions = np.load(cond_path)
        self.seq_length = seq_length
        
        assert len(self.data) == len(self.conditions), "Data and conditions length mismatch"
        
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        sequence = self.data[idx:idx + self.seq_length]
        condition = self.conditions[idx:idx + self.seq_length]
        return torch.FloatTensor(sequence), torch.FloatTensor(condition)

# ==================== CVAE模型 ====================
class Encoder(nn.Module):
    """编码器：将输入序列和条件映射到潜在空间"""
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        
        # 输入维度：input_dim + cond_dim
        input_size = config.input_dim + config.cond_dim
        
        # 使用LSTM处理序列
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )
        
        # 将双向LSTM的输出映射到潜在空间的均值和方差
        self.fc_mu = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.latent_dim)
        )
        
        self.fc_logvar = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.latent_dim)
        )
        
    def forward(self, x, cond):
        """
        x: [batch, seq_len, input_dim]
        cond: [batch, seq_len, cond_dim]
        """
        # 拼接输入和条件
        combined = torch.cat([x, cond], dim=-1)  # [batch, seq_len, input_dim + cond_dim]
        
        # LSTM编码
        lstm_out, (h_n, c_n) = self.lstm(combined)
        
        # 使用最后一个时间步的隐藏状态
        # 双向LSTM需要拼接最后的前向和后向输出
        last_hidden = lstm_out[:, -1, :]  # [batch, hidden_dim * 2]
        
        # 计算均值和方差
        mu = self.fc_mu(last_hidden)
        logvar = self.fc_logvar(last_hidden)
        
        return mu, logvar

class Decoder(nn.Module):
    """解码器：从潜在变量和条件重建序列"""
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        
        # 解码器初始状态
        self.fc_init = nn.Sequential(
            nn.Linear(config.latent_dim + config.cond_dim, config.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # LSTM解码器
        self.lstm = nn.LSTM(
            input_size=config.cond_dim,  # 每个时间步输入条件
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )
        
        # 输出层
        self.fc_out = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.input_dim)
        )
        
    def forward(self, z, cond):
        """
        z: [batch, latent_dim]
        cond: [batch, seq_len, cond_dim]
        """
        batch_size, seq_len, _ = cond.shape
        
        # 初始化LSTM状态
        init_state = self.fc_init(torch.cat([z, cond[:, 0, :]], dim=-1))
        h0 = init_state[:, :self.config.hidden_dim].unsqueeze(0).repeat(self.config.num_layers, 1, 1)
        c0 = init_state[:, self.config.hidden_dim:].unsqueeze(0).repeat(self.config.num_layers, 1, 1)
        
        # 解码
        lstm_out, _ = self.lstm(cond, (h0, c0))  # [batch, seq_len, hidden_dim]
        
        # 输出
        out = self.fc_out(lstm_out)  # [batch, seq_len, input_dim]
        
        return out

class CVAE(nn.Module):
    """条件变分自编码器"""
    def __init__(self, config):
        super(CVAE, self).__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, cond):
        """
        x: [batch, seq_len, input_dim]
        cond: [batch, seq_len, cond_dim]
        """
        # 编码
        mu, logvar = self.encoder(x, cond)
        
        # 采样
        z = self.reparameterize(mu, logvar)
        
        # 解码
        recon_x = self.decoder(z, cond)
        
        return recon_x, mu, logvar, z
    
    def generate(self, cond, num_samples=1):
        """从条件生成样本"""
        self.eval()
        with torch.no_grad():
            batch_size = cond.shape[0] * num_samples
            # 从标准正态分布采样
            z = torch.randn(batch_size, self.config.latent_dim).to(cond.device)
            
            # 扩展条件
            cond_expanded = cond.repeat_interleave(num_samples, dim=0)
            
            # 生成
            generated = self.decoder(z, cond_expanded)
            
        return generated
    
    def interpolate(self, cond1, cond2, num_steps=10):
        """在两个条件之间插值生成样本"""
        self.eval()
        with torch.no_grad():
            # 生成潜在向量
            z1 = torch.randn(1, self.config.latent_dim).to(cond1.device)
            z2 = torch.randn(1, self.config.latent_dim).to(cond2.device)
            
            # 插值
            alphas = torch.linspace(0, 1, num_steps).to(cond1.device)
            zs = []
            conds = []
            for alpha in alphas:
                z = (1 - alpha) * z1 + alpha * z2
                cond = (1 - alpha) * cond1 + alpha * cond2
                zs.append(z)
                conds.append(cond)
            
            zs = torch.cat(zs, dim=0)
            conds = torch.cat(conds, dim=0)
            
            # 生成
            generated = self.decoder(zs, conds)
            
        return generated

# ==================== 损失函数 ====================
def loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    计算CVAE损失
    recon_x: 重建序列
    x: 原始序列
    mu: 均值
    logvar: 对数方差
    beta: KL散度权重
    """
    # 重建损失 (MSE)
    recon_loss = nn.MSELoss(reduction='sum')(recon_x, x) / x.shape[0]
    
    # KL散度
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
    
    # 总损失
    loss = recon_loss + beta * kl_loss
    
    return loss, recon_loss, kl_loss

# ==================== 训练函数 ====================
def train():
    # 创建目录
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
    dataset = ConditionalDataset(config.data_path, config.cond_data_path, config.seq_length)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset size: {len(dataset)}")
    
    # 模型初始化
    model = CVAE(config).to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.num_epochs)
    
    # KL退火
    def get_beta(epoch):
        if not config.kl_annealing:
            return config.beta
        if epoch < config.kl_annealing_steps:
            return config.beta * (epoch + 1) / config.kl_annealing_steps
        return config.beta
    
    # 训练循环
    logger.info("Starting CVAE training...")
    best_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch_idx, (x, cond) in enumerate(progress_bar):
            x = x.to(config.device)
            cond = cond.to(config.device)
            
            # 前向传播
            recon_x, mu, logvar, _ = model(x, cond)
            
            # 计算损失
            beta = get_beta(epoch)
            loss, recon_loss, kl_loss = loss_function(recon_x, x, mu, logvar, beta)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': loss.item(),
                'recon': recon_loss.item(),
                'kl': kl_loss.item(),
                'beta': beta
            })
            
            # TensorBoard日志
            if batch_idx % 100 == 0:
                global_step = epoch * len(dataloader) + batch_idx
                writer.add_scalar('Train/Loss', loss.item(), global_step)
                writer.add_scalar('Train/Recon_Loss', recon_loss.item(), global_step)
                writer.add_scalar('Train/KL_Loss', kl_loss.item(), global_step)
                writer.add_scalar('Train/Beta', beta, global_step)
                writer.add_scalar('Train/LR', scheduler.get_last_lr()[0], global_step)
        
        # Epoch结束
        avg_loss = epoch_loss / len(dataloader)
        avg_recon_loss = epoch_recon_loss / len(dataloader)
        avg_kl_loss = epoch_kl_loss / len(dataloader)
        
        scheduler.step()
        
        logger.info(f"Epoch {epoch+1}/{config.num_epochs} - "
                   f"Loss: {avg_loss:.6f}, Recon: {avg_recon_loss:.6f}, KL: {avg_kl_loss:.6f}")
        
        writer.add_scalar('Epoch/Average_Loss', avg_loss, epoch)
        writer.add_scalar('Epoch/Average_Recon_Loss', avg_recon_loss, epoch)
        writer.add_scalar('Epoch/Average_KL_Loss', avg_kl_loss, epoch)
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
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
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'config': config
            }, os.path.join(config.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            logger.info(f"Checkpoint saved at epoch {epoch+1}")
            
            # 可视化
            if (epoch + 1) % 20 == 0:
                visualize_samples(model, dataloader, epoch + 1, logger)
    
    # 保存最终模型
    torch.save({
        'epoch': config.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_loss,
        'config': config
    }, os.path.join(config.save_dir, 'final_model.pth'))
    
    logger.info("CVAE training completed!")
    writer.close()

# ==================== 可视化函数 ====================
def visualize_samples(model, dataloader, epoch, logger):
    """可视化生成样本"""
    model.eval()
    with torch.no_grad():
        # 获取一批数据
        x, cond = next(iter(dataloader))
        x = x[:4].to(config.device)
        cond = cond[:4].to(config.device)
        
        # 重建
        recon_x, _, _, _ = model(x, cond)
        
        # 生成新样本
        generated = model.generate(cond, num_samples=1)
        
        # 绘制
        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        
        for i in range(4):
            # 原始序列
            for dim in range(min(3, config.input_dim)):
                axes[i, 0].plot(x[i, :, dim].cpu().numpy(), label=f'Dim {dim}')
            axes[i, 0].set_title(f'Original {i+1}')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            
            # 重建序列
            for dim in range(min(3, config.input_dim)):
                axes[i, 1].plot(recon_x[i, :, dim].cpu().numpy(), label=f'Dim {dim}')
            axes[i, 1].set_title(f'Reconstructed {i+1}')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)
            
            # 生成序列
            for dim in range(min(3, config.input_dim)):
                axes[i, 2].plot(generated[i, :, dim].cpu().numpy(), label=f'Dim {dim}')
            axes[i, 2].set_title(f'Generated {i+1}')
            axes[i, 2].legend()
            axes[i, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(config.log_dir, f'samples_epoch_{epoch}.png')
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Samples saved to {save_path}")

# ==================== 评估函数 ====================
def evaluate_model():
    """评估训练好的模型"""
    checkpoint_path = os.path.join(config.save_dir, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        print("No trained model found.")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model = CVAE(config).to(config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger = logging.getLogger(__name__)
    logger.info("Evaluating model...")
    
    # 加载数据
    dataset = ConditionalDataset(config.data_path, config.cond_data_path, config.seq_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    x, cond = next(iter(dataloader))
    x = x[:8].to(config.device)
    cond = cond[:8].to(config.device)
    
    with torch.no_grad():
        # 计算重建损失
        recon_x, mu, logvar, _ = model(x, cond)
        loss, recon_loss, kl_loss = loss_function(recon_x, x, mu, logvar, config.beta)
        
        logger.info(f"Reconstruction Loss: {recon_loss.item():.6f}")
        logger.info(f"KL Loss: {kl_loss.item():.6f}")
        logger.info(f"Total Loss: {loss.item():.6f}")
        
        # 潜在空间插值
        cond1 = cond[0:1]
        cond2 = cond[1:2]
        interpolated = model.interpolate(cond1, cond2, num_steps=10)
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i in range(10):
            row = i // 5
            col = i % 5
            for dim in range(min(3, config.input_dim)):
                axes[row, col].plot(interpolated[i, :, dim].cpu().numpy(), label=f'Dim {dim}')
            axes[row, col].set_title(f'Step {i+1}')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(config.log_dir, 'interpolation.png')
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Interpolation visualization saved to {save_path}")

# ==================== 主程序 ====================
if __name__ == "__main__":
    train()
    evaluate_model()