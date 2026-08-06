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
    data_path = "./data/train_data.npy"
    cond_data_path = "./data/conditions.npy"  # 条件数据路径
    seq_length = 50
    input_dim = 10
    cond_dim = 5  # 条件特征维度
    
    # 扩散模型参数
    T = 1000
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
    gradient_accumulation_steps = 1  # 梯度累积步数
    
    # 条件增强
    cond_drop_rate = 0.1  # 条件丢弃率（用于分类器无关引导）
    
    # 保存和日志
    save_dir = "./checkpoints_conditional"
    log_dir = "./logs_conditional"
    save_interval = 10

config = Config()

# ==================== 数据集（带条件） ====================
class ConditionalDriftingDataset(Dataset):
    def __init__(self, data_path, cond_path, seq_length):
        self.data = np.load(data_path)
        self.conditions = np.load(cond_path)
        self.seq_length = seq_length
        
        # 确保数据长度匹配
        assert len(self.data) == len(self.conditions), "Data and conditions length mismatch"
        
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        sequence = self.data[idx:idx + self.seq_length]
        condition = self.conditions[idx:idx + self.seq_length]  # 每个时间步都有条件
        # 或者条件可以是全局的：condition = self.conditions[idx]
        return torch.FloatTensor(sequence), torch.FloatTensor(condition)

# ==================== 条件扩散模型 ====================
class ConditionalDriftingModel(nn.Module):
    """条件漂移模型 - 基于Transformer的条件扩散模型"""
    def __init__(self, config):
        super(ConditionalDriftingModel, self).__init__()
        self.config = config
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_emb_dim, config.model_dim * 4),
            nn.SiLU(),
            nn.Linear(config.model_dim * 4, config.model_dim)
        )
        
        # 条件编码器
        self.cond_encoder = nn.Sequential(
            nn.Linear(config.cond_dim, config.model_dim // 2),
            nn.SiLU(),
            nn.Linear(config.model_dim // 2, config.model_dim)
        )
        
        # 主网络 - 使用Transformer
        self.input_proj = nn.Linear(config.input_dim, config.model_dim)
        
        # 条件注入方式：使用AdaLN (Adaptive Layer Norm)
        self.ada_ln = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim * 2),
            nn.SiLU(),
            nn.Linear(config.model_dim * 2, config.model_dim)
        )
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.model_dim,
                nhead=config.num_heads,
                dim_feedforward=config.model_dim * 4,
                dropout=config.dropout,
                batch_first=True,
                norm_first=True
            ),
            num_layers=config.num_layers
        )
        
        self.output_proj = nn.Linear(config.model_dim, config.input_dim)
        
        # 条件丢弃（用于训练时的无条件生成）
        self.cond_drop_rate = config.cond_drop_rate
        
    def forward(self, x, t, cond, cond_mask=None):
        """
        x: [batch, seq_len, input_dim]
        t: [batch] 时间步
        cond: [batch, seq_len, cond_dim] 条件
        cond_mask: [batch] 是否丢弃条件 (1: 使用条件, 0: 丢弃)
        """
        batch_size, seq_len, _ = x.shape
        
        # 条件丢弃
        if self.training and self.cond_drop_rate > 0:
            if cond_mask is None:
                cond_mask = torch.bernoulli(
                    torch.ones(batch_size, device=x.device) * (1 - self.cond_drop_rate)
                ).bool()
            cond_effective = cond * cond_mask.view(-1, 1, 1)
        else:
            cond_effective = cond
            cond_mask = torch.ones(batch_size, device=x.device).bool()
        
        # 时间嵌入
        t_emb = self.get_timestep_embedding(t, config.time_emb_dim)
        t_emb = self.time_mlp(t_emb)  # [batch, model_dim]
        
        # 条件嵌入
        cond_emb = self.cond_encoder(cond_effective)  # [batch, seq_len, model_dim]
        cond_emb_pooled = cond_emb.mean(dim=1)  # [batch, model_dim]
        
        # 投影输入
        h = self.input_proj(x)  # [batch, seq_len, model_dim]
        
        # 注入时间步和条件（使用AdaLN风格）
        # 这里我们简单地将时间和条件加到输入上
        h = h + t_emb.unsqueeze(1) + cond_emb
        
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

# ==================== 条件扩散过程 ====================
class ConditionalDiffusionProcess:
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
    
    def p_sample(self, model, xt, t, cond, cond_mask=None):
        """反向去噪过程：单步去噪"""
        # 预测噪声
        predicted_noise = model(xt, t, cond, cond_mask)
        
        # 计算均值和方差
        beta_t = self.beta[t].reshape(-1, 1, 1)
        alpha_t = self.alpha[t].reshape(-1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1, 1)
        sqrt_recip_alpha_t = 1 / torch.sqrt(alpha_t)
        
        if t[0] > 0:
            noise = torch.randn_like(xt)
            sigma_t = torch.sqrt(beta_t)
        else:
            noise = 0
            sigma_t = 0
            
        xt_prev = sqrt_recip_alpha_t * (xt - beta_t / torch.sqrt(1 - alpha_bar_t) * predicted_noise) + sigma_t * noise
        return xt_prev
    
    def ddpm_sample(self, model, cond, shape, cond_mask=None):
        """从噪声开始逐步去噪生成样本"""
        batch_size = shape[0]
        xt = torch.randn(shape, device=cond.device)
        
        for t in reversed(range(self.T)):
            t_tensor = torch.full((batch_size,), t, device=cond.device, dtype=torch.long)
            xt = self.p_sample(model, xt, t_tensor, cond, cond_mask)
        
        return xt

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
    dataset = ConditionalDriftingDataset(
        config.data_path, 
        config.cond_data_path, 
        config.seq_length
    )
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
    model = ConditionalDriftingModel(config).to(config.device)
    diffusion = ConditionalDiffusionProcess(config)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.num_epochs)
    
    # 梯度累积
    accumulation_steps = config.gradient_accumulation_steps
    
    # 训练循环
    logger.info("Starting conditional training...")
    best_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch_idx, (batch, cond) in enumerate(progress_bar):
            # 将数据移动到设备
            batch = batch.to(config.device)  # [batch, seq_len, input_dim]
            cond = cond.to(config.device)    # [batch, seq_len, cond_dim]
            
            # 采样时间步
            t = torch.randint(0, config.T, (batch.shape[0],), device=config.device)
            
            # 采样噪声
            noise = torch.randn_like(batch)
            
            # 前向扩散
            xt = diffusion.q_sample(batch, t, noise)
            
            # 随机丢弃条件（用于分类器无关引导）
            if config.cond_drop_rate > 0:
                cond_mask = torch.bernoulli(
                    torch.ones(batch.shape[0], device=config.device) * (1 - config.cond_drop_rate)
                ).bool()
            else:
                cond_mask = torch.ones(batch.shape[0], device=config.device).bool()
            
            # 预测噪声
            predicted_noise = model(xt, t, cond, cond_mask)
            
            # 计算损失 (MSE)
            loss = nn.MSELoss()(predicted_noise, noise)
            loss = loss / accumulation_steps
            
            # 反向传播
            loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * accumulation_steps
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': loss.item() * accumulation_steps,
                'lr': scheduler.get_last_lr()[0],
                'cond_used': cond_mask.float().mean().item()
            })
            
            # TensorBoard日志
            if batch_idx % 100 == 0:
                global_step = epoch * len(dataloader) + batch_idx
                writer.add_scalar('Train/Loss', loss.item() * accumulation_steps, global_step)
                writer.add_scalar('Train/LR', scheduler.get_last_lr()[0], global_step)
                writer.add_scalar('Train/Condition_Usage', cond_mask.float().mean().item(), global_step)
        
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
            
            # 采样可视化（每10个epoch生成样本）
            if (epoch + 1) % 20 == 0:
                visualize_samples(model, diffusion, epoch + 1, logger)
    
    # 保存最终模型
    torch.save({
        'epoch': config.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_loss,
        'config': config
    }, os.path.join(config.save_dir, 'final_model.pth'))
    
    logger.info("Conditional training completed!")
    writer.close()

# ==================== 样本可视化 ====================
def visualize_samples(model, diffusion, epoch, logger):
    """生成并可视化样本"""
    model.eval()
    with torch.no_grad():
        # 创建条件样本
        cond_sample = torch.randn(4, config.seq_length, config.cond_dim).to(config.device)
        
        # 生成样本
        shape = (4, config.seq_length, config.input_dim)
        samples = diffusion.ddpm_sample(model, cond_sample, shape)
        
        # 绘制
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for i, ax in enumerate(axes.flat):
            if i < samples.shape[0]:
                # 绘制序列
                for dim in range(min(3, config.input_dim)):
                    ax.plot(samples[i, :, dim].cpu().numpy(), label=f'Dim {dim}')
                ax.set_title(f'Sample {i+1}')
                ax.set_xlabel('Time step')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(config.log_dir, f'samples_epoch_{epoch}.png')
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Samples saved to {save_path}")

# ==================== 评估函数 ====================
def evaluate_model():
    """评估训练好的模型"""
    # 加载最佳模型
    checkpoint_path = os.path.join(config.save_dir, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        print("No trained model found.")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model = ConditionalDriftingModel(config).to(config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    diffusion = ConditionalDiffusionProcess(config)
    
    logger = logging.getLogger(__name__)
    logger.info("Evaluating model...")
    
    # 生成多个样本
    with torch.no_grad():
        for cond_type in ['random', 'zero']:
            if cond_type == 'random':
                cond = torch.randn(4, config.seq_length, config.cond_dim).to(config.device)
            else:
                cond = torch.zeros(4, config.seq_length, config.cond_dim).to(config.device)
            
            shape = (4, config.seq_length, config.input_dim)
            samples = diffusion.ddpm_sample(model, cond, shape)
            
            # 保存样本
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            for i, ax in enumerate(axes.flat):
                if i < samples.shape[0]:
                    for dim in range(min(3, config.input_dim)):
                        ax.plot(samples[i, :, dim].cpu().numpy(), label=f'Dim {dim}')
                    ax.set_title(f'Sample {i+1} ({cond_type} condition)')
                    ax.set_xlabel('Time step')
                    ax.set_ylabel('Value')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = os.path.join(config.log_dir, f'eval_samples_{cond_type}.png')
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Evaluation samples saved to {save_path}")

# ==================== 主程序 ====================
if __name__ == "__main__":
    train()
    evaluate_model()