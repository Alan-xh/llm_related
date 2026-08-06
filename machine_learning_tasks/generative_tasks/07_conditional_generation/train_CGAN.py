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

# ==================== 配置参数 ====================
class Config:
    # 数据参数
    data_path = "./data/train_data.npy"
    cond_data_path = "./data/conditions.npy"
    seq_length = 50
    input_dim = 10
    cond_dim = 5
    
    # 生成器架构
    latent_dim = 100  # 噪声维度
    gen_hidden_dim = 256
    gen_num_layers = 3
    
    # 判别器架构
    dis_hidden_dim = 256
    dis_num_layers = 3
    dis_dropout = 0.2
    
    # 训练参数
    batch_size = 64
    lr_g = 2e-4  # 生成器学习率
    lr_d = 2e-4  # 判别器学习率
    num_epochs = 200
    n_critic = 1  # 判别器更新次数/生成器更新次数
    gp_weight = 10.0  # 梯度惩罚权重（WGAN-GP）
    label_smoothing = 0.1  # 标签平滑
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 保存和日志
    save_dir = "./checkpoints_cgan"
    log_dir = "./logs_cgan"
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

# ==================== 条件生成器 ====================
class ConditionalGenerator(nn.Module):
    """条件生成器：从噪声和条件生成序列"""
    def __init__(self, config):
        super(ConditionalGenerator, self).__init__()
        self.config = config
        
        # 输入：噪声 + 条件（第一个时间步）
        self.fc_input = nn.Sequential(
            nn.Linear(config.latent_dim + config.cond_dim, config.gen_hidden_dim),
            nn.BatchNorm1d(config.gen_hidden_dim),
            nn.ReLU()
        )
        
        # LSTM生成器
        self.lstm = nn.LSTM(
            input_size=config.cond_dim,
            hidden_size=config.gen_hidden_dim,
            num_layers=config.gen_num_layers,
            batch_first=True,
            dropout=0.1 if config.gen_num_layers > 1 else 0
        )
        
        # 输出层
        self.fc_out = nn.Sequential(
            nn.Linear(config.gen_hidden_dim, config.gen_hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(config.gen_hidden_dim * 2),
            nn.Linear(config.gen_hidden_dim * 2, config.input_dim),
            nn.Tanh()  # 输出归一化到[-1, 1]
        )
        
    def forward(self, z, cond):
        """
        z: [batch, latent_dim]
        cond: [batch, seq_len, cond_dim]
        """
        batch_size, seq_len, _ = cond.shape
        
        # 初始化LSTM状态
        init_input = torch.cat([z, cond[:, 0, :]], dim=-1)
        init_state = self.fc_input(init_input)
        
        # 分离为h0和c0
        h0 = init_state.unsqueeze(0).repeat(self.config.gen_num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        
        # 生成序列
        lstm_out, _ = self.lstm(cond, (h0, c0))  # [batch, seq_len, hidden_dim]
        
        # 逐时间步输出
        outputs = []
        for t in range(seq_len):
            out = self.fc_out(lstm_out[:, t, :])
            outputs.append(out.unsqueeze(1))
        
        output = torch.cat(outputs, dim=1)  # [batch, seq_len, input_dim]
        
        return output

# ==================== 条件判别器 ====================
class ConditionalDiscriminator(nn.Module):
    """条件判别器：判别序列是否真实（带条件）"""
    def __init__(self, config):
        super(ConditionalDiscriminator, self).__init__()
        self.config = config
        
        # 输入：序列 + 条件
        input_size = config.input_dim + config.cond_dim
        
        # LSTM判别器
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.dis_hidden_dim,
            num_layers=config.dis_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dis_dropout if config.dis_num_layers > 1 else 0
        )
        
        # 输出层
        self.fc_out = nn.Sequential(
            nn.Linear(config.dis_hidden_dim * 2, config.dis_hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(config.dis_dropout),
            nn.Linear(config.dis_hidden_dim * 2, config.dis_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(config.dis_dropout),
            nn.Linear(config.dis_hidden_dim, 1)  # 输出真实度得分
        )
        
    def forward(self, x, cond):
        """
        x: [batch, seq_len, input_dim]
        cond: [batch, seq_len, cond_dim]
        """
        # 拼接序列和条件
        combined = torch.cat([x, cond], dim=-1)  # [batch, seq_len, input_dim + cond_dim]
        
        # LSTM编码
        lstm_out, _ = self.lstm(combined)  # [batch, seq_len, hidden_dim * 2]
        
        # 使用最后一个时间步
        last_out = lstm_out[:, -1, :]  # [batch, hidden_dim * 2]
        
        # 输出
        out = self.fc_out(last_out)  # [batch, 1]
        
        return out

# ==================== CGAN模型 ====================
class CGAN:
    def __init__(self, config):
        self.config = config
        self.generator = ConditionalGenerator(config).to(config.device)
        self.discriminator = ConditionalDiscriminator(config).to(config.device)
        
        # 优化器
        self.optimizer_g = optim.AdamW(
            self.generator.parameters(), 
            lr=config.lr_g, 
            betas=(0.5, 0.9),
            weight_decay=1e-5
        )
        self.optimizer_d = optim.AdamW(
            self.discriminator.parameters(), 
            lr=config.lr_d, 
            betas=(0.5, 0.9),
            weight_decay=1e-5
        )
        
        # 损失函数
        self.criterion = nn.BCEWithLogitsLoss()
        
    def train_discriminator(self, real_data, cond, fake_data):
        """训练判别器"""
        batch_size = real_data.size(0)
        
        # 真实数据的标签（带标签平滑）
        real_labels = torch.ones(batch_size, 1).to(self.config.device)
        if self.config.label_smoothing > 0:
            real_labels = real_labels * (1 - self.config.label_smoothing)
        
        fake_labels = torch.zeros(batch_size, 1).to(self.config.device)
        
        # 真实数据的判别
        real_validity = self.discriminator(real_data, cond)
        real_loss = self.criterion(real_validity, real_labels)
        
        # 伪造数据的判别
        fake_validity = self.discriminator(fake_data.detach(), cond)
        fake_loss = self.criterion(fake_validity, fake_labels)
        
        # 总损失
        d_loss = (real_loss + fake_loss) / 2
        
        return d_loss, real_validity.mean(), fake_validity.mean()
    
    def train_generator(self, cond):
        """训练生成器"""
        batch_size = cond.size(0)
        
        # 生成噪声
        z = torch.randn(batch_size, self.config.latent_dim).to(self.config.device)
        
        # 生成序列
        fake_data = self.generator(z, cond)
        
        # 判别
        fake_validity = self.discriminator(fake_data, cond)
        
        # 生成器损失：试图让判别器认为生成的序列是真实的
        g_loss = self.criterion(fake_validity, torch.ones(batch_size, 1).to(self.config.device))
        
        return g_loss, fake_data
    
    def gradient_penalty(self, real_data, fake_data, cond):
        """WGAN-GP梯度惩罚"""
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1, 1).to(self.config.device)
        epsilon = epsilon.expand_as(real_data)
        
        # 插值
        interpolated = epsilon * real_data + (1 - epsilon) * fake_data
        interpolated.requires_grad_(True)
        
        # 判别器对插值数据的输出
        disc_interpolated = self.discriminator(interpolated, cond)
        
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=disc_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(disc_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # 计算梯度惩罚
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty

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
    cgan = CGAN(config)
    logger.info(f"Generator parameters: {sum(p.numel() for p in cgan.generator.parameters()):,}")
    logger.info(f"Discriminator parameters: {sum(p.numel() for p in cgan.discriminator.parameters()):,}")
    
    # 训练循环
    logger.info("Starting CGAN training...")
    best_g_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        cgan.generator.train()
        cgan.discriminator.train()
        
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch_idx, (real_data, cond) in enumerate(progress_bar):
            real_data = real_data.to(config.device)
            cond = cond.to(config.device)
            
            batch_size = real_data.size(0)
            
            # ========== 训练判别器 ==========
            cgan.optimizer_d.zero_grad()
            
            # 生成伪造数据
            z = torch.randn(batch_size, config.latent_dim).to(config.device)
            fake_data = cgan.generator(z, cond).detach()
            
            # 判别器损失
            d_loss, real_score, fake_score = cgan.train_discriminator(real_data, cond, fake_data)
            
            # 梯度惩罚（WGAN-GP）
            gp = cgan.gradient_penalty(real_data, fake_data, cond)
            d_loss_total = d_loss + config.gp_weight * gp
            
            d_loss_total.backward()
            torch.nn.utils.clip_grad_norm_(cgan.discriminator.parameters(), 1.0)
            cgan.optimizer_d.step()
            
            # ========== 训练生成器 ==========
            if batch_idx % config.n_critic == 0:
                cgan.optimizer_g.zero_grad()
                
                g_loss, fake_data = cgan.train_generator(cond)
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(cgan.generator.parameters(), 1.0)
                cgan.optimizer_g.step()
            else:
                g_loss = torch.tensor(0.0)
            
            epoch_d_loss += d_loss_total.item()
            epoch_g_loss += g_loss.item() if g_loss.item() > 0 else 0
            
            # 更新进度条
            progress_bar.set_postfix({
                'D_loss': d_loss_total.item(),
                'G_loss': g_loss.item() if g_loss.item() > 0 else 0,
                'D_real': real_score.mean().item(),
                'D_fake': fake_score.mean().item(),
                'GP': gp.item()
            })
            
            # TensorBoard日志
            if batch_idx % 100 == 0:
                global_step = epoch * len(dataloader) + batch_idx
                writer.add_scalar('Train/D_Loss', d_loss_total.item(), global_step)
                writer.add_scalar('Train/G_Loss', g_loss.item() if g_loss.item() > 0 else 0, global_step)
                writer.add_scalar('Train/Real_Score', real_score.mean().item(), global_step)
                writer.add_scalar('Train/Fake_Score', fake_score.mean().item(), global_step)
                writer.add_scalar('Train/Gradient_Penalty', gp.item(), global_step)
        
        # Epoch结束
        avg_d_loss = epoch_d_loss / len(dataloader)
        avg_g_loss = epoch_g_loss / len(dataloader)
        
        logger.info(f"Epoch {epoch+1}/{config.num_epochs} - "
                   f"D Loss: {avg_d_loss:.6f}, G Loss: {avg_g_loss:.6f}")
        