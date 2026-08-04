"""
任务 7：条件生成（Conditional Generation）
代表模型：条件 DiT（Conditional Diffusion Transformer）
损失函数：MSE（预测噪声）
使用 DiT 架构替代 U-Net，演示条件生成训练。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from einops import rearrange

# 超参数
BATCH_SIZE = 64
EPOCHS = 10
LR = 2e-4
T = 1000
NUM_CLASSES = 10
IMAGE_SIZE = 32
PATCH_SIZE = 4
DIM = 128
DEPTH = 4
HEADS = 4
MLP_RATIO = 4.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 扩散过程参数
betas = torch.linspace(1e-4, 0.02, T) # torch.linspace 创建等间隔数值序列的函数
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)


class TimeEmbedding(nn.Module):
    """时间步嵌入"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        t = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(t), torch.cos(t)], dim=-1)


class MLP(nn.Module):
    """多层感知机"""
    def __init__(self, dim, hidden_dim, out_dim=None):
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    """多头注意力机制"""
    def __init__(self, dim, heads=4, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        return x


class DiTBlock(nn.Module):
    """DiT 基础块：包含自注意力和条件注入"""
    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads=heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * mlp_ratio)
        
        # 条件注入 (adaptive layer norm)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6, bias=True)
        )

    def forward(self, x, c):
        # c 是条件嵌入 (时间 + 类别)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        
        # 自适应 LayerNorm + 自注意力 + 残差连接
        x = x + gate_msa[:, None] * self.attn(
            self.norm1(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        )
        
        # 自适应 LayerNorm + MLP + 残差连接
        x = x + gate_mlp[:, None] * self.mlp(
            self.norm2(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )
        return x


class ConditionalDiT(nn.Module):
    """
    条件 Diffusion Transformer
    将图像分割成 patches，使用 Transformer 处理，支持时间步和类别条件
    """
    def __init__(
        self,
        in_ch=3,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_classes=NUM_CLASSES,
        dim=DIM,
        depth=DEPTH,
        heads=HEADS,
        mlp_ratio=MLP_RATIO,
        cond_dim=128,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = in_ch * patch_size * patch_size

        # 输入嵌入：将图像块线性映射到 dim 维
        self.patch_embed = nn.Linear(self.patch_dim, dim)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)

        # 条件嵌入
        self.time_embed = nn.Sequential(
            TimeEmbedding(cond_dim),
            nn.Linear(cond_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.class_embed = nn.Embedding(num_classes, dim)

        # DiT 块序列
        self.blocks = nn.ModuleList([
            DiTBlock(dim, heads, mlp_ratio)
            for _ in range(depth)
        ])

        # 输出层
        self.norm_out = nn.LayerNorm(dim)
        self.proj_out = nn.Linear(dim, self.patch_dim)

    def forward(self, x, t, y):
        """
        x: [B, C, H, W] 噪声图像
        t: [B] 时间步
        y: [B] 类别标签
        """
        B, C, H, W = x.shape
        p = self.patch_size
        
        # 分割成 patches: [B, C, H, W] -> [B, num_patches, patch_dim]
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_embed(x)
        
        # 添加位置编码
        x = x + self.pos_embed

        # 条件嵌入
        t_emb = self.time_embed(t)
        y_emb = self.class_embed(y)
        c = t_emb + y_emb  # [B, dim]

        # 通过 DiT 块
        for block in self.blocks:
            x = block(x, c)

        # 输出投影
        x = self.norm_out(x)
        x = self.proj_out(x)
        
        # 还原为图像 [B, C, H, W]
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                      h=H//p, w=W//p, p1=p, p2=p, c=C)
        return x


def q_sample(x0, t, noise=None):
    """前向扩散过程"""
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_acp = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1).to(x0.device)
    sqrt_omc = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1).to(x0.device)
    return sqrt_acp * x0 + sqrt_omc * noise


def get_synthetic_dataset(num_samples=1000):
    """生成合成数据集"""
    x0 = torch.randn(num_samples, 3, IMAGE_SIZE, IMAGE_SIZE)
    y = torch.randint(0, NUM_CLASSES, (num_samples,))
    return TensorDataset(x0, y)


def main():
    """主训练函数"""
    train_loader = DataLoader(
        get_synthetic_dataset(), batch_size=BATCH_SIZE, shuffle=True
    )

    model = ConditionalDiT().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"使用设备: {DEVICE}")

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for x0, y in train_loader:
            x0 = x0.to(DEVICE)
            y = y.to(DEVICE)
            
            # 采样时间步
            t = torch.randint(0, T, (x0.size(0),), device=DEVICE)
            
            # 添加噪声
            noise = torch.randn_like(x0)
            xt = q_sample(x0, t, noise)

            # 预测噪声
            pred_noise = model(xt, t, y)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}] Conditional Noise MSE: {avg_loss:.4f}")


if __name__ == "__main__":
    main()