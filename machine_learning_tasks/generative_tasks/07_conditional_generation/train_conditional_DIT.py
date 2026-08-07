"""
任务定义：
    - 任务编号：任务 7
    - 任务名称：条件图像生成 (Conditional Image Generation)
    - 领域分类：生成式 AI / 扩散模型 (Generative AI / Diffusion Models)

代表架构/算法：
    - 模型名称：Conditional Diffusion Transformer (Conditional DiT)
    - 主要论文：Peebles & Xie, "Scalable Diffusion Models with Transformers" (ICCV 2023)

核心思想与机制：
    1. 使用 ViT (Vision Transformer) 范式替代传统的 U-Net 作为 DDPM/DDIM 的 BackBone。
    2. 将图像 切块 (Patchify) 并线性投影为 1D Token 序列，叠加可学习的 1D 位置编码。
    3. 引入 adaptive LayerNorm (adaLN-Zero) 机制：将时间步步数 (t) 和类别条件 (y) 映射为调制参数，
       自适应控制 Transformer 块内部的 Scale, Shift 和 Gate 权重。

数学公式/目标函数：
    1. 前向加噪过程 (Forward Process):
       q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
       代码实现: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    2. adaLN 调制机制 (adaLN Modulation):
       adaLN(x, c) = gamma * LayerNorm(x) + beta
       在 DiT 中拓展为 6 个调制参数 (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp):
       x' = x + gate_msa * Attention((1 + scale_msa) * LayerNorm(x) + shift_msa)
    3. 优化目标 (Optimization Objective - MSE Loss):
       L_simple(theta) = E_{t, x_0, epsilon} [ || epsilon - epsilon_theta(x_t, t, y) ||^2 ]

数据输入规范：
    - 输入 x_0:  [B, C, H, W] = [Batch_Size, Channels, Image_Height, Image_Width]
    - 时间步 t:  [B] = [Batch_Size] (范围: 0 ~ T-1)
    - 条件标签 y:[B] = [Batch_Size] (范围: 0 ~ Num_Classes-1)
    - 输出 pred: [B, C, H, W] = [Batch_Size, Channels, Image_Height, Image_Width]
"""

# ==============================================================================
# 2. 依赖导入 (Imports)
# ==============================================================================
import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from einops import rearrange


# ==============================================================================
# 3. 超参数与全局配置 (Hyperparameters & Config)
# ==============================================================================
BATCH_SIZE: int = 64
EPOCHS: int = 10
LR: float = 2e-4
T: int = 1000
NUM_CLASSES: int = 10
IMAGE_SIZE: int = 32
PATCH_SIZE: int = 4
DIM: int = 128
DEPTH: int = 4
HEADS: int = 4
MLP_RATIO: float = 4.0
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 扩散过程全局 Schedule 参数计算
betas: torch.Tensor = torch.linspace(1e-4, 0.02, T)  # beta_t从 0.0001 到 0.02 线性递增
alphas: torch.Tensor = 1.0 - betas                   # alpha_t = 1 - beta_t
alphas_cumprod: torch.Tensor = torch.cumprod(alphas, dim=0)  # alpha_bar_t = \prod_{s=1}^t alpha_s
sqrt_alphas_cumprod: torch.Tensor = torch.sqrt(alphas_cumprod)  # \sqrt{\bar{\alpha}_t}
sqrt_one_minus_alphas_cumprod: torch.Tensor = torch.sqrt(1.0 - alphas_cumprod)  # \sqrt{1 - \bar{\alpha}_t}


# ==============================================================================
# 4. 数据处理与 Dataset 管道 (Data Pipeline & Utils)
# ==============================================================================
def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    前向扩散加噪过程 (Forward Diffusion Process)

    数学推导:
        x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon

    Args:
        x0 (torch.Tensor): 原始清晰图像，shape: [B, C, H, W]
        t (torch.Tensor): 采样时间步，shape: [B]
        noise (torch.Tensor, optional): 高斯噪声，shape: [B, C, H, W]

    Returns:
        xt (torch.Tensor): 加噪后的图像 x_t，shape: [B, C, H, W]
    """
    if noise is None:
        noise = torch.randn_like(x0)
        
    # 重塑为广播维度 [B, 1, 1, 1]
    sqrt_acp = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1).to(x0.device)
    sqrt_omc = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1).to(x0.device)
    
    # 广播计算加噪图像: [B, C, H, W]
    return sqrt_acp * x0 + sqrt_omc * noise


def get_synthetic_dataset(num_samples: int = 1000) -> TensorDataset:
    """
    构造合成条件图像数据集

    Args:
        num_samples (int): 样本数量

    Returns:
        dataset (TensorDataset): 包含图像与标签的 PyTorch Dataset
    """
    x0 = torch.randn(num_samples, 3, IMAGE_SIZE, IMAGE_SIZE)
    y = torch.randint(0, NUM_CLASSES, (num_samples,))
    return TensorDataset(x0, y)


# ==============================================================================
# 5. 核心子模块 (Sub-components)
# ==============================================================================
class TimeEmbedding(nn.Module):
    """
    时间步正弦位置嵌入 (Sinusoidal Time Embedding)

    数学原理:
        PE(t, 2i)   = sin(t / 10000^(2i/d))
        PE(t, 2i+1) = cos(t / 10000^(2i/d))

    Args:
        dim (int): 嵌入的目标维度 (必须为偶数)。

    Inputs:
        t (Tensor): 时间步离散张量，shape: [B]

    Outputs:
        emb (Tensor): 正弦高维嵌入张量，shape: [B, dim]
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B]
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        
        # 广播相乘: [B, 1] * [1, half_dim] -> [B, half_dim]
        t_emb = t[:, None].float() * emb[None, :]
        
        # 拼接 sin 和 cos: [B, dim]
        return torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)


class MLP(nn.Module):
    """
    多层感知机 (Feed-Forward Network)

    数学原理:
        FFN(x) = GELU(x \cdot W_1 + b_1) \cdot W_2 + b_2

    Args:
        dim (int): 输入与输出特征维度。
        hidden_dim (int): 隐藏层特征维度。
        out_dim (int, optional): 输出特征维度，默认为 dim。

    Inputs:
        x (Tensor): 输入特征，shape: [B, N, dim]

    Outputs:
        out (Tensor): 输出特征，shape: [B, N, out_dim]
    """
    def __init__(self, dim: int, hidden_dim: int, out_dim: Optional[int] = None):
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, dim] -> [B, N, hidden_dim]
        x = self.fc1(x)
        x = self.act(x)
        # x: [B, N, hidden_dim] -> [B, N, out_dim]
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    """
    多头自注意力机制 (Multi-Head Self-Attention)

    数学原理与映射:
        Attention(Q, K, V) = Softmax( (Q \cdot K^T) / \sqrt{d_k} ) \cdot V
        - Q, K, V 分别对应来自线性投影 q, k, v
        - scale 对应 1 / \sqrt{d_k}

    Args:
        dim (int): 输入总通道维度。
        heads (int): 注意力头数，默认 4。
        dropout (float): Dropout 概率，默认 0.0。

    Inputs:
        x (Tensor): 输入序列张量，shape: [B, N, dim]

    Outputs:
        out (Tensor): 自注意力交互后的张量，shape: [B, N, dim]
    """
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        # qkv 变换: [B, N, D] -> [B, N, 3 * D]
        qkv = self.qkv(x)
        
        # 维度拆分与转置: [B, N, 3*D] -> [3, B, heads, N, head_dim]
        qkv = qkv.reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 单个 shape: [B, heads, N, head_dim]
        
        # 关联度点积计算: [B, heads, N, head_dim] @ [B, heads, head_dim, N] -> [B, heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # 加权融合与还原: [B, heads, N, N] @ [B, heads, N, head_dim] -> [B, heads, N, head_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)  # -> [B, N, D]
        x = self.proj(x)  # -> [B, N, D]
        return x


class DiTBlock(nn.Module):
    """
    DiT 基础块：采用自适应 LayerNorm (adaLN-Zero) 进行条件控制注入

    计算流与公式映射:
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = Chunk(Linear(SiLU(c)), 6)
        x' = x + gate_msa * Attention( LayerNorm(x) * (1 + scale_msa) + shift_msa )
        out = x' + gate_mlp * MLP( LayerNorm(x') * (1 + scale_mlp) + shift_mlp )

    Args:
        dim (int): 特征通道维度。
        heads (int): 注意力头数。
        mlp_ratio (float): MLP 隐层维度放大倍率。
        dropout (float): Dropout 概率。

    Inputs:
        x (Tensor): 输入图像 Token 序列，shape: [B, N, dim]
        c (Tensor): 条件嵌入向量 (t_emb + y_emb)，shape: [B, dim]

    Outputs:
        out (Tensor): 经过条件调制与自注意力后的 Token 序列，shape: [B, N, dim]
    """
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, heads=heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        
        # 条件调制映射层 (adaLN)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6, bias=True)
        )
        
        # 初始化 adaLN linear 权值为 0，保证初始训练阶段残差分支接近恒等映射 (Zero-Init)
        nn.init.constant_(self.adaLN_modulation[1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # c: [B, dim] -> modulation parameters: 6 个 [B, dim] 张量
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        
        # --- 子块 1: adaLN + Self-Attention + Gate 残差 ---
        # 调制 LayerNorm 输入: [B, N, dim]
        norm_x1 = self.norm1(x) * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        # 自注意力并进行门控缩放加回输入: [B, N, dim]
        x = x + gate_msa[:, None, :] * self.attn(norm_x1)
        
        # --- 子块 2: adaLN + MLP + Gate 残差 ---
        # 调制 LayerNorm 输入: [B, N, dim]
        norm_x2 = self.norm2(x) * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        # MLP 并进行门控缩放加回输入: [B, N, dim]
        x = x + gate_mlp[:, None, :] * self.mlp(norm_x2)
        
        return x


# ==============================================================================
# 6. 顶层模型 / Pipeline 主体 (Top-level Architecture / Model)
# ==============================================================================
class ConditionalDiT(nn.Module):
    """
    条件 Diffusion Transformer 顶层主网络 (Conditional DiT)

    处理逻辑:
        1. Patchify: 2D 图像 [B, C, H, W] 分块展开为 1D 序列 [B, N, patch_dim]。
        2. Patch Embedding & Pos Embed: 映射至 [B, N, dim] 并加上 1D 位置编码。
        3. Condition Embedding: 正弦时间步编码与类别 Label Embedding 相加得到 c。
        4. Transformer Backbone: 堆叠 迭代多个 DiTBlock (包含 adaLN 调制)。
        5. Unpatchify: 将 Transformer 输出 Token 逆映射还原回图像维度 [B, C, H, W]。

    Args:
        in_ch (int): 输入图像通道数，默认 3。
        image_size (int): 输入图像边长。
        patch_size (int): 图像切块 Patch 边长。
        num_classes (int): 条件类别总数。
        dim (int): Transformer 主体隐层维度。
        depth (int): DiT Block 堆叠深度。
        heads (int): 注意力头数。
        mlp_ratio (float): MLP 隐层通道倍率。
        cond_dim (int): 时间步基础嵌入维度。

    Inputs:
        x (Tensor): 噪声图像，shape: [B, C, H, W]
        t (Tensor): 时间步离散索引，shape: [B]
        y (Tensor): 类别标签，shape: [B]

    Outputs:
        pred_noise (Tensor): 预测的噪声特征，shape: [B, C, H, W]
    """
    def __init__(
        self,
        in_ch: int = 3,
        image_size: int = IMAGE_SIZE,
        patch_size: int = PATCH_SIZE,
        num_classes: int = NUM_CLASSES,
        dim: int = DIM,
        depth: int = DEPTH,
        heads: int = HEADS,
        mlp_ratio: float = MLP_RATIO,
        cond_dim: int = 128,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = in_ch * patch_size * patch_size

        # Patch 线性投影层
        self.patch_embed = nn.Linear(self.patch_dim, dim)
        
        # 1D 可学习位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)

        # 时间与类别条件嵌入层
        self.time_embed = nn.Sequential(
            TimeEmbedding(cond_dim),
            nn.Linear(cond_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.class_embed = nn.Embedding(num_classes, dim)

        # DiT 核心 Block 堆叠
        self.blocks = nn.ModuleList([
            DiTBlock(dim, heads, mlp_ratio)
            for _ in range(depth)
        ])

        # 最终输出线性投影与归一化
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(dim, self.patch_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        
        # 1. Patchify: [B, C, H, W] -> [B, N, patch_dim]  (N = (H/p)*(W/p))
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        
        # 2. 线性投影与位置编码注入: [B, N, patch_dim] -> [B, N, dim]
        x = self.patch_embed(x)
        x = x + self.pos_embed

        # 3. 条件嵌入计算: [B] -> [B, dim]
        t_emb = self.time_embed(t)
        y_emb = self.class_embed(y)
        c = t_emb + y_emb  # 综合条件向量: [B, dim]

        # 4. 主体 Transformer 特征提取: [B, N, dim]
        for block in self.blocks:
            x = block(x, c)

        # 5. 还原输出维度: [B, N, dim] -> [B, N, patch_dim]
        x = self.norm_out(x)
        x = self.proj_out(x)
        
        # 6. Unpatchify 恢复 2D 图像张量: [B, N, patch_dim] -> [B, C, H, W]
        out = rearrange(
            x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
            h=H // p, w=W // p, p1=p, p2=p, c=C
        )
        return out


# ==============================================================================
# 7. 损失函数与评估指标 (Loss & Metrics)
# ==============================================================================
def compute_diffusion_loss(
    model: nn.Module, 
    x0: torch.Tensor, 
    y: torch.Tensor, 
    num_timesteps: int = T
) -> torch.Tensor:
    """
    计算扩散模型预测噪声的 MSE 损失

    Args:
        model (nn.Module): ConditionalDiT 模型实例
        x0 (torch.Tensor): 目标清晰图像，shape: [B, C, H, W]
        y (torch.Tensor): 类别标签，shape: [B]
        num_timesteps (int): 扩散总步数 T

    Returns:
        loss (torch.Tensor): 标量 MSE 损失值
    """
    # 随机采样每个样本的扩散时间步 t: [B]
    t = torch.randint(0, num_timesteps, (x0.size(0),), device=x0.device)
    
    # 采样标准高斯噪声: [B, C, H, W]
    noise = torch.randn_like(x0)
    
    # 计算加噪图像 x_t: [B, C, H, W]
    xt = q_sample(x0, t, noise)

    # 模型预测噪声: [B, C, H, W]
    pred_noise = model(xt, t, y)

    # 均方误差损失 MSE
    loss = F.mse_loss(pred_noise, noise)
    return loss


# ==============================================================================
# 8. 训练/推理逻辑与入口 (Training/Inference Execution)
# ==============================================================================
def main():
    """训练全流程入口函数"""
    # 1. 准备数据管道
    train_dataset = get_synthetic_dataset(num_samples=1000)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    # 2. 实例化模型与优化器
    model = ConditionalDiT().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    print("=" * 60)
    print(f" Conditional DiT Model Successfully Initialized")
    print(f" Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f" Running Device:    {DEVICE}")
    print("=" * 60)

    # 3. 训练循环
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for step, (x0, y) in enumerate(train_loader):
            x0 = x0.to(DEVICE)
            y = y.to(DEVICE)

            # 计算损失与反向传播
            optimizer.zero_grad()
            loss = compute_diffusion_loss(model, x0, y, num_timesteps=T)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1:02d}/{EPOCHS:02d}] | Conditional Noise MSE Loss: {avg_loss:.6f}")


if __name__ == "__main__":
    main()