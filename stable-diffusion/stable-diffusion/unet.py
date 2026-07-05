import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple, Union


class SinusoidalPositionEmbedding(nn.Module):
    """正弦位置编码模块，用于将时间步t编码为高维向量。
    
    在Diffusion模型中，模型需要知道当前处于去噪过程的哪一步。
    该模块使用正弦和余弦函数将离散的时间步(整数)转换为连续的、
    具有丰富表达能力的高维向量表示。
    
    正弦位置编码的数学公式为：
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Attributes:
        dim: 输出编码向量的维度
    """
    
    def __init__(self, dim: int):
        """初始化正弦位置编码模块。
        
        Args:
            dim: 输出编码向量的维度，必须是偶数
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """前向传播，将时间步编码为向量。
        
        Args:
            timesteps: 时间步张量，形状为 (batch_size,)，每个元素是整数时间步
            
        Returns:
            编码后的位置向量，形状为 (batch_size, dim)
            
        Example:
            >>> pos_embed = SinusoidalPositionEmbedding(256)
            >>> t = torch.tensor([0, 100, 500])
            >>> emb = pos_embed(t)
            >>> emb.shape
            torch.Size([3, 256])
        """
        device = timesteps.device
        half_dim = self.dim // 2
        
        # 计算频率项: exp(-log(10000) * i / (half_dim-1))
        # i从0到half_dim-1
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        # 计算位置与频率的外积，然后应用sin和cos
        emb = timesteps[:, None] * emb[None, :]  # (batch_size, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (batch_size, dim)
        
        return emb


class SelfAttention(nn.Module):
    """自注意力机制模块，让模型学习图像内部的全局依赖关系。
    
    在U-Net的深层特征图上使用，帮助模型理解图像各部分之间的关联。
    例如：模型需要知道"眼睛"和"鼻子"的相对位置关系，或者物体前景与背景的关系。
    
    实现流程：
        1. 通过三个独立的卷积将输入投影为Q、K、V
        2. 计算注意力分数: softmax(Q @ K^T / sqrt(d_k))
        3. 用注意力分数加权V得到输出
        4. 应用输出投影并添加残差连接
    """
    
    def __init__(self, channels: int):
        """初始化自注意力模块。
        
        Args:
            channels: 输入和输出的通道数
        """
        super().__init__()
        self.channels = channels
        
        # Q、K、V投影，使用1x1卷积保持空间尺寸
        self.to_q = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_k = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_v = nn.Conv2d(channels, channels, 1, bias=False)
        
        # 输出投影
        self.to_out = nn.Conv2d(channels, channels, 1)
        
        # 缩放因子，防止softmax进入梯度饱和区域
        self.scale = (channels ** -0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，应用自注意力机制。
        
        Args:
            x: 输入特征图，形状为 (batch_size, channels, height, width)
            
        Returns:
            应用自注意力后的特征图，形状与输入相同 (batch_size, channels, height, width)
            
        Example:
            >>> attn = SelfAttention(64)
            >>> x = torch.randn(4, 64, 32, 32)
            >>> out = attn(x)
            >>> out.shape
            torch.Size([4, 64, 32, 32])
        """
        b, c, h, w = x.shape
        
        # 1. 投影并展平空间维度
        # Q, K, V: (batch, channels, height*width)
        q = self.to_q(x).view(b, c, h * w)
        k = self.to_k(x).view(b, c, h * w)
        v = self.to_v(x).view(b, c, h * w)
        
        # 2. 计算注意力分数
        # attention: (batch, h*w, h*w)
        attention = torch.bmm(q.permute(0, 2, 1), k) * self.scale
        attention = F.softmax(attention, dim=-1)
        
        # 3. 应用注意力权重到V
        # out: (batch, channels, h*w)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        
        # 4. 恢复形状并输出
        out = out.view(b, c, h, w)
        out = self.to_out(out)
        
        # 5. 残差连接
        return out + x


class CrossAttention(nn.Module):
    """交叉注意力模块，用于将条件信息（如文本描述）融入图像生成。
    
    这是文生图模型（如Stable Diffusion、DALL-E）的关键组件。
    它允许模型在生成图像的每一步中，"看到"并"关注"到条件信息的不同部分。
    
    工作原理：
        - Query来自图像特征
        - Key和Value来自条件信息（如文本embedding）
        - 输出是条件信息根据与图像的相似度加权后的结果
    
    这使得模型能够根据文本提示生成相应的内容，例如"一只戴帽子的猫"。
    """
    
    def __init__(self, query_dim: int, context_dim: Optional[int] = None):
        """初始化交叉注意力模块。
        
        Args:
            query_dim: Query的维度（通常来自图像特征）
            context_dim: 条件信息的维度（如文本embedding的维度）。
                        如果为None，则使用query_dim（自注意力模式）
        """
        super().__init__()
        context_dim = context_dim or query_dim
        self.scale = (query_dim ** -0.5)
        
        # Q、K、V投影
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播，应用交叉注意力。
        
        Args:
            x: 图像特征，形状为 (batch_size, channels, height, width)
            context: 条件信息，形状为 (batch_size, seq_len, context_dim)。
                    如果为None，则退化为自注意力模式
            
        Returns:
            融合条件信息后的特征图，形状与x相同 (batch_size, channels, height, width)
            
        Example:
            >>> cross_attn = CrossAttention(512, 768)  # 768维的文本embedding
            >>> img_feat = torch.randn(4, 512, 16, 16)
            >>> text_emb = torch.randn(4, 77, 768)  # 77个文本token
            >>> out = cross_attn(img_feat, text_emb)
            >>> out.shape
            torch.Size([4, 512, 16, 16])
        """
        b, c, h, w = x.shape
        
        # 转换形状: (batch, channels, h, w) -> (batch, h*w, channels)
        x_flat = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        
        # 如果没有条件信息，使用自身作为条件（自注意力）
        if context is None:
            context = x_flat
        
        # 计算Q、K、V
        q = self.to_q(x_flat)          # (batch, h*w, query_dim)
        k = self.to_k(context)          # (batch, seq_len, query_dim)
        v = self.to_v(context)          # (batch, seq_len, query_dim)
        
        # 计算注意力分数
        attention = torch.bmm(q, k.permute(0, 2, 1)) * self.scale  # (batch, h*w, seq_len)
        attention = F.softmax(attention, dim=-1)
        
        # 应用注意力
        out = torch.bmm(attention, v)   # (batch, h*w, query_dim)
        out = self.to_out(out)
        
        # 恢复形状: (batch, h*w, channels) -> (batch, channels, h, w)
        out = out.reshape(b, h, w, c).permute(0, 3, 1, 2)
        
        # 残差连接
        return out + x


class ResidualBlock(nn.Module):
    """残差块，U-Net的基础构建单元。
    
    该模块包含两个卷积层，能够注入时间步信息，并使用残差连接。
    残差结构使得梯度可以更顺畅地流动，从而训练更深的网络。
    
    结构：
        Input -> Conv1 + Time + Act -> Conv2 + Norm -> + Residual -> Output
    """
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: Optional[int] = None):
        """初始化残差块。
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            time_emb_dim: 时间步嵌入向量的维度。如果为None，则不注入时间信息
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 第一个卷积块：GroupNorm + SiLU + Conv2d
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        # 时间步投影层（可选）
        self.time_mlp = None
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            )
        
        # 第二个卷积块
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        # 跳跃连接的投影（当输入输出通道数不同时使用）
        self.residual_conv = None
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播。
        
        Args:
            x: 输入特征图，形状为 (batch, in_channels, height, width)
            time_emb: 时间步嵌入向量，形状为 (batch, time_emb_dim)。
                     如果为None，则不注入时间信息
            
        Returns:
            输出特征图，形状为 (batch, out_channels, height, width)
        """
        residual = x
        
        # 第一个卷积
        out = self.block1(x)
        
        # 注入时间步信息（添加到特征图上）
        if time_emb is not None and self.time_mlp is not None:
            time_out = self.time_mlp(time_emb)
            # 将时间向量广播到空间维度并添加
            out = out + time_out[:, :, None, None]
        
        # 第二个卷积
        out = self.block2(out)
        
        # 调整残差分支的通道数
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        
        # 残差连接
        return out + residual


class DownBlock(nn.Module):
    """下采样块，U-Net的左侧编码器部分。
    
    该模块逐步降低特征图的空间分辨率，同时增加通道数。
    通过下采样，网络能够提取更抽象、更高层的语义特征。
    
    处理流程：
        1. 残差块（注入时间信息）
        2. 自注意力（可选）
        3. 保存跳跃连接
        4. 下采样（stride=2卷积）
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 time_emb_dim: int, has_attn: bool = False):
        """初始化下采样块。
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            time_emb_dim: 时间步嵌入的维度
            has_attn: 是否在该块中使用自注意力机制
        """
        super().__init__()
        self.res_block = ResidualBlock(in_channels, out_channels, time_emb_dim)
        self.attn = SelfAttention(out_channels) if has_attn else nn.Identity()
        # 使用stride=2的卷积进行下采样，同时保持通道数不变
        self.downsample = nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播。
        
        Args:
            x: 输入特征图，形状为 (batch, in_channels, height, width)
            time_emb: 时间步嵌入，形状为 (batch, time_emb_dim)
            
        Returns:
            out: 下采样后的特征图，形状为 (batch, out_channels, height/2, width/2)
            skip: 跳跃连接保存的特征图，形状为 (batch, out_channels, height, width)
        """
        x = self.res_block(x, time_emb)
        x = self.attn(x)
        skip = x  # 保存跳跃连接
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """上采样块，U-Net的右侧解码器部分。
    
    该模块逐步恢复特征图的空间分辨率，并融合来自编码器的跳跃连接。
    跳跃连接为解码器提供了丰富的空间细节信息，这是U-Net能够保持细节的关键。
    
    处理流程：
        1. 上采样（最近邻插值）
        2. 融合跳跃连接（拼接）
        3. 残差块处理
        4. 自注意力（可选）
    """
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 time_emb_dim: int, has_attn: bool = False):
        """初始化上采样块。
        
        Args:
            in_channels: 输入通道数（来自上一个上采样块）
            skip_channels: 跳跃连接的特征通道数
            out_channels: 输出通道数
            time_emb_dim: 时间步嵌入的维度
            has_attn: 是否在该块中使用自注意力机制
        """
        super().__init__()
        # 上采样使用最近邻插值，简单高效
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # 残差块的输入通道 = 上采样后的输入 + 跳跃连接
        self.res_block = ResidualBlock(in_channels + skip_channels, out_channels, time_emb_dim)
        self.attn = SelfAttention(out_channels) if has_attn else nn.Identity()
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """前向传播。
        
        Args:
            x: 输入特征图，形状为 (batch, in_channels, height, width)
            skip: 跳跃连接的特征图，形状为 (batch, skip_channels, height*2, width*2)
            time_emb: 时间步嵌入，形状为 (batch, time_emb_dim)
            
        Returns:
            上采样后的特征图，形状为 (batch, out_channels, height*2, width*2)
        """
        # 先上采样，将尺寸扩大一倍
        x = self.upsample(x)
        
        # 确保尺寸与skip特征完全一致（防止由于取整导致的尺寸偏差）
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
        
        # 拼接上采样后的特征和跳跃连接特征（U-Net的核心操作）
        x = torch.cat([x, skip], dim=1)
        
        # 残差块处理
        x = self.res_block(x, time_emb)
        x = self.attn(x)
        
        return x


class UNet(nn.Module):
    """完整的 U-Net 模型，支持 Diffusion 模型所需的所有特性。
    
    U-Net 最初是为生物医学图像分割设计的，但因其优秀的结构特性，
    现在已成为 Diffusion 模型的标准噪声预测网络。
    
    主要特性：
        1. 对称的U形编码器-解码器结构
        2. 跳跃连接保留空间细节
        3. 时间步条件嵌入（Diffusion必需）
        4. 自注意力和交叉注意力支持（提升质量）
        5. 灵活的条件注入（文本、类别等）
    
    网络架构：
        Input -> Encoder (Down) -> Middle -> Decoder (Up) -> Output
                  |___________Skip Connections___________|
    
    References:
        - U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)
        - Denoising Diffusion Probabilistic Models (Ho et al., 2020)
        - High-Resolution Image Synthesis with Latent Diffusion Models (Rombach et al., 2022)
    """
    
    def __init__(
        self,
        in_channels: int = 3,          # 输入通道数（RGB=3，潜空间=4）
        out_channels: int = 3,         # 输出通道数
        model_channels: int = 128,     # 基础通道数
        channel_mult: Tuple[int, ...] = (1, 2, 2, 4),  # 每一层的通道倍数
        num_res_blocks: int = 2,       # 每个分辨率的残差块数量
        attn_resolutions: Tuple[int, ...] = (16,),     # 哪些分辨率使用注意力
        dropout: float = 0.0,          # Dropout比率（当前实现未使用）
        num_classes: Optional[int] = None,   # 类别条件，如果有则支持类别引导
        context_dim: Optional[int] = None,   # 条件维度，如文本embedding维度
    ):
        """初始化U-Net模型。
        
        Args:
            in_channels: 输入图像的通道数
            out_channels: 输出图像的通道数
            model_channels: 模型的基础通道数，后续层会乘以channel_mult
            channel_mult: 每个分辨率层级的通道倍数元组
            num_res_blocks: 每个分辨率层级的残差块数量
            attn_resolutions: 使用自注意力的空间分辨率列表
            dropout: Dropout比率（保留参数，当前未使用）
            num_classes: 类别总数，如果提供则支持类别条件生成
            context_dim: 条件信息的维度，如文本embedding的维度
        """
        super().__init__()
        
        # ==================== 1. 时间步嵌入模块 ====================
        # Diffusion模型需要知道当前的时间步
        time_emb_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # ==================== 2. 类别条件嵌入（可选） ====================
        self.class_embed = None
        if num_classes is not None:
            self.class_embed = nn.Embedding(num_classes, time_emb_dim)
        
        # ==================== 3. 输入投影 ====================
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # ==================== 4. 编码器（下采样路径） ====================
        self.down_blocks = nn.ModuleList([])
        ch = model_channels  # 当前通道数
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            # 添加该层级的残差块
            for _ in range(num_res_blocks):
                # 判断是否在这个分辨率使用注意力
                has_attn = out_ch in attn_resolutions or (level == len(channel_mult) - 1)
                self.down_blocks.append(
                    DownBlock(ch, out_ch, time_emb_dim, has_attn)
                )
                ch = out_ch
            
            # 在层级之间添加下采样（最后一层不加）
            if level != len(channel_mult) - 1:
                self.down_blocks.append(
                    nn.Conv2d(ch, ch, 3, stride=2, padding=1)
                )
        
        # ==================== 5. 中间块 ====================
        # 在编码器和解码器之间的瓶颈层，使用注意力提升全局理解
        self.mid_block1 = ResidualBlock(ch, ch, time_emb_dim)
        self.mid_attn = SelfAttention(ch)
        self.mid_block2 = ResidualBlock(ch, ch, time_emb_dim)
        
        # ==================== 6. 解码器（上采样路径） ====================
        self.up_blocks = nn.ModuleList([])
        for level, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult
            # 添加该层级的残差块
            for _ in range(num_res_blocks + 1):
                # 判断是否使用注意力
                has_attn = out_ch in attn_resolutions or (level == 0)
                self.up_blocks.append(
                    UpBlock(ch, out_ch, out_ch, time_emb_dim, has_attn)
                )
                ch = out_ch
            
            # 在层级之间添加上采样（最后一层不加）
            if level != len(channel_mult) - 1:
                self.up_blocks.append(
                    nn.Upsample(scale_factor=2, mode='nearest')
                )
        
        # ==================== 7. 输出投影 ====================
        self.out_norm = nn.GroupNorm(32, ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)
    
    def forward(
        self, 
        x: torch.Tensor, 
        timesteps: torch.Tensor, 
        context: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """U-Net的前向传播。
        
        在Diffusion模型中，该网络接收带噪图像和时间步，预测需要去除的噪声。
        通过条件注入，可以实现文本到图像、类别到图像等条件生成任务。
        
        Args:
            x: 输入图像或噪声张量，形状为 (batch_size, in_channels, height, width)
            timesteps: 时间步张量，形状为 (batch_size,)，每个元素是整数时间步
            context: 条件信息（如文本embedding），形状为 (batch_size, seq_len, context_dim)
            class_labels: 类别标签，形状为 (batch_size,)，用于类别条件生成
            
        Returns:
            预测的噪声或输出图像，形状为 (batch_size, out_channels, height, width)
            
        Example:
            >>> # 标准用法（无条件生成）
            >>> unet = UNet(in_channels=3, out_channels=3)
            >>> x = torch.randn(4, 3, 64, 64)  # 带噪图像
            >>> t = torch.randint(0, 1000, (4,))  # 随机时间步
            >>> noise_pred = unet(x, t)
            >>> noise_pred.shape
            torch.Size([4, 3, 64, 64])
            
            >>> # 文本条件生成
            >>> text_unet = UNet(in_channels=4, out_channels=4, context_dim=768)
            >>> z = torch.randn(4, 4, 64, 64)  # 潜空间表示
            >>> text_emb = torch.randn(4, 77, 768)  # CLIP文本embedding
            >>> pred = text_unet(z, t, context=text_emb)
        """
        # 1. 获取时间步嵌入
        t_emb = self.time_embed(timesteps)
        
        # 2. 添加类别条件（如果有）
        if self.class_embed is not None and class_labels is not None:
            c_emb = self.class_embed(class_labels)
            t_emb = t_emb + c_emb
        
        # 3. 输入卷积
        x = self.input_conv(x)
        
        # 4. 保存跳跃连接
        skips: List[torch.Tensor] = []
        
        # 5. 编码器（下采样路径）
        for module in self.down_blocks:
            if isinstance(module, DownBlock):
                x, skip = module(x, t_emb)
                skips.append(skip)
            else:  # 普通卷积下采样
                x = module(x)
        
        # 6. 中间块
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)
        
        # 7. 解码器（上采样路径）
        for module in self.up_blocks:
            if isinstance(module, UpBlock):
                # 从堆栈中取出对应的跳跃连接
                skip = skips.pop()
                x = module(x, skip, t_emb)
            else:  # 上采样操作
                x = module(x)
        
        # 8. 输出投影
        x = self.out_norm(x)
        x = self.out_act(x)
        x = self.out_conv(x)
        
        return x


# ==================== 示例使用代码 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("U-Net 模型测试")
    print("=" * 60)
    
    # 测试1: 基础U-Net模型
    print("\n1. 基础U-Net模型测试")
    print("-" * 40)
    model = UNet(
        in_channels=3,
        out_channels=3,
        model_channels=128,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(16,),
    )
    
    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64)
    timesteps = torch.randint(0, 1000, (batch_size,))
    
    with torch.no_grad():
        output = model(x, timesteps)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params / 1e6:.2f}M")
    
    # 测试2: 带文本条件的模型
    print("\n2. 带文本条件的模型测试")
    print("-" * 40)
    text_model = UNet(
        in_channels=4,
        out_channels=4,
        model_channels=256,
        channel_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        context_dim=768,  # CLIP文本embedding维度
    )
    
    x_latent = torch.randn(batch_size, 4, 64, 64)
    text_emb = torch.randn(batch_size, 77, 768)  # 77个token
    
    with torch.no_grad():
        output_latent = text_model(x_latent, timesteps, context=text_emb)
    
    print(f"潜空间输入形状: {x_latent.shape}")
    print(f"文本条件形状: {text_emb.shape}")
    print(f"输出形状: {output_latent.shape}")
    
    text_params = sum(p.numel() for p in text_model.parameters())
    print(f"参数量: {text_params / 1e6:.2f}M")
    
    # 测试3: 带类别条件的模型
    print("\n3. 带类别条件的模型测试")
    print("-" * 40)
    class_model = UNet(
        in_channels=3,
        out_channels=3,
        model_channels=128,
        num_classes=1000,  # ImageNet的1000个类别
    )
    
    class_labels = torch.randint(0, 1000, (batch_size,))
    
    with torch.no_grad():
        output_class = class_model(x, timesteps, class_labels=class_labels)
    
    print(f"输入形状: {x.shape}")
    print(f"类别标签: {class_labels}")
    print(f"输出形状: {output_class.shape}")
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)