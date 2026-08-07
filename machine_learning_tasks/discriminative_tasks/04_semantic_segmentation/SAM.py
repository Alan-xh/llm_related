"""
任务定义：
    - 任务编号：Task 005
    - 任务名称：可提示语义/实例分割 (Promptable Segmentation)
    - 领域分类：计算机视觉 (Computer Vision)

代表架构/算法：
    - 模型名称：SAM 1 (Segment Anything Model v1)
    - 论文来源：Kirillov et al., "Segment Anything", ICCV 2023 (Meta AI).

核心思想与机制：
    1. Image Encoder (Heavy ViT)：采用 Vision Transformer (ViT) 提取图像的高维密特征 (Image Embedding)。
       由于图像编码计算开销大，在实际交互或批处理中仅需对单张图像运行一次。
    2. Prompt Encoder (Flexible Encoders)：
       - 稀疏提示 (Sparse Prompts)：点 (Points)、框 (Boxes)，通过位置编码 (Positional Embeddings) 映射至固定维度。
       - 密集提示 (Dense Prompts)：掩码 (Masks)，通过卷积降采样后与图像 Embedding 在空间维度按元素相加。
    3. Lightweight Mask Decoder (Two-Way Transformer Block)：
       - 结合 Image Embedding 与 Prompt Tokens，采用双向 Cross-Attention (Query-to-Key & Key-to-Query) 进行特征更新。
       - 解码出多候选 Mask (解决歧义性 Ambiguity) 及对应的 IoU 预测得分 (IoU Scores)。

数学公式 / 目标函数与代码映射：
    1. 二分类交叉熵损失 (Binary Cross Entropy Loss):
       L_BCE = - (1 / N) * \sum_{i=1}^{N} [ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]
       代码映射: torch.nn.functional.binary_cross_entropy_with_logits(pred_masks, gt_masks)

    2. 二分类 Dice 损失 (Binary Dice Loss):
       L_Dice = 1 - (2 * \sum (p * y) + \epsilon) / (\sum p + \sum y + \epsilon)
       符号映射: p -> pred_probs (torch.sigmoid(pred_masks)), y -> gt_masks_expand
       代码映射: 1.0 - (2.0 * intersection + smooth) / (cardinality + smooth)

    3. IoU 预测均方误差损失 (MSE Loss for IoU Prediction):
       L_IoU = MSE(Predicted_IoU, True_IoU)
       符号映射: Predicted_IoU -> pred_ious, True_IoU -> actual_iou
       代码映射: torch.nn.functional.mse_loss(pred_ious, actual_iou.detach())

    4. 综合 Focal/BCE + Dice + IoU 复合损失 (Combined Loss):
       L_total = \lambda_1 * L_BCE + \lambda_2 * L_Dice + \lambda_3 * L_IoU

数据输入输出规范：
    - 输入图像 (Input Images): [B, 3, H, W] = [Batch_Size, 3, 256, 256], 类型 float32
    - 提示点坐标 (Point Coords): [B, N_pts, 2], 类型 float32 (像素绝对坐标 [x, y])
    - 提示点标签 (Point Labels): [B, N_pts], 类型 int64 (1: 前景, 0: 背景, -1: 填充点)
    - 输出掩码 (Output Masks): [B, Multimask_Output_Num, H, W] = [Batch_Size, 3, 256, 256], 类型 float32
    - 输出 IoU 预测 (Output IoU Scores): [B, Multimask_Output_Num], 类型 float32
"""

import math
from typing import Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ==============================================================================
# 3. 超参数与全局配置 (Hyperparameters & Config)
# ==============================================================================
BATCH_SIZE = 2
EPOCHS = 3
LR = 1e-4
IMAGE_SIZE = 256
IN_CHANNELS = 3
EMBED_DIM = 256      # SAM 解码器与 Transformer 内部统一特征维度
TRANSFORMER_DEPTH = 2
NUM_HEADS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# 4. 数据处理与 Dataset 管道 (Data Pipeline & Utils)
# ==============================================================================
def get_synthetic_sam_dataset(num_samples: int = 16, image_size: int = IMAGE_SIZE) -> TensorDataset:
    """
    生成合成的 2D 图像、提示点与对应二进制掩码 (Binary Mask) 数据集。

    Args:
        num_samples (int): 合成样本数量，默认 16。
        image_size (int): 图像边长（高/宽），默认 256。

    Outputs:
        TensorDataset: 包含图像、提示点坐标、提示点标签以及 Ground-Truth 掩码。
            - images (Tensor): [num_samples, 3, image_size, image_size], Float32
            - point_coords (Tensor): [num_samples, 1, 2], Float32
            - point_labels (Tensor): [num_samples, 1], Long
            - masks (Tensor): [num_samples, 1, image_size, image_size], Float32
    """
    images = torch.randn(num_samples, 3, image_size, image_size, dtype=torch.float32)
    # 随机生成坐标 [x, y] 在 [0, image_size) 之间
    point_coords = torch.rand(num_samples, 1, 2, dtype=torch.float32) * image_size
    point_labels = torch.ones(num_samples, 1, dtype=torch.long)  # 均为前景点 1
    masks = torch.randint(0, 2, (num_samples, 1, image_size, image_size), dtype=torch.float32)
    return TensorDataset(images, point_coords, point_labels, masks)


# ==============================================================================
# 5. 核心子模块 / Component Modules
# ==============================================================================

class PositionEmbeddingRandom(nn.Module):
    """
    基于随机高斯特征矩阵的位置编码器 (Random Fourier Features Position Embedding)。
    用于将 2D 空间坐标及网格坐标映射至隐层维度 embed_dim。

    数学原理 / 变换逻辑:
        PE(x) = [sin(2 * \pi * B x), cos(2 * \pi * B x)]
        其中 B 为从标准正态分布采样的随机高斯矩阵，维度为 [2, num_pos_feats]。

    Args:
        num_pos_feats (int): 高斯投影特征通道数，默认 128 (拼接 sin/cos 后总维度为 256)。
        scale (float): 高斯矩阵缩放系数，默认 1.0。

    Inputs:
        size (Tuple[int, int]): 评估网格尺寸 (H, W)
        coords_input (Tensor): 稀疏坐标张量，shape: [B, N, 2]

    Outputs:
        pe (Tensor): 密集网格位置编码，shape: [C, H, W]
        pe_coords (Tensor): 稀疏坐标位置编码，shape: [B, N, C]
    """
    def __init__(self, num_pos_feats: int = 128, scale: float = 1.0) -> None:
        super().__init__()
        # 随机二维高斯矩阵 B: shape [2, num_pos_feats]
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """
        内部公式推导计算: 2 * \pi * (2 * coords - 1) @ B
        
        Args:
            coords (Tensor): 归一化坐标 [..., 2]
        Returns:
            pe (Tensor): 编码特征 [..., 2 * num_pos_feats]
        """
        coords = 2 * coords - 1                                        # 归一化至 [-1, 1]
        coords = coords @ self.positional_encoding_gaussian_matrix     # [..., num_pos_feats]
        coords = 2 * math.pi * coords                                  # 缩放至相位区间
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1) # [..., 2 * num_pos_feats]

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """生成密集网格 2D 位置编码: [C, H, W]"""
        h, w = size
        grid = torch.ones((h, w), device=self.positional_encoding_gaussian_matrix.device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1)) # [H, W, C]
        return pe.permute(2, 0, 1)                                     # [C, H, W]

    def forward_with_coords(self, coords_input: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """针对稀疏坐标点计算位置编码: coords_input [B, N, 2] -> [B, N, C]"""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords)                               # [B, N, C]


class ImageEncoderViT(nn.Module):
    """
    SAM 1 的主干图像编码器 (Simplified Vision Transformer Image Encoder)。
    将 [B, 3, H, W] 图像转化为下采样 16x 的稠密特征表达 [B, embed_dim, H/16, W/16]。

    Args:
        img_size (int): 输入图像边长，默认 256。
        patch_size (int): Patch 块大小，默认 16。
        in_chans (int): 输入通道数，默认 3。
        embed_dim (int): 隐藏层向量维度，默认 256。

    Inputs:
        x (Tensor): 输入图像张量，shape: [B, 3, H, W]

    Outputs:
        out (Tensor): 图像嵌入特征图，shape: [B, embed_dim, H/16, W/16]
    """
    def __init__(self, img_size: int = 256, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 256):
        super().__init__()
        self.img_size = img_size
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=NUM_HEADS, dim_feedforward=embed_dim * 4, batch_first=True, activation="gelu"
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        grid_h = img_size // patch_size
        grid_w = img_size // patch_size
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.LayerNorm([embed_dim, grid_h, grid_w]),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.LayerNorm([embed_dim, grid_h, grid_w]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]
        x = self.patch_embed(x)                                         # [B, C, H/16, W/16]
        B, C, H, W = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)                          # [B, HW, C]
        x_trans = self.blocks(x_flat)                                   # [B, HW, C]
        x_out = x_trans.permute(0, 2, 1).view(B, C, H, W)               # [B, C, H/16, W/16]
        out = self.neck(x_out)                                          # [B, C, H/16, W/16]
        return out


class PromptEncoder(nn.Module):
    """
    SAM 1 可提示编码器 (Prompt Encoder)。
    处理点 (Points)、框 (Boxes) 等稀疏提示与掩码 (Masks) 密集提示。

    Args:
        embed_dim (int): 特征编码维度。
        image_embedding_size (Tuple[int, int]): 特征图网格尺寸 (H', W')。
        input_image_size (Tuple[int, int]): 原始输入图像尺寸 (H, W)。

    Inputs:
        points (Tuple[Tensor, Tensor], optional): 提示点 (coords [B, N, 2], labels [B, N])
        boxes (Tensor, optional): 边界框提示 [B, N, 4]
        masks (Tensor, optional): 密集掩码提示 [B, 1, H, W]

    Outputs:
        sparse_embeddings (Tensor): 稀疏提示 Embedding, shape: [B, N_tokens, embed_dim]
        dense_embeddings (Tensor): 密集提示 Embedding, shape: [B, embed_dim, H', W']
    """
    def __init__(self, embed_dim: int, image_embedding_size: Tuple[int, int], input_image_size: Tuple[int, int]):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # 1. 稀疏点提示嵌入 (Foreground/Background Embedding)
        self.num_point_embeddings = 4  # pos, neg, not_a_point, box_corner
        self.point_embeddings = nn.ModuleList([nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)])
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
        """计算稀疏点的嵌入向量并加上位置编码"""
        points = points + 0.5                                           # 像素中心偏移
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size) # [B, N, C]

        # 根据标签类型融合类型 Embedding
        # labels: 1 为前景点, 0 为背景点, -1 为填充点
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = points[0].shape[0] if points is not None else 1
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=points[0].device if points is not None else "cpu")

        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None)) # [B, N_pts, C]
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1) # [B, N_pts, C]

        # Dense Embeddings 初始化 (无 Mask 提示时输入全 0 向量)
        dense_embeddings = torch.zeros(
            (bs, self.embed_dim, self.image_embedding_size[0], self.image_embedding_size[1]),
            device=sparse_embeddings.device,
        )                                                               # [B, C, H', W']

        return sparse_embeddings, dense_embeddings


class TwoWayAttentionBlock(nn.Module):
    """
    双向 Attention 模块：实现 Tokens (Prompts) 与 Image Embedding 的双向信息交互。

    数学原理 / 变换逻辑:
        1. Self-Attention (Tokens -> Tokens):
           Tokens' = LayerNorm(Tokens + MultiHeadAttention(Q=Tokens+PE, K=Tokens+PE, V=Tokens))
        2. Cross-Attention (Tokens -> Image):
           Tokens'' = LayerNorm(Tokens' + MultiHeadAttention(Q=Tokens'+PE_tok, K=Image+PE_img, V=Image))
        3. MLP Block:
           Tokens''' = LayerNorm(Tokens'' + MLP(Tokens''))
        4. Cross-Attention (Image -> Tokens):
           Image' = LayerNorm(Image + MultiHeadAttention(Q=Image+PE_img, K=Tokens'''+PE_tok, V=Tokens'''))

    Args:
        embedding_dim (int): 隐藏特征维度 (256)。
        num_heads (int): 注意力头数 (8)。
        mlp_dim (int): MLP 中间隐层维度，默认 2048。

    Inputs:
        queries (Tensor): Prompt Tokens, shape: [B, N_tokens, C]
        keys (Tensor): Image Embeddings, shape: [B, HW, C]
        query_pe (Tensor): Prompt Tokens Positional Encoding, shape: [B, N_tokens, C]
        key_pe (Tensor): Image Positional Encoding, shape: [B, HW, C]

    Outputs:
        queries (Tensor): 更新后的 Prompt Tokens, shape: [B, N_tokens, C]
        keys (Tensor): 更新后的 Image Embeddings, shape: [B, HW, C]
    """
    def __init__(self, embedding_dim: int, num_heads: int, mlp_dim: int = 2048):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embedding_dim),
        )
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.cross_attn_image_to_token = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.norm4 = nn.LayerNorm(embedding_dim)

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, query_pe: torch.Tensor, key_pe: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Self Attention on Queries (Tokens)
        q = queries + query_pe
        attn_out, _ = self.self_attn(q, q, queries)
        queries = self.norm1(queries + attn_out)                       # [B, N_tokens, C]

        # 2. Cross Attention: Tokens -> Image
        q = queries + query_pe
        k = keys + key_pe
        attn_out, _ = self.cross_attn_token_to_image(query=q, key=k, value=keys)
        queries = self.norm2(queries + attn_out)                       # [B, N_tokens, C]

        # 3. MLP block
        queries = self.norm3(queries + self.mlp(queries))              # [B, N_tokens, C]

        # 4. Cross Attention: Image -> Tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out, _ = self.cross_attn_image_to_token(query=k, key=q, value=queries)
        keys = self.norm4(keys + attn_out)                              # [B, HW, C]

        return queries, keys


class MaskDecoder(nn.Module):
    """
    SAM 1 轻量级掩码解码器 (Mask Decoder)。
    解析 Prompt Token 与 Image Feature，预测多候选分类 Mask 以及 IoU 质量分数。

    Args:
        transformer_dim (int): 特征通道数，默认 256。
        num_multimask_outputs (int): 多候选 Mask 输出数量，默认 3。

    Inputs:
        image_embeddings (Tensor): 图像编码特征，shape: [B, C, H', W']
        image_pe (Tensor): 图像位置编码，shape: [1, C, H', W']
        sparse_prompt_embeddings (Tensor): 稀疏提示词向量，shape: [B, N_pts, C]
        dense_prompt_embeddings (Tensor): 密集提示特征图，shape: [B, C, H', W']
        multimask_output (bool): 是否输出多候选 Mask，默认 True。

    Outputs:
        masks (Tensor): 预测掩码，shape: [B, K, H, W]
        iou_pred (Tensor): 预测 IoU 得分，shape: [B, K]
    """
    def __init__(self, transformer_dim: int = 256, num_multimask_outputs: int = 3):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.num_multimask_outputs = num_multimask_outputs

        # 特征 Tokens 维度设计: 1(iou_token) + 1(mask_token_default) + 3(multimask_tokens) = 5
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # 双向 Transformer Block 交互
        self.transformer_block = TwoWayAttentionBlock(embedding_dim=transformer_dim, num_heads=NUM_HEADS)

        # 上采样反卷积层 (4x 上采样，使图像特征恢复至 1/4 尺寸)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            nn.LayerNorm([transformer_dim // 4, 32, 32]),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )

        # MLP 预测头: 每一个 Mask Token 对应一个独立 MLP
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(transformer_dim, transformer_dim),
                    nn.ReLU(),
                    nn.Linear(transformer_dim, transformer_dim // 8),
                )
                for _ in range(self.num_mask_tokens)
            ]
        )
        self.iou_prediction_head = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.ReLU(),
            nn.Linear(transformer_dim, self.num_mask_tokens),
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. 组装输入 Tokens: [iou_token, mask_tokens, sparse_prompts]
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0) # [5, C]
        output_tokens = output_tokens.unsqueeze(0).expand(image_embeddings.shape[0], -1, -1) # [B, 5, C]
        tokens = torch.cat([output_tokens, sparse_prompt_embeddings], dim=1) # [B, 5 + N_pts, C]

        # 2. 准备图像特征打平与融合 Dense Prompt
        src = image_embeddings + dense_prompt_embeddings              # [B, C, H', W']
        b, c, h, w = src.shape
        src = src.flatten(2).permute(0, 2, 1)                          # [B, H'W', C]
        pos_src = image_pe.flatten(2).permute(0, 2, 1)                  # [B, H'W', C] (广播至 B)

        # 3. 运行 TwoWayAttention 模块
        hs, src = self.transformer_block(
            queries=tokens, keys=src, query_pe=tokens, key_pe=pos_src
        )                                                              # hs: [B, 5+N_pts, C], src: [B, H'W', C]

        iou_token_out = hs[:, 0, :]                                    # [B, C]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]     # [B, 5, C]

        # 4. 上采样解码特征图
        src = src.permute(0, 2, 1).view(b, c, h, w)                    # [B, C, H', W']
        upscaled_embedding = self.output_upscaling(src)                # [B, C/8, 4H', 4W'] = [B, 32, 64, 64]

        # 5. 计算点乘矩阵生成预测 Mask Logits
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)                    # [B, 5, C/8] = [B, 5, 32]

        b, c, h_up, w_up = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h_up * w_up)).view(b, self.num_mask_tokens, h_up, w_up)
                                                                        # [B, 5, H_up, W_up] = [B, 5, 64, 64]

        # 6. 计算预测的 IoU 得分
        iou_pred = self.iou_prediction_head(iou_token_out)             # [B, 5]

        # 选择输出 Multimask (index 1~3, 共3个通道) 还是 Singlemask (index 0, 共1个通道)
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)

        masks = masks[:, mask_slice, :, :]                              # [B, K, 64, 64] (K=3 或 1)
        iou_pred = iou_pred[:, mask_slice]                              # [B, K]

        # 双线性插值上采样恢复至原始图像分辨率 H, W
        masks = F.interpolate(masks, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
                                                                        # [B, K, H, W]
        return masks, iou_pred


# ==============================================================================
# 6. 顶层模型 / Pipeline 主体 (Top-level Architecture / Model)
# ==============================================================================
class SegmentAnythingModel(nn.Module):
    """
    SAM 1 (Segment Anything Model) 完整整合架构主体。

    架构组成：
        - ImageEncoderViT: 图像编码主干网
        - PromptEncoder: 点/框/掩码 提示编码器
        - MaskDecoder: 轻量级 Mask 交互解码器

    Args:
        image_size (int): 图像边长 (256)。
        embed_dim (int): 全局嵌入通道数 (256)。

    Inputs:
        images (Tensor): 图像张量，shape: [B, 3, H, W]
        point_coords (Tensor): 提示点坐标，shape: [B, N_pts, 2]
        point_labels (Tensor): 提示点类型，shape: [B, N_pts]
        multimask_output (bool): 是否开启多候选输出，默认 True。

    Outputs:
        low_res_masks (Tensor): 预测掩码，shape: [B, K, H, W]
        iou_predictions (Tensor): 预测 IoU 分数，shape: [B, K]
    """
    def __init__(self, image_size: int = IMAGE_SIZE, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.image_size = image_size
        self.embed_dim = embed_dim

        # 1. 图像编码器
        self.image_encoder = ImageEncoderViT(img_size=image_size, embed_dim=embed_dim)

        # 2. 提示编码器
        grid_size = image_size // 16
        self.prompt_encoder = PromptEncoder(
            embed_dim=embed_dim,
            image_embedding_size=(grid_size, grid_size),
            input_image_size=(image_size, image_size),
        )

        # 3. 掩码解码器
        self.mask_decoder = MaskDecoder(transformer_dim=embed_dim, num_multimask_outputs=3)

    def forward(
        self,
        images: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        multimask_output: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Step 1: 图像编码提取高层语义特征 [B, C, H/16, W/16]
        image_embeddings = self.image_encoder(images)

        # Step 2: 生成提示编码 (Sparse & Dense Embeddings)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )

        # Step 3: 计算位置编码，解码 Mask 与 IoU 得分
        image_pe = self.prompt_encoder.pe_layer(self.prompt_encoder.image_embedding_size).unsqueeze(0) # [1, C, H', W']
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        return low_res_masks, iou_predictions


# ==============================================================================
# 7. 损失函数与评估指标 (Loss & Metrics)
# ==============================================================================
def sam_loss(
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    pred_ious: torch.Tensor,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """
    SAM 复合损失函数: BCE Loss + Dice Loss + IoU 预测 MSE Loss。

    数学原理:
        1. L_BCE = BCEWithLogits(pred_masks, gt_masks_expand)
        2. L_Dice = 1 - (2 * \sum p * y + \epsilon) / (\sum p + \sum y + \epsilon)
        3. L_IoU = MSE(pred_ious, actual_iou)
        4. L_total = 20.0 * L_BCE + 1.0 * L_Dice + 1.0 * L_IoU

    Args:
        pred_masks (Tensor): 预测 Mask Logits [B, K, H, W]
        gt_masks (Tensor): 真实 Mask 标签 [B, 1, H, W]
        pred_ious (Tensor): 预测 IoU 得分 [B, K]
        smooth (float): 平滑项，默认 1e-5。

    Outputs:
        total_loss (Tensor): 综合标量损失
    """
    # 扩展 GT Mask 维度至多候选掩码维度 K
    num_masks = pred_masks.shape[1]
    gt_masks_expand = gt_masks.repeat(1, num_masks, 1, 1)             # [B, K, H, W]

    # 1. 二分类交叉熵损失 (BCE Loss)
    bce_loss = F.binary_cross_entropy_with_logits(pred_masks, gt_masks_expand)

    # 2. Dice 损失计算
    pred_probs = torch.sigmoid(pred_masks)
    intersection = torch.sum(pred_probs * gt_masks_expand, dim=(2, 3)) # [B, K]
    cardinality = torch.sum(pred_probs, dim=(2, 3)) + torch.sum(gt_masks_expand, dim=(2, 3)) # [B, K]
    dice_loss = 1.0 - (2.0 * intersection + smooth) / (cardinality + smooth)
    dice_loss = torch.mean(dice_loss)

    # 3. 实际真实 IoU 与预测 IoU 的 MSE 损失
    actual_iou = (intersection + smooth) / (cardinality - intersection + smooth)
    iou_mse_loss = F.mse_loss(pred_ious, actual_iou.detach())

    # 复合总损失
    total_loss = 20.0 * bce_loss + 1.0 * dice_loss + 1.0 * iou_mse_loss
    return total_loss


# ==============================================================================
# 8. 训练/推理逻辑与入口 (Training/Inference Execution)
# ==============================================================================
def main():
    print(f"[*] 使用设备: {DEVICE}")

    # 1. 初始化构建数据加载器 (DataLoader)
    dataset = get_synthetic_sam_dataset(num_samples=16, image_size=IMAGE_SIZE)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. 实例化 SAM 模型与优化器
    model = SegmentAnythingModel(image_size=IMAGE_SIZE, embed_dim=EMBED_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # 3. 训练循环
    model.train()
    print("[*] 开始 SAM 1 模型训练...")
    for epoch in range(EPOCHS):
        running_loss = 0.0

        for images, point_coords, point_labels, gt_masks in train_loader:
            images = images.to(DEVICE)           # [B, 3, H, W]
            point_coords = point_coords.to(DEVICE) # [B, 1, 2]
            point_labels = point_labels.to(DEVICE) # [B, 1]
            gt_masks = gt_masks.to(DEVICE)       # [B, 1, H, W]

            optimizer.zero_grad()

            # 前向传播 (输出 3 种不同抽象粒度的候选 Mask 及 IoU 评估)
            pred_masks, pred_ious = model(images, point_coords, point_labels, multimask_output=True)

            # 计算损失值
            loss = sam_loss(pred_masks, gt_masks, pred_ious)

            # 反向传播与优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}] | Total SAM Loss: {avg_loss:.4f}")

    print("[*] SAM 1 模型训练与前向测试完成！")


if __name__ == "__main__":
    main()