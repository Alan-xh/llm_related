# SAM 1 (Segment Anything Model v1) 技术架构与接口文档

## 1. 架构总览

SAM 1 采用了解耦的三阶段解构设计：**重型图像编码器 (Heavy Image Encoder)**、**轻量化提示编码器 (Prompt Encoder)** 以及 **双向交互掩码解码器 (Two-Way Mask Decoder)**。

```
                       ┌─────────────────────────┐
                       │   Input Image [B,3,H,W] │
                       └────────────┬────────────┘
                                    │
                                    ▼
                        [ ImageEncoderViT (4x) ]
                                    │
                                    ▼
                     Image Embeddings [B, 256, 16, 16]
                                    │
    ┌───────────────────────────────┴───────────────────────────────┐
    │                                                               │
    │    ┌──────────────────────────────┐                           │
    │    │ Point Prompts [B, N_pts, 2]  │                           │
    │    └──────────────┬───────────────┘                           │
    │                   │                                           │
    │                   ▼                                           │
    │         [ PromptEncoder ]                                     │
    │                   │                                           │
    │                   ▼                                           │
    │    Sparse Embeddings [B, N_pts, 256]                          │
    │                   │                                           │
    └───────────────────┼───────────────────────────────────────────┘
                        │
                        ▼
            [ TwoWayAttentionBlock ] ◄── (Tokens ◄─► Image Cross Attn)
                        │
                        ├──────────────────────────┐
                        ▼                          ▼
            [ Hypernetwork MLPs ]       [ Output Upscaling (ConvT) ]
                        │                          │
                        └────────────┬─────────────┘
                                     │ (Matrix Multiply / 点乘)
                                     ▼
                          Mask Logits [B, 3, 64, 64]
                                     │
                                     ▼ (Bilinear Upsample)
                           Final Masks [B, 3, 256, 256]

```

---

## 2. 张量 Shape 流动追踪 (Tensor Flow Table)

| 节点/模块 | 输入 Shape | 输出 Shape | 说明 / 维度变化原因 |
| --- | --- | --- | --- |
| **Input Image** | - | `[B, 3, 256, 256]` | 原始RGB图像输入 |
| **Patch Embedding** | `[B, 3, 256, 256]` | `[B, 256, 16, 16]` | Conv2d (kernel=16, stride=16) 进行 16 倍下采样 |
| **ViT Blocks** | `[B, 256, 16, 16]` | `[B, 256, 16, 16]` | 序列化打平为 `[B, 256, 256]` 进行自注意力计算后重构为特征图 |
| **Prompt Encoder** | `[B, N_pts, 2]` | `[B, N_pts, 256]` | 随机高斯位置编码 (PE) + 前/背景类型 Embedding 叠加 |
| **Tokens Assembly** | Tokens + Prompts | `[B, 5 + N_pts, 256]` | 拼接 1 个 IoU Token、4 个 Mask Tokens 与 N_pts 个 Prompt Tokens |
| **Two-Way Attention** | `Tokens`, `Image` | `[B, 5+N_pts, 256]`, `[B, 256, 256]` | Tokens 与 Image Flatten 特征循环进行自注意力与双向交叉注意力交互 |
| **Output Upscaling** | `[B, 256, 16, 16]` | `[B, 32, 64, 64]` | 转置卷积 (ConvTranspose2d 2x2) 进行 4 倍上采样提升空间分辨率 |
| **Hypernetwork MLPs** | `[B, 5, 256]` | `[B, 5, 32]` | 将解码后的 Mask Tokens 映射为卷积通道权重 |
| **Mask Logits Matmul** | `[B, 5, 32]`, `[B, 32, 4096]` | `[B, 5, 64, 64]` | 通过矩阵点乘计算最终点级预测分类结果 |
| **Mask Upsampling** | `[B, 3, 64, 64]` | `[B, 3, 256, 256]` | 截取前 3 通道 (Multimask) 并用双线性插值恢复至原始图像分辨率 |

---

## 3. 核心公式与代码映射

| 数学推导公式 | 代码变量 / 实现名称 | 位置与说明 |
| --- | --- | --- |
| $PE(x) = [\sin(2\pi B x), \cos(2\pi B x)]$ | `PositionEmbeddingRandom._pe_encoding` | 随机 Fourier 位置编码，用于稀疏点和密集特征图的位置赋予 |
| $A = \text{Softmax}\left(\frac{(Q + PE_q)(K + PE_k)^T}{\sqrt{d_k}}\right) V$ | `TwoWayAttentionBlock` 中各类 `nn.MultiheadAttention` | 显式将 Query 和 Key 添加对应的 Positional Encoding 后进行注意力矩阵运算 |
| $\mathcal{L}_{BCE} = -[y \log p + (1-y) \log(1-p)]$ | `F.binary_cross_entropy_with_logits` | 计算预测 Logits 与真实 Mask 间的逐像素交叉熵损失 |
| $\mathcal{L}_{Dice} = 1 - \frac{2 \sum p y + \epsilon}{\sum p + \sum y + \epsilon}$ | `sam_loss` 中的 `dice_loss` 计算 | 计算掩码的前景重叠面积与总体面积比值，缓解类别不平衡 |
| $\text{IoU}_{actual} = \frac{\vert{}P \cap Y\vert{}}{\vert{}P \cup Y\vert{}}$ | `sam_loss` 中的 `actual_iou` 表达式 | 利用矩阵计算真实 IoU 分数并与 `pred_ious` 计算 MSE 损失 |