# U-Net 语义分割网络 (Semantic Segmentation) 技术架构与接口文档

## 1. 架构总览

U-Net 是一种基于全卷积神经网络 (FCN) 的对称 Encoder-Decoder 语义分割架构。它通过下采样提取多尺度上下文语义，再通过上采样与跨层通道拼接 (Skip Connection) 恢复图像的空间细节。

```
[Input Image: Bx3x128x128]
       │
       ▼
[Encoder Stage 1] ──── (Skip Connection 1) ───────────────┐
 (DoubleConv) -> Bx32x128x128                             │
       │                                                  │
   [MaxPool2d]                                            │
       ▼                                                  │
[Encoder Stage 2] ──── (Skip Connection 2) ──────┐        │
 (DoubleConv) -> Bx64x64x64                      │        │
       │                                         │        │
   [MaxPool2d]                                   │        │
       ▼                                         │        │
[Bottleneck Stage]                               │        │
 (DoubleConv) -> Bx128x32x32                     │        │
       │                                         │        │
[ConvTranspose2d] -> Bx64x64x64                  │        │
       │                                         │        │
       ├─────────────────────────────────────────┘        │
       ▼ (Concat)                                         │
 [Decoder Stage 2] -> Bx128x64x64                         │
 (DoubleConv) -> Bx64x64x64                               │
       │                                                  │
[ConvTranspose2d] -> Bx32x128x128                         │
       │                                                  │
       ├──────────────────────────────────────────────────┘
       ▼ (Concat)
 [Decoder Stage 1] -> Bx64x128x128
 (DoubleConv) -> Bx32x128x128
       │
  [1x1 Conv]
       ▼
 [Logits Output: Bx4x128x128]

```

---

## 2. 张量 Shape 流动追踪 (Tensor Flow Table)

假定配置参数为：`Batch_Size = 8`, `In_Channels = 3`, `Base_Channels = 32`, `Num_Classes = 4`, 图像高宽 $H=128, W=128$。

| 节点 / 模块 | 输入 Shape | 输出 Shape | 变换说明 / 维度变化原因 |
| --- | --- | --- | --- |
| **Input Image** | - | `[8, 3, 128, 128]` | 原始合成 RGB 图像输入 |
| **Encoder Stage 1 (`enc1`)** | `[8, 3, 128, 128]` | `[8, 32, 128, 128]` | 双层 3x3 卷积，通道由 3 拓展至 32，保持空间高宽 |
| **MaxPool1 (`pool1`)** | `[8, 32, 128, 128]` | `[8, 32, 64, 64]` | $2 \times 2$ 最大池化，空间高宽下采样减半 |
| **Encoder Stage 2 (`enc2`)** | `[8, 32, 64, 64]` | `[8, 64, 64, 64]` | 双层 3x3 卷积，通道由 32 拓展至 64 |
| **MaxPool2 (`pool2`)** | `[8, 64, 64, 64]` | `[8, 64, 32, 32]` | $2 \times 2$ 最大池化，空间高宽下采样减半 |
| **Bottleneck (`bottleneck`)** | `[8, 64, 32, 32]` | `[8, 128, 32, 32]` | 瓶颈层特征抽取，通道数达到峰值 128 |
| **Decoder Up2 (`up2`)** | `[8, 128, 32, 32]` | `[8, 64, 64, 64]` | 2x2 转置卷积，空间高宽翻倍，通道数减半 |
| **Skip Connection 2 (Concat)** | `[8, 64, 64, 64]` & `[8, 64, 64, 64]` | `[8, 128, 64, 64]` | 拼接 `up2` 特征与编码器 `e2` 特征 (`dim=1`) |
| **Decoder Conv2 (`dec2`)** | `[8, 128, 64, 64]` | `[8, 64, 64, 64]` | 双层 3x3 卷积，通道由 128 压缩回 64 |
| **Decoder Up1 (`up1`)** | `[8, 64, 64, 64]` | `[8, 32, 128, 128]` | 2x2 转置卷积，空间高宽恢复至原始 128x128 |
| **Skip Connection 1 (Concat)** | `[8, 32, 128, 128]` & `[8, 32, 128, 128]` | `[8, 64, 128, 128]` | 拼接 `up1` 特征与编码器 `e1` 浅层高精特征 (`dim=1`) |
| **Decoder Conv1 (`dec1`)** | `[8, 64, 128, 128]` | `[8, 32, 128, 128]` | 双层 3x3 卷积，通道由 64 恢复至 32 |
| **Out Head (`out_conv`)** | `[8, 32, 128, 128]` | `[8, 4, 128, 128]` | 1x1 卷积逐像素通道映射，生成对应 4 类的 Logits |

---

## 3. 核心公式与代码映射

### 3.1 多分类 Dice Loss

数学推导公式：


$$L_{\text{Dice}} = 1 - \frac{1}{C} \sum_{c=0}^{C-1} \frac{2 \sum_{i,j} p_{i,j,c} \cdot y_{i,j,c} + \epsilon}{\sum_{i,j} p_{i,j,c} + \sum_{i,j} y_{i,j,c} + \epsilon}$$

变量与代码实现对应表：

| 公式符号 | 含义描述 | 对应代码变量 | 代码表达式 / 逻辑 |
| --- | --- | --- | --- |
| $p_{i,j,c}$ | 像素 $(i,j)$ 预测为类别 $c$ 的概率 | `probs` | `probs = F.softmax(pred, dim=1)` |
| $y_{i,j,c}$ | 像素 $(i,j)$ 类别 $c$ 的真实 One-Hot 标签 | `target_one_hot` | `F.one_hot(target, C).permute(0, 3, 1, 2)` |
| $\sum p \cdot y$ | 预测概率与 One-Hot 的重叠交集 | `intersection` | `torch.sum(probs * target_one_hot, dim=(2,3))` |
| $\sum p + \sum y$ | 预测与真实标签的总和 (并集分母) | `cardinality` | `torch.sum(probs, (2,3)) + torch.sum(target_one_hot, (2,3))` |
| $\epsilon$ | 平滑因子（防止除零） | `smooth` | 函数形参 `smooth=1.0` |
| $L_{\text{Dice}}$ | 最终均值 Dice 损失标量 | `loss` | `1.0 - torch.mean((2*intersection + smooth)/(cardinality + smooth))` |