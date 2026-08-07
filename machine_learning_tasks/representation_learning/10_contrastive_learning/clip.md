# CLIP 双塔对比学习模型 技术架构与接口文档

## 1. 架构总览

CLIP (Contrastive Language-Image Pre-training) 是一种典型的双塔（Two-Tower）跨模态表示学习架构。模型由独立的 **图像编码器 (Image Encoder)** 与 **文本编码器 (Text Encoder)** 组成，通过将两种模态的数据分别映射到统一维度的连续高维向量空间，并利用 L2 范数归一化与余弦相似度构建关联。

```
 [Image: B, 3, 64, 64]                  [Text: B, 32]
          │                                   │
   (CNN Backbone)                        (Embedding)
          │                                   │
   (Global Pooling)                        (BiGRU)
          │                                   │
   (Linear Proj)                        (Linear Proj)
          │                                   │
   [z_I: B, 128]  ─── (L2 Norm) ───►  [z_I_norm: B, 128]
                                              │
                                              ├───►  [ Logits Matrix: B x B ]
                                              │       (Scale * z_I @ z_T^T)
   [z_T: B, 128]  ─── (L2 Norm) ───►  [z_T_norm: B, 128]
                                              │
                                     (Symmetric InfoNCE)

```

---\n

## 2. 张量 Shape 流动追踪 (Tensor Flow Table)

| 节点 / 模块 | 输入 Shape | 输出 Shape | 说明 / 维度变化原因 |
| --- | --- | --- | --- |
| **Input Images** | - | `[B, 3, 64, 64]` | 原始图像 Batch 批量数据 |
| **Input Texts** | - | `[B, 32]` | 原始文本 Token 序号索引矩阵 |
| **Image Backbone (ConvNet)** | `[B, 3, 64, 64]` | `[B, 512, 4, 4]` | 经过 4 层 Stride=2 的卷积层完成 16x 空间下采样并扩展通道 |
| **Image Pooling & Flatten** | `[B, 512, 4, 4]` | `[B, 512]` | 自适应全局均值池化降为 `[B, 512, 1, 1]` 后按维度 1 展平 |
| **Image Projection** | `[B, 512]` | `[B, 128]` | 线性变换映射至 128 维共享跨模态空间 |
| **Image L2 Normalize** | `[B, 128]` | `[B, 128]` | 按最后一维特征做 $L2$ 归一化 $\frac{z}{\Vert{}z\Vert{}_2}$ |
| **Text Embedding** | `[B, 32]` | `[B, 32, 256]` | 词表查找映射（Lookup Table） |
| **Text BiGRU Layer** | `[B, 32, 256]` | `[B, 32, 512]` | 双向 GRU 特征提取，隐层维度 $256 \times 2 = 512$ |
| **Text Pooling (Last Token)** | `[B, 32, 512]` | `[B, 512]` | 截取序列最后一个时间步的时序表示 $H_{:, -1, :}$ |
| **Text Projection & L2 Norm** | `[B, 512]` | `[B, 128]` | 线性投影并做 $L2$ 归一化 |
| **Similarity Matrix Calculation** | `[B, 128]`, `[B, 128]` | `[B, B]` | 矩阵乘法 $S = \text{scale} \cdot (z_I \cdot z_T^T)$ 计算对偶相似度度量 |

---

## 3. 核心公式与代码映射

### 3.1 可学习温度系数 $\tau$ 与 Logit 缩放因子

* **数学公式**:

$$\text{Scale} = \exp(\text{logit\_scale}) = \frac{1}{\tau}$$


* **代码映射**:
```python
self.logit_scale = nn.Parameter(
    torch.ones([]) * math.log(1.0 / init_temperature)
)
logit_scale = self.logit_scale.exp()

```



```

### 3.2 跨模态余弦相似度矩阵
* **数学公式**:
  $$S_{i,j} = \text{Scale} \cdot \langle \hat{z}_i^I, \hat{z}_j^T \rangle$$
* **代码映射**:
  ```python
  image_features = F.normalize(self.proj(flat), p=2, dim=-1)
  text_features = F.normalize(self.proj(last_hidden), p=2, dim=-1)
  logits_per_image = logit_scale * image_features @ text_features.t()

```

### 3.3 对称 InfoNCE 对比损失

* **数学公式**:

$$\mathcal{L}_{I \to T} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp(S_{i,i})}{\sum_{j=1}^B \exp(S_{i,j})}$$


* **代码映射**:
```python
labels = torch.arange(batch_size, device=logits_per_image.device)
loss_i = F.cross_entropy(logits_per_image, labels)
loss_t = F.cross_entropy(logits_per_text, labels)
loss = (loss_i + loss_t) / 2.0

```



```

```