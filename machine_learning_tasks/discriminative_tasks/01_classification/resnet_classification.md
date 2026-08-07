# ResNet18 图像分类 Pipeline 技术架构与接口文档

## 1. 架构总览

本架构严格遵照标准 PyTorch 模块化工程进行设计，基于 **ResNet18** 实现端到端 64x64 合成图像的 10 分类任务。模型核心通过残差跳跃连接（Residual Shortcut）实现跨层特征相加，克服深层网络梯度衰减。

```
[Input Tensor: B x 3 x 64 x 64]
               │
               ▼
   ┌──────────────────────┐
   │ Conv7x7, s=2, p=3    │ ───► [B, 64, 32, 32]
   │ BatchNorm + ReLU     │
   │ MaxPool3x3, s=2, p=1 │ ───► [B, 64, 16, 16]
   └──────────────────────┘
               │
               ▼
   ┌──────────────────────┐
   │ Layer 1 (2x Block)   │ ───► [B, 64, 16, 16]  (Stride=1)
   └──────────────────────┘
               │
               ▼
   ┌──────────────────────┐
   │ Layer 2 (2x Block)   │ ───► [B, 128, 8, 8]   (Stride=2 下采样)
   └──────────────────────┘
               │
               ▼
   ┌──────────────────────┐
   │ Layer 3 (2x Block)   │ ───► [B, 256, 4, 4]   (Stride=2 下采样)
   └──────────────────────┘
               │
               ▼
   ┌──────────────────────┐
   │ Layer 4 (2x Block)   │ ───► [B, 512, 2, 2]   (Stride=2 下采样)
   └──────────────────────┘
               │
               ▼
   ┌──────────────────────┐
   │ AdaptiveAvgPool2d    │ ───► [B, 512, 1, 1]
   │ Flatten              │ ───► [B, 512]
   │ Linear FC            │ ───► [B, 10]
   └──────────────────────┘
               │
               ▼
    [Logits Tensor: B x 10]

```

---

## 2. 张量 Shape 流动追踪 (Tensor Flow Table)

假设 Batch Size $B = 64$，输入图像尺寸为 $3 \times 64 \times 64$：

| 节点 / 模块名称 | 输入 Shape | 输出 Shape | 维度变化主要原因 / 计算说明 |
| --- | --- | --- | --- |
| **Input Data** | - | `[64, 3, 64, 64]` | 批次图像输入数据 |
| **Stem Conv1** | `[64, 3, 64, 64]` | `[64, 64, 32, 32]` | $7 \times 7$ 卷积，`stride=2`, `padding=3` 降低一半分辨率 |
| **Stem MaxPool** | `[64, 64, 32, 32]` | `[64, 64, 16, 16]` | $3 \times 3$ 最大池化，`stride=2`, `padding=1` 二次下采样 |
| **Layer 1** | `[64, 64, 16, 16]` | `[64, 64, 16, 16]` | 包含 2 个 BasicBlock，`stride=1`，保持维度不变 |
| **Layer 2** | `[64, 64, 16, 16]` | `[64, 128, 8, 8]` | 首个 Block `stride=2`，通道翻倍，高宽减半 |
| **Layer 3** | `[64, 128, 8, 8]` | `[64, 256, 4, 4]` | 首个 Block `stride=2`，通道翻倍，高宽减半 |
| **Layer 4** | `[64, 256, 4, 4]` | `[64, 512, 2, 2]` | 首个 Block `stride=2`，通道翻倍，高宽减半 |
| **AvgPool** | `[64, 512, 2, 2]` | `[64, 512, 1, 1]` | 自适应全局平均池化（Adaptive Avg Pooling） |
| **Flatten** | `[64, 512, 1, 1]` | `[64, 512]` | 展平非 Batch 维度，进入全连接层 |
| **Linear (FC)** | `[64, 512]` | `[64, 10]` | 线性映射到 10 个类别的未归一化分值 Logits |

---

## 3. 核心公式与代码映射

### 1. 残差加法映射 (Residual Connection)

* **理论公式**：

$$y = \mathcal{F}(x, \{W_i\}) + W_s x$$


* **代码实现 (`BasicBlock.forward`)**：
```python
identity = self.shortcut(x)  # 对应 W_s * x
out = self.conv2(out)  # 对应 F(x, {W_i})
out = out + identity  # 对应 + 运算

```



```

### 2. 多分类交叉熵损失 (Cross-Entropy Loss)
* **理论公式**：
  $$\mathcal{L} = -\log \left( \frac{\exp(z_{y})}{\sum_{j=1}^{K} \exp(z_j)} \right)$$
* **代码实现 (`main`)**：
  ```python
  criterion = nn.CrossEntropyLoss()
  loss = criterion(logits, labels)  # logits 对应 z, labels 对应类别索引 y

```