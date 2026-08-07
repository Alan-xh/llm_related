# Vanilla GAN 图像生成技术架构与接口文档

## 1. 架构总览与数据流

Vanilla GAN 基于零和博弈理论，由**生成器 (Generator, G)** 与 **判别器 (Discriminator, D)** 两个核心网络相互对抗组成：

```
[ Random Noise z ~ N(0, I) ] ──> [ Generator (MLP) ] ──> Fake Image G(z) ──┐
                                                                           ├──> [ Discriminator (MLP) ] ──> Sigmoid Output [0,1]
[ Real Image x ~ p_data    ] ──────────────────────────────────────────────┘

```

* **生成流 (Generator Flow)**:
随机向量 $z \in \mathbb{R}^{100}$ 依次通过 4 层全连接块，特征通道按 `128 -> 256 -> 512 -> 1024` 逐步扩展，最后经线性层与 `Tanh` 映射展平成 shape 为 `[1, 64, 64]` 的伪造图像。
* **判别流 (Discriminator Flow)**:
真实图像 $x$ 或假图像 $G(z)$ 被展平为向量 $\mathbb{R}^{4096}$，通过 2 层隐藏全连接层（降维 `512 -> 256`），最终映射为标量预测值，代表输入张量为真实图像的概率。

---

## 2. 张量 Shape 流动追踪 (Tensor Flow Table)

假定配置为：`Batch_Size (B) = 128`, `Latent_Dim = 100`, `Channels (C) = 1`, `H = W = 64`（即单通道像素数 $64 \times 64 = 4096$）。

### 2.1 生成器 (Generator) 张量流动

| 阶段 / 模块 | 输入 Shape | 输出 Shape | 操作与维度变换说明 |
| --- | --- | --- | --- |
| **Input Noise ($z$)** | - | `[128, 100]` | 标准正态分布高斯噪声采样的 latent 向量 |
| **Linear Block 1** | `[128, 100]` | `[128, 128]` | Linear(100, 128) + LeakyReLU(0.2) |
| **Linear Block 2** | `[128, 128]` | `[128, 256]` | Linear(128, 256) + BatchNorm1d + LeakyReLU |
| **Linear Block 3** | `[128, 256]` | `[128, 512]` | Linear(256, 512) + BatchNorm1d + LeakyReLU |
| **Linear Block 4** | `[128, 512]` | `[128, 1024]` | Linear(512, 1024) + BatchNorm1d + LeakyReLU |
| **Output FC** | `[128, 1024]` | `[128, 4096]` | Linear(1024, 4096) + Tanh() |
| **Reshape** | `[128, 4096]` | `[128, 1, 64, 64]` | `.view(B, C, H, W)` 重构图像几何空间结构 |

### 2.2 判别器 (Discriminator) 张量流动

| 阶段 / 模块 | 输入 Shape | 输出 Shape | 操作与维度变换说明 |
| --- | --- | --- | --- |
| **Input Image ($x$)** | - | `[128, 1, 64, 64]` | 真实图像或生成器产出的假图像 |
| **Flatten** | `[128, 1, 64, 64]` | `[128, 4096]` | `.view(B, -1)` 展平 2D 像素阵列为 1D 向量 |
| **Linear Stage 1** | `[128, 4096]` | `[128, 512]` | Linear(4096, 512) + LeakyReLU(0.2) |
| **Linear Stage 2** | `[128, 512]` | `[128, 256]` | Linear(512, 256) + LeakyReLU(0.2) |
| **Output FC** | `[128, 256]` | `[128, 1]` | Linear(256, 1) + Sigmoid()，标量概率 |

---

## 3. 核心公式与代码映射

| 数学理论 / Objective | 数学表达 / Formula | 代码实现变量 / 算子映射 |
| --- | --- | --- |
| **二元交叉熵损失 (BCE)** | $\ell(y, \hat{y}) = - [y \log \hat{y} + (1 - y) \log (1 - \hat{y})]$ | `adversarial_loss = nn.BCELoss()` |
| **判别器真图 Loss ($L_{D,real}$)** | $-\log D(x)$ | `d_real_loss = adversarial_loss(discriminator(real_imgs), valid)` |
| **判别器假图 Loss ($L_{D,fake}$)** | $-\log (1 - D(G(z)))$ | `d_fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)` |
| **判别器总 Loss ($L_D$)** | $\frac{1}{2} (L_{D,real} + L_{D,fake})$ | `d_loss = (d_real_loss + d_fake_loss) / 2` |
| **生成器欺骗 Loss ($L_G$)** | $-\log D(G(z))$ | `g_loss = adversarial_loss(discriminator(gen_imgs), valid)` |
| **梯度截断控制** | $\nabla_{\theta_G} L_D = 0$ | `gen_imgs.detach()` (避免判别器更新时计算生成器梯度) |