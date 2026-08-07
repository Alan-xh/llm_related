# 变分自编码器 (Variational Autoencoder, VAE) 技术架构与接口文档

## 1. 架构总览

变分自编码器 (VAE) 结合了深度学习与概率图模型推断。模型由 **Encoder（编码器）**、**Reparameterization Trick（重参数化模块）** 和 **Decoder（解码器）** 三部分组成：

```
+-----------------------------------------------------------------------------------------+
|                                    VAE Data Flow                                        |
+-----------------------------------------------------------------------------------------+

 [Input Data: x]               [Encoder Network]                    [Latent Distributions]
 [B, 1, 28, 28]              Linear -> SiLU -> Linear             Mean \mu  : [B, Latent_Dim]
       │                                │                         LogVar    : [B, Latent_Dim]
       ▼                                ▼                                  │
[Flatten Layer]  ─────────►  [Feature Vector h]  ──────────►  [Predict \mu & \log(\sigma^2)]
 [B, 784]                    [B, Hidden_Dim2]                              │
                                                                           ▼
                                                                  [Reparameterization]
                                                                  \epsilon ~ N(0, I)
                                                                  z = \mu + \exp(0.5*\log(\sigma^2)) * \epsilon
                                                                           │
                                                                           ▼
 [Output Recon: \hat{x}]       [Decoder Network]                  [Sampled Latent z]
  [B, 1, 28, 28]             Linear -> SiLU -> Linear             [B, Latent_Dim]
       ▲                                ▲                                  │
       │                                │                                  │
[Reshape Layer]  ◄─────────  [Reconstructed Features]  ◄───────────────────┘
  [B, 784]                   Linear(Tanh)

```

1. **编码阶段**: 图像展平为特征向量进入 Encoder，经过两个全连接层及 `SiLU` 激活函数映射为高斯隐分布参数：均值 $\mu$ 与对数方差 $\log(\sigma^2)$。
2. **重参数化阶段**: 从标准高斯分布 $\mathcal{N}(0, I)$ 采样噪声 $\epsilon$，通过可导公式 $z = \mu + \sigma \odot \epsilon$ 生成隐变量 $z$，保证梯度顺畅回传。
3. **解码阶段**: 隐变量 $z$ 通过 Decoder 网络升维重建，经过 `Tanh` 激活输出与原始归一化图像范围保持一致的重建特征。

---

## 2. 张量 Shape 流动追踪 (Tensor Flow Table)

默认参数设为: Batch Size $B = 128$, Latent Dim $D_z = 20$, Flatten Size $D_x = 784$。

| 节点 / 模块名称 | 输入 Shape | 输出 Shape | 变换说明 / 维度变化原因 |
| --- | --- | --- | --- |
| **Data Input** | `[128, 1, 28, 28]` | `[128, 1, 28, 28]` | 原始 MNIST 批次张量 |
| **Input Flatten** | `[128, 1, 28, 28]` | `[128, 784]` | 将二维图像矩阵展平为一维特征向量 ($28 \times 28 = 784$) |
| **Encoder Feature Extractor** | `[128, 784]` | `[128, 200]` | 连续通过 `Linear(784, 400)` $\rightarrow$ `SiLU` $\rightarrow$ `Linear(400, 200)` $\rightarrow$ `SiLU` |
| **Encoder Mu Layer** | `[128, 200]` | `[128, 20]` | 线性映射得到高斯分布均值 $\mu$ |
| **Encoder LogVar Layer** | `[128, 200]` | `[128, 20]` | 线性映射得到高斯分布对数方差 $\log(\sigma^2)$ |
| **Reparameterization Trick** | `[128, 20]` ($\mu$, $\log\sigma^2$) | `[128, 20]` ($z$) | 采样随机噪声 $\epsilon \sim \mathcal{N}(0, I)$，按 $z = \mu + e^{0.5 \cdot \log\sigma^2} \odot \epsilon$ 合成 |
| **Decoder Net** | `[128, 20]` | `[128, 784]` | 连续通过 `Linear(20, 200)` $\rightarrow$ `SiLU` $\rightarrow$ `Linear(200, 400)` $\rightarrow$ `SiLU` $\rightarrow$ `Linear(400, 784)` $\rightarrow$ `Tanh` |
| **Reshape Reconstruction** | `[128, 784]` | `[128, 1, 28, 28]` | 将展平后的预测特征向量重构为二维图像张量格式 |

---

## 3. 核心公式与代码映射

为了提升可读性与公式追溯效率，下表整理了理论公式与代码实现的映射关系：

| 理论概念 | 物理 / 数学推导公式 | 代码变量 / 逻辑实现 |
| --- | --- | --- |
| **方差推导 (Standard Deviation)** | $\sigma = \exp\left(\frac{1}{2} \log(\sigma^2)\right)$ | `std = torch.exp(0.5 * logvar)` |
| **随机采样 (Standard Normal)** | $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ | `eps = torch.randn_like(std)` |
| **重参数化采样 (Sampling)** | $z = \mu + \sigma \odot \epsilon$ | `z = mu + eps * std` |
| **重建误差 (Recon Loss)** | $\mathcal{L}_{\text{recon}} = \sum_{i=1}^{D} (x_i - \hat{x}_i)^2$ | `recon_loss = F.mse_loss(recon_x, x_flat, reduction='sum')` |
| **KL 散度 (KL Loss)** | $D_{\text{KL}}(q_\phi(z\Vert{}x) \parallel p(z)) = -\frac{1}{2} \sum_{j=1}^{d} \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$ | `kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())` |
| **总体优化目标 (ELBO Loss)** | $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{kl}}$ | `total_loss = recon_loss + kl_loss` |