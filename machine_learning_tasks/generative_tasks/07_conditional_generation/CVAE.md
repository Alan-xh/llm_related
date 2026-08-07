# 条件时序变分自编码器 (TimeSeriesCVAE) 技术架构与接口文档

## 1. 架构总览与数据流

本系统实现了一个**条件时序变分自编码器 (Conditional Time-Series VAE)**，专用于在给定外部控制序列/条件变量（Condition Sequence $c_{1:T}$）的前提下，对多维时序数据（Time-Series Sequence $x_{1:T}$）进行概率建模、重构与可控生成。

### 数据流图示 (Data Flow Diagram)

```
[输入序列 X: B x T x D_x] ───┐
                          ├──> [Concat] ──> [BiLSTM Encoder] ──> [Last Hidden] ──┬─> [FC_mu] ────> μ  ──┐
[条件序列 C: B x T x D_c] ───┴────────────────────────────────────────────────────┼─> [FC_logvar] ─> σ² ─┴─> [Reparameterize: z = μ + ε⊙σ]
                                                                                  │                                      │
[首帧条件 c_0: B x D_c] ─────────────────────────────────────────────────────────┼──────────────────────────────────────┘
                                                                                  ▼
                                                                [Concat (z, c_0)] ──> [FC_init] ──> State (h_0, c_0)
                                                                                                          │
[完整条件 C: B x T x D_c] ────────────────────────────────────────────────────────────────────────────────┼─> [UniLSTM Decoder]
                                                                                                          │          │
                                                                                                          ▼          ▼
[重构/生成序列 X_hat: B x T x D_x] <───────────────────────────────────────────────────────────────── [FC_out] <─────┘

```

---

## 2. 张量 Shape 流动追踪 (Tensor Flow Table)

| 节点/模块 (Node/Module) | 输入 Shape | 输出 Shape | 说明 / 维度变化原因 |
| --- | --- | --- | --- |
| **Input x** | - | `[B, T, D_x]` | 原始输入特征序列 (`D_x = input_dim`) |
| **Input cond** | - | `[B, T, D_c]` | 控制条件序列 (`D_c = cond_dim`) |
| **Encoder Concatenation** | `[B, T, D_x]`, `[B, T, D_c]` | `[B, T, D_x + D_c]` | 特征维度拼接 (`dim=-1`) |
| **BiLSTM Output** | `[B, T, D_x + D_c]` | `[B, T, 2 * H]` | 双向 LSTM 编码，隐藏维度双倍 (`H = hidden_dim`) |
| **Encoder Last State Extraction** | `[B, T, 2 * H]` | `[B, 2 * H]` | 提取最后一个时间步作为序列高阶上下文 `[:, -1, :]` |
| **FC_mu / FC_logvar** | `[B, 2 * H]` | `[B, Latent_Dim]` | 投影映射到高斯分布均值 $\mu$ 与对数方差 $\log(\sigma^2)$ |
| **Reparameterization Sampling** | `[B, Latent_Dim]`, `[B, Latent_Dim]` | `[B, Latent_Dim]` | 随机采样 $z = \mu + \epsilon \odot \sigma$ |
| **Decoder State Initializer** | `[B, Latent_Dim + D_c]` | `[B, 2 * H]` | 拼接潜在向量 $z$ 与首帧条件 $c_0$，映射出隐藏状态 |
| **Decoder Hidden States ($h_0, c_0$)** | `[B, 2 * H]` | `[Num_Layers, B, H]` | 切分拆解为 $h_0, c_0$ 并广播重构成多层 LSTM 状态 |
| **UniLSTM Decoder Output** | `[B, T, D_c]`, State | `[B, T, H]` | 以条件序列为驱动进行单向自回归解码 |
| **FC Output Head** | `[B, T, H]` | `[B, T, D_x]` | 线性映射还原重构序列特征维度 $x_{hat}$ |

---

## 3. 核心公式与代码映射 (Theory to Code Mapping)

### 3.1 变分下界与损失函数 (Variational Bound & Loss)

* **数学公式**:

$$\mathcal{L}_{\text{total}} = \frac{1}{B} \sum_{i=1}^B \Vert{}x_i - \hat{x}_i\Vert{}_F^2 - \frac{\beta}{2B} \sum_{i=1}^B \sum_{j=1}^{d_z} \left( 1 + \log(\sigma_{ij}^2) - \mu_{ij}^2 - \sigma_{ij}^2 \right)$$


* **代码映射**:
* **重构损失 ($L_{\text{recon}}$)**:
`recon_loss = nn.MSELoss(reduction='sum')(recon_x, x) / x.shape[0]`
* **KL 散度 ($L_{\text{KL}}$)**:
`kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]`
* **总损失**:
`loss = recon_loss + beta * kl_loss`



### 3.2 重参数化技巧 (Reparameterization Trick)

* **数学公式**:

$$\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \quad \sigma = \exp\left(\frac{1}{2}\log(\sigma^2)\right), \quad z = \mu + \epsilon \odot \sigma$$


* **代码映射**:
```python
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
z = mu + eps * std

```



### 3.3 Dynamic KL-Annealing 策略

* **数学公式**:

$$\beta_e = \min\left(1.0, \frac{e + 1}{E_{\text{anneal}}}\right) \cdot \beta_{\text{target}}$$


* **代码映射**:
```python
def get_beta(epoch):
    if epoch < config.kl_annealing_steps:
        return config.beta * (epoch + 1) / config.kl_annealing_steps
    return config.beta

```