# Conditional Time-Series Transformer Diffusion (CTTD) 技术架构与接口文档

## 1. 架构总览

本模型实现了基于 **Transformer Backbone** 与 **Classifier-Free Guidance (CFG)** 的条件时间序列生成扩散模型。架构整体分为 **前向加噪过程 (Forward Diffusion)** 与 **反向去噪生成 (Reverse Denoising Process)**。

```
[无噪序列 x_0] ---> (q_sample + 高斯噪声 ε) ---> [加噪序列 x_t]
                                                     |
[条件序列 c]  ---> [Condition Encoder] ------------> + ---> [Transformer Encoder] ---> [Linear Out] ---> [预测噪声 ε_θ]
                                                     |
[时间步 t]    ---> [Sinusoidal PosEmb + MLP] ------->

```

1. **Conditioning Integration (条件融合)**：采用加性融合（Additive Conditioning Integration）策略，将输入序列投影特征 $H \in \mathbb{R}^{B \times L \times D}$、时间编码 $E_t \in \mathbb{R}^{B \times 1 \times D}$ 以及条件特征 $E_c \in \mathbb{R}^{B \times L \times D}$ 在隐空间无缝叠加。
2. **Backbone (Transformer Encoder)**：利用标准 Self-Attention 捕获长程时间序列依赖与不同特征维度间的动态漂移相关性。
3. **CFG (Classifier-Free Guidance)**：训练期以概率 $p_{\text{drop}}$ 将条件置零，推理期支持根据 Scale $w$ 对条件预测方向进行外推强化。

---

## 2. 张量 Shape 流动追踪 (Tensor Flow Table)

| 节点/模块 (Node/Module) | 输入 Shape | 输出 Shape | 说明 / 维度变化原因 |
| --- | --- | --- | --- |
| **Input: $x_0$** | - | `[B, 50, 10]` | 原始目标时间序列 (Seq_Len=50, Input_Dim=10) |
| **Input: $c$** | - | `[B, 50, 5]` | 控制条件序列 (Cond_Dim=5) |
| **Input: $t$** | - | `[B]` | 随机扩散时间步区间 $[0, T-1]$ |
| **$q\_sample$** | `[B, 50, 10]`, `[B]` | `[B, 50, 10]` | 解析解一步加噪输出 $x_t$ |
| **Input Proj** | `[B, 50, 10]` | `[B, 50, 128]` | 将原始通道数映射到模型隐层特征维度 (`model_dim=128`) |
| **Time Pos Emb** | `[B]` | `[B, 256]` | 256 维正弦位置编码 |
| **Time MLP** | `[B, 256]` | `[B, 128]` | 多层感知机特征投影为 $E_t$ |
| **Cond Encoder** | `[B, 50, 5]` | `[B, 50, 128]` | 将条件特征升维映射为 $E_c$ |
| **Feature Addition** | `x_proj + t_emb + c_emb` | `[B, 50, 128]` | 广播加法融合：`[B, 50, 128] + [B, 1, 128] + [B, 50, 128]` |
| **Transformer Block** | `[B, 50, 128]` | `[B, 50, 128]` | 4 层 Transformer Encoder，Self-Attention 依赖建模 |
| **Output Proj** | `[B, 50, 128]` | `[B, 50, 10]` | 线性重构映射回预测噪声 $\hat{\varepsilon}$ Dimensions |

---

## 3. 核心公式与代码映射

| 数学原理 / 概念 | 理论推导公式 | 代码变量 / 实现函数映射 |
| --- | --- | --- |
| **前向扩散加噪** | $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \varepsilon$ | `ConditionalDiffusionProcess.q_sample()` |
| **正弦时间步编码** | $PE_{(t, 2i)} = \sin\left(\frac{t}{10000^{2i/d}}\right)$ | `SinusoidalPosEmb.forward()` |
| **反向采样均值估计** | $\mu_\theta = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \varepsilon_\theta(x_t, t, c) \right)$ | `ConditionalDiffusionProcess.p_sample()` |
| **Classifier-Free Guidance** | $\tilde{\varepsilon}_\theta(x_t, c) = \varepsilon_\theta(x_t, \emptyset) + w \cdot (\varepsilon_\theta(x_t, c) - \varepsilon_\theta(x_t, \emptyset))$ | `ConditionalDiffusionProcess.p_sample(cfg_scale > 1.0)` |
| **优化目标 (MSE Loss)** | $\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \varepsilon} \left[ \Vert{}\varepsilon - \varepsilon_\theta(x_t, t, c)\Vert{}^2 \right]$ | `train()` 中的 `nn.MSELoss()(predicted_noise, noise)` |