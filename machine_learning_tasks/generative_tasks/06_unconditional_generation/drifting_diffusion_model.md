# Drifting Diffusion Model 技术架构与接口文档

## 1. 架构总览

本模型实现了用于连续时间序列预测与动态漂移轨迹生成的扩散模型（Diffusion Model）。系统整体分为三大核心模块：

1. **Diffusion Process（前向加噪与参数管理器）**：基于 DDPM 理论预计算前向 Markov 链加噪系数 $\beta_t, \alpha_t, \bar{\alpha}_t$。在训练阶段，直接采用 Closed-form 计算任意步数 $t$ 的加噪结果 $x_t$；在采样阶段，执行 $T$ 步迭代单步去噪。
2. **DriftingModel（Transformer 噪声预测主干）**：网络输入包含加噪后的连续序列 $x_t$ 与扩散时间步 $t$。通过正弦位置编码（Sinusoidal Positional Embedding）与 MLP 提取时间步特征，将时间上下文广播叠加至输入序列特征后，送入多层 `TransformerEncoder` 进行全序列长程依赖提取，最后经过线性层映射输出预测噪声 $\epsilon_\theta$。
3. **Training & Loss Optimization（训练管道）**：使用标准 MSE 损失匹配真实加性噪声 $\epsilon$ 与预测噪声 $\epsilon_\theta$，支持 Cosine 学习率退火、梯度裁剪以及 TensorBoard/日志全流程监控。

### 数据流图示 (Data Flow Diagram)

```
[原始序列 x0] (B, L, C_in) ---+
                              |
                     (q_sample 加噪) ---> [加噪序列 xt] (B, L, C_in) --+
                              |                                          |
[采样噪声  ε ] (B, L, C_in) ---+                                          |
                                                                         v
[时间步  t  ] (B) ----> [Sinusoidal Embed + MLP] -> (B, 1, C_model) -> [DriftingModel]
                                                                         |
                                                                         v
[MSE Loss] <--- (计算 L2 损失) <---------------------------------- [预测噪声 ε_θ] (B, L, C_in)

```

---

## 2. 张量 Shape 流动追踪 (Tensor Flow Table)

| 节点/模块 | 输入 Shape | 输出 Shape | 说明 / 维度变化原因 |
| --- | --- | --- | --- |
| **Input Data ($x_0$)** | - | `[B, Seq_Len, Input_Dim]` | 批次原始时间序列 |
| **Timestep Sampling ($t$)** | - | `[B]` | 在 $[0, T-1]$ 范围采样的整数时间步 |
| **Noise Addition ($q\_sample$)** | $x_0$: `[B, Seq_Len, Input_Dim]`<br>

<br>$t$: `[B]` | `[B, Seq_Len, Input_Dim]` | 融合高斯噪声后的 $x_t$ 张量 |
| **Time Embedding (`time_emb_func`)** | `[B]` | `[B, Time_Emb_Dim]` | 正弦位置编码维度扩充 |
| **Time MLP (`time_mlp`)** | `[B, Time_Emb_Dim]` | `[B, Model_Dim]` | 非线性升维变换，对其与主干模型维度 |
| **Input Projection (`input_proj`)** | `[B, Seq_Len, Input_Dim]` | `[B, Seq_Len, Model_Dim]` | 物理维度映射至 Transformer 隐层维度 |
| **Time Feature Broadcast** | $h$: `[B, Seq_Len, Model_Dim]`<br>

<br>$t_{emb}$: `[B, 1, Model_Dim]` | `[B, Seq_Len, Model_Dim]` | `unsqueeze(1)` 后沿序列维度广播并相加 |
| **Transformer Encoder** | `[B, Seq_Len, Model_Dim]` | `[B, Seq_Len, Model_Dim]` | 多头注意力与 FFN 提取序列时序依赖 |
| **Output Projection (`output_proj`)** | `[B, Seq_Len, Model_Dim]` | `[B, Seq_Len, Input_Dim]` | 恢复至原始物理维度，输出预测噪声 $\epsilon_\theta$ |
| **Loss Computation** | $\epsilon_\theta$: `[B, Seq_Len, Input_Dim]`<br>

<br>$\epsilon$: `[B, Seq_Len, Input_Dim]` | Scalar `()` | 计算 MSE Loss 进行梯度反向传播 |

---

## 3. 核心公式与代码映射

| 数学原理 / 目标 | 理论公式 | 代码实现映射 |
| --- | --- | --- |
| **前向加噪公式** | $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$ | `xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise` |
| **时间步正弦编码** | $PE_{(t, 2i)} = \sin\left(\frac{t}{10000^{2i/d}}\right)$ | `emb = torch.exp(torch.arange(half_dim) * -math.log(10000.0) / (half_dim - 1))` |
| **优化目标** | $\mathcal{L}_{simple}(\theta) = \Vert{}\epsilon - \epsilon_\theta(x_t, t)\Vert{}^2$ | `loss = nn.MSELoss()(predicted_noise, noise)` |
| **反向均值估计** | $\mu_\theta = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta \right)$ | `xt_prev = sqrt_recip_alpha_t * (xt - beta_t / torch.sqrt(1.0 - alpha_bar_t) * predicted_noise)` |
| **重参数化采样** | $x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z, \quad z \sim \mathcal{N}(0, \mathbf{I})$ | `xt_prev = ... + sigma_t * noise` |