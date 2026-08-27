# CGAN-Time-Series 技术架构与接口文档

## 1. 架构总览

本架构用于解决**带条件约束的多维时间序列生成**问题，核心在于借助 Wasserstein GAN 搭配梯度惩罚（WGAN-GP）克服传统 GAN 在时间序列数据上的梯度消失与模式崩溃（Mode Collapse）问题。

```
                       ┌────────────────────────┐
                       │  Latent Noise z [B, Dz] │
                       └───────────┬────────────┘
                                   │
                                   ▼
┌────────────────────────┐  ┌─────────────┐
│ Condition c [B, L, Dc] ├─►│ Generator   ├─► Generated Sequence x̂ [B, L, Din]
└───────────┬────────────┘  │ (LSTM-G)    │             │
            │               └─────────────┘             │
            │                                           ▼
            │               ┌─────────────┐     ┌────────────────┐
            └──────────────►│Discriminator│◄────┤ Real Data x    │
                            │ (BiLSTM-D)  │     │ [B, L, Din]    │
                            └──────┬──────┘     └────────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ Score Score [B, 1]     │
                       └────────────────────────┘

```

* **Generator (条件生成器)**：首先将噪声 vector $z$ 与第 0 时间步的条件特征 $c_0$ 拼接，通过 MLP 映射后作为多层 LSTM 的初始隐藏状态 $h_0$。随后将整条条件序列 $C$ 输入 LSTM 进行逐步推理，最后对各时间步隐藏状态并行做 MLP 投影与 `Tanh` 激活，生成合成序列 $\hat{x}$。
* **Discriminator (条件判别器)**：将目标序列（真实 $x$ 或生成 $\hat{x}$）在特征维度上与条件序列 $C$ 进行直接拼接，送入双向 LSTM (Bi-LSTM) 捕获双向上下文关系，提取末位时间步的特征表示，最后经多层感知机（MLP）输出无界真实度标量得分（Logits）。

---

## 2. 张量 Shape 流动追踪 (Tensor Flow Table)

| 节点 / 模块 | 输入 Shape | 输出 Shape | 说明 / 维度变化原因 |
| --- | --- | --- | --- |
| **Data Sample (x, c)** | - | `[B, L, 10]`, `[B, L, 5]` | 从 Dataset 采样的真实序列与条件序列 |
| **Generator Init State** | `[B, 100]`, `[B, 5]` | `[3, B, 256]` | 拼接 $z$ 与 $c_0$ 经 MLP 映射，Expand 为 3 层 LSTM 的 initial $h_0$ |
| **Generator Recurrent Pass** | `[B, L, 5]`, `h0` | `[B, L, 256]` | 以 $c$ 为 Driving Sequence 在 LSTM 展开运算 |
| **Generator Output Head** | `[B, L, 256]` | `[B, L, 10]` | Flatten 到 `[B*L, 256]` 经 MLP + Tanh 映射后 Reshape |
| **Discriminator Combined** | `[B, L, 10]`, `[B, L, 5]` | `[B, L, 15]` | 特征维度拼接：`10 + 5 = 15` |
| **Discriminator BiLSTM** | `[B, L, 15]` | `[B, L, 512]` | 双向 LSTM，输出维度为 `2 * hidden_dim = 512` |
| **Discriminator Pooling** | `[B, L, 512]` | `[B, 512]` | 提取最后一个时间步特征 `last_out = lstm_out[:, -1, :]` |
| **Discriminator Score Head** | `[B, 512]` | `[B, 1]` | 经过 MLP + LeakyReLU 输出无界 Logit 得分 |
| **WGAN Interpolation** | `[B, L, 10]`, `[B, L, 10]` | `[B, L, 10]` | $\tilde{x} = \epsilon x + (1-\epsilon) \hat{x}$ 凸组合插值 |
| **Gradient Penalty** | `[B, L, 10]` (grads) | `[1]` | 展平计算 $L_2$ Norm，计算 $(\vert{}\vert{}\nabla_{\tilde{x}} D\vert{}\vert{}_2 - 1)^2$ 均值 |

---

## 3. 核心公式与代码映射

| 数学理论 / 算法公式 | 代码实现变量 / 方法 | 逻辑与设计说明 |
| --- | --- | --- |
| $\tilde{x} = \epsilon x + (1 - \epsilon)\hat{x}$ | `interpolated = epsilon * real_data + (1 - epsilon) * fake_data` | 在真实与生成数据间做随机凸组合用于梯度惩罚采样 |
| $\nabla_{\tilde{x}} D(\tilde{x} \mid c)$ | `torch.autograd.grad(outputs=disc_interpolated, inputs=interpolated, ...)` | 自动求导计算 Discriminator 对插值样本的梯度的 Jacobian 矩阵 |
| $\mathcal{L}_{GP} = \left(\Vert{}\nabla_{\tilde{x}} D\Vert{}_2 - 1\right)^2$ | `gradient_norm = gradients.norm(2, dim=1); penalty = ((gradient_norm - 1) ** 2).mean()` | 约束判别器满足 1-Lipschitz 连续条件，防止梯度爆炸/消失 |
| $\mathcal{L}_D = \mathbb{E}[D(\hat{x})] - \mathbb{E}[D(x)] + \lambda \mathcal{L}_{GP}$ | `d_loss_wasserstein = fake_validity.mean() - real_validity.mean(); total_d_loss = d_loss_wasserstein + gp_weight * gp` | WGAN 判别器 Wasserstein 距离极大化转化为极小化损失 |
| $\mathcal{L}_G = -\mathbb{E}[D(G(z \mid c))]$ | `g_loss = -fake_validity.mean()` | 生成器目标是最大化判别器对合成数据的评分 |
| $h_0 = \text{SiLU}(\text{BN}(\mathbf{W}[z \,\Vert{}\, c_0] + b))$ | `init_state = self.fc_input(torch.cat([z, cond[:, 0, :]], dim=-1))` | 将静态高斯噪声与时序初始条件注入 LSTM 的 Hidden State 初始空间 |