# Conditional Flow Matching (CFM) 技术架构与接口文档

## 1. 架构总览

Flow Matching 是一类基于连续时间生成模型（Continuous Normalizing Flows, CNF）的新型生成范式。本工程实现了**条件 Flow Matching (Conditional Flow Matching, CFM)** 架构，能够根据指定的类别标签 $y$ 生成对应的图像。

### 数据流与推理路径

```
[先验噪声 x0 ~ N(0, I)] + [类别标签 y] + [时间步 t=0]
          │
          ▼
    ┌──────────┐
    │  ODE 求解 │◄────── [网络预测条件向量场 v_theta(x_t, t, y)]
    │  (Euler) │
    └────┬─────┘
         │  (沿着轨迹积分 t: 0.0 ───► 1.0)
         ▼
[最终生成图像 x1 ~ q(x1|y)]

```

* **训练阶段**：直接通过数据点 $x_1$ 和高斯噪声点 $x_0$ 构造线性概率路径 $x_t = (1-t)x_0 + t x_1$。模型 $v_\theta(x_t, t, y)$ 以均方误差（MSE）回归目标方向向量场 $u_t = x_1 - x_0$。无需模拟完整的 ODE 求解路径即可实现高效并行训练。
* **采样阶段**：在初始 $t=0$ 时采样高斯噪声 $x_0$，使用 Euler 步进法，沿着预测的向量场步进 $50$ 步推演至 $t=1$，获得高保真图像。

---

## 2. 张量 Shape 流动追踪 (Tensor Flow Table)

以下为以 `ConditionalMLP` 为例，在一个 Batch 训练过程中的张量维度流动变化表（假设 Batch Size $B=128$, 图像大小 $1 \times 28 \times 28$）：

| 节点 / 模块 | 输入 Shape | 输出 Shape | 说明 / 维度变化原因 |
| --- | --- | --- | --- |
| **Input Data (x1)** | `[128, 1, 28, 28]` | `[128, 784]` | 目标图像，展平为一维向量 |
| **Noise Data (x0)** | `[128, 784]` | `[128, 784]` | 标准高斯噪声 $N(0, I)$ |
| **Timestep Sampling (t)** | - | `[128, 1]` | 采样 $t \sim U[0, 1]$ |
| **Path Interpolation (x_t)** | `x1, x0, t` | `[128, 784]` | 计算概率轨迹点 $x_t = (1-t)x_0 + t x_1$ |
| **Timestep Embedding** | `[128, 1]` | `[128, 128]` | 正弦位置编码 + MLP 投影 |
| **Class Embedding** | `[128]` | `[128, 64]` | Lookup 映射类别标签 $y \to \mathbb{R}^{64}$ |
| **Concat Features** | `x_t, t_emb, y_emb` | `[128, 976]` | $784 + 128 + 64 = 976$ 维度拼接 |
| **MLP Dense Backbone** | `[128, 976]` | `[128, 512]` | 多层线性带 SiLU 激活网络 |
| **Vector Field Output** | `[128, 512]` | `[128, 784]` | 输出预测的目标向量场 $v_\theta(x_t, t, y)$ |
| **Loss Calculation** | `v_pred, target_vf` | `[]` (Scalar) | 计算 $\text{MSE}(v_{\text{pred}}, x_1 - x_0)$ |

---

## 3. 核心公式与代码映射

| 数学概念 / 物理推导 | 公式表达 | 代码变量 / 实现位置 |
| --- | --- | --- |
| **条件轨迹线性插值** | $x_t = (1-t)x_0 + t x_1$ | `x_t = (1 - t_view) * x0 + t_view * x1` |
| **目标向量场 (Target Field)** | $u_t(x_t \mid x_0, x_1) = x_1 - x_0$ | `target_vf = x1 - x0` |
| **Flow Matching 损失** | $\mathcal{L}_{\text{CFM}} = \mathbb{E} \Vert{}v_\theta(x_t, t, y) - (x_1 - x_0)\Vert{}^2$ | `F.mse_loss(predicted_vf, target_vf)` |
| **正弦时间编码** | $\sin(t / 10000^{2i/d}), \cos(t / 10000^{2i/d})$ | `TimestepEmbedding.forward()` 内的矩阵乘法 |
| **Euler ODE 数值采样步进** | $x_{t+\Delta t} = x_t + v_\theta(x_t, t, y) \cdot \Delta t$ | `x = x + v * dt` (位于 `sample_conditional_flow_matching`) |