# Flow Matching (连续流匹配) 技术架构与接口文档

## 1. 架构总览

连续流匹配 (Continuous Flow Matching, CFM) 是一种基于连续时间概率流与常微分方程 (ODE) 的生成模型。相比于扩散模型 (Diffusion Models) 的弯曲随机轨迹，CFM (特别是 OT-CFM) 学习从先验高斯分布 $p_0 \sim \mathcal{N}(0, I)$ 到目标数据分布 $p_1 \sim p_{\text{data}}$ 的**直线向量场** (Straight Paths)。

```
[噪声分布 p_0: N(0, I)] ──(t=0) ───────> [中间插值状态 x_t] ───────(t=1) ──> [目标数据 p_1: Image]
                                                │
                                                ▼
                                    ┌──────────────────────┐
                                    │ v_theta(x_t, t) 预测 │
                                    └──────────────────────┘
                                                │
                                                ▼
                                    Loss = || v_theta - (x_1 - x_0) ||^2

```

* **训练阶段**：在 $[0, 1]$ 范围内随机采样时间 $t$，利用公式 $x_t = (1-t)x_0 + t x_1$ 生成插值点，训练网络 $v_\theta(x_t, t)$ 逼近常数切线向量 $x_1 - x_0$。
* **推理阶段**：从噪声 $x_0$ 出发，以预测的 $v_\theta(x_t, t)$ 作为方向，采用欧拉显式积分步步推进（ODE 求解）还原生成图像。

---

## 2. 张量 Shape 流动追踪 (Tensor Flow Table)

以 **ConvFlowMatcher** 架构，处理批次大小 $B=128$ 的 $1 \times 28 \times 28$ MNIST 图像为例：

| 节点 / 模块 | 输入 Shape | 输出 Shape | 说明 / 维度变化原因 |
| --- | --- | --- | --- |
| **Input $x_1$** | `[128, 1, 28, 28]` | - | 原始目标图像 Batch |
| **Input $t$** | `[128, 1]` | - | 采样时间步 $t \sim U(0,1)$ |
| **Prior $x_0$** | `[128, 1, 28, 28]` | - | 采样自标准高斯分布 $\mathcal{N}(0, I)$ |
| **Conditional Path $x_t$** | `[128, 1, 28, 28]` | `[128, 1, 28, 28]` | 条件计算: $(1-t)x_0 + t x_1 + \sigma \sqrt{t(1-t)}\epsilon$ |
| **Time MLP** | `[128, 1]` | `[128, 128]` | 时间正弦/线性高维投影 |
| **Time Proj** | `[128, 128]` | `[128, 256, 1, 1]` | 线性变换后拓展维度以对齐 Spatial 特征 |
| **Encoder Conv1** | `[128, 1, 28, 28]` | `[128, 64, 28, 28]` | 保持分辨率卷积抽取 |
| **Encoder Conv2** | `[128, 64, 28, 28]` | `[128, 128, 14, 14]` | Stride=2 卷积下采样 |
| **Encoder Conv3** | `[128, 128, 14, 14]` | `[128, 256, 7, 7]` | Stride=2 卷积下采样瓶颈层 |
| **Time Addition** | `[128, 256, 7, 7]` + `[128, 256, 1, 1]` | `[128, 256, 7, 7]` | 空间维度广播相加注入时间信息 |
| **Decoder ConvT1** | `[128, 256, 7, 7]` | `[128, 128, 14, 14]` | 转置卷积上采样 |
| **Decoder ConvT2** | `[128, 128, 14, 14]` | `[128, 64, 28, 28]` | 转置卷积上采样 |
| **Decoder ConvOut** | `[128, 64, 28, 28]` | `[128, 1, 28, 28]` | 还原通道数，得到预测向量场 $v_\theta$ |
| **Target Vector Field** | `[128, 1, 28, 28]` | `[128, 1, 28, 28]` | 计算目标直线切向速度 $u = x_1 - x_0$ |
| **Loss Step** | $v_\theta$ & $u$ | `[]` (Scalar) | 计算元素级均方误差 $\text{Mean}((v_\theta - u)^2)$ |

---

## 3. 核心公式与代码映射

| 数学公式概念 | 数学表达 | 代码对应实现 | 说明 |
| --- | --- | --- | --- |
| **时间步采样** | $t \sim \text{Uniform}(0, 1)$ | `torch.rand(batch_size, 1)` | 随机生成时刻 $t$ |
| **概率轨迹点** | $x_t = (1-t)x_0 + t x_1 + \sigma\sqrt{t(1-t)}\epsilon$ | `(1-t)*x0 + t*x1 + sigma*torch.sqrt(t*(1-t))*eps` | 流匹配的边缘概率路径采样 |
| **最优传输向量场** | $u_t(x_t \mid x_0, x_1) = x_1 - x_0$ | `target_vf = x1 - x0` | 从起点直接指向终点的常数速度场 |
| **流匹配损失** | $\mathcal{L}_{\text{CFM}} = \mathbb{E} \Vert{} v_\theta(x_t, t) - u_t \Vert{}^2$ | `torch.mean((predicted_vf - target_vf)**2)` | 回归目标向量场 |
| **欧拉数值采样** | $x_{t+\Delta t} = x_t + v_\theta(x_t, t) \cdot \Delta t$ | `x = x + v * dt` | 推理求解 ODE 生成真实数据 |