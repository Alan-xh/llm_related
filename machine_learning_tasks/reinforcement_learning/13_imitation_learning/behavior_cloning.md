# 行为克隆 (Behavior Cloning) 模仿学习策略网络技术架构与接口文档

## 1. 架构总览

行为克隆 (Behavior Cloning) 管道由**环境状态输入**、**深度多层感知机策略网络 (Policy Network)**、**策略头 (Policy Head)** 以及**最大似然交叉熵损失驱动模块**四大核心部分构成。

数据流与处理控制逻辑如下：

```
[环境状态 s] (State Tensor) 
     │
     ▼  Shape: [B, State_Dim]
┌─────────────────────────────────────────────────────────┐
│ MLPBlock 1: Linear -> LayerNorm -> GELU -> Dropout      │
└─────────────────────────────────────────────────────────┘
     │
     ▼  Shape: [B, Hidden_Dim]
┌─────────────────────────────────────────────────────────┐
│ MLPBlock 2: Linear -> LayerNorm -> GELU -> Dropout      │
└─────────────────────────────────────────────────────────┘
     │
     ▼  Shape: [B, Hidden_Dim]
┌─────────────────────────────────────────────────────────┐
│ Policy Head: Linear (Hidden_Dim -> Action_Dim)          │
└─────────────────────────────────────────────────────────┘
     │
     ▼  Shape: [B, Action_Dim]
[动作 Logits z] ────────┐
                        ├──► [CrossEntropy Loss] ◄── [专家动作目标 a*]
[Softmax(z)] ───────────┘
     │
     ▼
[动作概率分布 pi(a|s)] ──► argmax ──► [最终预测动作]

```

---

## 2. 张量 Shape 流动追踪 (Tensor Flow Table)

以下为批次大小为 $B$，状态维度 $S = 10$，隐层维度 $H = 64$，动作类别数 $A = 4$ 的张量流动全过程：

| 节点 / 模块名称 | 输入 Shape | 输出 Shape | 维度变化原因 / 算子操作 说明 |
| --- | --- | --- | --- |
| **Input State** | - | `[B, 10]` | 包含 Batch 特征的环境状态向量序列 |
| **MLPBlock 1 (Linear)** | `[B, 10]` | `[B, 64]` | 线性升维映射 $W_1 \cdot x + b_1$ |
| **MLPBlock 1 (LN + GELU)** | `[B, 64]` | `[B, 64]` | 归一化与非线性激活函数，保持特征维度不变 |
| **MLPBlock 2 (Linear)** | `[B, 64]` | `[B, 64]` | 隐层深度特征提取与交互 |
| **Policy Head (Linear)** | `[B, 64]` | `[B, 4]` | 线性降维映射至动作空间，输出非归一化对数概率 Logits |
| **Softmax (Inference)** | `[B, 4]` | `[B, 4]` | 在动作维度上进行指数归一化得到概分布 $\pi(a\Vert{}s)$ |
| **Argmax (Inference)** | `[B, 4]` | `[B]` | 沿着动作维度选择概率最大的动作索引 $a_{pred}$ |
| **CrossEntropy (Loss)** | `[B, 4]`, `[B]` | `[]` (标量) | 计算预测 Logits 与专家动作 target 之间的交叉熵损失 |

---

## 3. 核心公式与代码映射

| 数学推导公式 | 代码变量 / 函数实现 | 功能及映射解释 |
| --- | --- | --- |
| $z = \text{PolicyNetwork}(s)$ | `logits = policy_net(states)` | 状态至动作 Logits 的神经映射 |
| $\text{GELU}(x) = x \cdot \Phi(x)$ | `nn.GELU()` | 现代高阶平滑激活函数 |
| $\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$ | `nn.LayerNorm(out_features)` | 隐层特征归一化，稳定深度网络梯度 |
| $\pi_\theta(a \mid s) = \frac{e^{z_a}}{\sum_{k=1}^A e^{z_k}}$ | `F.softmax(logits, dim=-1)` | 输出概率分布（对应推理动作选择） |
| $\mathcal{L}_{\text{BC}}(\theta) = -\sum_{i=1}^B \log \pi_\theta(a_i^* \mid s_i)$ | `nn.CrossEntropyLoss()` | 行为克隆核心标量损失，拟合专家行动 |
| $a_{\text{pred}} = \arg\max_{a} z_a$ | `torch.argmax(logits, dim=1)` | 贪婪策略下的决定性动作选择 |