# MLP 多维特征连续值回归模型 技术架构与接口文档

## 1. 架构总览

本模型实现了一个**通用多层感知机 (MLP)**，用于处理表格或多维向量数据的连续值回归任务。整体架构由模块化的 `MLPBlock` 堆叠而成，并在最终输出层使用线性投影。

### 数据流拓扑图 (Data Flow Diagram)

```text
  [Input Tensor]
  Shape: [B, D_in]
        │
        ▼
┌─────────────────────────┐
│       MLPBlock 1        │
│ ├─ Linear(D_in, D_hid)  │
│ ├─ LayerNorm(D_hid)     │
│ ├─ SiLU() Activation    │
│ └─ Dropout(p=0.1)       │
└─────────────────────────┘
        │ Shape: [B, D_hid]
        ▼
┌─────────────────────────┐
│       MLPBlock 2        │
│ ├─ Linear(D_hid, D_hid) │
│ ├─ LayerNorm(D_hid)     │
│ ├─ SiLU() Activation    │
│ └─ Dropout(p=0.1)       │
└─────────────────────────┘
        │ Shape: [B, D_hid]
        ▼
┌─────────────────────────┐
│      Regression Head    │
│ └─ Linear(D_hid, D_out) │  <-- 无激活函数/归一化，输出无界连续值
└─────────────────────────┘
        │
        ▼
  [Output Target]
  Shape: [B, D_out]

```

---

## 2. 张量 Shape 流动追踪 (Tensor Flow Table)

以下表格展示了单一批次大小为 $B$（默认 $B=64$），特征输入维度 $D_{in}=10$，隐藏维度 $D_{hidden}=64$，目标维度 $D_{out}=1$ 时，数据在各个节点中的维度变换全过程：

| 节点/模块 | 输入 Shape | 输出 Shape | 说明 / 维度变化原因 |
| --- | --- | --- | --- |
| **Input Data (`x`)** | `[B, 10]` | - | 原始特征批次输入 |
| **Block 1 - Linear** | `[B, 10]` | `[B, 64]` | 第一层线性投影：$X \cdot W_1 + b_1$ |
| **Block 1 - LayerNorm** | `[B, 64]` | `[B, 64]` | 对特征维度（最后一个轴）求均值方差并正规化 |
| **Block 1 - SiLU** | `[B, 64]` | `[B, 64]` | 逐元素非线性激活变换：$x \cdot \sigma(x)$ |
| **Block 1 - Dropout** | `[B, 64]` | `[B, 64]` | 训练阶段按概率 $p=0.1$ 随机将部分元素置零 |
| **Block 2 - Linear** | `[B, 64]` | `[B, 64]` | 第二层隐藏层线性变换 |
| **Block 2 - LayerNorm** | `[B, 64]` | `[B, 64]` | 隐藏特征空间正规化 |
| **Block 2 - SiLU** | `[B, 64]` | `[B, 64]` | 非线性特征映射 |
| **Block 2 - Dropout** | `[B, 64]` | `[B, 64]` | 随机正则化特征维度 |
| **Head - Linear** | `[B, 64]` | `[B, 1]` | 回归投影层：将 64 维隐藏向量映射为 1 维预测标量 |
| **MSE Loss Unit** | `[B, 1]`, `[B, 1]` | `[]` (Scalar) | 计算预测值与真实目标值之间的均方误差损失 |

---

## 3. 核心公式与代码映射

| 数学推导公式 (Mathematical Formula) | 代码实现组件/变量 (Code Implementation) | 物理/逻辑意义 |
| --- | --- | --- |
| $\mathbf{y} = \mathbf{X}\mathbf{W} + \mathbf{b} + \boldsymbol{\epsilon}$ | `y = torch.matmul(x, true_w) + true_b + noise` | 线性高斯数据生成机制 |
| $\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$ | `self.act = nn.SiLU()` | 平滑非线性激活函数 |
| $\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$ | `self.norm = nn.LayerNorm(out_features)` | 层归一化，稳定深度梯度传导 |
| $\mathcal{L}_{\text{MSE}} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$ | `self.mse_fn = nn.MSELoss()` | 优化目标函数 (Mean Squared Error) |
| $\text{MAE} = \frac{1}{N}\sum_{i=1}^{N}\Vert{}y_i - \hat{y}_i\Vert{}$ | `self.mae_fn = nn.L1Loss()` | 评估指标 (Mean Absolute Error) |