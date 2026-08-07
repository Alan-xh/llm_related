# Mamba (Selective State Space Model) 多维特征连续值回归模型 技术架构与接口文档

## 1. 架构总览

本模型实现了基于 **Mamba (Selective State Space Model, S6)** 的多维特征连续值回归架构。该模型通过将多维表格特征序列化投影，结合输入驱动的选择性离散化机制（Selective Mechanism）与深度局部卷积，实现了对表格数据的非线性序列化依赖表征与无界连续值回归。

### 数据流拓扑图 (Data Flow Diagram)

```text
  [Input Tensor]
  Shape: [B, D_in]
        │
        ▼
┌─────────────────────────┐
│ Sequence Tokenization   │
│ ├─ Unsqueeze(-1)        │  <-- 维度重塑: [B, D_in, 1]
│ └─ Linear(1, D_model)   │  <-- 特征投影: [B, D_in, D_model]
└─────────────────────────┘
        │ Shape: [B, D_in, D_model]
        ▼
┌─────────────────────────┐ ◄───────────────────────────┐
│       Mamba Block       │                             │
│ ├─ LayerNorm(D_model)   │                             │
│ ├─ Linear Expansion     │  <-- 维度拓展至 2 * D_inner │
│ ├─ Chunk: Main / Res    │                             │ (堆叠 N 层 Mamba Layer)
│ ├─ Conv1d (Depthwise)   │                             │
│ ├─ SiLU Activation      │                             │
│ ├─ S6 Selective Kernel  │  <-- 动态离散化 Δ, B, C    │
│ ├─ Gated Fusion (GLU)   │  <-- 门控融合主分支与残差分支│
│ ├─ Out Linear Projection│                             │
│ └─ Residual Add         │ ────────────────────────────┘
└─────────────────────────┘
        │ Shape: [B, D_in, D_model]
        ▼
┌─────────────────────────┐
│     Final LayerNorm     │
└─────────────────────────┘
        │ Shape: [B, D_in, D_model]
        ▼
┌─────────────────────────┐
│   Global Mean Pooling   │  <-- 沿特征序列维度 (dim=1) 聚合
└─────────────────────────┘
        │ Shape: [B, D_model]
        ▼
┌─────────────────────────┐
│     Regression Head     │
│ ├─ Linear(D_mod, D_mid) │
│ ├─ SiLU() Activation    │
│ └─ Linear(D_mid, D_out) │  <-- 无激活/归一化，输出无界预测标量
└─────────────────────────┘
        │
        ▼
  [Output Target]
  Shape: [B, D_out]

```

---

## 2. 张量 Shape 流动追踪 (Tensor Flow Table)

以下表格展示了单一批次大小 $B=64$，输入特征维度 $D_{in}=10$，模型隐层维度 $D_{model}=64$，SSM 状态维度 $D_{state}=16$，拓展倍数 $\text{Expand}=2$（即 $D_{inner}=128$），目标维度 $D_{out}=1$ 时，数据在各个节点中的维度变换全过程：

| 节点/模块 | 输入 Shape | 输出 Shape | 说明 / 维度变化原因 |
| --- | --- | --- | --- |
| **Input Data (`x`)** | `[B, 10]` | - | 原始表格/特征向量批次输入 |
| **Tokenization (`Unsqueeze`)** | `[B, 10]` | `[B, 10, 1]` | 增加序列维度，将每个特征标量视为序列中的一个 Token |
| **Input Linear Projection** | `[B, 10, 1]` | `[B, 10, 64]` | 将单维 Token 映射为 $D_{model}$ 高维特征表示 |
| **Pre-LN LayerNorm** | `[B, 10, 64]` | `[B, 10, 64]` | 层归一化，稳定骨干网络深层梯度传导 |
| **In Projection (`in_proj`)** | `[B, 10, 64]` | `[B, 10, 256]` | 线性升维至 $2 \times D_{inner}$（$2 \times 128 = 256$） |
| **Chunk Split** | `[B, 10, 256]` | Main: `[B, 10, 128]`<br>

<br>Res: `[B, 10, 128]` | 切分为 SSM 主处理分支 (`x_branch`) 与门控残差分支 (`res_branch`) |
| **Conv1d (Depthwise)** | `[B, 10, 128]` | `[B, 10, 128]` | 沿序列维度进行 1D 因果深度可分离卷积，捕捉局部相邻特征交互 |
| **Conv SiLU Activation** | `[B, 10, 128]` | `[B, 10, 128]` | 卷积特征非线性平滑激活 |
| **S6 Kernel - Parameter Proj** | `[B, 10, 128]` | $\Delta$: `[B, 10, 128]`<br>

<br>$B, C$: `[B, 10, 16]` | 输入驱动投影产生选择性参数：步长 $\Delta$、输入矩阵 $B$、输出矩阵 $C$ |
| **S6 Kernel - Recurrence (Scan)** | `[B, 10, 128]` | `[B, 10, 128]` | 离散化 SSM 状态递推更新：$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$，$y_t = C_t h_t + D x_t$ |
| **Gated Fusion (GLU)** | `[B, 10, 128]`, `[B, 10, 128]` | `[B, 10, 128]` | 门控相乘融合：$\text{SSM\_Out} \odot \text{SiLU}(\text{Res\_Branch})$ |
| **Out Projection (`out_proj`)** | `[B, 10, 128]` | `[B, 10, 64]` | 降维线性投影，恢复至 $D_{model}$ 通道维度 |
| **Residual Add** | `[B, 10, 64]`, `[B, 10, 64]` | `[B, 10, 64]` | 元素级残差跳跃连接 |
| **Final LayerNorm** | `[B, 10, 64]` | `[B, 10, 64]` | 骨干网末端隐层归一化 |
| **Global Mean Pooling** | `[B, 10, 64]` | `[B, 64]` | 沿序列维度求均值，聚合全局特征信息 |
| **Head - Linear 1** | `[B, 64]` | `[B, 32]` | 回归头第一层降维投影 |
| **Head - SiLU** | `[B, 32]` | `[B, 32]` | 非线性特征映射 |
| **Head - Linear 2** | `[B, 32]` | `[B, 1]` | 最终回归投影层：将 32 维特征向量映射为 1 维预测标量 |
| **MSE Loss Unit** | `[B, 1]`, `[B, 1]` | `[]` (Scalar) | 计算预测连续值与真实目标值之间的均方误差损失 |

---

## 3. 核心公式与代码映射

| 数学推导公式 (Mathematical Formula) | 代码实现组件/变量 (Code Implementation) | 物理/逻辑意义 |
| --- | --- | --- |
| $\Delta = \text{Softplus}(\mathbf{W}_{\Delta} x + b_{\Delta})$ | `delta = F.softplus(self.dt_proj(delta_rank))` | 输入驱动的离散化步长计算（选择性机制） |
| $\bar{\mathbf{A}} = \exp(\Delta \mathbf{A}), \quad \bar{\mathbf{B}} = \Delta \mathbf{B}$ | `delta_A = torch.exp(delta_expanded * A_expanded)`<br>

<br>`delta_B = delta_expanded * B_param.unsqueeze(-2)` | 零阶保持 (ZOH) 连续状态空间矩阵离散化 |
| $h_t = \bar{\mathbf{A}}_t h_{t-1} + \bar{\mathbf{B}}_t x_t$ | `h = delta_A[:, t] * h + dB_x[:, t]` | SSM 隐状态选择性时域递推更新方程 |
| $y_t = \mathbf{C}_t h_t + \mathbf{D} x_t$ | `y_t = torch.sum(h * c_t, dim=-1) + x * self.D` | 隐藏状态向输出空间的选择性线性投影与跳跃连接 |
| $\text{Out} = \text{S6}(x_{\text{branch}}) \odot \text{SiLU}(x_{\text{res}})$ | `gated_out = ssm_out * F.silu(res_branch)` | 门控线性单元 (GLU) 特征乘法选择与融合 |
| $\mathcal{L}_{\text{MSE}} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$ | `self.mse_fn = nn.MSELoss()` | 连续值回归预测优化目标函数 |
| $\text{MAE} = \frac{1}{N}\sum_{i=1}^{N}\Vert{}y_i - \hat{y}_i\Vert{}$ | `self.mae_fn = nn.L1Loss()` | 平均绝对误差回归评估指标 |