# Mamba 文本意图识别 Pipeline 技术架构与接口文档

## 1. 架构总览

本架构严格遵照标准 PyTorch 模块化工程进行设计，基于 **Mamba（Selective State Space Model, S6）** 实现端到端 Token 序列（最大长度为 32）的 8 分类意图识别任务。模型通过纯手写选择性扫描（Selective Scan）机制，实现 $O(N)$ 线性时间复杂度的长上下文依赖提取，解决传统 Transformer 在长序列下的二次计算开销与传统 SSM 无法根据输入选择性过滤信息的缺陷。

```
[Input Tensor: B x L] (Token IDs)
         │
         ▼
 ┌──────────────┐
 │ nn.Embedding │ ───► [B, L, 128]  (d_model=128)
 └──────────────┘
         │
         ▼
 ┌──────────────┐
 │ LayerNorm    │
 └──────────────┘
         │
         ▼
 ┌────────────────────────────────────────────────────────┐
 │ MambaBlock (Layer 1 & 2)                               │
 │                                                        │
 │ ┌──────────────────┐                                   │
 │ │ in_proj (Linear) │ ───► Split into x_branch & z      │
 │ └──────────────────┘      ([B, L, 256] 各半)           │
 │          │                                             │
 │          ▼                                             │
 │ ┌──────────────────┐                                   │
 │ │ Conv1d (k=4)     │ ───► SiLU Activation              │
 │ └──────────────────┘                                   │
 │          │                                             │
 │          ▼                                             │
 │ ┌──────────────────┐                                   │
 │ │ x_proj & dt_proj │ ───► Projections for B, C, Δ      │
 │ └──────────────────┘                                   │
 │          │                                             │
 │          ▼                                             │
 │ ┌──────────────────┐                                   │
 │ │ Selective Scan   │ ───► h_t = Ā * h_{t-1} + B̄ * x_t   │
 │ │ (SSM Discretize) │      y_t = C_t * h_t              │
 │ └──────────────────┘                                   │
 │          │                                             │
 │          ▼                                             │
 │ ┌──────────────────┐                                   │
 │ │ Gate & Out Proj  │ ───► (y + x * D) ⊙ SiLU(z)        │
 │ └──────────────────┘      ───► out_proj ───► [B, L, 128]│
 └────────────────────────────────────────────────────────┘
         │
         ▼
 ┌──────────────┐
 │ Pre-LN Res   │ ───► [B, L, 128]  (残差累加 + 规范化)
 └──────────────┘
         │
         ▼
 ┌──────────────┐
 │ Final Norm   │ ───► [B, L, 128]
 └──────────────┘
         │
         ▼
 ┌──────────────┐
 │ Mean Pooling │ ───► [B, 128]     (全局序列平均池化)
 └──────────────┘
         │
         ▼
 ┌──────────────┐
 │ Linear FC    │ ───► [B, 8]       (意图分类分值)
 └──────────────┘
         │
         ▼
[Logits Tensor: B x 8]

```

---

## 2. 张量 Shape 流动追踪 (Tensor Flow Table)

假设 Batch Size $B = 32$，序列最大长度 $L = 32$，隐藏维度 $D_{model} = 128$，展开维度 $D_{inner} = 256$（扩展倍率 $E=2$），SSM 状态维度 $d_{state} = 16$，$\Delta$ 投影秩 $dt\_rank = \lceil 128 / 16 \rceil = 8$：

| 节点 / 模块名称 | 输入 Shape | 输出 Shape | 维度变化主要原因 / 计算说明 |
| --- | --- | --- | --- |
| **Input Token IDs** | - | `[32, 32]` | 批次输入词 ID 序列（`dtype=torch.long`） |
| **Embedding Layer** | `[32, 32]` | `[32, 32, 128]` | Token 转换至连续向量空间（词表大小 1000 $\rightarrow 128$ 维） |
| **In Projection** | `[32, 32, 128]` | `[32, 32, 256, 2]` *(Chunk 为 2 个 `[32, 32, 256]`)* | 维度扩展两倍，线性投影拆分为主路径特征 $x_{branch}$ 和门控信号 $z$ |
| **Depthwise Conv1d** | `[32, 256, 32]` *(转置前为 `[32, 32, 256]`)* | `[32, 32, 256]` | $1D$ 深度卷积（核大小 $k=4$），捕获局部相邻 Token 的下文信息 |
| **Selective Projections** | `[32, 32, 256]` | `[32, 32, 8]` ($\Delta$), `[32, 32, 16]` ($B$), `[32, 32, 16]` ($C$) | 依靠当前输入动态导出选择性参数 $\Delta, B, C$ |
| **SSM Discretization** | `Δ: [32, 32, 256]`, `A: [256, 16]` | $\bar{A}$: `[32, 32, 256, 16]`, $\bar{B}x$: `[32, 32, 256, 16]` | 通过零阶保持法（ZOH）离散化系统矩阵：$\bar{A} = \exp(\Delta A)$, $\bar{B}x \approx (\Delta B) x$ |
| **Selective Scan Step** | $\bar{A}, \bar{B}x, C$ 以及上一时刻 $h_{t-1}$ | $h_t$: `[32, 256, 16]`, $y_{ssm}$: `[32, 32, 256]` | 沿序列长度 $L$ 进行递推循环更新，计算状态空间输出 $y_t = C_t \cdot h_t$ |
| **Gate & Out Proj** | `[32, 32, 256]` | `[32, 32, 128]` | 融合跳跃连接 $D$ 项，与门控信号 $\text{SiLU}(z)$ 点乘相乘后投影回 $D_{model}$ |
| **Global Mean Pool** | `[32, 32, 128]` | `[32, 128]` | 对序列维度 $L$ 执行自适应平均池化，聚合获得句级全局向量 |
| **Classifier Header** | `[32, 128]` | `[32, 8]` | 线性映射到 8 个意图类别的未归一化分值 Logits |

---

## 3. 核心公式与代码映射

### 1. 选择性状态空间离散化 (Selective SSM Discretization)

* **理论公式**：

$$\bar{A} = \exp(\Delta A), \quad \bar{B} = (\Delta A)^{-1} (\exp(\Delta A) - I) \cdot \Delta B \approx \Delta B$$

* **代码实现 (`MambaBlock.forward`)**：

```python
# 动态导出 dt, B_t, C_t 参数
dt, B_t, C_t = torch.split(x_proj_res, [self.dt_rank, self.d_state, self.d_state], dim=-1)
dt = F.softplus(self.dt_proj(dt))  # [B, L, D_inner]

# 离散化 Ā = exp(Δ * A)
deltaA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # [B, L, D_inner, D_state]

# 离散化 B̄x ≈ (Δ * B) * x
deltaB_x = (dt.unsqueeze(-1) * B_t.unsqueeze(2)) * x_active.unsqueeze(-1)  # [B, L, D_inner, D_state]

```

---

### 2. 时序状态递归与输出方程 (Selective Scan Recursion)

* **理论公式**：

$$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t, \quad y_t = C_t h_t$$

* **代码实现 (`MambaBlock.forward`)**：

```python
# 初始化状态向量 h
h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
y_ssm = []

for t in range(seq_len):
    # 状态转移递推
    h = deltaA[:, t] * h + deltaB_x[:, t]  # [B, D_inner, D_state]
    # 计算选择性输出
    y_t = torch.einsum("bdn,bn->bd", h, C_t[:, t])  # [B, D_inner]
    y_ssm.append(y_t)

y = torch.stack(y_ssm, dim=1)  # [B, L, D_inner]

```

---

### 3. 多分类交叉熵损失 (Cross-Entropy Loss)

* **理论公式**：

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log \left( \frac{\exp(z_{i, y_i})}{\sum_{j=1}^{K} \exp(z_{i, j})} \right)$$

* **代码实现 (`train_one_epoch`)**：

```python
criterion = nn.CrossEntropyLoss()
logits = model(input_ids)  # logits 对应未归一化的 z, labels 对应正确意图类别索引 y
loss = criterion(logits, labels)

```