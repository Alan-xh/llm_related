# GPT (Decoder-Only Transformer) 技术架构与接口文档

## 1. 架构总览

GPT 是一种纯解码器（Decoder-Only）自回归语言模型。整体流程分为三大阶段：**词嵌入与位置编码融入**、**多层因果掩码 Transformer 解码层堆叠**、以及**语言模型 Head 投影**。

```
[Input Tokens: B, L]
       │
       ├──► Token Embedding [B, L, C] ──┐
       │                                ├──► Add ──► Dropout ──► [B, L, C]
       └──► Pos Embedding   [L, C]    ──┘                           │
                                                                    ▼
                                                         ┌──────────────────────┐
                                                         │ Causal Mask [1,1,L,L]│
                                                         └──────────┬───────────┘
                                                                    │
                                                                    ▼
                                                      ┌────────────────────────────┐
                                                      │ Decoder Layer x NUM_LAYERS │
                                                      │  ├── Causal Self-Attention │
                                                      │  └── FeedForward (MLP)     │
                                                      └─────────────┬──────────────┘
                                                                    │
                                                                    ▼
                                                         LayerNorm & Head Proj
                                                                    │
                                                                    ▼
                                                       [Logits: B, L, Vocab_Size]

```

---

## 2. 张量 Shape 流动追踪 (Tensor Flow Table)

假定配置为：`Batch_Size = B`, `Seq_Len = L`, `D_Model = C`, `NHead = H`, `D_Head = C / H`, `Vocab_Size = V`。

| 节点 / 模块 | 输入 Shape | 输出 Shape | 说明 / 维度变化原因 |
| --- | --- | --- | --- |
| **Input Token IDs** | `[B, L]` | - | 原始整型输入序列 |
| **Token Embedding** | `[B, L]` | `[B, L, C]` | 映射整型 ID 为连续向量 |
| **Position Embedding** | `[L]` | `[L, C]` -> `[B, L, C]` | 映射位置序号为向量并广播 |
| **Embedding Fusion** | `[B, L, C]`, `[B, L, C]` | `[B, L, C]` | 逐元素相加（Token + Pos） |
| **Q/K/V Projection** | `[B, L, C]` | `[B, H, L, D_Head]` | 线性变换后拆分多头并转置维度 |
| **Attention Scores** | `[B, H, L, D_Head]`, `[B, H, D_Head, L]` | `[B, H, L, L]` | 计算矩阵乘法 $Q \cdot K^T / \sqrt{d_k}$ |
| **Causal Mask Fill** | `[B, H, L, L]`, `[1, 1, L, L]` | `[B, H, L, L]` | 将掩码矩阵为 `True` 位置设为 $-\infty$ |
| **Softmax & Dropout** | `[B, H, L, L]` | `[B, H, L, L]` | 在最后一个维度归一化得到注意力权重 |
| **Attention Context** | `[B, H, L, L]`, `[B, H, L, D_Head]` | `[B, H, L, D_Head]` | 权重矩阵乘以 Value 张量 |
| **Head Concat & Proj** | `[B, H, L, D_Head]` | `[B, L, C]` | 重塑维度拼接多头并过输出线性层 |
| **FeedForward Sub-layer** | `[B, L, C]` | `[B, L, C]` | 先升维至 `Dim_FF` 再降维回 `C` |
| **LM Head Projection** | `[B, L, C]` | `[B, L, V]` | 映射回词表空间计算各个 Token 概率 |
| **Loss Reshape** | `[B, L, V]`, `[B, L]` | `[B*L, V]`, `[B*L]` | 展平 Batch 与 Length 维度进行交叉熵计算 |

---

## 3. 核心公式与代码映射

| 数学推导公式 | 代码变量 / 实现名称 | 对应逻辑节点说明 |
| --- | --- | --- |
| $Q = X W_Q, K = X W_K, V = X W_V$ | `q`, `k`, `v` | `self.w_q(x)`, `self.w_k(x)`, `self.w_v(x)` |
| $S = \frac{Q K^T}{\sqrt{d_k}}$ | `scores` | `torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)` |
| $\tilde{S}_{i,j} = \begin{cases} S_{i,j}, & i \ge j \\ -\infty, & i < j \end{cases}$ | `scores.masked_fill` | `scores.masked_fill(attn_mask, float("-inf"))` |
| $A = \text{Softmax}(\tilde{S})$ | `attn_weights` | `torch.softmax(scores, dim=-1)` |
| $O = A \cdot V \cdot W_O$ | `out` | `torch.matmul(attn_weights, v)` 后跟 `self.w_o(...)` |
| $\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$ | `FeedForward.forward` | `self.fc2(self.activation(self.fc1(x)))` |
| $\mathcal{L} = -\sum \log P(x_t \mid x_{<t})$ | `AutoregressiveCrossEntropyLoss` | `nn.CrossEntropyLoss()` |