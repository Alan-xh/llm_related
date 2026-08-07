# BERT Masked Language Modeling (MLM) 技术架构与接口文档

## 1. 架构总览

BERT (Bidirectional Encoder Representations from Transformers) 是基于自监督双向编码器实现的语言表征模型。本实现手写构建了完整的 Transformer 模块，剥离了 `nn.TransformerEncoder` 高级封装，并严格采用 **Pre-LN** (LayerNorm 在注意力子层之前) 的工程架构以增强深层模型的训练稳定性。

### 数据流与模块架构拓扑：

```text
       Input Tokens [B, Seq_Len]
                   │
                   ▼
         Dynamic MLM Masking 80-10-10
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
    Masked Tokens          Labels
     [B, Seq_Len]       [B, Seq_Len] (包含 -100 忽略位)
         │
         ├── Token Embedding  [B, Seq_Len, D_MODEL]
         └── Position Embedding [B, Seq_Len, D_MODEL]
                   │
                   ▼ (Element-wise Addition & Dropout)
         Embedded Tensor [B, Seq_Len, D_MODEL]
                   │
                   ▼
     ┌───────────────────────────┐
     │ Transformer Encoder Layer │ x NUM_LAYERS (4)
     │ ┌───────────────────────┐ │
     │ │ Pre-LN Multi-Head Attn│ │
     │ └───────────────────────┘ │
     │ ┌───────────────────────┐ │
     │ │ Pre-LN FeedForward    │ │
     │ └───────────────────────┘ │
     └─────────────┬─────────────┘
                   │
                   ▼
     Encoder Output [B, Seq_Len, D_MODEL]
                   │
                   ▼
         MLM Prediction Head (Linear + GELU + LN + Linear)
                   │
                   ▼
         Logits [B, Seq_Len, VOCAB_SIZE]
                   │
                   ▼ (Compute CrossEntropyLoss with Labels)
            MLM Loss (Scalar)

```

---

## 2. 张量 Shape 流动追踪 (Tensor Flow Table)

| 节点/模块 | 输入 Shape | 输出 Shape | 说明 / 维度变化原因 |
| --- | --- | --- | --- |
| **Raw Input** | - | `[B, Seq_Len]` | 原始输入的 Token ID 整数序列 |
| **Masking Pipeline** | `[B, Seq_Len]` | `[B, Seq_Len]` | 产生 `inputs` (15% 被掩码) 和 `labels` (非遮挡位置 -100) |
| **Token Embedding** | `[B, Seq_Len]` | `[B, Seq_Len, D_MODEL]` | 将整数 Token ID 映射为连续高维向量 |
| **Position Embedding** | `[B, Seq_Len]` | `[B, Seq_Len, D_MODEL]` | 位置矩阵按行广播相加提供绝对位置信息 |
| **Embedding Sum** | `2 x [B, Seq_Len, D_MODEL]` | `[B, Seq_Len, D_MODEL]` | Token 与 Position 特征逐元素相加 |
| **MHA Projections (Q/K/V)** | `[B, Seq_Len, D_MODEL]` | `[B, NHead, Seq_Len, D_Head]` | 线性变换后 reshape 并 transpose 拆分多头 |
| **Attention Scores** | `Q, K^T` | `[B, NHead, Seq_Len, Seq_Len]` | `Q x K^T / sqrt(D_Head)` 点积算注意力权重 |
| **Attn Weighted Sum** | `Scores, V` | `[B, NHead, Seq_Len, D_Head]` | 注意力 Softmax 权重矩阵与 `V` 矩阵加权求和 |
| **MHA Output Projection** | `[B, NHead, Seq_Len, D_Head]` | `[B, Seq_Len, D_MODEL]` | 拼合所有 Head (`transpose`+`view`) 后过线性层 `w_o` |
| **FFN Layer 1** | `[B, Seq_Len, D_MODEL]` | `[B, Seq_Len, FF_DIM]` | 线性升维扩张表达能力，过 GELU 激活 |
| **FFN Layer 2** | `[B, Seq_Len, FF_DIM]` | `[B, Seq_Len, D_MODEL]` | 线性降维映射回模型主维度 |
| **Prediction Head** | `[B, Seq_Len, D_MODEL]` | `[B, Seq_Len, VOCAB_SIZE]` | 映射至全词表空间，输出未归一化的概率 Logits |

---

## 3. 核心公式与代码映射

### (1) 多头 Scaled Dot-Product 注意力 (Scaled Dot-Product Attention)

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

```python
# 代码实现映射:
# Q: [B, nhead, Seq_Len, d_head]
# K.transpose(-2, -1): [B, nhead, d_head, Seq_Len]
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
attn_weights = torch.softmax(scores, dim=-1)
out = torch.matmul(attn_weights, V)

```

### (2) 前馈神经网络 (Position-wise Feed-Forward Network)

$$\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$$

```python
# 代码实现映射:
x = self.fc1(x)  # x W_1 + b_1
x = self.activation(x)  # GELU(...)
x = self.fc2(x)  # (...) W_2 + b_2

```

### (3) 掩码语言模型交叉熵损失 (MLM Cross-Entropy Loss)

$$\mathcal{L}_{\text{MLM}} = -\frac{1}{\vert{}M\vert{}} \sum_{i \in M} \log P(x_i = y_i \mid \mathbf{\tilde{x}})$$

```python
# 代码实现映射:
criterion = nn.CrossEntropyLoss(ignore_index=-100)  # M 为标签 != -100 的索引集合
loss = criterion(logits.view(-1, VOCAB_SIZE), labels.view(-1))

```

---

## 4. 关键参数与接口配置说明

| 参数名称 | 类型 | 默认值 | 作用与推荐配置建议 |
| --- | --- | --- | --- |
| `VOCAB_SIZE` | `int` | `1000` | 字典维度。在实际任务中对应 Tokenizer 词表大小（如 Bert-Base 对应 30522）。 |
| `D_MODEL` | `int` | `128` | Transformer 的隐藏层向量通道数（如 Bert-Base 对应 768）。 |
| `NHEAD` | `int` | `4` | 多头注意力头的数量，需满足 `D_MODEL % NHEAD == 0`。 |
| `NUM_LAYERS` | `int` | `4` | Encoder 层数，决定模型的深度（如 Bert-Base 为 12 层）。 |
| `FF_DIM` | `int` | `256` | 前馈层内部升维大小，通常设置为 `4 * D_MODEL`。 |
| `mask_prob` | `float` | `0.15` | MLM 掩码比例，BERT 标准设定为 `0.15`。 |
| `ignore_index` | `int` | `-100` | PyTorch 损失函数忽略标志，屏蔽未掩码 Token 对梯度的贡献。 |