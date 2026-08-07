# Transformer Seq2Seq 技术架构与接口文档

## 1. 架构总览

本模块手写实现了标准的 Transformer 编码器-解码器 (Seq2Seq) 架构。整体数据路径分为 Encoder 编码阶段和 Decoder 增量/并行解码阶段。

```
       Source Sequence [B, S_src]                      Target Sequence [B, S_tgt]
                   │                                               │
           Embedding Scale                                 Embedding Scale
                   │                                               │
          Positional Encoding                             Positional Encoding
                   │                                               │
                   ▼                                               ▼
         ┌───────────────────┐                           ┌───────────────────┐
         │ TransformerEncLyr │                           │ Masked MHA (Self) │
         │   (Self-Attn)     │                           └─────────┬─────────┘
         └─────────┬─────────┘                                     │
                   │ (Loop N layers)                               ▼
                   │                                     ┌───────────────────┐
                   ├────────────────────────────────────►│    Cross-Attention│
                   │ Memory [B, S_src, D]                  └─────────┬─────────┘
                   ▼                                               │ (Loop N layers)
             Encoder Output                                        ▼
                                                           Linear Head Projection
                                                                   │
                                                                   ▼
                                                         Logits [B, S_tgt, Vocab]

```

---

## 2. 张量 Shape 流动追踪 (Tensor Flow Table)

以下为单次前向传播 (Forward Pass) 过程中所有核心节点的维度变化流程（设 Batch Size = $B$, Source Len = $S_{src}$, Target Len = $S_{tgt}$, Embedding Dim = $D$, Num Heads = $H$, $D_h = D/H$）：

| 节点 / 模块 | 输入 Shape | 输出 Shape | 说明 / 维度变化原因 |
| --- | --- | --- | --- |
| **Src Input** | - | `[B, S_src]` | 输入源语言 Token ID 序列 |
| **Tgt Input** | - | `[B, S_tgt]` | 解码器 Teacher Forcing 输入（已右移） |
| **Src Embedding + PE** | `[B, S_src]` | `[B, S_src, D]` | 查表得到向量并加上正弦/余弦位置编码 |
| **Encoder Self-Attn Q/K/V** | `[B, S_src, D]` | `[B, H, S_src, D_head]` | 线性变换投影并分拆多头 |
| **Encoder Attn Score** | `[B, H, S_src, D_head]` | `[B, H, S_src, S_src]` | $Q \cdot K^T / \sqrt{d_k}$ 点积相似度 |
| **Encoder Out (Memory)** | `[B, S_src, D]` | `[B, S_src, D]` | 经过 $N$ 层 Encoder Layer 后的上下文记忆表示 |
| **Decoder Masked Self-Attn** | `[B, S_tgt, D]` | `[B, S_tgt, D]` | 施加因果掩码 `[1, 1, S_tgt, S_tgt]` 防止窥探未来词 |
| **Decoder Cross-Attn** | Q: `[B, S_tgt, D]`<br>

<br>K,V: `[B, S_src, D]` | `[B, S_tgt, D]` | Target 查询 Source 编码上下文信息 |
| **Decoder FeedForward** | `[B, S_tgt, D]` | `[B, S_tgt, D]` | 逐位置两层全连接升维 ($D \to 2D \to D$) 再降维 |
| **Linear Head** | `[B, S_tgt, D]` | `[B, S_tgt, Vocab_Size]` | 线性投影至全词表空间，输出分类 logits |

---

## 3. 核心公式与代码映射

| 数学概念 / 公式 | 关键变量 / 代码实现 | 位置 |
| --- | --- | --- |
| **Scaled Dot-Product**<br>

<br>$\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ | `scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)`<br>

<br>`out = torch.matmul(attn, V)` | `MultiHeadAttention.forward` |
| **Positional Encoding**<br>

<br>$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$ | `pe[:, 0::2] = torch.sin(position * div_term)`<br>

<br>`pe[:, 1::2] = torch.cos(position * div_term)` | `PositionalEncoding.__init__` |
| **Causal Masking**<br>

<br>$M_{i,j} = \begin{cases} 0 & i \ge j \\ -\infty & i < j \end{cases}$ | `torch.triu(torch.ones(sz, sz), diagonal=1).bool()` | `Seq2SeqTransformer.generate_square_subsequent_mask` |
| **Embedding Scaling**<br>

<br>$\text{Embed}(x) \cdot \sqrt{d_{model}}$ | `self.src_emb(src) * math.sqrt(emb_dim)` | `Seq2SeqTransformer.forward` |