# MTP (Multi-Token Prediction) 大语言模型加速架构技术文档

## 0. 原论文介绍

### 0.1 论文信息

- **论文题目**：*Better & Faster Large Language Models via Multi-token Prediction*
- **作者**：Fabian Gloeckle、Badr Youbi Idrissi、Baptiste Roziere、David Lopez-Paz、Gabriel Synnaeve
- **提交时间**：2024 年 4 月
- **论文链接**：[arXiv:2404.19737](https://arxiv.org/abs/2404.19737)

### 0.2 研究背景

标准的语言模型通常采用 **Next-Token Prediction（NTP）**：给定当前位置之前的上下文，只预测下一个 Token：

$$
\mathcal{L}_{1}
=-\sum_t \log P_{\theta}(x_{t+1}\mid x_{1:t})
$$

这种训练方式简单有效，但每个位置只产生一个监督信号，模型主要关注局部的下一个 Token，可能没有充分利用当前位置对更远未来 Token 的信息。论文因此提出 **Multi-Token Prediction（MTP）**：在训练阶段，让模型从同一个上下文位置同时预测未来的多个 Token，从而提高训练样本效率，并促使模型学习更长程的模式。

### 0.3 核心方法

论文在 Transformer 主干网络之后增加多个独立的预测头。共享主干网络首先根据上下文生成隐藏表示：

$$
z_{1:t}=f_s(x_{1:t})
$$

随后，第 $i$ 个预测头使用同一个隐藏表示预测第 $i$ 个未来 Token：

$$
P_{\theta}(x_{t+i}\mid x_{1:t})
=\operatorname{softmax}
\left(f_u\left(f_{h_i}\left(f_s(x_{1:t})\right)\right)\right),
\quad i=1,\ldots,n
$$

其中：

- $f_s$ 是所有预测任务共享的 Transformer 主干；
- $f_{h_i}$ 是第 $i$ 个未来位置对应的独立预测头；
- $f_u$ 是共享的输出投影矩阵；
- $n$ 是一次预测的未来 Token 数量。

对应的多 Token 预测损失可以写成：

$$
\mathcal{L}_{n}
=-\sum_t\sum_{i=1}^{n}
\log P_{\theta}(x_{t+i}\mid z_{1:t})
$$

因此，当前位置可以同时产生 $n$ 个预测目标。通常将 next-token loss 作为主任务，并把更远未来 Token 的预测作为辅助训练目标。

### 0.4 工程实现

如果直接并行保存所有预测头的 logits 和梯度，显存开销会随着预测 Token 数量 $n$ 增长。论文采用顺序执行各个预测头的 forward/backward，并在每个预测头计算完成后释放其 logits 和梯度，只累积传回共享主干的梯度。

这种实现将峰值显存开销从：

$$
O(nV+d)
$$

降低到：

$$
O(V+d)
$$

其中 $V$ 是词表大小，$d$ 是隐藏表示维度，并且不会带来额外的训练时间开销。

### 0.5 论文结论

论文在代码和自然语言任务上进行了实验，主要结论包括：

1. MTP 的收益会随着模型规模增大而更加明显；
2. 在相同训练预算下，MTP 能提升代码生成等生成式任务的表现；
3. 13B 模型在 HumanEval 和 MBPP 上相较于对应的 NTP 模型取得了明显提升；
4. 训练得到的额外预测头可以用于 **self-speculative decoding**，不需要额外的 draft model，论文中报告了最高约 $3\times$ 的推理加速；
5. MTP 还有助于模型学习 induction pattern 和算法推理能力。

### 0.6 与 DeepSeek-V3 MTP 的关系

原论文的方法与 DeepSeek-V3 的实现都属于 MTP，但结构并不完全相同：

| 对比项 | 原论文 | DeepSeek-V3 |
| --- | --- | --- |
| 未来 Token 的预测方式 | 使用多个独立预测头并行预测 | 使用多个 MTP 模块顺序预测 |
| 预测之间的关系 | 各预测头共享主干表示，彼此相对独立 | 后一个预测深度接收前一个深度的表示，保留完整因果链 |
| 参数共享 | 共享 Transformer 主干和输出投影 | 共享 Embedding 和 output head，每个深度拥有自己的 Transformer 模块和投影 |
| 推理用途 | 可用于 self-speculative decoding | MTP 模块可以移除，也可以用于 speculative decoding |

DeepSeek-V3 明确说明其 MTP 设计受到该论文启发，但将“并行预测多个未来 Token”改造成了“顺序预测多个未来 Token”。这样可以在不同预测深度之间传递表示，使后续预测能够利用前面预测深度形成的上下文信息。

## 1. 架构总览

本项目实现了一种基于多 Token 预测（Multi-Token Prediction）的大语言模型架构。该框架在保持主干 Causal LLM（如 Qwen2.5）结构不变的前提下，引入了多个轻量级 `MTPModule`。在模型训练阶段，主干与多个 MTP 头并行预测后续连续的多个 Token，从而增强模型的上下文表征能力并提升推理阶段的生成吞吐量。

### 数据流动路径

```
[Input IDs] ---> [Embedding Layer] -------> [Main Model (Causal LLM)] ---> [Main Hidden State] ---> [Main Head Output]
                        |                                       |
                        +---> [Concat (h_prev, Embed)] --------> [MTP Module] ---> [MTP Hidden State] ---> [MTP Head Output]

```

---

## 2. 张量 Shape 流动追踪 (Tensor Flow Table)

| 节点 / 模块 | 输入 Shape | 输出 Shape | 说明 / 维度变化原因 |
| --- | --- | --- | --- |
| Input (input_ids) | `[B, Seq_Len]` | - | 原始输入的 Token 索引批次 |
| Main Model (`forward_main`) | `[B, Seq_Len]` | `[B, Seq_Len, Hidden_Size]` | 主干 LLM 输出的隐藏状态 |
| Main Output Head (`output_head`) | `[B, Seq_Len, Hidden_Size]` | `[B, Seq_Len, Vocab_Size]` | 主干模型词表概率分布 Logits |
| Token Embedding (`get_input_embeddings`) | `[B, Seq_Len]` | `[B, Seq_Len, Hidden_Size]` | 词嵌入向量映射 |
| MTP Concat 算子 | `[B, Seq_Len, Hidden_Size]` (x2) | `[B, Seq_Len, 2 * Hidden_Size]` | 前序隐藏状态与当前嵌入拼接 |
| MTP Sub-module (`MTPModule`) | `[B, Seq_Len, 2 * Hidden_Size]` | `[B, Seq_Len, Hidden_Size]` | 多层感知机特征降维与融合 |

---

## 3. 核心公式与代码映射

* **隐藏状态特征拼接映射**:
* 数学表示：$h_{in} = [h_{prev} \parallel \text{Embed}(x)]$
* 代码实现：`mtp_input = torch.cat([previous_hidden_output, input_embed], dim=-1)`


* **多层感知机非线性映射**:
* 数学表示：$h_{out} = W_2 \cdot \text{SiLU}(W_1 \cdot h_{in} + b_1) + b_2$
* 代码实现：`x = self.linear2(F.silu(self.linear1(x)))`