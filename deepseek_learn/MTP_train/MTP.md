# MTP (Multi-Token Prediction) 大语言模型加速架构技术文档

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