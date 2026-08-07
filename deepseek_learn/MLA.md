# MLA (Multi-Head Latent Attention) 技术架构与接口文档

## 1. 架构总览

MLA 是一种高效的注意力机制，旨在通过将 KV 缓存进行低秩压缩，显著降低大语言模型推理时的显存占用。

* **核心组件**：
* **Query 投影**：通过低秩分解进行投影，并拆分为非旋转（nope）与旋转位置编码（rope）部分。
* **KV 压缩**：将原始 KV 矩阵通过线性层 `W_kv_a` 压缩为低秩隐藏状态 `kv_cache`，并存储旋转部分 `pe_cache`。
* **注意力计算**：利用 `einsum` 在压缩后的低秩空间直接计算注意力分数，避免显式还原为原始高维 KV。



## 2. 张量 Shape 流动追踪 (Tensor Flow Table)

| 节点/模块 | 输入 Shape | 输出 Shape | 说明 / 维度变化原因 |
| --- | --- | --- | --- |
| **Input** | [B, S, D] | - | 原始输入 |
| **Wq_a** | [B, S, D] | [B, S, Q_rank] | Query 低秩降维 |
| **Wk_a** | [B, S, D] | [B, S, KV_rank + D_rope] | KV 低秩降维 + PE 拆分 |
| **RoPE (q_pe)** | [B, S, H, D_rope] | [B, S, H, D_rope] | 应用旋转位置编码 |
| **Scores** | - | [B, S, H, S] | 注意力得分矩阵 |
| **Attention Output** | [B, S, H, KV_rank] | [B, S, D] | 重构输出并 WO 映射 |

## 3. 核心公式与代码映射

| 数学公式 (LaTeX) | 代码实现 (变量) | 说明 |
| --- | --- | --- |
| `RMSNorm(x) = x / sqrt(mean(x^2) + eps)` | `RMSNorm` 类 | 归一化层 |
| `Q = W_q_b * Norm(W_q_a * x)` | `q = self.wq_b(...)` | Query 计算分支 |
| `KV_c = Norm(W_kv_a * x)` | `kv = self.kv_norm(...)` | KV 压缩存储 |
| `Scores = (Q_nope K_c^T + Q_rope PE_c^T) / sqrt(d)` | `scores = (scores_nope + scores_pe) / ...` | 低秩注意力计算 |

---

*注：本实现已针对推理优化，移除了不必要的冗余，确保 Cache 的高效更新与计算。*