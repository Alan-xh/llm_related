# PureMambaLM 技术架构与接口文档

## 1. 架构总览

本模型实现了轻量化的选择性状态空间语言模型（Mamba Language Model）。数据流经以下核心阶段：

1. **Embedding 层**：将输入的离散 Token ID 映射为连续向量表示。
2. **Mamba 堆叠层（SimpleMambaBlock）**：
* 线性投影与门控分流 (`in_proj`)
* 1D 因果卷积捕获局部特征 (`conv1d`)
* 选择性参数投影生成动态步长及状态参数 (`x_proj`)
* 状态空间核心循环（离散化与隐状态更新递归）
* 残差与门控激活融合输出


3. **LayerNorm 与语言模型头（lm_head）**：对齐特征并输出词表维度的对数概率。

---

## 2. 张量 Shape 流动追踪 (Tensor Flow Table)

| 节点/模块 | 输入 Shape | 输出 Shape | 说明 / 维度变化原因 |
| --- | --- | --- | --- |
| `Input (input_ids)` | `[B, Seq_Len]` | - | 原始 Token 索引输入 |
| `Embedding` | `[B, Seq_Len]` | `[B, Seq_Len, d_model]` | 词嵌入映射 |
| `SimpleMambaBlock.in_proj` | `[B, Seq_Len, d_model]` | `[B, Seq_Len, 2 * d_inner]` | 扩展内部维度并分流 |
| `conv1d` | `[B, d_inner, Seq_Len]` | `[B, d_inner, Seq_Len]` | 1D 因果卷积（保持长度不变） |
| `x_proj` | `[B, Seq_Len, d_inner]` | `[B, Seq_Len, d_inner + 2*d_state]` | 生成动态参数 delta, B, C |
| SSM Core Loop (`h`) | `[B, d_inner, d_state]` | `[B, d_inner, d_state]` | 递归更新隐状态 |
| SSM Core Loop (`y`) | `[B, Seq_Len, d_inner]` | `[B, Seq_Len, d_inner]` | 状态空间序列输出 |
| `out_proj` | `[B, Seq_Len, d_inner]` | `[B, Seq_Len, d_model]` | 投影回模型隐藏维度 |
| `lm_head` | `[B, Seq_Len, d_model]` | `[B, Seq_Len, vocab_size]` | 输出词表概率分布 |

---

## 3. 核心公式与代码映射

| 数学/理论公式 | 代码实现符号 / 变量 |
| --- | --- |
| $h'(t) = Ah(t) + Bx(t)$ | `h = dA * h + dB * x_t` (离散化更新) |
| $y(t) = Ch(t) + Dx(t)$ | `y_t = torch.sum(h * c_t, dim=-1)` 及 `+ x_conv * D` |
| $\Delta = \text{softplus}(\text{Linear}(x))$ | `delta = F.softplus(delta)` |
| $\bar{A} = \exp(\Delta A)$ | `dA = torch.exp(dt_t.unsqueeze(-1) * A)` |