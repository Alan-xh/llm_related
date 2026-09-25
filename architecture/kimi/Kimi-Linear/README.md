# Kimi Linear

这是一个 CPU 可运行的 Kimi Linear 风格教学实现，不兼容官方 checkpoint、
tokenizer 或 FLA kernel。它把 Kimi Linear 的核心结构压缩成可读的小模型：

- **KDA**：用 gated delta rule 更新有限状态 RNN memory。
- **Hybrid attention**：默认每 3 个 KDA 层插入 1 个全局 MLA 层。
- **State cache**：KDA 缓存 `[B, heads, D, D]` 的状态，而不是保存整段 KV。
- **MoE FFN**：保留 top-k 专家和 shared expert，便于观察与 K2 的共同点。

## KDA 数据流

```text
q, k, v, gate, decay                  [B, heads, T, D]
retrieved = q @ state                  [B, heads, T, D]
delta = v - k @ state                  [B, heads, T, D]
state <- decay * state + gate * k^T delta
output = retrieved + gate * delta       [B, T, H]
```

KDA 的循环版本用于展示状态更新；生产实现通常使用优化过的 chunkwise
kernel。MLA 层仍保存全局 KV cache，用于周期性全局交互。

## 张量 Shape 流动追踪

| 节点/模块 | 输入 Shape | 输出 Shape | 缓存 |
|---|---|---|---|
| `input_ids` | `[B, T]` | `[B, T, H]` | Embedding 输出 |
| KDA 投影 | `[B, T, H]` | `[B, heads, T, D]` | `state: [B, heads, D, D]` |
| KDA 递归输出 | `[B, heads, T, D]` | `[B, T, H]` | 只保留有限状态和 position |
| MLA 投影 | `[B, T, H]` | `[B, heads, T, D]` | K/V: `[B, heads, T_cache, D]` |
| MoE/SwiGLU | `[B, T, H]` | `[B, T, H]` | 无额外序列缓存 |
| `lm_head` | `[B, T, H]` | `[B, T, V]` | 输出词表 logits |

## 核心公式与代码映射

| 数学公式 | 代码位置 |
|---|---|
| `retrieved = qS` | `common.py::KDAttention.forward()` 的 `retrieved` |
| `prediction = kS` | `common.py::KDAttention.forward()` 的 `prediction` |
| `delta = v - prediction` | `common.py::KDAttention.forward()` 的 `delta` |
| `S <- decay*S + gate*k^T*delta` | `common.py::KDAttention.forward()` 的 `state` 更新 |
| `Attention(Q,K,V)` | `common.py::MLAAttention.forward()`，用于周期性全局交互 |
| `L_total = L_next_token + 0.01*L_MoE_aux` | `common.py::KimiForCausalLM.forward()` |

## 运行

```bash
python architecture/kimi/Kimi-Linear/model.py
python architecture/kimi/Kimi-Linear/train.py --steps 5 --checkpoint kimi_linear_tiny.pt
python architecture/kimi/Kimi-Linear/inference.py --checkpoint kimi_linear_tiny.pt
```

## 来源

- [Kimi Linear 官方仓库](https://github.com/MoonshotAI/Kimi-Linear)
- [Kimi Linear 技术报告](https://arxiv.org/abs/2510.26692)
