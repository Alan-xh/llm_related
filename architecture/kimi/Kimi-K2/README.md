# Kimi K2

这是一个 CPU 可运行的 Kimi K2 风格教学实现，不兼容官方 checkpoint、
tokenizer 或训练配方。它只保留最值得观察的结构：

- **MLA**：先把 K/V 内容压到 latent，再恢复多头表示；RoPE 部分单独处理。
- **MoE**：每个 token 选择 top-k 专家，同时保留一个 shared expert。
- **KV cache**：增量生成时只对最后一个 token 做前向计算。
- **Agent prompt**：`--thinking` 只改变文本模板，工具执行由外部系统负责。

## 数据流

```text
input_ids                         [B, T]
embedding                         [B, T, H]
q_lora -> q_up                    [B, heads, T, D]
kv_down -> k_up / v_up            [B, heads, T, D]
decoupled RoPE + MLA              [B, T, H]
top-k routed SwiGLU + shared FFN  [B, T, H]
lm_head                           [B, T, V]
```

## 张量 Shape 流动追踪

| 节点/模块 | 输入 Shape | 输出 Shape | 说明 |
|---|---|---|---|
| `input_ids` | `[B, T]` | `[B, T, H]` | Embedding 映射 |
| `q_lora -> q_up` | `[B, T, H]` | `[B, heads, T, D]` | 查询低秩压缩后恢复多头 |
| `kv_down -> k_up/v_up` | `[B, T, H]` | `[B, heads, T, D]` | K/V 内容 latent 展开 |
| 解耦 RoPE | `[B, heads, T, qk_rope_dim]` | 同 Shape | 对 Q/K 的位置分量旋转 |
| MLA 输出 | `[B, heads, T, D]` | `[B, T, H]` | 因果注意力聚合并合并多头 |
| Top-k MoE + shared FFN | `[B, T, H]` | `[B, T, H]` | 路由专家与共享专家相加 |
| `lm_head` | `[B, T, H]` | `[B, T, V]` | 输出词表 logits |

## 核心公式与代码映射

| 数学公式 | 代码位置 |
|---|---|
| `Attention(Q,K,V) = softmax(QK^T / sqrt(D))V` | `common.py::MLAAttention.forward()` |
| `K/V = Up(Down(hidden_states))` | `kv_down`、`k_up`、`v_up` |
| `y = sum_i w_i Expert_i(x) + SharedExpert(x)` | `common.py::TopKMoE.forward()` |
| `L = CrossEntropy(next_token) + 0.01*L_aux` | `common.py::KimiForCausalLM.forward()` |

训练目标为：

```text
L = CrossEntropy(next_token) + 0.01 * L_aux
```

`L_aux` 是简化的专家负载均衡项。教学实现为了可读性会计算所有小专家，
并不模拟官方的大规模稀疏 kernel。

## 运行

```bash
python architecture/kimi/Kimi-K2/model.py
python architecture/kimi/Kimi-K2/train.py --steps 5 --checkpoint kimi_k2_tiny.pt
python architecture/kimi/Kimi-K2/inference.py --checkpoint kimi_k2_tiny.pt
python architecture/kimi/Kimi-K2/inference.py --thinking --prompt "Plan a safe tool call"
```

## 来源

- [Kimi K2 官方仓库](https://github.com/MoonshotAI/Kimi-K2)
- [Kimi K2 技术报告](https://arxiv.org/abs/2507.20534)
