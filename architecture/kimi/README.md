# Kimi 系列架构

本目录提供 Kimi 系列关键架构机制的最小 PyTorch 教学实现。代码面向
CPU smoke test 和张量流学习，不兼容 Moonshot AI 官方 tokenizer、权重、
训练数据或推理 kernel。

| 目录 | 重点 | 入口 |
| --- | --- | --- |
| [`Kimi-K2/`](./Kimi-K2/) | MLA、共享专家与 top-k MoE、KV cache | `model.py` / `train.py` / `inference.py` |
| [`Kimi-Linear/`](./Kimi-Linear/) | KDA 有限状态记忆、3:1 KDA/MLA 混合注意力 | `model.py` / `train.py` / `inference.py` |

## 统一数据流

```text
input_ids [B, T]
  -> embedding [B, T, H]
  -> pre-norm attention
       MLA: latent K/V + decoupled RoPE + global KV cache
       KDA: gated delta update + recurrent state cache
  -> top-k routed SwiGLU + shared expert
  -> final RMSNorm
  -> logits [B, T, V]
```

训练目标为：

```text
L_total = L_next_token + 0.01 * L_MoE_aux
```

`ByteTokenizer` 只把 UTF-8 字节映射到 259 个 token，方便不依赖外部词表运行。
`generate()` 会在首轮处理完整 prompt，之后每轮只输入最后一个 token，并把
每层 MLA 的 KV 或 KDA 的 recurrent state 传回模型。

## 张量 Shape 流动追踪

| 节点/模块 | 输入 Shape | 输出 Shape | 说明 |
|---|---|---|---|
| `ByteTokenizer` | 文本 | `[B, T]` | UTF-8 字节映射为 token id |
| Embedding | `[B, T]` | `[B, T, H]` | token id 映射为隐藏状态 |
| MLA/KDA 输入投影 | `[B, T, H]` | `[B, heads, T, D]` | 按注意力头拆分隐藏维度 |
| MLA 注意力输出 | `[B, heads, T, D]` | `[B, T, H]` | 注意力聚合并合并多头 |
| KDA 注意力输出 | `[B, heads, T, D]` | `[B, T, H]` | 递归状态逐 token 更新 |
| MoE/SwiGLU | `[B, T, H]` | `[B, T, H]` | 专家变换后保持隐藏维度 |
| LM Head | `[B, T, H]` | `[B, T, V]` | 投影到词表 logits |

## 核心公式与代码映射

| 数学公式 | 代码位置 |
|---|---|
| `Attention(Q,K,V) = softmax(QK^T / sqrt(D))V` | `MLAAttention.forward()` 中的 `scores`、`weights`、`output` |
| `retrieved = qS` | `KDAttention.forward()` 中的 `retrieved = torch.einsum(...)` |
| `delta = v - kS` | `KDAttention.forward()` 中的 `prediction` 与 `delta` |
| `S <- decay*S + gate*k^T*delta` | `KDAttention.forward()` 中的 `state = ...` |
| `y = sum_i w_i Expert_i(x) + SharedExpert(x)` | `TopKMoE.forward()` 中的 `routed` 累加 |
| `L_total = L_next_token + 0.01*L_MoE_aux` | `KimiForCausalLM.forward()` 中的 `loss` |

## 快速运行

```bash
python architecture/kimi/Kimi-K2/model.py
python architecture/kimi/Kimi-K2/train.py --steps 2
python architecture/kimi/Kimi-K2/inference.py --max-new-tokens 16

python architecture/kimi/Kimi-Linear/model.py
python architecture/kimi/Kimi-Linear/train.py --steps 2
python architecture/kimi/Kimi-Linear/inference.py --max-new-tokens 16
```

## 边界

教学版为了可读性会计算所有小专家，KDA 使用 token 循环而不是官方优化
chunkwise kernel；因此它用于解释 shape、缓存和更新公式，不用于性能比较或
官方模型部署。真实模型的工具调用、权限控制和沙箱执行也必须由外部系统负责。
