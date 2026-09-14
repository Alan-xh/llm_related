# Qwen3

Qwen3 教学实现把三个容易观察的变化放在一个小模型中：GQA、Q/K normalization 和
top-k MoE，并在推理入口通过 prompt 控制 thinking/non-thinking 模式。模式切换主要发生
在 chat template、训练数据和推理预算层面，并不是把一个模型复制成两个网络。

## 来源

- 论文：[Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- 开源代码：[QwenLM/Qwen3](https://github.com/QwenLM/Qwen3)

## 数据流与 Shape

```text
input_ids                         [B, T]
embedding                         [B, T, H]
Q/K/V projection                  [B, 4/2, T, d]
GQA + QK RMSNorm + RoPE           [B, T, H]
router logits                     [B*T, E]
top-k expert weighted SwiGLU      [B*T, H] -> [B, T, H]
lm_head                           [B, T, V]
```

训练时 MoE 额外计算负载均衡项。`--thinking` 会把 prompt 包装为
`<|assistant|>\n<think>\n`，不带该参数则直接从 assistant 区域生成；这是教学版对
统一 thinking/non-thinking 接口的最小表达，不包含官方完整 chat template、thinking budget
或工具调用协议。

## 核心公式

```text
g = softmax(W_r x)
S = TopK(g)
y = sum_{e in S} g_e * SwiGLU_e(x)
L = L_lm + 0.01 * E * sum_e(mean(g_e) * mean(assign_e))
```

其中 `E` 是 expert 数量，`K` 是每个 token 激活的 expert 数量。

## 运行

```bash
python architecture/qwen/Qwen3/model.py
python architecture/qwen/Qwen3/train.py --steps 5 --checkpoint qwen3_tiny.pt
python architecture/qwen/Qwen3/inference.py --checkpoint qwen3_tiny.pt --thinking
python architecture/qwen/Qwen3/inference.py --prompt "Answer directly"
```

默认小配置为 4 个 expert、每 token 激活 2 个 expert，便于在 CPU 上观察路由。
