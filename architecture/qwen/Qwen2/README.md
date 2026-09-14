# Qwen2

Qwen2 教学实现重点展示 Grouped-Query Attention（GQA）和自回归推理的 KV cache。
真实 Qwen2 还包括新的 tokenizer、长上下文训练和多个尺寸的工程配置。

## 来源

- 论文：[Qwen2 Technical Report](https://arxiv.org/abs/2407.10671)
- 开源代码：[QwenLM/Qwen2](https://github.com/QwenLM/Qwen2)

## 数据流与 Shape

默认教学配置为 `n_heads=4`、`n_kv_heads=2`：

```text
hidden states                 [B, T, H]
Q                              [B, 4, T, d]
K/V                            [B, 2, T, d]
repeat KV for GQA              [B, 4, T, d]
attention output               [B, T, H]
logits                         [B, T, V]
```

当生成第一个片段后，每层缓存 `K_cache/V_cache: [B, 2, T_cache, d]`。下一步仅输入
`[B, 1]` 的 token，得到一个 query，再把新的 KV 拼到 cache 后计算注意力。

## 核心公式

```text
K' = repeat(K, n_heads / n_kv_heads)
V' = repeat(V, n_heads / n_kv_heads)
Attention = softmax(QK'^T / sqrt(d) + causal_mask) V'
K_cache <- concat(K_cache, K_new, dim=sequence)
```

## 运行

```bash
python architecture/qwen/Qwen2/model.py
python architecture/qwen/Qwen2/train.py --steps 5
python architecture/qwen/Qwen2/inference.py --prompt "Explain GQA" --max-new-tokens 24
```

`common.py` 的 `generate()` 会自动在首轮后使用 `past_key_values`。
