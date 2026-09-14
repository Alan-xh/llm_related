# Qwen1

Qwen 初代教学实现：dense decoder-only Transformer、RMSNorm、RoPE、SwiGLU 和 causal language
modeling。官方模型还包含专用 tokenizer、ChatML 对话格式和大规模预训练，本目录只保留网络主干。

## 来源

- 论文：[Qwen Technical Report](https://arxiv.org/abs/2309.16609)
- 开源代码：[QwenLM/Qwen](https://github.com/QwenLM/Qwen)

## 数据流与 Shape

对输入 `input_ids: [B, T]`：

```text
Embedding                  [B, T] -> [B, T, H]
N x TransformerBlock       [B, T, H] -> [B, T, H]
RMSNorm + lm_head          [B, T, H] -> [B, T, V]
shifted cross entropy      logits[:, :-1], labels[:, 1:] -> scalar
```

每个 block 是 `x + Attention(RMSNorm(x))`，再接
`x + SwiGLU(RMSNorm(x))`。注意力使用
`Q,K,V: [B, n_heads, T, H/n_heads]`，因 Qwen1 教学配置采用 MHA，所以 query 和 KV head 数相同。

## 核心公式

```text
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * g
SwiGLU(x) = W_down( SiLU(W_gate x) * W_up x )
Attention(Q,K,V) = softmax(QK^T / sqrt(d) + causal_mask) V
L = -sum_t log p(token_t | token_<t>)
```

`model.py` 中的 `generate()` 使用 temperature、top-k、top-p 和逐层 KV cache。

## 运行

```bash
python architecture/qwen/Qwen1/model.py
python architecture/qwen/Qwen1/train.py --steps 5 --checkpoint qwen1_tiny.pt
python architecture/qwen/Qwen1/inference.py --checkpoint qwen1_tiny.pt --prompt "Qwen is"
```

这是教学配置，不读取官方权重。
