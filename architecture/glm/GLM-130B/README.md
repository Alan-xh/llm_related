# GLM-130B

本例展示 GLM 的两个教学重点：自回归空白填充（blank infilling）和二维位置
编码。官方资料：[论文](https://arxiv.org/abs/2210.02414)、
[THUDM/GLM-130B](https://github.com/THUDM/GLM-130B)。

## 数据流与 Shape

```text
input_ids                         [B, T]
position_ids                      [B, 2, T]
token embedding + block embedding [B, T, H]
Q/K/V + RoPE                      [B, heads, T, d]
prefix-visible attention          [B, T, H]
SwiGLU                            [B, T, H]
lm_head                           [B, T, V]
```

训练样本形如 `[BOS] + prefix + <BLANK> + suffix + target`。prefix 和 suffix
构成 context，`prefix_length` 之前使用 prefix-visible mask，loss 对 context
位置设为 `-100`，只优化 target。

## 核心公式

```text
M(q, k) = causal(q, k) OR (q < prefix_length AND k < prefix_length)
RoPE(x, p) = x * cos(p * inv_freq) + rotate_half(x) * sin(p * inv_freq)
L = CE(logits[:, :-1], labels[:, 1:], ignore_index=-100)
```

二维 position 的第 0 行进入 RoPE，第 1 行进入可学习 block-position embedding。
这是对原理的可运行抽象，不是 130B checkpoint-compatible 实现。

## 运行

```bash
python architecture/glm/GLM-130B/model.py
python architecture/glm/GLM-130B/train.py --steps 5 --checkpoint glm130b_tiny.pt
python architecture/glm/GLM-130B/inference.py --checkpoint glm130b_tiny.pt
```

