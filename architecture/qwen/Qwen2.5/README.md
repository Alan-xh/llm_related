# Qwen2.5

Qwen2.5 教学实现以 Qwen2 的 GQA decoder 为基础，增加可配置的长上下文 RoPE 缩放。
代码聚焦语言模型主干；代码、数学、视觉扩展和后训练接口在真实发布中是数据与系统层面的
完整工程，而不是一个额外的线性层。

## 来源

- 论文：[Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- 发布说明：[Qwen2.5 blog](https://qwenlm.github.io/blog/qwen2.5/)
- 开源代码：[QwenLM/Qwen2.5](https://github.com/QwenLM/Qwen2.5)

## 数据流与 Shape

输入 `[B, T]` 经 embedding 后为 `[B, T, H]`，经过多层：

```text
RMSNorm -> GQA + scaled RoPE -> residual -> RMSNorm -> SwiGLU -> residual
                                          |
                                          v
                                 final lm_head [B, T, V]
```

RoPE 的 `cos/sin` cache 为 `[1, 1, T, d]`，Q/K 为
`[B, n_heads or n_kv_heads, T, d]`。本实现默认 `rope_scaling=4.0`，把位置坐标除以
该倍率后生成旋转角度，用于展示“位置频率缩放”的接口；它不等价于官方长上下文训练、
YaRN、稀疏注意力或 1M 上下文推理框架。

## 核心公式

```text
pos' = pos / s
cos, sin = cos/sin(pos' * inv_freq)
RoPE(x) = x * cos + rotate_half(x) * sin
L_lm = CrossEntropy(logits[:, :-1], labels[:, 1:])
```

## 运行

```bash
python architecture/qwen/Qwen2.5/model.py
python architecture/qwen/Qwen2.5/train.py --steps 5
python architecture/qwen/Qwen2.5/inference.py --prompt "Summarize long context"
```
