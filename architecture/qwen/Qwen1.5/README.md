# Qwen1.5

Qwen1.5 教学实现沿用 Qwen1 的 dense decoder，并把 feed-forward 层抽象成可选的 top-k
Mixture-of-Experts，用同一套训练和生成接口展示 dense/MoE 的差异。真实 Qwen1.5 发布系列、
tokenizer 和对齐数据远比本例完整。

## 来源

- 论文与系列说明：[Qwen Technical Report](https://arxiv.org/abs/2309.16609)
- 开源代码：[QwenLM/Qwen1.5](https://github.com/QwenLM/Qwen1.5)

## 数据流与 Shape

输入 `input_ids: [B, T]` 首先变为 `[B, T, H]`。每个 block 输出仍是 `[B, T, H]`：

```text
x -> RMSNorm -> MHA -> residual
  -> RMSNorm -> SwiGLU 或 MoE -> residual
  -> final RMSNorm -> lm_head -> logits [B, T, V]
```

MoE 路径中 router 产生 `[B*T, E]` 的 logits，选出每个 token 的 `K` 个 expert，
各 expert 返回 `[B*T, H]`，按 router 权重加和后恢复为 `[B, T, H]`。

## 核心公式

```text
p(e | x) = softmax(W_router x)
MoE(x) = sum_{e in TopK(p)} p(e | x) * Expert_e(x)
L_total = L_lm + 0.01 * E * sum_e(mean(p_e) * mean(assign_e))
```

`build_model(use_moe=False)` 生成 dense 版本，`use_moe=True` 打开教学 MoE。

## 运行

```bash
python architecture/qwen/Qwen1.5/model.py
python architecture/qwen/Qwen1.5/train.py --steps 5 --moe
python architecture/qwen/Qwen1.5/inference.py --moe --prompt "Qwen1.5 is"
```

由于目录名包含点号，推荐直接运行脚本；它仍然可以复用 `architecture.qwen.common`。
