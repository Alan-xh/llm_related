# GLM-4.5

本例用一个轻量 top-k MoE 和 thinking prompt 展示 agentic reasoning 与 coding
接口。官方资料：[GLM-4.5](https://github.com/zai-org/GLM-4.5)。

## 数据流与 Shape

```text
x                         [B, T, H]
router logits             [B*T, E]
top-k expert SwiGLU       [B*T, H]
weighted reduce           [B, T, H]
lm_head                   [B, T, V]
```

训练损失为：

```text
L_total = L_lm + 0.01 * L_balance
```

`--thinking` 会在 assistant 区域加入 `<think>`，它和直接回答共享同一个网络；
工具调用仍通过 `format_chat()` 的 schema/role 标记表达，模型不会自动执行工具。

## 运行

```bash
python architecture/glm/GLM-4.5/model.py
python architecture/glm/GLM-4.5/train.py --steps 5 --checkpoint glm45_tiny.pt
python architecture/glm/GLM-4.5/inference.py --thinking --prompt "修复一个测试"
```

