# ChatGLM2

本例在 ChatGLM 主干上突出多查询注意力（MQA）、RoPE 和自回归 KV cache。
官方资料：[THUDM/ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B)。

## 数据流与 Shape

默认配置为 `num_heads=4`、`num_kv_heads=1`：

```text
hidden states       [B, T, H]
Q                   [B, 4, T, d]
K/V                 [B, 1, T, d]
repeat K/V          [B, 4, T, d]
attention output    [B, T, H]
KV cache per layer  [B, 1, T_cache, d]
```

首轮把完整 prompt 放入模型，后续每步只输入 `[B, 1]`，并把新 K/V 拼到 cache
上。MQA 减少了缓存的 KV head 数量，代码位于 `common.py` 的 `SelfAttention`。

## 运行

```bash
python architecture/glm/ChatGLM2/model.py
python architecture/glm/ChatGLM2/train.py --steps 5
python architecture/glm/ChatGLM2/inference.py --prompt "解释 MQA" --max-new-tokens 24
```

