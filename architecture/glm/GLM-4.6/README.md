# GLM-4.6

本例在 GLM-4.5 风格的 MoE/推理接口上，增加更大的 RoPE 位置范围和 coding-agent
模板，用来观察长上下文与多步任务接口的关系。官方资料：
[GLM-4.6](https://github.com/zai-org/GLM-4.6)。

## 数据流与 Shape

```text
input_ids             [B, T]
GQA                   Q [B, 4, T, d], K/V [B, 2, T, d]
RoPE scaling=4.0      [B, heads, T, d]
top-k MoE             [B, T, H]
KV cache              [B, 2, T_cache, d]
logits                [B, T, V]
```

位置缩放是演示 API，并不代表官方上下文长度或训练配方。代码任务可以通过
`<|user|>`、`<|assistant|>` 和 `<think>` 区域组织为多轮样本。

## 运行

```bash
python architecture/glm/GLM-4.6/model.py
python architecture/glm/GLM-4.6/train.py --steps 5
python architecture/glm/GLM-4.6/inference.py --prompt "写一个 Python 函数"
```

