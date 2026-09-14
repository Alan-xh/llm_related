# GLM-4

本例展示 GLM-4 语言核心的三个接口方向：GQA、长上下文位置缩放和视觉
placeholder。官方资料：[THUDM/GLM-4](https://github.com/THUDM/GLM-4)。

## 数据流与 Shape

```text
input_ids             [B, T]
Q                     [B, 4, T, d]
K/V                   [B, 2, T, d]
repeat KV             [B, 4, T, d]
scaled RoPE           [B, heads, T, d]
SwiGLU                [B, T, H]
logits                [B, T, V]
```

本例的 `rope_scaling=2.0` 只是将位置坐标按倍率缩放的教学接口，不等价于
官方长上下文训练。`<image>` 也只是 byte-token 文本占位符，不包含视觉编码器。
工具和多模态系统需要外部模块把真实输入转换成模型可读的 token/embedding。

## 运行

```bash
python architecture/glm/GLM-4/model.py
python architecture/glm/GLM-4/train.py --steps 5
python architecture/glm/GLM-4/inference.py --prompt "描述 <image>"
```

