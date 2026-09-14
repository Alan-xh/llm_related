# GLM 系列教学实现

本目录用一个可在 CPU 上运行的小型 PyTorch 模型，展示 GLM/ChatGLM
各代最值得观察的结构和接口变化。实现关注数据流、Tensor Shape 和训练/推理
边界，不追求官方参数规模、tokenizer、checkpoint 或对齐数据兼容。

| 目录 | 重点 | 入口 |
| --- | --- | --- |
| [`GLM-130B/`](./GLM-130B/) | 空白填充目标、二维位置编码 | `model.py` / `train.py` / `inference.py` |
| [`ChatGLM/`](./ChatGLM/) | 双语对话模板、GLM 式生成 | 同上 |
| [`ChatGLM2/`](./ChatGLM2/) | MQA、RoPE、KV cache | 同上 |
| [`ChatGLM3/`](./ChatGLM3/) | 工具调用、代码解释器模板 | 同上 |
| [`GLM-4/`](./GLM-4/) | GQA、长上下文缩放、视觉占位符 | 同上 |
| [`GLM-4.5/`](./GLM-4.5/) | 混合推理、MoE、agent 工具模板 | 同上 |
| [`GLM-4.6/`](./GLM-4.6/) | 长上下文编码、代码 agent 工作流 | 同上 |
| [`GLM-4.7/`](./GLM-4.7/) | thinking-before-acting、终端工具模板 | 同上 |

## 共用实现

[`common.py`](./common.py) 提供：

- `RMSNorm`、RoPE、GQA/MQA、SwiGLU、可选 top-k MoE 和 KV cache。
- `use_2d_position_ids=True` 时，将绝对位置用于 RoPE，将 block position
  作为可学习 embedding 加到 hidden states。
- `prefix_length` 让 prefix 内 token 互相可见、生成区保持因果注意力，
  用于教学版 blank infilling。
- `ByteTokenizer`，词表为 256 个 UTF-8 byte 加上 `pad/bos/eos` 三个 token，
  不需要额外 tokenizer 依赖。
- `format_chat()`、`parse_tool_call()`，把对话、工具 schema 和工具结果
  组织成可观察的文本协议。

普通 causal LM 的数据流为：

```text
input_ids                         [B, T]
embedding                         [B, T, H]
Q/K/V                             [B, heads, T, head_dim]
GQA/MQA + RoPE + prefix mask      [B, T, H]
SwiGLU or top-k MoE               [B, T, H]
lm_head                           [B, T, V]
```

GLM-130B 的 blank infilling 示例将输入组织为：

```text
[BOS] + prefix + <BLANK> + suffix + target
```

prefix 和 suffix 构成可双向读取的 context，loss 只计算 target 区域。
二维位置示例为 `position_ids: [B, 2, T]`，其中第 0 行是绝对位置，第 1 行
是 block position。

## 运行

从仓库根目录运行：

```bash
python architecture/glm/GLM-130B/model.py
python architecture/glm/GLM-130B/train.py --steps 5 --checkpoint glm130b_tiny.pt
python architecture/glm/GLM-130B/inference.py --checkpoint glm130b_tiny.pt

python architecture/glm/ChatGLM3/inference.py --prompt "调用天气工具"
python architecture/glm/GLM-4.7/inference.py --thinking --prompt "分析下一步终端操作"
```

目录名包含连字符，因此推荐直接运行脚本。训练脚本默认使用 CPU 小配置；
`--device cuda` 可在本机 CUDA 环境中尝试。

## 边界

这些代码是架构教学样例，不是官方模型的复刻。真实模型还包含专用 tokenizer、
大规模预训练/后训练数据、分布式训练、量化、视觉编码器、工具执行沙箱和
checkpoint 转换逻辑。GLM-4、GLM-4.5、GLM-4.6、GLM-4.7 目录中的视觉、
agent 和终端能力以 placeholder/template 形式表达，模型本身不会真的执行工具。

