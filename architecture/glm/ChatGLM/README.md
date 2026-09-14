# ChatGLM

本例在 GLM 式 decoder 上增加双语对话模板，突出“对话对齐主要由数据格式和
训练目标表达”的事实。官方资料：[ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)。

## 数据流与 Shape

模型主干保持 `[B, T] -> [B, T, H] -> [B, T, V]`，使用二维 position ids、
prefix-visible attention、SwiGLU 和 KV cache。推理 prompt 为：

```text
[gMASK] <sop> <|user|>
问题
<|assistant|>
```

`format_chat()` 也支持 system、user、assistant 多轮消息。中文和英文都先编码
为 UTF-8 bytes，因此本例可以直接观察跨语言文本如何流过同一个 LM head。

## 运行

```bash
python architecture/glm/ChatGLM/model.py
python architecture/glm/ChatGLM/train.py --steps 5 --checkpoint chatglm_tiny.pt
python architecture/glm/ChatGLM/inference.py --prompt "介绍 RMSNorm"
```

这不是官方 tokenizer 或 ChatGLM checkpoint 的兼容实现。

