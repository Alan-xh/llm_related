# ChatGLM3

本例把 ChatGLM3 的工具调用和代码解释器方向表达为结构化对话模板。
官方资料：[THUDM/ChatGLM3](https://github.com/THUDM/ChatGLM3)。

## 模板与 Shape

工具 schema 进入 `<|tools|>` 区域，模型可生成：

```text
<tool_call>{"name": "weather", "arguments": {"city": "Beijing"}}</tool_call>
```

`format_chat()` 负责拼接 role、system、tools 和 thinking 标记，
`parse_tool_call()` 负责抽取首个 JSON 对象。语言模型仍然是：

```text
input_ids [B, T] -> Transformer [B, T, H] -> logits [B, T, V]
```

工具执行器不在模型中；解析后的 JSON 需要由外部应用校验并执行。

## 运行

```bash
python architecture/glm/ChatGLM3/model.py
python architecture/glm/ChatGLM3/train.py --steps 5
python architecture/glm/ChatGLM3/inference.py --prompt "查询北京天气"
```

