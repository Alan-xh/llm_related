# GLM-4.7

本例将 thinking-before-acting、多语言 agent coding 和终端操作表达为一个
结构化 prompt/工具协议。官方资料：[GLM-4.7](https://github.com/zai-org/GLM-4.7)。

## 数据流与 Shape

```text
input_ids             [B, T]
Q                     [B, 4, T, d]
K/V (MQA)             [B, 1, T, d]
top-k MoE             [B, T, H]
KV cache              [B, 1, T_cache, d]
logits                [B, T, V]
```

推理模板可以是：

```text
<|user|>任务
<|assistant|>
<think>计划与检查</think>
<tool_call>{"name": "terminal", "arguments": {"command": "..."}}</tool_call>
```

`parse_tool_call()` 只解析 JSON，绝不执行命令；真实终端 agent 必须在外部沙箱
中做权限、路径和结果校验。

## 运行

```bash
python architecture/glm/GLM-4.7/model.py
python architecture/glm/GLM-4.7/train.py --steps 5
python architecture/glm/GLM-4.7/inference.py --thinking --prompt "检查测试失败原因"
```

