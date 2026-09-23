# Qwen3-VL 实际使用

Qwen3-VL 覆盖 dense/MoE、Instruct/Thinking 等变体，面向图像、视频和空间理解任务。
实际应用应区分 checkpoint 变体，并使用其官方 processor/template，而不是沿用本仓库 tiny LM tokenizer。

## 架构与工程影响

- Interleaved-MRoPE 对时间、高度、宽度等位置维度编码，服务于图像/视频中的空间时间对齐。
- DeepStack 融合视觉编码器的多层级特征，有利于细节理解，但模型视觉路径不可简化成单个图片向量。
- 动态分辨率和视频帧采样会改变输入 token 数；图片像素、视频帧数、上下文长度应统一纳入服务限额。
- Thinking/Instruct 是不同 checkpoint/推理行为配置，应明确选择，不能仅通过在 prompt 中追加标签互换权重。

## Transformers 图像推理

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

model_id = "Qwen/Qwen3-VL-8B-Instruct"
model = AutoModelForImageTextToText.from_pretrained(
    model_id, dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "/data/sample.jpg"},
        {"type": "text", "text": "描述图中设备，并指出安全风险。"},
    ],
}]
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
images, videos = process_vision_info(messages, image_patch_size=16)
inputs = processor(
    text=text, images=images, videos=videos, return_tensors="pt", do_resize=False
).to(model.device)
generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
answer_ids = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(inputs.input_ids, generated)
]
print(processor.batch_decode(answer_ids, skip_special_tokens=True)[0])
```

视频输入建议用 Qwen 官方 `qwen-vl-utils` 中的 `process_vision_info`，而不是自行将整段视频转成无时间戳的静态图片列表。该工具负责模型所需的视觉/视频输入组织；帧采样和像素预算仍应按业务设置。图片尺寸已经由该工具缩放时，向 processor 传 `do_resize=False`，避免二次 resize。

## vLLM 服务

Qwen3-VL 官方建议使用 `vllm>=0.11.0` 或 SGLang 部署。实际启动参数依硬件、
checkpoint 精度和并行策略而异：

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --host 0.0.0.0 \
  --port 8000
```

示例参数只适合作为小规模服务起点，不代表适用于所有 GPU/模型尺寸。通过 OpenAI-compatible API 发送多模态请求前，先核验所用 vLLM 版本的图像/视频 schema、
本地媒体路径白名单与视频解码后端。不要直接对公网开放任意本地媒体读取路径。

Transformers 本地入口在 [model.py](./model.py)，可直接传入多个 `--image`/`--video`。
加载时可通过 `processor_kwargs` 配置处理器支持的 `min_pixels`/`max_pixels`：

```bash
python architecture/qwen/Qwen3-VL/inference.py \
  --image /data/sample.jpg \
  --prompt "描述图片中的设备。"
```

## 微调和评测

ms-swift 可用于 Qwen3-VL 的多模态 SFT/LoRA；MoE 大模型全参训练往往需专门的并行配置，
不能把单卡 dense 示例命令直接套用。微调配置需显式检查 `IMAGE_MAX_TOKEN_NUM`、
`VIDEO_MAX_TOKEN_NUM`、FPS/帧数、最大长度和可训练模块。按任务分别度量视觉问答、
文档理解、grounding、视频时序问答和空间输出质量。

跨框架工作流与限制见 [`production/README.md`](../production/README.md)。

## 官方资料

- [Qwen3-VL 官方代码、cookbooks 与部署指南](https://github.com/QwenLM/Qwen3-VL)
- [Transformers Qwen3-VL 文档](https://huggingface.co/docs/transformers/main/model_doc/qwen3_vl)
