# Qwen2.5-VL 实际使用

Qwen2.5-VL 是图像/视频理解模型族。实际推理依赖官方 checkpoint、`AutoProcessor`、
视觉预处理和兼容的 Transformers 版本；本目录是工程入口说明，不是重新实现模型权重。

## 主要机制与使用影响

- 动态分辨率使高分辨率图像可保留细节，同时意味着视觉 token 数、显存和延迟随像素预算变化。
- 视频按帧和时间信息编码；抽帧策略、最大帧数与视频时长是推理配置的一部分。
- OCR、文档、定位等任务需要按任务设计提示词和输出解析，不能只用通用图像描述评测。

## Transformers 图像推理

```python
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="sdpa",
)
processor = AutoProcessor.from_pretrained(
    model_id,
    min_pixels=256 * 28 * 28,
    max_pixels=1280 * 28 * 28,
)
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "url": "https://example.com/sample.jpg"},
        {"type": "text", "text": "提取图片中的表格字段，并以 JSON 返回。"},
    ],
}]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)
generated = model.generate(**inputs, max_new_tokens=512, do_sample=False)
answer_ids = generated[:, inputs.input_ids.shape[1]:]
print(processor.batch_decode(answer_ids, skip_special_tokens=True)[0])
```

本例采用 Transformers 消息格式中的 `url` 字段。生产任务宜将图片预先下载/校验并使用受控的本地路径或对象存储；不要允许任意用户 URL 触发服务器端请求。

## 视频输入

按官方示例安装 `qwen-vl-utils`，使用 `process_vision_info` 解析视频帧与时间信息。视频请求优先限制总时长、FPS、最大帧数和总像素，再进入模型；大视频不应默认一次性解码完整内容。具体消息字段和工具参数以当前 `qwen-vl-utils`/模型仓库版本为准。

## 本地运行

```bash
python architecture/qwen/Qwen2.5-VL/inference.py \
  --image /data/sample.jpg \
  --prompt "提取图片中的表格字段，并以 JSON 返回。"
python architecture/qwen/Qwen2.5-VL/inference.py \
  --video /data/clip.mp4 \
  --prompt "概括视频内容。" \
  --max-new-tokens 256
```

入口使用 [model.py](./model.py) 的 `Qwen2_5VLModel`。可通过 `processor_kwargs` 传入
`min_pixels`/`max_pixels` 限制视觉 token 预算；模型下载和显存需求取决于 checkpoint、精度与像素预算。

## 服务与微调

- 在线服务：可评估 vLLM 或 SGLang；确认部署版本支持目标 checkpoint、视频输入及所需输出格式。
- LoRA/SFT：可用 ms-swift，样本需包含一致的 `messages` 与 `images`/`videos` 字段；训练前抽样验证模板渲染和媒体读取。
- 评测：按 OCR、文档抽取、VQA、grounding、视频问答分别构造验证集；结构化任务同时度量字段准确率和 JSON 可解析率。

通用框架的训练、部署、数据治理建议见 [`production/README.md`](../production/README.md)。

## 官方资料

- [Qwen2.5-VL 官方代码与模型说明](https://github.com/QwenLM/Qwen2.5-VL)
- [Transformers Qwen2.5-VL 文档](https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl)
