# Qwen3-Omni 实际使用

Qwen3-Omni 面向文本、图像、音频、视频的统一感知，并可输出文本和自然语音。
Instruct、Thinking、Captioner 等 checkpoint 的组件和输出能力不同，部署时要按任务选择具体变体。

## Transformers 推理

官方 Python 接口使用 Omni 专用模型类、processor 和 `qwen_omni_utils`。模型较大且 MoE，
Transformers 路径更适合能力验证和完整模型功能接入；低延迟/高并发场景评估官方推荐的推理后端。

```python
import soundfile as sf
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

model_id = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    model_id,
    dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
processor = Qwen3OmniMoeProcessor.from_pretrained(model_id)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": "/data/request.wav"},
        {"type": "text", "text": "识别语音内容并用一句话回答。"},
    ],
}]
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
inputs = processor(
    text=text, audio=audios, images=images, videos=videos, return_tensors="pt",
    padding=True, use_audio_in_video=True,
).to(model.device).to(model.dtype)
text_output, audio = model.generate(
    **inputs,
    speaker="Ethan",
    thinker_return_dict_in_generate=True,
    use_audio_in_video=True,
)
answer_ids = text_output.sequences[:, inputs["input_ids"].shape[1]:]
print(processor.batch_decode(
    answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
))
if audio is not None:
    sf.write("response.wav", audio.reshape(-1).detach().cpu().numpy(), 24000)
```

若只需要文本回答，按模型 API 设置 `return_audio=False`，避免不必要的语音生成。音频、
图像和视频都应通过官方工具处理；实时流式交互还需要正确实现分块、时间对齐和取消机制，
上面的离线调用不是实时流服务。

## 生产部署注意

Transformers 封装位于 [model.py](./model.py)，可混合传入音频、图片和视频；设置 `--output-audio` 时生成并保存语音：

```bash
python architecture/qwen/Qwen3-Omni/inference.py \
  --audio /data/request.wav \
  --prompt "识别语音并简短回答。" \
  --output-audio /tmp/response.wav
```

- 官方推荐使用 vLLM-Omni 部署；支持范围与 Transformers 并不完全相同。上线前按指定
  vLLM-Omni 版本验证输入模态、输出模态、流式能力和 API schema。
- MoE 活跃参数量不代表全部权重都能放入小显存；权重加载、视觉/音频塔、KV cache、
  视频帧和语音输出都会增加峰值内存。
- 评测须同时覆盖 ASR/音频理解、图像问答、视频理解、语音生成自然度/内容一致性以及延迟。
- ms-swift 等框架的 Omni 微调支持范围可能只覆盖 Thinker 或特定任务；不要默认完整 Thinker-Talker 联训可用。

框架比较、数据准备和服务验收清单见 [`production/README.md`](../production/README.md)。

## 官方资料

- [Qwen3-Omni 官方代码、cookbooks、Transformers 与部署说明](https://github.com/QwenLM/Qwen3-Omni)
