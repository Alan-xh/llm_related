# Qwen2.5-Omni 实际使用

Qwen2.5-Omni 是端到端多模态模型，可处理文本、图像、音频和视频，并支持文本与语音响应。
其 Thinker-Talker 结构及音视频预处理不等同于把 VL 模型与 ASR/TTS 服务简单串接。

## Transformers 推理

运行环境需要支持 Qwen2.5-Omni 的 Transformers 版本、`qwen-omni-utils`、PyTorch，以及媒体解码相关依赖。
版本请依据官方仓库当前说明锁定。下面示例以视频输入并请求语音输出：

```python
import soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

model_id = "Qwen/Qwen2.5-Omni-7B"
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)
processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
messages = [{
    "role": "user",
    "content": [
        {"type": "video", "video": "/data/meeting.mp4"},
        {"type": "text", "text": "用中文概括讨论内容。"},
    ],
}]
use_audio_in_video = True
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
audios, images, videos = process_mm_info(
    messages, use_audio_in_video=use_audio_in_video
)
inputs = processor(
    text=text,
    audio=audios,
    images=images,
    videos=videos,
    return_tensors="pt",
    padding=True,
    use_audio_in_video=use_audio_in_video,
).to(model.device).to(model.dtype)
text_ids, audio = model.generate(
    **inputs, use_audio_in_video=use_audio_in_video
)
print(processor.batch_decode(
    text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
))
if audio is not None:
    sf.write("response.wav", audio.reshape(-1).detach().cpu().numpy(), 24000)
```

仅需文本结果时，应按官方 API 设置关闭音频输出，减少生成开销。音视频联合输入时，
`use_audio_in_video` 必须在媒体处理和模型生成配置中保持一致。不要把视频音轨是否参与理解留作隐式默认值。

## 部署与训练边界

本地封装位于 [model.py](./model.py)，请求语音输出时会启用 Talker 并补入官方建议的 system prompt：

```bash
python architecture/qwen/Qwen2.5-Omni/inference.py \
  --audio /data/request.wav \
  --prompt "识别语音并用中文回答。" \
  --output-audio /tmp/response.wav
```

- 官方提供专用运行环境/示例；普通 Transformers 环境遇到版本或媒体依赖冲突时，应优先对齐官方依赖组合。
- vLLM 适配可能需要特定分支/版本；不要照搬通用 `vllm serve` 命令，先核验官方 Omni 部署页和后端对输出音频的支持范围。
- ms-swift 的 Omni 微调能力有组件边界；其文档说明 Qwen2.5-Omni 支持 Thinker 侧训练、不支持 Talker 训练。若目标是端到端语音生成微调，应另行验证训练目标和完整音频输出链路。
- 媒体服务需加上视频时长、帧数、音频时长、并发数、超时和采样率限制。

框架选型和多模态数据注意事项见 [`production/README.md`](../production/README.md)。

## 官方资料

- [Qwen2.5-Omni 官方代码、cookbooks 与运行说明](https://github.com/QwenLM/Qwen2.5-Omni)
