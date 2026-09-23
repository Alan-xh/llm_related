"""Checkpoint-backed Transformers adapter for Qwen2.5-Omni."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class OmniGeneration:
    text: str
    audio: Any | None = None


class Qwen2_5OmniModel:
    """Load Qwen2.5-Omni and process text, image, audio, and video messages."""

    def __init__(self, model: Any, processor: Any) -> None:
        self.model = model
        self.processor = processor

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "Qwen/Qwen2.5-Omni-7B",
        *,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        enable_audio_output: bool = True,
        **load_kwargs: Any,
    ) -> "Qwen2_5OmniModel":
        try:
            from transformers import (
                Qwen2_5OmniForConditionalGeneration,
                Qwen2_5OmniProcessor,
            )
        except ImportError as exc:
            raise ImportError(
                "Install a Transformers version with Qwen2.5-Omni support."
            ) from exc
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            enable_audio_output=enable_audio_output,
            **load_kwargs,
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
        return cls(model, processor)

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int = 512,
        return_audio: bool = False,
        use_audio_in_video: bool = True,
        **generation_kwargs: Any,
    ) -> OmniGeneration:
        try:
            from qwen_omni_utils import process_mm_info
        except ImportError as exc:
            raise ImportError(
                "Install qwen-omni-utils to process multimodal inputs."
            ) from exc

        if return_audio and not any(message.get("role") == "system" for message in messages):
            messages = [
                {
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": (
                            "You are Qwen, a virtual human developed by the Qwen Team, "
                            "Alibaba Group, capable of perceiving auditory and visual "
                            "inputs, as well as generating text and speech."
                        ),
                    }],
                },
                *messages,
            ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        audios, images, videos = process_mm_info(
            messages, use_audio_in_video=use_audio_in_video
        )
        inputs = self.processor(
            text=prompt,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video,
        ).to(self.model.device).to(self.model.dtype)
        result = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_audio=return_audio,
            use_audio_in_video=use_audio_in_video,
            **generation_kwargs,
        )
        if return_audio:
            text_ids, audio = result
        else:
            text_ids, audio = result, None
        text = self.processor.batch_decode(
            text_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return OmniGeneration(text=text, audio=audio)
