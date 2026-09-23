"""Checkpoint-backed Transformers adapter for Qwen3-Omni."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class OmniGeneration:
    text: str
    audio: Any | None = None


class Qwen3OmniModel:
    """Load Qwen3-Omni and process text, image, audio, and video messages."""

    def __init__(self, model: Any, processor: Any) -> None:
        self.model = model
        self.processor = processor

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        *,
        device_map: str = "auto",
        dtype: str = "auto",
        attn_implementation: str | None = None,
        **load_kwargs: Any,
    ) -> "Qwen3OmniModel":
        try:
            from transformers import (
                Qwen3OmniMoeForConditionalGeneration,
                Qwen3OmniMoeProcessor,
            )
        except ImportError as exc:
            raise ImportError(
                "Install a Transformers version with Qwen3-Omni support."
            ) from exc
        model_kwargs = {"dtype": dtype, "device_map": device_map, **load_kwargs}
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_id, **model_kwargs
        )
        processor = Qwen3OmniMoeProcessor.from_pretrained(model_id)
        return cls(model, processor)

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int = 512,
        return_audio: bool = False,
        speaker: str = "Ethan",
        use_audio_in_video: bool = True,
        **generation_kwargs: Any,
    ) -> OmniGeneration:
        try:
            from qwen_omni_utils import process_mm_info
        except ImportError as exc:
            raise ImportError(
                "Install qwen-omni-utils to process multimodal inputs."
            ) from exc

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
            speaker=speaker,
            thinker_return_dict_in_generate=True,
            use_audio_in_video=use_audio_in_video,
            **generation_kwargs,
        )
        if isinstance(result, tuple):
            text_output, audio = result
        else:
            text_output, audio = result, None
        sequences = getattr(text_output, "sequences", text_output)
        answer_ids = sequences[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(
            answer_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return OmniGeneration(text=text, audio=audio)
