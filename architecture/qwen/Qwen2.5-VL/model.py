"""Checkpoint-backed Transformers adapter for Qwen2.5-VL."""

from __future__ import annotations

from typing import Any


class Qwen2_5VLModel:
    """Load an official Qwen2.5-VL checkpoint and run image/video chat."""

    def __init__(self, model: Any, processor: Any) -> None:
        self.model = model
        self.processor = processor

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        *,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        attn_implementation: str | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        **load_kwargs: Any,
    ) -> "Qwen2_5VLModel":
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Install a Transformers version with Qwen2.5-VL support."
            ) from exc

        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            **load_kwargs,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, **model_kwargs
        )
        processor = AutoProcessor.from_pretrained(
            model_id, **(processor_kwargs or {})
        )
        return cls(model, processor)

    @staticmethod
    def _prepare_media(messages: list[dict[str, Any]]) -> tuple[Any, Any, dict[str, Any]]:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise ImportError("Install qwen-vl-utils to process image/video inputs.") from exc

        images, videos, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        return images, videos, video_kwargs

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int = 512,
        **generation_kwargs: Any,
    ) -> str:
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos, video_kwargs = self._prepare_media(messages)
        inputs = self.processor(
            text=[prompt],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(self.model.device)
        generated = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, **generation_kwargs
        )
        answer_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated)
        ]
        return self.processor.batch_decode(
            answer_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


Qwen2_5_VLForConditionalGenerationAdapter = Qwen2_5VLModel
