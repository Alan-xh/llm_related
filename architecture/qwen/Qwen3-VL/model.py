"""Checkpoint-backed Transformers adapter for Qwen3-VL."""

from __future__ import annotations

from typing import Any


class Qwen3VLModel:
    """Load an official Qwen3-VL checkpoint and run image/video chat."""

    def __init__(self, model: Any, processor: Any) -> None:
        self.model = model
        self.processor = processor

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        *,
        device_map: str = "auto",
        dtype: str = "auto",
        attn_implementation: str | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        **load_kwargs: Any,
    ) -> "Qwen3VLModel":
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Install a Transformers version with Qwen3-VL support."
            ) from exc

        model_kwargs = {"dtype": dtype, "device_map": device_map, **load_kwargs}
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
        processor = AutoProcessor.from_pretrained(
            model_id, **(processor_kwargs or {})
        )
        return cls(model, processor)

    def _prepare_inputs(self, messages: list[dict[str, Any]]) -> Any:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise ImportError("Install qwen-vl-utils to process image/video inputs.") from exc

        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        video_metadata = None
        if videos is not None:
            videos, video_metadata = zip(*videos)
            videos = list(videos)
            video_metadata = list(video_metadata)
        inputs = self.processor(
            text=[prompt],
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        )
        return inputs.to(self.model.device)

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int = 512,
        **generation_kwargs: Any,
    ) -> str:
        inputs = self._prepare_inputs(messages)
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
