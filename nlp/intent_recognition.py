"""Intent recognition built on the shared text classification pipeline.

The only convention specific to this task is that labels are read from the
``intent`` JSONL field by default:

    {"text": "帮我查询订单", "intent": "query_order"}

Run with:

    python -m nlp.intent_recognition --train-file train.jsonl --valid-file valid.jsonl
"""

from __future__ import annotations

from typing import Sequence

from .text_classification import (
    BertForTextClassification as BertForIntentClassification,
    TextClassificationDataset as IntentDataset,
    TextClassificationExample as IntentExample,
    build_label_mapping,
    classification_metrics,
    evaluate,
    load_jsonl,
    predict,
    predict_ids,
    set_seed,
    train_classifier,
    train_epoch,
)
from .text_classification import main as _classification_main


def load_intent_jsonl(path: str, text_field: str = "text", intent_field: str = "intent"):
    """Load intent examples using an explicit ``intent`` field."""
    return load_jsonl(path, text_field=text_field, label_field=intent_field)


def main(argv: Sequence[str] | None = None) -> None:
    _classification_main(
        argv,
        default_label_field="intent",
        description="Train a BERT intent recognition model.",
    )


if __name__ == "__main__":
    main()


__all__ = [
    "BertForIntentClassification",
    "IntentDataset",
    "IntentExample",
    "build_label_mapping",
    "classification_metrics",
    "evaluate",
    "load_intent_jsonl",
    "predict",
    "predict_ids",
    "set_seed",
    "train_classifier",
    "train_epoch",
]
