"""Span-based named entity recognition with training and strict evaluation.

Each JSONL line is expected to look like:

    {"text": "张三在北京工作", "entities": [
        {"start": 0, "end": 2, "label": "PER"},
        {"start": 3, "end": 5, "label": "LOC"}
    ]}

Offsets use Python string indices and ``end`` is exclusive.  The model
classifies token spans as ``NONE`` or an entity type, so nested entities are
representable as long as they have different boundaries.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


NONE_LABEL = "NONE"


@dataclass(frozen=True)
class SpanEntity:
    start: int
    end: int
    label: str


@dataclass(frozen=True)
class SpanNERExample:
    text: str
    entities: tuple[SpanEntity, ...]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_entity(entity: Mapping[str, Any], text: str, line_number: int) -> SpanEntity:
    try:
        start = int(entity["start"])
        end = int(entity["end"])
        label = str(entity["label"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid entity at line {line_number}; expected start/end/label"
        ) from exc
    if start < 0 or start >= end or end > len(text):
        raise ValueError(
            f"Invalid entity offsets at line {line_number}: "
            f"({start}, {end}) for text length {len(text)}"
        )
    if not label or label == NONE_LABEL:
        raise ValueError(f"Invalid entity label at line {line_number}: {label!r}")
    return SpanEntity(start, end, label)


def load_jsonl(
    path: str | Path,
    text_field: str = "text",
    entity_field: str = "entities",
) -> list[SpanNERExample]:
    """Load span annotations from JSONL."""
    examples: list[SpanNERExample] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if text_field not in record or entity_field not in record:
                raise ValueError(
                    f"{path}:{line_number} must contain "
                    f"{text_field!r} and {entity_field!r}"
                )
            text = str(record[text_field])
            entities = tuple(
                _parse_entity(entity, text, line_number)
                for entity in record[entity_field]
            )
            examples.append(SpanNERExample(text=text, entities=entities))
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples


def build_label_mapping(
    examples: Iterable[SpanNERExample],
    labels: Sequence[str] | None = None,
) -> tuple[dict[str, int], dict[int, str]]:
    """Reserve zero for NONE and assign stable ids to entity types."""
    if labels is None:
        values = sorted(
            {entity.label for example in examples for entity in example.entities}
        )
    else:
        values = list(labels)
    if NONE_LABEL in values:
        raise ValueError(f"{NONE_LABEL!r} is reserved for non-entity spans")
    if len(set(values)) != len(values):
        raise ValueError("labels contains duplicate values")
    label2id = {NONE_LABEL: 0}
    label2id.update({label: index for index, label in enumerate(values, start=1)})
    return label2id, {index: label for label, index in label2id.items()}


def _entity_token_span(
    entity: SpanEntity,
    offsets: Sequence[tuple[int, int]],
    valid_tokens: Sequence[bool],
) -> tuple[int, int] | None:
    """Map a character span to the first/last token covering it."""
    start_candidates = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if valid_tokens[index] and token_start <= entity.start < token_end
    ]
    end_candidates = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if valid_tokens[index] and token_start < entity.end <= token_end
    ]
    if not start_candidates or not end_candidates:
        return None
    start_token = min(start_candidates)
    end_token = max(end_candidates)
    if end_token < start_token or not all(valid_tokens[start_token : end_token + 1]):
        return None
    return start_token, end_token


class SpanNERDataset(Dataset[dict[str, torch.Tensor]]):
    """Encode text and create a fixed-width candidate span grid."""

    def __init__(
        self,
        examples: Sequence[SpanNERExample],
        tokenizer: Any,
        label2id: Mapping[str, int],
        max_length: int = 128,
        max_span_width: int = 10,
    ) -> None:
        if max_span_width < 1:
            raise ValueError("max_span_width must be positive")
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.label2id = dict(label2id)
        self.max_length = max_length
        self.max_span_width = max_span_width

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        encoded = self.tokenizer(
            example.text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0).bool()
        offsets = encoded["offset_mapping"].squeeze(0).long()
        offset_list = [tuple(pair.tolist()) for pair in offsets]
        valid_tokens = [
            bool(attention_mask[token_index]) and token_start < token_end
            for token_index, (token_start, token_end) in enumerate(offset_list)
        ]

        span_starts = torch.zeros(
            (self.max_length, self.max_span_width), dtype=torch.long
        )
        span_ends = torch.zeros_like(span_starts)
        span_mask = torch.zeros_like(span_starts, dtype=torch.bool)
        span_labels = torch.full_like(span_starts, fill_value=-100)

        for start in range(self.max_length):
            if not valid_tokens[start]:
                continue
            for width in range(1, self.max_span_width + 1):
                end = start + width - 1
                if end >= self.max_length or not all(valid_tokens[start : end + 1]):
                    continue
                span_starts[start, width - 1] = start
                span_ends[start, width - 1] = end
                span_mask[start, width - 1] = True
                span_labels[start, width - 1] = self.label2id[NONE_LABEL]

        for entity in example.entities:
            token_span = _entity_token_span(entity, offset_list, valid_tokens)
            if token_span is None:
                # The entity was truncated or cannot be aligned to a tokenizer span.
                continue
            start_token, end_token = token_span
            width = end_token - start_token + 1
            if width > self.max_span_width:
                continue
            if not span_mask[start_token, width - 1]:
                continue
            label_id = self.label2id.get(entity.label)
            if label_id is None:
                raise ValueError(f"Unknown entity label {entity.label!r}")
            old_label = span_labels[start_token, width - 1].item()
            if old_label not in (-100, self.label2id[NONE_LABEL], label_id):
                raise ValueError(
                    f"Two labels overlap on the same token span: "
                    f"{entity.start}:{entity.end}"
                )
            span_labels[start_token, width - 1] = label_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask.long(),
            "span_starts": span_starts,
            "span_ends": span_ends,
            "span_mask": span_mask,
            "span_labels": span_labels,
            "offset_mapping": offsets,
            "sample_index": torch.tensor(index, dtype=torch.long),
        }


class BertSpanNER(nn.Module):
    """BERT encoder followed by token-span classification."""

    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        max_span_width: int = 10,
        width_embedding_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        self.max_span_width = max_span_width
        self.width_embeddings = nn.Embedding(max_span_width + 1, width_embedding_dim)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2 + width_embedding_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_starts: torch.Tensor,
        span_ends: torch.Tensor,
        span_mask: torch.Tensor | None = None,
        span_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence = outputs.last_hidden_state
        batch_size = input_ids.shape[0]
        batch_indices = torch.arange(batch_size, device=input_ids.device).view(
            batch_size, 1, 1
        )
        start_repr = sequence[batch_indices, span_starts]
        end_repr = sequence[batch_indices, span_ends]
        widths = (span_ends - span_starts + 1).clamp(
            min=1, max=self.max_span_width
        )
        width_repr = self.width_embeddings(widths)
        features = torch.cat((start_repr, end_repr, width_repr), dim=-1)
        logits = self.classifier(self.dropout(features))
        result = {"logits": logits}
        if span_labels is not None:
            result["loss"] = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                span_labels.reshape(-1),
                ignore_index=-100,
            )
        return result


def _move_batch(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _model_batch(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    model_keys = {
        "input_ids",
        "attention_mask",
        "span_starts",
        "span_ends",
        "span_mask",
        "span_labels",
    }
    return {key: value for key, value in batch.items() if key in model_keys}


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    device: torch.device,
    gradient_clip_norm: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="train", leave=False):
        batch = _move_batch(batch, device)
        output = model(**_model_batch(batch))
        loss = output["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / max(len(dataloader), 1)


def _decode_batch(
    logits: torch.Tensor,
    span_mask: torch.Tensor,
    offsets: torch.Tensor,
    id2label: Mapping[int, str],
) -> list[set[tuple[int, int, str]]]:
    predicted_ids = logits.argmax(dim=-1)
    decoded: list[set[tuple[int, int, str]]] = []
    for batch_index in range(logits.shape[0]):
        entities: set[tuple[int, int, str]] = set()
        for start in range(logits.shape[1]):
            for width_index in range(logits.shape[2]):
                if not span_mask[batch_index, start, width_index]:
                    continue
                label_id = int(predicted_ids[batch_index, start, width_index])
                if label_id == 0:
                    continue
                end = start + width_index
                char_start = int(offsets[batch_index, start, 0])
                char_end = int(offsets[batch_index, end, 1])
                entities.add((char_start, char_end, id2label[label_id]))
        decoded.append(entities)
    return decoded


def _gold_entities(example: SpanNERExample) -> set[tuple[int, int, str]]:
    return {(entity.start, entity.end, entity.label) for entity in example.entities}


def span_metrics(
    gold: Sequence[set[tuple[int, int, str]]],
    predicted: Sequence[set[tuple[int, int, str]]],
) -> dict[str, Any]:
    """Compute exact-match micro metrics and per-entity-type metrics."""
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must have the same number of examples")
    true_positive = sum(len(gold_set & pred_set) for gold_set, pred_set in zip(gold, predicted))
    total_gold = sum(len(items) for items in gold)
    total_predicted = sum(len(items) for items in predicted)
    precision = true_positive / total_predicted if total_predicted else 0.0
    recall = true_positive / total_gold if total_gold else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    labels = sorted(
        {entity[2] for items in gold for entity in items}
        | {entity[2] for items in predicted for entity in items}
    )
    per_type: dict[str, dict[str, float]] = {}
    for label in labels:
        gold_count = sum(sum(entity[2] == label for entity in items) for items in gold)
        predicted_count = sum(
            sum(entity[2] == label for entity in items) for items in predicted
        )
        hits = sum(
            len(
                {
                    entity
                    for entity in gold_set & pred_set
                    if entity[2] == label
                }
            )
            for gold_set, pred_set in zip(gold, predicted)
        )
        type_precision = hits / predicted_count if predicted_count else 0.0
        type_recall = hits / gold_count if gold_count else 0.0
        type_f1 = (
            2 * type_precision * type_recall / (type_precision + type_recall)
            if type_precision + type_recall
            else 0.0
        )
        per_type[label] = {
            "precision": type_precision,
            "recall": type_recall,
            "f1": type_f1,
            "support": float(gold_count),
        }
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "gold_count": total_gold,
        "predicted_count": total_predicted,
        "per_type": per_type,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    examples: Sequence[SpanNERExample],
    device: torch.device,
    id2label: Mapping[int, str],
) -> dict[str, Any]:
    """Evaluate strict span precision/recall/F1."""
    model.eval()
    all_predictions: list[set[tuple[int, int, str]]] = []
    all_gold: list[set[tuple[int, int, str]]] = []
    for batch in tqdm(dataloader, desc="evaluate", leave=False):
        moved = _move_batch(batch, device)
        output = model(**_model_batch(moved))
        decoded = _decode_batch(
            output["logits"],
            moved["span_mask"].bool(),
            moved["offset_mapping"],
            id2label,
        )
        all_predictions.extend(decoded)
        sample_indices = batch["sample_index"].tolist()
        all_gold.extend(_gold_entities(examples[index]) for index in sample_indices)
    metrics = span_metrics(all_gold, all_predictions)
    metrics["predictions"] = all_predictions
    metrics["gold"] = all_gold
    return metrics


def predict(
    text: str,
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    id2label: Mapping[int, str],
    *,
    max_length: int = 128,
    max_span_width: int = 10,
    threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """Predict all positive spans for one text, including nested spans."""
    example = SpanNERExample(text=text, entities=())
    label2id = {label: index for index, label in id2label.items()}
    dataset = SpanNERDataset(
        [example],
        tokenizer,
        label2id,
        max_length=max_length,
        max_span_width=max_span_width,
    )
    batch = {
        key: value.unsqueeze(0)
        for key, value in dataset[0].items()
        if key in {
            "input_ids",
            "attention_mask",
            "span_starts",
            "span_ends",
            "span_mask",
        }
    }
    batch = _move_batch(batch, device)
    model.eval()
    with torch.no_grad():
        logits = model(**batch)["logits"]
        probabilities = logits.softmax(dim=-1)
    results: list[dict[str, Any]] = []
    span_mask = batch["span_mask"].bool()[0]
    predicted_ids = probabilities.argmax(dim=-1)[0]
    offsets = dataset[0]["offset_mapping"]
    for start in range(span_mask.shape[0]):
        for width_index in range(span_mask.shape[1]):
            if not span_mask[start, width_index]:
                continue
            label_id = int(predicted_ids[start, width_index])
            score = float(probabilities[0, start, width_index, label_id])
            if label_id == 0 or score < threshold:
                continue
            end = start + width_index
            char_start = int(offsets[start, 0])
            char_end = int(offsets[end, 1])
            results.append(
                {
                    "text": text[char_start:char_end],
                    "start": char_start,
                    "end": char_end,
                    "label": id2label[label_id],
                    "score": score,
                }
            )
    return sorted(results, key=lambda item: (item["start"], item["end"], item["label"]))


def train_span_ner(
    train_examples: Sequence[SpanNERExample],
    valid_examples: Sequence[SpanNERExample],
    model_name_or_path: str,
    output_path: str | Path,
    *,
    label2id: Mapping[str, int] | None = None,
    max_length: int = 128,
    max_span_width: int = 10,
    batch_size: int = 8,
    epochs: int = 5,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    seed: int = 42,
    device: str | torch.device | None = None,
) -> tuple[nn.Module, dict[int, str], dict[str, Any]]:
    """Train and checkpoint a span NER model using validation F1."""
    set_seed(seed)
    if label2id is None:
        label2id, id2label = build_label_mapping(train_examples)
    else:
        label2id = dict(label2id)
        id2label = {index: label for label, index in label2id.items()}
    valid_labels = {
        entity.label for example in valid_examples for entity in example.entities
    }
    missing = valid_labels - set(label2id)
    if missing:
        raise ValueError(f"Validation labels are absent from training labels: {missing}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    train_dataset = SpanNERDataset(
        train_examples,
        tokenizer,
        label2id,
        max_length=max_length,
        max_span_width=max_span_width,
    )
    valid_dataset = SpanNERDataset(
        valid_examples,
        tokenizer,
        label2id,
        max_length=max_length,
        max_span_width=max_span_width,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    selected_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = BertSpanNER(
        model_name_or_path,
        num_labels=len(label2id),
        max_span_width=max_span_width,
    ).to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    total_steps = max(len(train_loader) * epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )

    best_f1 = float("-inf")
    best_metrics: dict[str, Any] = {}
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, selected_device)
        metrics = evaluate(model, valid_loader, valid_examples, selected_device, id2label)
        print(
            f"epoch={epoch} loss={loss:.4f} "
            f"precision={metrics['precision']:.4f} "
            f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
        )
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_metrics = metrics
            checkpoint_path = Path(output_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                    "model_name_or_path": model_name_or_path,
                    "label2id": label2id,
                    "max_length": max_length,
                    "max_span_width": max_span_width,
                },
                checkpoint_path,
            )
    return model, id2label, best_metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a span-based NER model.")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--valid-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path)
    parser.add_argument("--model", default="bert-base-chinese")
    parser.add_argument("--output", type=Path, default=Path("outputs/span_ner.pt"))
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--entity-field", default="entities")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-span-width", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    train_examples = load_jsonl(args.train_file, args.text_field, args.entity_field)
    valid_examples = load_jsonl(args.valid_file, args.text_field, args.entity_field)
    model, id2label, metrics = train_span_ner(
        train_examples,
        valid_examples,
        args.model,
        args.output,
        max_length=args.max_length,
        max_span_width=args.max_span_width,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
    )
    del model
    print(
        f"best precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
    )
    if args.test_file:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        label2id = {label: index for index, label in id2label.items()}
        test_examples = load_jsonl(args.test_file, args.text_field, args.entity_field)
        test_dataset = SpanNERDataset(
            test_examples,
            tokenizer,
            label2id,
            max_length=args.max_length,
            max_span_width=args.max_span_width,
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        selected_device = torch.device(
            args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint = torch.load(
            args.output,
            map_location=selected_device,
            weights_only=False,
        )
        test_model = BertSpanNER(
            args.model,
            num_labels=len(id2label),
            max_span_width=args.max_span_width,
        ).to(selected_device)
        test_model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate(
            test_model, test_loader, test_examples, selected_device, id2label
        )
        print(
            f"test precision={test_metrics['precision']:.4f} "
            f"recall={test_metrics['recall']:.4f} f1={test_metrics['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
