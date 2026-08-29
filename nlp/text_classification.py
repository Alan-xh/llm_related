"""BERT text classification training and evaluation utilities.

The module is intentionally self-contained so it can be used as either:

    python -m nlp.text_classification --train-file train.jsonl ...

or as a small library from another experiment.

JSONL input uses one object per line.  The default fields are ``text`` and
``label``; the label field can be changed from the command line.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


@dataclass(frozen=True)
class TextClassificationExample:
    text: str
    label: str | int


def set_seed(seed: int) -> None:
    """Make data order and model initialization repeatable where possible."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(
    path: str | Path,
    text_field: str = "text",
    label_field: str = "label",
) -> list[TextClassificationExample]:
    """Load a classification split from JSONL."""
    examples: list[TextClassificationExample] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if text_field not in record or label_field not in record:
                raise ValueError(
                    f"{path}:{line_number} must contain "
                    f"{text_field!r} and {label_field!r}"
                )
            text = str(record[text_field]).strip()
            if not text:
                raise ValueError(f"{path}:{line_number} contains empty text")
            examples.append(TextClassificationExample(text, record[label_field]))
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples


def build_label_mapping(
    examples: Iterable[TextClassificationExample],
    labels: Sequence[str | int] | None = None,
) -> tuple[dict[str | int, int], dict[int, str | int]]:
    """Build a stable label mapping, preserving an optional user order."""
    if labels is None:
        label_values = sorted({example.label for example in examples}, key=str)
    else:
        label_values = list(labels)
    if not label_values:
        raise ValueError("At least one label is required")
    if len(set(label_values)) != len(label_values):
        raise ValueError("labels contains duplicate values")
    label2id = {label: index for index, label in enumerate(label_values)}
    id2label = {index: label for label, index in label2id.items()}
    return label2id, id2label


class TextClassificationDataset(Dataset[dict[str, torch.Tensor]]):
    """Tokenized dataset for single-label text classification."""

    def __init__(
        self,
        examples: Sequence[TextClassificationExample],
        tokenizer: Any,
        label2id: Mapping[str | int, int] | None = None,
        max_length: int = 128,
    ) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        encoded = self.tokenizer(
            example.text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        if self.label2id is not None:
            if example.label not in self.label2id:
                raise ValueError(f"Unknown label {example.label!r}")
            item["labels"] = torch.tensor(
                self.label2id[example.label], dtype=torch.long
            )
        return item


class BertForTextClassification(nn.Module):
    """Transformer encoder with a dropout classification head."""

    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**model_inputs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            mask = attention_mask.unsqueeze(-1).to(outputs.last_hidden_state.dtype)
            pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        logits = self.classifier(self.dropout(pooled))
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = nn.functional.cross_entropy(logits, labels)
        return result


def _move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


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
        output = model(**batch)
        loss = output["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def predict_ids(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    model.eval()
    predictions: list[int] = []
    targets: list[int] = []
    for batch in tqdm(dataloader, desc="evaluate", leave=False):
        batch = _move_batch(batch, device)
        labels = batch.pop("labels")
        logits = model(**batch)["logits"]
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
        targets.extend(labels.cpu().tolist())
    return predictions, targets


def classification_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    id2label: Mapping[int, str | int],
) -> dict[str, Any]:
    """Return scalar metrics and a per-label sklearn report."""
    label_ids = sorted(id2label)
    target_names = [str(id2label[index]) for index in label_ids]
    return {
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(
            targets,
            predictions,
            labels=label_ids,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            targets,
            predictions,
            labels=label_ids,
            average="weighted",
            zero_division=0,
        ),
        "report": classification_report(
            targets,
            predictions,
            labels=label_ids,
            target_names=target_names,
            zero_division=0,
            digits=4,
        ),
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    id2label: Mapping[int, str | int],
) -> dict[str, Any]:
    predictions, targets = predict_ids(model, dataloader, device)
    metrics = classification_metrics(targets, predictions, id2label)
    metrics["predictions"] = predictions
    metrics["targets"] = targets
    return metrics


def predict(
    text: str,
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    id2label: Mapping[int, str | int],
    max_length: int = 128,
) -> dict[str, Any]:
    """Predict one text and return the label and class probabilities."""
    model.eval()
    encoded = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    encoded = _move_batch(encoded, device)
    with torch.no_grad():
        probabilities = model(**encoded)["logits"].softmax(dim=-1)[0]
    label_id = int(probabilities.argmax().item())
    return {
        "label": id2label[label_id],
        "label_id": label_id,
        "score": float(probabilities[label_id].item()),
        "probabilities": {
            id2label[index]: float(probabilities[index].item())
            for index in range(len(probabilities))
        },
    }


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Mapping[str, Any],
    model_name_or_path: str,
    label2id: Mapping[str | int, int],
    max_length: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": dict(metrics),
            "model_name_or_path": model_name_or_path,
            "label2id": dict(label2id),
            "max_length": max_length,
        },
        path,
    )


def train_classifier(
    train_examples: Sequence[TextClassificationExample],
    valid_examples: Sequence[TextClassificationExample],
    model_name_or_path: str,
    output_path: str | Path,
    *,
    label2id: Mapping[str | int, int] | None = None,
    max_length: int = 128,
    batch_size: int = 16,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    seed: int = 42,
    device: str | torch.device | None = None,
) -> tuple[nn.Module, dict[int, str | int], dict[str, Any]]:
    """Train a classifier and save the best validation checkpoint."""
    set_seed(seed)
    if label2id is None:
        label2id, id2label = build_label_mapping(train_examples)
    else:
        label2id = dict(label2id)
        id2label = {index: label for label, index in label2id.items()}
    unknown = {example.label for example in valid_examples} - set(label2id)
    if unknown:
        raise ValueError(f"Validation labels are absent from training labels: {unknown}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    train_dataset = TextClassificationDataset(
        train_examples, tokenizer, label2id, max_length
    )
    valid_dataset = TextClassificationDataset(
        valid_examples, tokenizer, label2id, max_length
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    selected_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = BertForTextClassification(
        model_name_or_path, num_labels=len(label2id)
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

    best_score = float("-inf")
    best_metrics: dict[str, Any] = {}
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, selected_device)
        metrics = evaluate(model, valid_loader, selected_device, id2label)
        print(
            f"epoch={epoch} loss={loss:.4f} "
            f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}"
        )
        if metrics["macro_f1"] > best_score:
            best_score = metrics["macro_f1"]
            best_metrics = metrics
            _save_checkpoint(
                Path(output_path),
                model,
                optimizer,
                epoch,
                metrics,
                model_name_or_path,
                label2id,
                max_length,
            )
    return model, id2label, best_metrics


def build_arg_parser(
    default_label_field: str = "label",
    description: str = "Train a BERT text classifier.",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--valid-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path)
    parser.add_argument("--model", default="bert-base-chinese")
    parser.add_argument("--output", type=Path, default=Path("outputs/text_classifier.pt"))
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--label-field", default=default_label_field)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    default_label_field: str = "label",
    description: str = "Train a BERT text classifier.",
) -> None:
    args = build_arg_parser(default_label_field, description).parse_args(argv)
    train_examples = load_jsonl(args.train_file, args.text_field, args.label_field)
    valid_examples = load_jsonl(args.valid_file, args.text_field, args.label_field)
    model, id2label, metrics = train_classifier(
        train_examples,
        valid_examples,
        args.model,
        args.output,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
    )
    del model
    print(metrics["report"])
    if args.test_file:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        label2id = {label: index for index, label in id2label.items()}
        test_dataset = TextClassificationDataset(
            load_jsonl(args.test_file, args.text_field, args.label_field),
            tokenizer,
            label2id,
            args.max_length,
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        checkpoint = torch.load(
            args.output,
            map_location=args.device or "cpu",
            weights_only=False,
        )
        test_model = BertForTextClassification(
            args.model, len(id2label)
        ).to(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        test_model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate(
            test_model,
            test_loader,
            next(test_model.parameters()).device,
            id2label,
        )
        print("test")
        print(test_metrics["report"])


if __name__ == "__main__":
    main()
