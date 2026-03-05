from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .config import TrainConfig
from .constants import EVAL_DIR, EVAL_LABELED_FILES, TRAIN_DIR, TRAIN_FILES


def _subset_keys(subset: str) -> List[str]:
    if subset == "all":
        return ["usual", "virus"]
    if subset in {"usual", "virus"}:
        return [subset]
    raise ValueError(f"Unsupported dataset_subset: {subset}")


def _resolve_files(dataset_root: str, subset: str, split: str) -> List[Path]:
    root = Path(dataset_root)
    keys = _subset_keys(subset)

    paths: List[Path] = []
    if split == "train":
        for key in keys:
            paths.append(root / TRAIN_DIR / TRAIN_FILES[key])
    elif split == "eval":
        for key in keys:
            paths.append(root / EVAL_DIR / EVAL_LABELED_FILES[key])
    else:
        raise ValueError(f"Unsupported split: {split}")

    missing = [p for p in paths if not p.exists()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing dataset files: {missing_str}")

    return paths


def _read_json_list(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def load_split_records(dataset_root: str, subset: str, split: str) -> List[Dict[str, str]]:
    files = _resolve_files(dataset_root=dataset_root, subset=subset, split=split)
    records: List[Dict[str, str]] = []
    for path in files:
        records.extend(_read_json_list(path))
    return records


def build_label_maps(labels: Sequence[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def prepare_examples(records: Iterable[Dict[str, str]], label2id: Dict[str, int]) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []

    for row in records:
        text = str(row.get("content", "")).strip()
        label = row.get("label")
        if not text:
            continue
        if label is None:
            continue
        if label not in label2id:
            continue

        texts.append(text)
        labels.append(label2id[label])

    if not texts:
        raise ValueError("No valid records after preprocessing.")

    return texts, labels


def compute_class_weights(label_ids: Sequence[int], num_labels: int) -> torch.Tensor:
    counts = Counter(label_ids)
    total = float(len(label_ids))

    weights = []
    for i in range(num_labels):
        c = float(counts.get(i, 0))
        if c == 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_labels * c))

    tensor = torch.tensor(weights, dtype=torch.float32)
    if tensor.sum() > 0:
        tensor = tensor / tensor.sum() * num_labels
    return tensor


class EmotionDataset(Dataset):
    def __init__(self, encodings: Dict[str, List[int]], labels: Sequence[int]) -> None:
        self.encodings = encodings
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: torch.tensor(v[idx], dtype=torch.long) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def _build_dataset(texts: Sequence[str], labels: Sequence[int], tokenizer, max_length: int) -> EmotionDataset:
    encodings = tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    return EmotionDataset(encodings=encodings, labels=labels)


def build_dataloaders(
    config: TrainConfig,
    tokenizer,
    label2id: Dict[str, int],
):
    train_records = load_split_records(config.dataset_root, config.dataset_subset, split="train")
    eval_records = load_split_records(config.dataset_root, config.dataset_subset, split="eval")

    train_texts, train_label_ids = prepare_examples(train_records, label2id)
    eval_texts, eval_label_ids = prepare_examples(eval_records, label2id)

    class_weights = compute_class_weights(train_label_ids, num_labels=len(label2id))

    train_dataset = _build_dataset(train_texts, train_label_ids, tokenizer, config.max_length)
    eval_dataset = _build_dataset(eval_texts, eval_label_ids, tokenizer, config.max_length)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    info = {
        "num_train": len(train_dataset),
        "num_eval": len(eval_dataset),
        "train_label_distribution": Counter(train_label_ids),
        "eval_label_distribution": Counter(eval_label_ids),
    }

    return train_loader, eval_loader, class_weights, info
