from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .config import TrainConfig
from .data import build_dataloaders, build_label_maps
from .metrics import classification_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, num_labels: int) -> Dict[str, float]:
    model.eval()

    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    for batch in tqdm(data_loader, desc="Eval", leave=False):
        labels = batch["labels"].to(device)
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}

        outputs = model(**inputs)
        logits = outputs.logits
        loss = criterion(logits, labels)

        losses.append(float(loss.detach().cpu().item()))
        preds = torch.argmax(logits, dim=-1)

        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())

    metrics = classification_metrics(y_true=y_true, y_pred=y_pred, num_labels=num_labels)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def _save_train_artifacts(
    output_dir: Path,
    config: TrainConfig,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    info: Dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        **config.__dict__,
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "dataset_info": {
            "num_train": info.get("num_train", 0),
            "num_eval": info.get("num_eval", 0),
            "train_label_distribution": {
                str(k): int(v) for k, v in info.get("train_label_distribution", {}).items()
            },
            "eval_label_distribution": {
                str(k): int(v) for k, v in info.get("eval_label_distribution", {}).items()
            },
        },
    }

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, ensure_ascii=False, indent=2)


def train(config: TrainConfig) -> Tuple[Path, Dict[str, float]]:
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = bool(config.fp16 and device.type == "cuda")

    output_dir = Path(config.output_dir).resolve()
    best_dir = output_dir / "best_model"

    label2id, id2label = build_label_maps(config.labels)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    train_loader, eval_loader, class_weights, info = build_dataloaders(
        config=config,
        tokenizer=tokenizer,
        label2id=label2id,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(config.labels),
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)

    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    updates_per_epoch = math.ceil(len(train_loader) / max(1, config.grad_accum_steps))
    total_steps = max(1, updates_per_epoch * config.epochs)
    warmup_steps = int(total_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    _save_train_artifacts(output_dir, config, label2id, id2label, info)

    best_macro_f1 = -1.0
    best_metrics: Dict[str, float] = {}

    print(f"Device: {device}")
    print(f"FP16 enabled: {use_fp16}")
    print(f"Train samples: {info['num_train']} | Eval samples: {info['num_eval']}")

    global_step = 0
    for epoch in range(1, config.epochs + 1):
        model.train()

        running_loss = 0.0
        epoch_true: List[int] = []
        epoch_pred: List[int] = []

        optimizer.zero_grad(set_to_none=True)

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}", leave=False)
        for step, batch in enumerate(progress, start=1):
            labels = batch["labels"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}

            with torch.cuda.amp.autocast(enabled=use_fp16):
                outputs = model(**inputs)
                logits = outputs.logits
                loss = criterion(logits, labels)
                loss = loss / max(1, config.grad_accum_steps)

            scaler.scale(loss).backward()

            if step % config.grad_accum_steps == 0 or step == len(train_loader):
                if config.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            raw_loss = float(loss.detach().cpu().item() * max(1, config.grad_accum_steps))
            running_loss += raw_loss

            preds = torch.argmax(logits.detach(), dim=-1)
            epoch_true.extend(labels.detach().cpu().tolist())
            epoch_pred.extend(preds.detach().cpu().tolist())

            if global_step > 0:
                progress.set_postfix(
                    loss=f"{raw_loss:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

        train_metrics = classification_metrics(epoch_true, epoch_pred, num_labels=len(config.labels))
        train_metrics["loss"] = running_loss / max(1, len(train_loader))

        eval_metrics = evaluate(
            model=model,
            data_loader=eval_loader,
            criterion=criterion,
            device=device,
            num_labels=len(config.labels),
        )

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"train_acc={train_metrics['accuracy']:.4f}, "
            f"train_macro_f1={train_metrics['macro_f1']:.4f}, "
            f"val_loss={eval_metrics['loss']:.4f}, "
            f"val_acc={eval_metrics['accuracy']:.4f}, "
            f"val_macro_f1={eval_metrics['macro_f1']:.4f}"
        )

        current_macro_f1 = eval_metrics["macro_f1"]
        should_save = (current_macro_f1 > best_macro_f1) or (not config.save_best_only)

        if should_save:
            best_macro_f1 = max(best_macro_f1, current_macro_f1)
            best_metrics = eval_metrics.copy()

            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

            with (best_dir / "metrics.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "epoch": epoch,
                        "metrics": eval_metrics,
                        "best_macro_f1": best_macro_f1,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    print(f"Training finished. Best macro_f1={best_macro_f1:.4f}")
    print(f"Best checkpoint: {best_dir}")

    return best_dir, best_metrics
