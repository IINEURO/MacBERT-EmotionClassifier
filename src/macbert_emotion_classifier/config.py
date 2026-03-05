from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .constants import DEFAULT_LABELS


@dataclass
class TrainConfig:
    seed: int = 3407
    model_name: str = "hfl/chinese-macbert-base"
    max_length: int = 128
    batch_size: int = 32
    eval_batch_size: int = 64
    epochs: int = 5
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    fp16: bool = True
    num_workers: int = 2
    dataset_subset: str = "all"
    dataset_root: str = "./WBDataSet"
    output_dir: str = "./outputs/macbert-base-wbdataset"
    save_best_only: bool = True
    labels: List[str] = field(default_factory=lambda: DEFAULT_LABELS.copy())


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping at top-level.")
    return data


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MacBERT emotion classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_macbert_base.yaml",
        help="Path to yaml config file",
    )

    # Optional overrides
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--dataset-subset", type=str, choices=["usual", "virus", "all"], default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--fp16", action="store_true", help="Force enable fp16")
    parser.add_argument("--no-fp16", action="store_true", help="Force disable fp16")
    return parser


def parse_train_args() -> TrainConfig:
    parser = _make_parser()
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    raw = _load_yaml(config_path)

    cfg = TrainConfig(**{k: v for k, v in raw.items() if hasattr(TrainConfig, k)})

    if args.dataset_root is not None:
        cfg.dataset_root = args.dataset_root
    if args.dataset_subset is not None:
        cfg.dataset_subset = args.dataset_subset
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.eval_batch_size is not None:
        cfg.eval_batch_size = args.eval_batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.max_length is not None:
        cfg.max_length = args.max_length

    if args.fp16 and args.no_fp16:
        raise ValueError("Cannot set both --fp16 and --no-fp16")
    if args.fp16:
        cfg.fp16 = True
    if args.no_fp16:
        cfg.fp16 = False

    return cfg
