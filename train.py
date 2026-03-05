from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from macbert_emotion_classifier import parse_train_args


if __name__ == "__main__":
    cfg = parse_train_args()
    from macbert_emotion_classifier.trainer import train

    train(cfg)
