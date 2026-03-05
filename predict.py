from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with MacBERT emotion classifier")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--text", type=str, default=None, help="Single text input")
    parser.add_argument("--input-json", type=str, default=None, help="JSON list file with 'content'")
    parser.add_argument("--output-json", type=str, default="./predictions.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from macbert_emotion_classifier.predictor import EmotionPredictor

    predictor = EmotionPredictor(args.checkpoint, max_length=args.max_length)

    if args.text is not None:
        pred = predictor.predict_texts([args.text], batch_size=1)[0]
        print(json.dumps(pred, ensure_ascii=False, indent=2))
        return

    if args.input_json is None:
        raise ValueError("Either --text or --input-json must be provided.")

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("input-json must be a JSON list.")

    texts = [str(x.get("content", "")) for x in data]
    preds = predictor.predict_texts(texts, batch_size=args.batch_size)

    merged = []
    for raw, pred in zip(data, preds):
        item = dict(raw)
        item["pred_label"] = pred["pred_label"]
        item["confidence"] = pred["confidence"]
        item["probabilities"] = pred["probabilities"]
        merged.append(item)

    out_path = Path(args.output_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()
