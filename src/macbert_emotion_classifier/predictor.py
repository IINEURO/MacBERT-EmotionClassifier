from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class EmotionPredictor:
    def __init__(self, checkpoint_dir: str, max_length: int = 128) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_texts(self, texts: Sequence[str], batch_size: int = 64) -> List[Dict]:
        if not texts:
            return []

        id2label = self.model.config.id2label
        outputs: List[Dict] = []

        for start in range(0, len(texts), batch_size):
            batch_texts = list(texts[start : start + batch_size])
            enc = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            conf, pred = torch.max(probs, dim=-1)

            for text, p, c, prob in zip(batch_texts, pred.tolist(), conf.tolist(), probs.tolist()):
                outputs.append(
                    {
                        "content": text,
                        "pred_label": id2label[str(p)] if isinstance(next(iter(id2label.keys())), str) else id2label[p],
                        "confidence": float(c),
                        "probabilities": [float(x) for x in prob],
                    }
                )

        return outputs
