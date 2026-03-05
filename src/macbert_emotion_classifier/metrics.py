from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    num_labels: int,
) -> Dict[str, float]:
    true_arr = np.asarray(list(y_true), dtype=np.int64)
    pred_arr = np.asarray(list(y_pred), dtype=np.int64)

    if true_arr.size == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0}

    conf = np.zeros((num_labels, num_labels), dtype=np.int64)
    for t, p in zip(true_arr, pred_arr):
        conf[t, p] += 1

    tp = np.diag(conf).astype(np.float64)
    pred_sum = conf.sum(axis=0).astype(np.float64)
    true_sum = conf.sum(axis=1).astype(np.float64)

    precision = np.divide(tp, pred_sum, out=np.zeros_like(tp), where=pred_sum > 0)
    recall = np.divide(tp, true_sum, out=np.zeros_like(tp), where=true_sum > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )

    accuracy = float((true_arr == pred_arr).mean())
    macro_f1 = float(f1.mean())

    return {"accuracy": accuracy, "macro_f1": macro_f1}
