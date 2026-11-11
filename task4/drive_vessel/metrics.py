from __future__ import annotations

import numpy as np


def binary_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    true_positive = np.logical_and(target, prediction).sum(dtype=np.float64)
    false_positive = np.logical_and(~target, prediction).sum(dtype=np.float64)
    false_negative = np.logical_and(target, ~prediction).sum(dtype=np.float64)
    true_negative = np.logical_and(~target, ~prediction).sum(dtype=np.float64)

    dice = (2.0 * true_positive) / max(2.0 * true_positive + false_positive + false_negative, 1.0)
    sensitivity = true_positive / max(true_positive + false_negative, 1.0)
    specificity = true_negative / max(true_negative + false_positive, 1.0)
    iou = true_positive / max(true_positive + false_positive + false_negative, 1.0)
    accuracy = (true_positive + true_negative) / target.size
    return {
        "dice": float(dice),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "iou": float(iou),
        "accuracy": float(accuracy),
    }


def summarize_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot summarize an empty metric list")
    keys = [
        key
        for key, value in rows[0].items()
        if key != "id" and isinstance(value, (int, float, np.floating))
    ]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}
