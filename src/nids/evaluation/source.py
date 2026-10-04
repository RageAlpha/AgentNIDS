from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_probability_metrics(
    labels: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    epsilon: float,
    ece_bins: int,
) -> dict:
    labels = labels.astype(np.uint8, copy=False)
    probability = probability.astype(np.float64, copy=False)
    prediction = (probability >= threshold).astype(np.uint8)
    tn = int(np.count_nonzero((labels == 0) & (prediction == 0)))
    fp = int(np.count_nonzero((labels == 0) & (prediction == 1)))
    fn = int(np.count_nonzero((labels == 1) & (prediction == 0)))
    tp = int(np.count_nonzero((labels == 1) & (prediction == 1)))
    accuracy = (tn + tp) / len(labels)
    tnr = tn / (tn + fp)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * precision * recall / (precision + recall)
    clipped = np.clip(probability, epsilon, 1 - epsilon)
    nll = -np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
    brier = np.mean((probability - labels) ** 2)
    bin_index = np.minimum((probability * ece_bins).astype(np.int64), ece_bins - 1)
    ece = 0.0
    for index in range(ece_bins):
        selected = bin_index == index
        if np.any(selected):
            ece += selected.mean() * abs(probability[selected].mean() - labels[selected].mean())
    return {
        "rows": len(labels),
        "prevalence": float(labels.mean()),
        "threshold": threshold,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "accuracy": float(accuracy),
        "balanced_accuracy": float((tnr + recall) / 2),
        "precision_attack": float(precision),
        "recall_attack": float(recall),
        "f1_attack": float(f1),
        "auroc": float(roc_auc_score(labels, probability)),
        "average_precision": float(average_precision_score(labels, probability)),
        "nll": float(nll),
        "brier": float(brier),
        "ece": float(ece),
        "ece_bins": ece_bins,
    }
