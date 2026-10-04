from __future__ import annotations

import time

import numpy as np

from .logistic import apply_logistic, fit_unpenalized_logistic, probability_logit


def fit_platt(raw_probability: np.ndarray, labels: np.ndarray, config: dict) -> dict:
    started = time.perf_counter()
    epsilon = float(config["calibration"]["input_probability_epsilon"])
    logits = probability_logit(raw_probability, epsilon)
    counts = np.bincount(labels.astype(np.int64), minlength=2)
    assert np.all(counts > 0)
    fit = fit_unpenalized_logistic(
        logits,
        labels,
        config["logistic_solver"],
        int(config["batches"]["logistic_closure_rows"]),
    )
    return {
        "method": "Platt",
        "input": "logit_raw_attack_probability",
        "input_probability_epsilon": epsilon,
        "penalty": "none",
        "fit_rows": len(labels),
        "label_counts": {"Benign": int(counts[0]), "Attack": int(counts[1])},
        "coefficient": fit["coefficients"][0],
        "intercept": fit["intercept"],
        "solver": fit,
        "solver_config": dict(config["logistic_solver"]),
        "closure_rows": int(config["batches"]["logistic_closure_rows"]),
        "fit_seconds": time.perf_counter() - started,
    }


def apply_platt(raw_probability: np.ndarray, calibrator: dict) -> np.ndarray:
    logits = probability_logit(raw_probability, calibrator["input_probability_epsilon"])
    return apply_logistic(logits, calibrator["coefficient"], calibrator["intercept"])


def historical_competence(probability: np.ndarray, labels: np.ndarray, threshold: float) -> list[dict]:
    predicted = (probability >= threshold).astype(np.uint8)
    table = []
    for predicted_class, name in ((0, "Benign"), (1, "Attack")):
        selected = predicted == predicted_class
        count = int(selected.sum())
        correct = int(np.count_nonzero(labels[selected] == predicted_class))
        table.append({
            "predicted_class": predicted_class,
            "predicted_class_name": name,
            "predicted_count": count,
            "correct_count": correct,
            "h": (correct + 1) / (count + 2),
        })
    return table
