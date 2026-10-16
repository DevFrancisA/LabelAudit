"""Compare rankings against injected ground truth. Results are printed, never saved."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from labelaudit.ranking import ranking


@dataclass(frozen=True)
class DetectionMetrics:
    auroc: float
    average_precision: float
    precision_at_k: float  # k = number of injected errors


def detection_metrics(suspicion: np.ndarray, is_noisy: np.ndarray) -> DetectionMetrics:
    is_noisy = np.asarray(is_noisy, dtype=bool)
    k = int(is_noisy.sum())
    if k == 0 or k == is_noisy.size:
        raise ValueError("need both corrupted and clean examples to evaluate")
    top = ranking(suspicion)[:k]
    return DetectionMetrics(
        auroc=float(roc_auc_score(is_noisy, suspicion)),
        average_precision=float(average_precision_score(is_noisy, suspicion)),
        precision_at_k=float(is_noisy[top].mean()),
    )
