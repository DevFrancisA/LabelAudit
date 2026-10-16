"""Turn the two signals into one ranking.

Each signal is first converted to a suspicion score in [0, 1] by rank normalization,
so signals on different scales (probabilities, losses, counts) can be blended. Higher
suspicion means the label is more likely wrong.
"""

from __future__ import annotations

import numpy as np

from labelaudit.loss_signals import LossStats


def rank_normalize(values: np.ndarray) -> np.ndarray:
    """Ranks scaled to [0, 1]; ties share their average rank. Smallest value maps to 0."""
    x = np.asarray(values, dtype=np.float64)
    if np.isnan(x).any():
        raise ValueError("cannot rank NaN values")
    n = x.size
    if n <= 1:
        return np.zeros(n)
    order = np.argsort(x, kind="stable")
    _, group, counts = np.unique(x[order], return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    average_rank = (ends - counts + ends - 1) / 2
    ranks = np.empty(n)
    ranks[order] = average_rank[group]
    return ranks / (n - 1)


def signal_a(confidence_score: np.ndarray) -> np.ndarray:
    """Suspicion from a confident-learning score where low values are suspicious."""
    return rank_normalize(-np.asarray(confidence_score, dtype=np.float64))


def signal_b(stats: LossStats) -> np.ndarray:
    """Suspicion from loss statistics: high loss, late loss and forgetting are suspicious."""
    parts = [rank_normalize(s) for s in (stats.mean_loss, stats.late_loss, stats.forgetting)]
    return rank_normalize(np.mean(parts, axis=0))


def loss_only(mean_loss: np.ndarray) -> np.ndarray:
    """Baseline suspicion: rank by mean training loss alone."""
    return rank_normalize(mean_loss)


def combine(a: np.ndarray, b: np.ndarray, weight_a: float, weight_b: float) -> np.ndarray:
    if weight_a < 0 or weight_b < 0 or weight_a + weight_b == 0:
        raise ValueError("weights must be non-negative and not both zero")
    return (weight_a * np.asarray(a) + weight_b * np.asarray(b)) / (weight_a + weight_b)


def ranking(suspicion: np.ndarray) -> np.ndarray:
    """Indices from most to least suspicious; ties keep the lower index first."""
    return np.argsort(-np.asarray(suspicion, dtype=np.float64), kind="stable")
