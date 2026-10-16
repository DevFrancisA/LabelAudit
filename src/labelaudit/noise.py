"""Seeded label-noise injection.

Two modes:

* ``uniform``: pick ``round(rate * n)`` examples anywhere in the dataset and give each a
  different label chosen uniformly from the other classes.
* ``class_dependent``: within each class ``c``, pick ``round(rate * n_c)`` examples and
  draw their new label from row ``c`` of a flip-target matrix. This mimics annotators who
  confuse specific pairs of classes.

The returned indices of corrupted examples are ground truth. They must be stored apart
from anything the detector reads.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from labelaudit.config import NoiseMode


@dataclass(frozen=True)
class NoiseResult:
    noisy_labels: np.ndarray
    flipped: np.ndarray  # sorted indices of examples whose label was changed


def flip_count(rate: float, n: int) -> int:
    """Number of examples to flip; rounds half up so the count is predictable."""
    return math.floor(rate * n + 0.5)


def flip_target_matrix(
    num_classes: int, flip_targets: Sequence[Sequence[float]] | None
) -> np.ndarray:
    """Row-normalized matrix of where flipped labels go, with a zero diagonal."""
    if flip_targets is None:
        matrix = np.ones((num_classes, num_classes)) - np.eye(num_classes)
    else:
        matrix = np.asarray(flip_targets, dtype=np.float64)
        if matrix.shape != (num_classes, num_classes):
            raise ValueError(f"flip_targets must be {num_classes}x{num_classes}")
        if (matrix < 0).any():
            raise ValueError("flip_targets must be non-negative")
        if np.diag(matrix).any():
            raise ValueError("flip_targets must have a zero diagonal (a flip changes the label)")
    sums = matrix.sum(axis=1, keepdims=True)
    if (sums == 0).any():
        raise ValueError("every row of flip_targets needs at least one target class")
    return matrix / sums


def inject_noise(
    labels: Sequence[int] | np.ndarray,
    num_classes: int,
    rate: float,
    mode: NoiseMode = "uniform",
    flip_targets: Sequence[Sequence[float]] | None = None,
    seed: int = 0,
) -> NoiseResult:
    if num_classes < 2:
        raise ValueError("need at least two classes to flip labels")
    if not 0 <= rate < 1:
        raise ValueError("rate must be in [0, 1)")
    clean = np.asarray(labels, dtype=np.int64)
    if clean.size and (clean.min() < 0 or clean.max() >= num_classes):
        raise ValueError("labels must lie in [0, num_classes)")
    rng = np.random.default_rng(seed)

    if mode == "uniform":
        idx = rng.choice(clean.size, size=flip_count(rate, clean.size), replace=False)
        offsets = rng.integers(1, num_classes, size=idx.size)
        new_labels = (clean[idx] + offsets) % num_classes
    elif mode == "class_dependent":
        targets = flip_target_matrix(num_classes, flip_targets)
        idx_parts: list[np.ndarray] = []
        new_parts: list[np.ndarray] = []
        for c in range(num_classes):
            members = np.flatnonzero(clean == c)
            chosen = rng.choice(members, size=flip_count(rate, members.size), replace=False)
            idx_parts.append(chosen)
            new_parts.append(rng.choice(num_classes, size=chosen.size, p=targets[c]))
        idx = np.concatenate(idx_parts)
        new_labels = np.concatenate(new_parts)
    else:
        raise ValueError(f"unknown noise mode {mode!r}")

    noisy = clean.copy()
    noisy[idx] = new_labels
    return NoiseResult(noisy_labels=noisy, flipped=np.sort(idx))
