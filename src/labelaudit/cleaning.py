"""Choose which examples to remove, and summarize how flags fall across classes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassFlagRate:
    label: int
    total: int
    flagged: int

    @property
    def rate(self) -> float:
        return self.flagged / self.total if self.total else 0.0


def class_limits(labels: np.ndarray, num_classes: int, class_cap: float | None) -> np.ndarray:
    counts = np.bincount(labels, minlength=num_classes)
    if class_cap is None:
        return counts
    return np.array([math.floor(class_cap * c) for c in counts])


def select_flags(
    order: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    n_flag: int,
    class_cap: float | None = None,
) -> np.ndarray:
    """Walk the ranking and flag up to `n_flag` examples.

    With a cap, at most floor(class_cap * n_c) examples of class c are flagged; an example
    whose class is full is skipped and the walk moves on, so rare classes are not
    stripped. Returns flagged indices in ranking order (possibly fewer than `n_flag`).
    """
    labels = np.asarray(labels, dtype=np.int64)
    limits = class_limits(labels, num_classes, class_cap)
    used = np.zeros(num_classes, dtype=np.int64)
    flagged: list[int] = []
    for i in order:
        if len(flagged) >= n_flag:
            break
        c = labels[i]
        if used[c] < limits[c]:
            used[c] += 1
            flagged.append(int(i))
    return np.array(flagged, dtype=np.int64)


def flag_rates(labels: np.ndarray, flagged: np.ndarray, num_classes: int) -> list[ClassFlagRate]:
    labels = np.asarray(labels, dtype=np.int64)
    totals = np.bincount(labels, minlength=num_classes)
    hits = np.bincount(labels[np.asarray(flagged, dtype=np.int64)], minlength=num_classes)
    return [ClassFlagRate(c, int(totals[c]), int(hits[c])) for c in range(num_classes)]
