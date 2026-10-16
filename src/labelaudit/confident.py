"""Signal A: confident learning (Northcutt, Jiang & Chuang, JAIR 2021).

Input is out-of-sample predicted probabilities, so a model never scores an example it
was trained on. From those we compute:

* per-class thresholds: t_j is the average predicted probability of class j over the
  examples labeled j, i.e. how confident the model typically is when class j is right;
* the confident joint C: an example with given label i lands in C[i, j] when class j is
  the most probable class among those whose probability clears its own threshold;
* the calibrated joint, which rescales each row of C to the number of examples with that
  given label; its off-diagonal mass estimates how many labels are wrong;
* per-example scores: self-confidence p(given label) and normalized margin
  p(given label) - max p(other label). Lower means more suspicious.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConfidentLearning:
    thresholds: np.ndarray  # (K,)
    confident_label: np.ndarray  # (n,), -1 when no class clears its threshold
    joint: np.ndarray  # (K, K) counts; rows are given labels, columns confident labels
    calibrated_joint: np.ndarray  # (K, K)
    suggested: np.ndarray  # (n,) confident label, or the argmax when there is none
    self_confidence: np.ndarray  # (n,)
    normalized_margin: np.ndarray  # (n,)

    @property
    def estimated_issues_per_class(self) -> np.ndarray:
        return self.calibrated_joint.sum(axis=1) - np.diag(self.calibrated_joint)

    @property
    def estimated_issues(self) -> int:
        return math.floor(float(self.estimated_issues_per_class.sum()) + 0.5)


def _check(pred_probs: np.ndarray, labels: np.ndarray) -> None:
    if pred_probs.ndim != 2 or labels.shape != (pred_probs.shape[0],):
        raise ValueError("pred_probs must be (n, K) and labels (n,)")
    missing = np.setdiff1d(np.arange(pred_probs.shape[1]), labels)
    if missing.size:
        raise ValueError(f"every class needs at least one labeled example; missing {missing}")


def per_class_thresholds(pred_probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    num_classes = pred_probs.shape[1]
    return np.array([pred_probs[labels == j, j].mean() for j in range(num_classes)])


def confident_labels(pred_probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """For each example, the most probable class among those clearing their threshold."""
    above = pred_probs >= thresholds[None, :]
    masked = np.where(above, pred_probs, -np.inf)
    best = masked.argmax(axis=1)
    return np.where(above.any(axis=1), best, -1)


def confident_joint(
    labels: np.ndarray, confident_label: np.ndarray, num_classes: int
) -> np.ndarray:
    joint = np.zeros((num_classes, num_classes), dtype=np.int64)
    counted = confident_label >= 0
    np.add.at(joint, (labels[counted], confident_label[counted]), 1)
    return joint


def calibrate_joint(joint: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Rescale row i of the confident joint so it sums to the count of label i.

    A row with no confidently counted examples carries no evidence of noise, so its whole
    mass goes on the diagonal.
    """
    num_classes = joint.shape[0]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    row_sums = joint.sum(axis=1).astype(np.float64)
    calibrated = np.zeros_like(joint, dtype=np.float64)
    for i in range(num_classes):
        if row_sums[i] > 0:
            calibrated[i] = joint[i] / row_sums[i] * counts[i]
        else:
            calibrated[i, i] = counts[i]
    return calibrated


def self_confidence(pred_probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return pred_probs[np.arange(labels.size), labels]


def normalized_margin(pred_probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    given = self_confidence(pred_probs, labels)
    others = pred_probs.copy()
    others[np.arange(labels.size), labels] = -np.inf
    return given - others.max(axis=1)


def confident_learning(pred_probs: np.ndarray, labels: np.ndarray) -> ConfidentLearning:
    pred_probs = np.asarray(pred_probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    _check(pred_probs, labels)
    thresholds = per_class_thresholds(pred_probs, labels)
    confident = confident_labels(pred_probs, thresholds)
    joint = confident_joint(labels, confident, pred_probs.shape[1])
    return ConfidentLearning(
        thresholds=thresholds,
        confident_label=confident,
        joint=joint,
        calibrated_joint=calibrate_joint(joint, labels),
        suggested=np.where(confident >= 0, confident, pred_probs.argmax(axis=1)),
        self_confidence=self_confidence(pred_probs, labels),
        normalized_margin=normalized_margin(pred_probs, labels),
    )
