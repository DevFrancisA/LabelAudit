"""Run both signals on a feature matrix and blend them into one ranking.

The audit sees only features and the labels as given. It never reads ground truth.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from labelaudit.confident import confident_learning
from labelaudit.config import Config
from labelaudit.loss_signals import loss_statistics
from labelaudit.model import cross_val_probs, loss_trajectories
from labelaudit.ranking import combine, loss_only, signal_a, signal_b


@dataclass(frozen=True)
class AuditResult:
    pred_probs: np.ndarray
    thresholds: np.ndarray
    confident_joint: np.ndarray
    calibrated_joint: np.ndarray
    suggested: np.ndarray
    self_confidence: np.ndarray
    normalized_margin: np.ndarray
    mean_loss: np.ndarray
    late_loss: np.ndarray
    forgetting: np.ndarray
    trajectory: np.ndarray
    signal_a: np.ndarray
    signal_b: np.ndarray
    combined: np.ndarray
    loss_only: np.ndarray

    @property
    def estimated_issues(self) -> int:
        off_diagonal = self.calibrated_joint.sum() - np.trace(self.calibrated_joint)
        return math.floor(float(off_diagonal) + 0.5)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **asdict(self))

    @classmethod
    def load(cls, path: Path) -> AuditResult:
        with np.load(path) as stored:
            return cls(**{key: stored[key] for key in stored.files})


def run_audit(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    config: Config,
    log: Callable[[str], None] = lambda _: None,
) -> AuditResult:
    audit = config.audit
    log(f"Signal A: {audit.folds}-fold cross-validated probabilities")
    probs = cross_val_probs(
        features, labels, num_classes, config.model, audit.folds, audit.cv_epochs, config.seed
    )
    cl = confident_learning(probs, labels)

    log(f"Signal B: loss trajectories over {audit.trajectory_seeds} seeds")
    seeds = [config.seed + 10_000 + s for s in range(audit.trajectory_seeds)]
    losses, correct = loss_trajectories(
        features, labels, num_classes, config.model, audit.trajectory_epochs, seeds
    )
    stats = loss_statistics(losses, correct, audit.late_fraction)

    confidence = cl.self_confidence if audit.cl_score == "self_confidence" else cl.normalized_margin
    a = signal_a(confidence)
    b = signal_b(stats)
    return AuditResult(
        pred_probs=probs,
        thresholds=cl.thresholds,
        confident_joint=cl.joint,
        calibrated_joint=cl.calibrated_joint,
        suggested=cl.suggested,
        self_confidence=cl.self_confidence,
        normalized_margin=cl.normalized_margin,
        mean_loss=stats.mean_loss,
        late_loss=stats.late_loss,
        forgetting=stats.forgetting,
        trajectory=stats.trajectory,
        signal_a=a,
        signal_b=b,
        combined=combine(a, b, audit.weight_a, audit.weight_b),
        loss_only=loss_only(stats.mean_loss),
    )


def flag_count(result: AuditResult, config: Config) -> int:
    """How many examples cleaning should remove: the configured fraction, or the
    confident-learning estimate of how many labels are wrong."""
    fraction = config.audit.flag_fraction
    n = result.combined.size
    return result.estimated_issues if fraction is None else math.floor(fraction * n + 0.5)
