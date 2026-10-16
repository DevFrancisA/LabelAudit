"""Signal B: statistics of each example's training-loss trajectory.

Mislabeled examples tend to stay hard to fit: their loss under the given label stays
high, drops only late in training (when the model starts memorizing), and the model
often flips between getting them right and wrong. From per-epoch losses recorded over
several seeds we compute, per example:

* mean loss over all seeds and epochs;
* late-training loss, the mean over the final `late_fraction` of epochs;
* forgetting, in the spirit of Toneva et al. (ICLR 2019): the number of times the model
  went from predicting the given label to not predicting it between consecutive epochs.
  An example never predicted correctly in a seed counts as `epochs` forgetting events
  for that seed, more than any learnable example can reach. Averaged over seeds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LossStats:
    mean_loss: np.ndarray  # (n,)
    late_loss: np.ndarray  # (n,)
    forgetting: np.ndarray  # (n,)
    trajectory: np.ndarray  # (n, epochs): loss per epoch, averaged over seeds


def late_epoch_count(epochs: int, late_fraction: float) -> int:
    return max(1, min(epochs, math.floor(epochs * late_fraction + 0.5)))


def forgetting_events(correct: np.ndarray) -> np.ndarray:
    """Forgetting count per example for one seed; `correct` is (epochs, n) boolean."""
    epochs = correct.shape[0]
    forgotten = (correct[:-1] & ~correct[1:]).sum(axis=0).astype(np.float64)
    never_learned = ~correct.any(axis=0)
    return np.where(never_learned, float(epochs), forgotten)


def loss_statistics(losses: np.ndarray, correct: np.ndarray, late_fraction: float) -> LossStats:
    """`losses` and `correct` are indexed [seed, epoch, example]."""
    losses = np.asarray(losses, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    if losses.ndim != 3 or losses.shape != correct.shape:
        raise ValueError("losses and correct must both be (seeds, epochs, n)")
    epochs = losses.shape[1]
    late = late_epoch_count(epochs, late_fraction)
    return LossStats(
        mean_loss=losses.mean(axis=(0, 1)),
        late_loss=losses[:, epochs - late :].mean(axis=(0, 1)),
        forgetting=np.mean([forgetting_events(seed) for seed in correct], axis=0),
        trajectory=losses.mean(axis=0).T,
    )
