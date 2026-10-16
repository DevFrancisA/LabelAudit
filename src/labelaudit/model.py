"""A small MLP head trained on frozen embeddings, plus the two training regimes the
signals need: K-fold cross-validation (out-of-sample probabilities for confident
learning) and full-data training with per-epoch loss logging (loss trajectories)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch import nn

from labelaudit.config import ModelConfig


class MLPHead(nn.Module):
    def __init__(self, dim: int, hidden: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_deterministic() -> None:
    torch.use_deterministic_algorithms(True)


def train_head(
    x: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    cfg: ModelConfig,
    epochs: int,
    seed: int,
    on_epoch: Callable[[MLPHead], None] | None = None,
) -> MLPHead:
    """Train from a fresh seeded initialization. `on_epoch` runs after every epoch."""
    make_deterministic()
    torch.manual_seed(seed)  # initialization and dropout masks
    order = torch.Generator().manual_seed(seed)  # minibatch order
    features = torch.as_tensor(x, dtype=torch.float32)
    targets = torch.as_tensor(y, dtype=torch.long)
    model = MLPHead(features.shape[1], cfg.hidden, num_classes, cfg.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(features.shape[0], generator=order)
        for start in range(0, perm.numel(), cfg.batch_size):
            batch = perm[start : start + cfg.batch_size]
            loss = F.cross_entropy(model(features[batch]), targets[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if on_epoch is not None:
            on_epoch(model)
    return model


@torch.no_grad()
def predict_proba(model: MLPHead, x: np.ndarray) -> np.ndarray:
    model.eval()
    logits = model(torch.as_tensor(x, dtype=torch.float32))
    return torch.softmax(logits, dim=1).double().numpy()


def cross_val_probs(
    x: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    cfg: ModelConfig,
    folds: int,
    epochs: int,
    seed: int,
) -> np.ndarray:
    """Out-of-sample predicted probabilities: each example is scored by the model trained
    on the other folds."""
    probs = np.zeros((y.size, num_classes))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train_idx, held_idx) in enumerate(splitter.split(x, y)):
        model = train_head(x[train_idx], y[train_idx], num_classes, cfg, epochs, seed + fold)
        probs[held_idx] = predict_proba(model, x[held_idx])
    return probs


def loss_trajectories(
    x: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    cfg: ModelConfig,
    epochs: int,
    seeds: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Train on all examples once per seed and record, after every epoch, each example's
    loss under its given label and whether the top class matches that label.

    Returns (losses, correct), both shaped (seeds, epochs, n).
    """
    losses = np.zeros((len(seeds), epochs, y.size))
    correct = np.zeros((len(seeds), epochs, y.size), dtype=bool)
    features = torch.as_tensor(x, dtype=torch.float32)
    targets = torch.as_tensor(y, dtype=torch.long)

    for s, seed in enumerate(seeds):
        epoch = 0

        @torch.no_grad()
        def record(model: MLPHead, s: int = s) -> None:
            nonlocal epoch
            model.eval()
            logits = model(features)
            losses[s, epoch] = F.cross_entropy(logits, targets, reduction="none").numpy()
            correct[s, epoch] = (logits.argmax(dim=1) == targets).numpy()
            epoch += 1

        train_head(x, y, num_classes, cfg, epochs, seed, on_epoch=record)
    return losses, correct


def accuracy(model: MLPHead, x: np.ndarray, y: np.ndarray) -> float:
    return float((predict_proba(model, x).argmax(axis=1) == y).mean())
