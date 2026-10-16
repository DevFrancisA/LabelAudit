"""Write the compact JSON that the web demo loads."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from labelaudit.audit import AuditResult, flag_count
from labelaudit.config import Config
from labelaudit.data import Split

FORMAT_VERSION = 1


def _rounded(values: np.ndarray, digits: int) -> list[float]:
    return [round(float(v), digits) for v in values]


def site_payload(
    train: Split,
    class_names: list[str],
    result: AuditResult,
    injected: np.ndarray,
    config: Config,
) -> dict[str, Any]:
    """A seeded random sample of the audited training set, stored column by column.

    Signals are exported as computed over the full training set, so the sample shows the
    pipeline's real scores rather than scores recomputed on fewer examples.
    """
    n = train.labels.size
    size = min(config.export.sample_size, n)
    rng = np.random.default_rng(config.seed + 20_000)
    picked = rng.choice(n, size=size, replace=False)
    picked = picked[np.argsort(-result.combined[picked], kind="stable")]
    audit = config.audit

    return {
        "version": FORMAT_VERSION,
        "meta": {
            "dataset": "AG News",
            "source": config.data.dataset,
            "encoder": config.model.encoder,
            "classes": class_names,
            "auditedExamples": n,
            "folds": audit.folds,
            "trajectorySeeds": audit.trajectory_seeds,
            "trajectoryEpochs": audit.trajectory_epochs,
            "clScore": audit.cl_score,
            "weightA": audit.weight_a,
            "weightB": audit.weight_b,
            "classCap": audit.class_cap,
            "suggestedFlagCount": math.floor(flag_count(result, config) * size / n + 0.5),
        },
        "rows": {
            "text": [train.texts[i] for i in picked],
            "given": train.labels[picked].tolist(),
            "suggested": result.suggested[picked].tolist(),
            "selfConfidence": _rounded(result.self_confidence[picked], 4),
            "meanLoss": _rounded(result.mean_loss[picked], 3),
            "lateLoss": _rounded(result.late_loss[picked], 3),
            "forgetting": _rounded(result.forgetting[picked], 3),
            "signalA": _rounded(result.signal_a[picked], 5),
            "signalB": _rounded(result.signal_b[picked], 5),
            "trajectory": [_rounded(result.trajectory[i], 2) for i in picked],
            "injected": injected[picked].astype(int).tolist(),
        },
    }
