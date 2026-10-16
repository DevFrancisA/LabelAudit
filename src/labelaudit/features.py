"""Sentence embeddings from a frozen encoder, cached on disk by content hash."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def cache_key(texts: Sequence[str], model_name: str) -> str:
    digest = hashlib.sha256(model_name.encode())
    for text in texts:
        digest.update(text.encode())
        digest.update(b"\0")
    return digest.hexdigest()[:32]


def embed(texts: Sequence[str], model_name: str, cache_dir: Path) -> np.ndarray:
    """Unit-normalized float32 embeddings, shape (len(texts), dim)."""
    path = cache_dir / "embeddings" / f"{cache_key(texts, model_name)}.npy"
    if path.exists():
        return np.load(path)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    vectors = model.encode(
        list(texts),
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npy")
    np.save(partial, vectors)
    partial.replace(path)
    return vectors
