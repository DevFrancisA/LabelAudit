"""Load a seeded, class-controlled subset of a Hugging Face text classification dataset."""

from __future__ import annotations

import html
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from labelaudit.config import DataConfig

_NUMERIC_ENTITY = re.compile(r"( ?)#(\d+);")
_QUOT = re.compile(r" ?quot;")
_TAG = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class Split:
    texts: list[str]
    labels: np.ndarray


def normalize_text(text: str) -> str:
    """Undo AG News escaping artifacts such as `#39;`, ` quot;`, `\\` and `&lt;b&gt;`."""
    text = text.replace("\\", " ")
    text = _NUMERIC_ENTITY.sub(
        lambda m: "'" if m.group(2) == "39" else m.group(1) + chr(int(m.group(2))), text
    )
    text = _QUOT.sub('"', text)
    text = _TAG.sub("", html.unescape(text))
    return " ".join(text.split())


def class_counts(n: int, num_classes: int, proportions: Sequence[float] | None) -> np.ndarray:
    """Split `n` into per-class counts by largest remainder (ties go to lower class ids)."""
    weights = np.ones(num_classes) if proportions is None else np.asarray(proportions, float)
    if weights.shape != (num_classes,):
        raise ValueError(f"expected {num_classes} class proportions, got {len(weights)}")
    exact = n * weights / weights.sum()
    counts = np.floor(exact).astype(np.int64)
    remainder = exact - counts
    order = np.argsort(-remainder, kind="stable")
    counts[order[: n - counts.sum()]] += 1
    return counts


def stratified_sample(
    labels: np.ndarray, counts: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Indices with exactly `counts[c]` examples of class c, in a shuffled order."""
    parts = []
    for c, k in enumerate(counts):
        members = np.flatnonzero(labels == c)
        if k > members.size:
            raise ValueError(f"class {c} has {members.size} examples, {k} requested")
        parts.append(rng.choice(members, size=int(k), replace=False))
    picked = np.concatenate(parts)
    return picked[rng.permutation(picked.size)]


def load_splits(cfg: DataConfig, seed: int) -> tuple[Split, Split, list[str]]:
    """Return (train, held-out test, class names). The test split is never corrupted."""
    from datasets import load_dataset

    dataset = load_dataset(cfg.dataset)
    class_names: list[str] = list(dataset["train"].features[cfg.label_column].names)
    rng = np.random.default_rng(seed)

    def take(split: str, n: int, proportions: Sequence[float] | None) -> Split:
        labels = np.asarray(dataset[split][cfg.label_column], dtype=np.int64)
        counts = class_counts(n, len(class_names), proportions)
        idx = stratified_sample(labels, counts, rng)
        texts = dataset[split][cfg.text_column]
        return Split([normalize_text(texts[i]) for i in idx], labels[idx])

    train = take("train", cfg.n_train, cfg.class_proportions)
    test = take("test", cfg.n_test, None)
    return train, test, class_names


def write_split(path: Path, split: Split) -> None:
    write_jsonl(
        path,
        (
            {"id": i, "text": t, "label": int(y)}
            for i, (t, y) in enumerate(zip(split.texts, split.labels, strict=True))
        ),
    )


def read_split(path: Path) -> Split:
    records = read_jsonl(path)
    return Split([r["text"] for r in records], np.array([r["label"] for r in records], np.int64))


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
