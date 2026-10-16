"""Configuration for a LabelAudit run.

`labelaudit prepare` resolves a config (defaults, optionally overridden by a TOML
file), and writes it into the run directory. Every later stage reads it back from
there, so all stages of a run share exactly the same settings.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

NoiseMode = Literal["uniform", "class_dependent"]
ClScore = Literal["self_confidence", "normalized_margin"]


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "fancyzhx/ag_news"
    text_column: str = "text"
    label_column: str = "label"
    n_train: int = 20_000
    n_test: int = 4_000
    # Share of each class in the training subset. None means equal counts per class.
    class_proportions: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.n_train <= 0 or self.n_test <= 0:
            raise ValueError("n_train and n_test must be positive")
        props = self.class_proportions
        if props is not None and (any(p < 0 for p in props) or sum(props) <= 0):
            raise ValueError("class_proportions must be non-negative with a positive sum")


@dataclass(frozen=True)
class NoiseConfig:
    rate: float = 0.1
    mode: NoiseMode = "uniform"
    # For class_dependent noise: row i gives the relative chance that a flipped example of
    # class i receives each other label. None means all other labels are equally likely.
    flip_targets: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.rate < 1:
            raise ValueError("noise rate must be in [0, 1)")
        if self.mode not in ("uniform", "class_dependent"):
            raise ValueError(f"unknown noise mode {self.mode!r}")


@dataclass(frozen=True)
class ModelConfig:
    encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    hidden: int = 256
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128


@dataclass(frozen=True)
class AuditConfig:
    folds: int = 5
    cv_epochs: int = 10
    trajectory_seeds: int = 3
    trajectory_epochs: int = 15
    # Fraction of the final epochs averaged into the late-training loss.
    late_fraction: float = 1 / 3
    cl_score: ClScore = "self_confidence"
    weight_a: float = 0.5
    weight_b: float = 0.5
    # Fraction of the training set to remove. None uses the confident-learning estimate.
    flag_fraction: float | None = None
    # Largest fraction of any single class that cleaning may remove. None disables the cap.
    class_cap: float | None = 0.5

    def __post_init__(self) -> None:
        if self.folds < 2:
            raise ValueError("folds must be at least 2")
        if self.trajectory_seeds < 1 or self.trajectory_epochs < 2 or self.cv_epochs < 1:
            raise ValueError("need at least one seed, two trajectory epochs, and one CV epoch")
        if not 0 < self.late_fraction <= 1:
            raise ValueError("late_fraction must be in (0, 1]")
        if self.cl_score not in ("self_confidence", "normalized_margin"):
            raise ValueError(f"unknown cl_score {self.cl_score!r}")
        if self.weight_a < 0 or self.weight_b < 0 or self.weight_a + self.weight_b == 0:
            raise ValueError("weights must be non-negative and not both zero")
        if self.flag_fraction is not None and not 0 <= self.flag_fraction <= 1:
            raise ValueError("flag_fraction must be in [0, 1]")
        if self.class_cap is not None and not 0 < self.class_cap <= 1:
            raise ValueError("class_cap must be in (0, 1]")


@dataclass(frozen=True)
class ExportConfig:
    sample_size: int = 2_500


@dataclass(frozen=True)
class Config:
    seed: int = 0
    cache_dir: str = ".cache/labelaudit"
    data: DataConfig = field(default_factory=DataConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        sections = {
            "data": DataConfig,
            "noise": NoiseConfig,
            "model": ModelConfig,
            "audit": AuditConfig,
            "export": ExportConfig,
        }
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        kwargs: dict[str, Any] = {}
        for key, value in raw.items():
            if key in sections:
                kwargs[key] = _build_section(sections[key], value, key)
            else:
                kwargs[key] = value
        return cls(**kwargs)


def _build_section(section_cls: type, raw: dict[str, Any], name: str) -> Any:
    allowed = {f.name for f in fields(section_cls)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown keys in [{name}]: {sorted(unknown)}")
    return section_cls(**{k: _freeze(v) for k, v in raw.items()})


def _freeze(value: Any) -> Any:
    """Turn JSON/TOML lists into tuples so frozen configs stay hashable."""
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def load_config(path: Path | None) -> Config:
    if path is None:
        return Config()
    with path.open("rb") as fh:
        return Config.from_dict(tomllib.load(fh))


def save_config(config: Config, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")


def read_run_config(path: Path) -> Config:
    return Config.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class RunPaths:
    """Where each stage reads and writes inside a run directory.

    Ground truth about injected noise lives under `truth/`, which only `evaluate` and
    `export-site` read. The audit stage reads `data/` only.
    """

    root: Path

    @property
    def config(self) -> Path:
        return self.root / "config.json"

    @property
    def train(self) -> Path:
        return self.root / "data" / "train.jsonl"

    @property
    def test(self) -> Path:
        return self.root / "data" / "test.jsonl"

    @property
    def classes(self) -> Path:
        return self.root / "data" / "classes.json"

    @property
    def truth(self) -> Path:
        return self.root / "truth" / "injected.json"

    @property
    def scores(self) -> Path:
        return self.root / "audit" / "scores.npz"

    @property
    def flags(self) -> Path:
        return self.root / "audit" / "flags.csv"

    @property
    def flag_rates(self) -> Path:
        return self.root / "audit" / "flag_rates.csv"

    @property
    def cleaned(self) -> Path:
        return self.root / "audit" / "cleaned.csv"
