"""Command-line interface: prepare -> audit -> evaluate / retrain / export-site."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from labelaudit.audit import AuditResult, flag_count, run_audit
from labelaudit.cleaning import flag_rates, select_flags
from labelaudit.config import RunPaths, load_config, read_run_config, save_config
from labelaudit.data import Split, load_splits, read_json, read_split, write_json, write_split
from labelaudit.noise import inject_noise
from labelaudit.ranking import ranking

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Find training examples whose labels are probably wrong.",
)

RunDir = Annotated[Path, typer.Option("--run-dir", "-r", help="Directory holding this run.")]
DEFAULT_RUN = Path("runs/agnews")


def _echo_table(header: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    typer.echo("  ".join(h.ljust(w) for h, w in zip(header, widths, strict=True)))
    typer.echo("  ".join("-" * w for w in widths))
    for row in rows:
        typer.echo("  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)))


def _require(path: Path, hint: str) -> None:
    if not path.exists():
        raise typer.BadParameter(f"{path} not found; {hint}")


def _read_flagged(paths: RunPaths) -> np.ndarray:
    with paths.flags.open(encoding="utf-8", newline="") as fh:
        return np.array([row["flagged"] == "1" for row in csv.DictReader(fh)])


def _embed(texts: list[str], config_encoder: str, cache_dir: str) -> np.ndarray:
    from labelaudit.features import embed

    return embed(texts, config_encoder, Path(cache_dir))


@app.command()
def prepare(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", exists=True, dir_okay=False, help="TOML config file."),
    ] = None,
    run_dir: RunDir = DEFAULT_RUN,
) -> None:
    """Download the dataset, take a seeded subset, and inject label noise."""
    cfg = load_config(config)
    paths = RunPaths(run_dir)
    train, test, classes = load_splits(cfg.data, cfg.seed)
    noise = inject_noise(
        train.labels,
        len(classes),
        cfg.noise.rate,
        mode=cfg.noise.mode,
        flip_targets=cfg.noise.flip_targets,
        seed=cfg.seed + 1,
    )
    write_split(paths.train, Split(train.texts, noise.noisy_labels))
    write_split(paths.test, test)
    write_json(paths.classes, classes)
    write_json(
        paths.truth,
        {
            "injected": noise.flipped.tolist(),
            "original_labels": train.labels[noise.flipped].tolist(),
        },
    )
    save_config(cfg, paths.config)
    typer.echo(f"Wrote {len(train.texts)} training and {len(test.texts)} held-out examples.")
    typer.echo(f"Injected {noise.flipped.size} {cfg.noise.mode} label flips into training only.")
    typer.echo(f"Ground truth is in {paths.truth.parent}; only evaluate and export-site read it.")


@app.command()
def audit(run_dir: RunDir = DEFAULT_RUN) -> None:
    """Score every training example, rank them, and write a cleaned dataset."""
    paths = RunPaths(run_dir)
    _require(paths.config, "run `labelaudit prepare` first")
    cfg = read_run_config(paths.config)
    train = read_split(paths.train)
    classes: list[str] = read_json(paths.classes)
    k = len(classes)

    typer.echo(f"Embedding {len(train.texts)} texts with {cfg.model.encoder}")
    features = _embed(train.texts, cfg.model.encoder, cfg.cache_dir)
    result = run_audit(features, train.labels, k, cfg, log=typer.echo)
    result.save(paths.scores)

    order = ranking(result.combined)
    flagged = select_flags(order, train.labels, k, flag_count(result, cfg), cfg.audit.class_cap)
    is_flagged = np.zeros(train.labels.size, dtype=bool)
    is_flagged[flagged] = True
    rank = np.empty(order.size, dtype=np.int64)
    rank[order] = np.arange(1, order.size + 1)

    with paths.flags.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["id", "rank", "combined", "signal_a", "signal_b", "given", "suggested", "flagged"]
        )
        for i in range(train.labels.size):
            writer.writerow(
                [
                    i,
                    rank[i],
                    f"{result.combined[i]:.6f}",
                    f"{result.signal_a[i]:.6f}",
                    f"{result.signal_b[i]:.6f}",
                    classes[train.labels[i]],
                    classes[result.suggested[i]],
                    int(is_flagged[i]),
                ]
            )
    with paths.cleaned.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["text", "label"])
        for i in np.flatnonzero(~is_flagged):
            writer.writerow([train.texts[i], classes[train.labels[i]]])

    rates = flag_rates(train.labels, flagged, k)
    with paths.flag_rates.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class", "total", "flagged", "rate"])
        for r in rates:
            writer.writerow([classes[r.label], r.total, r.flagged, f"{r.rate:.4f}"])

    typer.echo("\nPer-class confidence thresholds")
    _echo_table(
        ["class", "threshold"],
        [[c, f"{t:.3f}"] for c, t in zip(classes, result.thresholds, strict=True)],
    )
    typer.echo("\nConfident joint (rows: given label, columns: confident label)")
    _echo_table(
        ["given \\ confident", *classes],
        [
            [c, *(str(v) for v in row)]
            for c, row in zip(classes, result.confident_joint, strict=True)
        ],
    )
    typer.echo(f"\nConfident learning estimates {result.estimated_issues} label issues.")
    typer.echo(f"Flagged {flagged.size} examples for removal\n")
    _echo_table(
        ["class", "total", "flagged", "rate"],
        [[classes[r.label], str(r.total), str(r.flagged), f"{r.rate:.1%}"] for r in rates],
    )
    typer.echo(f"\nWrote {paths.scores}, {paths.flags}, {paths.flag_rates}, {paths.cleaned}")


@app.command()
def evaluate(run_dir: RunDir = DEFAULT_RUN) -> None:
    """Score rankings against the injected ground truth (terminal output only)."""
    from labelaudit.evaluation import detection_metrics

    paths = RunPaths(run_dir)
    _require(paths.scores, "run `labelaudit audit` first")
    result = AuditResult.load(paths.scores)
    is_noisy = np.zeros(result.combined.size, dtype=bool)
    is_noisy[read_json(paths.truth)["injected"]] = True

    rows = []
    for name, scores in [
        ("combined (A + B)", result.combined),
        ("loss-only baseline", result.loss_only),
        ("signal A alone", result.signal_a),
        ("signal B alone", result.signal_b),
    ]:
        m = detection_metrics(scores, is_noisy)
        rows.append(
            [name, f"{m.auroc:.4f}", f"{m.average_precision:.4f}", f"{m.precision_at_k:.4f}"]
        )
    typer.echo(f"Injected label errors: {int(is_noisy.sum())} of {is_noisy.size}\n")
    _echo_table(["ranking", "AUROC", "avg precision", "precision@k"], rows)
    typer.echo("k = number of injected errors")

    flagged = _read_flagged(paths)
    hits = int((flagged & is_noisy).sum())
    typer.echo(
        f"\nCleaning removed {int(flagged.sum())}: precision {hits / max(1, flagged.sum()):.4f}, "
        f"recall {hits / is_noisy.sum():.4f}"
    )


@app.command()
def retrain(
    run_dir: RunDir = DEFAULT_RUN,
    seeds: Annotated[int, typer.Option(min=1, help="Training seeds per condition.")] = 3,
) -> None:
    """Compare corrupted vs cleaned training on the clean held-out split (terminal only)."""
    from labelaudit.model import accuracy, train_head

    paths = RunPaths(run_dir)
    _require(paths.flags, "run `labelaudit audit` first")
    cfg = read_run_config(paths.config)
    train = read_split(paths.train)
    test = read_split(paths.test)
    k = len(read_json(paths.classes))
    keep = ~_read_flagged(paths)

    x_train = _embed(train.texts, cfg.model.encoder, cfg.cache_dir)
    x_test = _embed(test.texts, cfg.model.encoder, cfg.cache_dir)
    conditions = {
        "corrupted": (x_train, train.labels),
        "cleaned": (x_train[keep], train.labels[keep]),
    }
    rows = []
    for name, (x, y) in conditions.items():
        scores = [
            accuracy(
                train_head(x, y, k, cfg.model, cfg.audit.cv_epochs, cfg.seed + 30_000 + s),
                x_test,
                test.labels,
            )
            for s in range(seeds)
        ]
        spread = statistics.stdev(scores) if seeds > 1 else 0.0
        rows.append([name, str(y.size), f"{statistics.mean(scores):.4f}", f"{spread:.4f}"])
    typer.echo(f"Held-out accuracy over {seeds} seeds ({test.labels.size} clean test examples)\n")
    _echo_table(["training set", "examples", "mean accuracy", "std"], rows)


@app.command("export-site")
def export_site(
    run_dir: RunDir = DEFAULT_RUN,
    out: Annotated[Path, typer.Option(help="Where to write the demo JSON.")] = Path(
        "web/public/demo.json"
    ),
) -> None:
    """Write a compact JSON sample of the audit for the web demo."""
    from labelaudit.export import site_payload

    paths = RunPaths(run_dir)
    _require(paths.scores, "run `labelaudit audit` first")
    cfg = read_run_config(paths.config)
    train = read_split(paths.train)
    result = AuditResult.load(paths.scores)
    injected = np.zeros(train.labels.size, dtype=bool)
    injected[read_json(paths.truth)["injected"]] = True
    payload = site_payload(train, read_json(paths.classes), result, injected, cfg)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    typer.echo(
        f"Wrote {len(payload['rows']['text'])} examples to {out} ({out.stat().st_size // 1024} KB)"
    )
