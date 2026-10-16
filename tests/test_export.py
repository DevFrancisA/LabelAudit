import numpy as np

from labelaudit.audit import AuditResult
from labelaudit.config import Config, ExportConfig
from labelaudit.data import Split
from labelaudit.export import site_payload


def fake_result(n: int, k: int, rng: np.random.Generator) -> AuditResult:
    probs = rng.dirichlet(np.ones(k), size=n)
    return AuditResult(
        pred_probs=probs,
        thresholds=np.full(k, 0.5),
        confident_joint=np.eye(k, dtype=np.int64) * (n // k),
        calibrated_joint=np.full((k, k), n / k**2),
        suggested=probs.argmax(axis=1),
        self_confidence=probs[:, 0],
        normalized_margin=probs[:, 0] - probs[:, 1],
        mean_loss=rng.random(n),
        late_loss=rng.random(n),
        forgetting=rng.integers(0, 3, n).astype(float),
        trajectory=rng.random((n, 5)),
        signal_a=rng.random(n),
        signal_b=rng.random(n),
        combined=rng.random(n),
        loss_only=rng.random(n),
    )


def test_site_payload_is_a_sorted_columnar_sample():
    rng = np.random.default_rng(0)
    n, k = 200, 3
    train = Split([f"text {i}" for i in range(n)], rng.integers(0, k, n))
    result = fake_result(n, k, rng)
    injected = np.zeros(n, dtype=bool)
    injected[:20] = True
    config = Config(export=ExportConfig(sample_size=50))

    payload = site_payload(train, ["a", "b", "c"], result, injected, config)
    rows = payload["rows"]
    assert {len(column) for column in rows.values()} == {50}
    assert len(set(rows["text"])) == 50
    assert all(len(t) == 5 for t in rows["trajectory"])
    ids = [int(t.split()[1]) for t in rows["text"]]
    combined = result.combined[ids]
    assert (np.diff(combined) <= 0).all()
    assert rows["injected"] == injected[ids].astype(int).tolist()
    assert rows["given"] == train.labels[ids].tolist()
    assert payload == site_payload(train, ["a", "b", "c"], result, injected, config)
