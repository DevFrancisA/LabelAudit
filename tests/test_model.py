import numpy as np

from labelaudit.audit import AuditResult, run_audit
from labelaudit.config import AuditConfig, Config, ModelConfig
from labelaudit.model import cross_val_probs, loss_trajectories

SMALL = ModelConfig(hidden=16, batch_size=32)


def blobs(n_per_class=40, num_classes=3, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3, size=(num_classes, dim))
    y = np.repeat(np.arange(num_classes), n_per_class)
    x = centers[y] + rng.normal(size=(y.size, dim))
    return x.astype(np.float32), y


def test_cross_val_probs_are_reproducible_distributions():
    x, y = blobs()
    a = cross_val_probs(x, y, 3, SMALL, folds=3, epochs=3, seed=1)
    b = cross_val_probs(x, y, 3, SMALL, folds=3, epochs=3, seed=1)
    assert a.shape == (y.size, 3)
    np.testing.assert_allclose(a.sum(axis=1), 1, rtol=1e-6)
    np.testing.assert_array_equal(a, b)


def test_loss_trajectories_shape_and_reproducibility():
    x, y = blobs()
    losses, correct = loss_trajectories(x, y, 3, SMALL, epochs=4, seeds=[0, 1])
    again, _ = loss_trajectories(x, y, 3, SMALL, epochs=4, seeds=[0, 1])
    assert losses.shape == correct.shape == (2, 4, y.size)
    assert (losses >= 0).all()
    np.testing.assert_array_equal(losses, again)
    assert not np.array_equal(losses[0], losses[1])


def test_run_audit_is_deterministic_and_round_trips(tmp_path):
    x, y = blobs()
    config = Config(
        model=SMALL,
        audit=AuditConfig(folds=3, cv_epochs=2, trajectory_seeds=2, trajectory_epochs=3),
    )
    first = run_audit(x, y, 3, config)
    second = run_audit(x, y, 3, config)
    np.testing.assert_array_equal(first.combined, second.combined)
    assert first.trajectory.shape == (y.size, 3)
    assert ((first.combined >= 0) & (first.combined <= 1)).all()

    path = tmp_path / "scores.npz"
    first.save(path)
    loaded = AuditResult.load(path)
    np.testing.assert_array_equal(loaded.combined, first.combined)
    assert loaded.estimated_issues == first.estimated_issues
