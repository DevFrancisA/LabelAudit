import numpy as np

from labelaudit.loss_signals import forgetting_events, late_epoch_count, loss_statistics


def test_loss_statistics_on_toy(toy):
    t = toy["trajectories"]
    stats = loss_statistics(np.array(t["losses"]), np.array(t["correct"]), t["late_fraction"])
    np.testing.assert_allclose(stats.mean_loss, t["mean_loss"])
    np.testing.assert_allclose(stats.late_loss, t["late_loss"])
    np.testing.assert_allclose(stats.forgetting, t["forgetting"])
    np.testing.assert_allclose(stats.trajectory, t["trajectory"])


def test_never_learned_counts_as_more_than_any_learnable_example():
    correct = np.array([[1, 0], [0, 0], [1, 0], [0, 0], [1, 0]], dtype=bool)
    np.testing.assert_array_equal(forgetting_events(correct), [2, 5])


def test_late_epoch_count_rounds_half_up_and_stays_in_range():
    assert late_epoch_count(15, 1 / 3) == 5
    assert late_epoch_count(4, 0.5) == 2
    assert late_epoch_count(3, 0.01) == 1
    assert late_epoch_count(3, 1.0) == 3
