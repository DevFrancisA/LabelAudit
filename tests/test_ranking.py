import numpy as np
import pytest

from labelaudit.loss_signals import loss_statistics
from labelaudit.ranking import combine, loss_only, rank_normalize, ranking, signal_a, signal_b


def test_rank_normalize_averages_ties(toy):
    case = toy["rank_normalize"]
    np.testing.assert_allclose(rank_normalize(np.array(case["input"])), case["expected"])


def test_rank_normalize_edge_cases():
    assert rank_normalize(np.array([])).size == 0
    np.testing.assert_array_equal(rank_normalize(np.array([4.2])), [0.0])
    np.testing.assert_allclose(rank_normalize(np.array([7, 7, 7])), [0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        rank_normalize(np.array([1.0, np.nan]))


def test_rank_normalize_is_scale_free():
    x = np.array([0.3, 10.0, -2.0, 4.0])
    np.testing.assert_allclose(rank_normalize(x), rank_normalize(np.exp(x)))


def test_combined_ranking_on_toy(toy):
    r = toy["ranking"]
    a = signal_a(np.array(toy["confident_learning"]["self_confidence"]))
    b = loss_only(np.array(r["mean_loss"]))
    np.testing.assert_allclose(a, np.array(r["signal_a_twelfths"]) / 12)
    np.testing.assert_allclose(b, np.array(r["loss_only_twelfths"]) / 12)
    combined = combine(a, b, r["weight_a"], r["weight_b"])
    np.testing.assert_allclose(combined, np.array(r["combined_twelfths"]) / 12)
    np.testing.assert_array_equal(ranking(combined), r["combined_order"])


def test_signal_b_on_toy_trajectories(toy):
    t = toy["trajectories"]
    stats = loss_statistics(np.array(t["losses"]), np.array(t["correct"]), t["late_fraction"])
    np.testing.assert_allclose(signal_b(stats), t["signal_b"])


def test_combine_rejects_zero_weights():
    with pytest.raises(ValueError):
        combine(np.zeros(2), np.zeros(2), 0, 0)
