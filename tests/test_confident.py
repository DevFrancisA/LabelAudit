import numpy as np
import pytest

from labelaudit.confident import calibrate_joint, confident_learning
from labelaudit.ranking import ranking, signal_a


@pytest.fixture
def cl(toy):
    return confident_learning(np.array(toy["pred_probs"]), np.array(toy["labels"]))


def test_thresholds_are_mean_self_confidence_per_class(cl, toy):
    np.testing.assert_allclose(cl.thresholds, toy["confident_learning"]["thresholds"])


def test_confident_label_takes_argmax_among_classes_over_threshold(cl, toy):
    expected = toy["confident_learning"]["confident_label"]
    np.testing.assert_array_equal(cl.confident_label, expected)
    # Example 12's overall argmax (class 1) misses its threshold; class 0 clears it.
    assert cl.confident_label[12] == 0
    # Example 10 clears two thresholds and goes to the more probable of the two.
    assert cl.confident_label[10] == 1


def test_confident_joint(cl, toy):
    np.testing.assert_array_equal(cl.joint, toy["confident_learning"]["confident_joint"])


def test_calibrated_joint_rows_match_label_counts(cl, toy):
    expected = toy["confident_learning"]["calibrated_joint"]
    np.testing.assert_allclose(cl.calibrated_joint, expected)
    np.testing.assert_allclose(cl.calibrated_joint.sum(axis=1), np.bincount(toy["labels"]))


def test_issue_estimate_rounds_half_up(cl, toy):
    expected = toy["confident_learning"]
    np.testing.assert_allclose(
        cl.estimated_issues_per_class, expected["estimated_issues_per_class"]
    )
    assert cl.estimated_issues == expected["estimated_issues"]


def test_scores_and_suggestions(cl, toy):
    expected = toy["confident_learning"]
    np.testing.assert_allclose(cl.self_confidence, expected["self_confidence"])
    np.testing.assert_allclose(cl.normalized_margin, expected["normalized_margin"])
    np.testing.assert_array_equal(cl.suggested, expected["suggested_labels"])


def test_self_confidence_ranking(cl, toy):
    order = ranking(signal_a(cl.self_confidence))
    np.testing.assert_array_equal(order, toy["confident_learning"]["self_confidence_order"])


def test_every_class_row_of_the_joint_is_populated():
    # Some example of each class clears that class's threshold (the threshold is the
    # class mean), so no row of the confident joint is ever empty.
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(4), size=200)
    labels = rng.integers(0, 4, size=200)
    assert (confident_learning(probs, labels).joint.sum(axis=1) > 0).all()


def test_calibrating_an_empty_row_keeps_its_mass_on_the_diagonal():
    calibrated = calibrate_joint(np.array([[2, 0], [0, 0]]), np.array([0, 0, 1]))
    np.testing.assert_allclose(calibrated, [[2, 0], [0, 1]])


def test_missing_class_is_rejected():
    with pytest.raises(ValueError, match="missing"):
        confident_learning(np.array([[0.5, 0.5]]), np.array([0]))
