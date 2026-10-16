import numpy as np
import pytest

from labelaudit.noise import flip_count, flip_target_matrix, inject_noise


def balanced_labels(per_class: int = 250, num_classes: int = 4) -> np.ndarray:
    return np.repeat(np.arange(num_classes), per_class)


@pytest.mark.parametrize("mode", ["uniform", "class_dependent"])
def test_same_seed_is_reproducible(mode):
    labels = balanced_labels()
    a = inject_noise(labels, 4, 0.2, mode=mode, seed=7)
    b = inject_noise(labels, 4, 0.2, mode=mode, seed=7)
    np.testing.assert_array_equal(a.noisy_labels, b.noisy_labels)
    np.testing.assert_array_equal(a.flipped, b.flipped)


@pytest.mark.parametrize("mode", ["uniform", "class_dependent"])
def test_different_seed_changes_which_examples_flip(mode):
    labels = balanced_labels()
    a = inject_noise(labels, 4, 0.2, mode=mode, seed=1)
    b = inject_noise(labels, 4, 0.2, mode=mode, seed=2)
    assert not np.array_equal(a.flipped, b.flipped)


@pytest.mark.parametrize("rate", [0.0, 0.05, 0.2, 0.37])
def test_uniform_flips_exact_fraction(rate):
    labels = balanced_labels(per_class=251)
    result = inject_noise(labels, 4, rate, mode="uniform", seed=0)
    assert result.flipped.size == flip_count(rate, labels.size)
    changed = np.flatnonzero(result.noisy_labels != labels)
    np.testing.assert_array_equal(changed, result.flipped)


@pytest.mark.parametrize("rate", [0.1, 0.25])
def test_class_dependent_flips_exact_fraction_per_class(rate):
    labels = np.concatenate([np.full(n, c) for c, n in enumerate([400, 200, 100, 50])])
    result = inject_noise(labels, 4, rate, mode="class_dependent", seed=3)
    for c in range(4):
        members = labels == c
        flipped_here = np.isin(np.flatnonzero(members), result.flipped).sum()
        assert flipped_here == flip_count(rate, members.sum())
    assert (result.noisy_labels[result.flipped] != labels[result.flipped]).all()


def test_class_dependent_follows_flip_targets():
    targets = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]  # 0->1, 1->2, 2->0
    labels = balanced_labels(per_class=100, num_classes=3)
    result = inject_noise(labels, 3, 0.3, mode="class_dependent", flip_targets=targets, seed=0)
    original = labels[result.flipped]
    np.testing.assert_array_equal(result.noisy_labels[result.flipped], (original + 1) % 3)


def test_unflipped_labels_are_untouched():
    labels = balanced_labels()
    result = inject_noise(labels, 4, 0.3, seed=5)
    keep = np.setdiff1d(np.arange(labels.size), result.flipped)
    np.testing.assert_array_equal(result.noisy_labels[keep], labels[keep])


def test_input_labels_are_not_mutated():
    labels = balanced_labels()
    before = labels.copy()
    inject_noise(labels, 4, 0.3, seed=5)
    np.testing.assert_array_equal(labels, before)


def test_flip_target_matrix_rejects_self_flips():
    with pytest.raises(ValueError, match="zero diagonal"):
        flip_target_matrix(2, [[1, 1], [1, 0]])


def test_default_flip_targets_are_uniform_over_other_classes():
    matrix = flip_target_matrix(3, None)
    np.testing.assert_allclose(matrix, [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
