import numpy as np
import pytest

from labelaudit.data import class_counts, normalize_text, stratified_sample


def test_class_counts_equal_split_with_remainder():
    np.testing.assert_array_equal(class_counts(10, 4, None), [3, 3, 2, 2])


def test_class_counts_follow_proportions():
    np.testing.assert_array_equal(class_counts(100, 3, [0.5, 0.3, 0.2]), [50, 30, 20])


def test_class_counts_rejects_wrong_length():
    with pytest.raises(ValueError):
        class_counts(10, 3, [0.5, 0.5])


def test_stratified_sample_takes_exact_counts():
    labels = np.repeat(np.arange(3), 20)
    idx = stratified_sample(labels, np.array([5, 2, 7]), np.random.default_rng(0))
    assert len(set(idx.tolist())) == idx.size
    np.testing.assert_array_equal(np.bincount(labels[idx], minlength=3), [5, 2, 7])


def test_normalize_text_undoes_ag_news_escapes():
    raw = "Wall St.\\band Bush #39;s plan &lt;b&gt;Opinion&lt;/b&gt; said quot;hi quot; costs #36;5"
    assert normalize_text(raw) == 'Wall St. band Bush\'s plan Opinion said"hi" costs $5'
