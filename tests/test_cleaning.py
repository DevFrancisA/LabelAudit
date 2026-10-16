import numpy as np
import pytest

from labelaudit.cleaning import flag_rates, select_flags


@pytest.mark.parametrize("case_index", [0, 1, 2, 3])
def test_select_flags_on_toy(toy, case_index):
    case = toy["cleaning"][case_index]
    flagged = select_flags(
        np.array(toy["ranking"]["combined_order"]),
        np.array(toy["labels"]),
        toy["num_classes"],
        case["n_flag"],
        case["class_cap"],
    )
    np.testing.assert_array_equal(flagged, case["flagged"])


def test_class_cap_protects_rare_class():
    labels = np.array([0] * 10 + [1] * 4)
    order = np.array([10, 11, 12, 13, *range(10)])  # the rare class ranks first
    flagged = select_flags(order, labels, 2, n_flag=6, class_cap=0.5)
    assert (labels[flagged] == 1).sum() == 2
    assert flagged.size == 6


def test_flag_rates(toy):
    case = toy["cleaning"][0]
    rates = flag_rates(np.array(toy["labels"]), np.array(case["flagged"]), toy["num_classes"])
    expected = toy["flag_rates_for_first_cleaning_case"]
    assert [(r.label, r.total, r.flagged) for r in rates] == [
        (e["label"], e["total"], e["flagged"]) for e in expected
    ]
    assert rates[1].rate == 0.25
