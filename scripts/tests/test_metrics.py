"""Unit tests for the honest-evaluation metrics that back the mastery gates."""
import numpy as np

from common import metrics as M


def test_auroc_perfect_and_random():
    y = [0, 0, 1, 1]
    assert M.auroc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    # reversed scores -> AUROC 0
    assert M.auroc(y, [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_threshold_at_sensitivity_meets_floor():
    rng = np.random.default_rng(0)
    y = np.array([0] * 80 + [1] * 20)
    # well-separated scores
    proba = np.concatenate([rng.uniform(0.0, 0.5, 80), rng.uniform(0.5, 1.0, 20)])
    thr, sens, spec = M.threshold_at_sensitivity(y, proba, 0.90)
    assert sens >= 0.90
    # the chosen threshold should actually deliver >=0.90 sensitivity
    c = M.confusion_at(y, proba, thr)
    assert c["sensitivity"] >= 0.90


def test_threshold_monotone_property():
    y = np.array([0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.4, 0.45, 0.7, 0.9])
    thr_90, s90, _ = M.threshold_at_sensitivity(y, proba, 0.90)
    thr_60, s60, _ = M.threshold_at_sensitivity(y, proba, 0.60)
    # a higher sensitivity floor cannot require a higher threshold
    assert thr_90 <= thr_60 + 1e-9


def test_confusion_counts_consistent():
    y = [0, 1, 1, 0]
    c = M.confusion_at(y, [0.2, 0.8, 0.4, 0.6], 0.5)
    assert c["tp"] + c["fn"] == 2  # two positives
    assert c["tn"] + c["fp"] == 2  # two negatives


def test_ece_bounds():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    proba = rng.uniform(0, 1, 200)
    ece = M.expected_calibration_error(y, proba)
    assert 0.0 <= ece <= 1.0


def test_subgroup_report_shape():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 100)
    proba = rng.uniform(0, 1, 100)
    groups = np.where(np.arange(100) % 2 == 0, "A", "B")
    rep = M.subgroup_report(y, proba, groups)
    assert set(rep.keys()) == {"A", "B"}
    for g in rep.values():
        assert "auroc" in g and "ece" in g and "n" in g and "prevalence" in g


def test_auprc_matches_prevalence_for_random():
    # AUPRC of random scores ~ prevalence
    y = np.array([1] * 30 + [0] * 70)
    rng = np.random.default_rng(3)
    proba = rng.uniform(0, 1, 100)
    ap = M.auprc(y, proba)
    assert 0.15 < ap < 0.5  # loosely around the 0.30 prevalence
