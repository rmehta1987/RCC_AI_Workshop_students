"""Honest-evaluation metrics shared across Day-1 (clinical) and Day-2 (VEP).

These back the mastery gates, so they are written to be deterministic and to
return plain Python floats (easy to assert on).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

try:  # sklearn >=1.3 path
    from sklearn.calibration import calibration_curve
except Exception:  # pragma: no cover
    calibration_curve = None


def auroc(y_true, scores) -> float:
    """Area under ROC. `scores` must be oriented so higher = more positive."""
    return float(roc_auc_score(np.asarray(y_true), np.asarray(scores)))


def auprc(y_true, scores) -> float:
    """Average precision (area under PR curve) — the metric that matters under imbalance."""
    return float(average_precision_score(np.asarray(y_true), np.asarray(scores)))


def confusion_at(y_true, proba, threshold: float) -> dict:
    """Confusion counts + sensitivity/specificity at a decision threshold."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": sens, "specificity": spec, "threshold": float(threshold)}


def threshold_at_sensitivity(y_true, proba, target_sensitivity: float = 0.90):
    """Highest decision threshold that still achieves >= target sensitivity (recall).

    As the threshold rises, sensitivity is non-increasing, so we want the most
    specific threshold whose sensitivity floor is still met. Returns
    (threshold, achieved_sensitivity, specificity_at_threshold).
    This is the quantity Day-1 Mastery Gate 1 checks.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    candidates = np.unique(proba)  # ascending
    best = None
    for t in candidates:
        c = confusion_at(y_true, proba, t)
        if c["sensitivity"] >= target_sensitivity:
            best = (float(t), c["sensitivity"], c["specificity"])  # keep highest t meeting floor
    if best is None:  # nothing meets it; return the lowest threshold (max sensitivity)
        c = confusion_at(y_true, proba, candidates[0])
        return (float(candidates[0]), c["sensitivity"], c["specificity"])
    return best


def calibration_points(y_true, proba, n_bins: int = 10):
    """(prob_true, prob_pred) for a reliability diagram. Falls back to manual bins."""
    y_true = np.asarray(y_true, dtype=float)
    proba = np.asarray(proba, dtype=float)
    if calibration_curve is not None:
        try:
            pt, pp = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
            return pt, pp
        except Exception:
            pass
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(proba, bins) - 1, 0, n_bins - 1)
    pt, pp = [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            pt.append(y_true[m].mean())
            pp.append(proba[m].mean())
    return np.array(pt), np.array(pp)


def expected_calibration_error(y_true, proba, n_bins: int = 10) -> float:
    """ECE: sum over bins of |acc-conf| weighted by bin population. Lower is better."""
    y_true = np.asarray(y_true, dtype=float)
    proba = np.asarray(proba, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(proba, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    n = len(proba)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            acc = y_true[m].mean()
            conf = proba[m].mean()
            ece += (m.sum() / n) * abs(acc - conf)
    return float(ece)


def subgroup_report(y_true, proba, groups) -> dict:
    """Per-subgroup AUROC / AUPRC / ECE / prevalence — the ⚠️ fairness audit.

    `groups` is an array of group labels (e.g. sex A/B). Returns {group: {...}}.
    A subgroup with one class present cannot have an AUROC (reported as None).
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    groups = np.asarray(groups)
    out = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        yg, pg = y_true[m], proba[m]
        rep = {"n": int(m.sum()), "prevalence": float(yg.mean()) if len(yg) else float("nan")}
        rep["auroc"] = float(roc_auc_score(yg, pg)) if len(set(yg.tolist())) == 2 else None
        rep["auprc"] = float(average_precision_score(yg, pg)) if len(set(yg.tolist())) == 2 else None
        rep["ece"] = expected_calibration_error(yg, pg) if len(yg) else None
        out[str(g)] = rep
    return out


def bootstrap_auroc_ci(y_true, scores, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI for AUROC (used to defend whether a gap is real)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y_true[idx].tolist())) < 2:
            continue
        vals.append(roc_auc_score(y_true[idx], scores[idx]))
    if not vals:
        return (float("nan"), float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(np.mean(vals)), float(lo), float(hi))
