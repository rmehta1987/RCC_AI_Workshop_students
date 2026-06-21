"""Headless (Agg) plotting helpers used by both the scripts and the notebooks."""
from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Agg")  # safe on compute nodes / nbconvert
import matplotlib.pyplot as plt  # noqa: E402

from common import metrics as M  # noqa: E402


def calibration_plot(y_true, proba, n_bins=10, title="Calibration", ax=None):
    pt, pp = M.calibration_points(y_true, proba, n_bins=n_bins)
    ece = M.expected_calibration_error(y_true, proba, n_bins=n_bins)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax.plot(pp, pt, "o-", label=f"model (ECE={ece:.3f})")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    return ax


def delta_distribution_plot(deltas, is_lof, title="VEP delta distributions", ax=None):
    """Two histograms (benign vs LoF) of the delta score — the Day-2 'aha' figure."""
    deltas = np.asarray(deltas)
    is_lof = np.asarray(is_lof)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.hist(deltas[is_lof == 0], bins=40, alpha=0.6, density=True, label="benign (is_lof=0)")
    ax.hist(deltas[is_lof == 1], bins=40, alpha=0.6, density=True, label="LoF (is_lof=1)")
    ax.set_xlabel("delta = LL(alt) - LL(ref)   (more negative -> more disruptive)")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    return ax


def gradcam_overlay(image, cam, alpha=0.5, ax=None, title="Grad-CAM"):
    """Overlay a [H,W] CAM heatmap on a grayscale [H,W] (or [1,H,W]) image."""
    img = np.asarray(image, dtype=float)
    if img.ndim == 3:
        img = img[0]
    img = (img - img.min()) / (img.ptp() + 1e-8)
    cam = np.asarray(cam, dtype=float)
    cam = (cam - cam.min()) / (cam.ptp() + 1e-8)
    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(img, cmap="gray")
    ax.imshow(cam, cmap="jet", alpha=alpha)
    ax.set_title(title)
    ax.axis("off")
    return ax


def fusion_bar(aurocs: dict, title="Fusion vs single-modality AUROC", ax=None):
    """Bar chart of {modality: auroc} for image-only / tabular-only / fused."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.4, 3.4))
    keys = list(aurocs.keys())
    vals = [aurocs[k] for k in keys]
    ax.bar(keys, vals, color=["#4C72B0", "#DD8452", "#55A868"][: len(keys)])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUROC")
    ax.set_title(title)
    return ax


def savefig(fig_or_ax, path):
    fig = getattr(fig_or_ax, "figure", fig_or_ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return path
