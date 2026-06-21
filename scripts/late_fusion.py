#!/usr/bin/env python
"""Late-Fusion Patient Model (Day-1 Module 3a).

Combine a frozen image encoder's embeddings (the Module-2 CNN, penultimate layer)
with tabular features, and reason about when fusion helps and when it hurts.

The bundled `multimodal_smoke.npz` is the constructed/aligned fusion set: 600
patients, a 512-d image embedding + 8 tabular features + a label that depends on
BOTH modalities (so fusion *can* beat either alone). We compare three AUROCs —
image-only, tabular-only, fused — then run the ⚠️ missing-modality drill.

  python late_fusion.py --smoke
  python late_fusion.py --missing-frac 0.3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from common import aimed_config as cfg            # noqa: E402
from common import metrics as M                    # noqa: E402

SEED = 0


def load_multimodal():
    z = np.load(cfg.MULTIMODAL_SMOKE)
    return z["img_emb"].astype("float32"), z["tab"].astype("float32"), z["y"].astype(int)


def _fit_eval(Xtr, ytr, Xte, yte):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, random_state=SEED))
    clf.fit(Xtr, ytr)
    return M.auroc(yte, clf.predict_proba(Xte)[:, 1])


def three_way(img, tab, y):
    """Return {image_only, tabular_only, fused} AUROCs on a shared split."""
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.3, random_state=SEED, stratify=y)
    fused = np.concatenate([img, tab], axis=1)
    return {
        "image_only": _fit_eval(img[tr], y[tr], img[te], y[te]),
        "tabular_only": _fit_eval(tab[tr], y[tr], tab[te], y[te]),
        "fused": _fit_eval(fused[tr], y[tr], fused[te], y[te]),
    }


def missing_modality(img, tab, y, frac=0.3, strategy="flag"):
    """Null out the image for `frac` of patients; handle it three ways.

    strategy in {drop, zero_impute, flag}. Returns fused AUROC under that policy.
    """
    rng = np.random.default_rng(SEED)
    n = len(y)
    missing = rng.random(n) < frac
    idx = np.arange(n)
    tr, te = train_test_split(idx, test_size=0.3, random_state=SEED, stratify=y)

    if strategy == "drop":
        keep_tr = tr[~missing[tr]]
        keep_te = te[~missing[te]]
        fused = np.concatenate([img, tab], axis=1)
        if len(set(y[keep_te].tolist())) < 2:
            return float("nan")
        return _fit_eval(fused[keep_tr], y[keep_tr], fused[keep_te], y[keep_te])

    img2 = img.copy()
    img2[missing] = 0.0  # zero-impute the missing embedding
    if strategy == "zero_impute":
        fused = np.concatenate([img2, tab], axis=1)
    elif strategy == "flag":
        present = (~missing).astype("float32")[:, None]
        fused = np.concatenate([img2, tab, present], axis=1)  # modality-present flag
    else:
        raise ValueError(strategy)
    return _fit_eval(fused[tr], y[tr], fused[te], y[te])


def run_report(missing_frac=0.3, log=print):
    img, tab, y = load_multimodal()
    log(f"[fusion] img_emb={img.shape} tab={tab.shape} prevalence={y.mean():.2f}")
    aurocs = three_way(img, tab, y)
    for k, v in aurocs.items():
        log(f"[fusion] {k:12s} AUROC={v:.3f}")
    drills = {s: missing_modality(img, tab, y, frac=missing_frac, strategy=s)
              for s in ("drop", "zero_impute", "flag")}
    log(f"[fusion] fused AUROC under {missing_frac:.0%} missing images:")
    for s, v in drills.items():
        log(f"           {s:12s} -> {v:.3f}")
    return {"aurocs": aurocs, "missing": drills, "missing_frac": missing_frac}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--missing-frac", type=float, default=0.3)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg.setup_caches()
    st = run_report(missing_frac=args.missing_frac)
    a = st["aurocs"]
    assert all(0.4 < v < 1.0 for v in a.values()), "an AUROC fell outside a sane band"
    print(f"\nGate-3a quantities OK: image={a['image_only']:.3f} "
          f"tabular={a['tabular_only']:.3f} fused={a['fused']:.3f} "
          f"fused@missing(flag)={st['missing']['flag']:.3f}")


if __name__ == "__main__":
    main()
