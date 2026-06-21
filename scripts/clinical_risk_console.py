#!/usr/bin/env python
"""Clinical Risk Console (Day-1 Module 1).

Train and *honestly evaluate* a clinical risk model on the bundled Diabetes set
(442 patients; demographic groups sex=A/B; binary label high_progression):

  1. baseline + honest metrics: AUROC, AUPRC (the one that matters under imbalance),
     confusion matrix, and a calibration curve;
  2. threshold as a clinical decision: a slider trading FN vs FP;
  3. ⚠️ subgroup audit: AUROC + calibration per demographic subgroup.

  python clinical_risk_console.py --smoke      # train + metrics + gate, no UI
  python clinical_risk_console.py --serve       # launch the Gradio app
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from common import aimed_config as cfg            # noqa: E402
from common import metrics as M                    # noqa: E402

LABEL = "high_progression"
GROUP = "sex"
SEED = 0


def load_data():
    """Return (df, X, y, groups, feature_names). sex is BOTH a feature and the
    protected attribute for the audit."""
    df = pd.read_csv(cfg.DIABETES_CSV)
    df = df.copy()
    groups = df[GROUP].astype(str).to_numpy()
    feats = df.drop(columns=[LABEL]).copy()
    feats[GROUP] = (feats[GROUP].astype(str) == "B").astype(int)  # A->0, B->1
    X = feats.to_numpy(dtype=float)
    y = df[LABEL].to_numpy(dtype=int)
    return df, X, y, groups, list(feats.columns)


def split(X, y, groups, test_size=0.3):
    return train_test_split(X, y, groups, test_size=test_size,
                            random_state=SEED, stratify=y)


def train_models(X_train, y_train):
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=2000, random_state=SEED))
    hgb = HistGradientBoostingClassifier(random_state=SEED)
    lr.fit(X_train, y_train)
    hgb.fit(X_train, y_train)
    return {"logreg": lr, "histgb": hgb}


def evaluate(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    return {"auroc": M.auroc(y_test, proba),
            "auprc": M.auprc(y_test, proba),
            "ece": M.expected_calibration_error(y_test, proba),
            "proba": proba}


def subgroup_audit(proba, y_test, groups_test) -> dict:
    return M.subgroup_report(y_test, proba, groups_test)


def gate1(proba, y_test, target_sensitivity=0.90) -> dict:
    """Quantities Day-1 Mastery Gate 1 checks."""
    thr, sens, spec = M.threshold_at_sensitivity(y_test, proba, target_sensitivity)
    return {"threshold_at_90_sensitivity": thr, "sensitivity": sens,
            "specificity": spec, "auprc": M.auprc(y_test, proba)}


# --------------------------------------------------------------------------- #
def run_report(log=print):
    df, X, y, groups, feats = load_data()
    Xtr, Xte, ytr, yte, gtr, gte = split(X, y, groups)
    models = train_models(Xtr, ytr)
    log(f"[console] features={feats}")
    log(f"[console] n_train={len(ytr)} n_test={len(yte)} "
        f"prevalence={y.mean():.2f}")
    results = {}
    for name, m in models.items():
        ev = evaluate(m, Xte, yte)
        results[name] = ev
        log(f"[console] {name:8s}  AUROC={ev['auroc']:.3f}  "
            f"AUPRC={ev['auprc']:.3f}  ECE={ev['ece']:.3f}")
    # use the gradient-boosting model as the console default
    best = results["histgb"]
    g1 = gate1(best["proba"], yte)
    log(f"[console] threshold@>=0.90 sens = {g1['threshold_at_90_sensitivity']:.3f} "
        f"(sens={g1['sensitivity']:.2f}, spec={g1['specificity']:.2f})  "
        f"AUPRC={g1['auprc']:.3f}")
    audit = subgroup_audit(best["proba"], yte, gte)
    log("[console] subgroup audit (⚠️ look for equal AUROC but different ECE):")
    for grp, rep in audit.items():
        a = "n/a" if rep["auroc"] is None else f"{rep['auroc']:.3f}"
        log(f"           sex={grp}: n={rep['n']:3d} prev={rep['prevalence']:.2f} "
            f"AUROC={a} ECE={rep['ece']:.3f}")
    return {"models": models, "results": results, "gate1": g1, "audit": audit,
            "Xte": Xte, "yte": yte, "gte": gte}


# --------------------------------------------------------------------------- #
def build_app():
    """Gradio app: threshold slider -> live sensitivity/specificity/FN/FP."""
    import gradio as gr
    df, X, y, groups, feats = load_data()
    Xtr, Xte, ytr, yte, gtr, gte = split(X, y, groups)
    model = train_models(Xtr, ytr)["histgb"]
    proba = model.predict_proba(Xte)[:, 1]
    base_auroc, base_auprc = M.auroc(yte, proba), M.auprc(yte, proba)

    def at_threshold(threshold):
        c = M.confusion_at(yte, proba, threshold)
        return (f"AUROC {base_auroc:.3f} | AUPRC {base_auprc:.3f}\n"
                f"Threshold {threshold:.2f}\n"
                f"Sensitivity {c['sensitivity']:.2f} | Specificity {c['specificity']:.2f}\n"
                f"Missed cases (FN) {c['fn']} | False alarms (FP) {c['fp']}")

    with gr.Blocks(title="Clinical Risk Console") as demo:
        gr.Markdown("## Clinical Risk Console — threshold is a clinical decision")
        s = gr.Slider(0.05, 0.95, value=0.5, step=0.01, label="Decision threshold")
        out = gr.Textbox(label="Clinical impact", lines=4)
        s.change(at_threshold, s, out)
        demo.load(at_threshold, s, out)
    return demo


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serve", action="store_true", help="launch the Gradio app")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    st = run_report()
    # Gate-1 self-check (the notebook asserts the same)
    g1 = st["gate1"]
    assert g1["sensitivity"] >= 0.90 - 1e-9, "threshold did not reach 0.90 sensitivity"
    assert 0.3 < g1["auprc"] < 1.0, "AUPRC outside a sane band"
    print(f"\nGate-1 quantities OK: thr={g1['threshold_at_90_sensitivity']:.3f} "
          f"AUPRC={g1['auprc']:.3f}")

    if args.serve and not args.smoke:
        app = build_app()
        app.launch(server_name="0.0.0.0", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
