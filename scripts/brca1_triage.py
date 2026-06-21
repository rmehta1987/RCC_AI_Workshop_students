#!/usr/bin/env python
"""BRCA1 Clinical Triage (Day-2 Module 3).

Run the zero-shot VEP pipeline end-to-end on BRCA1, reproduce the benign-vs-
pathogenic separation, and build a triage table that ranks the VUS
(clinvar == "Uncertain significance") by delta, most-disruptive first.

With HyenaDNA expect a modest-but-real AUROC; the headline >0.9 is the Evo 2
result. `--compare-evo2` scores with both (or reads the precomputed Evo 2 table)
so students see the same method, performance scaling with model size.

  python brca1_triage.py --smoke
  python brca1_triage.py                       # full, primary model
  python brca1_triage.py --compare-evo2        # HyenaDNA vs Evo 2
  python brca1_triage.py --use-precomputed     # no GPU: analyse the precomputed table
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from common import aimed_config as cfg            # noqa: E402
from common import genomics                       # noqa: E402
from vep_scorer import run_vep                     # noqa: E402

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

VUS_LABEL = "Uncertain significance"


def vus_triage(df: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Rank VUS by delta (ascending = most disruptive first)."""
    vus = df[df["clinvar"] == VUS_LABEL].sort_values("delta")
    cols = [c for c in ["pos_hg19", "reference", "alt", "consequence",
                        "CADD.score", "delta"] if c in vus.columns]
    return vus[cols].head(top)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--compare-evo2", action="store_true",
                    help="also score with Evo 2 (or the precomputed Evo 2 table)")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--window", type=int, default=genomics.DEFAULT_WINDOW)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--use-precomputed", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg.setup_caches()
    if args.smoke:
        args.n = args.n or 60
        args.device = "cpu"
        args.window = min(args.window, 1024)
        print("[triage] SMOKE: 60 variants, CPU, window<=1024")

    print("=" * 64)
    print(f" BRCA1 triage — primary model: {args.model or cfg.PRIMARY_GENOMIC_MODEL}")
    print("=" * 64)
    df, auroc, info = run_vep(model_key=args.model, n=args.n, window=args.window,
                              device=args.device, batch_size=args.batch_size,
                              use_precomputed=args.use_precomputed, out=args.out)
    print(f"\n>>> Primary AUROC = {auroc:.3f}   ({info['source']})")

    print(f"\n--- VUS triage (top 10, most disruptive first) ---")
    try:
        print(vus_triage(df).to_string(index=False))
    except Exception as e:
        print(f"  (VUS table unavailable: {e})")

    if args.compare_evo2:
        print("\n" + "=" * 64)
        print(" Comparison: what does scale buy? (same method, bigger model)")
        print("=" * 64)
        try:
            # Prefer a real Evo 2 run; fall back to the precomputed Evo 2 table.
            df2, auroc2, info2 = run_vep(model_key="evo2_7b", n=args.n,
                                         window=args.window, device=args.device,
                                         batch_size=1, use_precomputed=False)
        except Exception as e:
            print(f"  Evo 2 model unavailable ({e}); using precomputed table.")
            df2, auroc2, info2 = run_vep(model_key="evo2_7b", n=args.n,
                                         window=args.window, use_precomputed=True)
        print(f"\n  HyenaDNA AUROC = {auroc:.3f}")
        print(f"  Evo 2    AUROC = {auroc2:.3f}   ({info2['source']})")
        print(f"  delta(AUROC)   = {auroc2 - auroc:+.3f}  "
              f"-> identical pipeline, performance scales with the model.")

    print("\n[triage] done. Remember: a variant score is a PRIOR, not a patient's risk.")


if __name__ == "__main__":
    main()
