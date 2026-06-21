#!/usr/bin/env python
"""Zero-Shot Variant-Effect Predictor (Day-2 Module 2).

The methodological heart of Day 2: for a single-nucleotide variant, score a
window of reference sequence and the same window with the variant substituted;
the delta is the signal.

    delta = LL(alt_window) - LL(ref_window)
    # more negative delta -> more disruptive -> more likely pathogenic
    AUROC  = roc_auc_score(is_lof, -delta)

Model-agnostic: identical for HyenaDNA-small-32k (default, CPU-ok) and Evo 2 7B.

Fallbacks (graceful degradation, Goal 13):
  * no GPU      -> HyenaDNA runs on CPU (slower; use --n to subset)
  * model/weights unavailable or --use-precomputed -> read the precomputed table
  * --smoke     -> tiny subset so the code path is provable offline in seconds

Examples
--------
  python vep_scorer.py --smoke                       # 24 variants, CPU, seconds
  python vep_scorer.py --n 400 --device cpu          # quick partial AUROC on CPU
  python vep_scorer.py --model evo2_7b --out scored.csv
  python vep_scorer.py --use-precomputed             # no model; uses the table
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from common import aimed_config as cfg            # noqa: E402
from common import genomics, metrics             # noqa: E402


# --------------------------------------------------------------------------- #
# The exact design function (imported by the Module-2 notebook + its gate)
# --------------------------------------------------------------------------- #
def score_variant(ref_window: str, alt_window: str, scorer) -> float:
    """delta = LL(alt) - LL(ref). The one function the whole day rests on."""
    return scorer.sequence_logprob(alt_window) - scorer.sequence_logprob(ref_window)


def auroc_from_table(path) -> float:
    """AUROC vs is_lof from a precomputed delta table (any of the known delta cols)."""
    if not path or not Path(path).exists():
        return None
    pre = pd.read_csv(path)
    dcol = next((c for c in _DELTA_COLS if c in pre.columns), None)
    if dcol is None or "is_lof" not in pre.columns:
        return None
    return metrics.auroc(pre["is_lof"].to_numpy(), -pre[dcol].to_numpy())


def evo2_auroc():
    """AUROC from the REAL Evo 2 7B precomputed table (measured 0.877 on Midway3),
    or None if Evo 2 has not been run/generated yet. HyenaDNA is ~chance (0.46),
    so this is the headline 'scale matters' number when present.
    (The bundled 1B reference notebook shows 0.73; 7B scores higher.)"""
    return auroc_from_table(cfg.evo2_precomputed_table())


# --------------------------------------------------------------------------- #
def load_brca1(n: int = None, seed: int = 0) -> pd.DataFrame:
    df = pd.read_csv(cfg.BRCA1_CSV)
    if n:
        # stratified-ish subset: keep both classes
        pos = df[df.is_lof == 1]
        neg = df[df.is_lof == 0]
        k = max(1, n // 2)
        df = pd.concat([pos.sample(min(k, len(pos)), random_state=seed),
                        neg.sample(min(n - k, len(neg)), random_state=seed)])
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


_DELTA_COLS = ["delta", "delta_score", "evo2_delta_score", "evo_delta_score", "score"]


def deltas_from_precomputed(df: pd.DataFrame):
    """Merge a precomputed delta table onto df (by pos/ref/alt). Returns aligned deltas."""
    table = cfg.precomputed_vep_table()
    pre = pd.read_csv(table)
    dcol = next((c for c in _DELTA_COLS if c in pre.columns), None)
    if dcol is None:
        raise ValueError(f"No delta column in {table} (have {list(pre.columns)})")
    keys = ["pos_hg19", "reference", "alt"]
    merged = df.merge(pre[keys + [dcol]].rename(columns={dcol: "delta"}),
                      on=keys, how="left")
    coverage = merged["delta"].notna().mean()
    return merged["delta"].to_numpy(), float(coverage), str(table)


def evo2_scored_table():
    """BRCA1 variants with the REAL Evo 2 7B delta merged on as a 'delta' column — ready for
    brca1_triage.vus_triage() and metrics.auroc(df.is_lof, -df.delta). Mirrors
    deltas_from_precomputed() but uses the Evo 2 table; returns the merged DataFrame, or
    None if that table is absent (smoke / no-Evo2 fallback)."""
    table = cfg.evo2_precomputed_table()
    if table is None:
        return None
    df = load_brca1()
    pre = pd.read_csv(table)
    dcol = next((c for c in _DELTA_COLS if c in pre.columns), None)
    if dcol is None:
        return None
    keys = ["pos_hg19", "reference", "alt"]
    return df.merge(pre[keys + [dcol]].rename(columns={dcol: "delta"}), on=keys, how="left")


def run_vep(model_key: str = None, n: int = None, window: int = genomics.DEFAULT_WINDOW,
            device: str = "auto", batch_size: int = 8, use_precomputed: bool = False,
            out: str = None, seed: int = 0, log=print):
    """Score BRCA1 variants and return (df_with_delta, auroc, info)."""
    df = load_brca1(n=n, seed=seed)
    info = {"n": len(df), "window": window, "source": None, "model": model_key}

    deltas = None
    if not use_precomputed:
        try:
            from common import models
            log(f"[vep] loading model '{model_key or cfg.PRIMARY_GENOMIC_MODEL}' "
                f"(device={device}) ...")
            scorer = models.load_dna_model(model_key, device=device)
            full = genomics.load_chr_sequence(str(cfg.CHR17_FASTA))
            deltas, sinfo = genomics.score_variants(
                df, scorer, full, window=window, batch_size=batch_size, log=log)
            info.update(sinfo)
            info["source"] = f"model:{sinfo.get('model')}"
        except Exception as e:
            log(f"[vep] model scoring unavailable ({e}); falling back to precomputed table.")
            deltas = None

    if deltas is None:
        deltas, cov, table = deltas_from_precomputed(df)
        info["source"] = f"precomputed:{Path(table).name} (coverage {cov:.0%})"
        if np.isnan(deltas).any():
            keep = ~np.isnan(deltas)
            n_drop = int((~keep).sum())
            log(f"[vep] precomputed table covers {keep.mean():.0%} of the subset; "
                f"dropping {n_drop} uncovered rows.")
            df = df[keep].reset_index(drop=True)
            deltas = deltas[keep]

    df = df.copy()
    df["delta"] = deltas
    # AUROC: negate delta so that 'more negative -> more pathogenic' -> higher risk score
    auroc = metrics.auroc(df["is_lof"].to_numpy(), -df["delta"].to_numpy())
    auprc = metrics.auprc(df["is_lof"].to_numpy(), -df["delta"].to_numpy())
    info["auroc"] = auroc
    info["auprc"] = auprc

    log(f"[vep] source = {info['source']}")
    log(f"[vep] n={len(df)}  window={window}  "
        f"AUROC={auroc:.3f}  AUPRC={auprc:.3f}")
    log(f"[vep] mean delta  benign={df.loc[df.is_lof==0,'delta'].mean():+.3f}  "
        f"LoF={df.loc[df.is_lof==1,'delta'].mean():+.3f}  (LoF should be lower)")

    if out:
        df.to_csv(out, index=False)
        log(f"[vep] wrote scored table -> {out}")
    return df, auroc, info


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None,
                    help="model key (default: configured primary, hyenadna-small-32k)")
    ap.add_argument("--n", type=int, default=None, help="subset to N variants (both classes)")
    ap.add_argument("--window", type=int, default=genomics.DEFAULT_WINDOW)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--use-precomputed", action="store_true",
                    help="skip the model; read the precomputed delta table")
    ap.add_argument("--out", default=None, help="write scored CSV here")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny subset on CPU (proves the path in seconds)")
    args = ap.parse_args()

    cfg.setup_caches()
    if args.smoke:
        args.n = args.n or 24
        args.device = "cpu"
        args.window = min(args.window, 1024)  # smaller window => fast CPU smoke
        print("[vep] SMOKE: 24 variants, CPU, window<=1024")

    df, auroc, info = run_vep(model_key=args.model, n=args.n, window=args.window,
                              device=args.device, batch_size=args.batch_size,
                              use_precomputed=args.use_precomputed, out=args.out)
    print(f"\nRESULT  AUROC={auroc:.3f}  ({info['source']})")


if __name__ == "__main__":
    main()
