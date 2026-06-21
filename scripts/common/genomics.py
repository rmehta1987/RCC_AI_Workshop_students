"""GRCh37 chr17 reference handling + SNV window construction + scoring driver.

Mirrors the official Evo 2 BRCA1 zero-shot notebook exactly:
    WINDOW_SIZE = 8192 ; p = pos-1 ; ref window centered on the SNV ;
    var = ref[:off] + alt + ref[off+1:] ; assert ref[off] == reference base.

The same windows feed HyenaDNA or Evo 2 — the pipeline is model-agnostic; only
the `scorer.score_sequences(...)` call differs. HyenaDNA-small-32k (32 kb context)
comfortably handles the 8192-bp window, so the comparison to Evo 2 is apples-to-apples.
"""
from __future__ import annotations

import functools
import gzip
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_WINDOW = 8192


@functools.lru_cache(maxsize=2)
def load_chr_sequence(fasta_gz: str) -> str:
    """Load a single-record (optionally gzipped) FASTA into one uppercase string.

    chr17 GRCh37 is ~81 Mb (~81 MB in memory) — fine to hold once and slice. Cached
    so repeated variant scoring does not re-read the file.
    """
    opener = gzip.open if str(fasta_gz).endswith(".gz") else open
    chunks: List[str] = []
    with opener(fasta_gz, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                continue
            chunks.append(line.strip())
    return "".join(chunks).upper()


def build_window(pos: int, ref: str, alt: str, full_seq: str,
                 window: int = DEFAULT_WINDOW) -> Tuple[str, str, int, str]:
    """Build (ref_window, alt_window, snv_offset, observed_ref_base) for one SNV.

    `pos` is 1-based (hg19). Returns the reference base actually found at the SNV
    offset so callers can verify it matches the table's `reference` column.
    """
    p = pos - 1  # 0-indexed
    start = max(0, p - window // 2)
    end = min(len(full_seq), p + window // 2)
    ref_w = full_seq[start:end]
    off = min(window // 2, p)
    observed = ref_w[off] if 0 <= off < len(ref_w) else ""
    alt_w = ref_w[:off] + str(alt) + ref_w[off + 1:]
    return ref_w, alt_w, off, observed


def build_variant_windows(df: pd.DataFrame, full_seq: str,
                          window: int = DEFAULT_WINDOW, verify: bool = True):
    """Build ref/alt windows for every row of a BRCA1-style dataframe.

    Expects columns pos_hg19, reference, alt. Returns
    (ref_windows, alt_windows, offsets, n_ref_mismatch).
    A nonzero mismatch count means the coordinate build (hg19 vs hg38) is wrong —
    a load-bearing sanity check the notebooks surface.
    """
    ref_ws: List[str] = []
    alt_ws: List[str] = []
    offs: List[int] = []
    mism = 0
    for pos, ref, alt in zip(df["pos_hg19"], df["reference"], df["alt"]):
        rw, aw, off, obs = build_window(int(pos), str(ref), str(alt), full_seq, window)
        if verify and obs.upper() != str(ref).upper():
            mism += 1
        ref_ws.append(rw)
        alt_ws.append(aw)
        offs.append(off)
    return ref_ws, alt_ws, offs, mism


def dedup_reference_windows(ref_windows: Sequence[str]):
    """Map identical reference windows to a single index (the notebook's speed trick).

    Many SNVs share a position (e.g. T>A, T>C, T>G), hence the same ref window.
    Returns (unique_windows, index_array) where index_array[i] points the i-th
    variant at its unique ref window.
    """
    uniq: dict = {}
    order: List[str] = []
    idx: List[int] = []
    for w in ref_windows:
        if w not in uniq:
            uniq[w] = len(order)
            order.append(w)
        idx.append(uniq[w])
    return order, np.asarray(idx, dtype=int)


def score_variants(df: pd.DataFrame, scorer, full_seq: str,
                   window: int = DEFAULT_WINDOW, batch_size: int = 8,
                   verify: bool = True, log=print):
    """Compute delta = LL(alt) - LL(ref) for every variant in `df`.

    Returns (deltas: np.ndarray, info: dict). `scorer` is any object exposing
    `score_sequences(list[str], batch_size=...) -> list[float]` (HyenaDNA or Evo 2).
    Reference windows are de-duplicated so shared positions are scored once.
    """
    ref_ws, alt_ws, offs, mism = build_variant_windows(df, full_seq, window, verify)
    if verify and mism:
        log(f"[genomics] WARNING: {mism}/{len(df)} variants' reference base did not "
            f"match the chr17 sequence at the SNV offset (check hg19 vs hg38 / strand).")
    uniq_ref, ref_idx = dedup_reference_windows(ref_ws)
    log(f"[genomics] scoring {len(uniq_ref)} unique reference windows + "
        f"{len(alt_ws)} variant windows (window={window} bp) ...")
    ref_scores = np.asarray(scorer.score_sequences(uniq_ref, batch_size=batch_size))
    alt_scores = np.asarray(scorer.score_sequences(alt_ws, batch_size=batch_size))
    deltas = alt_scores - ref_scores[ref_idx]
    info = {
        "n": int(len(df)),
        "n_unique_ref": int(len(uniq_ref)),
        "window": int(window),
        "ref_mismatch": int(mism),
        "model": getattr(scorer, "name", "unknown"),
    }
    return deltas, info


# --- tiny biology helpers used by the sequence-modeling playground ----------- #
_COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


def insert_frameshift(seq: str, at: int = None, base: str = "A") -> str:
    """Insert a single base to shift the reading frame (collapses LL for a coding seq)."""
    if at is None:
        at = len(seq) // 2
    return seq[:at] + base + seq[at:]
