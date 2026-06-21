"""Unit tests for the load-bearing Day-2 scoring math (delta = LL(alt) - LL(ref))."""
import numpy as np
import pandas as pd

from common import genomics
from vep_scorer import score_variant


class FakeScorer:
    """Deterministic stand-in: log-likelihood penalizes 'A' bases by -2 each.

    So substituting a non-A base with 'A' must DECREASE the score (delta < 0),
    which is the sign convention the whole VEP method depends on.
    """
    name = "fake"

    def sequence_logprob(self, seq: str) -> float:
        return -2.0 * seq.upper().count("A")

    def score_sequences(self, seqs, batch_size=8, progress=False):
        return [self.sequence_logprob(s) for s in seqs]


FULL = "ACGT" * 10  # 40 bp deterministic reference


def test_score_variant_sign():
    s = FakeScorer()
    # turning a 'G' into 'A' adds one A -> score drops by 2 -> delta = -2
    assert score_variant("CGTCGT", "CATCGT", s) == -2.0
    # turning an 'A' into 'G' removes one A -> delta = +2
    assert score_variant("CATCGT", "CGTCGT", s) == +2.0


def test_score_variant_is_alt_minus_ref():
    s = FakeScorer()
    ref, alt = "GGGG", "GAGG"
    assert score_variant(ref, alt, s) == s.sequence_logprob(alt) - s.sequence_logprob(ref)


def test_build_window_centers_and_substitutes():
    # pos 11 (1-based) sits on a 'G' in ACGT-repeat; window 8 -> flank 4
    ref_w, alt_w, off, observed = genomics.build_window(11, "G", "A", FULL, window=8)
    assert observed == "G"                      # center matches the reference base
    assert ref_w[off] == "G"
    assert alt_w[off] == "A"                     # substitution applied at the SNV offset
    assert len(ref_w) == len(alt_w) == 8         # SNV preserves length
    assert ref_w[:off] == alt_w[:off] and ref_w[off + 1:] == alt_w[off + 1:]


def test_build_variant_windows_flags_mismatch():
    df_ok = pd.DataFrame({"pos_hg19": [11], "reference": ["G"], "alt": ["A"]})
    _, _, _, mism = genomics.build_variant_windows(df_ok, FULL, window=8)
    assert mism == 0
    df_bad = pd.DataFrame({"pos_hg19": [11], "reference": ["T"], "alt": ["A"]})  # wrong ref
    _, _, _, mism_bad = genomics.build_variant_windows(df_bad, FULL, window=8)
    assert mism_bad == 1


def test_dedup_reference_windows():
    windows = ["AAA", "BBB", "AAA", "CCC", "BBB"]
    uniq, idx = genomics.dedup_reference_windows(windows)
    assert uniq == ["AAA", "BBB", "CCC"]
    assert list(idx) == [0, 1, 0, 2, 1]
    # reconstruct original from unique + index
    assert [uniq[i] for i in idx] == windows


def test_score_variants_end_to_end():
    s = FakeScorer()
    # three SNVs at pos 11 (G): G>A (adds A), G>C (no A change), G>T (no A change)
    df = pd.DataFrame({
        "pos_hg19": [11, 11, 11],
        "reference": ["G", "G", "G"],
        "alt": ["A", "C", "T"],
        "is_lof": [1, 0, 0],
    })
    deltas, info = genomics.score_variants(df, s, FULL, window=8, batch_size=2)
    assert info["n"] == 3
    assert info["n_unique_ref"] == 1            # all share the same ref window
    assert deltas[0] == -2.0                     # G>A introduces an A -> delta -2
    assert deltas[1] == 0.0 and deltas[2] == 0.0 # G>C, G>T do not change A count


def test_reverse_complement_and_frameshift():
    assert genomics.reverse_complement("ACGT") == "ACGT"      # palindrome
    assert genomics.reverse_complement("AAAA") == "TTTT"
    seq = "ACGTACGT"
    fs = genomics.insert_frameshift(seq, at=4, base="A")
    assert len(fs) == len(seq) + 1
    assert fs == "ACGTAACGT"
