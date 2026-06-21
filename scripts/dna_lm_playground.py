#!/usr/bin/env python
"""DNA Language Model Playground (Day-2 Module 1).

Treat DNA as a language: get likelihoods and generations out of a DNA LM
(HyenaDNA by default; --model evo2_7b to upgrade) and read them biologically.

Three guided probes from the design:
  1. generate & inspect a sampled sequence;
  2. likelihood as a sensor -> a frameshift collapses the score (Mastery Gate 1);
  3. context matters -> score a short vs a longer window around a site.

  python dna_lm_playground.py --smoke              # CPU, phiX gene, seconds
  python dna_lm_playground.py --generate 80
  python dna_lm_playground.py --model evo2_7b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from common import aimed_config as cfg            # noqa: E402
from common import genomics                       # noqa: E402


def load_coding_example() -> str:
    """A real protein-coding sequence: phiX174 G-protein (bundled extras).

    NB: phiX174 is a *bacteriophage* — out-of-distribution for the human-genome-
    trained HyenaDNA, so it scores a high perplexity. For the Module-1 'real DNA
    looks natural' demo use :func:`load_human_example` instead; this stays for the
    prokaryotic/seqmodeling extras.
    """
    fa = cfg.SEQ_EXTRAS / "NC_001422.1_Gprotein.fasta"
    seq = []
    if fa.exists():
        for line in fa.read_text().splitlines():
            if not line.startswith(">"):
                seq.append(line.strip())
    s = "".join(seq).upper()
    # fallback tiny coding-ish sequence if the file is missing
    return s or ("ATG" + "GCTGCAGCTGGTGCT" * 8 + "TAA")


# A small real human window committed to the repo so Module 1 runs on a bare git
# clone — the full chr17 FASTA is gitignored (22 MB; bundle-only). It is a verbatim
# slice of GRCh37 chr17 and covers the default BRCA1-locus windows used in M1.
_HUMAN_SNIPPET = cfg.SEQ_EXTRAS / "GRCh37_chr17_BRCA1_4kb.fasta"
_HUMAN_SNIPPET_START = 41_250_000   # hg19 chr17 coord the snippet's first base maps to


def _read_fasta_seq(path) -> str:
    return "".join(l.strip() for l in path.read_text().splitlines()
                   if not l.startswith(">")).upper()


def load_human_example(window: int = 300, start: int = 41_250_000) -> str:
    """A real window of the HUMAN genome: chr17 in the BRCA1 locus (hg19) — the
    same chromosome/region scored in Modules 2-3.

    In-distribution for the human-genome-trained HyenaDNA, so (unlike an out-of-
    organism example) its likelihood beats the coin-flip-among-4 baseline.

    Source order, so the call works everywhere with identical results:
      1. the small committed snippet (always present, even on a bare git clone),
         whenever it covers the requested ``[start, start+window)``;
      2. otherwise the full chr17 reference from the dataset bundle, scanning
         forward past any assembly-gap (``N``) run.
    """
    if _HUMAN_SNIPPET.exists():
        snip = _read_fasta_seq(_HUMAN_SNIPPET)
        off = int(start) - _HUMAN_SNIPPET_START
        if 0 <= off and off + window <= len(snip):
            return snip[off:off + window]
    if cfg.CHR17_FASTA.exists():
        full = genomics.load_chr_sequence(str(cfg.CHR17_FASTA))
        s = max(0, min(int(start), len(full) - window))
        for _ in range(2000):                   # nudge past any N-run; the locus is clean
            seq = full[s:s + window].upper()
            if "N" not in seq and len(seq) == window:
                return seq
            s += window
        raise ValueError("could not find an N-free chr17 window near the requested start")
    raise FileNotFoundError(
        f"need the committed human snippet ({_HUMAN_SNIPPET.name}) or the full chr17 "
        f"reference ({cfg.CHR17_FASTA.name}). On the cluster, unpack the dataset bundle; "
        f"locally, the snippet ships in the repo (data/day2_genomics/seqmodeling_extras/).")


def frameshift_demo(scorer, seq: str = None) -> dict:
    """Score a coding sequence and its frameshifted copy. Frameshift must score lower.

    This is the property Day-2 Mastery Gate 1 asserts.
    """
    seq = (seq or load_human_example(window=1000))[:1000]  # real human DNA, CPU-friendly
    ll_ref = scorer.sequence_logprob(seq)
    shifted = genomics.insert_frameshift(seq)  # insert one base -> frame breaks
    ll_fs = scorer.sequence_logprob(shifted)
    return {"ll_ref": float(ll_ref), "ll_frameshift": float(ll_fs),
            "drop": float(ll_ref - ll_fs), "frameshift_lower": bool(ll_fs < ll_ref),
            "len": len(seq)}


def context_demo(scorer, seq: str = None) -> dict:
    """Score a short vs a longer window around the same start site."""
    seq = seq or load_human_example(window=1024)
    short = seq[: min(128, len(seq))]
    long = seq[: min(1024, len(seq))]
    return {"ll_short_128": float(scorer.sequence_logprob(short)),
            "ll_long_1024": float(scorer.sequence_logprob(long)),
            "len_short": len(short), "len_long": len(long)}


def generate_demo(scorer, prompt: str = "ATGGCG", n: int = 64, seed: int = 0) -> str:
    try:
        return scorer.generate(prompt, max_new_tokens=n, seed=seed)
    except TypeError:  # evo2 generate has a different signature (no seed kwarg)
        try:
            return scorer.generate(prompt, max_new_tokens=n)
        except Exception as e:
            return f"[generate unavailable: {e}]"
    except Exception as e:
        return f"[generate unavailable: {e}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--generate", type=int, default=0, help="sample N tokens from a prompt")
    ap.add_argument("--prompt", default="ATGGCG")
    ap.add_argument("--seq", default=None, help="override the coding sequence")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg.setup_caches()
    device = "cpu" if args.smoke else args.device
    from common import models
    print(f"[playground] loading '{args.model or cfg.PRIMARY_GENOMIC_MODEL}' (device={device}) ...")
    scorer = models.load_dna_model(args.model, device=device)

    print("\n== Probe 2: likelihood as a sensor (frameshift) ==")
    fs = frameshift_demo(scorer, args.seq)
    print(f"  LL(reference)  = {fs['ll_ref']:.2f}")
    print(f"  LL(frameshift) = {fs['ll_frameshift']:.2f}")
    print(f"  drop           = {fs['drop']:+.2f}   frameshift_lower={fs['frameshift_lower']}")
    assert fs["frameshift_lower"], "Expected the frameshifted sequence to score lower!"

    print("\n== Probe 3: context matters ==")
    cd = context_demo(scorer, args.seq)
    print(f"  LL(128bp)={cd['ll_short_128']:.2f}   LL(1024bp)={cd['ll_long_1024']:.2f}")

    if args.generate:
        print("\n== Probe 1: generate & inspect ==")
        gen = generate_demo(scorer, args.prompt, n=args.generate)
        print(f"  prompt={args.prompt!r} -> {gen[:160]}")

    print("\n[playground] OK")


if __name__ == "__main__":
    main()
