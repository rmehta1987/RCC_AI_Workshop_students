"""Model-agnostic DNA language-model scorers.

Every scorer exposes the same tiny contract the whole of Day 2 relies on:

    scorer.sequence_logprob(seq: str) -> float          # higher = 'more natural'
    scorer.score_sequences(list[str], batch_size) -> list[float]
    scorer.generate(prompt: str, ...) -> str            # playground only

so `delta = scorer.sequence_logprob(alt) - scorer.sequence_logprob(ref)` is
identical across HyenaDNA-small-32k (default, CPU-ok) and Evo 2 7B (GPU upgrade).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import aimed_config as cfg  # noqa: E402


def resolve_repo(model_key: str) -> str:
    """Map a short key (or a full HF id) to a Hugging Face repo id."""
    key = model_key.lower()
    if "/" in model_key:  # already a full repo id
        return model_key
    if key in cfg.HF_REPOS:
        return cfg.HF_REPOS[key]
    # tolerant aliases
    if key.startswith("hyenadna") and "tiny" in key:
        return cfg.HF_REPOS["hyenadna-tiny-1k"]
    if key.startswith("hyenadna"):
        return cfg.HF_REPOS["hyenadna-small-32k"]
    if key.startswith("evo2"):
        return cfg.HF_REPOS["evo2_7b"]
    raise ValueError(f"Unknown model key: {model_key}")


# --------------------------------------------------------------------------- #
# HyenaDNA — the primary, CPU-capable path (transformers + trust_remote_code)
# --------------------------------------------------------------------------- #
class HyenaDNAScorer:
    """Autoregressive log-likelihood scorer for HyenaDNA `-hf` checkpoints.

    sequence_logprob = sum_t log p(x_{t+1} | x_<=t)   (the design's exact formula).
    Right-padding is safe because the model is causal: real positions' logits are
    unaffected by padding to their right, and padded targets are masked out.
    """

    name = "hyenadna"

    def __init__(self, repo: str, device: str = "auto", dtype: str = "float32"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.repo = repo
        self.device = ("cuda" if (device in ("auto", "cuda") and torch.cuda.is_available())
                       else "cpu")
        self.tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)
        self.model.eval()
        if self.device == "cuda" and dtype in ("float16", "half"):
            self.model = self.model.half()
        self.model.to(self.device)
        self.pad_id = (self.tok.pad_token_id
                       if getattr(self.tok, "pad_token_id", None) is not None else 0)

    def _encode(self, seq: str):
        return self.tok(seq, return_tensors="pt")["input_ids"][0]

    def score_sequences(self, seqs: Sequence[str], batch_size: int = 8,
                        progress: bool = False) -> List[float]:
        torch = self.torch
        enc = [self._encode(s) for s in seqs]
        out: List[float] = []
        rng = range(0, len(enc), batch_size)
        if progress:
            try:
                from tqdm import tqdm
                rng = tqdm(rng, desc=f"{self.name} score", unit="batch")
            except Exception:
                pass
        for i in rng:
            chunk = enc[i:i + batch_size]
            L = max(int(t.numel()) for t in chunk)
            ids = torch.full((len(chunk), L), self.pad_id, dtype=torch.long)
            lengths = torch.zeros(len(chunk), dtype=torch.long)
            for j, t in enumerate(chunk):
                ids[j, : t.numel()] = t
                lengths[j] = t.numel()
            ids = ids.to(self.device)
            with torch.inference_mode():
                logits = self.model(ids).logits.float()
            logp = torch.log_softmax(logits[:, :-1], dim=-1)
            tgt = ids[:, 1:]
            tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
            ar = torch.arange(L - 1, device=self.device).unsqueeze(0)
            mask = ar < (lengths.to(self.device) - 1).unsqueeze(1)
            seq_lp = (tok_lp * mask).sum(dim=1)
            out.extend(seq_lp.detach().float().cpu().tolist())
        return out

    def sequence_logprob(self, seq: str) -> float:
        return self.score_sequences([seq], batch_size=1)[0]

    def per_base_logprob(self, seq: str):
        """Per-position log-prob of the realized base (for the exon-structure plot)."""
        torch = self.torch
        ids = self._encode(seq).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(ids).logits.float()
        logp = torch.log_softmax(logits[:, :-1], dim=-1)
        tgt = ids[:, 1:]
        return logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0].detach().cpu().numpy()

    def generate(self, prompt: str, max_new_tokens: int = 64,
                 do_sample: bool = True, temperature: float = 1.0, seed: int = 0) -> str:
        # NOTE: the HyenaDNA `-hf` checkpoints expose no LM head, so HF `.generate()`
        # is unsupported. We never raise — generation is an optional probe; scoring
        # (sequence_logprob) is the load-bearing capability.
        torch = self.torch
        try:
            torch.manual_seed(seed)
            ids = self._encode(prompt).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                gen = self.model.generate(
                    ids, max_new_tokens=max_new_tokens, do_sample=do_sample,
                    temperature=temperature, pad_token_id=self.pad_id)
            return self.tok.decode(gen[0], skip_special_tokens=True).replace(" ", "")
        except Exception as e:
            return (f"[generation not supported by this HyenaDNA-hf checkpoint "
                    f"({type(e).__name__}); use it for SCORING. Original prompt: {prompt}]")


# --------------------------------------------------------------------------- #
# Evo 2 — optional upgrade tier (needs a >32 GB GPU + the evo2 GPU stack)
# --------------------------------------------------------------------------- #
class Evo2Scorer:
    """Wraps the `evo2` package. `score_sequences` returns the model's likelihoods,
    matching the reference notebook (delta = var - ref; AUROC vs is_lof uses -delta).
    """

    name = "evo2"

    def __init__(self, model_key: str = "evo2_7b", device: str = "auto"):
        try:
            from evo2 import Evo2  # noqa
        except Exception as e:  # pragma: no cover - only on GPU nodes with evo2 built
            raise RuntimeError(
                "evo2 is not importable. Build it inside a GPU job "
                "(scripts/build_evo2_gpu.sh), or use the precomputed score table "
                "fallback. Original error: %s" % e
            ) from e
        from evo2 import Evo2
        short = model_key if str(model_key).startswith("evo2") else "evo2_7b"
        self.name = short
        self.model = Evo2(short)

    def score_sequences(self, seqs: Sequence[str], batch_size: int = 1,
                        progress: bool = False) -> List[float]:
        # evo2 handles its own batching internally; chunk to bound memory.
        out: List[float] = []
        seqs = list(seqs)
        for i in range(0, len(seqs), max(1, batch_size)):
            out.extend(float(x) for x in self.model.score_sequences(seqs[i:i + batch_size]))
        return out

    def sequence_logprob(self, seq: str) -> float:
        return self.score_sequences([seq])[0]

    def generate(self, prompt: str, max_new_tokens: int = 64, **kw) -> str:
        try:
            res = self.model.generate(prompt_seqs=[prompt], n_tokens=max_new_tokens, **kw)
            # evo2 returns an object with .sequences in recent versions
            seqs = getattr(res, "sequences", None) or res
            return (seqs[0] if isinstance(seqs, (list, tuple)) else str(seqs))
        except Exception as e:  # pragma: no cover
            return f"[evo2 generate unavailable: {e}]"


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def load_dna_model(model_key: str = None, device: str = "auto", dtype: str = "float32"):
    """Return a scorer for the requested model. Defaults to the configured primary."""
    model_key = model_key or cfg.PRIMARY_GENOMIC_MODEL
    key = model_key.lower()
    if key.startswith("evo2") or "arcinstitute" in key:
        return Evo2Scorer(model_key, device=device)
    if key.startswith("evo1") or "evo-1" in key:
        raise ValueError(
            "Evo 1 is prokaryote-trained (OpenGenome) — wrong organism for human "
            "BRCA1. Use it only for the prokaryotic sequence-modeling extras."
        )
    # default: HyenaDNA family (CPU-capable)
    return HyenaDNAScorer(resolve_repo(model_key), device=device, dtype=dtype)
