"""Central configuration for the AImed course scripts.

Single source of truth = ``CLUSTER_PROFILE.md`` (the ``AIMED_CONFIG`` block).
Import this FIRST, call :func:`setup_caches` BEFORE importing torch/transformers,
so every process points at the shared HF / MedMNIST / torch caches and never
tries to reach the internet from a compute node.

    from common import aimed_config as cfg
    cfg.setup_caches()
    import torch, transformers   # now cache-aware
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# --------------------------------------------------------------------------- #
# Locate PROJECT_DIR and the profile
# --------------------------------------------------------------------------- #
def _find_project_dir() -> Path:
    if os.environ.get("PROJECT_DIR"):
        return Path(os.environ["PROJECT_DIR"])
    # this file lives at PROJECT_DIR/scripts/common/aimed_config.py
    return Path(__file__).resolve().parents[2]


PROJECT_DIR = _find_project_dir()
PROFILE_PATH = Path(os.environ.get("AIMED_PROFILE", PROJECT_DIR / "CLUSTER_PROFILE.md"))


def _parse_profile(path: Path) -> dict:
    cfg: dict[str, str] = {}
    if not path.exists():
        return cfg
    inside = False
    for line in path.read_text().splitlines():
        if ">>> AIMED_CONFIG >>>" in line:
            inside = True
            continue
        if "<<< AIMED_CONFIG <<<" in line:
            inside = False
            continue
        if inside:
            m = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line.strip())
            if m:
                cfg[m.group(1)] = m.group(2).split("#", 1)[0].strip()
    return cfg


_CONFIG = _parse_profile(PROFILE_PATH)


def get(key: str, default=None):
    """Env var wins, then CLUSTER_PROFILE.md, then the supplied default."""
    return os.environ.get(key, _CONFIG.get(key, default))


# --------------------------------------------------------------------------- #
# Resolved paths / values
# --------------------------------------------------------------------------- #
PROJECT_DIR = Path(get("PROJECT_DIR", str(PROJECT_DIR)))
DATA_DIR = Path(get("DATA_DIR", str(PROJECT_DIR / "data")))
ENV_PREFIX = Path(get("ENV_PREFIX", str(PROJECT_DIR / "env" / "aimed")))
HF_HOME = get("HF_HOME", str(PROJECT_DIR / "caches" / "hf"))
HF_HUB_CACHE = get("HF_HUB_CACHE", str(Path(HF_HOME) / "hub"))
MEDMNIST_ROOT = get("MEDMNIST_ROOT", str(PROJECT_DIR / "caches" / "medmnist"))
TORCH_HOME = get("TORCH_HOME", str(PROJECT_DIR / "caches" / "torch"))

RCC_ACCOUNT = get("RCC_ACCOUNT", "rcc-staff")
GPU_PARTITION = get("GPU_PARTITION", "test")
CPU_PARTITION = get("CPU_PARTITION", "caslake")

PRIMARY_GENOMIC_MODEL = get("PRIMARY_GENOMIC_MODEL", "hyenadna-small-32k")
EVO2_TIER = get("EVO2_TIER", "disabled")
EVO2_MODEL = get("EVO2_MODEL", "evo2_7b")

# HF repo ids for the genomic models (resolved by --model name in the scripts)
HF_REPOS = {
    "hyenadna-tiny-1k": "LongSafari/hyenadna-tiny-1k-seqlen-hf",
    "hyenadna-small-32k": "LongSafari/hyenadna-small-32k-seqlen-hf",
    "evo2_7b": "arcinstitute/evo2_7b",
    "evo1": "togethercomputer/evo-1-8k-base",
}

# Key data artifacts
BRCA1_CSV = DATA_DIR / "day2_genomics" / "brca1_variants.csv"
CHR17_FASTA = DATA_DIR / "day2_genomics" / "GRCh37.p13_chr17.fna.gz"
DIABETES_CSV = DATA_DIR / "day1_ai_medicine" / "tabular_clinical_diabetes.csv"
SEQ_EXTRAS = DATA_DIR / "day2_genomics" / "seqmodeling_extras"
FIXTURES = DATA_DIR / "fixtures"
CXR_SMOKE = FIXTURES / "cxr_smoke.npz"
MULTIMODAL_SMOKE = FIXTURES / "multimodal_smoke.npz"
PRECOMPUTED_VEP_SYNTH = FIXTURES / "precomputed_vep_scores_SYNTHETIC.csv"
# The REAL precomputed tables written by the pre-warm job (preferred when present).
# HyenaDNA-small-32k is ~chance on BRCA1 (measured AUROC 0.46) — the honest "small
# models fail at VEP" baseline. The Evo 2 7B table is the headline result: measured
# AUROC 0.877 (generated on beagle3 via the NGC container; cols incl. evo2_delta_score).
PRECOMPUTED_VEP_REAL = DATA_DIR / "day2_genomics" / "precomputed_vep_scores.csv"        # HyenaDNA
PRECOMPUTED_VEP_EVO2 = DATA_DIR / "day2_genomics" / "precomputed_vep_scores_evo2.csv"   # Evo 2


def precomputed_vep_table() -> Path:
    """Prefer the real HyenaDNA precomputed table; fall back to the synthetic fixture."""
    return PRECOMPUTED_VEP_REAL if PRECOMPUTED_VEP_REAL.exists() else PRECOMPUTED_VEP_SYNTH


def evo2_precomputed_table():
    """The Evo 2 headline table, or None if it hasn't been generated yet."""
    return PRECOMPUTED_VEP_EVO2 if PRECOMPUTED_VEP_EVO2.exists() else None


# --------------------------------------------------------------------------- #
# Environment setup
# --------------------------------------------------------------------------- #
def load_secrets() -> None:
    """Load per-user secrets from ~/.config/aimed/.env (never from PROJECT_DIR)."""
    env_file = Path.home() / ".config" / "aimed" / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def setup_caches(offline: bool | None = None, verbose: bool = False) -> None:
    """Point HF / MedMNIST / torch at the shared caches. Call before importing torch.

    offline=None  -> auto: offline if the shared HF cache exists AND no internet flag.
    """
    os.environ.setdefault("HF_HOME", HF_HOME)
    os.environ.setdefault("HF_HUB_CACHE", HF_HUB_CACHE)
    # NB: do NOT set TRANSFORMERS_CACHE (deprecated in transformers v5; HF_HOME covers it)
    os.environ.setdefault("TORCH_HOME", TORCH_HOME)
    # Enable HF's fast Rust downloader only when it's actually installed. Forcing it
    # on unconditionally crashes a lean CPU env that lacks the hf_transfer wheel
    # (ValueError: ... 'hf_transfer' package is not available). It ships in
    # requirements-cpu.txt, so this is normally on.
    import importlib.util
    if importlib.util.find_spec("hf_transfer") is not None:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    load_secrets()
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if verbose:
        print(f"[aimed_config] PROJECT_DIR={PROJECT_DIR}")
        print(f"[aimed_config] HF_HOME={HF_HOME}")
        print(f"[aimed_config] MEDMNIST_ROOT={MEDMNIST_ROOT}  TORCH_HOME={TORCH_HOME}")
        print(f"[aimed_config] EVO2_TIER={EVO2_TIER}  primary={PRIMARY_GENOMIC_MODEL}")


def pick_device(prefer: str = "auto") -> str:
    """Return 'cuda' if a GPU is visible else 'cpu'. The whole course degrades to CPU."""
    if prefer in ("cpu", "cuda"):
        return prefer
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def summary() -> str:
    return (
        f"PROJECT_DIR={PROJECT_DIR}\nENV_PREFIX={ENV_PREFIX}\nDATA_DIR={DATA_DIR}\n"
        f"HF_HOME={HF_HOME}\nMEDMNIST_ROOT={MEDMNIST_ROOT}\nTORCH_HOME={TORCH_HOME}\n"
        f"GPU_PARTITION={GPU_PARTITION}  CPU_PARTITION={CPU_PARTITION}  ACCOUNT={RCC_ACCOUNT}\n"
        f"EVO2_TIER={EVO2_TIER}  PRIMARY={PRIMARY_GENOMIC_MODEL}\n"
    )


if __name__ == "__main__":
    setup_caches(verbose=True)
    print(summary())
