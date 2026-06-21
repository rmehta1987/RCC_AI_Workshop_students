"""Shared utilities for the AImed two-day course scripts.

Submodules
----------
aimed_config : single source of truth (reads CLUSTER_PROFILE.md), cache setup
metrics      : honest-evaluation metrics (AUROC/AUPRC/calibration/subgroup/threshold)
genomics     : chr17 FASTA loading, SNV window construction, variant scoring driver
models       : model-agnostic DNA scorers (HyenaDNA via transformers; Evo 2 via evo2)
llm_backends : mock + OpenAI-compatible (vLLM/Ollama) LLM backends for the CXR agent
plotting     : calibration / delta-distribution / Grad-CAM / fusion plots (Agg)
"""
__all__ = [
    "aimed_config",
    "metrics",
    "genomics",
    "models",
    "llm_backends",
    "plotting",
]
