# SECRETS.md — keys, gated models, and external endpoints

**The course runs fully degraded-but-functional with NO secrets.** Keys only
unlock upgrades. Nothing here is required to clear any mastery gate.

## Where secrets live (and don't)

- **Per-user file:** `~/.config/aimed/.env`, `chmod 600`. Written by
  `bash scripts/setup_user_secrets.sh`. Auto-loaded by every script via
  `scripts/common/load_profile.sh`.
- **Never** in `PROJECT_DIR`, never in a notebook cell, never in git
  (`.gitignore` blocks `.env`, `*token*`, `*.key`).

## The three optional secrets

| Variable | Used by | If absent → fallback |
|---|---|---|
| `OPENAI_API_KEY` | Day-1 Module 3b agent / MedRAX LLM | **MockLLM** (offline, deterministic) |
| `OPENAI_BASE_URL` | point the agent at a local LLM | hosted OpenAI, else MockLLM |
| `AIMED_LLM_MODEL` | served model name | `gpt-4o-mini` default |
| `HF_TOKEN` | gated HF weights | none needed — HyenaDNA/Evo 2 are public |

The **genomic path needs no key** — HyenaDNA and Evo 2 run locally from the shared
HF cache.

## Driving the agent with a real LLM (optional, no GPU)

The Day-1 agent (Module 3b) runs offline on the **MockLLM** by default. To use a real LLM instead, point
it at any OpenAI-compatible endpoint — a hosted API or a local Ollama — by setting `OPENAI_BASE_URL` +
`OPENAI_API_KEY`, then `python scripts/minimal_cxr_agent.py --backend openai`. The agent falls back to the
MockLLM if the endpoint is unreachable, so this never blocks the module. *(No GPU is involved — the
genomic models don't run through an LLM server; Day-2 uses the precomputed score tables.)*

## Gated weights

HyenaDNA (`LongSafari/*`) and Evo 2 7B (`arcinstitute/evo2_7b`) are **public** — no
acceptance step. If you later add a gated model, accept its license once with the
instructor `HF_TOKEN` and pre-stage it into the shared cache so students don't each
need a token.
