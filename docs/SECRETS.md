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

## Recommended: drive the agent with the local vLLM (no OpenAI key) — turnkey

One shared GPU server for the whole cohort, using your `marshal_env_torch260_vllm084.sif`
(vLLM 0.8.4) + the cached **Qwen3-4B** (fits any GPU, reliable JSON tool-calls):

```bash
# instructor: start the shared server (self-tests the agent, then keeps serving)
sbatch slurm/serve_vllm_agent.sbatch
cat logs/vllm-agent-<jobid>.log        # shows the endpoint + the self-test transcript

# anyone (same node, or with the printed HOST / a tunnel): point the agent at it
source logs/vllm_endpoint.env          # sets OPENAI_BASE_URL / AIMED_LLM_MODEL / OPENAI_API_KEY=EMPTY
python scripts/minimal_cxr_agent.py --backend openai     # now driven by Qwen3-4B
```

Swap the model with `AIMED_VLLM_MODEL=Qwen2.5-Coder-32B-Instruct sbatch ...` (needs a
≥40 GB card — use the **beagle3** partition). The agent's JSON extraction strips Qwen3
`<think>` blocks and falls back to the offline MockLLM if the endpoint is unreachable,
so a flaky server never breaks the gate.

> **Note (answers "can we do Module 3 with vLLM?")** vLLM serves the *agent's*
> LLM controller (Module 3b) beautifully. It **cannot** serve the *genomic* models
> (Module 3 / Day 2): vLLM supports Transformer + Mamba/SSM architectures, but
> **HyenaDNA and Evo 2 are Hyena/StripedHyena models that vLLM does not implement**,
> and the VEP task needs prompt log-likelihoods, not generation. For Day-2 scale,
> use the shared HF cache + the precomputed score table instead.

## Gated weights

HyenaDNA (`LongSafari/*`) and Evo 2 7B (`arcinstitute/evo2_7b`) are **public** — no
acceptance step. If you later add a gated model, accept its license once with the
instructor `HF_TOKEN` and pre-stage it into the shared cache so students don't each
need a token.
