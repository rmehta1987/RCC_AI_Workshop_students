#!/usr/bin/env bash
# =============================================================================
# setup_user_secrets.sh — write PER-USER secrets to ~/.config/aimed/.env (chmod 600).
# Secrets NEVER live in PROJECT_DIR, notebooks, or git. The course runs fully
# without any of these (mock LLM + local HyenaDNA); they only enable upgrades.
#
#   bash scripts/setup_user_secrets.sh                 # interactive prompts
#   OPENAI_BASE_URL=http://gpu0123:8000/v1 \
#   AIMED_LLM_MODEL=Qwen2.5-7B-Instruct \
#     bash scripts/setup_user_secrets.sh --noninteractive
# =============================================================================
set -uo pipefail
CFG_DIR="$HOME/.config/aimed"
ENV_FILE="$CFG_DIR/.env"
mkdir -p "$CFG_DIR"; chmod 700 "$CFG_DIR"
NONINT=0; for a in "$@"; do [ "$a" = "--noninteractive" ] && NONINT=1; done

# keep existing values unless overridden
[ -f "$ENV_FILE" ] && . "$ENV_FILE" 2>/dev/null || true

ask() {  # ask VAR "prompt" -> echoes current-or-new value
  local var="$1" prompt="$2" cur="${!var:-}"
  if [ "$NONINT" = 1 ]; then echo "${!var:-}"; return; fi
  local shown="$cur"; [ -n "$cur" ] && shown="(keep current)"
  read -r -p "$prompt ${shown:+[$shown]}: " val
  echo "${val:-$cur}"
}

echo "== AImed per-user secrets =="
echo "Leave blank to skip / keep. The course works with ALL of these empty."
OPENAI_API_KEY="$(ask OPENAI_API_KEY 'OPENAI_API_KEY (MedRAX / agent LLM; "EMPTY" works for local vLLM)')"
OPENAI_BASE_URL="$(ask OPENAI_BASE_URL 'OPENAI_BASE_URL (local vLLM/Ollama, e.g. http://gpuNNNN:8000/v1)')"
AIMED_LLM_MODEL="$(ask AIMED_LLM_MODEL 'AIMED_LLM_MODEL (served model name, e.g. Qwen2.5-7B-Instruct)')"
HF_TOKEN="$(ask HF_TOKEN 'HF_TOKEN (only for gated HF weights; HyenaDNA is public)')"

umask 077
{
  echo "# AImed per-user secrets — written $(date). chmod 600. Do NOT commit."
  [ -n "$OPENAI_API_KEY" ]  && echo "OPENAI_API_KEY=$OPENAI_API_KEY"
  [ -n "$OPENAI_BASE_URL" ] && echo "OPENAI_BASE_URL=$OPENAI_BASE_URL"
  [ -n "$AIMED_LLM_MODEL" ] && echo "AIMED_LLM_MODEL=$AIMED_LLM_MODEL"
  [ -n "$HF_TOKEN" ]        && echo "HF_TOKEN=$HF_TOKEN"
} > "$ENV_FILE"
chmod 600 "$ENV_FILE"
echo "Wrote $ENV_FILE (chmod 600):"
sed 's/=.*/=********/' "$ENV_FILE" | grep -v '^#' || echo "  (no secrets set — fully fine)"
echo "Scripts pick these up automatically via scripts/common/load_profile.sh."
