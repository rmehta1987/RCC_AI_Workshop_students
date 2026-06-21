#!/usr/bin/env bash
# =============================================================================
# load_profile.sh — single source of truth loader (shell side).
#   source scripts/common/load_profile.sh
# Exports the AIMED_CONFIG block from CLUSTER_PROFILE.md + the shared cache env
# vars, and defines aimed_activate() to enter the shared 'aimed' env.
# =============================================================================
_lp_self="${BASH_SOURCE[0]:-$0}"
_lp_dir="$(cd "$(dirname "$_lp_self")" && pwd)"
: "${PROJECT_DIR:=$(cd "$_lp_dir/../.." && pwd)}"
PROFILE="${AIMED_PROFILE:-$PROJECT_DIR/CLUSTER_PROFILE.md}"

if [ -f "$PROFILE" ]; then
  while IFS= read -r _line; do
    case "$_line" in
      [A-Z]*=*) export "${_line%%#*}" 2>/dev/null || true ;;
    esac
  done < <(sed -n '/# >>> AIMED_CONFIG >>>/,/# <<< AIMED_CONFIG <<</p' "$PROFILE" \
           | grep -E '^[A-Z_][A-Z0-9_]*=')
fi

# Derived / defensive defaults (profile is authoritative; these fill gaps).
export PROJECT_DIR
export ENV_PREFIX="${ENV_PREFIX:-$PROJECT_DIR/env/aimed}"
export MINIFORGE_MODULE="${MINIFORGE_MODULE:-python/miniforge-25.3.0}"
export DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/caches/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
# (TRANSFORMERS_CACHE intentionally unset — deprecated; HF_HOME covers it)
export MEDMNIST_ROOT="${MEDMNIST_ROOT:-$PROJECT_DIR/caches/medmnist}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_DIR/caches/torch}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export RCC_ACCOUNT="${RCC_ACCOUNT:-rcc-staff}"
export GPU_PARTITION="${GPU_PARTITION:-test}"
export CPU_PARTITION="${CPU_PARTITION:-caslake}"

# `module purge` unsets $SOFTPATH, after which Midway3 modulefiles deref an empty
# var and every load fails; a fresh sbatch job may also not inherit it. Restore it
# defensively so `module load python/miniforge-... | cuda/...` works.
export SOFTPATH="${SOFTPATH:-/software}"

# Per-user secrets (never in PROJECT_DIR): OPENAI_API_KEY / OPENAI_BASE_URL / HF_TOKEN
if [ -f "$HOME/.config/aimed/.env" ]; then
  set -a; . "$HOME/.config/aimed/.env"; set +a
fi

# Enter the shared 'aimed' env. Two cases, auto-detected:
#  (1) a RELOCATED (conda-pack'd) env ships a self-contained `bin/activate` — prefer
#      it: needs NO module/conda at all (this is the student deployment env); or
#  (2) a normal prefix env (the dev tree) has none, so fall through to the CANONICAL
#      Midway3 path (cf. ../L2LGWAS_DFE/scripts/_hpc_env.sh): miniforge module ->
#      mamba shell hook -> mamba activate. The `eval "$(mamba shell hook ...)"` is
#      REQUIRED — a bare `mamba/conda activate` can silently no-op in a non-interactive
#      (sbatch) shell, leaving CONDA_PREFIX unset and the job on system python.
# Never `module purge` (it unsets SOFTPATH and breaks subsequent loads).
aimed_activate() {
  export SOFTPATH="${SOFTPATH:-/software}"
  if [ -f "$ENV_PREFIX/bin/activate" ]; then
    # conda-pack's activate isn't `set -u`-clean (refs unbound CONDA_PREFIX on first
    # entry); our sbatch jobs run under `set -u`, so relax nounset just for the source.
    case $- in *u*) _aimed_u=1 ;; *) _aimed_u=0 ;; esac
    set +u
    # shellcheck disable=SC1091
    . "$ENV_PREFIX/bin/activate"
    [ "$_aimed_u" = 1 ] && set -u
    unset _aimed_u
  else
    # `module` (Environment Modules, not Lmod) may be undefined in a fresh sbatch shell.
    [ -z "$(type -t module)" ] && source /software/modules/init/bash 2>/dev/null || true
    module load "${MINIFORGE_MODULE:-python/miniforge-25.3.0}" 2>/dev/null || true
    eval "$(mamba shell hook --shell bash)"
    mamba activate "$ENV_PREFIX"
  fi
  case "${CONDA_PREFIX:-}" in
    "$ENV_PREFIX") : ;;
    *) echo "[aimed_activate] FATAL: '$ENV_PREFIX' not active (CONDA_PREFIX='${CONDA_PREFIX:-unset}') — activation failed" >&2
       return 1 ;;
  esac
}
