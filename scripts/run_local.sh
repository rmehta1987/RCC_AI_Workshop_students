#!/usr/bin/env bash
# run_local.sh — start/stop a local JupyterLab in the BACKGROUND for the CPU-only laptop path.
#
#   bash scripts/run_local.sh          # start: opens your browser AND gives your terminal back
#   bash scripts/run_local.sh stop     # stop the background server
#   bash scripts/run_local.sh status   # is it running? print the URL
#
# It launches from the repo root with AIMED_PROFILE=/dev/null so the notebooks resolve the
# local ./data and ./caches instead of the cluster's /project paths, and uses the venv's own
# jupyter. The server runs detached — no console window to keep open or quit.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JUP="$ROOT/.venv/bin/jupyter"
PORT="${PORT:-8888}"
LOG="$ROOT/logs/jupyter-local.log"

if [ ! -x "$JUP" ]; then
  echo "ERROR: $JUP not found — create the CPU env first:" >&2
  echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements-cpu.txt" >&2
  exit 1
fi

open_browser() {  # best-effort; the URL is always printed too
  local url="$1"
  if grep -qiE "microsoft|wsl" /proc/version 2>/dev/null; then
    command -v wslview     >/dev/null 2>&1 && { wslview     "$url" >/dev/null 2>&1 & return; }
    command -v explorer.exe >/dev/null 2>&1 && { explorer.exe "$url" >/dev/null 2>&1 & return; }
  fi
  for opener in xdg-open open; do
    command -v "$opener" >/dev/null 2>&1 && { "$opener" "$url" >/dev/null 2>&1 & return; }
  done
}

running_url() { "$JUP" server list 2>/dev/null | grep ":$PORT/" | awk '{print $1}' | head -1; }

case "${1:-start}" in
  stop)
    if "$JUP" lab stop "$PORT" 2>/dev/null; then
      echo "Stopped the Jupyter server on port $PORT."
    else
      echo "No Jupyter server was running on port $PORT."
    fi
    ;;
  status)
    url="$(running_url || true)"
    [ -n "$url" ] && echo "Running: $url" || echo "Not running on port $PORT."
    ;;
  start|"")
    if [ -n "$(running_url || true)" ]; then
      echo "JupyterLab is already running at: $(running_url)"
      exit 0
    fi
    mkdir -p "$ROOT/logs"
    cd "$ROOT"
    AIMED_PROFILE=/dev/null nohup "$JUP" lab --port "$PORT" --no-browser >"$LOG" 2>&1 &
    for _ in $(seq 1 30); do
      [ -n "$(running_url || true)" ] && break
      sleep 1
    done
    url="$(running_url || true)"
    if [ -z "$url" ]; then
      echo "Jupyter didn't come up in time — see the log: $LOG" >&2
      exit 1
    fi
    open_browser "$url"
    cat <<MSG

  JupyterLab is running in the background — your browser should open at:

    $url

  Your terminal is free to use. When you're done, shut it down with:

    bash scripts/run_local.sh stop

MSG
    ;;
  *)
    echo "usage: bash scripts/run_local.sh [start|stop|status]" >&2
    exit 2
    ;;
esac
