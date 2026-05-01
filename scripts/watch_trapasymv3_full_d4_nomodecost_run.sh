#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEED="${1:-1}"
POLL_SECONDS="${PSAGENT_TRAPASYM_WATCH_POLL_SECONDS:-120}"
LOG_DIR="${ROOT_DIR}/tmp/watchdog_logs"
LOG_FILE="${LOG_DIR}/trapasymv3_full_d4_nomodecost_seed${SEED}_watchdog.log"
LOCK_DIR="${LOG_DIR}/trapasymv3_full_d4_nomodecost_seed${SEED}_watchdog.lock"
SESSION_NAME="psagent_trapasymv3_full_d4_nomodecost_seed${SEED}_11m"
LAUNCH_COMMAND="bash scripts/launch_trapasymv3_full_d4_nomodecost_llm_run.sh ${SEED}"

mkdir -p "$LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_FILE"
}

cleanup_lock() {
  rm -rf "$LOCK_DIR"
}

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "watchdog lock exists at $LOCK_DIR; another watchdog may already be running" >&2
  exit 1
fi
trap cleanup_lock EXIT

session_healthy() {
  local window_count pane_count dead_panes window_index window_dead_panes

  if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    return 1
  fi

  window_count="$(tmux list-windows -t "$SESSION_NAME" 2>/dev/null | wc -l | tr -d ' ')"
  pane_count=0
  dead_panes=0
  while IFS= read -r window_index; do
    pane_count=$((pane_count + $(tmux list-panes -t "${SESSION_NAME}:${window_index}" 2>/dev/null | wc -l | tr -d ' ')))
    window_dead_panes="$(tmux list-panes -t "${SESSION_NAME}:${window_index}" -F '#{pane_dead}' 2>/dev/null | rg -c '^1$' || true)"
    dead_panes=$((dead_panes + window_dead_panes))
  done < <(tmux list-windows -t "$SESSION_NAME" -F '#{window_index}' 2>/dev/null)

  if [[ "$window_count" != "11" ]]; then
    return 1
  fi
  if [[ "$pane_count" != "11" ]]; then
    return 1
  fi
  if [[ "${dead_panes:-0}" != "0" ]]; then
    return 1
  fi

  return 0
}

log "watchdog started; seed=${SEED} poll_seconds=${POLL_SECONDS}"

while true; do
  if session_healthy; then
    log "healthy full_d4_nomodecost session=${SESSION_NAME}"
  else
    log "unhealthy full_d4_nomodecost session=${SESSION_NAME}; attempting restart from checkpoint"
    eval "$LAUNCH_COMMAND" >>"$LOG_FILE" 2>&1
    log "relaunch finished for full_d4_nomodecost"
  fi

  sleep "$POLL_SECONDS"
done
