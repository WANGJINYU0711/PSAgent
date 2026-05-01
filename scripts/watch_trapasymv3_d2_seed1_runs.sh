#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

POLL_SECONDS="${PSAGENT_TRAPASYM_WATCH_POLL_SECONDS:-120}"
LOG_DIR="${ROOT_DIR}/tmp/watchdog_logs"
LOG_FILE="${LOG_DIR}/trapasymv3_d2_seed1_watchdog.log"
LOCK_DIR="${LOG_DIR}/trapasymv3_d2_seed1_watchdog.lock"

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

SESSION_NAMES=(
  "psagent_trapasymv3_d2_activegapv2_seed1_4m"
  "psagent_trapasymv3_d2_activegapv2_nomodecost_seed1_4m"
)

LAUNCH_COMMANDS=(
  "bash scripts/launch_trapasymv3_d2_activegap_v2_llm_run.sh 1"
  "bash scripts/launch_trapasymv3_d2_activegap_v2_nomodecost_llm_run.sh 1"
)

RUN_LABELS=(
  "d2_activegapv2_seed1"
  "d2_activegapv2_nomodecost_seed1"
)

session_healthy() {
  local session_name="$1"
  local window_count pane_count dead_panes

  if ! tmux has-session -t "$session_name" 2>/dev/null; then
    return 1
  fi

  window_count="$(tmux list-windows -t "$session_name" 2>/dev/null | wc -l | tr -d ' ')"
  pane_count="$(tmux list-panes -t "$session_name" 2>/dev/null | wc -l | tr -d ' ')"
  dead_panes="$(tmux list-panes -t "$session_name" -F '#{pane_dead}' 2>/dev/null | rg -c '^1$' || true)"

  if [[ "$window_count" != "4" ]]; then
    return 1
  fi
  if [[ "$pane_count" != "4" ]]; then
    return 1
  fi
  if [[ "${dead_panes:-0}" != "0" ]]; then
    return 1
  fi

  return 0
}

relaunch_run() {
  local label="$1"
  local launch_cmd="$2"

  log "relaunching ${label} with: ${launch_cmd}"
  eval "$launch_cmd" >>"$LOG_FILE" 2>&1
  log "relaunch finished for ${label}"
}

log "watchdog started; poll_seconds=${POLL_SECONDS}"

while true; do
  for idx in "${!SESSION_NAMES[@]}"; do
    session_name="${SESSION_NAMES[$idx]}"
    launch_cmd="${LAUNCH_COMMANDS[$idx]}"
    label="${RUN_LABELS[$idx]}"

    if session_healthy "$session_name"; then
      log "healthy ${label} session=${session_name}"
      continue
    fi

    log "unhealthy ${label} session=${session_name}; attempting restart from checkpoint"
    relaunch_run "$label" "$launch_cmd"
  done

  sleep "$POLL_SECONDS"
done
