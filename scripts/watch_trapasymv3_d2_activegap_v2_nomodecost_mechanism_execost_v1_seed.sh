#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEED="${1:-1}"
POLL_SECONDS="${PSAGENT_TRAPASYM_WATCH_POLL_SECONDS:-120}"
RUNNER_PROGRESS_STALE_SECONDS="${PSAGENT_TRAPASYM_RUNNER_PROGRESS_STALE_SECONDS:-1800}"
WANDB_STATE_STALE_SECONDS="${PSAGENT_TRAPASYM_WANDB_STATE_STALE_SECONDS:-900}"
LOG_DIR="${ROOT_DIR}/tmp/watchdog_logs"
LOG_FILE="${LOG_DIR}/trapasymv3_d2_activegapv2_nomodecost_mechanisms_execost_v1_seed${SEED}_watchdog.log"
LOCK_DIR="${LOG_DIR}/trapasymv3_d2_activegapv2_nomodecost_mechanisms_execost_v1_seed${SEED}_watchdog.lock"

EXPERIMENT_NAME="llm_v8_trapasymv3efficientanchor4of5_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_d2_activegapv2_nomodecost_reason135_eps001_10x10_mechanisms_execost_v1_seed${SEED}"
RUN_DIR="tmp/${EXPERIMENT_NAME}"
MECHANISM_SESSION="psagent_trapasymv3_d2_activegapv2_nomodecost_seed${SEED}_mech_execost_v1"
LIVE_SESSION="psagent_trapasymv3_d2_activegapv2_nomodecost_seed${SEED}_mech_execost_v1_wandb"
MERGE_SESSION="psagent_trapasymv3_d2_activegapv2_nomodecost_seed${SEED}_mech_execost_v1_merge"
LAUNCH_COMMAND="bash scripts/launch_trapasymv3_d2_activegap_v2_nomodecost_mechanism_llm_run_execost_v1.sh ${SEED}"
ENV_FILE="scripts/trapasymv3_d2_activegapv2_nomodecost_seed${SEED}_execost_v1_env.sh"
WANDB_GROUP="trapasymv3_d2_activegapv2_nomodecost_mechanisms_execost_v1_seed${SEED}"
MECHANISMS=(
  theta_guided_agent
  agent_only
)

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

run_complete() {
  [[ -f "${RUN_DIR}/theta_guided_agent/summary.json" && -f "${RUN_DIR}/agent_only/summary.json" ]]
}

mechanism_session_healthy() {
  if ! tmux has-session -t "$MECHANISM_SESSION" 2>/dev/null; then
    return 1
  fi
  return 0
}

support_sessions_healthy() {
  tmux has-session -t "$MERGE_SESSION" 2>/dev/null || return 1
  return 0
}

mechanism_process_alive() {
  local mechanism="$1"
  ps -eo args \
    | rg -F -- "python scripts/run_shared_basin_mechanism_repeated_smoke.py run-mechanism" \
    | rg -F -- "--run-dir ${RUN_DIR}" \
    | rg -F -- "--mechanism ${mechanism}" >/dev/null
}

progress_complete() {
  local mechanism="$1"
  python - "$RUN_DIR" "$mechanism" <<'PY'
import json
import sys
from pathlib import Path

progress_path = Path(sys.argv[1]) / sys.argv[2] / "progress.json"
if not progress_path.exists():
    raise SystemExit(1)
progress = json.loads(progress_path.read_text())
completed = int(progress.get("completed_episodes", 0) or 0)
scheduled = int(progress.get("scheduled_episodes", 0) or 0)
status = str(progress.get("status", ""))
raise SystemExit(0 if scheduled and completed >= scheduled and status in {"complete", "completed"} else 1)
PY
}

progress_summary() {
  local mechanism="$1"
  python - "$RUN_DIR" "$mechanism" <<'PY'
import json
import sys
from pathlib import Path

progress_path = Path(sys.argv[1]) / sys.argv[2] / "progress.json"
if not progress_path.exists():
    print("no progress.json")
    raise SystemExit(0)
progress = json.loads(progress_path.read_text())
print(
    f"{progress.get('completed_episodes', 0)}/"
    f"{progress.get('scheduled_episodes', '?')} status={progress.get('status', '?')} "
    f"updated_at={progress.get('updated_at', '?')}"
)
PY
}

runner_progress_stale() {
  local mechanism="$1"
  python - "$RUN_DIR" "$mechanism" "$RUNNER_PROGRESS_STALE_SECONDS" <<'PY'
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

progress_path = Path(sys.argv[1]) / sys.argv[2] / "progress.json"
stale_seconds = float(sys.argv[3])
if not progress_path.exists():
    raise SystemExit(1)
progress = json.loads(progress_path.read_text())
completed = int(progress.get("completed_episodes", 0) or 0)
scheduled = int(progress.get("scheduled_episodes", 0) or 0)
status = str(progress.get("status", ""))
if status in {"complete", "completed"} or (scheduled and completed >= scheduled):
    raise SystemExit(1)
updated_at = progress.get("updated_at")
try:
    dt = datetime.fromisoformat(str(updated_at).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    age = time.time() - dt.timestamp()
except Exception:
    age = time.time() - progress_path.stat().st_mtime
raise SystemExit(0 if age >= stale_seconds else 1)
PY
}

wandb_process_alive() {
  ps -eo args \
    | rg -F -- "python scripts/live_wandb_partial_uploader.py" \
    | rg -F -- "--run-dir ${RUN_DIR}" >/dev/null
}

wandb_state_stale() {
  python - "$RUN_DIR" "$WANDB_STATE_STALE_SECONDS" "${MECHANISMS[@]}" <<'PY'
import json
import sys
import time
from pathlib import Path

run_dir = Path(sys.argv[1])
stale_seconds = float(sys.argv[2])
mechanisms = sys.argv[3:]
state_files = sorted(run_dir.glob("live_wandb_uploader_state*.json"))
if not state_files:
    raise SystemExit(1)
state_path = max(state_files, key=lambda path: path.stat().st_mtime)
try:
    state = json.loads(state_path.read_text())
except Exception:
    raise SystemExit(0)
for mechanism in mechanisms:
    progress_path = run_dir / mechanism / "progress.json"
    if not progress_path.exists():
        continue
    progress = json.loads(progress_path.read_text())
    completed = int(progress.get("completed_episodes", 0) or 0)
    if completed <= 0 or str(progress.get("status", "")) == "completed":
        continue
    last_completed = int(progress.get("last_completed_episode_index", completed - 1) or -1)
    last_uploaded = int(state.get(mechanism, {}).get("last_uploaded_episode_index", -1))
    if last_uploaded < last_completed:
        age = time.time() - state_path.stat().st_mtime
        raise SystemExit(0 if age >= stale_seconds else 1)
raise SystemExit(1)
PY
}

restart_mechanism() {
  local mechanism="$1"
  local window_name="$mechanism"
  if [[ "$mechanism" == "theta_guided_agent" ]]; then
    window_name="theta"
  fi
  tmux kill-window -t "${MECHANISM_SESSION}:${window_name}" 2>/dev/null || true
  mkdir -p "${RUN_DIR}/${mechanism}"
  log "restarting mechanism ${mechanism} from checkpoint"
  if tmux has-session -t "$MECHANISM_SESSION" 2>/dev/null; then
    tmux new-window -t "$MECHANISM_SESSION" -n "$window_name" \
      "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/run_shared_basin_mechanism_repeated_smoke.py run-mechanism --run-dir '${RUN_DIR}' --mechanism '${mechanism}' 2>&1 | tee -a '${RUN_DIR}/${mechanism}/tmux_runner.log'"
  else
    tmux new-session -d -s "$MECHANISM_SESSION" -n "$window_name" \
      "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/run_shared_basin_mechanism_repeated_smoke.py run-mechanism --run-dir '${RUN_DIR}' --mechanism '${mechanism}' 2>&1 | tee -a '${RUN_DIR}/${mechanism}/tmux_runner.log'"
  fi
}

restart_wandb() {
  tmux kill-session -t "$LIVE_SESSION" 2>/dev/null || true
  log "restarting wandb uploader for mechanisms"
  tmux new-session -d -s "$LIVE_SESSION" -n wandb \
    "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/live_wandb_partial_uploader.py --run-dir '${RUN_DIR}' --project psagent-llm-smoke --entity wangjinyu0711-microsoft --run-group '${WANDB_GROUP}' --run-name-prefix '${EXPERIMENT_NAME}_' --run-id-suffix analysis_d2_activegapv2_nomodecost_mechanisms_execost_v1 --methods ${MECHANISMS[*]} --poll-seconds 20 --finish-when-complete 2>&1 | tee -a '${RUN_DIR}/live_wandb_uploader_analysis_d2_activegapv2_nomodecost_mechanisms_execost_v1.log'"
}

relaunch_run() {
  log "relaunching mechanisms from checkpoint with: ${LAUNCH_COMMAND}"
  eval "$LAUNCH_COMMAND" >>"$LOG_FILE" 2>&1
  log "relaunch finished for mechanisms"
}

log "mechanism execost_v1 watchdog started; seed=${SEED} poll_seconds=${POLL_SECONDS}"
log "stale thresholds: runner_progress=${RUNNER_PROGRESS_STALE_SECONDS}s wandb_state=${WANDB_STATE_STALE_SECONDS}s"

while true; do
  if run_complete; then
    log "run complete; summaries found for theta_guided_agent and agent_only; exiting watchdog"
    exit 0
  fi

  if mechanism_session_healthy && support_sessions_healthy; then
    log "healthy mechanisms session=${MECHANISM_SESSION} live=${LIVE_SESSION} merge=${MERGE_SESSION}"
  else
    log "unhealthy mechanisms; attempting restart from checkpoint"
    relaunch_run
  fi

  for mechanism in "${MECHANISMS[@]}"; do
    summary="$(progress_summary "$mechanism")"
    if progress_complete "$mechanism"; then
      log "complete mechanism ${mechanism}: ${summary}"
    elif mechanism_process_alive "$mechanism" && ! runner_progress_stale "$mechanism"; then
      log "healthy mechanism ${mechanism}: ${summary}"
    else
      log "unhealthy mechanism ${mechanism}: ${summary}"
      restart_mechanism "$mechanism"
    fi
  done

  if wandb_process_alive && ! wandb_state_stale; then
    log "healthy wandb mechanisms"
  else
    log "unhealthy wandb mechanisms"
    restart_wandb
  fi

  sleep "$POLL_SECONDS"
done
