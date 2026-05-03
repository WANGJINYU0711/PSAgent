#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEED="${1:-1}"
POLL_SECONDS="${PSAGENT_TRAPASYM_SHAREVAR_WATCH_POLL_SECONDS:-120}"
RUNNER_PROGRESS_STALE_SECONDS="${PSAGENT_TRAPASYM_RUNNER_PROGRESS_STALE_SECONDS:-1800}"
WANDB_STATE_STALE_SECONDS="${PSAGENT_TRAPASYM_WANDB_STATE_STALE_SECONDS:-900}"
LOG_DIR="${ROOT_DIR}/tmp/watchdog_logs"
LOG_FILE="${LOG_DIR}/trapasymv3_sharevariants_riskyps_seed${SEED}_watchdog.log"
LOCK_DIR="${LOG_DIR}/trapasymv3_sharevariants_riskyps_seed${SEED}_watchdog.lock"
ENV_FILE="scripts/trapasymv3_sharevariants_d2_activegapv2_nomodecost_riskyps_seed${SEED}_env.sh"
METHOD="risky_ps"
WANDB_PROJECT="psagent-llm-smoke"
WANDB_ENTITY="wangjinyu0711-microsoft"
WANDB_GROUP="trapasymv3_sharevariants_d2_activegapv2_nomodecost_riskyps_seed${SEED}"

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

LABELS=(
  "allshare"
  "2of5share"
  "allunshare"
)
KEYS=(
  "allshare"
  "2of5"
  "allunshare"
)
RUN_DIRS=(
  "tmp/llm_v8_trapasymv3efficientanchorallshare_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_d2_activegapv2_nomodecost_reason135_eps001_10x10_riskyps_seed${SEED}"
  "tmp/llm_v8_trapasymv3efficientanchor2of5share_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_d2_activegapv2_nomodecost_reason135_eps001_10x10_riskyps_seed${SEED}"
  "tmp/llm_v8_trapasymv3efficientanchorallunshare_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_d2_activegapv2_nomodecost_reason135_eps001_10x10_riskyps_seed${SEED}"
)
RUNNER_SESSIONS=(
  "psagent_trapasymv3_allshare_d2_activegapv2_nomodecost_seed${SEED}_riskyps"
  "psagent_trapasymv3_2of5_d2_activegapv2_nomodecost_seed${SEED}_riskyps"
  "psagent_trapasymv3_allunshare_d2_activegapv2_nomodecost_seed${SEED}_riskyps"
)
WANDB_SESSIONS=(
  "psagent_trapasymv3_allshare_d2_activegapv2_nomodecost_seed${SEED}_live_wandb"
  "psagent_trapasymv3_2of5_d2_activegapv2_nomodecost_seed${SEED}_live_wandb"
  "psagent_trapasymv3_allunshare_d2_activegapv2_nomodecost_seed${SEED}_live_wandb"
)

progress_complete() {
  local run_dir="$1"
  python - "$run_dir" <<'PY'
import json
import sys
from pathlib import Path

progress_path = Path(sys.argv[1]) / "risky_ps" / "progress.json"
if not progress_path.exists():
    raise SystemExit(1)
progress = json.loads(progress_path.read_text())
completed = int(progress.get("completed_episodes", 0))
scheduled = int(progress.get("scheduled_episodes", 0))
status = str(progress.get("status", ""))
raise SystemExit(0 if scheduled and completed >= scheduled and status in {"complete", "completed"} else 1)
PY
}

progress_summary() {
  local run_dir="$1"
  python - "$run_dir" <<'PY'
import json
import sys
from pathlib import Path

progress_path = Path(sys.argv[1]) / "risky_ps" / "progress.json"
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

runner_process_alive() {
  local run_dir="$1"
  ps -eo args \
    | rg -F -- "python scripts/run_shared_basin_repeated_smoke.py run-method" \
    | rg -F -- "--run-dir ${run_dir}" \
    | rg -F -- "--method ${METHOD}" >/dev/null
}

wandb_process_alive() {
  local run_dir="$1"
  ps -eo args \
    | rg -F -- "python scripts/live_wandb_partial_uploader.py" \
    | rg -F -- "--run-dir ${run_dir}" >/dev/null
}

runner_progress_stale() {
  local run_dir="$1"
  local stale_seconds="$2"
  python - "$run_dir" "$stale_seconds" <<'PY'
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

run_dir = Path(sys.argv[1])
stale_seconds = float(sys.argv[2])
progress_path = run_dir / "risky_ps" / "progress.json"
if not progress_path.exists():
    raise SystemExit(0)
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

wandb_state_stale() {
  local run_dir="$1"
  local stale_seconds="$2"
  python - "$run_dir" "$stale_seconds" <<'PY'
import json
import sys
import time
from pathlib import Path

run_dir = Path(sys.argv[1])
stale_seconds = float(sys.argv[2])
progress_path = run_dir / "risky_ps" / "progress.json"
if not progress_path.exists():
    raise SystemExit(1)
progress = json.loads(progress_path.read_text())
completed = int(progress.get("completed_episodes", 0) or 0)
last_completed = int(progress.get("last_completed_episode_index", completed - 1) or -1)
if completed <= 0 or str(progress.get("status", "")) == "completed":
    raise SystemExit(1)
state_files = sorted(run_dir.glob("live_wandb_uploader_state*.json"))
if not state_files:
    raise SystemExit(0)
state_path = max(state_files, key=lambda path: path.stat().st_mtime)
try:
    state = json.loads(state_path.read_text())
except Exception:
    raise SystemExit(0)
last_uploaded = int(state.get("risky_ps", {}).get("last_uploaded_episode_index", -1))
if last_uploaded >= last_completed:
    raise SystemExit(1)
age = time.time() - state_path.stat().st_mtime
raise SystemExit(0 if age >= stale_seconds else 1)
PY
}

restart_runner() {
  local label="$1"
  local run_dir="$2"
  local session="$3"

  tmux kill-session -t "$session" 2>/dev/null || true
  mkdir -p "${run_dir}/${METHOD}"
  log "restarting runner ${label} from checkpoint; run_dir=${run_dir}"
  tmux new-session -d -s "$session" -n "$METHOD" \
    "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/run_shared_basin_repeated_smoke.py run-method --run-dir '${run_dir}' --method '${METHOD}' 2>&1 | tee -a '${run_dir}/${METHOD}/tmux_runner.log'"
}

restart_wandb() {
  local label="$1"
  local key="$2"
  local run_dir="$3"
  local session="$4"
  local experiment_name

  experiment_name="$(basename "$run_dir")"
  tmux kill-session -t "$session" 2>/dev/null || true
  log "restarting wandb uploader ${label}; run_dir=${run_dir}"
  tmux new-session -d -s "$session" -n wandb \
    "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/live_wandb_partial_uploader.py --run-dir '${run_dir}' --project '${WANDB_PROJECT}' --entity '${WANDB_ENTITY}' --run-group '${WANDB_GROUP}' --run-name-prefix '${experiment_name}_' --run-id-suffix 'analysis_${key}_d2_activegapv2_nomodecost_riskyps' --methods '${METHOD}' --poll-seconds 20 --finish-when-complete 2>&1 | tee -a '${run_dir}/live_wandb_uploader_analysis_${key}_d2_activegapv2_nomodecost_riskyps.log'"
}

log "share-variant watchdog started; seed=${SEED} poll_seconds=${POLL_SECONDS}"
log "stale thresholds: runner_progress=${RUNNER_PROGRESS_STALE_SECONDS}s wandb_state=${WANDB_STATE_STALE_SECONDS}s"

while true; do
  for idx in "${!LABELS[@]}"; do
    label="${LABELS[$idx]}"
    key="${KEYS[$idx]}"
    run_dir="${RUN_DIRS[$idx]}"
    runner_session="${RUNNER_SESSIONS[$idx]}"
    wandb_session="${WANDB_SESSIONS[$idx]}"
    summary="$(progress_summary "$run_dir")"

    if progress_complete "$run_dir"; then
      log "complete ${label}: ${summary}"
      continue
    fi

    if runner_process_alive "$run_dir" && ! runner_progress_stale "$run_dir" "$RUNNER_PROGRESS_STALE_SECONDS"; then
      log "healthy runner ${label}: ${summary}"
    else
      log "unhealthy runner ${label}: ${summary}"
      restart_runner "$label" "$run_dir" "$runner_session"
    fi

    if wandb_process_alive "$run_dir" && ! wandb_state_stale "$run_dir" "$WANDB_STATE_STALE_SECONDS"; then
      log "healthy wandb ${label}"
    else
      log "unhealthy wandb ${label}"
      restart_wandb "$label" "$key" "$run_dir" "$wandb_session"
    fi
  done

  sleep "$POLL_SECONDS"
done
