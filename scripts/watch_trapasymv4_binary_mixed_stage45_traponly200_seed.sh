#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEED="${1:-1}"
POLL_SECONDS="${PSAGENT_TRAPASYM_V4_WATCH_POLL_SECONDS:-120}"
RUNNER_PROGRESS_STALE_SECONDS="${PSAGENT_TRAPASYM_RUNNER_PROGRESS_STALE_SECONDS:-1200}"
WANDB_STATE_STALE_SECONDS="${PSAGENT_TRAPASYM_WANDB_STATE_STALE_SECONDS:-900}"
LOG_DIR="${ROOT_DIR}/tmp/watchdog_logs"
LOG_FILE="${LOG_DIR}/trapasymv4_binarymixed_traponly200_seed${SEED}_watchdog.log"
LOCK_DIR="${LOG_DIR}/trapasymv4_binarymixed_traponly200_seed${SEED}_watchdog.lock"
ENV_FILE="scripts/trapasymv4_binarymixed_traponly200_seed${SEED}_env.sh"
WANDB_PROJECT="psagent-llm-smoke"
WANDB_ENTITY="wangjinyu0711-microsoft"
WANDB_GROUP="trapasymv4_binarymixed_traponly200_eta03_eps001_etashare005_seed${SEED}"

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
  "base_exp"
  "base_eps"
  "base_ps"
  "allshare_ps"
  "allunshare_ps"
)
METHODS=(
  "direct_multistage_exp3"
  "epsilon_exp3"
  "risky_ps"
  "risky_ps"
  "risky_ps"
)
RUN_DIRS=(
  "tmp/llm_v8_trapasymv4binarymixed_base_exp_traponly200_stage45_contract_promptv11b_stage45_gpt41mini_cleanv3_eta03_eps001_etashare005_seed${SEED}"
  "tmp/llm_v8_trapasymv4binarymixed_base_eps_traponly200_stage45_contract_promptv11b_stage45_gpt41mini_cleanv3_eta03_eps001_etashare005_seed${SEED}"
  "tmp/llm_v8_trapasymv4binarymixed_base_ps_traponly200_stage45_contract_promptv11b_stage45_gpt41mini_cleanv3_eta03_eps001_etashare005_seed${SEED}"
  "tmp/llm_v8_trapasymv4binarymixed_allshare_ps_traponly200_stage45_contract_promptv11b_stage45_gpt41mini_cleanv3_eta03_eps001_etashare005_seed${SEED}"
  "tmp/llm_v8_trapasymv4binarymixed_allunshare_ps_traponly200_stage45_contract_promptv11b_stage45_gpt41mini_cleanv3_eta03_eps001_etashare005_seed${SEED}"
)
RUNNER_SESSIONS=(
  "psagent_trapasymv4_base_exp_traponly200_seed${SEED}_runner"
  "psagent_trapasymv4_base_eps_traponly200_seed${SEED}_runner"
  "psagent_trapasymv4_base_ps_traponly200_seed${SEED}_runner"
  "psagent_trapasymv4_allshare_ps_traponly200_seed${SEED}_runner"
  "psagent_trapasymv4_allunshare_ps_traponly200_seed${SEED}_runner"
)
WANDB_SESSIONS=(
  "psagent_trapasymv4_base_exp_traponly200_seed${SEED}_live_wandb"
  "psagent_trapasymv4_base_eps_traponly200_seed${SEED}_live_wandb"
  "psagent_trapasymv4_base_ps_traponly200_seed${SEED}_live_wandb"
  "psagent_trapasymv4_allshare_ps_traponly200_seed${SEED}_live_wandb"
  "psagent_trapasymv4_allunshare_ps_traponly200_seed${SEED}_live_wandb"
)

progress_complete() {
  local run_dir="$1"
  local method="$2"
  python - "$run_dir" "$method" <<'PY'
import json
import sys
from pathlib import Path

progress_path = Path(sys.argv[1]) / sys.argv[2] / "progress.json"
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
  local method="$2"
  python - "$run_dir" "$method" <<'PY'
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

runner_process_alive() {
  local run_dir="$1"
  local method="$2"
  python - "$run_dir" "$method" <<'PY'
import subprocess
import sys

run_dir = sys.argv[1]
method = sys.argv[2]
needle = "python scripts/run_shared_basin_repeated_smoke.py run-method"
run_dir_arg = f"--run-dir {run_dir}"
method_arg = f"--method {method}"
out = subprocess.check_output(["ps", "-eo", "args="], text=True)
for line in out.splitlines():
    if needle in line and run_dir_arg in line and method_arg in line:
        raise SystemExit(0)
raise SystemExit(1)
PY
}

wandb_process_alive() {
  local run_dir="$1"
  python - "$run_dir" <<'PY'
import subprocess
import sys

run_dir = sys.argv[1]
needle = "python scripts/live_wandb_partial_uploader.py"
run_dir_arg = f"--run-dir {run_dir}"
out = subprocess.check_output(["ps", "-eo", "args="], text=True)
for line in out.splitlines():
    if needle in line and run_dir_arg in line:
        raise SystemExit(0)
raise SystemExit(1)
PY
}

runner_progress_stale() {
  local run_dir="$1"
  local method="$2"
  local stale_seconds="$3"
  python - "$run_dir" "$method" "$stale_seconds" <<'PY'
import json
import sys
import time
from pathlib import Path

run_dir = Path(sys.argv[1])
method = sys.argv[2]
stale_seconds = float(sys.argv[3])
progress_path = run_dir / method / "progress.json"
if not progress_path.exists():
    raise SystemExit(0)
progress = json.loads(progress_path.read_text())
completed = int(progress.get("completed_episodes", 0) or 0)
scheduled = int(progress.get("scheduled_episodes", 0) or 0)
status = str(progress.get("status", ""))
if status in {"complete", "completed"} or (scheduled and completed >= scheduled):
    raise SystemExit(1)
age = time.time() - progress_path.stat().st_mtime
raise SystemExit(0 if age >= stale_seconds else 1)
PY
}

wandb_state_stale() {
  local run_dir="$1"
  local method="$2"
  local stale_seconds="$3"
  python - "$run_dir" "$method" "$stale_seconds" <<'PY'
import json
import sys
import time
from pathlib import Path

run_dir = Path(sys.argv[1])
method = sys.argv[2]
stale_seconds = float(sys.argv[3])
progress_path = run_dir / method / "progress.json"
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
last_uploaded = int(state.get(method, {}).get("last_uploaded_episode_index", -1))
if last_uploaded >= last_completed:
    raise SystemExit(1)
age = time.time() - state_path.stat().st_mtime
raise SystemExit(0 if age >= stale_seconds else 1)
PY
}

restart_runner() {
  local label="$1"
  local run_dir="$2"
  local method="$3"
  local session="$4"

  tmux kill-session -t "$session" 2>/dev/null || true
  mkdir -p "${run_dir}/${method}"
  log "restarting runner ${label} from checkpoint; run_dir=${run_dir}"
  tmux new-session -d -s "$session" -n "$method" \
    "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/run_shared_basin_repeated_smoke.py run-method --run-dir '${run_dir}' --method '${method}' 2>&1 | tee -a '${run_dir}/${method}/tmux_runner.log'"
}

restart_wandb() {
  local label="$1"
  local run_dir="$2"
  local method="$3"
  local session="$4"
  local experiment_name

  experiment_name="$(basename "$run_dir")"
  tmux kill-session -t "$session" 2>/dev/null || true
  log "restarting wandb uploader ${label}; run_dir=${run_dir}"
  tmux new-session -d -s "$session" -n wandb \
    "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/live_wandb_partial_uploader.py --run-dir '${run_dir}' --project '${WANDB_PROJECT}' --entity '${WANDB_ENTITY}' --run-group '${WANDB_GROUP}' --run-name-prefix '${experiment_name}_' --run-id-suffix 'analysis_${label}_traponly200' --methods '${method}' --poll-seconds 20 --finish-when-complete 2>&1 | tee -a '${run_dir}/live_wandb_uploader_analysis_${label}_traponly200.log'"
}

log "v4 binary-mixed trap-only watchdog started; seed=${SEED} poll_seconds=${POLL_SECONDS}"
log "stale thresholds: runner_progress=${RUNNER_PROGRESS_STALE_SECONDS}s wandb_state=${WANDB_STATE_STALE_SECONDS}s"

while true; do
  all_done=1
  for idx in "${!LABELS[@]}"; do
    label="${LABELS[$idx]}"
    method="${METHODS[$idx]}"
    run_dir="${RUN_DIRS[$idx]}"
    runner_session="${RUNNER_SESSIONS[$idx]}"
    wandb_session="${WANDB_SESSIONS[$idx]}"
    summary="$(progress_summary "$run_dir" "$method")"

    if progress_complete "$run_dir" "$method"; then
      log "complete ${label}: ${summary}"
      continue
    fi
    all_done=0

    if runner_process_alive "$run_dir" "$method" && ! runner_progress_stale "$run_dir" "$method" "$RUNNER_PROGRESS_STALE_SECONDS"; then
      log "healthy runner ${label}: ${summary}"
    else
      log "unhealthy runner ${label}: ${summary}"
      restart_runner "$label" "$run_dir" "$method" "$runner_session"
    fi

    if wandb_process_alive "$run_dir" && ! wandb_state_stale "$run_dir" "$method" "$WANDB_STATE_STALE_SECONDS"; then
      log "healthy wandb ${label}"
    else
      log "unhealthy wandb ${label}"
      restart_wandb "$label" "$run_dir" "$method" "$wandb_session"
    fi
  done

  if [[ "$all_done" == "1" ]]; then
    log "all v4 binary-mixed trap-only runs complete; exiting watchdog"
    exit 0
  fi
  sleep "$POLL_SECONDS"
done
