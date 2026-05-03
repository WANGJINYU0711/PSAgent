#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEED="${1:-1}"
POLL_SECONDS="${PSAGENT_TRAPASYM_WATCH_POLL_SECONDS:-120}"
RUNNER_PROGRESS_STALE_SECONDS="${PSAGENT_TRAPASYM_RUNNER_PROGRESS_STALE_SECONDS:-1800}"
WANDB_STATE_STALE_SECONDS="${PSAGENT_TRAPASYM_WANDB_STATE_STALE_SECONDS:-900}"
LOG_DIR="${ROOT_DIR}/tmp/watchdog_logs"
LOG_FILE="${LOG_DIR}/trapasymv3_full_d4_nomodecost_seed${SEED}_watchdog.log"
LOCK_DIR="${LOG_DIR}/trapasymv3_full_d4_nomodecost_seed${SEED}_watchdog.lock"
SESSION_NAME="psagent_trapasymv3_full_d4_nomodecost_seed${SEED}_11m"
LIVE_SESSION="psagent_trapasymv3_full_d4_nomodecost_seed${SEED}_live_wandb"
LAUNCH_COMMAND="bash scripts/launch_trapasymv3_full_d4_nomodecost_llm_run.sh ${SEED}"
ENV_FILE="scripts/trapasymv3_full_d4_nomodecost_seed${SEED}_env.sh"
EXPERIMENT_NAME="llm_v8_trapasymv3efficientanchor4of5_stage45_contract_promptv11b_stage45_gpt41mini_full_d4_nomodecost_reason135_eps001_3x100_v3_11methods_seed${SEED}"
RUN_DIR="tmp/${EXPERIMENT_NAME}"
WANDB_GROUP="trapasymv3_full_d4_nomodecost_seed${SEED}"
METHODS=(
  naive_mixed
  naive_mixed_avg
  direct_multistage_exp3
  direct_multistage_exp3_local
  epsilon_exp3
  random_path
  risky_ps_old
  risky_ps
  risky_ps_safe_conditional
  risky_ps_ix
  risky_ps_safe_conditional_ix
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

session_healthy() {
  if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    return 1
  fi
  return 0
}

method_process_alive() {
  local method="$1"
  python - "$RUN_DIR" "$method" <<'PY'
import shlex
import subprocess
import sys

run_dir = sys.argv[1]
method = sys.argv[2]
try:
    output = subprocess.check_output(["ps", "-eo", "args"], text=True)
except Exception:
    raise SystemExit(1)
for line in output.splitlines():
    try:
        parts = shlex.split(line)
    except ValueError:
        continue
    if "scripts/run_shared_basin_repeated_smoke.py" not in parts:
        continue
    if "run-method" not in parts:
        continue
    try:
        if parts[parts.index("--run-dir") + 1] != run_dir:
            continue
        if parts[parts.index("--method") + 1] != method:
            continue
    except (ValueError, IndexError):
        continue
    raise SystemExit(0)
raise SystemExit(1)
PY
}

progress_complete() {
  local method="$1"
  python - "$RUN_DIR" "$method" <<'PY'
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
  local method="$1"
  python - "$RUN_DIR" "$method" <<'PY'
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
  local method="$1"
  python - "$RUN_DIR" "$method" "$RUNNER_PROGRESS_STALE_SECONDS" <<'PY'
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

progress_path = Path(sys.argv[1]) / sys.argv[2] / "progress.json"
run_dir = sys.argv[1]
method = sys.argv[2]
stale_seconds = float(sys.argv[3])

def matching_runner_pids() -> list[int]:
    try:
        output = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    except Exception:
        return []
    pids: list[int] = []
    for line in output.splitlines()[1:]:
        pid_text, _, args = line.strip().partition(" ")
        try:
            pid = int(pid_text)
            parts = shlex.split(args)
        except (ValueError, IndexError):
            continue
        if "scripts/run_shared_basin_repeated_smoke.py" not in parts:
            continue
        if "run-method" not in parts:
            continue
        try:
            if parts[parts.index("--run-dir") + 1] != run_dir:
                continue
            if parts[parts.index("--method") + 1] != method:
                continue
        except (ValueError, IndexError):
            continue
        pids.append(pid)
    return pids

def process_started_at(pid: int) -> float | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        start_ticks = int(stat.rsplit(") ", 1)[1].split()[19])
        boot_time = None
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("btime "):
                boot_time = int(line.split()[1])
                break
        if boot_time is None:
            return None
        ticks_per_second = int(subprocess.check_output(["getconf", "CLK_TCK"], text=True))
        return boot_time + (start_ticks / ticks_per_second)
    except Exception:
        return None

runner_starts = [ts for pid in matching_runner_pids() for ts in [process_started_at(pid)] if ts]
if runner_starts and time.time() - max(runner_starts) < stale_seconds:
    raise SystemExit(1)

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

wandb_process_alive() {
  ps -eo args \
    | rg -F -- "python scripts/live_wandb_partial_uploader.py" \
    | rg -F -- "--run-dir ${RUN_DIR}" >/dev/null
}

wandb_state_stale() {
  python - "$RUN_DIR" "$WANDB_STATE_STALE_SECONDS" "${METHODS[@]}" <<'PY'
import json
import sys
import time
from pathlib import Path

run_dir = Path(sys.argv[1])
stale_seconds = float(sys.argv[2])
methods = sys.argv[3:]
state_files = sorted(run_dir.glob("live_wandb_uploader_state*.json"))
if not state_files:
    raise SystemExit(0)
state_path = max(state_files, key=lambda path: path.stat().st_mtime)
try:
    state = json.loads(state_path.read_text())
except Exception:
    raise SystemExit(0)
for method in methods:
    progress_path = run_dir / method / "progress.json"
    if not progress_path.exists():
        continue
    progress = json.loads(progress_path.read_text())
    completed = int(progress.get("completed_episodes", 0) or 0)
    if completed <= 0 or str(progress.get("status", "")) == "completed":
        continue
    last_completed = int(progress.get("last_completed_episode_index", completed - 1) or -1)
    last_uploaded = int(state.get(method, {}).get("last_uploaded_episode_index", -1))
    if last_uploaded < last_completed:
        age = time.time() - state_path.stat().st_mtime
        raise SystemExit(0 if age >= stale_seconds else 1)
raise SystemExit(1)
PY
}

restart_method() {
  local method="$1"
  local window_id=""
  window_id="$(
    tmux list-windows -t "$SESSION_NAME" -F '#{window_id}	#{window_name}' 2>/dev/null \
      | awk -v name="$method" -F '\t' '$2 == name {print $1; exit}'
  )"
  if [[ -n "$window_id" ]]; then
    tmux kill-window -t "$window_id" 2>/dev/null || true
  fi
  mkdir -p "${RUN_DIR}/${method}"
  log "restarting method ${method} from checkpoint"
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    tmux new-window -t "$SESSION_NAME" -n "$method" \
      "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/run_shared_basin_repeated_smoke.py run-method --run-dir '${RUN_DIR}' --method '${method}' 2>&1 | tee -a '${RUN_DIR}/${method}/tmux_runner.log'"
  else
    tmux new-session -d -s "$SESSION_NAME" -n "$method" \
      "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/run_shared_basin_repeated_smoke.py run-method --run-dir '${RUN_DIR}' --method '${method}' 2>&1 | tee -a '${RUN_DIR}/${method}/tmux_runner.log'"
  fi
}

restart_wandb() {
  tmux kill-session -t "$LIVE_SESSION" 2>/dev/null || true
  log "restarting wandb uploader for full_d4_nomodecost"
  tmux new-session -d -s "$LIVE_SESSION" -n wandb \
    "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/live_wandb_partial_uploader.py --run-dir '${RUN_DIR}' --project psagent-llm-smoke --entity wangjinyu0711-microsoft --run-group '${WANDB_GROUP}' --run-name-prefix '${EXPERIMENT_NAME}_' --run-id-suffix analysis_full_d4_nomodecost --methods ${METHODS[*]} --poll-seconds 20 --finish-when-complete 2>&1 | tee -a '${RUN_DIR}/live_wandb_uploader_analysis_full_d4_nomodecost.log'"
}

log "watchdog started; seed=${SEED} poll_seconds=${POLL_SECONDS}"
log "stale thresholds: runner_progress=${RUNNER_PROGRESS_STALE_SECONDS}s wandb_state=${WANDB_STATE_STALE_SECONDS}s"

while true; do
  if ! session_healthy; then
    log "unhealthy full_d4_nomodecost session=${SESSION_NAME}; attempting restart from checkpoint"
    eval "$LAUNCH_COMMAND" >>"$LOG_FILE" 2>&1
    log "relaunch finished for full_d4_nomodecost"
  else
    log "healthy full_d4_nomodecost session=${SESSION_NAME}"
    for method in "${METHODS[@]}"; do
      summary="$(progress_summary "$method")"
      if progress_complete "$method"; then
        log "complete method ${method}: ${summary}"
      elif method_process_alive "$method" && ! runner_progress_stale "$method"; then
        log "healthy method ${method}: ${summary}"
      else
        log "unhealthy method ${method}: ${summary}"
        restart_method "$method"
      fi
    done
  fi

  if wandb_process_alive && ! wandb_state_stale; then
    log "healthy wandb full_d4_nomodecost"
  else
    log "unhealthy wandb full_d4_nomodecost"
    restart_wandb
  fi

  sleep "$POLL_SECONDS"
done
