#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEED="${1:-1}"
LABEL="${2:-base_4of5_exp_local_rerun_same_seed1}"
EXPERIMENT_NAME="llm_v8_trapasymv6small30_full_d3_3x100_4of5baselines_pssharevariants_stage45_contract_promptv11b_stage45_gpt41mini_cleanv3_eta03_eps001_seed${SEED}"
GROUP_DIR="tmp/${EXPERIMENT_NAME}"
RUN_DIR="${GROUP_DIR}/${LABEL}"
METHOD="direct_multistage_exp3_local"
ENV_FILE="scripts/trapasymv6small30_full_d3_4of5_11conditions_seed${SEED}_env.sh"
WANDB_PROJECT="psagent-llm-smoke"
WANDB_ENTITY="wangjinyu0711-microsoft"
WANDB_GROUP="trapasymv6small30_full_d3_4of5baselines_pssharevariants_eta03_eps001_seed${SEED}"
RUNNER_SESSION="psagent_trapasymv6small30_full_d3_${LABEL}_seed${SEED}_runner"
WANDB_SESSION="psagent_trapasymv6small30_full_d3_${LABEL}_seed${SEED}_wandb"

POLL_SECONDS="${PSAGENT_TRAPASYM_EXP_LOCAL_RERUN_WATCH_POLL_SECONDS:-120}"
RUNNER_STALE_SECONDS="${PSAGENT_TRAPASYM_EXP_LOCAL_RERUN_RUNNER_STALE_SECONDS:-1800}"
WANDB_STALE_SECONDS="${PSAGENT_TRAPASYM_EXP_LOCAL_RERUN_WANDB_STALE_SECONDS:-900}"
LOG_DIR="${ROOT_DIR}/tmp/watchdog_logs"
LOG_FILE="${LOG_DIR}/trapasymv6small30_exp_local_rerun_same_seed${SEED}_watchdog.log"
LOCK_DIR="${LOG_DIR}/trapasymv6small30_exp_local_rerun_same_seed${SEED}_watchdog.lock"

mkdir -p "$LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_FILE"
}

cleanup_lock() {
  rm -rf "$LOCK_DIR"
}

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "watchdog lock exists at $LOCK_DIR; another rerun watchdog may already be running" >&2
  exit 1
fi
trap cleanup_lock EXIT

progress_summary() {
  python - "$RUN_DIR" "$METHOD" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) / sys.argv[2] / "progress.json"
if not path.exists():
    print("no progress.json")
    raise SystemExit(0)
d = json.loads(path.read_text())
print(
    f"{d.get('completed_episodes', 0)}/{d.get('scheduled_episodes', '?')} "
    f"status={d.get('status', '?')} updated_at={d.get('updated_at', '?')}"
)
PY
}

progress_complete() {
  python - "$RUN_DIR" "$METHOD" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) / sys.argv[2] / "progress.json"
if not path.exists():
    raise SystemExit(1)
d = json.loads(path.read_text())
completed = int(d.get("completed_episodes", 0) or 0)
scheduled = int(d.get("scheduled_episodes", 0) or 0)
status = str(d.get("status", ""))
raise SystemExit(0 if scheduled and completed >= scheduled and status in {"complete", "completed"} else 1)
PY
}

runner_alive() {
  python - "$RUN_DIR" "$METHOD" <<'PY'
import shlex
import subprocess
import sys

run_dir, method = sys.argv[1:3]
try:
    out = subprocess.check_output(["ps", "-eo", "args="], text=True)
except Exception:
    raise SystemExit(1)
for line in out.splitlines():
    try:
        parts = shlex.split(line)
    except ValueError:
        continue
    if "scripts/run_shared_basin_repeated_smoke.py" not in parts or "run-method" not in parts:
        continue
    try:
        if parts[parts.index("--run-dir") + 1] == run_dir and parts[parts.index("--method") + 1] == method:
            raise SystemExit(0)
    except (ValueError, IndexError):
        continue
raise SystemExit(1)
PY
}

runner_stale() {
  python - "$RUN_DIR" "$METHOD" "$RUNNER_STALE_SECONDS" <<'PY'
import json
import sys
import time
from pathlib import Path

run_dir = Path(sys.argv[1])
method = sys.argv[2]
stale = float(sys.argv[3])
progress = run_dir / method / "progress.json"
log = run_dir / method / "tmux_runner.log"
now = time.time()
if log.exists() and now - log.stat().st_mtime < stale:
    raise SystemExit(1)
if not progress.exists():
    raise SystemExit(0)
d = json.loads(progress.read_text())
completed = int(d.get("completed_episodes", 0) or 0)
scheduled = int(d.get("scheduled_episodes", 0) or 0)
status = str(d.get("status", ""))
if status in {"complete", "completed"} or (scheduled and completed >= scheduled):
    raise SystemExit(1)
raise SystemExit(0 if now - progress.stat().st_mtime >= stale else 1)
PY
}

wandb_alive() {
  python - "$RUN_DIR" <<'PY'
import shlex
import subprocess
import sys

run_dir = sys.argv[1]
try:
    out = subprocess.check_output(["ps", "-eo", "args="], text=True)
except Exception:
    raise SystemExit(1)
for line in out.splitlines():
    try:
        parts = shlex.split(line)
    except ValueError:
        continue
    if "scripts/live_wandb_partial_uploader.py" not in parts:
        continue
    try:
        if parts[parts.index("--run-dir") + 1] == run_dir:
            raise SystemExit(0)
    except (ValueError, IndexError):
        continue
raise SystemExit(1)
PY
}

wandb_stale() {
  python - "$RUN_DIR" "$METHOD" "$WANDB_STALE_SECONDS" <<'PY'
import json
import sys
import time
from pathlib import Path

run_dir = Path(sys.argv[1])
method = sys.argv[2]
stale = float(sys.argv[3])
progress_path = run_dir / method / "progress.json"
if not progress_path.exists():
    raise SystemExit(1)
progress = json.loads(progress_path.read_text())
completed = int(progress.get("completed_episodes", 0) or 0)
scheduled = int(progress.get("scheduled_episodes", 0) or 0)
status = str(progress.get("status", ""))
if completed <= 0 or status in {"complete", "completed"} or (scheduled and completed >= scheduled):
    raise SystemExit(1)
last_completed = int(progress.get("last_completed_episode_index", completed - 1) or -1)
state_files = sorted(run_dir.glob("live_wandb_uploader_state*.json"))
if not state_files:
    raise SystemExit(0)
state_path = max(state_files, key=lambda p: p.stat().st_mtime)
try:
    state = json.loads(state_path.read_text())
except Exception:
    raise SystemExit(0)
last_uploaded = int(state.get(method, {}).get("last_uploaded_episode_index", -1))
if last_uploaded >= last_completed:
    raise SystemExit(1)
raise SystemExit(0 if time.time() - state_path.stat().st_mtime >= stale else 1)
PY
}

restart_runner() {
  tmux kill-session -t "$RUNNER_SESSION" 2>/dev/null || true
  mkdir -p "${RUN_DIR}/${METHOD}"
  log "restarting runner label=${LABEL} run_dir=${RUN_DIR}"
  tmux new-session -d -s "$RUNNER_SESSION" -n "$METHOD" \
    "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/run_shared_basin_repeated_smoke.py run-method --run-dir '${RUN_DIR}' --method '${METHOD}' 2>&1 | tee -a '${RUN_DIR}/${METHOD}/tmux_runner.log'"
}

restart_wandb() {
  tmux kill-session -t "$WANDB_SESSION" 2>/dev/null || true
  log "restarting wandb uploader label=${LABEL} run_dir=${RUN_DIR}"
  tmux new-session -d -s "$WANDB_SESSION" -n wandb \
    "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/live_wandb_partial_uploader.py --run-dir '${RUN_DIR}' --project '${WANDB_PROJECT}' --entity '${WANDB_ENTITY}' --run-group '${WANDB_GROUP}' --run-name-prefix '${EXPERIMENT_NAME}_${LABEL}_' --run-id-suffix 'analysis_${LABEL}' --methods '${METHOD}' --poll-seconds 20 --finish-when-complete 2>&1 | tee -a '${RUN_DIR}/live_wandb_uploader_analysis_${LABEL}.log'"
}

merge_if_complete() {
  python scripts/run_shared_basin_repeated_smoke.py merge-method --run-dir "$RUN_DIR" --method "$METHOD" >/dev/null
  python scripts/run_shared_basin_repeated_smoke.py merge-all --run-dir "$RUN_DIR" >/dev/null
}

log "v6 small30 exp_local rerun watchdog started seed=${SEED} label=${LABEL} poll=${POLL_SECONDS}s"
log "run_dir=${RUN_DIR}"

while true; do
  summary="$(progress_summary)"
  if progress_complete; then
    log "complete label=${LABEL}: ${summary}"
    if [[ ! -d "${RUN_DIR}/${METHOD}/merged" ]]; then
      log "merging completed label=${LABEL}"
      merge_if_complete || log "merge failed label=${LABEL}; will retry"
    fi
    exit 0
  fi

  if runner_alive && ! runner_stale; then
    log "runner ok label=${LABEL}: ${summary}"
  else
    log "runner missing/stale label=${LABEL}: ${summary}"
    restart_runner
  fi

  if wandb_alive && ! wandb_stale; then
    log "wandb ok label=${LABEL}"
  else
    log "wandb missing/stale label=${LABEL}"
    restart_wandb
  fi

  sleep "$POLL_SECONDS"
done
