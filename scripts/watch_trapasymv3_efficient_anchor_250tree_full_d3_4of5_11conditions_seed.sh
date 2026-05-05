#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEED="${1:-1}"
POLL_SECONDS="${PSAGENT_TRAPASYM_V3_EFFICIENT_ANCHOR_250_FULL_D3_WATCH_POLL_SECONDS:-120}"
RUNNER_PROGRESS_STALE_SECONDS="${PSAGENT_TRAPASYM_RUNNER_PROGRESS_STALE_SECONDS:-1200}"
RUNNER_STARTUP_GRACE_SECONDS="${PSAGENT_TRAPASYM_V3_EFFICIENT_ANCHOR_250_FULL_D3_RUNNER_STARTUP_GRACE_SECONDS:-240}"
RUNNER_LOG_STALE_SECONDS="${PSAGENT_TRAPASYM_V3_EFFICIENT_ANCHOR_250_FULL_D3_RUNNER_LOG_STALE_SECONDS:-1200}"
WANDB_STATE_STALE_SECONDS="${PSAGENT_TRAPASYM_WANDB_STATE_STALE_SECONDS:-900}"
EXPERIMENT_NAME="llm_v8_trapasymv3efficientanchor250_full_d3_3x100_4of5baselines_pssharevariants_stage45_contract_promptv11b_stage45_gpt41mini_cleanv3_eta03_eps001_seed${SEED}"
WANDB_PROJECT="psagent-llm-smoke"
WANDB_ENTITY="wangjinyu0711-microsoft"
WANDB_GROUP="trapasymv3efficientanchor250_full_d3_4of5baselines_pssharevariants_eta03_eps001_seed${SEED}"
GROUP_DIR="tmp/${EXPERIMENT_NAME}"
ENV_FILE="scripts/trapasymv3efficientanchor250_full_d3_4of5_11conditions_seed${SEED}_env.sh"
LOG_DIR="${ROOT_DIR}/tmp/watchdog_logs"
LOG_FILE="${LOG_DIR}/trapasymv3efficientanchor250_full_d3_4of5_11conditions_seed${SEED}_watchdog.log"
LOCK_DIR="${LOG_DIR}/trapasymv3efficientanchor250_full_d3_4of5_11conditions_seed${SEED}_watchdog.lock"

LABELS=(
  "base_4of5_naive_mixed_avg"
  "base_4of5_random"
  "base_4of5_exp_local"
  "base_4of5_exp"
  "base_4of5_eps"
  "base_4of5_ps_etashare005"
  "base_4of5_theta_guided"
  "base_4of5_agent_only"
  "ps_2of5_etashare005"
  "ps_allunshare_etashare005"
  "ps_allshare_leafratio_etashare015"
)
RUNNER_TYPES=(
  "repeated"
  "repeated"
  "repeated"
  "repeated"
  "repeated"
  "repeated"
  "mechanism"
  "mechanism"
  "repeated"
  "repeated"
  "repeated"
)
METHODS=(
  "naive_mixed_avg"
  "random_path"
  "direct_multistage_exp3_local"
  "direct_multistage_exp3"
  "epsilon_exp3"
  "risky_ps"
  "theta_guided_agent"
  "agent_only"
  "risky_ps"
  "risky_ps"
  "risky_ps_const_init_leaf_ratio_decay"
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

run_dir_for_label() {
  printf '%s/%s' "$GROUP_DIR" "$1"
}

runner_session_for_label() {
  printf 'psagent_trapasymv3efficientanchor250_full_d3_%s_seed%s_runner' "$1" "$SEED"
}

wandb_session_for_label() {
  printf 'psagent_trapasymv3efficientanchor250_full_d3_%s_seed%s_wandb' "$1" "$SEED"
}

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
completed = int(progress.get("completed_episodes", 0) or 0)
scheduled = int(progress.get("scheduled_episodes", 0) or 0)
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
  local runner_type="$3"
  python - "$run_dir" "$method" "$runner_type" <<'PY'
import shlex
import subprocess
import sys

run_dir, method, runner_type = sys.argv[1:4]
script = (
    "scripts/run_shared_basin_repeated_smoke.py"
    if runner_type == "repeated"
    else "scripts/run_shared_basin_mechanism_repeated_smoke.py"
)
subcommand = "run-method" if runner_type == "repeated" else "run-mechanism"
method_flag = "--method" if runner_type == "repeated" else "--mechanism"
try:
    output = subprocess.check_output(["ps", "-eo", "args="], text=True)
except Exception:
    raise SystemExit(1)
for line in output.splitlines():
    try:
        parts = shlex.split(line)
    except ValueError:
        continue
    if script not in parts or subcommand not in parts:
        continue
    try:
        if parts[parts.index("--run-dir") + 1] != run_dir:
            continue
        if parts[parts.index(method_flag) + 1] != method:
            continue
    except (ValueError, IndexError):
        continue
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
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

run_dir_text = sys.argv[1]
run_dir = Path(run_dir_text)
method = sys.argv[2]
stale_seconds = float(sys.argv[3])
progress_path = run_dir / method / "progress.json"
log_path = run_dir / method / "tmux_runner.log"

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
        repeated = (
            "scripts/run_shared_basin_repeated_smoke.py" in parts
            and "run-method" in parts
            and "--method" in parts
        )
        mechanism = (
            "scripts/run_shared_basin_mechanism_repeated_smoke.py" in parts
            and "run-mechanism" in parts
            and "--mechanism" in parts
        )
        if not repeated and not mechanism:
            continue
        method_flag = "--method" if repeated else "--mechanism"
        try:
            if parts[parts.index("--run-dir") + 1] != run_dir_text:
                continue
            if parts[parts.index(method_flag) + 1] != method:
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

runner_starts = [
    ts
    for pid in matching_runner_pids()
    for ts in [process_started_at(pid)]
    if ts is not None
]
startup_grace_seconds = float(
    os.environ.get(
        "PSAGENT_TRAPASYM_V3_EFFICIENT_ANCHOR_250_FULL_D3_RUNNER_STARTUP_GRACE_SECONDS",
        str(stale_seconds),
    )
)
log_stale_seconds = float(
    os.environ.get(
        "PSAGENT_TRAPASYM_V3_EFFICIENT_ANCHOR_250_FULL_D3_RUNNER_LOG_STALE_SECONDS",
        str(stale_seconds),
    )
)
if runner_starts and time.time() - max(runner_starts) < startup_grace_seconds:
    raise SystemExit(1)

if not progress_path.exists():
    if not log_path.exists():
        raise SystemExit(0)
    raise SystemExit(0 if time.time() - log_path.stat().st_mtime >= log_stale_seconds else 1)
if log_path.exists():
    log_age = time.time() - log_path.stat().st_mtime
    if log_age >= log_stale_seconds:
        raise SystemExit(0)
    if log_path.stat().st_mtime > progress_path.stat().st_mtime:
        raise SystemExit(1)
progress = json.loads(progress_path.read_text())
completed = int(progress.get("completed_episodes", 0) or 0)
scheduled = int(progress.get("scheduled_episodes", 0) or 0)
status = str(progress.get("status", ""))
if status in {"complete", "completed"} or (scheduled and completed >= scheduled):
    raise SystemExit(1)
raise SystemExit(0 if time.time() - progress_path.stat().st_mtime >= stale_seconds else 1)
PY
}

wandb_process_alive() {
  local run_dir="$1"
  python - "$run_dir" <<'PY'
import shlex
import subprocess
import sys

run_dir = sys.argv[1]
try:
    output = subprocess.check_output(["ps", "-eo", "args="], text=True)
except Exception:
    raise SystemExit(1)
for line in output.splitlines():
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
scheduled = int(progress.get("scheduled_episodes", 0) or 0)
status = str(progress.get("status", ""))
if completed <= 0 or status in {"complete", "completed"} or (scheduled and completed >= scheduled):
    raise SystemExit(1)
last_completed = int(progress.get("last_completed_episode_index", completed - 1) or -1)
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
raise SystemExit(0 if time.time() - state_path.stat().st_mtime >= stale_seconds else 1)
PY
}

restart_runner() {
  local label="$1"
  local run_dir="$2"
  local method="$3"
  local runner_type="$4"
  local session="$5"

  tmux kill-session -t "$session" 2>/dev/null || true
  mkdir -p "${run_dir}/${method}"
  log "restarting runner label=${label} type=${runner_type} run_dir=${run_dir}"
  if [[ "$runner_type" == "repeated" ]]; then
    tmux new-session -d -s "$session" -n "$method" \
      "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/run_shared_basin_repeated_smoke.py run-method --run-dir '${run_dir}' --method '${method}' 2>&1 | tee -a '${run_dir}/${method}/tmux_runner.log'"
  else
    tmux new-session -d -s "$session" -n "$method" \
      "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/run_shared_basin_mechanism_repeated_smoke.py run-mechanism --run-dir '${run_dir}' --mechanism '${method}' 2>&1 | tee -a '${run_dir}/${method}/tmux_runner.log'"
  fi
}

restart_wandb() {
  local label="$1"
  local run_dir="$2"
  local method="$3"
  local session="$4"

  tmux kill-session -t "$session" 2>/dev/null || true
  log "restarting wandb uploader label=${label} run_dir=${run_dir}"
  tmux new-session -d -s "$session" -n wandb \
    "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/live_wandb_partial_uploader.py --run-dir '${run_dir}' --project '${WANDB_PROJECT}' --entity '${WANDB_ENTITY}' --run-group '${WANDB_GROUP}' --run-name-prefix '${EXPERIMENT_NAME}_${label}_' --run-id-suffix 'analysis_${label}' --methods '${method}' --poll-seconds 20 --finish-when-complete 2>&1 | tee -a '${run_dir}/live_wandb_uploader_analysis_${label}.log'"
}

merge_if_complete() {
  local run_dir="$1"
  local method="$2"
  local runner_type="$3"
  if [[ "$runner_type" == "repeated" ]]; then
    python scripts/run_shared_basin_repeated_smoke.py merge-method --run-dir "$run_dir" --method "$method" >/dev/null
    python scripts/run_shared_basin_repeated_smoke.py merge-all --run-dir "$run_dir" >/dev/null
  else
    python scripts/run_shared_basin_mechanism_repeated_smoke.py merge-mechanism --run-dir "$run_dir" --mechanism "$method" >/dev/null
    python scripts/run_shared_basin_mechanism_repeated_smoke.py merge-all --run-dir "$run_dir" >/dev/null
  fi
}

write_group_compare() {
  python - "$GROUP_DIR" "${LABELS[@]}" <<'PY'
import json
import sys
from pathlib import Path

group_dir = Path(sys.argv[1])
labels = sys.argv[2:]
rows = []
for label in labels:
    run_dir = group_dir / label
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        continue
    config = json.loads(config_path.read_text())
    names = config.get("methods") or config.get("mechanisms") or []
    for name in names:
        summary_path = run_dir / name / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        rows.append(
            {
                "label": label,
                "name": name,
                "runner_type": "mechanism" if "mechanisms" in config else "repeated",
                "family_kind": config.get("family_kind"),
                "raw_total_cost_mean": summary.get("raw_total_cost_mean"),
                "raw_terminal_penalty_mean": summary.get("raw_terminal_penalty_mean"),
                "raw_reasoning_cost_component_mean": summary.get("raw_reasoning_cost_component_mean"),
                "exact_match_mean": summary.get("exact_match_mean"),
                "post_switch_raw_total_cost_mean": summary.get("post_switch_raw_total_cost_mean"),
                "shared_path_fraction": summary.get("shared_path_fraction"),
                "unshared_path_fraction": summary.get("unshared_path_fraction"),
            }
        )
rows.sort(key=lambda row: (
    float("inf") if row["raw_total_cost_mean"] is None else float(row["raw_total_cost_mean"]),
    row["label"],
))
(group_dir / "group_compare.json").write_text(json.dumps(rows, indent=2) + "\n")
if rows:
    header = list(rows[0].keys())
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join("" if row.get(k) is None else str(row.get(k)) for k in header))
    (group_dir / "group_compare.csv").write_text("\n".join(lines) + "\n")
PY
}

log "v3 efficient-anchor 250 full d3 4of5 11-condition watchdog started seed=${SEED} poll=${POLL_SECONDS}s"
log "group_dir=${GROUP_DIR}"
log "stale thresholds runner=${RUNNER_PROGRESS_STALE_SECONDS}s startup_grace=${RUNNER_STARTUP_GRACE_SECONDS}s log=${RUNNER_LOG_STALE_SECONDS}s wandb=${WANDB_STATE_STALE_SECONDS}s"

while true; do
  all_done=1
  for idx in "${!LABELS[@]}"; do
    label="${LABELS[$idx]}"
    method="${METHODS[$idx]}"
    runner_type="${RUNNER_TYPES[$idx]}"
    run_dir="$(run_dir_for_label "$label")"
    runner_session="$(runner_session_for_label "$label")"
    wandb_session="$(wandb_session_for_label "$label")"
    summary="$(progress_summary "$run_dir" "$method")"

    if progress_complete "$run_dir" "$method"; then
      log "complete label=${label}: ${summary}"
      if [[ ! -d "${run_dir}/${method}/merged" ]]; then
        log "merging completed label=${label}"
        merge_if_complete "$run_dir" "$method" "$runner_type" || log "merge failed label=${label}; will retry"
      fi
      continue
    fi

    all_done=0
    if runner_process_alive "$run_dir" "$method" "$runner_type" && ! PSAGENT_TRAPASYM_V3_EFFICIENT_ANCHOR_250_FULL_D3_RUNNER_STARTUP_GRACE_SECONDS="$RUNNER_STARTUP_GRACE_SECONDS" PSAGENT_TRAPASYM_V3_EFFICIENT_ANCHOR_250_FULL_D3_RUNNER_LOG_STALE_SECONDS="$RUNNER_LOG_STALE_SECONDS" runner_progress_stale "$run_dir" "$method" "$RUNNER_PROGRESS_STALE_SECONDS"; then
      log "runner ok label=${label}: ${summary}"
    else
      log "runner missing/stale label=${label}: ${summary}"
      restart_runner "$label" "$run_dir" "$method" "$runner_type" "$runner_session"
    fi

    if wandb_process_alive "$run_dir" && ! wandb_state_stale "$run_dir" "$method" "$WANDB_STATE_STALE_SECONDS"; then
      log "wandb ok label=${label}"
    else
      log "wandb missing/stale label=${label}"
      restart_wandb "$label" "$run_dir" "$method" "$wandb_session"
    fi
  done

  write_group_compare || log "group compare write failed; will retry"
  if [[ "$all_done" == "1" ]]; then
    log "all labels complete; group compare written; watchdog exiting"
    exit 0
  fi
  sleep "$POLL_SECONDS"
done
