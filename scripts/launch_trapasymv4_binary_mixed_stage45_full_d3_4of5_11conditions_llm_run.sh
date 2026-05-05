#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEED="${1:-1}"
EXPERIMENT_NAME="llm_v8_trapasymv4binarymixed_full_d3_3x100_4of5baselines_pssharevariants_stage45_contract_promptv11b_stage45_gpt41mini_cleanv3_eta03_eps001_seed${SEED}"
WANDB_PROJECT="psagent-llm-smoke"
WANDB_ENTITY="wangjinyu0711-microsoft"
WANDB_GROUP="trapasymv4_binarymixed_full_d3_4of5baselines_pssharevariants_eta03_eps001_seed${SEED}"
GROUP_DIR="tmp/${EXPERIMENT_NAME}"
ENV_FILE="scripts/trapasymv4_binarymixed_full_d3_4of5_11conditions_seed${SEED}_env.sh"
WATCHDOG_SESSION="psagent_trapasymv4_full_d3_4of5_11c_seed${SEED}_watchdog"

DATASET="data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v3_100/tasks.json"
BUCKETS="analysis/shared_basin_prefix_dedup_profile_switch_local_exec_clean_v3_100_schedule_buckets.json"
BASE_4OF5_FAMILY="shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45_4of5"
PS_2OF5_FAMILY="shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45_2of5"
PS_ALLUNSHARE_FAMILY="shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45_all_unshare"
PS_ALLSHARE_FAMILY="shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45_all_share"

export PSAGENT_LLM_BENCH_MODEL="gpt-4o-mini"
export PSAGENT_REPEATED_SMOKE_SEED="${SEED}"
export PSAGENT_TELECOM_EXEC_CLEAN_TERMINAL_V4="1"
export PSAGENT_TELECOM_REASONING_WEIGHT_CALIBRATION_V3="1"
export PSAGENT_TELECOM_REASONING_COST_MULTIPLIER="1.35"
export PSAGENT_TELECOM_REASONING_CALIBRATED_FAST_MATCH_DISCOUNT="0.45"
export PSAGENT_TELECOM_REASONING_CALIBRATED_DEEP_MATCH_DISCOUNT="0.70"
export PSAGENT_TELECOM_REASONING_CALIBRATED_FAST_ON_DEEP_PENALTY="1.55"
export PSAGENT_TELECOM_REASONING_CALIBRATED_DEEP_ON_FAST_PENALTY="2.25"
export PSAGENT_TELECOM_STAGE45_CONTRACT_PROMPT_V1_1B="1"
export PSAGENT_TELECOM_STAGE45_MODEL="gpt-4.1-mini"
export PSAGENT_TELECOM_LLM_BRIDGE_RETRY_ATTEMPTS="${PSAGENT_TELECOM_LLM_BRIDGE_RETRY_ATTEMPTS:-8}"
export PSAGENT_TELECOM_LLM_BRIDGE_RETRY_SLEEP_SECONDS="${PSAGENT_TELECOM_LLM_BRIDGE_RETRY_SLEEP_SECONDS:-30}"
export PSAGENT_TELECOM_LLM_BRIDGE_TIMEOUT_SECONDS="${PSAGENT_TELECOM_LLM_BRIDGE_TIMEOUT_SECONDS:-600}"

cat > "$ENV_FILE" <<EOF
export PSAGENT_LLM_BENCH_MODEL="gpt-4o-mini"
export PSAGENT_REPEATED_SMOKE_SEED="${SEED}"
export PSAGENT_TELECOM_EXEC_CLEAN_TERMINAL_V4="1"
export PSAGENT_TELECOM_REASONING_WEIGHT_CALIBRATION_V3="1"
export PSAGENT_TELECOM_REASONING_COST_MULTIPLIER="1.35"
export PSAGENT_TELECOM_REASONING_CALIBRATED_FAST_MATCH_DISCOUNT="0.45"
export PSAGENT_TELECOM_REASONING_CALIBRATED_DEEP_MATCH_DISCOUNT="0.70"
export PSAGENT_TELECOM_REASONING_CALIBRATED_FAST_ON_DEEP_PENALTY="1.55"
export PSAGENT_TELECOM_REASONING_CALIBRATED_DEEP_ON_FAST_PENALTY="2.25"
export PSAGENT_TELECOM_STAGE45_CONTRACT_PROMPT_V1_1B="1"
export PSAGENT_TELECOM_STAGE45_MODEL="gpt-4.1-mini"
export PSAGENT_TELECOM_LLM_BRIDGE_RETRY_ATTEMPTS="${PSAGENT_TELECOM_LLM_BRIDGE_RETRY_ATTEMPTS}"
export PSAGENT_TELECOM_LLM_BRIDGE_RETRY_SLEEP_SECONDS="${PSAGENT_TELECOM_LLM_BRIDGE_RETRY_SLEEP_SECONDS}"
export PSAGENT_TELECOM_LLM_BRIDGE_TIMEOUT_SECONDS="${PSAGENT_TELECOM_LLM_BRIDGE_TIMEOUT_SECONDS}"
EOF

mkdir -p "$GROUP_DIR"

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
FAMILY_KINDS=(
  "$BASE_4OF5_FAMILY"
  "$BASE_4OF5_FAMILY"
  "$BASE_4OF5_FAMILY"
  "$BASE_4OF5_FAMILY"
  "$BASE_4OF5_FAMILY"
  "$BASE_4OF5_FAMILY"
  "$BASE_4OF5_FAMILY"
  "$BASE_4OF5_FAMILY"
  "$PS_2OF5_FAMILY"
  "$PS_ALLUNSHARE_FAMILY"
  "$PS_ALLSHARE_FAMILY"
)
PS_ETA_SHAREDS=(
  ""
  ""
  ""
  ""
  ""
  "0.05"
  ""
  ""
  "0.05"
  "0.05"
  "0.15"
)

runner_session_for_label() {
  local label="$1"
  printf 'psagent_trapasymv4_full_d3_%s_seed%s_runner' "$label" "$SEED"
}

wandb_session_for_label() {
  local label="$1"
  printf 'psagent_trapasymv4_full_d3_%s_seed%s_wandb' "$label" "$SEED"
}

validate_setup() {
  local run_dir="$1"
  local runner_type="$2"
  local method="$3"
  local family_kind="$4"
  local ps_eta_shared="$5"
  python - "$run_dir" "$runner_type" "$method" "$family_kind" "$ps_eta_shared" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

run_dir = Path(sys.argv[1])
runner_type, method, family_kind, ps_eta_shared = sys.argv[2:6]
config = json.loads((run_dir / "run_config.json").read_text())
schedule = json.loads((run_dir / "schedule.json").read_text())
metadata = config.get("schedule_metadata") or {}
errors = []
if config.get("dataset") != "data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v3_100/tasks.json":
    errors.append(f"dataset={config.get('dataset')}")
if config.get("schedule_buckets") != "analysis/shared_basin_prefix_dedup_profile_switch_local_exec_clean_v3_100_schedule_buckets.json":
    errors.append(f"schedule_buckets={config.get('schedule_buckets')}")
if config.get("family_kind") != family_kind:
    errors.append(f"family_kind={config.get('family_kind')} expected={family_kind}")
if config.get("schedule_mode") != "trap_switch":
    errors.append(f"schedule_mode={config.get('schedule_mode')}")
if config.get("trap_switch_cycle_source") != "dataset":
    errors.append(f"trap_switch_cycle_source={config.get('trap_switch_cycle_source')}")
if int(config.get("repeats", -1)) != 3:
    errors.append(f"repeats={config.get('repeats')}")
if int(config.get("horizon", -1)) != 300:
    errors.append(f"horizon={config.get('horizon')}")
if int(config.get("switch_denominator", -1)) != 3:
    errors.append(f"switch_denominator={config.get('switch_denominator')}")
if int(metadata.get("cycle_length", -1)) != 100:
    errors.append(f"cycle_length={metadata.get('cycle_length')}")
if int(metadata.get("switch_episode", -1)) != 100:
    errors.append(f"switch_episode={metadata.get('switch_episode')}")
phase_counts = Counter(row.get("schedule_phase") for row in schedule)
if phase_counts != {"trap_pre_switch": 100, "target_post_switch": 200}:
    errors.append(f"phase_counts={dict(phase_counts)}")
if runner_type == "repeated":
    methods = config.get("methods") or []
    if methods != [method]:
        errors.append(f"methods={methods} expected={[method]}")
    kwargs = (config.get("policy_kwargs_by_method") or {}).get(method, {})
    if method in {"direct_multistage_exp3", "direct_multistage_exp3_local", "epsilon_exp3", "risky_ps", "risky_ps_const_init_leaf_ratio_decay"}:
        if float(kwargs.get("eta", -1)) != 0.3:
            errors.append(f"{method}.eta={kwargs.get('eta')}")
    if method in {"epsilon_exp3", "risky_ps", "risky_ps_const_init_leaf_ratio_decay"}:
        if float(kwargs.get("epsilon", -1)) != 0.01:
            errors.append(f"{method}.epsilon={kwargs.get('epsilon')}")
    if ps_eta_shared:
        if abs(float(kwargs.get("eta_shared", -999)) - float(ps_eta_shared)) > 1e-12:
            errors.append(f"{method}.eta_shared={kwargs.get('eta_shared')} expected={ps_eta_shared}")
elif runner_type == "mechanism":
    mechanisms = config.get("mechanisms") or []
    if mechanisms != [method]:
        errors.append(f"mechanisms={mechanisms} expected={[method]}")
    theta_kwargs = config.get("theta_policy_kwargs") or {}
    if float(theta_kwargs.get("eta", -1)) != 0.3:
        errors.append(f"theta_eta={theta_kwargs.get('eta')}")
else:
    errors.append(f"unknown runner_type={runner_type}")
if errors:
    raise SystemExit("setup validation failed for " + str(run_dir) + ": " + "; ".join(errors))
print(f"[validate] ok {run_dir} phases={dict(phase_counts)}")
PY
}

stop_existing_experiment_processes() {
  local label
  for label in "${LABELS[@]}"; do
    tmux kill-session -t "$(runner_session_for_label "$label")" 2>/dev/null || true
    tmux kill-session -t "$(wandb_session_for_label "$label")" 2>/dev/null || true
  done
  tmux kill-session -t "$WATCHDOG_SESSION" 2>/dev/null || true
  python - "$GROUP_DIR" <<'PY'
import os
import signal
import subprocess
import sys

group_dir = sys.argv[1]
out = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
self_pid = os.getpid()
for line in out.splitlines()[1:]:
    pid_text, _, args = line.strip().partition(" ")
    try:
        pid = int(pid_text)
    except ValueError:
        continue
    if pid == self_pid:
        continue
    if group_dir in args:
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"[stop] SIGTERM pid={pid} args={args[:180]}")
        except ProcessLookupError:
            pass
PY
}

stop_existing_experiment_processes

for index in "${!LABELS[@]}"; do
  label="${LABELS[$index]}"
  runner_type="${RUNNER_TYPES[$index]}"
  method="${METHODS[$index]}"
  family_kind="${FAMILY_KINDS[$index]}"
  ps_eta_shared="${PS_ETA_SHAREDS[$index]}"
  run_dir="${GROUP_DIR}/${label}"
  mkdir -p "${run_dir}/${method}"

  if [[ "$runner_type" == "repeated" ]]; then
    setup_cmd=(
      python scripts/run_shared_basin_repeated_smoke.py setup
      --data "$DATASET"
      --schedule-buckets "$BUCKETS"
      --output-dir "$run_dir"
      --family-kind "$family_kind"
      --executor-name llm_bench
      --schedule-mode trap_switch
      --trap-switch-cycle-source dataset
      --switch-denominator 3
      --repeats 3
      --common-eta-override 0.3
      --common-epsilon-override 0.01
      --methods "$method"
    )
    if [[ -n "$ps_eta_shared" ]]; then
      setup_cmd+=(--ps-eta-shared-override "$ps_eta_shared")
    fi
    "${setup_cmd[@]}"
  elif [[ "$runner_type" == "mechanism" ]]; then
    python scripts/run_shared_basin_mechanism_repeated_smoke.py setup \
      --data "$DATASET" \
      --schedule-buckets "$BUCKETS" \
      --output-dir "$run_dir" \
      --family-kind "$family_kind" \
      --executor-name llm_bench \
      --schedule-mode trap_switch \
      --trap-switch-cycle-source dataset \
      --switch-denominator 3 \
      --repeats 3 \
      --theta-eta 0.3 \
      --mechanisms "$method"
  else
    echo "unknown runner_type=${runner_type}" >&2
    exit 1
  fi

  validate_setup "$run_dir" "$runner_type" "$method" "$family_kind" "$ps_eta_shared"

  runner_session="$(runner_session_for_label "$label")"
  wandb_session="$(wandb_session_for_label "$label")"
  tmux kill-session -t "$runner_session" 2>/dev/null || true
  tmux kill-session -t "$wandb_session" 2>/dev/null || true

  if [[ "$runner_type" == "repeated" ]]; then
    tmux new-session -d -s "$runner_session" -n "$method" \
      "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/run_shared_basin_repeated_smoke.py run-method --run-dir '${run_dir}' --method '${method}' 2>&1 | tee -a '${run_dir}/${method}/tmux_runner.log'"
  else
    tmux new-session -d -s "$runner_session" -n "$method" \
      "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/run_shared_basin_mechanism_repeated_smoke.py run-mechanism --run-dir '${run_dir}' --mechanism '${method}' 2>&1 | tee -a '${run_dir}/${method}/tmux_runner.log'"
  fi

  tmux new-session -d -s "$wandb_session" -n wandb \
    "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/live_wandb_partial_uploader.py --run-dir '${run_dir}' --project '${WANDB_PROJECT}' --entity '${WANDB_ENTITY}' --run-group '${WANDB_GROUP}' --run-name-prefix '${EXPERIMENT_NAME}_${label}_' --run-id-suffix 'analysis_${label}' --methods '${method}' --poll-seconds 20 --finish-when-complete 2>&1 | tee -a '${run_dir}/live_wandb_uploader_analysis_${label}.log'"

  echo "${label}"
  echo "${run_dir}"
  echo "${runner_session}"
  echo "${wandb_session}"
done

tmux kill-session -t "$WATCHDOG_SESSION" 2>/dev/null || true
tmux new-session -d -s "$WATCHDOG_SESSION" -n watchdog \
  "cd '$ROOT_DIR' && source '$ENV_FILE' && scripts/watch_trapasymv4_binary_mixed_stage45_full_d3_4of5_11conditions_seed.sh '${SEED}' 2>&1 | tee -a '${GROUP_DIR}/watchdog.stdout'"

cat > "${GROUP_DIR}/experiment_manifest.json" <<EOF
{
  "experiment_name": "${EXPERIMENT_NAME}",
  "wandb_group": "${WANDB_GROUP}",
  "seed": ${SEED},
  "dataset": "${DATASET}",
  "schedule_buckets": "${BUCKETS}",
  "schedule_mode": "trap_switch",
  "trap_switch_cycle_source": "dataset",
  "switch_denominator": 3,
  "repeats": 3,
  "horizon": 300,
  "switch_episode": 100,
  "base_4of5_family": "${BASE_4OF5_FAMILY}",
  "partial_share_ps_eta_shared": 0.05,
  "all_share_leafratio_ps_eta_shared": 0.15
}
EOF

echo "experiment_name=${EXPERIMENT_NAME}"
echo "group_dir=${GROUP_DIR}"
echo "wandb_group=${WANDB_GROUP}"
echo "watchdog_session=${WATCHDOG_SESSION}"
