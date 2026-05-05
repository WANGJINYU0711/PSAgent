#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <seed>" >&2
  exit 2
fi

SEED="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EXPERIMENT_NAME="llm_v8_trapasymv3efficientanchor4of5_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_d4_eta03_eps001_10x10_4methods_seed${SEED}"
RUN_DIR="tmp/${EXPERIMENT_NAME}"
METHOD_SESSION="psagent_trapasymv3efficientanchor4of5_seed${SEED}_4m"
LIVE_SESSION="psagent_trapasymv3efficientanchor4of5_seed${SEED}_live_wandb"
EARLY_SESSION="psagent_trapasymv3efficientanchor4of5_seed${SEED}_earlystop"

export PSAGENT_LLM_BENCH_MODEL="gpt-4o-mini"
export PSAGENT_REPEATED_SMOKE_SEED="${SEED}"
export PSAGENT_TELECOM_EXEC_CLEAN_TERMINAL_V4="1"
export PSAGENT_TELECOM_REASONING_WEIGHT_CALIBRATION_V3="1"
export PSAGENT_TELECOM_STAGE45_CONTRACT_PROMPT_V1_1B="1"
export PSAGENT_TELECOM_STAGE45_MODEL="gpt-4.1-mini"
export PSAGENT_TELECOM_LLM_BRIDGE_RETRY_ATTEMPTS="${PSAGENT_TELECOM_LLM_BRIDGE_RETRY_ATTEMPTS:-4}"
export PSAGENT_TELECOM_LLM_BRIDGE_RETRY_SLEEP_SECONDS="${PSAGENT_TELECOM_LLM_BRIDGE_RETRY_SLEEP_SECONDS:-20}"

cat > "scripts/trapasymv3_seed${SEED}_env.sh" <<EOF
export PSAGENT_LLM_BENCH_MODEL="gpt-4o-mini"
export PSAGENT_REPEATED_SMOKE_SEED="${SEED}"
export PSAGENT_TELECOM_EXEC_CLEAN_TERMINAL_V4="1"
export PSAGENT_TELECOM_REASONING_WEIGHT_CALIBRATION_V3="1"
export PSAGENT_TELECOM_STAGE45_CONTRACT_PROMPT_V1_1B="1"
export PSAGENT_TELECOM_STAGE45_MODEL="gpt-4.1-mini"
export PSAGENT_TELECOM_LLM_BRIDGE_RETRY_ATTEMPTS="${PSAGENT_TELECOM_LLM_BRIDGE_RETRY_ATTEMPTS}"
export PSAGENT_TELECOM_LLM_BRIDGE_RETRY_SLEEP_SECONDS="${PSAGENT_TELECOM_LLM_BRIDGE_RETRY_SLEEP_SECONDS}"
EOF

python scripts/run_shared_basin_repeated_smoke.py setup \
  --data data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v2_100/tasks.json \
  --schedule-buckets analysis/shared_basin_prefix_dedup_profile_switch_local_exec_clean_v2_100_smoke10_schedule_buckets.json \
  --output-dir "${RUN_DIR}" \
  --family-kind shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_4of5 \
  --executor-name llm_bench \
  --schedule-mode trap_switch \
  --switch-denominator 4 \
  --repeats 10 \
  --common-eta-override 0.3 \
  --common-epsilon-override 0.01 \
  --methods risky_ps direct_multistage_exp3 epsilon_exp3 risky_ps_old

tmux kill-session -t "${METHOD_SESSION}" 2>/dev/null || true
tmux kill-session -t "${LIVE_SESSION}" 2>/dev/null || true
tmux kill-session -t "${EARLY_SESSION}" 2>/dev/null || true
mkdir -p \
  "${RUN_DIR}/risky_ps" \
  "${RUN_DIR}/direct_multistage_exp3" \
  "${RUN_DIR}/epsilon_exp3" \
  "${RUN_DIR}/risky_ps_old"

tmux new-session -d -s "${METHOD_SESSION}" -n risky_ps \
  "cd '$ROOT_DIR' && source scripts/trapasymv3_seed${SEED}_env.sh && python scripts/run_shared_basin_repeated_smoke.py run-method --run-dir '${RUN_DIR}' --method risky_ps 2>&1 | tee '${RUN_DIR}/risky_ps/tmux_runner.log'"
tmux new-window -t "${METHOD_SESSION}" -n direct \
  "cd '$ROOT_DIR' && source scripts/trapasymv3_seed${SEED}_env.sh && python scripts/run_shared_basin_repeated_smoke.py run-method --run-dir '${RUN_DIR}' --method direct_multistage_exp3 2>&1 | tee '${RUN_DIR}/direct_multistage_exp3/tmux_runner.log'"
tmux new-window -t "${METHOD_SESSION}" -n epsilon \
  "cd '$ROOT_DIR' && source scripts/trapasymv3_seed${SEED}_env.sh && python scripts/run_shared_basin_repeated_smoke.py run-method --run-dir '${RUN_DIR}' --method epsilon_exp3 2>&1 | tee '${RUN_DIR}/epsilon_exp3/tmux_runner.log'"
tmux new-window -t "${METHOD_SESSION}" -n risky_old \
  "cd '$ROOT_DIR' && source scripts/trapasymv3_seed${SEED}_env.sh && python scripts/run_shared_basin_repeated_smoke.py run-method --run-dir '${RUN_DIR}' --method risky_ps_old 2>&1 | tee '${RUN_DIR}/risky_ps_old/tmux_runner.log'"

tmux new-session -d -s "${LIVE_SESSION}" \
  "cd '$ROOT_DIR' && source scripts/trapasymv3_seed${SEED}_env.sh && python scripts/live_wandb_partial_uploader.py --run-dir '${RUN_DIR}' --project psagent-llm-smoke --entity wangjinyu0711-microsoft --run-group '${EXPERIMENT_NAME}' --run-name-prefix '${EXPERIMENT_NAME}_' --run-id-suffix analysis_v3 --methods risky_ps direct_multistage_exp3 epsilon_exp3 risky_ps_old --poll-seconds 30 --finish-when-complete 2>&1 | tee '${RUN_DIR}/live_wandb_uploader_analysis_v3.log'"

tmux new-session -d -s "${EARLY_SESSION}" \
  "cd '$ROOT_DIR' && source scripts/trapasymv3_seed${SEED}_env.sh && python scripts/monitor_repeated_smoke_early_stop.py --run-dir '${RUN_DIR}' --ps-method risky_ps --baseline-methods direct_multistage_exp3 epsilon_exp3 --min-episodes 75 --raw-gap-threshold 0.5 --tmux-session '${METHOD_SESSION}' --poll-seconds 60 2>&1 | tee '${RUN_DIR}/early_stop_monitor.stdout'"

echo "${EXPERIMENT_NAME}"
echo "${RUN_DIR}"
echo "${METHOD_SESSION}"
echo "${LIVE_SESSION}"
echo "${EARLY_SESSION}"
