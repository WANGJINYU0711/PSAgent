#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEED="${1:-1}"
EXPERIMENT_NAME="llm_v8_trapasymv3efficientanchor4of5_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_d2_activegapv2_nomodecost_reason135_eps001_10x10_mechanisms_seed${SEED}"
WANDB_GROUP="trapasymv3_d2_activegapv2_nomodecost_mechanisms_seed${SEED}"
RUN_DIR="tmp/${EXPERIMENT_NAME}"
MECHANISM_SESSION="psagent_trapasymv3_d2_activegapv2_nomodecost_seed${SEED}_mech"
LIVE_SESSION="psagent_trapasymv3_d2_activegapv2_nomodecost_seed${SEED}_mech_wandb"
MERGE_SESSION="psagent_trapasymv3_d2_activegapv2_nomodecost_seed${SEED}_mech_merge"
ENV_FILE="scripts/trapasymv3_d2_activegapv2_nomodecost_seed${SEED}_env.sh"

export PSAGENT_LLM_BENCH_MODEL="gpt-4o-mini"
export PSAGENT_REPEATED_SMOKE_SEED="${SEED}"
export PSAGENT_TELECOM_EXEC_CLEAN_TERMINAL_V4="1"
unset PSAGENT_TELECOM_MODE_MISMATCH_COST_V2
export PSAGENT_TELECOM_MODE_MISMATCH_REPORT_ONLY_V2="1"
export PSAGENT_TELECOM_MODE_MISMATCH_FAST_ON_DEEP_COST="1.5"
export PSAGENT_TELECOM_MODE_MISMATCH_DEEP_ON_FAST_COST="1.75"
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
unset PSAGENT_TELECOM_MODE_MISMATCH_COST_V2
export PSAGENT_TELECOM_MODE_MISMATCH_REPORT_ONLY_V2="1"
export PSAGENT_TELECOM_MODE_MISMATCH_FAST_ON_DEEP_COST="1.5"
export PSAGENT_TELECOM_MODE_MISMATCH_DEEP_ON_FAST_COST="1.75"
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

python scripts/run_shared_basin_mechanism_repeated_smoke.py setup \
  --data data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v2_100/tasks.json \
  --schedule-buckets analysis/shared_basin_prefix_dedup_profile_switch_local_exec_clean_v2_100_smoke10_schedule_buckets.json \
  --output-dir "${RUN_DIR}" \
  --family-kind shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_4of5 \
  --executor-name llm_bench \
  --schedule-mode trap_switch \
  --switch-denominator 2 \
  --repeats 10 \
  --theta-eta 0.3 \
  --mechanisms theta_guided_agent agent_only

tmux kill-session -t "${MECHANISM_SESSION}" 2>/dev/null || true
tmux kill-session -t "${LIVE_SESSION}" 2>/dev/null || true
tmux kill-session -t "${MERGE_SESSION}" 2>/dev/null || true
mkdir -p "${RUN_DIR}/theta_guided_agent" "${RUN_DIR}/agent_only"

tmux new-session -d -s "${MECHANISM_SESSION}" -n theta \
  "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/run_shared_basin_mechanism_repeated_smoke.py run-mechanism --run-dir '${RUN_DIR}' --mechanism theta_guided_agent 2>&1 | tee '${RUN_DIR}/theta_guided_agent/tmux_runner.log'"
tmux new-window -t "${MECHANISM_SESSION}" -n agent_only \
  "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/run_shared_basin_mechanism_repeated_smoke.py run-mechanism --run-dir '${RUN_DIR}' --mechanism agent_only 2>&1 | tee '${RUN_DIR}/agent_only/tmux_runner.log'"

tmux new-session -d -s "${LIVE_SESSION}" -n wandb \
  "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/live_wandb_partial_uploader.py --run-dir '${RUN_DIR}' --project psagent-llm-smoke --entity wangjinyu0711-microsoft --run-group '${WANDB_GROUP}' --run-name-prefix '${EXPERIMENT_NAME}_' --run-id-suffix analysis_d2_activegapv2_nomodecost_mechanisms --methods theta_guided_agent agent_only --poll-seconds 20 --finish-when-complete 2>&1 | tee '${RUN_DIR}/live_wandb_uploader_analysis_d2_activegapv2_nomodecost_mechanisms.log'"

tmux new-session -d -s "${MERGE_SESSION}" -n merge \
  "cd '$ROOT_DIR' && source '$ENV_FILE' && while [ ! -f '${RUN_DIR}/theta_guided_agent/summary.json' ] || [ ! -f '${RUN_DIR}/agent_only/summary.json' ]; do date; sleep 30; done; python scripts/run_shared_basin_mechanism_repeated_smoke.py merge-all --run-dir '${RUN_DIR}' 2>&1 | tee '${RUN_DIR}/merge_all.log'"

echo "${EXPERIMENT_NAME}"
echo "${RUN_DIR}"
echo "${MECHANISM_SESSION}"
echo "${LIVE_SESSION}"
echo "${MERGE_SESSION}"
