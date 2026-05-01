#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEED="${1:-1}"
EXPERIMENT_NAME="llm_v8_trapasymv3efficientanchor4of5_stage45_contract_promptv11b_stage45_gpt41mini_full_d4_nomodecost_reason135_eps001_3x100_v3_11methods_seed${SEED}"
WANDB_GROUP="trapasymv3_full_d4_nomodecost_seed${SEED}"
RUN_DIR="tmp/${EXPERIMENT_NAME}"
METHOD_SESSION="psagent_trapasymv3_full_d4_nomodecost_seed${SEED}_11m"
LIVE_SESSION="psagent_trapasymv3_full_d4_nomodecost_seed${SEED}_live_wandb"
ENV_FILE="scripts/trapasymv3_full_d4_nomodecost_seed${SEED}_env.sh"

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
export PSAGENT_TELECOM_LLM_BRIDGE_RETRY_ATTEMPTS="${PSAGENT_TELECOM_LLM_BRIDGE_RETRY_ATTEMPTS:-4}"
export PSAGENT_TELECOM_LLM_BRIDGE_RETRY_SLEEP_SECONDS="${PSAGENT_TELECOM_LLM_BRIDGE_RETRY_SLEEP_SECONDS:-20}"

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
EOF

python scripts/run_shared_basin_repeated_smoke.py setup \
  --data data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v3_100/tasks.json \
  --schedule-buckets analysis/shared_basin_prefix_dedup_profile_switch_local_exec_clean_v3_100_schedule_buckets.json \
  --output-dir "${RUN_DIR}" \
  --family-kind shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_4of5 \
  --executor-name llm_bench \
  --schedule-mode trap_switch \
  --switch-denominator 4 \
  --trap-switch-cycle-source dataset \
  --repeats 3 \
  --common-eta-override 0.3 \
  --common-epsilon-override 0.01 \
  --methods "${METHODS[@]}"

tmux kill-session -t "${METHOD_SESSION}" 2>/dev/null || true
tmux kill-session -t "${LIVE_SESSION}" 2>/dev/null || true

for method in "${METHODS[@]}"; do
  mkdir -p "${RUN_DIR}/${method}"
done

first_method="${METHODS[0]}"
tmux new-session -d -s "${METHOD_SESSION}" -n "${first_method}" \
  "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/run_shared_basin_repeated_smoke.py run-method --run-dir '${RUN_DIR}' --method '${first_method}' 2>&1 | tee '${RUN_DIR}/${first_method}/tmux_runner.log'"

for method in "${METHODS[@]:1}"; do
  tmux new-window -t "${METHOD_SESSION}" -n "${method}" \
    "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/run_shared_basin_repeated_smoke.py run-method --run-dir '${RUN_DIR}' --method '${method}' 2>&1 | tee '${RUN_DIR}/${method}/tmux_runner.log'"
done

tmux new-session -d -s "${LIVE_SESSION}" \
  "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/live_wandb_partial_uploader.py --run-dir '${RUN_DIR}' --project psagent-llm-smoke --entity wangjinyu0711-microsoft --run-group '${WANDB_GROUP}' --run-name-prefix '${EXPERIMENT_NAME}_' --run-id-suffix analysis_full_d4_nomodecost --methods ${METHODS[*]} --poll-seconds 20 --finish-when-complete 2>&1 | tee '${RUN_DIR}/live_wandb_uploader_analysis_full_d4_nomodecost.log'"

echo "${EXPERIMENT_NAME}"
echo "${RUN_DIR}"
echo "${METHOD_SESSION}"
echo "${LIVE_SESSION}"
