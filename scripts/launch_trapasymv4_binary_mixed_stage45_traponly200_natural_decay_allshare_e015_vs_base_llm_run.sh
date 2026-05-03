#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEED="${1:-1}"
WANDB_PROJECT="psagent-llm-smoke"
WANDB_ENTITY="wangjinyu0711-microsoft"
WANDB_GROUP="trapasymv4_binarymixed_traponly200_natural_decay_allshare_e015_vs_base_eta03_eps001_seed${SEED}"
GROUP_NAME="llm_v8_trapasymv4binarymixed_natural_decay_allshare_e015_vs_direct_baseps_baseequalchild_traponly200_stage45_contract_promptv11b_stage45_gpt41mini_cleanv3_eta03_eps001_seed${SEED}"
GROUP_DIR="tmp/${GROUP_NAME}"
ENV_FILE="scripts/trapasymv4_binarymixed_traponly200_natural_decay_allshare_e015_vs_base_seed${SEED}_env.sh"
WATCHDOG_SESSION="psagent_trapasymv4_natural_decay_allshare_e015_vs_base_seed${SEED}_watchdog"

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

DATASET="data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v3_100/tasks.json"
BUCKETS="analysis/shared_basin_prefix_dedup_profile_switch_local_exec_clean_v3_100_schedule_buckets.json"

mkdir -p "$GROUP_DIR"

LABELS=(
  "allshare_equalchild_natural_ps_e015"
  "base_4of5_direct_exp_e015"
  "base_4of5_weight_ps_e015"
  "base_4of5_equalchild_ps_e015"
)
METHODS=(
  "risky_ps_const_init_natural_decay"
  "direct_multistage_exp3"
  "risky_ps"
  "risky_ps_const_init"
)
FAMILY_KINDS=(
  "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45_all_share"
  "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45"
  "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45"
  "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45"
)

for index in "${!LABELS[@]}"; do
  label="${LABELS[$index]}"
  method="${METHODS[$index]}"
  family_kind="${FAMILY_KINDS[$index]}"
  experiment_name="llm_v8_trapasymv4binarymixed_${label}_traponly200_stage45_contract_promptv11b_stage45_gpt41mini_cleanv3_eta03_eps001_etashare015_seed${SEED}"
  run_dir="${GROUP_DIR}/${label}"
  method_session="psagent_trapasymv4_natdecay_${label}_seed${SEED}_runner"
  live_session="psagent_trapasymv4_natdecay_${label}_seed${SEED}_live_wandb"

  python scripts/run_shared_basin_repeated_smoke.py setup \
    --data "${DATASET}" \
    --schedule-buckets "${BUCKETS}" \
    --output-dir "${run_dir}" \
    --family-kind "${family_kind}" \
    --executor-name llm_bench \
    --schedule-mode trap_only_random \
    --switch-denominator 1 \
    --repeats 200 \
    --common-eta-override 0.3 \
    --common-epsilon-override 0.01 \
    --ps-eta-shared-override 0.15 \
    --methods "${method}"

  tmux kill-session -t "${method_session}" 2>/dev/null || true
  tmux kill-session -t "${live_session}" 2>/dev/null || true
  mkdir -p "${run_dir}/${method}"

  tmux new-session -d -s "${method_session}" -n "${method}" \
    "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/run_shared_basin_repeated_smoke.py run-method --run-dir '${run_dir}' --method '${method}' 2>&1 | tee -a '${run_dir}/${method}/tmux_runner.log'"

  tmux new-session -d -s "${live_session}" -n wandb \
    "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/live_wandb_partial_uploader.py --run-dir '${run_dir}' --project '${WANDB_PROJECT}' --entity '${WANDB_ENTITY}' --run-group '${WANDB_GROUP}' --run-name-prefix '${experiment_name}_' --run-id-suffix 'analysis_${label}_traponly200' --methods '${method}' --poll-seconds 20 --finish-when-complete 2>&1 | tee -a '${run_dir}/live_wandb_uploader_analysis_${label}_traponly200.log'"

  echo "${label}"
  echo "${run_dir}"
  echo "${method_session}"
  echo "${live_session}"
done

tmux kill-session -t "${WATCHDOG_SESSION}" 2>/dev/null || true
tmux new-session -d -s "${WATCHDOG_SESSION}" -n watchdog \
  "cd '$ROOT_DIR' && source '$ENV_FILE' && scripts/watch_trapasymv4_binary_mixed_stage45_traponly200_natural_decay_allshare_e015_vs_base_seed.sh '${SEED}' 2>&1 | tee -a '${GROUP_DIR}/watchdog.stdout'"

echo "group_dir=${GROUP_DIR}"
echo "watchdog_session=${WATCHDOG_SESSION}"
