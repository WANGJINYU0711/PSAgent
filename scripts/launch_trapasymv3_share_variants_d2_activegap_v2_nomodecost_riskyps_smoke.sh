#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEED="${1:-1}"
METHOD="risky_ps"
WANDB_PROJECT="psagent-llm-smoke"
WANDB_ENTITY="wangjinyu0711-microsoft"
WANDB_GROUP="trapasymv3_sharevariants_d2_activegapv2_nomodecost_riskyps_seed${SEED}"
ENV_FILE="scripts/trapasymv3_sharevariants_d2_activegapv2_nomodecost_riskyps_seed${SEED}_env.sh"

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

VARIANT_KEYS=(
  "allshare"
  "2of5"
  "allunshare"
)
VARIANT_LABELS=(
  "allshare"
  "2of5share"
  "allunshare"
)
FAMILY_KINDS=(
  "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_all_share"
  "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_2of5"
  "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_all_unshare"
)

for index in "${!VARIANT_KEYS[@]}"; do
  key="${VARIANT_KEYS[$index]}"
  label="${VARIANT_LABELS[$index]}"
  family_kind="${FAMILY_KINDS[$index]}"
  experiment_name="llm_v8_trapasymv3efficientanchor${label}_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_d2_activegapv2_nomodecost_reason135_eps001_10x10_riskyps_seed${SEED}"
  run_dir="tmp/${experiment_name}"
  method_session="psagent_trapasymv3_${key}_d2_activegapv2_nomodecost_seed${SEED}_riskyps"
  live_session="psagent_trapasymv3_${key}_d2_activegapv2_nomodecost_seed${SEED}_live_wandb"

  python scripts/run_shared_basin_repeated_smoke.py setup \
    --data data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v2_100/tasks.json \
    --schedule-buckets analysis/shared_basin_prefix_dedup_profile_switch_local_exec_clean_v2_100_smoke10_schedule_buckets.json \
    --output-dir "${run_dir}" \
    --family-kind "${family_kind}" \
    --executor-name llm_bench \
    --schedule-mode trap_switch \
    --switch-denominator 2 \
    --repeats 10 \
    --common-eta-override 0.3 \
    --common-epsilon-override 0.01 \
    --methods "${METHOD}"

  tmux kill-session -t "${method_session}" 2>/dev/null || true
  tmux kill-session -t "${live_session}" 2>/dev/null || true
  mkdir -p "${run_dir}/${METHOD}"

  tmux new-session -d -s "${method_session}" -n "${METHOD}" \
    "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/run_shared_basin_repeated_smoke.py run-method --run-dir '${run_dir}' --method '${METHOD}' 2>&1 | tee '${run_dir}/${METHOD}/tmux_runner.log'"

  tmux new-session -d -s "${live_session}" -n wandb \
    "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/live_wandb_partial_uploader.py --run-dir '${run_dir}' --project '${WANDB_PROJECT}' --entity '${WANDB_ENTITY}' --run-group '${WANDB_GROUP}' --run-name-prefix '${experiment_name}_' --run-id-suffix 'analysis_${key}_d2_activegapv2_nomodecost_riskyps' --methods '${METHOD}' --poll-seconds 20 --finish-when-complete 2>&1 | tee '${run_dir}/live_wandb_uploader_analysis_${key}_d2_activegapv2_nomodecost_riskyps.log'"

  echo "${label}"
  echo "${run_dir}"
  echo "${method_session}"
  echo "${live_session}"
done
