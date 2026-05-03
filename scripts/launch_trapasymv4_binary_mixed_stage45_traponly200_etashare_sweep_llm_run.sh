#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEED="${1:-1}"
WANDB_PROJECT="psagent-llm-smoke"
WANDB_ENTITY="wangjinyu0711-microsoft"
WANDB_GROUP="trapasymv4_binarymixed_traponly200_eta03_eps001_etashare_sweep010_015_020_seed${SEED}"
ENV_FILE="scripts/trapasymv4_binarymixed_traponly200_etashare_sweep_seed${SEED}_env.sh"

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

LABELS=(
  "base_exp_e015"
  "base_eps_e015"
  "base_ps_e015"
  "allshare_ps_e015"
  "allunshare_ps_e015"
  "allshare_ps_e010"
  "allshare_ps_e020"
)
METHODS=(
  "direct_multistage_exp3"
  "epsilon_exp3"
  "risky_ps"
  "risky_ps"
  "risky_ps"
  "risky_ps"
  "risky_ps"
)
FAMILY_KINDS=(
  "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45"
  "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45"
  "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45"
  "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45_all_share"
  "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45_all_unshare"
  "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45_all_share"
  "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45_all_share"
)
ETA_SHARED_VALUES=(
  "0.15"
  "0.15"
  "0.15"
  "0.15"
  "0.15"
  "0.10"
  "0.20"
)

for index in "${!LABELS[@]}"; do
  label="${LABELS[$index]}"
  method="${METHODS[$index]}"
  family_kind="${FAMILY_KINDS[$index]}"
  eta_shared="${ETA_SHARED_VALUES[$index]}"
  eta_tag="${eta_shared/./}"
  experiment_name="llm_v8_trapasymv4binarymixed_${label}_traponly200_stage45_contract_promptv11b_stage45_gpt41mini_cleanv3_eta03_eps001_etashare${eta_tag}_seed${SEED}"
  run_dir="tmp/${experiment_name}"
  method_session="psagent_trapasymv4_esweep_${label}_seed${SEED}_runner"
  live_session="psagent_trapasymv4_esweep_${label}_seed${SEED}_live_wandb"

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
    --ps-eta-shared-override "${eta_shared}" \
    --methods "${method}"

  tmux kill-session -t "${method_session}" 2>/dev/null || true
  tmux kill-session -t "${live_session}" 2>/dev/null || true
  mkdir -p "${run_dir}/${method}"

  tmux new-session -d -s "${method_session}" -n "${method}" \
    "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/run_shared_basin_repeated_smoke.py run-method --run-dir '${run_dir}' --method '${method}' 2>&1 | tee '${run_dir}/${method}/tmux_runner.log'"

  tmux new-session -d -s "${live_session}" -n wandb \
    "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/live_wandb_partial_uploader.py --run-dir '${run_dir}' --project '${WANDB_PROJECT}' --entity '${WANDB_ENTITY}' --run-group '${WANDB_GROUP}' --run-name-prefix '${experiment_name}_' --run-id-suffix 'analysis_${label}_traponly200' --methods '${method}' --poll-seconds 20 --finish-when-complete 2>&1 | tee '${run_dir}/live_wandb_uploader_analysis_${label}_traponly200.log'"

  echo "${label}"
  echo "${run_dir}"
  echo "${method_session}"
  echo "${live_session}"
done

MECH_LABEL="mechanisms_e015"
MECH_EXPERIMENT_NAME="llm_v8_trapasymv4binarymixed_${MECH_LABEL}_traponly200_stage45_contract_promptv11b_stage45_gpt41mini_cleanv3_eta03_seed${SEED}"
MECH_RUN_DIR="tmp/${MECH_EXPERIMENT_NAME}"
MECH_RUNNER_SESSION="psagent_trapasymv4_esweep_${MECH_LABEL}_seed${SEED}_runner"
MECH_LIVE_SESSION="psagent_trapasymv4_esweep_${MECH_LABEL}_seed${SEED}_live_wandb"
MECH_MERGE_SESSION="psagent_trapasymv4_esweep_${MECH_LABEL}_seed${SEED}_merge"

python scripts/run_shared_basin_mechanism_repeated_smoke.py setup \
  --data "${DATASET}" \
  --schedule-buckets "${BUCKETS}" \
  --output-dir "${MECH_RUN_DIR}" \
  --family-kind shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45 \
  --executor-name llm_bench \
  --schedule-mode trap_only_random \
  --switch-denominator 1 \
  --repeats 200 \
  --theta-eta 0.3 \
  --mechanisms theta_guided_agent agent_only

tmux kill-session -t "${MECH_RUNNER_SESSION}" 2>/dev/null || true
tmux kill-session -t "${MECH_LIVE_SESSION}" 2>/dev/null || true
tmux kill-session -t "${MECH_MERGE_SESSION}" 2>/dev/null || true
mkdir -p "${MECH_RUN_DIR}/theta_guided_agent" "${MECH_RUN_DIR}/agent_only"

tmux new-session -d -s "${MECH_RUNNER_SESSION}" -n theta \
  "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/run_shared_basin_mechanism_repeated_smoke.py run-mechanism --run-dir '${MECH_RUN_DIR}' --mechanism theta_guided_agent 2>&1 | tee -a '${MECH_RUN_DIR}/theta_guided_agent/tmux_runner.log'"
tmux new-window -t "${MECH_RUNNER_SESSION}" -n agent_only \
  "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/run_shared_basin_mechanism_repeated_smoke.py run-mechanism --run-dir '${MECH_RUN_DIR}' --mechanism agent_only 2>&1 | tee -a '${MECH_RUN_DIR}/agent_only/tmux_runner.log'"

tmux new-session -d -s "${MECH_LIVE_SESSION}" -n wandb \
  "cd '$ROOT_DIR' && source '$ENV_FILE' && python scripts/live_wandb_partial_uploader.py --run-dir '${MECH_RUN_DIR}' --project '${WANDB_PROJECT}' --entity '${WANDB_ENTITY}' --run-group '${WANDB_GROUP}' --run-name-prefix '${MECH_EXPERIMENT_NAME}_' --run-id-suffix analysis_${MECH_LABEL}_traponly200 --methods theta_guided_agent agent_only --poll-seconds 20 --finish-when-complete 2>&1 | tee -a '${MECH_RUN_DIR}/live_wandb_uploader_analysis_${MECH_LABEL}_traponly200.log'"

tmux new-session -d -s "${MECH_MERGE_SESSION}" -n merge \
  "cd '$ROOT_DIR' && source '$ENV_FILE' && while [ ! -f '${MECH_RUN_DIR}/theta_guided_agent/summary.json' ] || [ ! -f '${MECH_RUN_DIR}/agent_only/summary.json' ]; do date; sleep 30; done; python scripts/run_shared_basin_mechanism_repeated_smoke.py merge-all --run-dir '${MECH_RUN_DIR}' 2>&1 | tee -a '${MECH_RUN_DIR}/merge_all.log'"

echo "${MECH_LABEL}"
echo "${MECH_RUN_DIR}"
echo "${MECH_RUNNER_SESSION}"
echo "${MECH_LIVE_SESSION}"
echo "${MECH_MERGE_SESSION}"
