#!/usr/bin/env bash
set -u

cd /home/ubuntu/data/PSAgent

export PSAGENT_LLM_BENCH_MODEL=gpt-4o-mini
export PSAGENT_TELECOM_EXEC_CLEAN_TERMINAL_V4=1
export PSAGENT_TELECOM_REASONING_WEIGHT_CALIBRATION_V3=1
export PSAGENT_TELECOM_MODE_MISMATCH_REPORT_ONLY_V2=1
unset PSAGENT_TELECOM_MODE_MISMATCH_COST_V2

DATA="data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v2_100/tasks.json"
BUCKETS="analysis/shared_basin_prefix_dedup_profile_switch_local_exec_clean_v2_100_smoke10_schedule_buckets.json"
BASE="tmp/llm_v8_ps_update_stability_cconfig_d4_eta03_eps001_10x10"

run_one() {
  local seed="$1"
  local variant="$2"
  shift 2
  local out="${BASE}_seed${seed}_${variant}"
  local log="${out}_orchestrate.log"
  mkdir -p "$out"
  {
    echo "[start] $(date -Is) seed=${seed} variant=${variant} out=${out}"
    PSAGENT_REPEATED_SMOKE_SEED="$seed" \
      python scripts/run_shared_basin_repeated_smoke.py orchestrate \
        --data "$DATA" \
        --output-dir "$out" \
        --repeats 10 \
        --family-kind shared_basin_strong_prefix_dedup_profile_switch \
        --schedule-mode trap_switch \
        --switch-denominator 4 \
        --schedule-buckets "$BUCKETS" \
        --common-eta-override 0.3 \
        --common-epsilon-override 0.01 \
        --executor-name llm_bench \
        --methods risky_ps \
        "$@"
    rc=$?
    echo "[orchestrate-exit] $(date -Is) seed=${seed} variant=${variant} rc=${rc}"
    if [ "$rc" -eq 0 ]; then
      python scripts/analyze_llm_repeated_smoke_modes.py \
        --run-dir "$out" \
        --title "llm_v8 ps update stability seed${seed} ${variant}"
      echo "[analysis-exit] $(date -Is) seed=${seed} variant=${variant} rc=$?"
    fi
    exit "$rc"
  } >"$log" 2>&1
}

run_one 0 eta_shared002 --ps-eta-shared-override 0.02 &
run_one 0 clip100 --ps-loss-clip 100 &
run_one 0 probfloor0002 --ps-prob-floor 0.002 &
run_one 1 eta_shared002 --ps-eta-shared-override 0.02 &
run_one 1 clip100 --ps-loss-clip 100 &
run_one 1 probfloor0002 --ps-prob-floor 0.002 &

wait
echo "[all-done] $(date -Is)"
