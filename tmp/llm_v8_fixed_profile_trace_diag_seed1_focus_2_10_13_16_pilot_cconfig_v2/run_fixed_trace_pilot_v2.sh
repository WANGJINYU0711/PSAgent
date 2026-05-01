#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/data/PSAgent
export PSAGENT_LLM_BENCH_MODEL=gpt-4o-mini
export PSAGENT_TELECOM_EXEC_CLEAN_TERMINAL_V4=1
export PSAGENT_TELECOM_REASONING_WEIGHT_CALIBRATION_V3=1
export PSAGENT_TELECOM_MODE_MISMATCH_REPORT_ONLY_V2=1
python scripts/run_llm_fixed_profile_trace_diagnostic.py \
  --output-dir tmp/llm_v8_fixed_profile_trace_diag_seed1_focus_2_10_13_16_pilot_cconfig_v2 \
  --seed 1 \
  --model gpt-4o-mini \
  --dataset-indices 2 10 13 16 \
  --patterns fdddd ffddd ddddd \
  --repeats 1 \
  --parallelism 4 \
  --include-observed-high-terminal \
  --max-observed-paths 8
