#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "${SCRIPT_DIR}/_common.sh"
require_env BASE_MODEL
require_env CAGUI_ROOT
require_env SPLIT_MANIFEST
require_env E1_ADAPTER
require_env E2_ADAPTER
require_env OUTPUT_DIR

NUM_SHARDS=${NUM_SHARDS:-4}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
EXTRA_ARGS=()
[ "${BF16:-1}" = 1 ] && EXTRA_ARGS+=(--bf16)
[ "${LOAD_IN_4BIT:-1}" = 1 ] && EXTRA_ARGS+=(--load_in_4bit)
mkdir -p "${OUTPUT_DIR}/e1" "${OUTPUT_DIR}/e2"
: > "${OUTPUT_DIR}/pids.txt"
pids=()

launch() {
  local label=$1
  local adapter=$2
  local gpu=$3
  local shard=$4
  CUDA_VISIBLE_DEVICES=${gpu} nohup "${PYTHON}" -u -m gui_agent.evaluation.cagui_offline \
    --model_name_or_path "${BASE_MODEL}" \
    --adapter_path "${adapter}" \
    --dataset_dir "${CAGUI_ROOT}" \
    --split_manifest "${SPLIT_MANIFEST}" \
    --output_file "${OUTPUT_DIR}/${label}/shard${shard}.json" \
    --label "${label}-validation-shard${shard}" \
    --partition validation \
    --history_window "${HISTORY_WINDOW:-4}" \
    --max_ui_boxes "${MAX_UI_BOXES:-30}" \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --num_shards "${NUM_SHARDS}" \
    --shard_index "${shard}" \
    --max_image_side "${MAX_IMAGE_SIDE:-448}" \
    --max_new_tokens "${EVAL_MAX_NEW_TOKENS:-48}" \
    "${EXTRA_ARGS[@]}" > "${OUTPUT_DIR}/${label}/shard${shard}.log" 2>&1 < /dev/null &
  pids+=("$!")
  echo "$! ${label} ${shard} ${gpu}" >> "${OUTPUT_DIR}/pids.txt"
}

for ((shard=0; shard<NUM_SHARDS; shard++)); do
  launch e1 "${E1_ADAPTER}" 0 "${shard}"
  launch e2 "${E2_ADAPTER}" 1 "${shard}"
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
if [ "${failed}" -ne 0 ]; then
  echo "At least one CAGUI evaluation shard failed; inspect ${OUTPUT_DIR}/*/*.log" >&2
  exit 1
fi

for label in e1 e2; do
  inputs=()
  for ((shard=0; shard<NUM_SHARDS; shard++)); do
    inputs+=("${OUTPUT_DIR}/${label}/shard${shard}.json")
  done
  "${PYTHON}" -m gui_agent.evaluation.merge_cagui_offline \
    --inputs "${inputs[@]}" \
    --output_file "${OUTPUT_DIR}/${label}/validation.json" \
    --label "${label}-validation-agentcpm-compatible"
done
