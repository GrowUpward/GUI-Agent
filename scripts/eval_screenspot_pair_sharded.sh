#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "${SCRIPT_DIR}/_common.sh"
require_env BASE_MODEL
require_env DATASET_ROOT
require_env EVAL_LABEL
require_env OUTPUT_DIR

NUM_SHARDS=${NUM_SHARDS:-4}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
mkdir -p "${OUTPUT_DIR}"
: > "${OUTPUT_DIR}/pids.txt"

ADAPTER_ARGS=()
[ -n "${EVAL_ADAPTER:-}" ] && ADAPTER_ARGS+=(--adapter_path "${EVAL_ADAPTER}")
pids=()

launch() {
  local dataset_name=$1
  local data_json=$2
  local image_dir=$3
  local shard=$4
  local gpu=$5
  local prefix=$6
  CUDA_VISIBLE_DEVICES=${gpu} nohup "${PYTHON}" -u -m gui_agent.evaluation.screenspot \
    --model_name_or_path "${BASE_MODEL}" \
    "${ADAPTER_ARGS[@]}" \
    --data_json "${data_json}" \
    --image_dir "${image_dir}" \
    --output_file "${OUTPUT_DIR}/${prefix}-shard${shard}.json" \
    --label "${EVAL_LABEL}-${dataset_name}-shard${shard}" \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --num_shards "${NUM_SHARDS}" \
    --shard_index "${shard}" \
    --max_image_side "${MAX_IMAGE_SIDE:-448}" \
    --max_new_tokens "${EVAL_MAX_NEW_TOKENS:-48}" \
    --bf16 > "${OUTPUT_DIR}/${prefix}-shard${shard}.log" 2>&1 < /dev/null &
  pids+=("$!")
  echo "$! ${dataset_name} ${shard} ${gpu}" >> "${OUTPUT_DIR}/pids.txt"
}

for ((shard=0; shard<NUM_SHARDS; shard++)); do
  launch screenspot \
    "${DATASET_ROOT}/ScreenSpot/screenspot_test.json" \
    "${DATASET_ROOT}/ScreenSpot/eval_images" \
    "${shard}" "$((shard % 2))" screenspot
  launch screenspotv2 \
    "${DATASET_ROOT}/ScreenSpotV2/screenspotv2_test.json" \
    "${DATASET_ROOT}/ScreenSpotV2/eval_images" \
    "${shard}" "$(((shard + 1) % 2))" screenspotv2
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
if [ "${failed}" -ne 0 ]; then
  echo "At least one ScreenSpot shard failed; inspect ${OUTPUT_DIR}/*.log" >&2
  exit 1
fi

screenspot_inputs=()
screenspotv2_inputs=()
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  screenspot_inputs+=("${OUTPUT_DIR}/screenspot-shard${shard}.json")
  screenspotv2_inputs+=("${OUTPUT_DIR}/screenspotv2-shard${shard}.json")
done
"${PYTHON}" -m gui_agent.evaluation.merge_screenspot \
  --inputs "${screenspot_inputs[@]}" \
  --output_file "${OUTPUT_DIR}/screenspot.json" \
  --label "${EVAL_LABEL}-screenspot"
"${PYTHON}" -m gui_agent.evaluation.merge_screenspot \
  --inputs "${screenspotv2_inputs[@]}" \
  --output_file "${OUTPUT_DIR}/screenspotv2.json" \
  --label "${EVAL_LABEL}-screenspotv2"
