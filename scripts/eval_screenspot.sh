#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "${SCRIPT_DIR}/_common.sh"
require_env BASE_MODEL
require_env SCREENSPOT_JSON
require_env SCREENSPOT_IMAGE_DIR
require_env EVAL_LABEL
require_env OUTPUT_FILE

EXTRA_ARGS=()
[ "${BF16:-1}" = 1 ] && EXTRA_ARGS+=(--bf16)
[ "${LOAD_IN_4BIT:-0}" = 1 ] && EXTRA_ARGS+=(--load_in_4bit)

exec "${PYTHON}" -u -m gui_agent.evaluation.screenspot \
  --model_name_or_path "${BASE_MODEL}" \
  --adapter_path "${EVAL_ADAPTER:-}" \
  --data_json "${SCREENSPOT_JSON}" \
  --image_dir "${SCREENSPOT_IMAGE_DIR}" \
  --output_file "${OUTPUT_FILE}" \
  --label "${EVAL_LABEL}" \
  --batch_size "${EVAL_BATCH_SIZE:-8}" \
  --max_image_side "${MAX_IMAGE_SIDE:-448}" \
  --max_new_tokens "${EVAL_MAX_NEW_TOKENS:-48}" \
  --max_samples "${MAX_TEST_SAMPLES:-0}" \
  --num_shards "${NUM_SHARDS:-1}" \
  --shard_index "${SHARD_INDEX:-0}" \
  --seed "${SEED:-42}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
