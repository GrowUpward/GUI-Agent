#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "${SCRIPT_DIR}/_common.sh"
require_env BASE_MODEL
require_env CAGUI_ROOT
require_env EVAL_LABEL
require_env OUTPUT_FILE

EXTRA_ARGS=()
[ "${BF16:-1}" = 1 ] && EXTRA_ARGS+=(--bf16)
[ "${LOAD_IN_4BIT:-1}" = 1 ] && EXTRA_ARGS+=(--load_in_4bit)

exec "${PYTHON}" -u -m gui_agent.evaluation.sft \
  --model_name_or_path "${BASE_MODEL}" \
  --adapter_path "${EVAL_ADAPTER:-}" \
  --dataset_dir "${CAGUI_ROOT}" \
  --split_manifest "${SPLIT_MANIFEST:-${PROJECT_ROOT}/configs/splits/cagui-domestic-seed42-v1.json}" \
  --output_file "${OUTPUT_FILE}" \
  --label "${EVAL_LABEL}" \
  --split "${SPLIT:-domestic}" \
  --history_window "${HISTORY_WINDOW:-4}" \
  --max_ui_boxes "${MAX_UI_BOXES:-30}" \
  --max_image_side "${MAX_IMAGE_SIDE:-448}" \
  --max_new_tokens "${EVAL_MAX_NEW_TOKENS:-48}" \
  --max_samples "${MAX_TEST_SAMPLES:-0}" \
  --seed "${SEED:-42}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
