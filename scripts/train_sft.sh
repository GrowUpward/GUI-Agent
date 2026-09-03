#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "${SCRIPT_DIR}/_common.sh"
require_env BASE_MODEL
require_env CAGUI_ROOT

RUN_NAME=${RUN_NAME:-sft-cagui}
OUTPUT_DIR=${OUTPUT_DIR:-${ARTIFACT_ROOT}/train/${RUN_NAME}}
EXTRA_ARGS=
[ "${BF16:-1}" = 1 ] && EXTRA_ARGS="${EXTRA_ARGS} --bf16"
[ "${FP16:-0}" = 1 ] && EXTRA_ARGS="${EXTRA_ARGS} --fp16"
[ "${LOAD_IN_4BIT:-1}" = 1 ] && EXTRA_ARGS="${EXTRA_ARGS} --load_in_4bit"
if [ "${GRADIENT_CHECKPOINTING:-1}" = 1 ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --gradient_checkpointing"
else
  EXTRA_ARGS="${EXTRA_ARGS} --no_gradient_checkpointing"
fi

LAUNCHER=("${PYTHON}" -u)
if [ "${NUM_GPUS:-1}" -gt 1 ]; then
  LAUNCHER+=(
    -m torch.distributed.run
    --standalone
    --nproc_per_node "${NUM_GPUS}"
    --module gui_agent.training.sft
  )
else
  LAUNCHER+=(-m gui_agent.training.sft)
fi

exec "${LAUNCHER[@]}" \
  --model_name_or_path "${BASE_MODEL}" \
  --dataset_dir "${CAGUI_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --init_adapter "${INIT_ADAPTER:-}" \
  --split "${SPLIT:-domestic}" \
  --max_episodes "${MAX_EPISODES:-0}" \
  --history_window "${HISTORY_WINDOW:-4}" \
  --max_ui_boxes "${MAX_UI_BOXES:-30}" \
  --max_image_side "${MAX_IMAGE_SIDE:-448}" \
  --val_ratio "${VAL_RATIO:-0.1}" \
  --test_ratio "${TEST_RATIO:-0.1}" \
  --max_eval_samples "${MAX_EVAL_SAMPLES:-256}" \
  --max_test_samples "${MAX_TEST_SAMPLES:-256}" \
  --eval_steps "${EVAL_STEPS:-10}" \
  --eval_max_new_tokens "${EVAL_MAX_NEW_TOKENS:-48}" \
  --max_steps "${MAX_STEPS:-300}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}" \
  --per_device_train_batch_size "${BATCH_SIZE:-1}" \
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE:-${BATCH_SIZE:-1}}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
  --warmup_ratio "${WARMUP_RATIO:-0.03}" \
  --logging_steps "${LOGGING_STEPS:-5}" \
  --save_steps "${SAVE_STEPS:-100}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-0}" \
  --preview_samples "${PREVIEW_SAMPLES:-5}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
  ${EXTRA_ARGS} \
  "$@"
