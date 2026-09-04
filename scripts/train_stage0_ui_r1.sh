#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "${SCRIPT_DIR}/_common.sh"
require_env BASE_MODEL
require_env UI_R1_WARMUP_JSON
require_env UI_R1_WARMUP_IMAGES

RUN_NAME=${RUN_NAME:-stage0-ui-r1-warmup}
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
  --dataset_format ui_r1_warmup \
  --data_file "${UI_R1_WARMUP_JSON}" \
  --image_dir "${UI_R1_WARMUP_IMAGES}" \
  --dataset_dir "${CAGUI_ROOT:-${PROJECT_ROOT}/data/CAGUI}" \
  --output_dir "${OUTPUT_DIR}" \
  --split_manifest "" \
  --max_ui_boxes 0 \
  --max_image_side "${MAX_IMAGE_SIDE:-448}" \
  --max_eval_samples 0 \
  --max_test_samples 0 \
  --max_steps "${MAX_STEPS:--1}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-3}" \
  --per_device_train_batch_size "${BATCH_SIZE:-1}" \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
  --learning_rate "${LEARNING_RATE:-1e-5}" \
  --warmup_ratio "${WARMUP_RATIO:-0.05}" \
  --logging_steps "${LOGGING_STEPS:-1}" \
  --save_steps "${SAVE_STEPS:-20}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-2}" \
  --preview_samples "${PREVIEW_SAMPLES:-5}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
  ${EXTRA_ARGS} \
  "$@"
