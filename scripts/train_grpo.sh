#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "${SCRIPT_DIR}/_common.sh"
require_env BASE_MODEL
require_env CAGUI_ROOT
require_env SFT_ADAPTER

RUN_NAME=${RUN_NAME:-grpo-cagui}
OUTPUT_DIR=${OUTPUT_DIR:-${ARTIFACT_ROOT}/train/${RUN_NAME}}
mkdir -p "${OUTPUT_DIR}"

"${PYTHON}" -u -m gui_agent.training.grpo \
  --base_model "${BASE_MODEL}" \
  --sft_adapter "${SFT_ADAPTER}" \
  --dataset_dir "${CAGUI_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --use_lora \
  --lora_rank "${LORA_RANK:-16}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --num_generations "${NUM_GENERATIONS:-4}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-48}" \
  --history_window "${HISTORY_WINDOW:-4}" \
  --max_ui_boxes "${MAX_UI_BOXES:-30}" \
  --max_image_side "${MAX_IMAGE_SIDE:-448}" \
  --generation_micro_batch_size "${GEN_MICRO_BATCH_SIZE:-1}" \
  --logprob_micro_batch_size "${LOGPROB_MICRO_BATCH_SIZE:-1}" \
  --learning_rate "${LEARNING_RATE:-1e-6}" \
  --beta_kl "${BETA_KL:-0.08}" \
  --temperature "${TEMPERATURE:-0.3}" \
  --top_p "${TOP_P:-1.0}" \
  --generation_prefix "${GENERATION_PREFIX:-\{'action': }" \
  --repetition_penalty "${REPETITION_PENALTY:-1.05}" \
  --max_steps "${MAX_STEPS:-10}" \
  --save_steps "${SAVE_STEPS:-10}" \
  --logging_steps "${LOGGING_STEPS:-1}" \
  --bf16 \
  --load_in_4bit \
  --gradient_checkpointing \
  "$@" \
  2>&1 | tee "${OUTPUT_DIR}/run.log"
