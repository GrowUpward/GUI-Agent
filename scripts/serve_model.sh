#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "${SCRIPT_DIR}/_common.sh"
MODEL_PATH=${MODEL_PATH:-${BASE_MODEL:-}}
ADAPTER_PATH=${ADAPTER_PATH:-${SFT_ADAPTER:-}}
require_env MODEL_PATH

ARGS=(
  --model-path "${MODEL_PATH}"
  --host "${HOST:-0.0.0.0}"
  --port "${PORT:-8000}"
  --device "${DEVICE:-auto}"
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}"
)
[ -n "${ADAPTER_PATH:-}" ] && ARGS+=(--adapter-path "${ADAPTER_PATH}")
[ "${LOAD_IN_4BIT:-1}" = 1 ] && ARGS+=(--load-in-4bit)
[ "${LEGACY_YX_OUTPUT:-0}" = 1 ] && ARGS+=(--legacy-yx-output)

exec "${PYTHON}" -m gui_agent.deployment.model_server "${ARGS[@]}" "$@"
