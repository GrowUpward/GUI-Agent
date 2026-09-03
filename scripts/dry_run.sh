#!/usr/bin/env sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "${SCRIPT_DIR}/_common.sh"
require_env CAGUI_ROOT
exec "${PYTHON}" -m gui_agent.training.grpo \
  --base_model "${BASE_MODEL:-unused-for-dry-run}" \
  --dataset_dir "${CAGUI_ROOT}" \
  --output_dir "${ARTIFACT_ROOT}/dry-run" \
  --max_episodes "${MAX_EPISODES:-2}" \
  --dry_run \
  "$@"
