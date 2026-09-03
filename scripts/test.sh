#!/usr/bin/env sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "${SCRIPT_DIR}/_common.sh"

"${PYTHON}" "${PROJECT_ROOT}/tests/smoke.py"
if "${PYTHON}" -c "import pytest" >/dev/null 2>&1; then
  exec "${PYTHON}" -m pytest "$@"
fi
echo "pytest is not installed; dependency-light smoke tests completed."
