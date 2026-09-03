#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
PYTHON=${PYTHON:-python}
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

require_env() {
  name=$1
  eval "value=\${${name}:-}"
  if [ -z "${value}" ]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
}

ARTIFACT_ROOT=${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}
