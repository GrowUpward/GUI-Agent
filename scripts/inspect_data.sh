#!/usr/bin/env sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "${SCRIPT_DIR}/_common.sh"
require_env CAGUI_ROOT
exec "${PYTHON}" -m gui_agent.data.inspect_cagui --cagui_root "${CAGUI_ROOT}" "$@"
