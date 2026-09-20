#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-${PROJECT_ROOT}/config/paper_faithful_assumed_v8_3_0.yaml}"
OUTPUT_PREFIX="${2:-${PROJECT_ROOT}/runs/reactive_reference_v8_3_0/result}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_EXECUTABLE="${PYTHON_BIN}"
elif [[ -x "${PROJECT_ROOT}/../../venv/bin/python" ]]; then
  PYTHON_EXECUTABLE="${PROJECT_ROOT}/../../venv/bin/python"
else
  PYTHON_EXECUTABLE="python3"
fi

PYTHONPATH="${PROJECT_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_EXECUTABLE}" -m ecsp_reactive.cli \
  "${CONFIG_PATH}" \
  --output-prefix "${OUTPUT_PREFIX}"
