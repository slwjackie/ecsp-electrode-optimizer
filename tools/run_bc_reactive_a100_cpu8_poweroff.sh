#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CONFIG="${CONFIG:-$ROOT/config/nsga2_bc_reactive_a100_cpu8_poweroff.yaml}"
export ECSP_REQUIRE_POST_ONSET_POWER_OFF=1
exec bash "$ROOT/tools/run_bc_reactive_a100_cpu8.sh" "${1:-$ROOT/runs/v842_reactive_poweroff}"
