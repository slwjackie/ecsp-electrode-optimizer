#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${1:-$ROOT/runs/m2_cpp_fp64_benchmark}"
HORIZON="${ECSP_CPP_BENCHMARK_HORIZON_S:-0.2}"
TARGET_HORIZON="${ECSP_CPP_TARGET_HORIZON_S:-2.0}"
PARALLEL="${ECSP_CPP_PARALLEL_CASES:-4}"
CONFIG="${CONFIG:-$ROOT/config/nsga2_condensed_phase_no_f_m2_cpp_fp64.yaml}"
if [[ -e "$WORKDIR" ]]; then
  echo "ERROR: benchmark workdir exists: $WORKDIR" >&2
  exit 2
fi
mkdir -p "$WORKDIR"
"$ROOT/tools/build_cpp_cpu.sh"
export PYTHONPATH="$ROOT/python:${PYTHONPATH:-}"

run_gate() {
  local target="$1"
  local horizon="$2"
  python "$ROOT/python/run_nsga2_electrical_solid_loop.py" \
    --package-root "$ROOT" \
    --config "$CONFIG" \
    --device cpu \
    --population-size 4 \
    --generations 1 \
    --grid-size 193 \
    --end-time-s "$horizon" \
    --allow-no-feasible \
    --workdir "$target"
}

run_gate "$WORKDIR/one_step" 0.00025
run_gate "$WORKDIR/horizon" "$HORIZON"

python - "$WORKDIR" "$HORIZON" "$TARGET_HORIZON" "$PARALLEL" <<'PY'
import json, math, statistics, sys
from pathlib import Path

root = Path(sys.argv[1])
horizon = float(sys.argv[2])
target_horizon = float(sys.argv[3])
parallel = max(1, int(sys.argv[4]))

def load_rows(folder: Path):
    rows = {}
    for path in (folder / "generation_000" / "physics").glob("*/condensed_metrics.json"):
        rows[path.parent.name] = json.loads(path.read_text())
    rejected = list((folder / "generation_000" / "physics").glob("*/physics_rejection.txt"))
    return rows, rejected

one, one_rejected = load_rows(root / "one_step")
bench, bench_rejected = load_rows(root / "horizon")
status = "passed" if len(one) == 4 and len(bench) == 4 and not one_rejected and not bench_rejected else "failed"
config_paths = list((root / "horizon" / "generation_000" / "physics").glob("*/cpp_physics_config.kv"))
if not config_paths:
    raise SystemExit("benchmark produced no C++ config files")
kv = {}
for line in config_paths[0].read_text().splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        kv[key] = value
dt = float(kv["dt"])
electrical_interval = float(kv["electrical_interval"])

projections = []
for geometry_id in sorted(set(one) & set(bench)):
    a = one[geometry_id]
    b = bench[geometry_id]
    bench_updates = max(1, int(b.get("electricalSolveCount", 1)) - 1)
    # Each electrical interval also contains the same fixed number of explicit
    # species/thermal timesteps, so this scales both costs after removing the
    # initial nonlinear solve.
    target_steps = int(round(target_horizon / dt))
    solve_every = max(1, int(round(electrical_interval / dt)))
    target_updates = max(0, (target_steps - 1) // solve_every)
    startup = float(a["wallClockTime_s"])
    incremental = max(0.0, float(b["wallClockTime_s"]) - startup)
    projected = startup + incremental * target_updates / bench_updates
    projections.append({
        "geometry_id": geometry_id,
        "one_step_wall_s": startup,
        "benchmark_wall_s": float(b["wallClockTime_s"]),
        "benchmark_electrical_solves": int(b.get("electricalSolveCount", 0)),
        "projected_2s_case_wall_s": projected,
    })

values = [row["projected_2s_case_wall_s"] for row in projections]
median_case = statistics.median(values) if values else None
maximum_case = max(values) if values else None
waves_100x2 = math.ceil(200 / parallel)
report = {
    "status": status,
    "grid_size": 193,
    "benchmark_horizon_s": horizon,
    "target_horizon_s": target_horizon,
    "parallel_cases": parallel,
    "time_step_s": dt,
    "electrical_update_interval_s": electrical_interval,
    "one_step_successful": len(one),
    "horizon_successful": len(bench),
    "one_step_rejected": len(one_rejected),
    "horizon_rejected": len(bench_rejected),
    "case_projections": projections,
    "median_projected_2s_case_wall_s": median_case,
    "maximum_projected_2s_case_wall_s": maximum_case,
    "projected_100x2_physics_wall_hours_median": (
        median_case * waves_100x2 / 3600.0 if median_case is not None else None
    ),
    "projected_100x2_physics_wall_hours_conservative": (
        maximum_case * waves_100x2 / 3600.0 if maximum_case is not None else None
    ),
    "projection_note": (
        "Projection removes the expensive initial nonlinear solve and scales the "
        "measured post-startup electrical-interval cost. It remains a planning "
        "estimate; geometry difficulty and M2 thermal throttling can change it."
    ),
}
(root / "m2_cpp_fp64_benchmark_summary.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
raise SystemExit(0 if status == "passed" else 3)
PY
