#!/usr/bin/env python3
"""Benchmark the unoptimised CPU FP64 reactive reference without overclaiming."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = ROOT / "python"
if str(PYTHON_SOURCE) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE))

from ecsp_reactive.configuration import load_reactive_config  # noqa: E402
from ecsp_reactive.solver import ReactiveSolver  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the single-process NumPy CPU FP64 reference. This does not "
            "benchmark or validate MPI, CUDA, A100, or the preserved v8.2.1 backend."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "paper_faithful_assumed_v8_3_0.yaml",
    )
    parser.add_argument("--grids", type=int, nargs="+", default=[20, 40, 80])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--maximum-steps",
        type=int,
        default=1000,
        help="Explicit benchmark fail-closed step budget (default: 1000)",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _state_digest(state: np.ndarray) -> str:
    canonical = np.ascontiguousarray(state, dtype=np.float64)
    return hashlib.sha256(canonical.view(np.uint8)).hexdigest()


def _atomic_json(path: Path, report: dict[str, object]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary_name).replace(path)
    finally:
        if temporary_name is not None and Path(temporary_name).exists():
            Path(temporary_name).unlink()


def main() -> int:
    args = _arguments()
    if args.repeats < 2:
        raise SystemExit("--repeats must be at least 2")
    if args.maximum_steps <= 0:
        raise SystemExit("--maximum-steps must be positive")
    if any(grid < 8 for grid in args.grids) or len(set(args.grids)) != len(args.grids):
        raise SystemExit("--grids must contain unique integers >= 8")

    base = load_reactive_config(args.config)
    rows: list[dict[str, object]] = []
    all_deterministic = True
    for grid in args.grids:
        config = replace(
            base, nx=grid, ny=grid, maximum_steps=args.maximum_steps
        )
        # One untimed run warms import/allocation paths and also catches failures
        # before timings are reported.
        ReactiveSolver(config).run()
        samples: list[float] = []
        hashes: list[str] = []
        steps: list[int] = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            result = ReactiveSolver(config).run()
            elapsed = time.perf_counter() - started
            samples.append(elapsed)
            hashes.append(_state_digest(result.conservative_state))
            steps.append(result.accepted_steps)
        deterministic = len(set(hashes)) == 1 and len(set(steps)) == 1
        all_deterministic = all_deterministic and deterministic
        median = statistics.median(samples)
        work_units = grid * grid * steps[0]
        rows.append(
            {
                "grid_cells_per_axis": grid,
                "cell_count": grid * grid,
                "accepted_steps": steps[0],
                "repeat_count": args.repeats,
                "wall_time_s": {
                    "minimum": min(samples),
                    "median": median,
                    "maximum": max(samples),
                },
                "cell_steps_per_second_using_median": work_units / median,
                "bitwise_final_state_deterministic": deterministic,
                "final_state_sha256": hashes[0],
            }
        )

    report: dict[str, object] = {
        "schema": "ecsp.paper-reactive-benchmark/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "single_process_numpy_cpu_fp64_reference",
        "configuration": str(args.config.expanduser().resolve()),
        "explicit_maximum_steps": args.maximum_steps,
        "comparison_status": "NO_OPTIMIZED_REACTIVE_BACKEND_IMPLEMENTED",
        "overall_status": "PASS" if all_deterministic else "FAIL_NONDETERMINISTIC",
        "scope_warning": (
            "Timing is host-specific reference data, not a production throughput "
            "claim and not a comparison with the preserved v8.2.1 solver."
        ),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "capability_status": {
            "MPI": "NOT_TESTED_NO_MPI_RUNTIME",
            "CUDA": "NOT_IMPLEMENTED_NOT_TESTED_NO_TOOLCHAIN",
            "A100": "NOT_IMPLEMENTED_NOT_TESTED_NO_HARDWARE",
        },
        "runs": rows,
    }
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if all_deterministic else 1


if __name__ == "__main__":
    raise SystemExit(main())
