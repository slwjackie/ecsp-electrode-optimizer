#!/usr/bin/env python3
"""Run the v8.3 paper-reactive verification matrix and emit strict JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = ROOT / "python"
if str(PYTHON_SOURCE) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE))

from ecsp_reactive.validation import (  # noqa: E402
    PASS,
    run_validation,
    write_validation_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run CPU manufactured checks. A successful process does not imply "
            "paper reproduction, MPI, CUDA, or A100 validation."
        )
    )
    parser.add_argument(
        "--deterministic-repeats",
        type=int,
        default=50,
        help="number of bitwise deterministic repeats (minimum 2; default 50)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON output path; the report is always printed to stdout",
    )
    parser.add_argument(
        "--require-full-verification",
        action="store_true",
        help="return nonzero while the top-level verdict is NOT_FULLY_VERIFIED",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_validation(deterministic_repeats=args.deterministic_repeats)
    if args.output is not None:
        write_validation_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if report["runnable_cpu_verification"]["status"] != PASS:
        return 1
    if args.require_full_verification and report["verdict"] != "FULLY_VERIFIED":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
