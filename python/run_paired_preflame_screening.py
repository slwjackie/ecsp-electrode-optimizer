#!/usr/bin/env python3
"""Fixed-catalogue paired screening; no NSGA-II, global rank or scalar score."""
from __future__ import annotations
import argparse
import fcntl
from pathlib import Path
import sys

from ecsp_image_design.paired_screening import ScreeningPolicy
from ecsp_image_design.paired_workflow import (
    read_json, write_json, prepare_library, audit_library, evaluate_library, reclassify,
)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)
    prepare = sub.add_parser("prepare", help="Import frozen 143 plus five unchanged main geometries")
    prepare.add_argument("--base-archive", type=Path, required=True)
    prepare.add_argument("--five-library", type=Path)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    audit = sub.add_parser("audit", help="Read-only CAD/mask checks")
    evaluate = sub.add_parser("evaluate", help="Run paired preflame cases, not generations")
    for cmd in (audit, evaluate):
        cmd.add_argument("--library", type=Path, required=True)
        cmd.add_argument("--out", type=Path, required=True)
        cmd.add_argument("--ids", help="Explicit comma-separated subset; omit for all 148")
        cmd.add_argument("--expected-count", type=int, help="Default: 148, or number of explicit --ids")
    evaluate.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    evaluate.add_argument("--pcg-max-iterations", type=int, default=None,
                          help="Optional explicit iteration budget; residual tolerances are unchanged")
    evaluate.add_argument("--resume", action="store_true")
    evaluate.add_argument("--retry-failed", action="store_true")
    report = sub.add_parser("report", help="Reclassify saved result.json files without PDE reruns")
    report.add_argument("--results", type=Path, nargs="+", required=True)
    report.add_argument("--out", type=Path, required=True)
    for cmd in (evaluate, report):
        cmd.add_argument("--policy", type=Path, help="Optional JSON ScreeningPolicy; no physical parameters")
    args = ap.parse_args(argv)
    if args.mode == "prepare":
        print(prepare_library(args.project_root, args.base_archive, args.out, args.five_library))
        return 0
    ids = [x.strip() for x in args.ids.split(",") if x.strip()] if getattr(args, "ids", None) else None
    expected = getattr(args, "expected_count", None)
    if expected is None:
        expected = len(ids) if ids else 148
    if args.mode == "audit":
        result = audit_library(args.library, ids, expected)
        write_json(args.out / "audit.json", result)
        print(f"Geometry audit: {result['count']} cases; PDE not executed")
        return 0
    policy = ScreeningPolicy(**read_json(args.policy)) if args.policy else ScreeningPolicy()
    if args.mode == "report":
        rows = reclassify(args.results, args.out, policy)
        print(f"Reclassified {len(rows)} pairs; PDE not executed; no cross-design rank")
        return 0
    if args.retry_failed and not args.resume:
        ap.error("--retry-failed requires --resume")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".paired_screening.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another paired screening process owns this output directory")
        rows = evaluate_library(args.project_root, args.config, args.library, args.out,
                only_ids=ids, expected_count=expected, device=args.device,
                pcg_max_iterations=args.pcg_max_iterations, resume=args.resume,
                retry_failed=args.retry_failed, policy=policy)
    return 1 if any(r["comparison_state"] == "execution_failed" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
