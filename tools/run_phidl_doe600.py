#!/usr/bin/env python3
"""Separate PHIDL DOE entry point. Geometry-only unless explicitly enabled."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
# The optional per-project dependency directory isolates PHIDL's pinned CAD
# stack from the existing NSGA-II environment. Conventional pip installs work too.
if (ROOT / ".doe-deps").is_dir():
    sys.path.insert(0, str(ROOT / ".doe-deps"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/nsga2_preflame_only_200x3.yaml",
                        help="Existing experiment YAML; sole source for geometry and physics constraints")
    parser.add_argument("--doe-config", type=Path, default=ROOT / "config/phidl_doe600.yaml")
    parser.add_argument("--run-dir", type=Path, help="New run directory, or exact original directory for --resume")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute-physics", action="store_true", help="Explicitly run all pre-flame evaluations and baseline")
    parser.add_argument("--postflame", action="store_true", help="Also refine only fixed final20 + staggered")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed physics; completed results remain cached")
    args = parser.parse_args(argv)
    if args.resume and args.run_dir is None:
        parser.error("--resume requires --run-dir")
    if args.postflame and not args.execute_physics:
        parser.error("--postflame requires --execute-physics")
    import yaml
    from ecsp_doe.workflow import DOEWorkflow
    config = yaml.safe_load(args.config.read_text())
    doe_config = yaml.safe_load(args.doe_config.read_text()) or {}
    config["phidl_doe"] = doe_config.get("phidl_doe", doe_config)
    run_dir = args.run_dir or ROOT / "runs" / f"doe600_phidl_{datetime.now():%Y%m%d_%H%M%S}"
    result = DOEWorkflow(ROOT, config, run_dir).run(resume=args.resume,
        execute_physics=args.execute_physics, postflame=args.postflame, retry_failed=args.retry_failed)
    print(result, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
