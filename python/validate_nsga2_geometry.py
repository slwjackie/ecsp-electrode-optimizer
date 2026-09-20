#!/usr/bin/env python3
"""Generate all unique feasible bootstrap geometries without a physics solver."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import yaml
from ecsp_nsga2.bootstrap import (build_bootstrap, resolved_physics_grid,
                                 require_post_onset_power_off)


def main():
    root=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--require-power-off',action='store_true')
    args=ap.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit('Geometry output exists; choose a fresh directory (nothing is deleted)')
    cfg=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if not isinstance(cfg,dict):raise SystemExit('Configuration must be a mapping')
    if args.require_power_off:require_post_onset_power_off(cfg)
    _,report=build_bootstrap(cfg,physics_grid_size=resolved_physics_grid(cfg,root),audit_dir=args.output)
    print(json.dumps({k:report[k] for k in ('status','accepted_count','design_unique_count',
        'physics_unique_count','total_attempts','elapsed_s','maximum_design_area_relative_error',
        'maximum_physics_area_relative_error','minimum_design_gap_mm','minimum_physics_gap_mm',
        'maximum_used_width_mm')},indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
