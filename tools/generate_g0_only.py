#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import csv
import yaml

from ecsp_nsga2.bootstrap import build_bootstrap, resolved_physics_grid
from ecsp_nsga2.geometry import save_geometry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--package-root', type=Path, default=Path('.'))
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    root = args.package_root.resolve()
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    grid = resolved_physics_grid(cfg, root)
    out = args.out.resolve()
    geomdir = out / 'generation_000' / 'geometries'
    audit = out / 'geometry_bootstrap'
    entries, report = build_bootstrap(
        cfg,
        physics_grid_size=grid,
        audit_dir=audit,
        cache_dir=None,
        verbose=True,
    )
    geomdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for genome, raster in entries:
        save_geometry(genome, raster, geomdir)
        rows.append({
            'geometry_id': genome['geometry_id'],
            'topology_id': genome['topology_id'],
            'family': genome.get('family', ''),
            'anode_components': len(genome['anode']['components']),
            'cathode_components': len(genome['cathode']['components']),
            'anode_area_fraction': raster.descriptors['anode_area_fraction'],
            'cathode_area_fraction': raster.descriptors['cathode_area_fraction'],
            'minimum_gap_mm': raster.descriptors['minimum_gap_mm'],
        })
    with (out/'generation_000'/'geometry_manifest.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f'[g0-only] complete: {len(entries)} geometries -> {geomdir}')
    print(f'[g0-only] bootstrap status: {report["status"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
