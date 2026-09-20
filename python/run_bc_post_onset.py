#!/usr/bin/env python3
"""Continue an existing authorized BC handoff, without rerunning NSGA-II.

Example:
 python python/run_bc_post_onset.py --config config/nsga2_bc_reactive_debug.yaml \
   --handoff-dir RUN/final/propagation_candidates/ID/bc_handoff \
   --resolved-bc-config RUN/adapter/resolved_physics_config.json --output NEW_DIR
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from ecsp_v6.physics.composition_model import build_composition
from ecsp_nsga2.post_onset import run_post_onset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True,type=Path)
    p.add_argument("--handoff-dir",required=True,type=Path)
    p.add_argument("--resolved-bc-config",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    if a.output.exists(): p.error("output already exists: choose a new directory")
    cfg=yaml.safe_load(a.config.read_text(encoding="utf-8"))
    full=json.loads(a.resolved_bc_config.read_text(encoding="utf-8"))
    h=json.loads((a.handoff_dir/"bc_handoff_metadata.json").read_text(encoding="utf-8"))
    with np.load(a.handoff_dir/"bc_handoff_fields.npz",allow_pickle=False) as f:
        h.update({k:f[k] for k in f.files})
    prop=dict(cfg["propagation_refinement"])
    density=full["bcGlobal"]["thermal"].get("density_kg_per_m3")
    prop.update(domain_size_m=float(full["geometry"]["domainSize_m"]),
                surface_layer_thickness_m=float(full["geometry"]["surfaceLayerThickness_m"]),
                gas_constant_J_per_molK=float(full["transport"]["gasConstant_J_per_molK"]),
                density_kg_per_m3=float(build_composition(full).density_kg_per_m3 if density is None else density))
    # Legacy handoffs without contact masks cannot be guessed from heat fields.
    result=run_post_onset(h,prop,full["bcGlobal"],a.output,post_onset_config=cfg.get("post_onset",{}),full_bc_config=full)
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=="__main__": main()
