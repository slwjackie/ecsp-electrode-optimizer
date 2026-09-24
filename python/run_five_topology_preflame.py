#!/usr/bin/env python3
"""Generate/audit the original five geometries; evaluate by paired screening.

The geometry functions, native solver and postflame implementation are unchanged.
For the full fixed catalogue or reclassification use run_paired_preflame_screening.py.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import shapely
from scipy.io import savemat

from ecsp_image_design.five_topologies import fit_reference, gap_guard, render, signature, first_erosion_width, rolling_open
from ecsp_image_design.objectives import strict_json

IDS=('E058','E114','R038','R050','R091')

def write_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(strict_json(value),indent=2,allow_nan=False),encoding='utf-8')

def load_refs(path):
    obj=json.loads(Path(path).read_text(encoding='utf-8')); recs={r['source_id']:r for r in obj['records']}
    missing=[x for x in IDS if x not in recs]
    if missing:raise ValueError('Missing references: '+','.join(missing))
    return recs

def generate(refs,out):
    out=Path(out);lib=out/'library';lib.mkdir(parents=True,exist_ok=True);rows=[]
    for sid in IDS:
        meta,res=fit_reference(refs[sid])
        rows.append(meta);d=lib/sid;d.mkdir(exist_ok=True);write_json(d/'metadata.json',meta)
        if res is None:continue
        a,c,am,cm=res
        np.savez_compressed(d/'mask.npz',anode=am,cathode=cm,propellant=np.ones_like(am,bool),domain_mm=meta['domain_mm'])
        savemat(d/'mask.mat',{'anodeMask':am,'cathodeMask':cm,'propellantMask':np.ones_like(am,bool),'domain_mm':meta['domain_mm']})
        write_json(d/'master.json',{'anode':shapely.to_geojson(a),'cathode':shapely.to_geojson(c),'metadata':meta})
    write_json(out/'geometry_qc.json',rows)
    return rows

def audit(out):
    out=Path(out);passed=[]
    for sid in IDS:
        d=out/'library'/sid;m=json.loads((d/'metadata.json').read_text())
        if m['status']!='geometry_accepted':raise RuntimeError(f'{sid}: {m}')
        with np.load(d/'mask.npz',allow_pickle=False) as f:a=f['anode'].astype(bool);c=f['cathode'].astype(bool)
        master=json.loads((d/'master.json').read_text());pa,pc=[shapely.from_geojson(master[k]) for k in ('anode','cathode')]
        L=float(m['domain_mm']);n=int(m['grid_size'])
        if abs(pa.area-pc.area)/max(pa.area,pc.area)>1e-6:raise RuntimeError(sid+' area imbalance')
        if pa.distance(pc)<3 or gap_guard(a,c,L/n)<3:raise RuntimeError(sid+' gap failure')
        if min(first_erosion_width(pa),first_erosion_width(pc))<1.998:raise RuntimeError(sid+' width failure')
        if any(p.difference(rolling_open(p,1.)).area/p.area>=.002 for p in (pa,pc)):raise RuntimeError(sid+' rolling width failure')
        if not np.array_equal(render(pa,n,L),a) or not np.array_equal(render(pc,n,L),c):raise RuntimeError(sid+' master/mask mismatch')
        passed.append(sid)
    write_json(out/'audit.json',{'verified_geometry_ids':passed,'count':len(passed),'pde_executed':False})
    return passed

def evaluate(project_root, config_path, out):
    """Compatibility entry point; results are labels/ratios, not a Pareto vector."""
    import yaml
    from ecsp_image_design.paired_workflow import evaluate_library
    base=yaml.safe_load(Path(config_path).read_text())
    return evaluate_library(project_root, config_path, Path(out)/'library', out,
                            only_ids=IDS, expected_count=len(IDS),
                            device=base.get('evaluator',{}).get('device','cuda'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['generate','audit','evaluate'])
    ap.add_argument('--project-root',type=Path,default=Path.cwd());ap.add_argument('--refs',type=Path,default=Path('data/image_design/five_topology_references.json'))
    ap.add_argument('--out',type=Path,default=Path('runs/five_topologies'));ap.add_argument('--config',type=Path,default=Path('config/nsga2_bc_reactive_a100_cpu8_poweroff_200x3.yaml'))
    args=ap.parse_args();refs=args.refs if args.refs.is_absolute() else args.project_root/args.refs;out=args.out if args.out.is_absolute() else args.project_root/args.out
    if args.mode=='generate':generate(load_refs(refs),out)
    elif args.mode=='audit':audit(out)
    else:
        cfg=args.config if args.config.is_absolute() else args.project_root/args.config
        rows=evaluate(args.project_root,cfg,out)
        if any(r['comparison_state']=='execution_failed' for r in rows):
            raise SystemExit(1)
if __name__=='__main__':main()
