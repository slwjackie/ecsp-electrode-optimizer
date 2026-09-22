#!/usr/bin/env python3
"""Generate/audit/evaluate only E058, E114, R038, R050, R091.

This is additive orchestration. It does not modify ECSP preflame/postflame physics,
solvers, voltage-search code, or the existing generic geometry generator.
"""
from __future__ import annotations
import argparse, copy, dataclasses, hashlib, json, sys
from pathlib import Path
import numpy as np
import shapely
from scipy.io import savemat

from ecsp_image_design.five_topologies import fit_reference, gap_guard, render, signature, first_erosion_width, rolling_open
from ecsp_image_design.objectives import runtime_config, context, normalized_objectives, digest, strict_json

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

def _meta(m,baseline=False):
    return {'geometry_id':m['source_id']+('_staggered' if baseline else ''),
            'topology_id':'independent_area_matched_staggered' if baseline else m['source_id'],
            'intended_anode_components':2 if baseline else int(m['anode_components']),
            'intended_cathode_components':2 if baseline else int(m['cathode_components']),
            'surface_contact_model':True,'hidden_bus_assumed':True,
            'electrode_area_fraction':float(m['electrode_area_fraction']),
            'domain_mm':float(m['domain_mm']),
            'source_role':'external_reference_only' if baseline else 'five_topology_candidate',
            'baseline_type':'area_matched_staggered' if baseline else None}

def _valid(evaluator,row):
    primary=getattr(evaluator,'gpu_or_main',evaluator)
    if not hasattr(primary,'_trial_valid'):return False,'missing_production_validity_api'
    return primary._trial_valid(row)

def evaluate(project_root,config_path,out):
    root=Path(project_root).resolve();out=Path(out).resolve();lib=out/'library'
    sys.path.insert(0,str(root/'python'))
    import yaml
    from ecsp_nsga2.evaluator import create_evaluator
    from ecsp_nsga2.geometry import GeometryLimits
    from ecsp_nsga2.baselines import generate_area_matched_staggered
    base=yaml.safe_load(Path(config_path).read_text())
    results=[]
    for sid in IDS:
        d=lib/sid;m=json.loads((d/'metadata.json').read_text())
        if m['status']!='geometry_accepted':raise RuntimeError(sid+' geometry not accepted')
        with np.load(d/'mask.npz',allow_pickle=False) as f:a=f['anode'].astype(bool);c=f['cathode'].astype(bool)
        cfg=runtime_config(base,m);ctx=context(m,digest(cfg))
        fields={x.name for x in dataclasses.fields(GeometryLimits)}
        limits=GeometryLimits(**{k:v for k,v in cfg['geometry'].items() if k in fields})
        bc=cfg.get('baselines',{}).get('area_matched_staggered',{})
        opts={k:bc[k] for k in ('fingers_per_polarity','target_interdigitation_overlap_fraction','minimum_interdigitation_overlap_fraction','maximum_gap_safety_pixels') if k in bc}
        _,raster,params=generate_area_matched_staggered(limits,physics_grid_size=m['grid_size'],target_area_fraction_per_polarity=m['electrode_area_fraction']/2,**opts)
        ba=np.asarray(raster.anode_mask,bool);bca=np.asarray(raster.cathode_mask,bool);L=float(m['domain_mm'])
        bctx={**ctx,'anode_area_mm2':float(ba.mean()*L*L),'cathode_area_mm2':float(bca.mean()*L*L)}
        adapter=copy.deepcopy(cfg['evaluator']);adapter['physics_config']=copy.deepcopy(cfg)
        evaluator=create_evaluator(root,adapter,out/'preflame'/sid/'adapter',False)
        try:
            raw=evaluator.evaluate_batch([(a,c,_meta(m),out/'preflame'/sid/'candidate'),(ba,bca,_meta(m,True),out/'preflame'/sid/'staggered')])
            cand,bl=[dict(x) for x in raw]
            for row in (cand,bl):
                ok,reason=_valid(evaluator,row);row['external_numerical_valid']=bool(ok);row['external_numerical_reason']=str(reason)
        finally:
            if hasattr(evaluator,'close'):evaluator.close()
        norm=normalized_objectives(cand,bl,ctx,bctx)
        result={'source_id':sid,'context':ctx,'baseline_context':bctx,'candidate_raw':cand,'baseline_raw':bl,**norm}
        results.append(result);write_json(out/'preflame'/sid/'result.json',result);write_json(out/'preflame'/sid/'baseline_parameters.json',dataclasses.asdict(params))
    write_json(out/'preflame'/'comparison.json',results)
    return results

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['generate','audit','evaluate'])
    ap.add_argument('--project-root',type=Path,default=Path.cwd());ap.add_argument('--refs',type=Path,default=Path('data/image_design/five_topology_references.json'))
    ap.add_argument('--out',type=Path,default=Path('runs/five_topologies'));ap.add_argument('--config',type=Path,default=Path('config/nsga2_bc_reactive_a100_cpu8_poweroff_200x3.yaml'))
    args=ap.parse_args();refs=args.refs if args.refs.is_absolute() else args.project_root/args.refs;out=args.out if args.out.is_absolute() else args.project_root/args.out
    if args.mode=='generate':generate(load_refs(refs),out)
    elif args.mode=='audit':audit(out)
    else:
        cfg=args.config if args.config.is_absolute() else args.project_root/args.config
        evaluate(args.project_root,cfg,out)
if __name__=='__main__':main()
