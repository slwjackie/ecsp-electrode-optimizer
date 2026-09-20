from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy import ndimage

from ecsp_nsga2.geometry import (GeometryLimits, _fit_area_components, instantiate_variant,
    make_topology_templates, rasterize_and_validate, save_geometry, topology_signature)
from ecsp_nsga2.bootstrap import (BootstrapGeometryError, build_bootstrap,
    require_post_onset_power_off, validate_solver_grid)

ROOT=Path(__file__).resolve().parents[2]


def config():
    return yaml.safe_load((ROOT/'config/nsga2_bc_reactive_a100_cpu8_poweroff.yaml').read_text())


def small_config():
    c=config()
    c['optimization'].update(initial_topologies=1,variants_per_topology=2,population_size=2,
                              bootstrap_max_attempts_per_topology=100)
    return c


def test_production_keeps_5mm_and_each_polarity_1percent():
    paths=[]
    for p in (ROOT/'config').glob('nsga2_bc_*.yaml'):
        cfg=yaml.safe_load(p.read_text())
        if cfg.get('optimization',{}).get('population_size')==1000:
            assert cfg['geometry']['maximum_width_mm']==5.0
            assert cfg['geometry']['area_tolerance_fraction']==0.01
            paths.append(p)
    assert len(paths)>=11
    require_post_onset_power_off(config())
    assert config()['evaluator']['base_overrides']['coupled']['useFullNernstPlanckTransport'] is True


def test_length_is_extended_before_width_when_space_exists():
    limits=GeometryLimits(grid_size=200,minimum_width_mm=.3,maximum_width_mm=5.0,
                          maximum_segment_mm=15.0,target_area_fraction_per_polarity=.01,
                          area_tolerance_fraction=.01)
    paths=[[([(7.5,10.0),(12.5,10.0)],.5)]]
    mask,ws,lo,hi,plan=_fit_area_components(paths,limits,return_plan=True)
    assert plan['status']=='fitted'
    assert plan['phase']=='length_first'
    assert plan['length_scale']>1.0
    assert ws==1.0
    assert abs(float(mask.mean())-.01)/.01<=.01


def test_no_hidden_reduction_of_five_mm_ceiling():
    limits=GeometryLimits(grid_size=200,maximum_width_mm=5.0,maximum_segment_mm=15.0,
                          target_area_fraction_per_polarity=.17,area_tolerance_fraction=.01)
    paths=[[([(6.0,10.0),(14.0,10.0)],5.0)]]
    mask,ws,lo,hi,plan=_fit_area_components(paths,limits,return_plan=True)
    assert plan['status']=='fitted'
    assert hi>4.9 and hi<=5.0+1e-12
    assert ndimage.label(mask,structure=np.ones((3,3)))[1]==1


def test_impossible_width_fit_never_returns_merged_components():
    limits=GeometryLimits(grid_size=120,maximum_width_mm=5.0,area_tolerance_fraction=.01,
                          target_area_fraction_per_polarity=.30)
    paths=[[([(5.0,9.0),(15.0,9.0)],.5)],
           [([(5.0,10.0),(15.0,10.0)],.5)]]
    mask,*rest=_fit_area_components(paths,limits,return_plan=True)
    plan=rest[-1]
    assert plan['status']!='fitted'
    assert ndimage.label(mask,structure=np.ones((3,3)))[1] in (0,2)
    assert plan['global_maximum_width_mm']==5.0 if 'global_maximum_width_mm' in plan else True


def test_fitted_primitives_reproduce_masks_and_do_not_refit_on_reload(tmp_path):
    c=config();limits=GeometryLimits(**c['geometry'])
    template=make_topology_templates(20,c['project']['seed'],limits)[3]
    g=instantiate_variant(template,0,c['project']['seed'],limits)
    raw=copy.deepcopy(g)
    r=rasterize_and_validate(g,limits)
    assert g==raw, 'rasterization must not mutate caller-owned genome'
    assert r.constraint_violation<=1e-12
    fitted=r.fitted_genome
    assert topology_signature(fitted)==topology_signature(g)
    r2=rasterize_and_validate(fitted,limits)
    np.testing.assert_array_equal(r.anode_mask,r2.anode_mask)
    np.testing.assert_array_equal(r.cathode_mask,r2.cathode_mask)
    save_geometry(g,r,tmp_path)
    saved=json.loads((tmp_path/(g['geometry_id']+'.json')).read_text())
    r3=rasterize_and_validate(saved,limits)
    np.testing.assert_array_equal(r.anode_mask,r3.anode_mask)
    np.testing.assert_array_equal(r.cathode_mask,r3.cathode_mask)
    assert r.descriptors['maximum_effective_width_mm']>1.2


def test_revalidation_checks_each_polarity_not_average():
    c=config();limits=GeometryLimits(**c['geometry'])
    limits=GeometryLimits(**{**c['geometry'],'grid_size':100})
    a=np.zeros((100,100),bool);b=a.copy()
    # Aa=1780 (1.714% high), Ac=1750 exact: the former average test would pass.
    a[10:80,5:30]=True;a[80,5:30]=True;a[81,5:10]=True
    b[10:80,65:90]=True
    g={'anode':{'components':[{}]},'cathode':{'components':[{}]}}
    report=validate_solver_grid(a,b,g,limits,100)
    assert 'physics_per_polarity_area' in report['violations']
    assert (report['anode_area_relative_error']+report['cathode_area_relative_error'])/2<.01


def test_all_20_topologies_have_unique_feasible_small_quotas(tmp_path):
    c=config();c['optimization'].update(population_size=40,variants_per_topology=2)
    entries,report=build_bootstrap(c,physics_grid_size=193,audit_dir=tmp_path/'audit',verbose=False)
    assert len(entries)==40
    assert len(report['topologies'])==20
    assert all(t['accepted']==2 for t in report['topologies'])
    assert report['design_unique_count']==40 and report['physics_unique_count']==40
    assert report['maximum_design_area_relative_error']<=.01+1e-12
    assert report['maximum_physics_area_relative_error']<=.01+1e-12
    assert report['contract']['limits']['maximum_width_mm']==5.0


def test_bootstrap_never_pads_invalid_candidates(tmp_path,monkeypatch):
    import ecsp_nsga2.bootstrap as module
    original=module.rasterize_and_validate
    calls=[]
    def invalid(g,lim):
        calls.append(1)
        r=original(g,lim)
        r.constraint_violation=1.0
        r.violation_details['forced_test_failure']=1.0
        return r
    monkeypatch.setattr(module,'rasterize_and_validate',invalid)
    c=small_config();c['optimization']['bootstrap_max_attempts_per_topology']=3
    with pytest.raises(BootstrapGeometryError,match='no INVALID padding'):
        build_bootstrap(c,physics_grid_size=193,audit_dir=tmp_path/'failed',verbose=False)
    assert len(calls)==3
    report=json.loads((tmp_path/'failed/geometry_bootstrap_report.json').read_text())
    assert report['status']=='failed' and report['accepted_count']==0
    assert report['all_candidates_feasible_before_physics'] is False
    assert not (tmp_path/'failed/bootstrap_genomes.json').exists()


def test_duplicate_masks_do_not_fill_a_topology_quota(tmp_path,monkeypatch):
    import ecsp_nsga2.bootstrap as module
    original=module.instantiate_variant
    monkeypatch.setattr(module,'instantiate_variant',lambda t,i,s,l:original(t,0,s,l))
    c=small_config();c['optimization']['bootstrap_max_attempts_per_topology']=3
    with pytest.raises(BootstrapGeometryError):
        build_bootstrap(c,physics_grid_size=193,audit_dir=tmp_path/'duplicate',verbose=False)
    report=json.loads((tmp_path/'duplicate/geometry_bootstrap_report.json').read_text())
    assert report['accepted_count']==1
    assert report['topologies'][0]['rejections']['duplicate_mask']==2


def test_bootstrap_cache_is_revalidated_and_bound_to_seed_and_grids(tmp_path):
    c=small_config()
    entries,report=build_bootstrap(c,physics_grid_size=193,audit_dir=tmp_path/'cache',verbose=False)
    reused,r=build_bootstrap(c,physics_grid_size=193,cache_dir=tmp_path/'cache',verbose=False)
    assert r['reused_cache'] is True
    for (_,first),(_,second) in zip(entries,reused):
        np.testing.assert_array_equal(first.anode_mask,second.anode_mask)
        np.testing.assert_array_equal(first.cathode_mask,second.cathode_mask)
    c['project']['seed']+=1
    with pytest.raises(BootstrapGeometryError,match='stale'):
        build_bootstrap(c,physics_grid_size=193,cache_dir=tmp_path/'cache',verbose=False)


@pytest.mark.parametrize('bad',[True,'false',0,None])
def test_poweroff_requires_real_yaml_false(bad):
    c=config();c['propagation_refinement']['continued_electrical_heating']=bad
    with pytest.raises(ValueError):require_post_onset_power_off(c)


def test_poweroff_does_not_accept_recomputed_mode():
    c=config();c['propagation_refinement']['electrical_heating_mode']='recomputed'
    with pytest.raises(ValueError):require_post_onset_power_off(c)


def test_a100_launch_gates_geometry_before_gpu_and_reuses_cache():
    script=(ROOT/'tools/run_bc_reactive_a100_cpu8.sh').read_text()
    assert script.index('validate_nsga2_geometry.py')<script.index('preflight_reactive_a100.py')
    assert '--bootstrap-cache' in script
    assert '--require-power-off' in script
