from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import yaml
import networkx as nx
import numpy as np
import pytest

from ecsp_doe.geometry import (GeometryReject, _curve, graph_statistics, realize,
                               validate_topology_mask)
from ecsp_doe.sampling import (derived_seed, generate_variants, lhs_parameters,
                               parameter_bounds, parameter_key)
from ecsp_doe.topology import (_target_sequence, generate_topology_library, grammar_spec,
                               graph_signature, interdigitated_grammar_spec,
                               parametric_grammar_spec, representative_iou)
from ecsp_nsga2.geometry import GeometryLimits


@pytest.fixture(scope='module')
def limits():
    config=yaml.safe_load((Path(__file__).parents[2]/'config/nsga2_preflame_only_200x3.yaml').read_text())
    return GeometryLimits(**config['geometry'])


@pytest.fixture(scope='module')
def library(limits):
    return generate_topology_library(limits,4,20260828,193,
                                     {'maximum_topology_attempts':200, 'representative_iou_threshold':.8})


def test_library_unique_balanced_and_similarity(library):
    assert len(library)==4
    assert len({s['topology_graph_signature'] for s in library})==4
    assert [s['component_pair'] for s in library]==[[1,1],[1,1],[2,2],[2,2]]
    assert max(s['maximum_representative_iou'] for s in library)<=.8
    assert all(s['representative_validation']['passed'] for s in library)


def test_structural_signature_ignores_parameters_labels_coordinates():
    spec=grammar_spec(22,3,1)
    other=deepcopy(spec)
    other['topology_id']='unrelated'
    other['representative_parameters']={'irrelevant':32}
    other['nodes']=[[float(x)*2+1,float(y)*3] for x,y in other['nodes']]
    other['edges'].reverse()
    assert graph_signature(spec)==graph_signature(other)
    e=other['edges'][0]
    e['primitive']='SPLINE' if e['primitive']!='SPLINE' else 'LINE'
    assert graph_signature(spec)!=graph_signature(other)
    perm=list(reversed(range(len(spec['nodes']))))
    remap={old:new for new,old in enumerate(perm)}
    relabel=deepcopy(spec)
    relabel['nodes']=[spec['nodes'][i] for i in perm]
    for e in relabel['edges']:
        e['u'],e['v']=remap[e['u']],remap[e['v']]
    assert graph_signature(spec)==graph_signature(relabel)


def test_interdigitated_signature_records_graph_roles_not_embedding():
    spec=interdigitated_grammar_spec(22,3,1)
    moved=deepcopy(spec)
    moved['nodes']=[[float(x)*3+2,float(y)*5-1] for x,y in moved['nodes']]
    assert graph_signature(spec)==graph_signature(moved)
    changed=deepcopy(spec)
    changed['edges'][-1]['role']='spine'
    assert graph_signature(spec)!=graph_signature(changed)


def test_hybrid_library_contains_valid_interdigitated_candidates(limits):
    options={'topology_style':'hybrid_interdigitated','interdigitated_fraction':.5,
             'interdigitated_reach_fraction':.78,'interdigitated_curve_scale':.2,
             'require_opposed_active_edges':True,
             'minimum_opposed_edge_fraction':.45,'maximum_topology_attempts':1000,
             'representative_iou_threshold':.9}
    generated=generate_topology_library(limits,4,20260904,193,options)
    styles=[s['placement_style'] for s in generated]
    assert styles.count('interdigitated')==2
    assert styles.count('separated')==2
    assert [s['component_pair'] for s in generated].count([1,1])==2
    assert [s['component_pair'] for s in generated].count([2,2])==2
    for spec in generated:
        candidate=realize(spec,spec['representative_parameters'],limits,193,options)
        if spec['placement_style']=='interdigitated':
            metrics=candidate.metadata['interdigitation']
            assert metrics['opposed_active_edge_fraction']>=.45
            assert metrics['row_alternation_fraction']>.8
            a=np.asarray(candidate.polygons['anode'][0]['exterior'])
            c=np.asarray(candidate.polygons['cathode'][0]['exterior'])
            assert min(a[:,0].max(),c[:,0].max())>max(a[:,0].min(),c[:,0].min())


def test_shape_family_quota_is_exact_and_deterministic():
    fractions={'lattice':.55,'wave':.15,'spiral':.15,'circle':.15}
    targets=_target_sequence(100,50,5,fractions)
    assert targets==_target_sequence(100,50,5,fractions)
    families=[family for _,family in targets]
    assert {family:families.count(family) for family in set(families)}=={
        'interdigitated':5,'lattice':53,'wave':14,'spiral':14,'circle':14}
    assert [components for components,_ in targets].count(1)==50
    assert [components for components,_ in targets].count(2)==50
    assert all(components==1 for components,family in targets
               if family in ('wave','spiral','circle'))
    with pytest.raises(ValueError,match='1A1C quota'):
        _target_sequence(10,4,1,{'lattice':.1,'wave':.3,'spiral':.3,'circle':.3})


@pytest.mark.parametrize('family',['wave','spiral','circle'])
def test_parametric_family_is_true_distinct_valid_geometry(limits,family):
    spec=parametric_grammar_spec(20260916,7,1,family)
    bounds=parameter_bounds(family)
    candidate=None
    for params in lhs_parameters(32,derived_seed(20260916,'representative',family),bounds):
        try:
            candidate=realize(spec,params,limits,193,{'maximum_width_adjustment_fraction':.15})
            break
        except GeometryReject:
            pass
    assert candidate is not None
    assert candidate.metadata['shape_family']==family
    assert candidate.metadata['validation']['passed']
    assert graph_statistics(spec)['loops']==(1 if family=='circle' else 0)
    assert not np.any(candidate.anode_mask & candidate.cathode_mask)


def test_parametric_families_have_different_masks(limits):
    masks=[]
    for family in ('wave','spiral','circle'):
        spec=parametric_grammar_spec(20260916,7,1,family)
        for params in lhs_parameters(32,derived_seed(20260916,'masks',family),parameter_bounds(family)):
            try:
                masks.append(realize(spec,params,limits,193).design_anode_mask)
                break
            except GeometryReject:
                pass
        else:
            pytest.fail(f'no valid {family} geometry in deterministic sample')
    assert all(not np.array_equal(a,b) for i,a in enumerate(masks) for b in masks[i+1:])


def test_primitive_geometry_realizes_line_circle_and_spline():
    line=_curve([0,0],[5,0],'LINE',.05,1)
    arc=_curve([0,0],[5,0],'ARC',.05,1)
    spline=_curve([0,0],[5,0],'SPLINE',.05,1)
    assert np.allclose(line[:,1],0)
    assert np.max(np.abs(arc[:,1]))>.2
    assert spline[:,1].min()<0<spline[:,1].max()
    radius=5*5/(8*.25)+.25/2
    center=np.array([2.5,-(radius-.25)])
    assert np.ptp(np.linalg.norm(arc-center,axis=1))<1e-10
    assert np.allclose(arc[[0,-1]],line)
    assert np.allclose(spline[[0,-1]],line)


def test_realized_cad_and_both_rasters_preserve_graph(limits,library):
    for spec in library:
        c=realize(spec,spec['representative_parameters'],limits,193)
        assert c.anode_mask.shape==(193,193)
        assert c.design_anode_mask.shape==(96,96)
        assert not np.any(c.anode_mask & c.cathode_mask)
        assert c.metadata['topology_graph_signature']==spec['topology_graph_signature']
        report=c.metadata['validation']
        assert report['passed']
        assert report['existing_solver_grid']['violations']==[]
        expected=report['expected_graph']
        for p in ['anode','cathode']:
            cad=report['cad'][p]
            assert abs(cad['area_mm2']-70)/70<=limits.area_tolerance_fraction
            assert limits.minimum_width_mm<=cad['width_mm']<=limits.maximum_width_mm
            assert abs(cad['width_adjustment_fraction'])<=.15
            for grid in ['design','physics']:
                stats=report['rasters'][grid]['polarities'][p]
                assert all(stats[k]==expected[k] for k in expected)
                assert stats['area_relative_error']<=limits.area_tolerance_fraction
        # Every output (including CAD hole boundaries) is persistence ready.
        json.dumps(c.metadata)
        json.dumps(c.polygons)


def test_loop_filled_by_blob_is_rejected():
    loop=np.zeros((50,50),bool)
    loop[8:42,8:42]=True
    loop[15:35,15:35]=False
    expected={'components':1,'endpoints':0,'branch_points':0,'loops':1}
    validate_topology_mask(loop,expected,.1)
    blob=loop.copy();blob[15:35,15:35]=True
    with pytest.raises(GeometryReject,match='topology_loops'):
        validate_topology_mask(blob,expected,.1)


def test_branch_contact_or_branch_loss_is_rejected():
    fork=np.zeros((64,64),bool)
    fork[12:52,28:35]=True
    fork[12:19,10:54]=True
    expected={'components':1,'endpoints':3,'branch_points':1,'loops':0}
    validate_topology_mask(fork,expected,.1)
    fork[12:19,10:28]=False
    with pytest.raises(GeometryReject,match='topology_endpoints'):
        validate_topology_mask(fork,expected,.1)


def test_asymmetric_pair_fails_closed(limits,library):
    spec=deepcopy(library[0]);spec['component_pair']=[1,2]
    with pytest.raises(GeometryReject,match='asymmetric_component_pair'):
        realize(spec,spec['representative_parameters'],limits,193)


def test_lhs_reproducibility_and_stratification():
    x=lhs_parameters(8,123)
    assert x==lhs_parameters(8,123)
    assert x!=lhs_parameters(8,124)
    strata=np.floor((np.array([p['stretch_x'] for p in x])-.95)/.05*8).astype(int)
    assert sorted(strata)==list(range(8))
    assert derived_seed(1,'stage1')!=derived_seed(1,'stage2')


def test_variants_preserve_signature_and_stage2_is_new(limits,library):
    spec=library[0]
    stage1=generate_variants(spec,limits,2,derived_seed(43,'stage1'),193)
    stage2=generate_variants(spec,limits,2,derived_seed(43,'stage2'),193,
                             excluded_parameters=[c.parameter_vector for c in stage1])
    replay=generate_variants(spec,limits,2,derived_seed(43,'stage1'),193)
    assert [c.geometry_hash for c in stage1]==[c.geometry_hash for c in replay]
    assert len({parameter_key(c.parameter_vector) for c in stage1+stage2})==4
    assert len({c.metadata['physics_mask_hash'] for c in stage1+stage2})==4
    assert {c.metadata['topology_graph_signature'] for c in stage1+stage2}=={spec['topology_graph_signature']}


def test_variant_proposals_stratify_requested_count_not_attempt_budget(limits,library,monkeypatch):
    from types import SimpleNamespace
    import ecsp_doe.sampling as sampling
    proposed=[]
    def accept(spec,params,*args):
        proposed.append(params)
        return SimpleNamespace(metadata={'physics_mask_hash':str(len(proposed))})
    monkeypatch.setattr(sampling,'realize',accept)
    sampling.generate_variants(library[0],limits,3,43,193,{'maximum_attempts_per_topology':1000})
    assert len(proposed)==3
    for name,(low,high) in sampling.PARAMETER_BOUNDS.items():
        strata=np.floor((np.array([p[name] for p in proposed])-low)/(high-low)*3).astype(int)
        assert sorted(strata)==[0,1,2]


def test_attempt_limit_records_rejects(limits):
    rejects=[]
    with pytest.raises(GeometryReject,match='topology_attempt_limit'):
        generate_topology_library(limits,2,2,193,
          {'maximum_topology_attempts':1, 'maximum_width_adjustment_fraction':0},
          rejection_callback=rejects.append)
    assert rejects and rejects[0]['reason']


def test_legacy_generator_and_fitter_are_not_called(limits,library,monkeypatch):
    import ecsp_nsga2.geometry as old_geometry
    import ecsp_nsga2.geometry_fit as old_fit
    def forbidden(*args,**kwargs):
        raise AssertionError("Legacy generator/fitter used by PHIDL DOE")
    for name in ('make_topology_templates','instantiate_variant','rasterize_and_validate','_fit_area_components'):
        monkeypatch.setattr(old_geometry,name,forbidden)
    monkeypatch.setattr(old_fit,'fit_area_components',forbidden)
    spec=library[0]
    assert realize(spec,spec['representative_parameters'],limits,193).metadata['validation']['passed']
