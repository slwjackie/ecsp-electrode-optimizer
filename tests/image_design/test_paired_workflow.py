"""Mock evaluator contract tests. These tests do not claim CUDA PDE validation."""
from __future__ import annotations
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import types
import zipfile

import numpy as np
import pytest
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'))
from ecsp_image_design import paired_workflow as w
from ecsp_image_design.paired_screening import ScreeningPolicy


@dataclass
class Limits:
    domain_mm: float = 25.
    grid_size: int = 101
    target_area_fraction_per_polarity: float = .1


@dataclass
class Params:
    physics_grid_size: int = 101
    physics_area_fraction_per_polarity: float = 20/101


def scalar(gid):
    return dict(geometry_id=gid, converged=True, allElectricalLinearSolvesConverged=True,
        allNonlinearRobinSolvesConverged=True, ignitionSucceeded=False, ignitionDelay_s=None,
        minimumIgnitionVoltageSearchValid=True, minimumIgnitionVoltageLeftCensored=False,
        minimumIgnitionVoltageRightCensored=True, minimumIgnitionVoltage_V=None,
        peakMaximumTemperature_K=366.68 if gid=='E114' else 394.3,
        peakCurrentCongestionToEvaluationTime=8.185 if gid=='E114' else 35.405)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root=tmp_path/'repo';lib=root/'library';out=root/'run';lib.mkdir(parents=True)
    a=np.zeros((101,101),bool);c=a.copy();a[:,:20]=True;c[:,-20:]=True
    for sid in ('E114','R050'):
        d=lib/sid;d.mkdir()
        meta=dict(source_id=sid,status='geometry_accepted',domain_mm=25.,grid_size=101,
             anode_area_mm2=float(a.mean()*625),cathode_area_mm2=float(c.mean()*625),
             electrode_area_fraction=float(a.mean()+c.mean()),anode_components=1,cathode_components=1)
        (d/'metadata.json').write_text(json.dumps(meta));(d/'master.json').write_text('{}')
        np.savez_compressed(d/'mask.npz',anode=a,cathode=c,propellant=np.ones_like(a),domain_mm=25.)
    base={'geometry':{},'evaluator':{'device':'cuda','backend':'bc_global_native_hybrid',
          'native':{'cpu_workers':6},'base_overrides':{'numerics':{'potentialSolver':{
              'relativeToleranceCoupled':1e-9,'absoluteTolerance':1e-12,'maximumIterationsCoupled':1500}}}},
          'physics':{'voltage_V':260.,'end_time_s':2.},'post_onset':{'backend':'unchanged'}}
    cfg=root/'input.yaml';cfg.write_text(yaml.safe_dump(base))
    monkeypatch.setattr(w,'load_case',lambda p:(w.read_json(Path(p)/'metadata.json'),a,c))
    fake_utils=types.ModuleType('ecsp_image_design.five_topologies');fake_utils.gap_guard=lambda *args:3.5
    monkeypatch.setitem(sys.modules,'ecsp_image_design.five_topologies',fake_utils)
    calls=[];adapters=[];failures=set()
    class Evaluator:
        def __init__(self,adapter):
            self.device=adapter['device'];self.grid_size=adapter['grid_size'];self.voltage=260.
            self.config={'geometry':{'domainSize_m':.025},'bcGlobal':{
                'evaluationTime_s':2.,'thermal':{'initialTemperature_K':298.15}}}
        def evaluate_batch(self,items):
            ids=[x[2]['geometry_id'] for x in items];calls.append(ids)
            if ids[0] in failures:raise RuntimeError('synthetic PCG failure')
            return [scalar(x) for x in ids]
        def _trial_valid(self,row):return True,'valid'
        def close(self):pass
    def create(root,adapter,out,debug):
        adapters.append(copy.deepcopy(adapter));return Evaluator(adapter)
    def baseline(limits,**kwargs):
        assert kwargs['physics_grid_size']==101
        assert kwargs['target_area_fraction_per_polarity']==pytest.approx(a.mean())
        return None,types.SimpleNamespace(anode_mask=a,cathode_mask=c),Params()
    monkeypatch.setattr(w,'_production_dependencies',lambda:(create,baseline,Limits))
    return types.SimpleNamespace(root=root,lib=lib,out=out,cfg=cfg,calls=calls,
                                  adapters=adapters,failures=failures,base=base,a=a,c=c)


def test_same_pair_gpu_no_genetic_loop_and_config_preserved(setup):
    s=setup;before=s.cfg.read_bytes()
    rows=w.evaluate_library(s.root,s.cfg,s.lib,s.out,only_ids=['E114'],expected_count=1,pcg_max_iterations=6000)
    assert s.calls==[['E114','E114_staggered']]
    assert rows[0]['screening_valid'] and rows[0]['R_T'] < 1 and rows[0]['R_J'] < 1
    assert s.adapters[0]['backend']=='bc_global_native' and s.adapters[0]['native']['cpu_workers']==0
    assert s.adapters[0]['save_representative_fields'] is True
    ps=s.adapters[0]['base_overrides']['numerics']['potentialSolver']
    assert ps==dict(relativeToleranceCoupled=1e-9,absoluteTolerance=1e-12,maximumIterationsCoupled=6000)
    assert s.adapters[0]['physics_config']['post_onset']==s.base['post_onset']
    assert s.cfg.read_bytes()==before
    saved=w.read_json(s.out/'preflame/E114/result.json')
    assert saved['candidate_raw']['minimumIgnitionVoltage_V'] is None
    assert saved['context']['area_basis']=='solver_mask'


def test_resume_reuses_raw_but_reclassifies_new_policy(setup):
    s=setup
    w.evaluate_library(s.root,s.cfg,s.lib,s.out,only_ids=['E114'])
    rows=w.evaluate_library(s.root,s.cfg,s.lib,s.out,only_ids=['E114'],resume=True,
                            policy=ScreeningPolicy(congestion_gain=.9))
    assert len(s.calls)==1
    assert rows[0]['congestion_screen']=='no_resolved_difference'


def test_changed_mask_refuses_cached_result(setup):
    s=setup;w.evaluate_library(s.root,s.cfg,s.lib,s.out,only_ids=['E114'])
    with (s.lib/'E114/mask.npz').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='changed inputs'):
        w.evaluate_library(s.root,s.cfg,s.lib,s.out,only_ids=['E114'],resume=True)


def test_existing_run_without_resume_rejected(setup):
    s=setup;w.evaluate_library(s.root,s.cfg,s.lib,s.out,only_ids=['E114'])
    with pytest.raises(ValueError,match='existing result'):
        w.evaluate_library(s.root,s.cfg,s.lib,s.out,only_ids=['E114'])


def test_execution_failure_isolated_and_not_physical_reject(setup):
    s=setup;s.failures.add('E114')
    rows=w.evaluate_library(s.root,s.cfg,s.lib,s.out,expected_count=2)
    assert [r['comparison_state'] for r in rows]==['execution_failed','neither_ignited']
    assert (s.out/'preflame/E114/failure.json').is_file()
    assert not (s.out/'preflame/E114/raw_pair.json').exists()
    assert (s.out/'preflame/R050/raw_pair.json').is_file()
    assert w.read_json(s.out/'RUN_FINISHED.json')['execution_failures']==1


def test_retry_failed_requires_explicit_request(setup):
    s=setup;s.failures.add('E114')
    w.evaluate_library(s.root,s.cfg,s.lib,s.out,only_ids=['E114'])
    s.failures.clear()
    with pytest.raises(ValueError,match='previous execution failure'):
        w.evaluate_library(s.root,s.cfg,s.lib,s.out,only_ids=['E114'],resume=True)
    rows=w.evaluate_library(s.root,s.cfg,s.lib,s.out,only_ids=['E114'],resume=True,retry_failed=True)
    assert rows[0]['screening_valid']


def test_expected_148_does_not_silently_evaluate_two(setup):
    s=setup
    with pytest.raises(ValueError,match='Expected 148'):
        w.evaluate_library(s.root,s.cfg,s.lib,s.out,expected_count=148)
    assert not s.calls


def test_duplicate_and_missing_ids(setup):
    with pytest.raises(ValueError):w.case_ids(setup.lib,['E114','E114'])
    with pytest.raises(ValueError):w.case_ids(setup.lib,['NO_SUCH_ID'])


def test_legacy_context_recovers_saved_e114_without_pde(setup):
    s=setup;case=s.root/'old/preflame/E114';case.mkdir(parents=True)
    oldlib=s.root/'old/library/E114';oldlib.mkdir(parents=True)
    np.savez(oldlib/'mask.npz',anode=s.a,cathode=s.c,domain_mm=25.)
    area=float(s.a.mean()*625)
    ctx=dict(domain_mm=25.,grid_size=101,anode_area_mm2=area,cathode_area_mm2=area,
             electrode_area_fraction=2*area/625,minimum_gap_mm=3.,minimum_width_mm=2.,physics_config_hash='same-old')
    record=dict(source_id='E114',context=ctx,baseline_context=copy.deepcopy(ctx),
                candidate_raw=scalar('E114'),baseline_raw=scalar('E114_staggered'))
    for row in (record['candidate_raw'],record['baseline_raw']):row['external_numerical_valid']=True
    path=case/'result.json';w.write_json(path,record);before=path.read_bytes()
    w.write_json(case/'adapter/primary/resolved_physics_config.json',
        {'geometry':{'domainSize_m':.025},'coupled':{'voltage_V':260.},
         'bcGlobal':{'evaluationTime_s':2.,'thermal':{'initialTemperature_K':298.15}}})
    w.write_json(case/'baseline_parameters.json',{'physics_grid_size':101,
                  'physics_area_fraction_per_polarity':float(s.a.mean())})
    rows=w.reclassify([path],s.root/'reports')
    assert rows[0]['screening_valid'] and rows[0]['R_T'] < 1 and not s.calls
    assert path.read_bytes()==before


def test_legacy_missing_saved_context_stays_unverified(setup):
    s=setup;rec={'source_id':'E114','candidate_raw':scalar('E114'),'baseline_raw':scalar('E114_staggered')}
    path=s.root/'legacy.json';w.write_json(path,rec)
    assert w.reclassify([path],s.root/'report')[0]['comparison_state']=='invalid_pair'


def test_prepare_frozen_143_and_five_are_copied_byte_for_byte(tmp_path,monkeypatch):
    root=tmp_path/'repo';root.mkdir();zpath=tmp_path/'archive.zip'
    ids=[f'Z{i:03d}' for i in range(143)];payload=b'unchanged-geometry-evidence'
    with zipfile.ZipFile(zpath,'w') as z:
        for sid in ids:
            for name in ('mask.npz','master.json','metadata.json'):
                z.writestr(f'original/library/{sid}/{name}',payload)
    catalog={'frozen_143_ids':ids,'accepted_base_archive_sha256':[w.file_hash(zpath)]}
    w.write_json(root/'data/image_design/paired_catalogue_148.json',catalog)
    five=tmp_path/'five'
    for sid in w.FIVE_IDS:
        (five/sid).mkdir(parents=True)
        for name in ('mask.npz','master.json','metadata.json'):(five/sid/name).write_bytes(payload)
    monkeypatch.setattr(w,'load_case',lambda *args:None)
    out=tmp_path/'out';library=w.prepare_library(root,zpath,out,five)
    assert len(w.case_ids(library))==148
    for sid in ids+list(w.FIVE_IDS):
        for name in ('mask.npz','master.json','metadata.json'):
            assert (library/sid/name).read_bytes()==payload
    with pytest.raises(ValueError,match='Output exists'):w.prepare_library(root,zpath,out,five)


def test_unknown_geometry_archive_is_rejected(tmp_path):
    root=tmp_path/'repo';root.mkdir();z=tmp_path/'bad.zip';z.write_bytes(b'not the reference')
    w.write_json(root/'data/image_design/paired_catalogue_148.json',{'accepted_base_archive_sha256':['different']})
    with pytest.raises(ValueError,match='Unrecognised'):w.prepare_library(root,z,tmp_path/'out')
