from __future__ import annotations
import copy,json,math,os
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from ecsp_nsga2.workflow import NSGA2ElectricalSolidWorkflow
from ecsp_nsga2.bc_vmin import batched_voltage_search
from ecsp_native import run_native_batch,load_native
from ecsp_v6.physics.bc_global import run_bc_global_batch

ROOT=Path(__file__).resolve().parents[2]


def config():
    return yaml.safe_load((ROOT/'config/nsga2_bc_global_native_debug.yaml').read_text())

def evaluator(tmp_path, *, cfg=None):
    return NSGA2ElectricalSolidWorkflow(ROOT,cfg or config(),tmp_path).evaluator

def item(tmp_path,idx=0):
    a=np.zeros((32,32),bool);b=a.copy();a[5:8,5:26]=True;b[24:27,5:26]=True
    return a,b,{'geometry_id':f'native_case_{idx}'},tmp_path/f'case_{idx}'


def _assert_named_tensor_close(actual, expected, key):
    if actual.dtype == torch.bool or expected.dtype == torch.bool:
        assert torch.equal(actual, expected), key
        return
    atol = 1e-3 if "potential" in key.lower() else 2e-4 if "congestion" in key.lower() else 2e-5 if "anodecathodecurrentmismatch" in key.lower() else 1e-5
    rtol = 5e-4 if ("qj" in key.lower() or "qechem" in key.lower() or "heat" in key.lower()) else 2e-6
    torch.testing.assert_close(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
        msg=lambda message: f"{key}: {message}",
    )


def _batch_prefix(value, prefix_size, batch_size):
    if not isinstance(value, torch.Tensor) or value.ndim == 0:
        return value
    if value.shape[0] == batch_size:
        return value[:prefix_size]
    if value.ndim >= 2 and value.shape[1] == batch_size:
        return value[:, :prefix_size]
    return value


@pytest.mark.parametrize('voltage',[1.,20.,260.])
def test_native_reference_fp64_full_fields(tmp_path,voltage):
    e=evaluator(tmp_path/'eval');g=e._build_geometry_batch([item(tmp_path)])
    ref=run_bc_global_batch(g,e.config,e.composition,voltage,torch.float64,save_fields=True,save_handoff=True)
    got=run_native_batch(g,e.config,e.composition,voltage,torch.float64,save_fields=True,save_handoff=True)
    # Different converged linear algorithms can shift low-current potential
    # gauge slightly; compare at stricter than the original 0.01 V outer tolerance.
    for key in ref['finalFields']:
        torch.testing.assert_close(got['finalFields'][key],ref['finalFields'][key],rtol=2e-6,atol=1e-3 if key=='potential_V' else 1e-5)
    assert set(got['histories'])==set(ref['histories'])
    for key in ref['histories']:
        torch.testing.assert_close(got['histories'][key],ref['histories'][key],rtol=5e-4,atol=2e-4 if key=='currentCongestion' else .002)
    for key in ('ignitionDelay_s','remainingReactiveMassFractionAt2s','inputElectricalEnergyAt2s_J'):
        torch.testing.assert_close(got[key],ref[key],rtol=1e-5,atol=1e-9,equal_nan=True)
    assert got['nativeExecution']['compiled_time_loop']
    assert not got['nativeExecution']['trialOnly']
    assert got['nativeExecution']['completedFullHorizon'].all()
    assert got['finalFields']['cation_mol_per_m3'].dtype==torch.float64
    assert set(ref['handoffFields'])==set(got['handoffFields'])
    torch.testing.assert_close(got['handoffFields']['times_s'],ref['handoffFields']['times_s'])


def test_native_mixed_voltages_matches_independent_calls(tmp_path):
    e=evaluator(tmp_path/'eval');items=[item(tmp_path,i) for i in range(3)];g=e._build_geometry_batch(items)
    volts=[10.,20.,50.]
    batch=run_native_batch(g,e.config,e.composition,volts,save_fields=True)
    for i,v in enumerate(volts):
        single=run_native_batch(e._build_geometry_batch([items[i]]),e.config,e.composition,v,save_fields=True)
        for key in ('temperature_K','globalProgress','cation_mol_per_m3'):
            torch.testing.assert_close(batch['finalFields'][key][i],single['finalFields'][key][0],rtol=2e-6,atol=2e-5)
        torch.testing.assert_close(batch['peakCurrent_A'][i],single['peakCurrent_A'][0],rtol=.01,atol=1e-8)


def early_config(e):
    e.config['bcGlobal']['endTime_s']=.02;e.config['bcGlobal']['evaluationTime_s']=.02
    for ch in e.config['bcGlobal']['kinetics']['channels']:ch['heat_release_J_per_kg']=0.
    e.config['bcGlobal']['onsetCriterion']['temperature_K']=298.1502


def test_candidate_freezes_at_first_onset_and_trial_mode_keeps_partial_metrics_explicit(tmp_path):
    e=evaluator(tmp_path/'eval');early_config(e);g=e._build_geometry_batch([item(tmp_path,i) for i in range(3)])
    full=run_native_batch(g,e.config,e.composition,[1.,20.,260.],save_fields=True)
    short=run_native_batch(g,e.config,e.composition,[1.,20.,260.],save_fields=True,stop_on_onset=True)
    torch.testing.assert_close(short['ignitionDelay_s'],full['ignitionDelay_s'],equal_nan=True)
    configured_steps=round(e.config['bcGlobal']['endTime_s']/e.config['bcGlobal']['timeStep_s'])
    expected_steps=[
        configured_steps if not math.isfinite(float(delay))
        else round(float(delay)/e.config['bcGlobal']['timeStep_s'])
        for delay in full['ignitionDelay_s']
    ]
    assert short['nativeExecution']['stepsExecutedPerCandidate'].tolist()==expected_steps
    assert full['nativeExecution']['stepsExecutedPerCandidate'].tolist()==expected_steps
    assert full['preflameTermination']['candidateStatesFrozenAtFirstOnset']
    assert full['preflameTermination']['historyTailPolicy']==(
        'state_and_cumulative_hold_with_zero_instantaneous_tail'
    )
    assert full['nativeExecution']['historyCoversFullHorizon'].all()
    assert not full['nativeExecution']['integrationAdvancedFullHorizon'][2]
    assert torch.isnan(short['remainingReactiveMassFractionAt2s'][2])
    assert torch.isnan(short['inputElectricalEnergyAt2s_J'][2])
    assert torch.isfinite(full['inputElectricalEnergyAt2s_J']).all()
    assert short['nativeExecution']['validity_window']=='through_first_onset'


def test_onset_only_handoff_is_forbidden(tmp_path):
    e=evaluator(tmp_path/'eval');g=e._build_geometry_batch([item(tmp_path)])
    with pytest.raises(RuntimeError,match='onset-only'):
        run_native_batch(g,e.config,e.composition,20.,save_handoff=True,stop_on_onset=True)


def test_invalid_voltages_and_precision_rejected(tmp_path):
    e=evaluator(tmp_path/'eval');g=e._build_geometry_batch([item(tmp_path)])
    for value in [float('nan'),-1.,[1.,2.]]:
        with pytest.raises(ValueError):run_native_batch(g,e.config,e.composition,value)
    with pytest.raises(ValueError,match='FP64'):run_native_batch(g,e.config,e.composition,1.,torch.float32)


def test_unsupported_native_physics_never_silently_dropped(tmp_path):
    from ecsp_native.adapter import pack_config
    e=evaluator(tmp_path/'eval');c=copy.deepcopy(e.config)
    c['coupled']['usePaperMassTransferSaturation']=True
    with pytest.raises(ValueError,match='saturation'):pack_config(c,e.composition,17)
    c=copy.deepcopy(e.config);c['numerics']['potentialSolver']['preconditioner']='multigrid'
    with pytest.raises(ValueError,match='PCG'):pack_config(c,e.composition,17)


def test_native_chemistry_heat_caps_retained(tmp_path):
    e=evaluator(tmp_path/'eval');g=e._build_geometry_batch([item(tmp_path)])
    c=copy.deepcopy(e.config)
    for ch in c['bcGlobal']['kinetics']['channels']:
        ch['ln_Af_per_s']=[20.]*len(ch['alpha_grid'])
    c['bcGlobal']['thermal']['maximumTemperature_K']=298.151
    ref=run_bc_global_batch(g,c,e.composition,1.,torch.float64)
    got=run_native_batch(g,c,e.composition,1.)
    for k in ('maximumChemicalRateCapFraction','maximumTemperatureCapFraction','maximumSpeciesLimiterFraction'):
        torch.testing.assert_close(got[k],ref[k],atol=1e-12,rtol=0)
    assert got['maximumChemicalRateCapFraction'].item()>0
    assert got['maximumTemperatureCapFraction'].item()>0


def test_batched_search_cases_numerical_failure_and_censoring():
    rows=[{'ignitionSucceeded':i!=3,'converged':i!=4,'sentinel':i} for i in range(5)]
    thresholds=[15.,125.,185.,999.,100.];waves=[]
    def valid(row):return (bool(row['converged']),'valid' if row['converged'] else 'electrical_nonconvergence')
    def run(indices,volts,role):
        waves.append((indices[:],volts[:],role))
        return [dict(ignitionSucceeded=v>=thresholds[i],converged=not(i==2 and role=='bisection'),ignitionDelay_s=.1) for i,v in zip(indices,volts)]
    batched_voltage_search(rows,run,valid,enabled=True,low_voltage=20,high_voltage=260,tolerance=5,max_iterations=8,invalid_penalty=100,censor_penalty=50)
    assert rows[0]['minimumIgnitionVoltageLeftCensored']
    assert 125<=rows[1]['minimumIgnitionVoltage_V']<=130
    assert not rows[2]['minimumIgnitionVoltageSearchValid']
    assert rows[2]['minimumIgnitionVoltage_V'] is None
    assert rows[3]['minimumIgnitionVoltageRightCensored']
    assert not rows[4]['minimumIgnitionVoltageSearchValid']
    assert max(len(x[0]) for x in waves)==3
    assert all(r['sentinel']==i for i,r in enumerate(rows))


def test_vmin_real_batched_and_full_reference_preserved(tmp_path):
    cfg=config();cfg['physics']['voltage_V']=260.;cfg['evaluator']['voltage_V']=260.
    cfg['minimum_ignition_voltage_search'].update(enabled=True,lower_bound_V=20.,upper_bound_V=260.,tolerance_V=20.)
    cfg['bc_global']['endTime_s']=.02;cfg['bc_global']['evaluationTime_s']=.02
    cfg['physics']['end_time_s']=.02;cfg['evaluator']['end_time_s']=.02
    # Keep the lower 20 V bound non-igniting after conservative FV/contact
    # Joule heat replaced the old under-integrating cell-gradient estimate.
    cfg['bc_global']['onsetCriterion']['temperature_K']=298.1503
    cfg['condensed_ignition']['onset_temperature_K']=298.1503
    for ch in cfg['bc_global']['kinetics']['channels']:ch['heat_release_J_per_kg']=0.
    e=evaluator(tmp_path/'eval',cfg=cfg);items=[item(tmp_path,i) for i in range(2)]
    rows=e.evaluate_batch(items)
    logs=[json.loads(line) for line in e.wave_log.read_text().splitlines()]
    assert any(x['role']=='bisection' and x['count']==2 for x in logs)
    assert any(x['role']=='bisection' and min(x['steps_per_candidate'])<10 for x in logs)
    for row in rows:
        assert row['minimumIgnitionVoltageSearchValid']
        assert row['minimumIgnitionVoltageBracketWidth_V']<=20.
        assert row['nativeExecution']['completedFullHorizon']
        assert row['remainingReactiveMassFractionAt2s'] is not None
        assert any(t['role']=='final_upper_full_horizon_verification' for t in row['minimumIgnitionVoltageTrials'])


def test_native_full_pipeline_keeps_pareto_handoff_staggered(tmp_path):
    workflow=NSGA2ElectricalSolidWorkflow(ROOT,config(),tmp_path/'run')
    result=workflow.run();final=tmp_path/'run'/'final'
    assert result.objectives.shape==(8,)
    assert result.metrics['backend']=='bc_global_native'
    for name in ('preflame_final_pareto_designs.csv','propagation_selection.csv','all_propagation_refined_designs.csv',
                 'final_pareto_designs.csv','recommended_design.json','area_matched_staggered.json',
                 'area_matched_staggered_propagation.json','recommended_vs_area_matched_staggered_propagation.json'):
        assert (final/name).is_file()
    complete=json.loads((tmp_path/'run'/'RUN_COMPLETE.json').read_text())
    assert complete['bc_global_preflame_used'] and complete['post_onset_condensed_propagation_used']
    assert not complete['gas_phase_cfd_used']


def test_spawned_cpu_pool_uses_native_workers(tmp_path):
    from ecsp_nsga2.bc_native import NativeBCHybridEvaluator
    from ecsp_native.loader import required_cpp_standard
    cfg=config();cfg['evaluator']['backend']='bc_global_native_cpu_pool'
    cfg['evaluator']['native'].update(cpu_workers=2,host_reserve=0,cpu_budget=2)
    workflow=NSGA2ElectricalSolidWorkflow(ROOT,cfg,tmp_path/'run');e=workflow.evaluator
    try:
        rows=e.evaluate_batch([item(tmp_path,i) for i in range(4)])
        assert len(rows)==4
        standard_tag=required_cpp_standard().replace('+','p')
        assert all(r['physicsDevice']=='cpu' and r['nativeExecution']['engine']==f'{standard_tag}_cpu_fp64' for r in rows)
        assert all(r['nativeWorkerPid']!=os.getpid() for r in rows)
    finally:e.close()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA hardware unavailable: run on target A100')
def test_native_cuda_cpu_full_field_and_mixed_voltage_parity(tmp_path):
    e=evaluator(tmp_path/'cpu');g=e._build_geometry_batch([item(tmp_path,i) for i in range(3)])
    cpu=run_native_batch(g,e.config,e.composition,[10.,20.,50.],save_fields=True,save_handoff=True)
    from ecsp_v6.physics.geometry import GeometryBatch
    gg=GeometryBatch(geometry_ids=g.geometry_ids,anode=g.anode.cuda(),cathode=g.cathode.cuda(),fixed=g.fixed.cuda(),
        propellant=g.propellant.cuda(),grid_size=g.grid_size,domain_size_m=g.domain_size_m,minimum_gap_m=g.minimum_gap_m)
    gpu=run_native_batch(gg,e.config,e.composition,[10.,20.,50.],save_fields=True,save_handoff=True)
    for group in ('finalFields','histories','handoffFields'):
        assert set(gpu[group])==set(cpu[group])
        for key,value in cpu[group].items():
            other=gpu[group][key]
            if isinstance(value,torch.Tensor):
                _assert_named_tensor_close(other.cpu(),value,f'{group}.{key}')
            else:
                assert other==value,key
    for key in ('ignitionDelay_s','remainingReactiveMassFractionAtEvaluationTime',
                'peakCurrent_A','peakCurrentCongestion','inputElectricalEnergyAtEvaluationTime_J',
                'maximumTemperatureAtEvaluationTime_K','maximumThermalStabilityCFL',
                'equation32ElectricalHeatEnergyAtEvaluationTime_J','finalGlobalProgress'):
        _assert_named_tensor_close(gpu[key].cpu(),cpu[key],f'metrics.{key}')
    assert gpu['nativeExecution']['custom_cuda_kernels']


@pytest.mark.parametrize(
    ('profile','expected'),
    (
        ('nsga2_bc_global_native_a100_batch32.yaml',32),
        ('nsga2_bc_global_native_a100_batch64.yaml',64),
    ),
)
def test_native_a100_profile_gpu_subbatch_and_hybrid_wave_contract(profile,expected):
    """Profile names describe the GPU sub-batch, not GPU+CPU wave size."""
    profile_cfg=yaml.safe_load((ROOT/'config'/profile).read_text())
    assert profile_cfg['evaluator']['internal_batch_size']==expected
    assert profile_cfg['evaluator']['base_overrides']['numerics']['coupledBatchSize']==expected
    cpu_workers=profile_cfg['evaluator']['native']['cpu_workers']
    assert profile_cfg['optimization']['physics_batch_size']==expected+cpu_workers


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA hardware unavailable: run on target A100')
def test_native_cuda32_64_batch_size_invariance(tmp_path):
    for profile,expected in (
        ('nsga2_bc_global_native_a100_batch32.yaml',32),
        ('nsga2_bc_global_native_a100_batch64.yaml',64),
    ):
        profile_cfg=yaml.safe_load((ROOT/'config'/profile).read_text())
        assert profile_cfg['evaluator']['internal_batch_size']==expected
        assert profile_cfg['evaluator']['base_overrides']['numerics']['coupledBatchSize']==expected
    cfg=config();cfg['evaluator']['device']='cuda';cfg['project']['device']='cuda'
    cfg['evaluator']['base_overrides']['numerics']['physicsDevice']='cuda'
    e=evaluator(tmp_path/'gpu',cfg=cfg)
    items=[item(tmp_path,i) for i in range(64)];g=e._build_geometry_batch(items)
    voltages=[(1.,20.,260.)[i%3] for i in range(64)]
    out64=run_native_batch(g,e.config,e.composition,voltages,save_fields=True,save_handoff=True)
    out32=run_native_batch(
        e._build_geometry_batch(items[:32]),e.config,e.composition,
        voltages[:32],save_fields=True,save_handoff=True,
    )
    for group in ('finalFields','histories','handoffFields'):
        assert set(out64[group])==set(out32[group])
        for key,expected in out32[group].items():
            actual=out64[group][key]
            if isinstance(expected,torch.Tensor):
                actual=_batch_prefix(actual,32,64)
                _assert_named_tensor_close(actual,expected,f'{group}.{key}')
            else:
                assert actual==expected,key
    for key in ('ignitionDelay_s','remainingReactiveMassFractionAtEvaluationTime',
                'peakCurrent_A','peakCurrentCongestion','inputElectricalEnergyAtEvaluationTime_J',
                'maximumTemperatureAtEvaluationTime_K','maximumThermalStabilityCFL',
                'equation32ElectricalHeatEnergyAtEvaluationTime_J','finalGlobalProgress'):
        _assert_named_tensor_close(out64[key][:32],out32[key],f'metrics.{key}')
