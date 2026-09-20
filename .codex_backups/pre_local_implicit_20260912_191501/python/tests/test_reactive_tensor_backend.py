"""v8.4.1: numerical identity, independent lanes, resource and CUDA contracts."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import numpy as np
import pytest
import torch
from ecsp_reactive.condensed.validation_cases import synthetic_condensed_case as fixture
from ecsp_reactive.condensed import BCReactiveHandoffAdapter,BCReactiveSolver
from ecsp_reactive.condensed.tensor_math import TensorCondensedKernel,conduction_and_diagonal
from ecsp_reactive.condensed.tensor_solver import run_tensor_solvers,validate_execution_config
from ecsp_reactive.condensed.finite_volume import flux_divergence
from ecsp_nsga2.propagation import PropagationConfigurationError,PropagationCandidateNumericalError,_div_k_grad,_thermal_diffusive_cfl
from ecsp_nsga2.post_onset import validate_post_onset_config
from ecsp_nsga2.nsga2 import Individual,assign_crowding_distance

ROOT=Path(__file__).resolve().parents[2]


def solvers(n=2,shape=(8,8),**kwargs):
    h,p,b,r=fixture(shape=shape,**kwargs)
    result=[]
    for i in range(n):
        a=BCReactiveHandoffAdapter(p,b,r).adapt(h)
        result.append(BCReactiveSolver(a,p,b,r))
    return result


@pytest.mark.parametrize('cp',[2200.,{'mode':'constant','value':1500.},
    {'mode':'table','temperature_K':[200.,300.,400.,900.],'values':[1500.,2300.,1900.,2800.]}])
def test_tensor_thermo_inverse_and_cp_matches_reference(cp):
    h,p,b,r=fixture();b['thermal']['heat_capacity']=cp
    a=BCReactiveHandoffAdapter(p,b,r).adapt(h);s=BCReactiveSolver(a,p,b,r);k=TensorCondensedKernel([s],'cpu')
    T=np.linspace(100,1200,64).reshape(1,8,8);rho=np.linspace(1000,1000.01,64).reshape(1,8,8)
    e=s.thermo.internal_energy(rho,T);got=k.temperature(k.tensor(rho),k.tensor(e)).numpy()
    np.testing.assert_allclose(got,T,rtol=2e-14,atol=1e-10)
    np.testing.assert_allclose(k.sensible(k.tensor(T)).numpy(),s.thermo.heat.sensible_energy(T),rtol=3e-14,atol=1e-10)


@pytest.mark.parametrize('kind',['hllc','hll'])
@pytest.mark.parametrize('boundary',['reflective','periodic','transmissive'])
def test_full_dynamic_tensor_flux_matches_numpy(kind,boundary):
    ss=solvers(2);rng=np.random.default_rng(841)
    for i,s in enumerate(ss):
        s.fast=False;s.riemann=kind;s.boundary=boundary
        P=s.thermo.primitive(s.U);P[...,0]*=1+1e-7*rng.random((8,8));P[...,1:3]=rng.normal(size=(8,8,2))*1e-5
        P[...,3]+=rng.random((8,8));s.U=s.thermo.conservative(P)
    k=TensorCondensedKernel(ss,'cpu');a,ba,d=k.hydro(k.initial,torch.zeros(2,dtype=torch.bool))
    for i,s in enumerate(ss):
        b,bb,db=flux_divergence(s.U,s.thermo,s.a.dx,s.a.thickness,boundary=boundary,kind=kind)
        np.testing.assert_allclose(a[i].numpy(),b,rtol=2e-11,atol=1e-7)
        np.testing.assert_allclose(ba[i].numpy(),bb,rtol=2e-11,atol=1e-12)


@pytest.mark.parametrize('n',[1,2,4])
def test_independent_batch_lanes_equal_single_numpy_runs(n,tmp_path):
    ss=solvers(n)
    for i,s in enumerate(ss):
        P=s.thermo.primitive(s.U);P[...,3]+=.5*i*np.cos(np.pi*(np.arange(8)[None,:]+.5)/8)
        s.U=s.thermo.conservative(P)
    reference=[]
    for i,s in enumerate(ss):
        c=copy.deepcopy(s);m=c.run(tmp_path/f'ref{i}');reference.append((c.U,m))
    m=run_tensor_solvers(ss,[tmp_path/f'tensor{i}' for i in range(n)],{'backend':'torch_batch','device':'cpu'})
    for i in range(n):
        assert isinstance(m[i],dict)
        np.testing.assert_allclose(ss[i].U,reference[i][0],rtol=3e-13,atol=2e-9)
        assert m[i]['maximumEnergyBudgetRelativeResidual']<1e-10
        assert m[i]['finalUnreactedAreaFraction']==reference[i][1]['finalUnreactedAreaFraction']
        assert m[i]['independentCandidateTimesteps']


def test_distinct_candidate_timestep_does_not_slow_fast_lane(tmp_path):
    h,p,b,r=fixture(rates=(.001,.001))
    # Sharp but admissible temperature-dependent k gives independent thermal dt.
    b['thermal']['thermal_conductivity']={'mode':'table','temperature_K':[300,400], 'values':[.4,2e6]}
    p.update(duration_s=1e-4,time_step_s=1e-4,snapshot_interval_s=1e-4)
    states=[]
    for temp in (300,400):
        hh=copy.deepcopy(h);hh['temperatureAtOnset_K'][:]=temp
        a=BCReactiveHandoffAdapter(p,b,r).adapt(hh);states.append(BCReactiveSolver(a,p,b,r))
    metrics=run_tensor_solvers(states,[tmp_path/'cold',tmp_path/'hot'],{'backend':'torch_batch'})
    assert all(isinstance(m,dict) for m in metrics)
    assert metrics[0]['acceptedTimeSteps']==1
    assert metrics[1]['acceptedTimeSteps']>10
    assert all(m['durationAfterOnset_s']==p['duration_s'] for m in metrics)


def test_one_failed_lane_does_not_invalidate_sibling(tmp_path):
    ss=solvers(2)
    # Narrow temperature ceiling: the cold lane is good; hot one runs out of headroom.
    for s in ss:s.thermo.tmax=300.0005
    P=ss[1].thermo.primitive(ss[1].U);P[...,3]=299.;ss[1].U=ss[1].thermo.conservative(P)
    result=run_tensor_solvers(ss,[tmp_path/'hot',tmp_path/'cold'],{'backend':'torch_batch'})
    assert isinstance(result[0],PropagationCandidateNumericalError)
    assert result[1]['status']=='complete'
    assert not (tmp_path/'hot/propagation_metrics.json').exists()
    assert (tmp_path/'hot/INCOMPLETE_STATE.npz').exists()


def test_fourier_and_local_stability_diagonal_match_reference():
    rng=np.random.default_rng(841)
    T=300+rng.random((2,8,8));k=np.exp(rng.normal(size=(2,8,8))*4);dx=.002
    q,diag=conduction_and_diagonal(torch.tensor(T),torch.tensor(k),dx)
    for i in range(2):
        np.testing.assert_allclose(q[i].numpy(),_div_k_grad(T[i],k[i],np.ones((8,8),bool),dx),rtol=5e-14,atol=1e-5)
        d=_thermal_diffusive_cfl(k[i],np.ones((8,8)),np.ones((8,8),bool),dx,1.)
        np.testing.assert_allclose(diag[i].numpy(),d,rtol=5e-14)
        assert abs(q[i].sum().item())<1e-5


def test_numpy_thermal_bound_uses_face_conductance_not_4k_heuristic():
    h,p,b,r=fixture(rates=(1e-30,1e-30),heats=(0,0))
    h['temperatureAtOnset_K'][:]=400.;h['temperatureAtOnset_K'][3,3]=300.
    b['thermal'].update(heat_capacity={'mode':'table','temperature_K':[300.,400.],'values':[1.,1e6]},
                       thermal_conductivity={'mode':'table','temperature_K':[300.,400.],'values':[1.,1000.]})
    p['time_step_s']=1.;p['duration_s']=1.
    a=BCReactiveHandoffAdapter(p,b,r).adapt(h);s=BCReactiveSolver(a,p,b,r)
    P,cp,k,*_=s._thermal_terms(s.U);dt=s.time_step(s.U,1.)
    actual=_thermal_diffusive_cfl(k,P[...,0]*cp,a.propellant_mask,a.dx,dt)
    assert actual.max()<=s.thermal_cfl*(1+1e-12)
    olddt=s.thermal_cfl/np.max(4*k/a.dx**2/(P[...,0]*cp))
    assert _thermal_diffusive_cfl(k,P[...,0]*cp,a.propellant_mask,a.dx,olddt).max()>1.


def test_exhausted_stock_does_not_force_impossible_tiny_timestep():
    ss=solvers(1,rates=(1e20,1e20));s=ss[0]
    s.U[...,6:8]=0.;s.U[...,11]=s.chem.initial_lp_per_kg*s.U[...,0]-1.45*s.chem.xi_per_kg*(s.U[...,4:6]*s.chem.weights).sum(-1)
    assert s.time_step(s.U,s.dtmax)==s.dtmax
    k=TensorCondensedKernel(ss,'cpu')
    assert float(k.step_size(k.initial,k.tensor([s.dtmax]))[0])==s.dtmax
    # This only removes an unnecessary dt constraint. Strict raw-rate cap gate
    # remains a separate modelling diagnostic and can still reject the case.


def test_constant_objectives_cannot_create_artificial_crowding_endpoints():
    F=[Individual(str(i),{},np.array([2.,2.])) for i in range(4)]
    assign_crowding_distance(F)
    assert all(x.crowding_distance==0 for x in F)


@pytest.mark.parametrize('config',[{'backend':'cupy'},{'device':'mps','backend':'torch_batch'},
    {'backend':'numpy_cpu','device':'cuda'},{'batch_size':0},{'batch_size':True},
    {'cuda_graph':'false'},{'cuda_graph':True},{'cpu_budget':8,'cpu_workers':8,'host_reserve':2},
    {'unknown_option':True}])
def test_execution_config_rejects_silent_ignores(config):
    with pytest.raises(PropagationConfigurationError):validate_execution_config(config)


@pytest.mark.parametrize('key',['stationary_mechanics_fast_path','continued_electrical_heating'])
def test_string_false_is_not_treated_as_true(key):
    h,p,b,r=fixture()
    if key=='stationary_mechanics_fast_path':r[key]='false'
    else:p[key]='false'
    a=BCReactiveHandoffAdapter(p,b,r).adapt(h)
    with pytest.raises(PropagationConfigurationError):BCReactiveSolver(a,p,b,r)


def test_no_cuda_never_silently_runs_on_cpu(tmp_path):
    if torch.cuda.is_available():pytest.skip('CPU-only failure-path test')
    with pytest.raises(PropagationConfigurationError,match='CUDA'):
        run_tensor_solvers(solvers(1),[tmp_path],{'backend':'torch_batch','device':'cuda'})


def test_memory_guard_before_tensor_allocations(tmp_path):
    with pytest.raises(PropagationConfigurationError,match='memory budget'):
        run_tensor_solvers(solvers(2),[tmp_path/'a',tmp_path/'b'],{'backend':'torch_batch','maximum_batch_working_bytes':1024})


def test_cpu_spawn_and_tensor_pair_dispatch(tmp_path):
    from ecsp_nsga2.post_onset_batch import run_post_onset_batch
    h,p,b,r=fixture()
    cfg={'backend':'reactive_euler','compare_backends':True,'reactive_euler':r,
        'execution':{'backend':'torch_batch','device':'cpu','cpu_budget':4,'host_reserve':2,'cpu_workers':2,'batch_size':2}}
    result=run_post_onset_batch([h,h],p,b,[tmp_path/'a',tmp_path/'b'],post_onset_config=cfg)
    assert all(m['postOnsetBackendComparison']['bothBackendsCompleted'] for m in result)
    report=json.loads((tmp_path/'a/post_onset_execution.json').read_text())
    assert report['process_start_method']=='spawn'
    assert report['cpu_workers']+report['host_reserve']<=report['detected_cpu_budget']


@pytest.mark.parametrize('reactive_device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA hardware unavailable; real BC/NP/BV integration GPU preflight'))])
def test_tensor_NSGA_workflow_really_dispatches_batch(tmp_path,reactive_device):
    from ecsp_nsga2.workflow import NSGA2ElectricalSolidWorkflow
    import yaml
    cfg=yaml.safe_load((ROOT/'config/nsga2_bc_reactive_debug.yaml').read_text())
    cfg['post_onset']['execution']={'backend':'torch_batch','device':reactive_device,'batch_size':4,'cpu_budget':4,'host_reserve':2,'cpu_workers':2}
    wf=NSGA2ElectricalSolidWorkflow(ROOT,cfg,tmp_path/'workflow');ind=wf.run()
    assert ind.metrics['computeBackend']==('torch_cuda_batch' if reactive_device=='cuda' else 'torch_cpu_batch')
    assert ind.metrics['postOnsetBatchSize']>1
    assert ind.metrics['postOnsetBackendComparison']['bothBackendsCompleted']
    # Reuse actual BC snapshot for a nonzero-current tensor electrical callback parity test.
    from ecsp_v6.physics.composition_model import build_composition
    hd=next((tmp_path/'workflow/final/propagation_candidates').glob('*/bc_handoff'))
    h=json.loads((hd/'bc_handoff_metadata.json').read_text())
    with np.load(hd/'bc_handoff_fields.npz') as z:h.update({name:z[name] for name in z.files})
    full=json.loads((tmp_path/'workflow/adapter/resolved_physics_config.json').read_text());full['coupled']['voltage_V']=5.
    prop=copy.deepcopy(cfg['propagation_refinement']);prop.update(domain_size_m=full['geometry']['domainSize_m'],
        surface_layer_thickness_m=full['geometry']['surfaceLayerThickness_m'],density_kg_per_m3=build_composition(full).density_kg_per_m3,
        gas_constant_J_per_molK=full['transport']['gasConstant_J_per_molK'],continued_electrical_heating=True,
        electrical_heating_mode='recomputed',duration_s=1e-5,time_step_s=5e-6,snapshot_interval_s=5e-6)
    from ecsp_reactive.condensed.tensor_solver import run_tensor_propagation_batch
    baseline_a=BCReactiveHandoffAdapter(prop,full['bcGlobal'],cfg['post_onset']['reactive_euler']).adapt(h)
    baseline=BCReactiveSolver(baseline_a,prop,full['bcGlobal'],cfg['post_onset']['reactive_euler'],full_bc_config=full)
    ref=baseline.run(tmp_path/'electric_numpy')
    result=run_tensor_propagation_batch([h,h],prop,full['bcGlobal'],[tmp_path/'electric_tensor0',tmp_path/'electric_tensor1'],
        reactive_config=cfg['post_onset']['reactive_euler'],execution_config={'backend':'torch_batch','device':reactive_device},full_bc_config=full)
    for m in result:
        assert isinstance(m,dict),str(m)
        assert m['integratedJouleHeat_J']>0 and m['integratedElectrochemicalHeat_J']>0
        assert m['postOnsetElectricalSolveCalls']==6
        assert m['integratedJouleHeat_J']==pytest.approx(ref['integratedJouleHeat_J'],rel=2e-7,abs=1e-15)
        assert m['maximumEnergyBudgetRelativeResidual']<1e-9


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA hardware unavailable; mandatory target A100 preflight')
def test_cuda_fp64_stationary_dynamic_and_batch_parity(tmp_path):
    torch.set_num_threads(1)
    for dynamic in (False,True):
        ss=solvers(2)
        for s in ss:
            s.duration=2e-7;s.dtmax=1e-8;s.snapshot_interval=2e-7
            if dynamic:
                P=s.thermo.primitive(s.U);P[...,0]*=1+1e-7*np.cos(np.arange(8)[None,:]);s.U=s.thermo.conservative(P)
        ref=copy.deepcopy(ss)
        rm=run_tensor_solvers(ref,[tmp_path/f'cpu{dynamic}{i}' for i in range(2)],{'backend':'torch_batch'})
        gm=run_tensor_solvers(ss,[tmp_path/f'gpu{dynamic}{i}' for i in range(2)],{'backend':'torch_batch','device':'cuda'})
        for a,b,m in zip(ss,ref,gm):
            assert isinstance(m,dict)
            np.testing.assert_allclose(a.U,b.U,rtol=2e-10,atol=2e-7)
            assert m['maximumEnergyBudgetRelativeResidual']<1e-9


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA hardware unavailable; mandatory CUDA graph replay test')
def test_cuda_graph_matches_eager_with_changing_timestep(tmp_path):
    for dynamic in (False,True):
        ss=solvers(2)
        for s in ss:
            if dynamic:
                s.duration=2.04e-7;s.dtmax=1e-8;s.snapshot_interval=7.3e-8
                P=s.thermo.primitive(s.U);P[...,0]*=1+1e-7*np.cos(np.arange(8)[None,:]);s.U=s.thermo.conservative(P)
            else:
                s.duration=.0104;s.snapshot_interval=.0037
        gg=copy.deepcopy(ss)
        em=run_tensor_solvers(ss,[tmp_path/f'eager{dynamic}{i}' for i in range(2)],{'backend':'torch_batch','device':'cuda'})
        gm=run_tensor_solvers(gg,[tmp_path/f'graph{dynamic}{i}' for i in range(2)],{'backend':'torch_batch','device':'cuda','cuda_graph':True})
        for a,b,m in zip(ss,gg,gm):
            assert m['cudaGraphReplayUsed']
            np.testing.assert_allclose(a.U,b.U,rtol=2e-12,atol=2e-7)


def test_hll_stationary_contact_does_not_skip_acoustic_diffusion():
    ss=solvers(1)
    s=ss[0];s.riemann='hll'
    P=s.thermo.primitive(s.U);P[...,3]+=np.cos(np.pi*(np.arange(8)[None,:]+.5)/8);s.U=s.thermo.conservative(P)
    k=TensorCondensedKernel(ss,'cpu')
    assert not k.stationary_all
    rhs,_,_=k.hydro(k.initial,torch.zeros(1,dtype=torch.bool))
    reference,_,_=flux_divergence(s.U,s.thermo,s.a.dx,s.a.thickness,kind='hll',stationary_fast_path=True)
    assert np.max(abs(reference[...,3]))>0
    np.testing.assert_allclose(rhs[0].numpy(),reference,rtol=1e-11,atol=1e-6)
    acoustic=s.cfl*s.a.dx/(2*s.thermo.sound_speed(P[...,0]).max())
    assert s.time_step(s.U,.001)<=acoustic*(1+1e-12)
    assert float(k.step_size(k.initial,k.tensor([.001]))[0])<=acoustic*(1+1e-12)


def test_preflight_rejects_skipped_cuda_results(tmp_path):
    from preflight_reactive_a100 import NODES,require_gpu_tests
    from xml.etree.ElementTree import Element,SubElement,ElementTree
    root=Element('testsuites');suite=SubElement(root,'testsuite')
    for node in NODES:SubElement(suite,'testcase',name=node.split('::')[-1])
    path=tmp_path/'result.xml';ElementTree(root).write(path)
    assert len(require_gpu_tests(path))==3
    SubElement(list(suite)[0],'skipped');ElementTree(root).write(path)
    with pytest.raises(RuntimeError):require_gpu_tests(path)
