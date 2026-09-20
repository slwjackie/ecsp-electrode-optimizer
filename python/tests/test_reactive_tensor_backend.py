"""v8.4.1: numerical identity, independent lanes, resource and CUDA contracts."""
from __future__ import annotations
import ast
import copy
import inspect
import json
import math
from pathlib import Path
import textwrap
import numpy as np
import pytest
import torch
from ecsp_reactive.condensed.validation_cases import synthetic_condensed_case as fixture
from ecsp_reactive.condensed import BCReactiveHandoffAdapter,BCReactiveSolver
from ecsp_reactive.condensed.tensor_math import (
    LOCAL_CHEMISTRY_DIAGNOSTIC_NAMES,NCONS,TensorCondensedKernel,
    conduction_and_diagonal,
)
from ecsp_reactive.condensed.tensor_solver import run_tensor_solvers,validate_execution_config
from ecsp_reactive.condensed.chemistry import (
    A1,A2,ANION,CATION,EC_LP,ENERGY,PRODUCT_WATER,PVA,RHO,WATER,
)
from ecsp_reactive.condensed.finite_volume import flux_divergence
from ecsp_nsga2.propagation import (
    PropagationConfigurationError,PropagationCandidateNumericalError,
    _div_k_grad,_thermal_diffusive_cfl,final_refinement_objectives,
)
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


def local_solvers(n=2,shape=(8,8),**kwargs):
    h,p,b,r=fixture(shape=shape,**kwargs)
    r.update(
        chemistry_integration_mode="local_adaptive_thermochemical",
        chemistry_relative_tolerance=1.0e-7,
        chemistry_absolute_tolerance=1.0e-10,
        chemistry_temperature_tolerance_K=1.0e-4,
        maximum_chemistry_corrector_iterations=8,
        maximum_chemistry_depletion_iterations=40,
        maximum_chemistry_local_refinements=10,
    )
    return [
        BCReactiveSolver(BCReactiveHandoffAdapter(p,b,r).adapt(h),p,b,r)
        for _ in range(n)
    ]


def scalar_local_kwargs(solver):
    return {
        "concentration_floor":solver.concentration_floor,
        "relative_tolerance":solver.chemistry_relative_tolerance,
        "absolute_tolerance":solver.chemistry_absolute_tolerance,
        "temperature_tolerance_K":solver.chemistry_temperature_tolerance,
        "maximum_corrector_iterations":(
            solver.max_chemistry_corrector_iterations),
        "maximum_depletion_iterations":(
            solver.max_chemistry_depletion_iterations),
        "maximum_local_refinements":solver.max_chemistry_local_refinements,
        "maximum_reaction_coordinate_steps":(
            solver.max_chemistry_reaction_coordinate_steps),
    }


def v009_coordinate_solver():
    """Small synthetic solver carrying the production V009 hard-cell tables."""
    alpha=(0.6320268988661255,0.04700627127685306)
    h,p,b,r=fixture(shape=(5,5),alpha=alpha)
    h["temperatureAtOnset_K"][:]=1188.6287368808814
    b["thermal"]["heat_capacity"]={
        "mode":"table",
        "temperature_K":[298.15,373.15,473.15,573.15,773.15],
        "values":[2200.0,2250.0,2350.0,2500.0,2700.0],
    }
    b["kinetics"]["mass_conversion_weights"]=[2.0/3.0,1.0/3.0]
    grid=[value/10.0 for value in range(11)]
    b["kinetics"]["channels"][0].update(
        alpha_grid=grid,
        activation_energy_J_per_mol=[
            85000,85000,91000,98000,105000,112000,
            119000,127000,136000,150000,150000,
        ],
        ln_Af_per_s=[
            13.0,13.0,14.5,15.5,16.5,18.0,
            19.0,20.0,21.0,23.8,23.8,
        ],
        heat_release_J_per_kg=881000.0,
    )
    b["kinetics"]["channels"][1].update(
        alpha_grid=grid,
        activation_energy_J_per_mol=[
            150000,150000,145000,140000,140000,150000,
            190000,220000,190000,150000,150000,
        ],
        ln_Af_per_s=[
            19.0,19.0,18.5,18.0,18.0,20.0,
            26.0,29.9,26.0,18.0,18.0,
        ],
        heat_release_J_per_kg=1162000.0,
    )
    r.update(
        chemistry_integration_mode="local_adaptive_thermochemical",
        chemistry_relative_tolerance=1.0e-7,
        chemistry_absolute_tolerance=1.0e-10,
        chemistry_temperature_tolerance_K=1.0e-4,
        maximum_chemistry_corrector_iterations=8,
        maximum_chemistry_depletion_iterations=40,
        maximum_chemistry_local_refinements=10,
        maximum_chemistry_reaction_coordinate_steps=64,
    )
    adapted=BCReactiveHandoffAdapter(p,b,r).adapt(h)
    return BCReactiveSolver(adapted,p,b,r)


def heterogeneous_v009_local_solver():
    """One-step field spanning cold/hot, complete, and depleted cells."""
    solver=v009_coordinate_solver()
    solver.duration=2.5e-4
    solver.dtmax=2.5e-4
    solver.snapshot_interval=2.5e-4
    solver.prop.update(
        duration_s=solver.duration,
        time_step_s=solver.dtmax,
        snapshot_interval_s=solver.snapshot_interval,
    )
    temperature=np.asarray([
        [550.,650.,750.,850.,950.],
        [1050.,1150.,1250.,1350.,1450.],
        [600.,800.,1000.,1200.,1400.],
        [700.,900.,1100.,1300.,1500.],
        [575.,775.,975.,1175.,1375.],
    ])
    alpha1=np.asarray([
        [.05,.20,.40,.632,.80],
        [.95,.999,1.0,.30,.70],
        [.10,.50,.90,.9995,.65],
        [.02,.25,.55,.85,1.0],
        [.15,.45,.75,.98,.9999],
    ])
    alpha2=np.asarray([
        [.10,.25,.35,.047,.75],
        [.90,.9995,1.0,.20,.60],
        [.05,.40,.85,.999,.55],
        [.01,.15,.50,.80,1.0],
        [.12,.30,.70,.97,.9998],
    ])
    primitive=solver.thermo.primitive(solver.U)
    primitive[...,3]=temperature
    primitive[...,A1]=alpha1
    primitive[...,A2]=alpha2
    rho=primitive[...,RHO]
    progress=(solver.chem.weights[0]*alpha1
              +solver.chem.weights[1]*alpha2)
    extent=rho*solver.chem.xi_per_kg*progress
    primitive[...,CATION]=(rho*solver.chem.initial_lp_per_kg
                           -1.45*extent)/rho
    primitive[...,ANION]=primitive[...,CATION]
    primitive[...,WATER]=solver.chem.initial_water_per_kg
    primitive[...,PVA]=(rho*solver.chem.initial_pva_per_kg-extent)/rho
    primitive[...,PRODUCT_WATER]=2.0*extent/rho
    primitive[...,EC_LP]=0.0
    solver.U=solver.thermo.conservative(primitive)
    return solver


def shared_inventory_coordinate_solver():
    alpha=(0.2880679176765724,0.18780234884152658)
    h,p,b,r=fixture(shape=(5,5),alpha=alpha)
    h["temperatureAtOnset_K"][:]=654.3292090178273
    b["thermal"]["heat_capacity"]={"mode":"constant","value":2200.0}
    b["kinetics"]["mass_conversion_weights"]=[2.0/3.0,1.0/3.0]
    b["kinetics"]["channels"][0].update(
        alpha_grid=[0.0,0.3,0.65,1.0],
        activation_energy_J_per_mol=[
            60752.57709167853,64042.73245563079,
            93550.51531785612,112096.54780396441,
        ],
        ln_Af_per_s=[
            13.619872636989093,13.354648693058708,
            20.85971808792269,23.22496209551369,
        ],
        heat_release_J_per_kg=130302.24250943401,
    )
    b["kinetics"]["channels"][1].update(
        alpha_grid=[0.0,0.3,0.65,1.0],
        activation_energy_J_per_mol=[
            113170.64105899706,91491.52460466264,
            42742.0932168346,94640.19269046221,
        ],
        ln_Af_per_s=[
            20.876023514938478,19.862414311402752,
            8.799155310372413,16.435364080705273,
        ],
        heat_release_J_per_kg=289086.1062147321,
    )
    r.update(
        chemistry_integration_mode="local_adaptive_thermochemical",
        chemistry_relative_tolerance=1.0e-7,
        chemistry_absolute_tolerance=1.0e-10,
        chemistry_temperature_tolerance_K=1.0e-4,
        maximum_chemistry_corrector_iterations=12,
        maximum_chemistry_depletion_iterations=40,
        maximum_chemistry_local_refinements=10,
        maximum_chemistry_reaction_coordinate_steps=64,
    )
    adapted=BCReactiveHandoffAdapter(p,b,r).adapt(h)
    return BCReactiveSolver(adapted,p,b,r)


def fake_local_trial(failure_reason_by_call):
    calls={"count":0}

    def trial(kernel,U,dt,first_order):
        del first_order
        call=calls["count"]
        calls["count"]+=1
        batch=U.shape[0]
        candidate=U.clone()
        integrated=torch.zeros(
            (batch,NCONS+5),dtype=torch.float64,device=U.device
        )
        transport=torch.zeros(
            (batch,3,6),dtype=torch.float64,device=U.device
        )
        fields=torch.zeros(
            (*U.shape[:3],5),dtype=torch.float64,device=U.device
        )
        valid=torch.ones(batch,dtype=torch.bool,device=U.device)
        reasons=torch.zeros(batch,dtype=torch.int64,device=U.device)
        configured=failure_reason_by_call.get(call,{})
        for lane,reason in configured.items():
            valid[lane]=False
            reasons[lane]=reason
            # A deliberately visible candidate mutation verifies that the
            # rejected lane is restored transactionally by the runner.
            candidate[lane,...,ENERGY]+=123.0
        diagnostics=torch.zeros(
            (batch,2,len(LOCAL_CHEMISTRY_DIAGNOSTIC_NAMES)),
            dtype=torch.float64,device=U.device,
        )
        diagnostics[:,0,0]=1.0
        diagnostics[:,0,1]=1.0
        for lane in configured:
            diagnostics[lane,0,1]=0.0
        kernel.trial_local_diagnostics=diagnostics
        kernel.trial_post_stability_limit=torch.where(
            reasons==2,0.4*dt,torch.full_like(dt,math.inf)
        )
        kernel.trial_post_rechecked=reasons==2
        kernel.trial_post_rejected=reasons==2
        kernel.trial_failure_reason=reasons
        return candidate,integrated,transport,fields,valid

    trial.calls=calls
    return trial


@pytest.mark.parametrize('cp',[2200.,{'mode':'constant','value':1500.},
    {'mode':'table','temperature_K':[200.,300.,400.,900.],'values':[1500.,2300.,1900.,2800.]}])
def test_tensor_thermo_inverse_and_cp_matches_reference(cp):
    h,p,b,r=fixture();b['thermal']['heat_capacity']=cp
    a=BCReactiveHandoffAdapter(p,b,r).adapt(h);s=BCReactiveSolver(a,p,b,r);k=TensorCondensedKernel([s],'cpu')
    T=np.linspace(100,1200,64).reshape(1,8,8);rho=np.linspace(1000,1000.01,64).reshape(1,8,8)
    e=s.thermo.internal_energy(rho,T);got=k.temperature(k.tensor(rho),k.tensor(e)).numpy()
    np.testing.assert_allclose(got,T,rtol=2e-14,atol=1e-10)
    np.testing.assert_allclose(k.sensible(k.tensor(T)).numpy(),s.thermo.heat.sensible_energy(T),rtol=3e-14,atol=1e-10)


def test_tensor_v009_dominant_coordinate_matches_scalar_fp64():
    s=v009_coordinate_solver();k=TensorCondensedKernel([s],"cpu")
    alpha0=np.asarray([[0.6320268988661255,0.04700627127685306]])
    temperature0=np.asarray([1188.6287368808814])
    sensible0=np.asarray(s.thermo.heat.sensible_energy(temperature0))
    duration=np.asarray([1.25e-4])
    capacity=np.asarray([0.5629799769969657])
    scalar=s.chem._solve_local_cells(
        alpha0,sensible0,temperature0,duration,capacity,s.thermo,
        relative_tolerance=s.chemistry_relative_tolerance,
        absolute_tolerance=s.chemistry_absolute_tolerance,
        temperature_tolerance_K=s.chemistry_temperature_tolerance,
        maximum_corrector_iterations=s.max_chemistry_corrector_iterations,
        maximum_depletion_iterations=s.max_chemistry_depletion_iterations,
        maximum_local_refinements=s.max_chemistry_local_refinements,
        maximum_reaction_coordinate_steps=(
            s.max_chemistry_reaction_coordinate_steps),
    )
    tensor=k._solve_local_cells(
        k.tensor(alpha0),k.tensor(sensible0),k.tensor(temperature0),
        k.tensor(duration),k.tensor(capacity),
    )
    assert bool(tensor["valid"][0])
    np.testing.assert_allclose(
        tensor["alpha"].numpy(),scalar["alpha"],rtol=0.0,atol=2.0e-8)
    np.testing.assert_allclose(
        tensor["temperature"].numpy(),scalar["temperature"],
        rtol=0.0,atol=1.0e-5)
    assert int(tensor["coupled_coordinate_cell_count"][0])==1
    assert int(tensor["coupled_coordinate_fallback_count"][0])==0
    assert int(tensor["coupled_coordinate_accepted_steps"][0])<=64
    progress=float((tensor["alpha"].numpy()[0]-alpha0[0])@s.chem.weights)
    assert progress<=capacity[0]+2.0e-15


def test_tensor_shared_inventory_event_matches_scalar_actual_time_endpoint():
    torch.set_num_threads(1)
    s=shared_inventory_coordinate_solver()
    k=TensorCondensedKernel([s],"cpu")
    alpha0=np.asarray([[0.2880679176765724,0.18780234884152658]])
    temperature0=np.asarray([654.3292090178273])
    sensible0=np.asarray(s.thermo.heat.sensible_energy(temperature0))
    duration=np.asarray([0.2])
    capacity=np.asarray([0.1708788006988188])
    scalar=s.chem._solve_local_cells(
        alpha0,sensible0,temperature0,duration,capacity,s.thermo,
        relative_tolerance=s.chemistry_relative_tolerance,
        absolute_tolerance=s.chemistry_absolute_tolerance,
        temperature_tolerance_K=s.chemistry_temperature_tolerance,
        maximum_corrector_iterations=s.max_chemistry_corrector_iterations,
        maximum_depletion_iterations=s.max_chemistry_depletion_iterations,
        maximum_local_refinements=s.max_chemistry_local_refinements,
        maximum_reaction_coordinate_steps=(
            s.max_chemistry_reaction_coordinate_steps),
    )
    tensor=k._solve_local_cells(
        k.tensor(alpha0),k.tensor(sensible0),k.tensor(temperature0),
        k.tensor(duration),k.tensor(capacity),
    )
    assert bool(tensor["valid"][0])
    assert bool(tensor["depleted"][0])
    np.testing.assert_allclose(
        tensor["alpha"].numpy(),scalar["alpha"],rtol=0.0,atol=8.0e-8)
    np.testing.assert_allclose(
        tensor["temperature"].numpy(),scalar["temperature"],
        rtol=0.0,atol=5.0e-5)
    np.testing.assert_allclose(
        tensor["depletion_time"].numpy(),scalar["depletion_time"],
        rtol=0.0,atol=5.0e-8)
    assert 0.0<=float(tensor["capacity_left"][0])<=(
        s.chemistry_absolute_tolerance
        +s.chemistry_relative_tolerance*capacity[0])
    assert int(tensor["coupled_coordinate_cell_count"][0])==1
    assert int(tensor["coupled_coordinate_fallback_count"][0])==0
    assert int(tensor["coupled_coordinate_rate_evaluations"][0])<1200


def test_tensor_coordinate_attempt_block_has_no_inner_host_decision():
    source=textwrap.dedent(inspect.getsource(
        TensorCondensedKernel._coupled_coordinate_attempt_block))
    tree=ast.parse(source)
    assert not any(isinstance(node,ast.Break) for node in ast.walk(tree))
    forbidden=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call):
            continue
        if isinstance(node.func,ast.Attribute) and node.func.attr in {
                "item","cpu","numpy"}:
            forbidden.append(node.func.attr)
        if isinstance(node.func,ast.Name) and node.func.id=="bool":
            forbidden.append(node.func.id)
    assert forbidden==[]


def test_tensor_tiny_positive_chemistry_interval_is_not_a_noop():
    gas_constant=8.31446261815324
    h,p,b,r=fixture(shape=(5,5),alpha=(0.1,0.1))
    h["temperatureAtOnset_K"][:]=300.0
    for channel,rate in zip(
            b["kinetics"]["channels"],(1.0e11,5.0e10)):
        activation=8.0e4
        channel.update(
            activation_energy_J_per_mol=[activation,activation],
            ln_Af_per_s=[
                math.log(rate)+activation/(gas_constant*300.0)
            ]*2,
            heat_release_J_per_kg=1.0e6,
        )
    r.update(
        chemistry_integration_mode="local_adaptive_thermochemical",
        chemistry_relative_tolerance=1.0e-7,
        chemistry_absolute_tolerance=1.0e-10,
        chemistry_temperature_tolerance_K=1.0e-4,
        maximum_chemistry_corrector_iterations=8,
        maximum_chemistry_depletion_iterations=40,
        maximum_chemistry_local_refinements=10,
        maximum_chemistry_reaction_coordinate_steps=64,
    )
    a=BCReactiveHandoffAdapter(p,b,r).adapt(h);s=BCReactiveSolver(a,p,b,r)
    k=TensorCondensedKernel([s],"cpu")
    after,valid,diagnostic=k.advance_local(
        k.initial,torch.tensor([1.0e-14],dtype=torch.float64))
    assert bool(valid[0])
    before_alpha=k.initial[0,0,0,A1:A2+1]/k.initial[0,0,0,RHO]
    after_alpha=after[0,0,0,A1:A2+1]/after[0,0,0,RHO]
    assert torch.all(after_alpha-before_alpha>torch.tensor(
        [1.0e-3,5.0e-4],dtype=torch.float64))
    assert diagnostic.shape[-1]==len(LOCAL_CHEMISTRY_DIAGNOSTIC_NAMES)
    row=dict(zip(LOCAL_CHEMISTRY_DIAGNOSTIC_NAMES,diagnostic[0].tolist()))
    assert row["coupled_reaction_coordinate_attempt_count"]>0
    assert row[
        "maximum_raw_rate_per_s_traversed_upper_bound"]>=row[
            "maximum_raw_rate_per_s_endpoint"]


@pytest.mark.parametrize("zero_inventory",[False,True])
def test_zero_duration_local_chemistry_is_exact_identity_without_work(
        zero_inventory):
    s=local_solvers(1,shape=(5,5))[0]
    before=s.U.copy()
    if zero_inventory:
        before[..., [CATION,ANION,PVA]]=0.0

    scalar_after,scalar_diagnostics=s.chem.advance_local(
        before,s.thermo,0.0,**scalar_local_kwargs(s))
    np.testing.assert_array_equal(scalar_after,before)
    assert scalar_diagnostics["inventory_depleted_fraction"]==0.0
    for name in (
            "endpoint_evaluation_count","evaluated_cell_count",
            "panel_attempt_count","accepted_panel_count",
            "one_active_reaction_coordinate_cell_count",
            "coupled_reaction_coordinate_attempt_count"):
        assert scalar_diagnostics[name]==0

    kernel=TensorCondensedKernel([s],"cpu")
    tensor_before=kernel.tensor(before[None,...])
    tensor_after,valid,diagnostic=kernel.advance_local(
        tensor_before,torch.zeros(1,dtype=torch.float64))
    assert bool(valid[0])
    torch.testing.assert_close(tensor_after,tensor_before,rtol=0.0,atol=0.0)
    row=dict(zip(LOCAL_CHEMISTRY_DIAGNOSTIC_NAMES,diagnostic[0].tolist()))
    assert row["attempted"]==0.0
    assert row["inventory_limiter_fraction"]==0.0
    for name in (
            "endpoint_evaluation_count","evaluated_cell_count",
            "panel_attempt_count","accepted_panel_count",
            "one_active_reaction_coordinate_cell_count",
            "coupled_reaction_coordinate_attempt_count"):
        assert row[name]==0.0


def test_negative_duration_local_chemistry_fails_transactionally():
    s=local_solvers(1,shape=(5,5))[0]
    before=s.U.copy()
    with pytest.raises(PropagationCandidateNumericalError,
                       match="Invalid local chemistry timestep"):
        s.chem.advance_local(
            before,s.thermo,-1.0e-12,**scalar_local_kwargs(s))

    kernel=TensorCondensedKernel([s],"cpu")
    tensor_before=kernel.initial.clone()
    tensor_after,valid,diagnostic=kernel.advance_local(
        tensor_before,torch.tensor([-1.0e-12],dtype=torch.float64))
    assert not bool(valid[0])
    torch.testing.assert_close(tensor_after,tensor_before,rtol=0.0,atol=0.0)
    row=dict(zip(LOCAL_CHEMISTRY_DIAGNOSTIC_NAMES,diagnostic[0].tolist()))
    assert row["attempted"]==0.0
    for name in (
            "endpoint_evaluation_count","evaluated_cell_count",
            "panel_attempt_count","accepted_panel_count",
            "one_active_reaction_coordinate_cell_count",
            "coupled_reaction_coordinate_attempt_count"):
        assert row[name]==0.0


@pytest.mark.parametrize("primitive_residue,valid_expected",[
    (-0.5e-12,True),(-2.0e-12,False),
])
def test_local_chemistry_matches_transport_trace_positivity_contract(
        primitive_residue,valid_expected):
    s=local_solvers(1,shape=(5,5))[0]
    before=s.U.copy()
    before[..., [CATION,ANION,PVA]]=(
        primitive_residue*before[...,RHO,None]
    )

    if valid_expected:
        scalar_after,_=s.chem.advance_local(
            before,s.thermo,1.0e-4,**scalar_local_kwargs(s))
        np.testing.assert_array_equal(scalar_after,before)
    else:
        with pytest.raises(PropagationCandidateNumericalError,
                           match="Invalid conversion or inventory"):
            s.chem.advance_local(
                before,s.thermo,1.0e-4,**scalar_local_kwargs(s))

    kernel=TensorCondensedKernel([s],"cpu")
    tensor_before=kernel.tensor(before[None,...])
    tensor_after,tensor_valid,_=kernel.advance_local(
        tensor_before,torch.tensor([1.0e-4],dtype=torch.float64))
    assert bool(tensor_valid[0]) is valid_expected
    torch.testing.assert_close(
        tensor_after,tensor_before,rtol=0.0,atol=0.0)


@pytest.mark.parametrize(
    ("alpha","duration","path_diagnostic"),
    [
        ((1.0,0.2),4.0e-12,
         "one_active_reaction_coordinate_cell_count"),
        ((0.2,0.2),1.0e-12,"endpoint_evaluation_count"),
    ],
)
def test_positive_shared_inventory_below_chemistry_atol_is_not_suppressed(
        alpha,duration,path_diagnostic):
    s=local_solvers(
        1,shape=(5,5),rates=(2.0,1.0),heats=(0.0,0.0),alpha=alpha
    )[0]
    progress_capacity=1.0e-12
    assert progress_capacity<s.chemistry_absolute_tolerance
    before=s.U.copy()
    extent=(before[...,RHO]*s.chem.xi_per_kg*progress_capacity)
    before[...,CATION]=before[...,ANION]=1.45*extent
    before[...,PVA]=extent

    scalar_after,scalar_diagnostics=s.chem.advance_local(
        before,s.thermo,duration,**scalar_local_kwargs(s))
    scalar_delta=(
        scalar_after[...,A1:A2+1]-before[...,A1:A2+1]
    )/before[...,RHO,None]
    scalar_progress=scalar_delta@s.chem.weights
    assert np.all(scalar_progress>0.5*progress_capacity)
    assert np.all(scalar_progress<=progress_capacity)
    assert np.all(scalar_after[...,PVA]<before[...,PVA])
    assert np.min(scalar_after[..., [CATION,ANION,PVA]])>=0.0
    assert scalar_diagnostics["inventory_depleted_fraction"]==1.0
    assert scalar_diagnostics[path_diagnostic]>0

    kernel=TensorCondensedKernel([s],"cpu")
    tensor_before=kernel.tensor(before[None,...])
    tensor_after,valid,diagnostic=kernel.advance_local(
        tensor_before,
        torch.tensor([duration],dtype=torch.float64),
    )
    assert bool(valid[0])
    tensor_delta=(
        tensor_after[...,A1:A2+1]-tensor_before[...,A1:A2+1]
    )/tensor_before[...,RHO,None]
    tensor_progress=tensor_delta@kernel.weights
    assert bool(torch.all(tensor_progress>0.5*progress_capacity))
    assert bool(torch.all(tensor_progress<=progress_capacity))
    np.testing.assert_allclose(
        tensor_after.numpy()[0],scalar_after,rtol=2.0e-15,atol=2.0e-13)
    row=dict(zip(LOCAL_CHEMISTRY_DIAGNOSTIC_NAMES,diagnostic[0].tolist()))
    assert row["inventory_limiter_fraction"]==1.0
    assert row[path_diagnostic]>0.0


def test_tensor_one_active_inventory_bound_matches_scalar_nextafter():
    start=0.6394267984578837
    weight=2.51288193118338e-16
    progress_capacity=2.4919798255746028e-17
    raw_bound=0.7385950003001831
    conservative_bound=math.nextafter(raw_bound,start)
    assert start+progress_capacity/weight==raw_bound

    s=local_solvers(
        1,shape=(5,5),rates=(0.2,0.1),heats=(0.0,0.0),
        alpha=(start,1.0),
    )[0]
    # Handoff construction was already validated with its physical weights;
    # this direct numerical-kernel regression installs the adversarial weight
    # after adaptation to isolate division-rounding at the inventory bound.
    s.chem.weights=np.asarray([weight,1.0],dtype=np.float64)
    kernel=TensorCondensedKernel([s],"cpu")
    alpha0=np.asarray([[start,1.0]],dtype=np.float64)
    temperature0=np.asarray([300.0],dtype=np.float64)
    sensible0=np.asarray(s.thermo.heat.sensible_energy(temperature0))
    scalar=s.chem._solve_one_active_reaction_coordinate_cell(
        alpha0[0],sensible0[0],temperature0[0],1.0,
        progress_capacity,0,s.thermo,
        relative_tolerance=s.chemistry_relative_tolerance,
        absolute_tolerance=s.chemistry_absolute_tolerance,
        temperature_tolerance_K=s.chemistry_temperature_tolerance,
        maximum_depletion_iterations=s.max_chemistry_depletion_iterations,
        maximum_local_refinements=s.max_chemistry_local_refinements,
    )
    tensor=kernel._solve_one_active_reaction_coordinate(
        kernel.tensor(alpha0),kernel.tensor(sensible0),
        kernel.tensor(temperature0),kernel.tensor([1.0]),
        kernel.tensor([progress_capacity]),
        torch.tensor([0],dtype=torch.int64),
        torch.tensor([True]),
    )

    assert scalar["alpha"][0]==conservative_bound
    assert float(tensor["alpha"][0,0])==conservative_bound
    assert bool(tensor["valid"][0])
    assert bool(tensor["depleted"][0])
    assert weight*(float(tensor["alpha"][0,0])-start)<=progress_capacity
    assert float(tensor["capacity_left"][0])==scalar["capacity_left"]


@pytest.mark.parametrize(
    ("capacity_factor","expected_depleted"),
    [(1.0-5.0e-9,True),(1.0+5.0e-9,False)],
)
def test_near_coincident_duration_and_inventory_events_match_scalar_tensor(
        capacity_factor,expected_depleted):
    s=local_solvers(
        1,shape=(5,5),rates=(2.0,1.0),heats=(0.0,0.0),
        alpha=(0.2,0.2),
    )[0]
    s.chemistry_relative_tolerance=1.0e-10
    s.chemistry_absolute_tolerance=1.0e-13
    s.chemistry_temperature_tolerance=1.0e-7
    s.chemistry_temperature_tolerance_K=1.0e-7
    s.max_chemistry_depletion_iterations=50
    kernel=TensorCondensedKernel([s],"cpu")
    alpha0=np.asarray([[0.2,0.2]],dtype=np.float64)
    temperature0=np.asarray([300.0],dtype=np.float64)
    sensible0=np.asarray(s.thermo.heat.sensible_energy(temperature0))
    duration=np.asarray([0.05],dtype=np.float64)
    duration_progress=float(
        duration[0]*np.asarray([2.0,1.0])@s.chem.weights
    )
    capacity=np.asarray(
        [capacity_factor*duration_progress],dtype=np.float64
    )
    scalar=s.chem._solve_local_cells(
        alpha0,sensible0,temperature0,duration,capacity,s.thermo,
        relative_tolerance=s.chemistry_relative_tolerance,
        absolute_tolerance=s.chemistry_absolute_tolerance,
        temperature_tolerance_K=s.chemistry_temperature_tolerance,
        maximum_corrector_iterations=s.max_chemistry_corrector_iterations,
        maximum_depletion_iterations=s.max_chemistry_depletion_iterations,
        maximum_local_refinements=s.max_chemistry_local_refinements,
        maximum_reaction_coordinate_steps=(
            s.max_chemistry_reaction_coordinate_steps),
    )
    tensor=kernel._solve_local_cells(
        kernel.tensor(alpha0),kernel.tensor(sensible0),
        kernel.tensor(temperature0),kernel.tensor(duration),
        kernel.tensor(capacity),
    )

    assert bool(scalar["depleted"][0]) is expected_depleted
    assert bool(tensor["depleted"][0]) is expected_depleted
    assert bool(tensor["valid"][0])
    np.testing.assert_allclose(
        tensor["alpha"].numpy(),scalar["alpha"],rtol=0.0,atol=2.0e-14)
    scalar_progress=float((scalar["alpha"][0]-alpha0[0])@s.chem.weights)
    tensor_progress=float(
        ((tensor["alpha"]-kernel.tensor(alpha0))@kernel.weights)[0]
    )
    assert scalar_progress<=capacity[0]
    assert tensor_progress<=capacity[0]
    if expected_depleted:
        assert 0.0<=float(scalar["depletion_time"][0])<=duration[0]
        assert float(tensor["depletion_time"][0])==pytest.approx(
            float(scalar["depletion_time"][0]),abs=2.0e-14)
    else:
        assert math.isnan(float(scalar["depletion_time"][0]))
        assert math.isnan(float(tensor["depletion_time"][0]))


def test_half_panel_inventory_event_roundoff_routes_both_backends_safely():
    """An event a few ulps above q/2 uses the same guarded route."""
    s=local_solvers(
        1,shape=(5,5),rates=(1.0,1.0),heats=(0.0,0.0),
        alpha=(0.2,0.2),
    )[0]
    s.chemistry_relative_tolerance=1.0e-10
    s.chemistry_absolute_tolerance=1.0e-13
    s.chemistry_temperature_tolerance=1.0e-7
    s.chemistry_temperature_tolerance_K=1.0e-7
    s.max_chemistry_depletion_iterations=50
    kernel=TensorCondensedKernel([s],"cpu")
    alpha0=np.asarray([[0.2,0.2]],dtype=np.float64)
    temperature0=np.asarray([300.0],dtype=np.float64)
    sensible0=np.asarray(s.thermo.heat.sensible_energy(temperature0))
    duration=np.asarray([0.1],dtype=np.float64)
    capacity=np.asarray([
        0.05*(1.0+4.0*np.finfo(np.float64).eps)
    ],dtype=np.float64)
    scalar=s.chem._solve_local_cells(
        alpha0,sensible0,temperature0,duration,capacity,s.thermo,
        relative_tolerance=s.chemistry_relative_tolerance,
        absolute_tolerance=s.chemistry_absolute_tolerance,
        temperature_tolerance_K=s.chemistry_temperature_tolerance,
        maximum_corrector_iterations=s.max_chemistry_corrector_iterations,
        maximum_depletion_iterations=s.max_chemistry_depletion_iterations,
        maximum_local_refinements=s.max_chemistry_local_refinements,
        maximum_reaction_coordinate_steps=(
            s.max_chemistry_reaction_coordinate_steps),
    )
    tensor=kernel._solve_local_cells(
        kernel.tensor(alpha0),kernel.tensor(sensible0),
        kernel.tensor(temperature0),kernel.tensor(duration),
        kernel.tensor(capacity),
    )

    assert bool(tensor["valid"][0])
    assert scalar["coupled_reaction_coordinate_attempt_count"][0]==1
    assert int(tensor["coupled_coordinate_attempt_count"][0])==1
    assert scalar["refinement_depth"][0]==1
    assert int(tensor["refinement_depth"][0])==1
    np.testing.assert_allclose(
        tensor["alpha"].numpy(),scalar["alpha"],rtol=0.0,atol=2.0e-14)
    assert float(tensor["depletion_time"][0])==pytest.approx(
        float(scalar["depletion_time"][0]),abs=2.0e-14)


def test_coordinate_rescue_is_not_counted_as_midpoint_refinement():
    s=local_solvers(
        1,shape=(5,5),rates=(2.0,1.0),heats=(0.0,0.0),
        alpha=(0.2,0.2),
    )[0]
    s.chemistry_relative_tolerance=1.0e-10
    s.chemistry_absolute_tolerance=1.0e-13
    s.chemistry_temperature_tolerance=1.0e-7
    s.chemistry_temperature_tolerance_K=1.0e-7
    s.max_chemistry_depletion_iterations=50
    kernel=TensorCondensedKernel([s],"cpu")
    alpha0=np.asarray([[0.2,0.2]],dtype=np.float64)
    temperature0=np.asarray([300.0],dtype=np.float64)
    sensible0=np.asarray(s.thermo.heat.sensible_energy(temperature0))
    duration=np.asarray([0.05],dtype=np.float64)
    capacity=np.asarray([
        0.5*duration[0]*float(np.asarray([2.0,1.0])@s.chem.weights)
    ])
    scalar=s.chem._solve_local_cells(
        alpha0,sensible0,temperature0,duration,capacity,s.thermo,
        relative_tolerance=s.chemistry_relative_tolerance,
        absolute_tolerance=s.chemistry_absolute_tolerance,
        temperature_tolerance_K=s.chemistry_temperature_tolerance,
        maximum_corrector_iterations=s.max_chemistry_corrector_iterations,
        maximum_depletion_iterations=s.max_chemistry_depletion_iterations,
        maximum_local_refinements=s.max_chemistry_local_refinements,
        maximum_reaction_coordinate_steps=(
            s.max_chemistry_reaction_coordinate_steps),
    )
    tensor=kernel._solve_local_cells(
        kernel.tensor(alpha0),kernel.tensor(sensible0),
        kernel.tensor(temperature0),kernel.tensor(duration),
        kernel.tensor(capacity),
    )

    assert scalar['coupled_reaction_coordinate_cell_count'][0]==1
    assert int(tensor['coupled_coordinate_cell_count'][0])==1
    assert scalar['refinement_depth'][0]==0
    assert int(tensor['refinement_depth'][0])==0
    np.testing.assert_allclose(
        tensor['alpha'].numpy(),scalar['alpha'],rtol=0.0,atol=2.0e-14)


def test_tensor_material_negative_local_increment_is_not_clamped(
        monkeypatch):
    s=local_solvers(1,shape=(5,5))[0]
    k=TensorCondensedKernel([s],"cpu")
    original=k._solve_local_cells

    def injected(alpha0,*args,**kwargs):
        result=original(alpha0,*args,**kwargs)
        result["alpha"]=result["alpha"]-1.0e-11
        return result

    monkeypatch.setattr(k,"_solve_local_cells",injected)
    before=k.initial.clone()
    after,valid,_=k.advance_local(
        before,torch.zeros(1,dtype=torch.float64))
    assert not bool(valid[0])
    torch.testing.assert_close(after,before,rtol=0.0,atol=0.0)


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


def test_full_local_heterogeneous_numpy_tensor_state_metrics_and_objectives(
        tmp_path):
    """Exercise the production mode through both complete solver runners."""
    torch.set_num_threads(1)
    base=heterogeneous_v009_local_solver()
    initial_temperature=base.thermo.primitive(base.U)[...,3]
    initial_alpha=base.U[...,A1:A2+1]/base.U[...,RHO,None]
    initial_rates=base.chem._rates_from_alpha(
        initial_alpha,initial_temperature)
    active=initial_alpha<1.0
    assert np.min(initial_rates[active])<1000.0
    assert np.max(initial_rates[active])>1000.0
    assert np.any(np.all(initial_alpha==1.0,axis=-1))
    assert np.min(base.U[...,PVA][base.U[...,PVA]>0.0])<0.02

    scalar=copy.deepcopy(base)
    tensor=copy.deepcopy(base)
    changed_threshold=copy.deepcopy(base)
    changed_threshold.chem.maximum_rate=1.0e12
    scalar_metrics=scalar.run(tmp_path/'local_numpy')
    tensor_metrics=run_tensor_solvers(
        [tensor],[tmp_path/'local_tensor'],
        {'backend':'torch_batch','device':'cpu'},
    )[0]
    changed_metrics=changed_threshold.run(tmp_path/'changed_threshold')

    # The inherited threshold is diagnostics-only: it changes the reported
    # exceedance, not the accepted state or any post-onset objective.
    np.testing.assert_array_equal(changed_threshold.U,scalar.U)
    np.testing.assert_array_equal(
        final_refinement_objectives(np.arange(4.0),changed_metrics),
        final_refinement_objectives(np.arange(4.0),scalar_metrics),
    )
    assert scalar_metrics[
        'maximumFractionAboveInheritedChemicalRateDiagnosticThreshold'
    ]>0.0
    assert changed_metrics[
        'maximumFractionAboveInheritedChemicalRateDiagnosticThreshold'
    ]==0.0

    # Full FP64 conservative and primitive states agree.  Energy is large in
    # absolute units, so retain an explicit absolute bound as well.
    np.testing.assert_allclose(
        tensor.U,scalar.U,rtol=3.0e-13,atol=1.0e-9)
    assert np.max(abs(tensor.U[...,ENERGY]-scalar.U[...,ENERGY]))<1.0e-5
    np.testing.assert_allclose(
        tensor.thermo.primitive(tensor.U),
        scalar.thermo.primitive(scalar.U),rtol=3.0e-12,atol=2.0e-9)

    exact_telemetry=(
        'acceptedTimeSteps','rejectedTimeSteps',
        'chemistryHalfStepAttempts','acceptedChemistryHalfSteps',
        'failedChemistryHalfSteps',
        'maximumChemistryCorrectorIterationsPerHalfStep',
        'maximumChemistryDepletionIterationsPerHalfStep',
        'maximumFractionAboveInheritedChemicalRateDiagnosticThreshold',
        'maximumChemicalInventoryDepletedFraction',
    )
    for key in exact_telemetry:
        assert tensor_metrics[key]==scalar_metrics[key]
    physical_metrics=(
        'finalUnreactedAreaFraction','establishedTimeAfterOnset_s',
        'meanEffectiveRegressionVelocity_m_per_s',
        'reactionFrontNonuniformity','finalMaximumTemperature_K',
        'maximumTemperatureDuringPropagation_K',
        'maximumRawChemicalRate_per_s','maximumChemistryTemperatureRise_K',
    )
    for key in physical_metrics:
        assert tensor_metrics[key]==pytest.approx(
            scalar_metrics[key],rel=1.0e-7,abs=2.0e-10)
    np.testing.assert_allclose(
        final_refinement_objectives(np.arange(4.0),tensor_metrics),
        final_refinement_objectives(np.arange(4.0),scalar_metrics),
        rtol=1.0e-7,atol=2.0e-10,
    )
    for metrics in (scalar_metrics,tensor_metrics,changed_metrics):
        assert metrics['chemistryEndpointEvaluationCount']>0
        assert metrics['chemistryEvaluatedCellCount']>0
        assert metrics['maximumAbsoluteMassBudgetResidual_kg']<1.0e-15
        assert metrics['maximumAbsoluteEnergyBudgetResidual_J']<2.0e-12
        assert metrics[
            'maximumAbsoluteEnergyPlusChemicalBudgetResidual_J'
        ]<2.0e-12
        assert metrics['maximumLocalInventoryResidual_mol_per_m3']<1.0e-10
        assert 'maximumChemicalRateCapFraction' not in metrics
        assert 'maximumChemicalInventoryLimiterFraction' not in metrics
        assert 'maximumChemistrySubcyclesPerPDEStage' not in metrics


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


def test_local_chemistry_certification_failure_is_terminal_per_lane(
        tmp_path,monkeypatch):
    states=local_solvers(2,shape=(5,5))
    initial=[state.U.copy() for state in states]
    fake=fake_local_trial({0:{0:1}})
    monkeypatch.setattr(TensorCondensedKernel,"trial",fake)

    result=run_tensor_solvers(
        states,[tmp_path/'failed_chemistry',tmp_path/'sibling'],
        {'backend':'torch_batch','device':'cpu'},
    )

    assert isinstance(result[0],PropagationCandidateNumericalError)
    assert "chemistry_failure_reason=1" in str(result[0])
    assert "retry=0" in str(result[0])
    assert result[1]["status"] == "complete"
    assert result[1]["chemicalRawRateObservationScope"] == (
        "chemistry_half_step_initial_endpoint_and_monotone_temperature_"
        "traversed_knot_upper_bound"
    )
    assert states[0].rejected_steps == 1
    np.testing.assert_array_equal(states[0].U,initial[0])
    np.testing.assert_array_equal(
        np.load(tmp_path/'failed_chemistry/INCOMPLETE_STATE.npz')["U"],
        initial[0],
    )
    assert not (tmp_path/'failed_chemistry/propagation_metrics.json').exists()
    assert (tmp_path/'sibling/propagation_metrics.json').exists()


def test_post_chemistry_stability_failure_retries_only_failed_lane(
        tmp_path,monkeypatch):
    states=local_solvers(2,shape=(5,5))
    initial=[state.U.copy() for state in states]
    fake=fake_local_trial({0:{0:2}})
    monkeypatch.setattr(TensorCondensedKernel,"trial",fake)
    monkeypatch.setattr(
        TensorCondensedKernel,"local_trial_phase_wall_times",
        lambda _kernel:(0.25,0.125),
    )

    result=run_tensor_solvers(
        states,[tmp_path/'retried',tmp_path/'sibling'],
        {'backend':'torch_batch','device':'cpu'},
    )

    assert all(value["status"] == "complete" for value in result)
    assert result[0]["rejectedTimeSteps"] == 1
    assert result[0]["postChemistryTimestepRechecks"] == 1
    assert result[0]["postChemistryTimestepRecheckRejections"] == 1
    assert result[1]["rejectedTimeSteps"] == 0
    assert result[0]["chemistryWallClockTime_s"]>(
        result[1]["chemistryWallClockTime_s"])
    assert result[0]["nonchemicalSSPRKWallClockTime_s"]>(
        result[1]["nonchemicalSSPRKWallClockTime_s"])
    np.testing.assert_array_equal(states[0].U,initial[0])
    np.testing.assert_array_equal(states[1].U,initial[1])


def test_tensor_local_phase_wall_times_are_reported(tmp_path):
    state=local_solvers(1,shape=(5,5))[0]
    metrics=run_tensor_solvers(
        [state],[tmp_path/'timed'],
        {'backend':'torch_batch','device':'cpu'},
    )[0]

    assert metrics['status']=='complete'
    assert metrics['chemistryWallClockTime_s']>0.0
    assert metrics['nonchemicalSSPRKWallClockTime_s']>0.0
    assert metrics['chemistryWallClockTime_s']<metrics['wallClockTime_s']
    assert metrics['nonchemicalSSPRKWallClockTime_s']<metrics['wallClockTime_s']


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='CUDA hardware unavailable; local chemistry event-timing preflight',
)
def test_cuda_local_phase_events_report_completed_times():
    state=local_solvers(1,shape=(5,5))[0]
    kernel=TensorCondensedKernel([state],'cuda')
    remaining=kernel.tensor([min(state.dtmax,state.duration)])
    dt=kernel.step_size(kernel.initial,remaining)
    first=torch.zeros(1,dtype=torch.bool,device=kernel.device)
    *_,valid=kernel.trial(kernel.initial,dt,first)
    assert bool(valid.cpu()[0])
    chemistry_wall,nonchemical_wall=kernel.local_trial_phase_wall_times()
    assert chemistry_wall>0.0
    assert nonchemical_wall>0.0


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


def test_cpu_worker_initializer_tolerates_missing_apple_blas_version(
        monkeypatch):
    import threadpoolctl
    from ecsp_nsga2 import post_onset_batch

    def missing_version(*_args,**_kwargs):
        raise AttributeError("Apple BLAS returned no version string")

    monkeypatch.setattr(threadpoolctl,"threadpool_limits",missing_version)
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        monkeypatch.setenv(name,post_onset_batch.os.environ.get(name,''))
    post_onset_batch._cpu_init()
    assert post_onset_batch.os.environ['OMP_NUM_THREADS']=='1'
    assert post_onset_batch.os.environ['MKL_NUM_THREADS']=='1'
    assert post_onset_batch.os.environ['OPENBLAS_NUM_THREADS']=='1'


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
