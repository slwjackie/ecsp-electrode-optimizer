from __future__ import annotations
import copy,io,json,math,pickle,subprocess
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
from ecsp_native import run_native_batch
from ecsp_native.standalone import build_standalone,_read_tensor
from ecsp_nsga2.bc_vmin import batched_voltage_search
from ecsp_nsga2.bc_native import NativeBCGlobalEvaluator
from ecsp_v6.physics.bc_global import BCCandidateBatchError
from test_bc_native import config,evaluator,item,early_config

def test_native_compiler_standard_tracks_libtorch_requirement():
    from ecsp_native.loader import build_info,required_cpp_standard
    assert required_cpp_standard('2.13.9')=='c++17'
    assert required_cpp_standard('2.14.0a0+cu130')=='c++20'
    assert required_cpp_standard('3.0.0')=='c++20'
    info=build_info(False)
    assert info['compiler_language_standard']==required_cpp_standard()
    assert info['core_source_language_floor']=='c++17'


def test_pillow_mode1_masks_are_canonicalized_at_generation_and_resize():
    from PIL import Image,ImageDraw
    from ecsp_cuda.solver import _canonical_numpy_bool_mask
    from ecsp_nsga2.geometry import GeometryLimits,_draw_polylines
    from ecsp_v6.physics.numerics import resize_nearest_numpy

    image=Image.new('1',(8,8),0)
    ImageDraw.Draw(image).rectangle((1,1,5,5),fill=1)
    pillow_bool=np.asarray(image,dtype=bool)
    assert 0xff in set(pillow_bool.view(np.uint8).flat)

    resized=resize_nearest_numpy(pillow_bool,12)
    assert resized.dtype==np.bool_
    assert set(resized.view(np.uint8).flat)<={0,1}
    cuda_boundary=_canonical_numpy_bool_mask(pillow_bool)
    assert cuda_boundary.dtype==np.bool_
    assert set(cuda_boundary.view(np.uint8).flat)<={0,1}

    generated=_draw_polylines(
        [([(1.0,1.0),(10.0,1.0)],1.0)],
        GeometryLimits(grid_size=32),
    )
    assert np.any(generated)
    assert generated.dtype==np.bool_
    assert set(generated.view(np.uint8).flat)<={0,1}


def test_native_scalar_finite_helper_has_separate_host_and_cuda_paths():
    source=(Path(__file__).resolve().parents[2]/'cpp/bc_native/bc_scalar.h').read_text()
    assert 'BC_HD bool finite_fp64(double value)' in source
    assert '#if defined(__CUDA_ARCH__)' in source
    assert 'return ::isfinite(value);' in source
    assert 'return std::isfinite(value);' in source
    assert 'if(!finite_fp64(a)||!finite_fp64(b)' in source
    for name in ('exp','abs','pow','sqrt'):
        assert f'{name}_fp64' in source
    cpu_source=(Path(__file__).resolve().parents[2]/'cpp/bc_native/bc_ops_cpu.cpp').read_text()
    assert '[&]' not in cpu_source
    assert 'if(m==0)return out;' in cpu_source


def _noncanonical_ff_bool(mask: torch.Tensor) -> torch.Tensor:
    raw=mask.contiguous().view(torch.uint8).clone()
    raw[raw!=0]=0xff
    result=raw.view(torch.bool)
    assert set(result.view(torch.uint8).cpu().numpy().flat)<={0,0xff}
    return result


def test_native_adapter_and_direct_binding_canonicalize_ff_bool_storage(tmp_path):
    from ecsp_native.adapter import _canonical_native_bool_mask,pack_config
    from ecsp_native.loader import load_native
    from ecsp_v6.physics.geometry import GeometryBatch

    e=evaluator(tmp_path/'e')
    cfg=copy.deepcopy(e.config)
    dt=float(cfg['bcGlobal']['timeStep_s'])
    cfg['bcGlobal']['endTime_s']=dt
    cfg['bcGlobal']['evaluationTime_s']=dt
    cfg['bcGlobal']['electricalUpdateInterval_s']=dt
    cfg['bcGlobal']['onsetCriterion']['temperature_K']=1.0e9

    n=e.grid_size
    anode=torch.zeros((1,n,n),dtype=torch.bool)
    cathode=torch.zeros_like(anode)
    anode[:,:8,:]=True
    cathode[:,-8:,:]=True
    geometry=GeometryBatch(
        geometry_ids=['ff-mask'],
        anode=anode,
        cathode=cathode,
        fixed=torch.zeros_like(anode),
        propellant=torch.ones_like(anode),
        grid_size=n,
        domain_size_m=e.domain_size_m,
        minimum_gap_m=e.minimum_gap_m,
    )
    ff_geometry=replace(
        geometry,
        anode=_noncanonical_ff_bool(geometry.anode),
        cathode=_noncanonical_ff_bool(geometry.cathode),
        fixed=_noncanonical_ff_bool(geometry.fixed),
        propellant=_noncanonical_ff_bool(geometry.propellant),
    )
    noncontiguous_ff=_noncanonical_ff_bool(geometry.anode).transpose(-1,-2)
    canonicalized=_canonical_native_bool_mask(noncontiguous_ff,'test')
    assert canonicalized.is_contiguous()
    assert set(canonicalized.view(torch.uint8).numpy().flat)<={0,1}
    assert torch.equal(canonicalized,geometry.anode.transpose(-1,-2))

    expected=run_native_batch(
        geometry,cfg,e.composition,20.0,save_fields=True,
    )
    through_adapter=run_native_batch(
        ff_geometry,cfg,e.composition,20.0,save_fields=True,
    )
    for key in ('potential_V','temperature_K','globalProgress'):
        torch.testing.assert_close(
            through_adapter['finalFields'][key],
            expected['finalFields'][key],
            rtol=0.0,atol=0.0,
        )

    packed,table=pack_config(cfg,e.composition,n)
    packed.update(stop_on_onset=0.0,save_handoff=0.0,time_check=16.0,linear_check=8.0)
    voltage=torch.tensor([20.0],dtype=torch.float64)
    table_tensor=torch.tensor(table,dtype=torch.float64)
    native=load_native(False)
    assert native.version=='8.2.1'
    direct_expected=native.run(
        geometry.anode,geometry.cathode,voltage,table_tensor,packed,
    )
    direct_ff=native.run(
        ff_geometry.anode,ff_geometry.cathode,voltage,table_tensor,packed,
    )
    for key in ('state','histories','current_mismatch'):
        torch.testing.assert_close(
            direct_ff[key],direct_expected[key],rtol=0.0,atol=0.0,equal_nan=True,
        )

def test_native_time_loop_only_synchronizes_at_bounded_host_checks():
    source=(Path(__file__).resolve().parents[2]/'cpp/bc_native/bc_engine.cpp').read_text()
    marker='for(int step=0;step<steps;step++){'
    start=source.index(marker)
    brace=start+source[start:].index('{')
    depth=0
    end=None
    for index in range(brace,len(source)):
        if source[index]=='{':depth+=1
        elif source[index]=='}':
            depth-=1
            if depth==0:
                end=index+1
                break
    assert end is not None
    loop=source[start:end]
    assert '.item<' not in loop and '.item(' not in loop
    assert 'all_onset' not in loop
    assert 'pending_invalid_thermal_property' in loop
    assert 'pending_invalid_updated_thermal_property' in loop
    assert 'pending_rejected_echem' in loop
    assert 'pending_invalid_thermal_cfl' in loop
    assert 'host_check_due=(step+1)%time_check==0||step==steps-1' in loop
    assert loop.count('check_time_loop_failures(')==1

    helper=source[source.index('static TimeLoopHostStatus check_time_loop_failures'):start]
    # All flags, running state, step count and the CFL diagnostic share one
    # compact device-to-host transfer per configured check point.
    assert helper.count('.cpu()')==1

def test_native_host_check_interval_preserves_early_stop_outputs(tmp_path):
    e=evaluator(tmp_path/'e');early_config(e)
    items=[item(tmp_path,i) for i in range(3)]
    g=e._build_geometry_batch(items)

    def run(interval,stop):
        return run_native_batch(
            g,e.config,e.composition,[260.0]*3,
            save_fields=True,save_handoff=not stop,stop_on_onset=stop,
            native_options={'host_check_interval_steps':interval},
        )

    for stop in (True,False):
        immediate=run(1,stop)
        deferred=run(16,stop)
        torch.testing.assert_close(
            deferred['ignitionDelay_s'],immediate['ignitionDelay_s'],rtol=0,atol=0,
        )
        torch.testing.assert_close(
            deferred['nativeExecution']['stepsExecutedPerCandidate'],
            immediate['nativeExecution']['stepsExecutedPerCandidate'],rtol=0,atol=0,
        )
        assert deferred['time_s'].shape==immediate['time_s'].shape
        for group in ('finalFields','histories'):
            for name,expected in immediate[group].items():
                torch.testing.assert_close(
                    deferred[group][name],expected,rtol=0,atol=0,equal_nan=True,
                )
        if not stop:
            for name in (
                'times_s','qJ_W_per_m3','qEchem_W_per_m3',
                'temperatureHistory_K','globalProgressHistory',
            ):
                torch.testing.assert_close(
                    deferred['handoffFields'][name],
                    immediate['handoffFields'][name],rtol=0,atol=0,equal_nan=True,
                )

def test_adapter_step_count_matches_native_integral_ratio_rounding():
    from ecsp_native.adapter import _is_time_grid_aligned,_native_step_count
    assert 2.1/0.3>7.0  # binary FP would make a raw ceil spuriously return 8
    assert _native_step_count(2.1,0.3)==7
    assert _is_time_grid_aligned(2.1,0.3)
    assert not _is_time_grid_aligned(2.11,0.3)


def test_native_potential_assembly_handles_large_finite_conductivity(tmp_path):
    """The tensor operator must use the same overflow-safe harmonic mean."""
    from ecsp_v6.physics.bc_global import run_bc_global_batch

    e = evaluator(tmp_path / "e")
    g = e._build_geometry_batch([item(tmp_path)])
    cfg = copy.deepcopy(e.config)
    cfg["electrical"]["conductivityMaximum_S_per_m"] = 1.0e154
    cfg["bcGlobal"]["augmentedElectronicConduction"]["enabled"] = True
    cfg["bcGlobal"]["augmentedElectronicConduction"]["conductivity"] = {
        "mode": "constant",
        "value": 1.0e154,
    }
    cfg["bcGlobal"]["endTime_s"] = cfg["bcGlobal"]["timeStep_s"]
    cfg["bcGlobal"]["evaluationTime_s"] = cfg["bcGlobal"]["timeStep_s"]
    voltage = 1.0e-150

    reference = run_bc_global_batch(
        g, cfg, e.composition, voltage, torch.float64, save_fields=True
    )
    native = run_native_batch(
        g, cfg, e.composition, voltage, torch.float64, save_fields=True
    )
    for key in ("potential_V", "temperature_K", "globalProgress"):
        torch.testing.assert_close(
            native["finalFields"][key],
            reference["finalFields"][key],
            rtol=2.0e-6,
            atol=1.0e-10,
            equal_nan=True,
        )


def test_zero_initial_water_and_zero_configured_floor_keep_python_native_parity(
    tmp_path,
):
    from ecsp_v6.physics.bc_global import bc_transport_fields, run_bc_global_batch
    from ecsp_v6.physics.electrochem import initial_state

    e = evaluator(tmp_path / "e")
    composition = replace(e.composition, initial_water_mol_per_m3=0.0)
    cfg = copy.deepcopy(e.config)
    cfg["numerics"]["physicalFloors"]["concentration_mol_per_m3"] = 0.0
    cfg["bcGlobal"]["endTime_s"] = cfg["bcGlobal"]["timeStep_s"]
    cfg["bcGlobal"]["evaluationTime_s"] = cfg["bcGlobal"]["timeStep_s"]
    g = e._build_geometry_batch([item(tmp_path)])
    transport = bc_transport_fields(
        initial_state(g, cfg, composition, 1.0, torch.float64),
        g,
        cfg,
        composition,
    )
    assert torch.isfinite(transport["waterActivity"]).all()
    assert torch.count_nonzero(transport["waterActivity"]) == 0

    reference = run_bc_global_batch(
        g, cfg, composition, 1.0, torch.float64, save_fields=True
    )
    native = run_native_batch(
        g, cfg, composition, 1.0, torch.float64, save_fields=True
    )
    for key in ("temperature_K", "globalProgress", "mobileWater_mol_per_m3"):
        torch.testing.assert_close(
            native["finalFields"][key],
            reference["finalFields"][key],
            rtol=2.0e-6,
            atol=1.0e-10,
            equal_nan=True,
        )


def test_surface_outer_relaxation_below_1e3_is_not_silently_clamped(tmp_path):
    from ecsp_v6.physics.bc_global import run_bc_global_batch

    e = evaluator(tmp_path / "e")
    cfg = copy.deepcopy(e.config)
    cfg["interface"]["nonlinearRobin"]["potentialUnderRelaxation"] = 1.0e-4
    cfg["bcGlobal"]["endTime_s"] = cfg["bcGlobal"]["timeStep_s"]
    cfg["bcGlobal"]["evaluationTime_s"] = cfg["bcGlobal"]["timeStep_s"]
    g = e._build_geometry_batch([item(tmp_path)])
    reference = run_bc_global_batch(
        g, cfg, e.composition, 20.0, torch.float64, save_fields=True
    )
    native = run_native_batch(
        g, cfg, e.composition, 20.0, torch.float64, save_fields=True
    )
    torch.testing.assert_close(
        native["peakCurrent_A"],
        reference["peakCurrent_A"],
        rtol=2.0e-6,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        native["finalFields"]["potential_V"],
        reference["finalFields"]["potential_V"],
        rtol=2.0e-6,
        atol=1.0e-3,
    )

def test_native_engine_near_integral_horizon_has_no_zero_tail_step(tmp_path):
    e=evaluator(tmp_path/'e');g=e._build_geometry_batch([item(tmp_path)])
    e.config['bcGlobal']['timeStep_s']=0.3
    e.config['bcGlobal']['endTime_s']=2.1
    e.config['bcGlobal']['evaluationTime_s']=2.1
    e.config['bcGlobal']['electricalUpdateInterval_s']=0.3
    e.config['bcGlobal']['onsetCriterion']['temperature_K']=1.0e9
    e.config['bcGlobal']['thermal']['thermal_conductivity']={'mode':'constant','value':0.0}
    result=run_native_batch(g,e.config,e.composition,1.0,save_fields=True)
    assert result['time_s'].numel()==7
    assert result['nativeExecution']['stepsExecutedPerCandidate'].tolist()==[7]
    assert result['nativeExecution']['integrationAdvancedFullHorizon'].tolist()==[True]


def test_bc_resource_limits_reject_before_history_allocation(tmp_path):
    from ecsp_v6.physics.bc_global import run_bc_global_batch

    e=evaluator(tmp_path/'e');g=e._build_geometry_batch([item(tmp_path)])

    too_many_steps=copy.deepcopy(e.config)
    dt=float(too_many_steps['bcGlobal']['timeStep_s'])
    too_many_steps['bcGlobal']['endTime_s']=2.0*dt
    too_many_steps['bcGlobal']['evaluationTime_s']=2.0*dt
    too_many_steps['bcGlobal']['electricalUpdateInterval_s']=dt
    too_many_steps['numericalQuality']['maximumTimeSteps']=1
    with pytest.raises(ValueError,match='maximumTimeSteps'):
        run_bc_global_batch(g,too_many_steps,e.composition,1.0,torch.float64)
    with pytest.raises(ValueError,match='time-step count exceeds'):
        run_native_batch(g,too_many_steps,e.composition,1.0)

    too_much_history=copy.deepcopy(e.config)
    too_much_history['numericalQuality']['maximumHistoryAllocationBytes']=24*8
    with pytest.raises(ValueError,match='retained-history allocation exceeds'):
        run_bc_global_batch(g,too_much_history,e.composition,1.0,torch.float64)
    with pytest.raises(ValueError,match='retained-history allocation exceeds'):
        run_native_batch(g,too_much_history,e.composition,1.0)


def test_deferred_native_onset_snapshot_is_trimmed_to_exact_python_handoff(tmp_path):
    from ecsp_v6.physics.bc_global import run_bc_global_batch

    e=evaluator(tmp_path/'e');g=e._build_geometry_batch([item(tmp_path)])
    cfg=copy.deepcopy(e.config);dt=1.0e-14;end=2.0e-13
    cfg['bcGlobal'].update(
        timeStep_s=dt,
        endTime_s=end,
        evaluationTime_s=end,
        electricalUpdateInterval_s=dt,
        handoffSnapshotInterval_s=dt,
    )
    cfg['bcGlobal']['onsetCriterion'].update(
        temperature_K=float(cfg['thermal']['initialTemperature_K']),
        minimum_progress=1.0e-12,
        minimum_area_fraction=0.01,
    )
    cfg['bcGlobal']['kinetics']['maximum_rate_per_s']=1.0e6
    for channel in cfg['bcGlobal']['kinetics']['channels']:
        count=len(channel['alpha_grid'])
        channel['activation_energy_J_per_mol']=[0.0]*count
        channel['ln_Af_per_s']=[math.log(1.0e6)]*count
        channel['heat_release_J_per_kg']=0.0

    reference=run_bc_global_batch(
        g,cfg,e.composition,1.0,torch.float64,save_fields=True,save_handoff=True,
    )
    native=run_native_batch(
        g,cfg,e.composition,1.0,torch.float64,save_fields=True,save_handoff=True,
        native_options={'host_check_interval_steps':16},
    )
    assert reference['ignitionDelay_s'].item()==dt
    torch.testing.assert_close(
        native['ignitionDelay_s'],reference['ignitionDelay_s'],rtol=0.0,atol=0.0,
    )
    for name in ('times_s','qJ_W_per_m3','qEchem_W_per_m3',
                 'temperatureHistory_K','globalProgressHistory'):
        torch.testing.assert_close(
            native['handoffFields'][name],reference['handoffFields'][name],
            rtol=2.0e-6,atol=1.0e-10,
        )
    assert native['handoffFields']['times_s'][-1].item()==dt

def test_native_conservative_surface_joule_heat_matches_reference(tmp_path):
    from ecsp_v6.physics.bc_global import run_bc_global_batch
    e=evaluator(tmp_path/'e');g=e._build_geometry_batch([item(tmp_path)])
    e.config['electrical']['jouleHeatModel']='total_j_dot_e'
    reference=run_bc_global_batch(
        g,e.config,e.composition,260.0,torch.float64,save_fields=True,save_handoff=True,
    )
    native=run_native_batch(
        g,e.config,e.composition,260.0,torch.float64,save_fields=True,save_handoff=True,
    )
    expected=reference['handoffFields']['qJ_W_per_m3']
    actual=native['handoffFields']['qJ_W_per_m3']
    assert float(expected.max())>0.0
    assert float(reference['finalFields']['cation_mol_per_m3'].std())>0.0
    # The native and Torch PCG implementations can stop at different admissible
    # iterates inside the shared nonlinear tolerance.  Re-evaluating either
    # converged potential with the other backend's finite-volume heat formula
    # is bit-identical; allow only the resulting small local field sensitivity.
    torch.testing.assert_close(actual,expected,rtol=5e-5,atol=1e-5)
    # The physically integrated heat must remain much tighter than the local
    # field comparison, so a genuine face-counting/scaling regression cannot
    # be hidden by the local tolerance above.
    torch.testing.assert_close(
        actual[0].sum(dim=(-2,-1)),
        expected[0].sum(dim=(-2,-1)),
        rtol=1e-7,
        atol=1e-8,
    )
    # The volume mean saved by the compiled loop is the mean of the same
    # conservative in-plane-plus-contact-normal heat field.
    torch.testing.assert_close(
        actual[0].mean(dim=(-2,-1)),
        native['histories']['meanJouleHeat_W_per_m3'][0],
        rtol=1e-12,atol=1e-9,
    )
    mismatch=native['histories']['anodeCathodeCurrentMismatch']
    torch.testing.assert_close(native['maximumAnodeCathodeCurrentMismatch'],mismatch.amax(0))
    torch.testing.assert_close(native['meanAnodeCathodeCurrentMismatch'],mismatch.mean(0))
    torch.testing.assert_close(native['finalAnodeCathodeCurrentMismatch'],mismatch[-1])

def test_mixed_batch_handoff_never_replays_post_onset_sources(tmp_path):
    e=evaluator(tmp_path/'e');early_config(e)
    items=[item(tmp_path,i) for i in range(3)];g=e._build_geometry_batch(items)
    native=run_native_batch(
        g,e.config,e.composition,[1.0,20.0,260.0],torch.float64,
        save_handoff=True,
    )
    qj=native['handoffFields']['qJ_W_per_m3']
    qe=native['handoffFields']['qEchem_W_per_m3']
    assert float(qj[0,2].abs().max())>0.0
    assert not bool(torch.any(qj[1:,2]))
    assert not bool(torch.any(qe[1:,2]))

def test_standalone_executable_has_no_python_entrypoint(tmp_path):
    binary=build_standalone()
    result=subprocess.run([str(binary),'--version'],capture_output=True,text=True,check=True)
    assert 'ecsp_bc_native_cpu 8.2.1' in result.stdout
    assert 'no Python runtime' in result.stdout
    bad=tmp_path/'bad.bin';bad.write_bytes(b'bad')
    proc=subprocess.run([str(binary),str(bad),str(tmp_path/'out.bin')],capture_output=True,text=True)
    assert proc.returncode==2 and 'invalid request magic' in proc.stderr

def test_standalone_extension_same_engine_all_fields(tmp_path):
    e=evaluator(tmp_path/'e');g=e._build_geometry_batch([item(tmp_path,i) for i in range(2)])
    args=(g,e.config,e.composition,[20.,260.])
    ext=run_native_batch(*args,save_fields=True,save_handoff=True)
    cli=run_native_batch(*args,save_fields=True,save_handoff=True,native_options={'cpu_runtime':'standalone'})
    for group in ('finalFields','histories','handoffFields'):
        assert set(ext[group])==set(cli[group])
        for name,val in ext[group].items():
            if isinstance(val,torch.Tensor):torch.testing.assert_close(cli[group][name],val,rtol=0,atol=0,equal_nan=True)
    assert cli['nativeExecution']['cpu_runtime']=='standalone_executable'

def test_standalone_early_trial_matches_extension(tmp_path):
    e=evaluator(tmp_path/'e');early_config(e);g=e._build_geometry_batch([item(tmp_path,i) for i in range(3)])
    kwargs=dict(save_fields=True,stop_on_onset=True)
    a=run_native_batch(g,e.config,e.composition,[1.,20.,260.],**kwargs)
    b=run_native_batch(g,e.config,e.composition,[1.,20.,260.],native_options={'cpu_runtime':'standalone'},**kwargs)
    torch.testing.assert_close(a['ignitionDelay_s'],b['ignitionDelay_s'],rtol=0,atol=0,equal_nan=True)
    torch.testing.assert_close(a['finalFields']['temperature_K'],b['finalFields']['temperature_K'],rtol=0,atol=0)

def test_standalone_protocol_rejects_truncated_output():
    with pytest.raises(RuntimeError,match='Truncated'): _read_tensor(io.BytesIO(b''))

def test_batched_search_matches_serial_boolean_reference():
    thresholds=[1,20,20.1,61,128,129,250,260,300]
    rows=[dict(ignitionSucceeded=t<=260,converged=True) for t in thresholds]
    calls=[]
    def run(ix,vs,role):
        calls.append((ix,vs));return [dict(ignitionSucceeded=v>=thresholds[i],converged=True) for i,v in zip(ix,vs)]
    batched_voltage_search(rows,run,lambda r:(r['converged'],'ok'),enabled=True,low_voltage=20,
        high_voltage=260,tolerance=5,max_iterations=8,invalid_penalty=100,censor_penalty=50)
    for t,row in zip(thresholds,rows):
        if t>260:assert row['minimumIgnitionVoltageRightCensored'];continue
        lo,hi=20.,260.
        if t<=20:hi=20
        else:
            while hi-lo>5:
                mid=(lo+hi)/2
                if mid>=t:hi=mid
                else:lo=mid
        assert row['minimumIgnitionVoltage_V']==hi
    assert any(len(vs)>1 and len(set(vs))>1 for _,vs in calls)

def test_oom_split_retries_every_candidate_in_order(tmp_path,monkeypatch):
    # Synthetic OOM deliberately does not allocate a GPU; it exercises retries.
    e=object.__new__(NativeBCGlobalEvaluator)
    e.actual_batch_limit=8;e.device=torch.device('cpu');e.wave_log=tmp_path/'waves.jsonl'
    visits=[]
    def model(items,volts,**kw):
        if len(items)>2:raise torch.OutOfMemoryError('synthetic CUDA out of memory')
        visits.extend(items);return [dict(index=i) for i in items],None
    e._run_model=model
    rows,_=e._safe_run(list(range(17)),[1.]*17)
    assert [x['index'] for x in rows]==list(range(17))
    assert visits==list(range(17)) and e.actual_batch_limit==2

@pytest.mark.parametrize('failure_message',(
    'BC native PCG failed',
    'BC native initial PCG failed',
    'BC native nonlinear Robin',
    'BC native surface-overlay BV did not converge',
    'BC native electrochemical inventory limiter triggered',
    'BC native explicit thermal CFL limit exceeded',
    'BC native thermal properties became nonfinite/nonphysical',
    'BC native nonfinite integrated state',
    'electrode overlap',
    'minimum gap',
    'gap is below',
    'singular/nonfinite potential',
    'total explicit thermal stability CFL limit exceeded',
    'objective continuation total explicit thermal stability CFL exceeded',
    'objective continuation inventory invariant failed',
    'invalid or non-conservative B/C onset state',
    'common evaluation state has a non-finite or out-of-range value',
    'touching electrodes',
    'solver-grid failure',
    'candidate became empty',
    'minimum_gap violation',
    'unsafe geometry after physics-grid resize',
    'polarity vanished after physics-grid resize',
    'polarity overlap after physics-grid resize',
    'component topology/cap changed after B/C physics-grid resize',
    'surface-contact area constraint changed after B/C physics-grid resize',
    'narrower than the manufacturing minimum after B/C physics-grid resize',
))
def test_plain_runtime_error_messages_are_never_candidate_failures(
    tmp_path,failure_message,
):
    e=object.__new__(NativeBCGlobalEvaluator);e.actual_batch_limit=8;e.device=torch.device('cpu');e.wave_log=tmp_path/'waves.jsonl'
    def model(items,volts,**kw):
        raise RuntimeError(failure_message)
    e._run_model=model
    with pytest.raises(RuntimeError,match='.*') as caught:
        e._safe_run([(None,None,{'id':0},tmp_path/'0')],[1.])
    assert str(caught.value)==failure_message


def test_scaled_l2_norm_prevents_infinite_norm_false_convergence():
    from ecsp_v6.physics.potential import _norm_batch,_solve_pcg_system

    rhs=torch.full((1,2,2),1.0e200,dtype=torch.float64)
    norm=_norm_batch(rhs)
    assert torch.isfinite(norm).all()
    torch.testing.assert_close(
        norm,torch.tensor([2.0e200],dtype=torch.float64),rtol=2e-15,atol=0.0,
    )

    zeros=torch.zeros_like(rhs)
    active=torch.ones_like(rhs,dtype=torch.bool)
    inactive_neighbor=torch.zeros_like(active)
    system={
        'active':active,
        'active_e':inactive_neighbor,
        'active_w':inactive_neighbor,
        'active_n':inactive_neighbor,
        'active_s':inactive_neighbor,
        'diagonal':torch.ones_like(rhs),
        'ce':zeros,
        'cw':zeros,
        'cn':zeros,
        'cs':zeros,
        'b':rhs,
    }
    solution,diagnostics=_solve_pcg_system(
        zeros,system,
        {
            'iterativeRefinementRounds':1,
            'fallbackMethod':'none',
            'maximumKrylovRestarts':0,
            'convergenceCheckInterval':1,
        },
        maximum_iterations=1,
        relative_tolerance=1.0e-9,
        absolute_tolerance=1.0e-12,
    )
    assert not bool(diagnostics.converged.item())
    assert torch.isfinite(diagnostics.relative_residual).all()
    torch.testing.assert_close(
        diagnostics.relative_residual,torch.ones(1,dtype=torch.float64),
        rtol=0.0,atol=0.0,
    )
    torch.testing.assert_close(solution,zeros,rtol=0.0,atol=0.0)


def test_native_pcg_source_requires_scaled_finite_norm_convergence():
    source=(Path(__file__).resolve().parents[2]/'cpp/bc_native/bc_engine.cpp').read_text()
    assert 'static Tensor scaled_l2_norm' in source
    assert 'static Tensor finite_norm_converged' in source
    assert 'finite_norm_converged(norm,tol)&at::isfinite(relative)' in source
    assert 'sum2(original_r*original_r).sqrt()' not in source


@pytest.mark.parametrize('category',(
    'continuation_invalid_ignition_delay',
    'continuation_invalid_onset_state',
    'continuation_inventory_invariant',
    'continuation_thermal_properties',
    'continuation_thermal_stability_cfl',
    'continuation_nonfinite_temperature',
    'continuation_remaining_reactive_mass',
    'continuation_common_evaluation_state',
))
def test_structured_continuation_failure_isolated_without_message_matching(
    tmp_path,category,
):
    e=object.__new__(NativeBCGlobalEvaluator);e.actual_batch_limit=8;e.device=torch.device('cpu');e.wave_log=tmp_path/'waves.jsonl'
    def model(items,volts,**kw):
        bad=[index for index,value in enumerate(items) if value[2]['id']==2]
        if bad:
            raise BCCandidateBatchError('opaque candidate failure',bad,category)
        return [dict(index=value[2]['id'],converged=True) for value in items],None
    e._run_model=model;e._failed_geometry_row=lambda metadata,path,exc:dict(index=metadata['id'],converged=False)
    rows,_=e._safe_run([(None,None,{'id':i},tmp_path/str(i)) for i in range(5)],[1.]*5)
    assert [r['index'] for r in rows]==list(range(5))
    assert [r['converged'] for r in rows]==[True,True,False,True,True]


def test_structured_candidate_failure_keeps_metadata_when_pickled():
    original=BCCandidateBatchError('opaque',(3,1,3),'continuation_test')
    restored=pickle.loads(pickle.dumps(original))
    assert str(restored)=='opaque'
    assert restored.candidate_indices==(1,3)
    assert restored.category=='continuation_test'


def test_native_handoff_batch_quarantines_only_typed_failed_lane(tmp_path):
    e=object.__new__(NativeBCGlobalEvaluator)
    e.actual_batch_limit=8;e.handoff_batch=8;e.voltage=260.0
    e.device=torch.device('cpu');e.wave_log=tmp_path/'waves.jsonl'
    def model(items,volts,**kw):
        bad=[index for index,value in enumerate(items) if value[2]['id']==2]
        if bad:raise BCCandidateBatchError('typed handoff failure',bad,'handoff_test')
        return [dict(index=value[2]['id'],converged=True) for value in items],{}
    e._run_model=model
    e._extract_handoff_batch=lambda rows,out,items:[
        {'metrics':row,'handoff':{'marker':item[2]['id']}}
        for row,item in zip(rows,items)
    ]
    e._failed_geometry_row=lambda metadata,path,exc:dict(
        index=metadata['id'],converged=False,physicsRejected=True,
        physicsRejectionReason=str(exc),
    )
    items=[(None,None,{'id':i,'geometry_id':f'g{i}'},tmp_path/str(i)) for i in range(4)]
    results=e.evaluate_handoff_batch(items)
    assert [result.get('handoffPreparationFailed',False) for result in results]==[
        False,False,True,False,
    ]
    assert results[0]['handoff']['marker']==0 and results[3]['handoff']['marker']==3


def test_real_continuation_property_failure_quarantines_only_bad_voltage_lane(tmp_path):
    e=evaluator(tmp_path/'e');early_config(e)
    # Both lanes remain valid through native pre-flame onset.  Continuation
    # reaction heat then separates their temperatures enough that only the
    # 260 V lane crosses the deliberately extreme Arrhenius overflow limit.
    for channel in e.config['bcGlobal']['kinetics']['channels']:
        channel['heat_release_J_per_kg']=1.0e6
    e.config['bcGlobal']['thermal']['thermal_conductivity']={
        'mode':'reference_arrhenius',
        'reference_value':0.0,
        'reference_temperature_K':298.15,
        'activation_energy_J_per_mol':4.25e9,
    }
    items=[item(tmp_path,i) for i in range(2)]
    rows,_=e._safe_run(items,[20.0,260.0])
    assert rows[0]['converged']
    assert not rows[0].get('physicsRejected',False)
    assert not rows[1]['converged']
    assert rows[1]['physicsRejected']
    assert 'objective continuation thermal properties are invalid' in rows[1]['physicsRejectionReason']


def test_native_fails_closed_before_committing_rejected_faradaic_current(tmp_path):
    e=evaluator(tmp_path/'e');g=e._build_geometry_batch([item(tmp_path)])
    e.config['bcGlobal']['endTime_s']=0.002
    e.config['bcGlobal']['evaluationTime_s']=0.002
    for species,key in (
        ('water','waterStoichiometry_mol_per_molElectron'),
        ('lp','saltStoichiometry_mol_per_molElectron'),
    ):
        for polarity in ('anode','cathode'):
            e.config['interface'][species][polarity][key]=1.0e20
    with pytest.raises(RuntimeError,match='electrochemical inventory limiter triggered'):
        run_native_batch(
            g,e.config,e.composition,260.0,save_fields=True,
            native_options={'host_check_interval_steps':999},
        )

def test_deferred_native_property_failure_stays_quarantined_until_host_check(tmp_path):
    e=evaluator(tmp_path/'e');g=e._build_geometry_batch([item(tmp_path)])
    e.config['bcGlobal']['endTime_s']=0.012
    e.config['bcGlobal']['evaluationTime_s']=0.012
    e.config['bcGlobal']['onsetCriterion']['temperature_K']=1.0e9
    # Initially k=0 is valid.  The first positive temperature increment makes
    # 0*exp(large) nonfinite.  A later scheduled electrical solve must not see
    # that quarantined transport bundle and replace the intended diagnostic.
    e.config['bcGlobal']['thermal']['thermal_conductivity']={
        'mode':'reference_arrhenius',
        'reference_value':0.0,
        'reference_temperature_K':298.15,
        'activation_energy_J_per_mol':1.0e10,
    }
    with pytest.raises(
        RuntimeError,match='thermal properties became nonfinite/nonphysical',
    ):
        run_native_batch(
            g,e.config,e.composition,260.0,
            native_options={'host_check_interval_steps':999},
        )

def test_final_step_rejects_new_temperature_with_invalid_properties(tmp_path):
    from ecsp_v6.physics.bc_global import run_bc_global_batch

    e=evaluator(tmp_path/'e');g=e._build_geometry_batch([item(tmp_path)])
    dt=e.config['bcGlobal']['timeStep_s']
    e.config['bcGlobal']['endTime_s']=dt
    e.config['bcGlobal']['evaluationTime_s']=dt
    e.config['bcGlobal']['electricalUpdateInterval_s']=dt
    e.config['bcGlobal']['onsetCriterion']['temperature_K']=1.0e9
    # k(T0)=0 is admissible.  The positive one-step temperature increment
    # makes the Arrhenius exponential overflow, so the candidate final state
    # must be rejected even though no following step would re-evaluate k.
    e.config['bcGlobal']['thermal']['thermal_conductivity']={
        'mode':'reference_arrhenius',
        'reference_value':0.0,
        'reference_temperature_K':298.15,
        'activation_energy_J_per_mol':1.0e12,
    }
    with pytest.raises(
        RuntimeError,match='accepted updated temperature',
    ):
        run_bc_global_batch(
            g,e.config,e.composition,260.0,torch.float64,
        )
    with pytest.raises(
        RuntimeError,match='accepted updated temperature',
    ):
        run_native_batch(
            g,e.config,e.composition,260.0,
            native_options={'host_check_interval_steps':999},
        )


def test_final_step_rejects_invalid_updated_transport_constitutive_state(tmp_path):
    from ecsp_v6.physics.bc_global import run_bc_global_batch

    e=evaluator(tmp_path/'e');g=e._build_geometry_batch([item(tmp_path)])
    dt=e.config['bcGlobal']['timeStep_s']
    e.config['bcGlobal']['endTime_s']=dt
    e.config['bcGlobal']['evaluationTime_s']=dt
    e.config['bcGlobal']['electricalUpdateInterval_s']=dt
    e.config['bcGlobal']['onsetCriterion']['temperature_K']=1.0e9
    # D(T0)=0 is admissible, while the first positive temperature increment
    # makes 0*exp(large) nonfinite.  The final accepted state must validate the
    # complete constitutive bundle, not only cp and k.
    e.config['bcGlobal']['transport']['cation_diffusivity']={
        'mode':'reference_arrhenius',
        'reference_value':0.0,
        'reference_temperature_K':298.15,
        'activation_energy_J_per_mol':1.0e12,
    }
    for runner in (
        lambda: run_bc_global_batch(
            g,e.config,e.composition,260.0,torch.float64,
        ),
        lambda: run_native_batch(
            g,e.config,e.composition,260.0,
            native_options={'host_check_interval_steps':999},
        ),
    ):
        with pytest.raises(
            RuntimeError,match='transport constitutive properties.*accepted updated temperature',
        ):
            runner()


def test_nonfinite_bv_arithmetic_never_becomes_zero_current(tmp_path):
    from ecsp_v6.physics.bc_global import run_bc_global_batch

    e=evaluator(tmp_path/'e');g=e._build_geometry_batch([item(tmp_path)])
    dt=e.config['bcGlobal']['timeStep_s']
    e.config['bcGlobal']['endTime_s']=dt
    e.config['bcGlobal']['evaluationTime_s']=dt
    e.config['bcGlobal']['electricalUpdateInterval_s']=dt
    e.config['bcGlobal']['onsetCriterion']['temperature_K']=1.0e9
    e.config['interface']['exponentialArgumentLimit']=1000.0
    for species in ('water','lp'):
        for polarity in ('anode','cathode'):
            e.config['interface'][species][polarity][
                'exchangeCurrentDensity_A_per_m2'
            ]=1.0e308

    with pytest.raises(RuntimeError,match='singular or non-finite'):
        run_bc_global_batch(g,e.config,e.composition,260.0,torch.float64)
    # Either the nonfinite boundary linearisation is rejected by PCG or it
    # reaches the nonlinear surface residual guard; both are valid fail-closed
    # outcomes and, crucially, neither accepts it as zero current.
    with pytest.raises(
        RuntimeError,match='PCG failed|surface-overlay BV did not converge',
    ):
        run_native_batch(g,e.config,e.composition,260.0)


def test_harmonic_mean_is_overflow_safe_and_invalid_inputs_propagate():
    from ecsp_v6.physics.numerics import harmonic_mean

    tiny=torch.nextafter(torch.tensor(0.0,dtype=torch.float64),torch.tensor(1.0,dtype=torch.float64))
    values=torch.tensor([0.0,float(tiny),1.0,1.0e-300,1.0e308],dtype=torch.float64)
    torch.testing.assert_close(harmonic_mean(values,values),values,rtol=0.0,atol=0.0)
    unequal=harmonic_mean(
        torch.tensor([1.0e-300],dtype=torch.float64),
        torch.tensor([1.0e308],dtype=torch.float64),
    )
    torch.testing.assert_close(
        unequal,torch.tensor([2.0e-300],dtype=torch.float64),rtol=0.0,atol=0.0,
    )
    invalid=harmonic_mean(
        torch.tensor([-1.0,torch.inf,torch.nan],dtype=torch.float64),
        torch.ones(3,dtype=torch.float64),
    )
    assert bool(torch.all(torch.isnan(invalid)))


def test_extreme_finite_limiter_arithmetic_is_not_silently_clipped(tmp_path):
    from ecsp_v6.physics.bc_global import run_bc_global_batch
    from ecsp_v6.physics.species import _limited_update

    e=evaluator(tmp_path/'e');g=e._build_geometry_batch([item(tmp_path)])
    cfg=copy.deepcopy(e.config)
    dt=float(cfg['bcGlobal']['timeStep_s'])
    cfg['bcGlobal']['endTime_s']=dt
    cfg['bcGlobal']['evaluationTime_s']=dt
    cfg['bcGlobal']['electricalUpdateInterval_s']=dt
    cfg['bcGlobal']['onsetCriterion']['temperature_K']=1.0e9
    # Every configured value is finite, but max_delta * concentration
    # overflows.  Clamping with an infinite limit used to make that invalid
    # limiter arithmetic indistinguishable from an accepted update.
    cfg['transport']['maximumRelativeConcentrationChangePerStep']=1.0e308
    probe=torch.tensor([2.0],dtype=torch.float64)
    updated,limited=_limited_update(probe,torch.zeros_like(probe),dt,cfg,2.0)
    assert bool(torch.isnan(updated).all())
    assert bool(limited.all())

    with pytest.raises(RuntimeError,match='non-finite state'):
        run_bc_global_batch(g,cfg,e.composition,20.0,torch.float64)
    with pytest.raises(RuntimeError,match='nonfinite integrated state'):
        run_native_batch(
            g,cfg,e.composition,20.0,
            native_options={'host_check_interval_steps':999},
        )


def test_extreme_finite_thermal_energy_is_not_silently_clamped(tmp_path):
    from ecsp_v6.physics.bc_global import run_bc_global_batch

    e=evaluator(tmp_path/'e');g=e._build_geometry_batch([item(tmp_path)])
    cfg=copy.deepcopy(e.config)
    dt=float(cfg['bcGlobal']['timeStep_s'])
    cfg['bcGlobal']['endTime_s']=dt
    cfg['bcGlobal']['evaluationTime_s']=dt
    cfg['bcGlobal']['electricalUpdateInterval_s']=dt
    cfg['bcGlobal']['onsetCriterion']['temperature_K']=1.0e9
    # Finite inputs produce an overflowing chemical-energy sum.  The raw
    # temperature is therefore nonfinite and must fail instead of being
    # silently saturated at maximumTemperature_K.
    for channel in cfg['bcGlobal']['kinetics']['channels']:
        channel['heat_release_J_per_kg']=1.0e308
        channel['ln_Af_per_s']=[60.0]*len(channel['alpha_grid'])

    with pytest.raises(RuntimeError,match='energy update produced a non-finite state'):
        run_bc_global_batch(g,cfg,e.composition,20.0,torch.float64)
    with pytest.raises(RuntimeError,match='nonfinite integrated state'):
        run_native_batch(
            g,cfg,e.composition,20.0,
            native_options={'host_check_interval_steps':999},
        )


def test_extreme_finite_thermal_conductivity_fails_cfl_not_as_insulation(tmp_path):
    from ecsp_v6.physics.bc_global import run_bc_global_batch

    e=evaluator(tmp_path/'e');g=e._build_geometry_batch([item(tmp_path)])
    dt=e.config['bcGlobal']['timeStep_s']
    e.config['bcGlobal']['endTime_s']=dt
    e.config['bcGlobal']['evaluationTime_s']=dt
    e.config['bcGlobal']['electricalUpdateInterval_s']=dt
    e.config['bcGlobal']['onsetCriterion']['temperature_K']=1.0e9
    e.config['bcGlobal']['thermal']['thermal_conductivity']={
        'mode':'constant','value':1.0e308,
    }
    with pytest.raises(RuntimeError,match='thermal stability CFL'):
        run_bc_global_batch(g,e.config,e.composition,1.0,torch.float64)
    with pytest.raises(RuntimeError,match='thermal stability CFL'):
        run_native_batch(g,e.config,e.composition,1.0)

def test_congestion_objective_is_bounded_by_evaluation_time(tmp_path):
    from ecsp_v6.physics.bc_global import run_bc_global_batch

    e=evaluator(tmp_path/'e');g=e._build_geometry_batch([item(tmp_path)])
    e.config['bcGlobal']['endTime_s']=0.012
    e.config['bcGlobal']['evaluationTime_s']=0.004
    e.config['bcGlobal']['onsetCriterion']['temperature_K']=1.0e9
    reference=run_bc_global_batch(
        g,e.config,e.composition,260.0,torch.float64,
    )
    native=run_native_batch(g,e.config,e.composition,260.0)
    evaluation_steps=2
    for result in (reference,native):
        history=result['histories']['currentCongestion']
        expected_evaluation=history[:evaluation_steps].amax(0)
        expected_full=history.amax(0)
        torch.testing.assert_close(
            result['peakCurrentCongestionToEvaluationTime'],
            expected_evaluation,rtol=0.0,atol=0.0,
        )
        torch.testing.assert_close(
            result['peakCurrentCongestionTo2s'],
            expected_evaluation,rtol=0.0,atol=0.0,
        )
        torch.testing.assert_close(
            result['peakCurrentCongestion'],expected_full,rtol=0.0,atol=0.0,
        )
        assert bool(torch.all(expected_full>expected_evaluation))
    torch.testing.assert_close(
        native['peakCurrentCongestionToEvaluationTime'],
        reference['peakCurrentCongestionToEvaluationTime'],
        rtol=5.0e-4,atol=2.0e-4,
    )

def test_zero_initial_mobile_water_is_not_created_by_numerical_floor(tmp_path):
    from dataclasses import replace
    e=evaluator(tmp_path/'e');g=e._build_geometry_batch([item(tmp_path)])
    composition=replace(e.composition,initial_water_mol_per_m3=0.0)
    result=run_native_batch(g,e.config,composition,1.0,save_fields=True)
    assert not bool(torch.any(result['finalFields']['mobileWater_mol_per_m3']))

def test_infrastructure_error_not_reported_as_physics_failure(tmp_path):
    e=object.__new__(NativeBCGlobalEvaluator);e.actual_batch_limit=8;e.device=torch.device('cpu');e.wave_log=tmp_path/'waves.jsonl'
    def model(*a,**kw):raise RuntimeError('compiler not found')
    e._run_model=model
    with pytest.raises(RuntimeError,match='compiler'):e._safe_run([0],[1.])

def test_cpupool_full_pipeline_standalone(tmp_path):
    from ecsp_nsga2.workflow import NSGA2ElectricalSolidWorkflow
    cfg=config();cfg['evaluator']['backend']='bc_global_native_cpu_pool'
    cfg['evaluator']['native'].update(cpu_runtime='standalone',cpu_workers=2,host_reserve=0,cpu_budget=2)
    workflow=NSGA2ElectricalSolidWorkflow(Path(__file__).resolve().parents[2],cfg,tmp_path/'run')
    try:
        workflow.run()
        complete=json.loads((tmp_path/'run'/'RUN_COMPLETE.json').read_text())
        assert complete['bc_global_preflame_used'] and complete['post_onset_condensed_propagation_used']
        assert (tmp_path/'run'/'final'/'recommended_vs_area_matched_staggered_propagation.json').is_file()
    finally:workflow.evaluator.close()

def test_native_invalid_time_and_table_configs_fail_closed(tmp_path):
    from ecsp_native.adapter import pack_config
    e=evaluator(tmp_path/'e')
    for dt in (0,-1):
        c=copy.deepcopy(e.config);c['bcGlobal']['timeStep_s']=dt
        with pytest.raises(ValueError,match='timeStep'):pack_config(c,e.composition,17)
    c=copy.deepcopy(e.config);c['bcGlobal']['thermal']['heat_capacity']={'mode':'table','temperature_K':[300,300],'values':[1,2]}
    with pytest.raises(ValueError,match='table'):pack_config(c,e.composition,17)
    c=copy.deepcopy(e.config);c['bcGlobal']['evaluationTime_s']=.0095
    with pytest.raises(ValueError,match='align'):pack_config(c,e.composition,17)
    c=copy.deepcopy(e.config);c['bcGlobal']['timeStep_s']=.01
    c['bcGlobal']['endTime_s']=c['bcGlobal']['evaluationTime_s']=.075
    c['bcGlobal']['electricalUpdateInterval_s']=.01
    pack_config(c,e.composition,17)  # the fractional final step is an event boundary
    c=copy.deepcopy(e.config);c['bcGlobal']['electricalUpdateInterval_s']=.009
    with pytest.raises(ValueError,match='integer multiple'):pack_config(c,e.composition,17)
    c=copy.deepcopy(e.config);c['bcGlobal']['electricalUpdateInterval_s']=.001
    with pytest.raises(ValueError,match='at least one step'):pack_config(c,e.composition,17)

def test_native_material_properties_and_physical_bounds_fail_closed(tmp_path):
    from ecsp_native.adapter import pack_config
    e=evaluator(tmp_path/'e')
    invalid_properties=(
        {'mode':'table','temperature_K':[280.,320.],'values':[1.,-1.]},
        {'mode':'reference_arrhenius','reference_value':-1.,'reference_temperature_K':298.15,'activation_energy_J_per_mol':1.},
        {'mode':'reference_arrhenius','reference_value':1.,'reference_temperature_K':0.,'activation_energy_J_per_mol':1.},
        {'mode':'reference_arrhenius','reference_value':1.,'reference_temperature_K':298.15,'activation_energy_J_per_mol':-1.},
        {'mode':'constant','value':-1.},
    )
    for raw in invalid_properties:
        c=copy.deepcopy(e.config);c['bcGlobal']['transport']['cation_diffusivity']=raw
        with pytest.raises(ValueError):pack_config(c,e.composition,17)
    c=copy.deepcopy(e.config);c['electrical']['conductivityMinimum_S_per_m']=0.
    with pytest.raises(ValueError,match='conductivity'):pack_config(c,e.composition,17)
    c=copy.deepcopy(e.config);c['bcGlobal']['thermal']['emissivity']=1.1
    with pytest.raises(ValueError,match='heat-loss'):pack_config(c,e.composition,17)
    c=copy.deepcopy(e.config);c['bcGlobal']['thermal']['heat_capacity']={'mode':'constant','value':0.}
    with pytest.raises(ValueError,match='Heat capacity'):pack_config(c,e.composition,17)
    for value in (0.0,1.0001,float('nan')):
        c=copy.deepcopy(e.config);c['numericalQuality']['maximumExplicitThermalCFL']=value
        with pytest.raises(ValueError,match='maximumExplicitThermalCFL'):pack_config(c,e.composition,17)

def test_strict_backends_share_density_weight_and_outer_iteration_contract(tmp_path):
    from ecsp_native.adapter import pack_config
    from ecsp_v6.physics.bc_global import _validate_bc_runtime_contract
    e=evaluator(tmp_path/'e')

    # A configured thermal density overrides the composition density, so the
    # resolved value itself must be finite and strictly positive in both paths.
    for value in (0.0,-1.0,float('nan'),float('inf')):
        c=copy.deepcopy(e.config)
        c['bcGlobal']['thermal']['density_kg_per_m3']=value
        with pytest.raises(ValueError,match='thermal density'):
            _validate_bc_runtime_contract(c,e.composition)
        with pytest.raises(ValueError,match='thermal density'):
            pack_config(c,e.composition,17)

    # This error is small enough to pass the former 1e-9 check, but exceeds
    # the canonical 1e-12 absolute normalization tolerance.
    c=copy.deepcopy(e.config)
    c['bcGlobal']['kinetics']['mass_conversion_weights']=[0.5,0.50000000001]
    with pytest.raises(ValueError,match='summing to 1'):
        _validate_bc_runtime_contract(c,e.composition)
    with pytest.raises(ValueError,match='summing to 1'):
        pack_config(c,e.composition,17)

    c=copy.deepcopy(e.config)
    c['interface']['nonlinearRobin']['maximumIterations']=1
    c['interface']['nonlinearRobin']['minimumIterationsCoupled']=1
    with pytest.raises(ValueError,match='maximumIterations'):
        _validate_bc_runtime_contract(c,e.composition)
    with pytest.raises(ValueError,match='maximumIterations'):
        pack_config(c,e.composition,17)

def test_surface_local_iteration_defaults_ignore_legacy_budget(tmp_path):
    from ecsp_native.adapter import pack_config
    from ecsp_v6.physics.bc_global import _validate_bc_runtime_contract
    e=evaluator(tmp_path/'e')
    for legacy_budget in (1,37,10_000):
        c=copy.deepcopy(e.config)
        robin=c['interface']['nonlinearRobin']
        robin.pop('localInterfaceMinimumIterations',None)
        robin.pop('localInterfaceMaximumIterations',None)
        robin['localInterfaceIterations']=legacy_budget
        _validate_bc_runtime_contract(c,e.composition)
        packed,_=pack_config(c,e.composition,17)
        assert packed['local_min']==8.0
        assert packed['local_max']==60.0

def test_native_thermal_cfl_matches_reference_and_fails_closed(tmp_path):
    from ecsp_v6.physics.bc_global import run_bc_global_batch
    e=evaluator(tmp_path/'e');g=e._build_geometry_batch([item(tmp_path)])
    reference=run_bc_global_batch(g,e.config,e.composition,20.0,torch.float64)
    native=run_native_batch(g,e.config,e.composition,20.0,torch.float64)
    torch.testing.assert_close(
        native['histories']['thermalDiffusiveCFL'],
        reference['histories']['thermalDiffusiveCFL'],rtol=2e-12,atol=1e-15,
    )
    torch.testing.assert_close(
        native['maximumThermalDiffusiveCFL'],
        native['histories']['thermalDiffusiveCFL'].amax(0),rtol=0,atol=0,
    )
    torch.testing.assert_close(
        native['stabilityDiagnostics']['maximum_thermal_diffusive_CFL'],
        reference['stabilityDiagnostics']['maximum_thermal_diffusive_CFL'],
        rtol=2e-12,atol=1e-15,
    )
    unstable=copy.deepcopy(e.config)
    unstable['numericalQuality']['maximumExplicitThermalCFL']=1.0e-12
    with pytest.raises(
        RuntimeError,
        match='total explicit thermal stability CFL limit exceeded',
    ):
        run_native_batch(
            g,unstable,e.composition,20.0,torch.float64,
            native_options={'host_check_interval_steps':999},
        )

def test_missing_joule_model_defaults_to_reference_conductive_mode(tmp_path):
    from ecsp_native.adapter import pack_config
    e=evaluator(tmp_path/'e');c=copy.deepcopy(e.config)
    c['electrical'].pop('jouleHeatModel',None)
    packed,_=pack_config(c,e.composition,17)
    assert packed['joule_mode']==1.0


def test_benchmark_geometry_workload_and_profiles(tmp_path):
    import yaml
    from benchmark_bc_native_batches import benchmark_items
    from ecsp_nsga2.workflow import NSGA2ElectricalSolidWorkflow
    root=Path(__file__).resolve().parents[2]
    w=NSGA2ElectricalSolidWorkflow(root,config(),tmp_path/'workflow')
    items=benchmark_items(w,tmp_path/'candidates')
    assert len(items)==4
    assert all(a.dtype==bool and c.dtype==bool and a.shape==c.shape for a,c,_,_ in items)
    baseline=yaml.safe_load((root/'config/nsga2_bc_global_preflame_propagation_a100.yaml').read_text())
    for b in (32,64):
        cfg=yaml.safe_load((root/f'config/nsga2_bc_global_native_a100_batch{b}.yaml').read_text())
        for key in ('bc_global','physics','geometry','minimum_ignition_voltage_search'):
            assert cfg[key]==baseline[key],key
        assert cfg['evaluator']['internal_batch_size']==b
        assert cfg['evaluator']['native']['cpu_runtime']=='standalone'
        assert cfg['optimization']['objectives_minimise']==baseline['optimization']['objectives_minimise']
