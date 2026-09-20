from __future__ import annotations
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import yaml
from ecsp_reactive.condensed import BCReactiveHandoffAdapter, BCReactiveSolver
from ecsp_reactive.condensed.thermo import BCHeatCapacity
from ecsp_reactive.condensed.chemistry import *
from ecsp_reactive.condensed.finite_volume import flux_divergence
from ecsp_nsga2.propagation import (PropagationConfigurationError,PropagationCandidateInputError,
                                    PropagationCandidateNumericalError,run_condensed_propagation)
from ecsp_nsga2.post_onset import (run_post_onset,validate_post_onset_config,handoff_digest,
                                   write_backend_rank_comparison)
from ecsp_v6.physics.bc_global import _channel_rate,_as_channel

ROOT=Path(__file__).resolve().parents[2]


def fixture(shape=(8,8),rates=(.2,.1),heats=(1000.,2000.),alpha=(.2,.2)):
    f=lambda v:np.full(shape,v,dtype=float)
    a=np.zeros(shape,bool);c=a.copy();a[:,0]=True;c[:,-1]=True
    w=np.asarray([2/3,1/3]);X=w@alpha;xi=100*X
    h={'handoffSchemaVersion':'ecsp_bc_surface_onset_v8.2.0','onsetSucceeded':True,'onsetReportedByPhysics':True,
       'numericallyValidForPropagationHandoff':True,'propagationHandoffAuthorizationReason':'ignition_and_numerics_valid',
       'ignitionDelay_s':.1,'continuedElectricalHeating':False,'temperatureAtOnset_K':f(300),'propellantMask':f(1),
       'anodeContactMask':a,'cathodeContactMask':c,'alphaChannel1AtOnset':f(alpha[0]),'alphaChannel2AtOnset':f(alpha[1]),
       'globalProgressAtOnset':f(X),'xiMax_mol_per_m3':100.,'initialMobileLP_mol_per_m3':145.,
       'initialPVARepeat_mol_per_m3':100.,'initialMobileWater_mol_per_m3':50.,'molarMassLP_kg_per_mol':.1,
       'molarMassPVARepeat_kg_per_mol':.05,'initialReactiveMass_kg_per_m3':19.5,
       'cationAtOnset_mol_per_m3':f(145-1.45*xi),'anionAtOnset_mol_per_m3':f(145-1.45*xi),
       'mobileLPAtOnset_mol_per_m3':f(145-1.45*xi),'pvaReactiveRepeatAtOnset_mol_per_m3':f(100-xi),
       'generatedWaterProductAtOnset_mol_per_m3':f(2*xi),'mobileWaterAtOnset_mol_per_m3':f(50),
       'electrochemicalLPConsumedAtOnset_mol_per_m3':f(0),'potentialAtOnset_V':f(0),
       'qJAtOnset_W_per_m3':f(0),'qEchemAtOnset_W_per_m3':f(0)}
    p={'duration_s':.01,'time_step_s':.001,'snapshot_interval_s':.005,'domain_size_m':.02,
       'density_kg_per_m3':1000.,'surface_layer_thickness_m':.001,'gas_constant_J_per_molK':8.31446261815324,
       'front_progress_threshold':.5,'established_reacted_area_fraction':.5,'continued_electrical_heating':False,
       'electrical_heating_mode':'off'}
    ch=lambda k,q:dict(alpha_grid=[0.,1.],activation_energy_J_per_mol=[0.,0.],ln_Af_per_s=[math.log(k)]*2,heat_release_J_per_kg=q)
    b={'endTime_s':1.,'onsetCriterion':{'temperature_K':300.,'minimum_progress':.001,'minimum_area_fraction':.5},
       'thermal':{'heat_capacity':2200.,'thermal_conductivity':.4,'initialTemperature_K':298.15,
                  'ambientTemperature_K':298.15,'minimumTemperature_K':1,'maximumTemperature_K':10000},
       'kinetics':{'mass_conversion_weights':w.tolist(),'channels':[ch(rates[0],heats[0]),ch(rates[1],heats[1])]}}
    r={'eos':{'A_Pa':101325.,'B_Pa':2e9,'N':7.,'provenance':'synthetic numerical test not a material calibration'}}
    return h,p,b,r


def solver_fixture(**kwargs):
    h,p,b,r=fixture(**kwargs)
    a=BCReactiveHandoffAdapter(p,b,r).adapt(h)
    return h,p,b,r,a,BCReactiveSolver(a,p,b,r)


@pytest.mark.parametrize('cp',[2200.,{'mode':'constant','value':1900.},
                               {'mode':'table','temperature_K':[250.,350.,500.,1000.],'values':[1000.,2000.,1700.,3000.]}])
def test_caloric_integral_round_trip(cp):
    cal=BCHeatCapacity(cp,298.15)
    t=np.array([1.,249.,250.,251.,298.15,349.,350.,400.,500.,900.,1000.,1500.])
    np.testing.assert_allclose(cal.temperature(cal.sensible_energy(t)),t,atol=2e-12,rtol=1e-14)
    eps=1e-3
    derivative=(cal.sensible_energy(t+eps)-cal.sensible_energy(t-eps))/(2*eps)
    np.testing.assert_allclose(derivative,cal.cp(t),atol=.05,rtol=1e-6)


def test_invalid_caloric_replacement_refused():
    with pytest.raises(PropagationConfigurationError): BCHeatCapacity({'mode':'reference_arrhenius'},300)
    with pytest.raises(PropagationConfigurationError): BCHeatCapacity(-1.,300)


def test_handoff_exact_two_channel_energy_and_inventories():
    h,p,b,r,a,s=solver_fixture()
    P=a.thermo.primitive(a.U)
    np.testing.assert_allclose(P[...,3],h['temperatureAtOnset_K'],atol=1e-12)
    np.testing.assert_array_equal(a.U[...,RHO],1000.)
    np.testing.assert_array_equal(a.U[...,MX:MY+1],0.)
    np.testing.assert_allclose(P[...,A1],h['alphaChannel1AtOnset'],atol=1e-16)
    np.testing.assert_allclose(P[...,A2],h['alphaChannel2AtOnset'],atol=1e-16)
    np.testing.assert_allclose(a.U[...,WATER],h['mobileWaterAtOnset_mol_per_m3'])
    np.testing.assert_allclose(a.U[...,PRODUCT_WATER],h['generatedWaterProductAtOnset_mol_per_m3'])
    assert not np.shares_memory(a.U,h['temperatureAtOnset_K'])
    assert a.audit['temperature_roundtrip_max_error_K']<1e-10


@pytest.mark.parametrize('change',[
    lambda h:h.pop('anodeContactMask'),
    lambda h:h.update(onsetSucceeded=False),
    lambda h:h.update(onsetReportedByPhysics=False),
    lambda h:h.update(numericallyValidForPropagationHandoff=False),
    lambda h:h.update(bcGlobalConfigSHA256='wrong'),
    lambda h:h.update(ignitionDelay_s=2.),
    lambda h:h.update(continuedElectricalHeating='false'),
    lambda h:h['globalProgressAtOnset'].__setitem__((0,0),.3),
    lambda h:h['generatedWaterProductAtOnset_mol_per_m3'].__setitem__((0,0),1000.),
    lambda h:h['cationAtOnset_mol_per_m3'].__setitem__((0,0),-1.),
    lambda h:h['qEchemAtOnset_W_per_m3'].__setitem__((3,3),1.),
    lambda h:h['propellantMask'].__setitem__((3,3),0.),
    lambda h:h['temperatureAtOnset_K'].__setitem__((3,3),np.nan),
])
def test_invalid_handoff_rejected(change):
    h,p,b,r=fixture();change(h)
    with pytest.raises((PropagationCandidateInputError,PropagationCandidateNumericalError)):
        BCReactiveHandoffAdapter(p,b,r).adapt(h)


def test_same_X_does_not_erase_channel_information():
    h,p,b,r=fixture(alpha=(.5,0.))
    a=BCReactiveHandoffAdapter(p,b,r).adapt(h)
    h2,p2,b2,r2=fixture(alpha=(0.,1.))
    a2=BCReactiveHandoffAdapter(p2,b2,r2).adapt(h2)
    np.testing.assert_allclose(a.chemistry.progress(a.U),a2.chemistry.progress(a2.U))
    q=a.chemistry.source(a.U,h['temperatureAtOnset_K'],.001)[0][...,ENERGY]
    q2=a2.chemistry.source(a2.U,h2['temperatureAtOnset_K'],.001)[0][...,ENERGY]
    assert np.max(abs(q-q2))>0
    assert np.all(a2.chemistry.raw_rates(a2.U,h2['temperatureAtOnset_K'])[...,1]==0)


def test_kinetics_identical_to_BC_torch_tables():
    h,p,b,r,a,s=solver_fixture()
    for i,ch in enumerate(b['kinetics']['channels']):
        raw,_,_=_channel_rate(torch.from_numpy(a.U[...,A1+i]/a.U[...,RHO]),
                         torch.from_numpy(h['temperatureAtOnset_K']),_as_channel(ch,str(i)),p['gas_constant_J_per_molK'])
        np.testing.assert_allclose(a.chemistry.raw_rates(a.U,h['temperatureAtOnset_K'])[...,i],raw.numpy(),rtol=1e-14)


def test_accepted_heat_equals_two_conservative_channel_increments():
    h,p,b,r,a,s=solver_fixture(rates=(.8,.4),alpha=(.999,.999))
    src,d=a.chemistry.source(a.U,h['temperatureAtOnset_K'],.1)
    np.testing.assert_allclose(src[...,ENERGY],a.chemistry.Q[0]*src[...,A1]+a.chemistry.Q[1]*src[...,A2],rtol=1e-15)
    np.testing.assert_array_equal(src[...,RHO],0.)
    result=a.U+.1*src
    assert np.max(result[...,A1:A2+1]/result[...,RHO,None])<=1+1e-15
    for v in a.chemistry.conserved_inventory_residuals(result).values(): assert np.max(abs(v))<1e-10


def test_exhausted_LP_blocks_progress_and_heat():
    h,p,b,r,a,s=solver_fixture()
    U=a.U.copy(); U[...,EC_LP]+=U[...,CATION];U[...,CATION:ANION+1]=0.
    src,d=a.chemistry.source(U,h['temperatureAtOnset_K'],.1)
    np.testing.assert_array_equal(src[...,A1:A2+1],0)
    np.testing.assert_array_equal(src[...,ENERGY],0)
    assert d['inventory_limiter_fraction']==1.


def test_no_mobile_water_created_from_product_water():
    h,p,b,r,a,s=solver_fixture()
    S,d=a.chemistry.source(a.U,h['temperatureAtOnset_K'],.001)
    assert np.all(S[...,PRODUCT_WATER]>0)
    np.testing.assert_array_equal(S[...,WATER],0)


def test_no_extra_Q_weights():
    h,p,b,r,a,s=solver_fixture()
    source,_=a.chemistry.source(a.U,h['temperatureAtOnset_K'],.001)
    expected=p['density_kg_per_m3']*(1000*.2+2000*.1)
    np.testing.assert_allclose(source[...,ENERGY],expected)


def test_tait_cold_work_integral_and_barotropic_pressure():
    h,p,b,r,a,s=solver_fixture()
    rho=np.array([1000.,1000.1,1001.]); dr=1e-3
    derivative=(a.thermo.cold_energy(rho+dr)-a.thermo.cold_energy(rho-dr))/(2*dr)
    np.testing.assert_allclose(derivative,a.thermo.pressure(rho)/rho**2,rtol=1e-6,atol=1e-5)
    for t in [300.,500.,800.]:
        U=a.U.copy();P=a.thermo.primitive(U);P[...,3]=t;U=a.thermo.conservative(P)
        np.testing.assert_allclose(a.thermo.pressure(U[...,RHO]),101325.)


def test_hllc_stationary_thermal_contact_has_no_acoustic_diffusion():
    h,p,b,r,a,s=solver_fixture()
    P=a.thermo.primitive(a.U);P[:,4:,3]=600.;U=a.thermo.conservative(P)
    for boundary in ['reflective','periodic','transmissive']:
        rhs,ledger,d=flux_divergence(U,a.thermo,a.dx,a.thickness,boundary=boundary,stationary_fast_path=False)
        np.testing.assert_allclose(rhs,0.,atol=1e-7)
        np.testing.assert_allclose(ledger,0.,atol=1e-12)
    # HLL is intentionally available as a diagnostic. It must not be confused
    # with physical Fourier conduction at a stationary thermal contact.
    hll,_,_=flux_divergence(U,a.thermo,a.dx,a.thickness,kind='hll')
    assert np.max(abs(hll[...,ENERGY]))>1.


def test_shared_scalar_transport_preserves_stoichiometric_invariants():
    h,p,b,r,a,s=solver_fixture(shape=(12,12))
    P=a.thermo.primitive(a.U);x=np.arange(12)[None,:]
    P[...,A1]=.2+.08*np.sin(2*np.pi*x/12);P[...,A2]=.25+.1*np.cos(2*np.pi*x/12)
    X=P[...,A1:A2+1]@a.chemistry.weights
    P[...,CATION]=P[...,ANION]=a.chemistry.initial_lp_per_kg-1.45*a.chemistry.xi_per_kg*X
    P[...,PVA]=a.chemistry.initial_pva_per_kg-a.chemistry.xi_per_kg*X
    P[...,PRODUCT_WATER]=2*a.chemistry.xi_per_kg*X
    P[...,MX]=.02;U=a.thermo.conservative(P)
    rhs,ledger,d=flux_divergence(U,a.thermo,a.dx,a.thickness,boundary='periodic')
    rate_X=a.chemistry.weights[0]*rhs[...,A1]+a.chemistry.weights[1]*rhs[...,A2]
    np.testing.assert_allclose(rhs[...,PVA]+a.chemistry.xi_per_kg*rate_X-a.chemistry.initial_pva_per_kg*rhs[...,RHO],0,atol=2e-9)
    np.testing.assert_allclose(rhs[...,PRODUCT_WATER]-2*a.chemistry.xi_per_kg*rate_X,0,atol=2e-9)
    np.testing.assert_allclose(rhs.sum(axis=(0,1))*a.dx*a.dx*a.thickness,ledger,atol=1e-9)


def test_uniform_reaction_exact_energy_mass_and_same_baseline(tmp_path):
    h,p,b,r,a,s=solver_fixture()
    m=s.run(tmp_path/'reactive')
    base=run_condensed_propagation(h,p,b,tmp_path/'baseline')
    f=np.load(tmp_path/'reactive/propagation_fields.npz')
    np.testing.assert_allclose(f['final_alpha_channel1'],.202,atol=1e-14)
    np.testing.assert_allclose(f['final_alpha_channel2'],.201,atol=1e-14)
    np.testing.assert_allclose(f['final_temperature_K'],300.+4/2200.,atol=1e-12)
    assert m['finalMassBudgetResidual_kg']==0.
    assert abs(m['finalEnergyBudgetResidual_J'])<1e-12
    assert m['exactStationaryMechanicsSkippedSteps']==m['acceptedTimeSteps']
    assert base['finalMaximumTemperature_K']==pytest.approx(m['finalMaximumTemperature_K'],abs=1e-10)


def test_hotspot_diffusion_conserves_energy_without_gas_or_duplicate_solid(tmp_path):
    h,p,b,r=fixture(rates=(1e-30,1e-30),heats=(0,0))
    h['temperatureAtOnset_K'][2:5,2:5]=600.
    b['thermal']['heat_capacity']={'mode':'table','temperature_K':[250,350,700],'values':[1500,2000,3000]}
    a=BCReactiveHandoffAdapter(p,b,r).adapt(h);s=BCReactiveSolver(a,p,b,r)
    m=s.run(tmp_path)
    assert m['finalMaximumTemperature_K']<600
    assert m['maximumEnergyBudgetRelativeResidual']<1e-11
    assert m['maximumSpeedDuringPropagation_m_per_s']==0
    assert m['separateSolidEnergyEquationUsed'] is False
    assert m['gasMassSourceUsed'] is False


def test_saved_preflame_heat_is_not_replayed(tmp_path):
    h,p,b,r=fixture(rates=(1e-30,1e-30),heats=(0,0))
    h['qJAtOnset_W_per_m3'][...]=1e9
    h['qEchemAtOnset_W_per_m3'][h['anodeContactMask']]=2e9
    a=BCReactiveHandoffAdapter(p,b,r).adapt(h);s=BCReactiveSolver(a,p,b,r)
    m=s.run(tmp_path)
    assert m['integratedJouleHeat_J']==m['integratedElectrochemicalHeat_J']==0
    assert m['finalMaximumTemperature_K']==pytest.approx(300.)


def test_reaction_front_comes_from_both_channels_without_density_removal(tmp_path):
    h,p,b,r=fixture(rates=(.2,.1),alpha=(.499,.499))
    a=BCReactiveHandoffAdapter(p,b,r).adapt(h);s=BCReactiveSolver(a,p,b,r);m=s.run(tmp_path)
    f=np.load(tmp_path/'propagation_fields.npz')
    np.testing.assert_allclose(f['final_reaction_level_set'],.5-(2*f['final_alpha_channel1']+f['final_alpha_channel2'])/3,atol=1e-15)
    assert np.all(f['reacted_material_mask'])
    np.testing.assert_array_equal(f['U_final'][...,RHO],1000.)
    assert m['finalUnreactedAreaFraction']==0
    # Not an extinguished chemistry mask at X=.5: both channels keep reacting.
    assert np.all(f['final_alpha_channel1']>.5)


def test_paired_dispatch_and_input_immutability(tmp_path):
    h,p,b,r=fixture();before=handoff_digest(h)
    cfg={'backend':'reactive_euler','compare_backends':True,'reactive_euler':r}
    m=run_post_onset(h,p,b,tmp_path,post_onset_config=cfg)
    assert handoff_digest(h)==before
    pair=json.loads((tmp_path/'backend_comparison.json').read_text())
    assert pair['bothBackendsCompleted'] is True
    assert (tmp_path/'condensed_propagation/propagation_fields.npz').is_file()
    assert (tmp_path/'reactive_euler/propagation_fields.npz').is_file()
    assert m['postOnsetBackend']=='reactive_euler'


@pytest.mark.parametrize('cfg',[{'backend':'unknown'},{'backend':'reactive_euler'},
    {'backend':'condensed_propagation','compare_backends':'true'},
    {'backend':'reactive_euler','reactive_euler':{'eos':{}}}])
def test_invalid_or_unacknowledged_ranking_config(cfg):
    with pytest.raises(PropagationConfigurationError):validate_post_onset_config(cfg,for_optimization=True)


def test_primary_backend_failure_never_silently_falls_back(tmp_path):
    h,p,b,r=fixture();r['maximum_time_steps']=1
    with pytest.raises(PropagationConfigurationError):
        run_post_onset(h,p,b,tmp_path,post_onset_config={'backend':'reactive_euler','compare_backends':True,'reactive_euler':r})
    assert not (tmp_path/'reactive_euler/propagation_metrics.json').exists()


def test_legacy_stale_heating_mode_rejected():
    h,p,b,r=fixture();p.update(continued_electrical_heating=True,electrical_heating_mode='legacy_preflame_history_replay')
    a=BCReactiveHandoffAdapter(p,b,r).adapt(h)
    with pytest.raises(PropagationConfigurationError):BCReactiveSolver(a,p,b,r)


def test_dynamic_acoustic_update_is_not_replaced_by_stationary_fast_path(tmp_path):
    h,p,b,r=fixture(shape=(10,10),rates=(1e-30,1e-30),heats=(0,0))
    p.update(duration_s=1e-6,time_step_s=1e-6,snapshot_interval_s=1e-6)
    b['thermal']['thermal_conductivity']=0.
    a=BCReactiveHandoffAdapter(p,b,r).adapt(h);s=BCReactiveSolver(a,p,b,r)
    P=a.thermo.primitive(s.U)
    P[...,RHO]*=1+1e-7*np.sin(2*np.pi*np.arange(10)[None,:]/10)
    s.U=a.thermo.conservative(P)
    m=s.run(tmp_path)
    assert m['maximumSpeedDuringPropagation_m_per_s']>1e-6
    assert m['exactStationaryMechanicsSkippedSteps']==0
    assert m['maximumMassBudgetRelativeResidual']<1e-10
    assert m['maximumEnergyBudgetRelativeResidual']<1e-9


def test_rank_comparison_can_detect_reversal(tmp_path):
    items=[]
    for i in range(3):
        bs={}
        for name,index in [('condensed_propagation',i),('reactive_euler',2-i)]:
            bs[name]={'status':'complete','finalUnreactedAreaFraction':index/10,
                      'establishedTimeAfterOnset_s':.1+index*.1,'meanEffectiveRegressionVelocity_m_per_s':0.,
                      'reactionFrontNonuniformity':index/10}
        items.append(SimpleNamespace(geometry_id=str(i),metrics={'preflame_objective_vector':[0.1,0.2,10.,1.],
                      'postOnsetBackendComparison':{'bothBackendsCompleted':True,'backends':bs}}))
    write_backend_rank_comparison(items,tmp_path)
    result=json.loads((tmp_path/'backend_rank_comparison.json').read_text())
    assert result['spearman_utopia_rank']==pytest.approx(-1)
    assert result['pair_count']==3


def test_full_nsga2_reactive_branch_and_same_handoff_baseline(tmp_path):
    from ecsp_nsga2.workflow import NSGA2ElectricalSolidWorkflow
    cfg=yaml.safe_load((ROOT/'config/nsga2_bc_reactive_debug.yaml').read_text())
    wf=NSGA2ElectricalSolidWorkflow(package_root=ROOT,config=cfg,workdir=tmp_path/'run',allow_debug_physics=False)
    recommended=wf.run()
    assert recommended.objectives.shape==(8,)
    assert recommended.metrics['postOnsetBackend']=='reactive_euler'
    final=tmp_path/'run/final'
    assert (final/'backend_rank_comparison.csv').is_file()
    assert (final/'area_matched_staggered_propagation/backend_comparison.json').is_file()
    done=json.loads((tmp_path/'run/RUN_COMPLETE.json').read_text())
    assert done['post_onset_reactive_euler_used'] is True
    assert done['post_onset_condensed_propagation_used'] is True
    assert done['gas_phase_cfd_used'] is False
    from ecsp_v6.physics.composition_model import build_composition
    hd=next((final/'propagation_candidates').glob('*/bc_handoff'))
    h=json.loads((hd/'bc_handoff_metadata.json').read_text())
    with np.load(hd/'bc_handoff_fields.npz',allow_pickle=False) as f:h.update({k:f[k] for k in f.files})
    full=json.loads((tmp_path/'run/adapter/resolved_physics_config.json').read_text())
    full['coupled']['voltage_V']=5.0  # Numerical supply-step test after onset, not calibration.
    prop=copy.deepcopy(cfg['propagation_refinement'])
    prop.update(domain_size_m=full['geometry']['domainSize_m'],
                surface_layer_thickness_m=full['geometry']['surfaceLayerThickness_m'],
                density_kg_per_m3=build_composition(full).density_kg_per_m3,
                gas_constant_J_per_molK=full['transport']['gasConstant_J_per_molK'],
                continued_electrical_heating=True,electrical_heating_mode='recomputed',
                duration_s=1e-5,time_step_s=5e-6,snapshot_interval_s=5e-6)
    post=copy.deepcopy(cfg['post_onset']);post['compare_backends']=False
    powered=run_post_onset(h,prop,full['bcGlobal'],tmp_path/'powered',post_onset_config=post,full_bc_config=full)
    assert powered['postOnsetElectricalSolveCalls']==6
    assert powered['integratedJouleHeat_J']>0 and powered['integratedElectrochemicalHeat_J']>0
    assert powered['maximumEnergyBudgetRelativeResidual']<1e-9
    with np.load(tmp_path/'powered/reactive_euler/propagation_fields.npz') as f:
        contacts=f['anode_contact_mask']|f['cathode_contact_mask']
        qe=f['last_electrical_stage_qEchem_W_per_m3']
        assert np.max(qe)>0 and np.count_nonzero(qe[~contacts])==0
        np.testing.assert_allclose(f['last_electrical_stage_qJ_W_per_m3'],
            f['last_electrical_stage_inPlaneJouleHeat_W_per_m3']+f['last_electrical_stage_contactNormalJouleHeat_W_per_m3'])


def test_exact_stationary_fast_path_matches_full_flux_update(tmp_path):
    h,p,b,r=fixture(shape=(8,8))
    p.update(duration_s=2e-7,time_step_s=1e-8,snapshot_interval_s=2e-7)
    h['temperatureAtOnset_K']+=np.cos(np.pi*(np.arange(8)[None,:]+.5)/8)
    final=[]
    for fast in (False,True):
        opts={**r,'stationary_mechanics_fast_path':fast}
        a=BCReactiveHandoffAdapter(p,b,opts).adapt(h);s=BCReactiveSolver(a,p,b,opts)
        m=s.run(tmp_path/str(fast));final.append(s.U)
        assert m['maximumSpeedDuringPropagation_m_per_s']<1e-12
        assert m['maximumEnergyBudgetRelativeResidual']<1e-9
    np.testing.assert_allclose(final[0],final[1],rtol=2e-12,atol=2e-10)


def test_fourier_grid_convergence_retained_in_single_energy_equation(tmp_path):
    errors=[]
    for n in (8,16,32):
        h,p,b,r=fixture(shape=(n,n),rates=(1e-30,1e-30),heats=(0,0))
        b['thermal']['thermal_conductivity']=2000.
        p.update(duration_s=.001,time_step_s=2e-5,snapshot_interval_s=.001)
        mode=np.cos(np.pi*(np.arange(n)[None,:]+.5)/n)
        h['temperatureAtOnset_K']+=10*mode
        a=BCReactiveHandoffAdapter(p,b,r).adapt(h);s=BCReactiveSolver(a,p,b,r);s.run(tmp_path/str(n))
        alpha=2000/(1000*2200)
        exact=300+10*mode*np.exp(-alpha*(np.pi/.02)**2*p['duration_s'])
        computed=s.thermo.primitive(s.U)[...,3]
        errors.append(float(np.sqrt(np.mean((computed-exact)**2))))
    assert errors[0]/errors[1]>3.5 and errors[1]/errors[2]>3.5
    (tmp_path/'grid_convergence.json').write_text(json.dumps({'grids':[8,16,32],'L2_temperature_errors_K':errors}))
