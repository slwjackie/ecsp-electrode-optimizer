from __future__ import annotations
from dataclasses import asdict
from typing import Any, Mapping, Sequence
import math
import torch
from .loader import load_native, required_cpp_standard
from ecsp_v6.physics.bc_global import (
    _validate_bc_config, _validate_bc_runtime_contract, _reaction_inventory,
    _masked_integral_ratio, BCCandidateBatchError, BC_GLOBAL_REACTION,
    BC_PAPER_EQUATIONS_USED, BC_PAPER_EQUATIONS_NOT_USED_IN_PREFLAME,
    BC_EQUATION_32_ROLE, BC_SCALAR_HISTORY_FIELDS, _bc_resource_limits,
    _positive_time_step_count, _time_match_tolerance,
    complete_bc_evaluation_output,
)

HISTORY_NAMES = BC_SCALAR_HISTORY_FIELDS

def _native_step_count(
    duration_s: float,
    dt_s: float,
    maximum_steps: int = (1 << 31) - 1,
) -> int:
    """Match the native engine's tolerance for nearly integral FP ratios."""
    return _positive_time_step_count(
        float(duration_s), float(dt_s), maximum_steps=maximum_steps
    )

def _is_time_grid_aligned(duration_s: float, dt_s: float) -> bool:
    duration=float(duration_s);step=float(dt_s)
    if not math.isfinite(duration) or not math.isfinite(step) or step<=0.0:
        return False
    ratio=duration/step
    if not math.isfinite(ratio):
        return False
    nearest=round(ratio)
    tolerance=64.0*math.ulp(1.0)*max(1.0,abs(ratio))
    return abs(ratio-nearest)<=tolerance

class NativeDiagnostics:
    def __init__(self, raw: Mapping[str, torch.Tensor], device: str):
        self.converged = raw['linear_converged']
        self.relative_residual = raw['linear_residual']
        self.iterations = raw['linear_iterations']
        standard_tag=required_cpp_standard().replace('+','p')
        self.method = f'{standard_tag}_{device}_fp64_equilibrated_pcg_surface_overlay_bv'
    def to_cpu_dicts(self) -> list[dict]:
        cv=self.converged.cpu().tolist();rr=self.relative_residual.cpu().tolist();it=self.iterations.cpu().tolist()
        return [dict(converged=bool(c),relative_residual=float(r),iterations=int(n),method=self.method)
                for c,r,n in zip(cv,rr,it)]

def pack_config(config: dict, composition: Any, grid_size: int) -> tuple[dict[str,float],list[float]]:
    """One-time host configuration marshalling; no time-stepping in Python."""
    if isinstance(grid_size,bool) or int(grid_size)!=grid_size or int(grid_size)<5:
        raise ValueError('Native B/C requires an integer grid_size >= 5')
    grid_size=int(grid_size)
    channels=_validate_bc_config(config)
    _validate_bc_runtime_contract(config, composition)
    bc=config['bcGlobal'];tr=config['transport'];el=config['electrical'];it=config['interface']
    nu=config['numerics'];ro=it.get('nonlinearRobin',{});ps=nu['potentialSolver'];th=bc['thermal']
    # Fail closed on old names: in prior releases ``nonlinear_robin`` could
    # still mean perimeter faces/embedded electrode cells.  The explicit name
    # below is the production surface-overlay contract.
    if str(it.get('boundaryCouplingModel','')).lower()!='surface_overlay_bv':
        raise ValueError('Native B/C requires interface.boundaryCouplingModel=surface_overlay_bv')
    if str(it.get('contactModel','surface_overlay_full_propellant')).lower() not in {
        'surface_overlay_full_propellant','surface_overlay','surface_contact_overlay'}:
        raise ValueError('Native B/C requires surface contact labels over the full propellant domain')
    if bool(it.get('electrodeMasksRemovePropellant',False)):
        raise ValueError('electrodeMasksRemovePropellant must be false for the surface-overlay model')
    if not bool(it.get('hiddenBusConnectionAssumed',True)):
        raise ValueError('Native shared-terminal masks require hiddenBusConnectionAssumed=true')
    if str(it.get('currentDirectionModel','signed_normal')).lower()!='signed_normal':
        raise ValueError('Native surface-overlay B/C requires currentDirectionModel=signed_normal')
    if bool(bc.get('continuedElectricalHeatingAfterOnset',False)):
        raise ValueError('Strict pre-flame B/C freezes at first onset; continuedElectricalHeatingAfterOnset must be false')
    if config['coupled'].get('usePaperMassTransferSaturation',False) or it.get('nernst',{}).get('enabled',False):
        raise ValueError('Native strict B/C does not support legacy saturation/activity corrections; use the preserved reference backend.')
    if it.get('includeActivationHeat',False) or float(it.get('liquidKineticsGain',0)) != 0:
        raise ValueError('Native strict B/C requires the v8 BC proxy/activation-heat exclusions.')
    if any(it.get('blocking',{}).get(k,{}).get('enabled',False) for k in ('passivation','gasCoverage')):
        raise ValueError('Legacy blocking dynamics are not part of strict B/C.')
    mode=str(ps.get('methodCoupled',ps.get('method','auto'))).lower()
    pre=str(ps.get('preconditioner','jacobi')).lower()
    if mode not in ('auto','pcg') or pre not in ('jacobi','diagonal','identity','none'):
        raise ValueError('Native backend explicitly uses equilibrated PCG; set methodCoupled=pcg, preconditioner=jacobi. '
                         'Legacy direct/BiCGStab/multigrid remains available via bc_global_preflame.')
    if float(tr['chargeNumberCation'])!=1 or float(tr['chargeNumberAnion'])!=-1:
        raise ValueError('v8 B/C electroneutral averaging assumes a 1:1 binary electrolyte.')
    floors=nu.get('physicalFloors',{})
    def floor(k,d): return float(floors.get(k,d))
    density=th.get('density_kg_per_m3')
    numerical_quality=config.get('numericalQuality',{})
    if not isinstance(numerical_quality,Mapping):numerical_quality={}
    maximum_time_steps,maximum_history_bytes=_bc_resource_limits(config)
    cfg={
     'R':float(tr['gasConstant_J_per_molK']), 'F':float(tr['faradayConstant_C_per_mol']),
     'zp':float(tr['chargeNumberCation']),'zm':float(tr['chargeNumberAnion']),
     'dx':float(config['geometry']['domainSize_m'])/grid_size,'dt':float(bc['timeStep_s']),
     'end':float(bc['endTime_s']),'evaluation':float(bc.get('evaluationTime_s',bc['endTime_s'])),
     'electrical_interval':float(bc['electricalUpdateInterval_s']), 'snapshot':float(bc.get('handoffSnapshotInterval_s',.02)),
     'layer':float(config['geometry']['surfaceLayerThickness_m']),
     'sigma_min':float(el['conductivityMinimum_S_per_m']),'sigma_max':float(el['conductivityMaximum_S_per_m']),
     'initial_T':float(th['initialTemperature_K']),
     'ambient':float(th.get('ambientTemperature_K',298.15)),
     'cp0':composition.initial_cation_mol_per_m3,'cm0':composition.initial_anion_mol_per_m3,
     'water0':composition.initial_water_mol_per_m3,'lp0':composition.initial_lp_mol_per_m3,
     'pva0':composition.initial_pva_repeat_mol_per_m3,'m_lp':composition.molar_mass_lp_kg_per_mol,
     'm_pva':composition.molar_mass_pva_repeat_kg_per_mol,'density':composition.density_kg_per_m3 if density is None else float(density),
     'w1':float(bc['kinetics']['mass_conversion_weights'][0]),'w2':float(bc['kinetics']['mass_conversion_weights'][1]),
     'rate_max':float(bc['kinetics'].get('maximum_rate_per_s',1e6)), 'min_T':float(th.get('minimumTemperature_K',200)),
     'max_T':float(th.get('maximumTemperature_K',3000)),'convection':float(th.get('convectionCoefficient_W_per_m2K',0)),
     'emissivity':float(th.get('emissivity',0)),'sb':float(th.get('stefanBoltzmann_W_per_m2K4',5.670374419e-8)),
     'min_fraction':float(tr['concentrationMinimumFraction']),'max_multiple':float(tr['concentrationMaximumMultiple']),
     'max_delta':float(tr['maximumRelativeConcentrationChangePerStep']),'lim_rtol':float(nu.get('limiterComparisonRelativeTolerance',1e-7)),
     'conc_floor':floor('concentration_mol_per_m3',1e-12),'current_floor':floor('current_A',1e-15),
     'j_floor':floor('currentDensity_A_per_m2',1e-12),'slope_floor':floor('currentDensitySlope_A_per_m2_V',1e-12),
     'conductivity_floor':floor('conductivity_S_per_m',1e-12),'cathode':float(el['cathodeVoltage_V']),
     'contact_normal':float(it['contactNormalConductionLength_m']),
     'exp_limit':float(it['exponentialArgumentLimit']),'full_bv':float(bool(it['useFullButlerVolmer'])),
     'full_np':float(bool(config['coupled'].get('useFullNernstPlanckTransport',True))),
     'diffusion':float(bool(el.get('diffusionPotentialEnabled',True))),
     'augmented':float(bool(bc.get('augmentedElectronicConduction',{}).get('enabled',False))),
     'onset_T':float(bc['onsetCriterion']['temperature_K']),'onset_X':float(bc['onsetCriterion']['minimum_progress']),
     'onset_area':float(bc['onsetCriterion']['minimum_area_fraction']),
     'thermal_cfl_safety':float(numerical_quality.get('maximumExplicitThermalCFL',
                                                      numerical_quality.get('maximumDiffusiveCFLFraction',1.0))),
     'max_steps':float(maximum_time_steps),'max_history_bytes':float(maximum_history_bytes),
     'linear_rtol':float(ps.get('relativeToleranceCoupled',1e-9)), 'linear_atol':float(ps.get('absoluteTolerance',1e-12)),
     'linear_max':float(ps.get('maximumIterationsCoupled',1500)),
    }
    names={
     'local_min':('localInterfaceMinimumIterations',8),'local_max':('localInterfaceMaximumIterations',60),
     'newton':('localNewtonPolishIterations',2),'local_abs':('localRobinResidualTolerance_A_per_m2',.01),
     'local_rel':('localRobinRelativeResidualTolerance',1e-4),'roundoff':('localRoundoffSafetyFactor',2),
     'balance_rel':('currentBalanceTolerance',.005),'balance_abs':('currentBalanceAbsoluteTolerance_A',1e-12),
     'outer_max':('maximumIterations',120),'outer_min':('minimumIterationsCoupled',ro.get('minimumIterations',2)),
     'outer_relax':('potentialUnderRelaxation',ro.get('underRelaxation',.35)),
     'outer_phi_tol':('potentialTolerance_V',.01),'outer_current_tol':('relativeReactionCurrentTolerance',.005),
    }
    cfg.update({k:float(ro.get(src,default)) for k,(src,default) in names.items()})
    integer_settings=('local_min','local_max','newton','linear_max','outer_min','outer_max')
    if any(not float(cfg[k]).is_integer() for k in integer_settings):
        raise ValueError('Native iteration budgets must be integers')
    end_match_tolerance=64.0*max(math.ulp(abs(cfg['end'])),math.ulp(abs(cfg['evaluation'])))
    if (cfg['dt']<=0 or cfg['end']<=0 or not 0<cfg['evaluation']<=cfg['end']
            or cfg['electrical_interval']<=0 or cfg['snapshot']<=0):
        raise ValueError('Require positive timeStep/update/snapshot intervals and 0<evaluationTime_s<=endTime_s')
    evaluation_is_end=abs(cfg['evaluation']-cfg['end'])<=end_match_tolerance
    if not evaluation_is_end and not _is_time_grid_aligned(cfg['evaluation'],cfg['dt']):
        raise ValueError('evaluationTime_s must align with the fixed timeStep_s grid')
    _native_step_count(cfg['end'],cfg['dt'],maximum_time_steps)
    _native_step_count(cfg['evaluation'],cfg['dt'],maximum_time_steps)
    electrical_ratio=cfg['electrical_interval']/cfg['dt']
    if electrical_ratio<1 or not _is_time_grid_aligned(cfg['electrical_interval'],cfg['dt']):
        raise ValueError('electricalUpdateInterval_s must be an integer multiple of timeStep_s and at least one step')
    if cfg['layer']<=0 or cfg['contact_normal']<=0 or cfg['density']<=0 or cfg['dx']<=0:
        raise ValueError('Grid spacing, surface/contact-normal thicknesses, and density must be positive')
    if (cfg['local_min']<1 or cfg['local_max']<cfg['local_min'] or cfg['linear_max']<1
            or cfg['outer_min']<1 or cfg['outer_max']<2 or cfg['outer_max']<cfg['outer_min']):
        raise ValueError('Invalid native iteration budget')
    if not 0<cfg['outer_relax']<=1 or cfg['linear_rtol']<=0 or cfg['linear_atol']<0:
        raise ValueError('Invalid native solver tolerance or relaxation')
    if cfg['balance_rel']<0 or cfg['balance_abs']<0 or (cfg['balance_rel']==0 and cfg['balance_abs']==0):
        raise ValueError('At least one non-negative current-balance tolerance must be positive')
    if cfg['local_abs']<0 or cfg['local_rel']<0 or (cfg['local_abs']==0 and cfg['local_rel']==0):
        raise ValueError('At least one non-negative local Robin tolerance must be positive')
    if cfg['min_T']>=cfg['max_T'] or cfg['rate_max']<=0 or cfg['R']<=0 or cfg['F']<=0:
        raise ValueError('Invalid native physical bounds/constants')
    if not 0<cfg['thermal_cfl_safety']<=1:
        raise ValueError('numericalQuality.maximumExplicitThermalCFL must lie in (0,1]')
    if min(cfg['cp0'],cfg['cm0'],cfg['lp0'],cfg['pva0'],cfg['m_lp'],cfg['m_pva'])<=0 or cfg['water0']<0:
        raise ValueError('Invalid initial condensed-phase inventory')
    if cfg['w1']<0 or cfg['w2']<0 or not math.isclose(cfg['w1']+cfg['w2'],1.0,rel_tol=0,abs_tol=1e-12):
        raise ValueError('Native kinetic mass-conversion weights must be non-negative and sum to one')
    if not 0<=cfg['min_fraction']<1 or cfg['max_multiple']<1 or cfg['max_delta']<=0:
        raise ValueError('Invalid native concentration bounds')
    if cfg['sigma_min']<=0 or cfg['sigma_max']<cfg['sigma_min'] or cfg['conductivity_floor']<=0:
        raise ValueError('Invalid native conductivity bounds')
    if (cfg['initial_T']<=0 or cfg['ambient']<=0
            or not cfg['min_T']<=cfg['initial_T']<=cfg['max_T']
            or not cfg['min_T']<=cfg['ambient']<=cfg['max_T']):
        raise ValueError('Invalid native initial/ambient temperature')
    if cfg['convection']<0 or not 0<=cfg['emissivity']<=1 or cfg['sb']<=0:
        raise ValueError('Invalid native surface heat-loss coefficients')
    if cfg['exp_limit']<=0 or cfg['roundoff']<1 or cfg['newton']<0:
        raise ValueError('Invalid native Butler-Volmer numerical bounds')
    if cfg['outer_phi_tol']<0 or cfg['outer_current_tol']<0:
        raise ValueError('Native outer convergence tolerances must be non-negative')
    if min(cfg['conc_floor'],cfg['slope_floor'])<0 or min(cfg['current_floor'],cfg['j_floor'])<=0:
        raise ValueError('Native concentration/slope floors must be non-negative and current/current-density floors strictly positive')
    heatmode=str(el.get('jouleHeatModel','conductive_sigma_E2')).lower()
    if heatmode in {'total_j_dot_e','j_dot_e','total'}:cfg['joule_mode']=0
    elif heatmode in {'conductive_sigma_e2','conductive','sigma_e2','jcond_dot_e'}:cfg['joule_mode']=1
    elif heatmode in {'legacy_magnitude_e_times_j','legacy_magnitude','e_times_jmag'}:cfg['joule_mode']=2
    else:raise ValueError(f'Unsupported Joule model: {heatmode}')
    table:list[float]=[]
    def prop(raw: Any, prefix:str) -> float:
        if isinstance(raw,(int,float)):
            value=float(raw)
            if not math.isfinite(value) or value<0:raise ValueError('Material-property constants must be finite and non-negative')
            cfg[prefix+'mode']=0;cfg[prefix+'value']=value;return value
        mode=str(raw.get('mode','table')).lower()
        if mode=='table':
            xs=list(map(float,raw['temperature_K']));ys=list(map(float,raw['values']))
            if (len(xs)<2 or len(xs)!=len(ys) or any(not math.isfinite(x) for x in xs+ys)
                    or any(b<=a for a,b in zip(xs,xs[1:])) or any(y<0 for y in ys)):
                raise ValueError('Material-property tables require finite, non-negative values on a strictly increasing temperature grid')
            cfg[prefix+'mode']=1;cfg[prefix+'offset']=len(table);cfg[prefix+'size']=len(xs);table.extend(xs+ys);return min(ys)
        elif mode in ('arrhenius','reference_arrhenius'):
            value=float(raw['reference_value']);tref=float(raw['reference_temperature_K']);ea=float(raw.get('activation_energy_J_per_mol',0))
            if not all(math.isfinite(x) for x in (value,tref,ea)) or value<0 or tref<=0 or ea<0:
                raise ValueError('Arrhenius properties require finite non-negative reference/activation values and positive reference temperature')
            cfg.update({prefix+'mode':2,prefix+'value':value,prefix+'tref':tref,prefix+'ea':ea});return value
        elif mode=='constant':
            value=float(raw['value'])
            if not math.isfinite(value) or value<0:raise ValueError('Material-property constants must be finite and non-negative')
            cfg[prefix+'mode']=0;cfg[prefix+'value']=value;return value
        else:raise ValueError(f'Unsupported material property {mode}')
    raws=[bc['transport'][k] for k in ('cation_diffusivity','anion_diffusivity','water_diffusivity')]
    raws += [bc.get('augmentedElectronicConduction',{}).get('conductivity',0.),th['heat_capacity'],th['thermal_conductivity']]
    property_minima=[prop(raw,f'p{i}.') for i,raw in enumerate(raws)]
    if property_minima[4]<=0:
        raise ValueError('Heat capacity must be strictly positive')
    for i,k in enumerate(channels):
        cfg.update({f'k{i}.offset':len(table),f'k{i}.size':len(k.alpha_grid),f'k{i}.heat':k.heat_release_J_per_kg})
        table.extend(k.alpha_grid+k.activation_energy_J_per_mol+k.ln_Af_per_s)
    for i,(species,polarity) in enumerate((('water','anode'),('water','cathode'),('lp','anode'),('lp','cathode'))):
        ch=it[species][polarity];prefix=f'ch{i}.'
        for name,src,default in [('eq','equilibriumPotential_V',0),('j0','exchangeCurrentDensity_A_per_m2',0),('alpha','chargeTransferCoefficient',.5),('n','electronNumber',1),('reverse','reverseAvailabilityFraction',1),('heat','reactionEnthalpy_J_per_mol',0),('stoich','waterStoichiometry_mol_per_molElectron' if species=='water' else 'saltStoichiometry_mol_per_molElectron',0)]:
            cfg[prefix+name]=float(ch.get(src,default))
        if cfg[prefix+'j0']<0 or not 0<=cfg[prefix+'alpha']<=1 or cfg[prefix+'n']<=0 or cfg[prefix+'reverse']<0:
            raise ValueError(f'Invalid Butler-Volmer parameters for {species}/{polarity}')
        if cfg[prefix+'heat']<0 or cfg[prefix+'stoich']<0:
            raise ValueError(f'Native tracked reaction heat/stoichiometric sinks must be non-negative for {species}/{polarity}')
    # All numeric payloads are finite. Configuration/table validation precedes native calls.
    if any(not math.isfinite(float(v)) for v in cfg.values()) or any(not math.isfinite(v) for v in table):raise ValueError('Nonfinite native configuration')
    return {k:float(v) for k,v in cfg.items()},table

def _canonical_native_bool_mask(mask: torch.Tensor, name: str) -> torch.Tensor:
    """Canonicalise a Torch bool mask by inspecting its underlying bytes.

    Pillow mode-1 arrays may reach ``torch.as_tensor`` with dtype bool but
    0xff true bytes.  Reading such storage through a C++ ``bool*`` is undefined
    behaviour.  Reinterpret as uint8 first, then compare against zero so the
    result owns canonical 0/1 bool bytes on either CPU or CUDA.
    """
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise ValueError(f'Native {name} mask must be a Torch bool tensor')
    # Reinterpret before making the tensor contiguous: a preliminary bool
    # copy could itself read a noncanonical C++ bool representation.
    return mask.view(torch.uint8).ne(0).contiguous()

def run_native_batch(geometry, config, composition, voltage, dtype=torch.float64, *, save_fields=False, save_handoff=False,
                     stop_on_onset=False, native_options:Mapping[str,Any]|None=None):
    if dtype!=torch.float64:raise ValueError('Native B/C requires FP64')
    device=geometry.propellant.device
    if device.type not in ('cpu','cuda'):raise ValueError('Use CPU FP64 on Apple Silicon; MPS does not support this native backend')
    if geometry.batch_size<1:
        raise ValueError('Native integration requires at least one candidate')
    native_anode=_canonical_native_bool_mask(geometry.anode,'anode')
    native_cathode=_canonical_native_bool_mask(geometry.cathode,'cathode')
    native_fixed=_canonical_native_bool_mask(geometry.fixed,'fixed')
    native_propellant=_canonical_native_bool_mask(geometry.propellant,'propellant')
    if any(mask.device!=device for mask in (native_anode,native_cathode,native_fixed,native_propellant)):
        raise ValueError('All native geometry masks must share one device')
    expected=torch.ones_like(native_propellant,dtype=torch.bool)
    if not torch.equal(native_propellant,expected):
        raise ValueError('Native surface-overlay B/C requires propellant on every cell, including both contact footprints')
    if bool(torch.any(native_fixed).item()):
        raise ValueError('Native surface-overlay B/C requires no fixed bulk cells; electrodes are separate contact labels')
    _validate_bc_runtime_contract(config,composition,geometry)
    cfg,table=pack_config(config,composition,geometry.grid_size);opt=dict(native_options or {})
    configured_steps=_native_step_count(
        cfg['end'],cfg['dt'],int(cfg['max_steps'])
    )
    history_allocation_bytes=(
        len(HISTORY_NAMES)*configured_steps*geometry.batch_size*8
    )
    if save_handoff:
        snapshot_period=max(cfg['snapshot'],cfg['dt'])
        snapshot_count_upper_bound=min(
            configured_steps,
            2+_native_step_count(
                cfg['end'],snapshot_period,int(cfg['max_steps'])
            ),
        )
        # Four full-grid fields are retained and then stacked, so account for
        # both the list storage and the transient stacked tensors.
        history_allocation_bytes+=(
            2*4*snapshot_count_upper_bound*geometry.batch_size
            *geometry.grid_size*geometry.grid_size*8
        )
    if history_allocation_bytes>int(cfg['max_history_bytes']):
        raise ValueError(
            'Native B/C retained-history allocation exceeds '
            'numericalQuality.maximumHistoryAllocationBytes: '
            f'required={history_allocation_bytes}, maximum={int(cfg["max_history_bytes"])}'
        )
    cfg.update(stop_on_onset=float(stop_on_onset),save_handoff=float(save_handoff),
               time_check=float(opt.get('host_check_interval_steps',16)),linear_check=float(opt.get('linear_check_interval',8)))
    volts=torch.as_tensor(voltage,device=device,dtype=dtype).reshape(-1)
    if volts.numel()==1:volts=volts.expand(geometry.batch_size).contiguous()
    if volts.numel()!=geometry.batch_size or not torch.isfinite(volts).all() or not (volts>cfg['cathode']).all():
        raise ValueError('Each independent trial needs one finite voltage above the cathode potential')
    use_cli=device.type=='cpu' and opt.get('cpu_runtime','extension')=='standalone'
    with torch.inference_mode():
        args=(native_anode,native_cathode,volts.contiguous(),
              torch.tensor(table,device=device,dtype=dtype),cfg)
        if use_cli:
            from .standalone import run_standalone
            raw=run_standalone(*args,threads=int(opt.get('torch_threads',1)),
                               timeout_s=float(opt.get('trial_timeout_s',0)))
        else:
            native=load_native(device.type=='cuda',bool(opt.get('verbose_build',False)))
            raw=native.run(*args)
        out=_package(raw,geometry,config,composition,volts,save_fields,save_handoff,stop_on_onset,native_propellant)
        out['nativeExecution']['cpu_runtime']='standalone_executable' if use_cli else 'libtorch_extension'
        return out

def _package(raw,geometry,config,composition,volts,save_fields,save_handoff,stop_only,masked):
    bc=config['bcGlobal'];dt=float(bc['timeStep_s']);end=float(bc['endTime_s']);evaluation=float(bc.get('evaluationTime_s',end));device=volts.device
    numerical_quality=config.get('numericalQuality',{})
    if not isinstance(numerical_quality,Mapping):numerical_quality={}
    s=raw['state'];hist={k:raw['histories'][i] for i,k in enumerate(HISTORY_NAMES)};nsteps=raw['histories'].shape[1]
    maximum_time_steps,_=_bc_resource_limits(config)
    configured_steps=_native_step_count(end,dt,maximum_time_steps);evaluation_steps=_native_step_count(evaluation,dt,maximum_time_steps)
    steps=raw['steps_completed'];advanced_complete=steps>=configured_steps;delay=raw['ignitionDelay'];success=torch.isfinite(delay)
    history_complete=torch.full_like(success,nsteps>=configured_steps,dtype=torch.bool)
    idx=min(max(evaluation_steps-1,0),nsteps-1)
    def evaluation_value(name):
        value=hist[name][idx]
        if stop_only:value=torch.where(steps>=evaluation_steps,value,torch.full_like(value,torch.nan))
        return value
    def evaluation_peak(name):
        value=hist[name][:idx+1].amax(0)
        if stop_only:value=torch.where(steps>=evaluation_steps,value,torch.full_like(value,torch.nan))
        return value
    inventory=_reaction_inventory(
        s[6],composition,
        mobile_lp_mol_per_m3=.5*(s[1]+s[2]),
        electrochemical_lp_consumed_mol_per_m3=s[9],
        generated_water_product_mol_per_m3=s[8],
    )
    rem=_masked_integral_ratio(
        inventory['reactiveMass_kg_per_m3'],
        inventory['initialReactiveMass_kg_per_m3'],
        masked,
    )
    remaining_tolerance=128.0*torch.finfo(volts.dtype).eps
    invalid_rem=(~torch.isfinite(rem))|(rem < -remaining_tolerance)|(rem > 1.0+remaining_tolerance)
    if bool(torch.any(invalid_rem).item()):
        bad=torch.nonzero(invalid_rem).flatten().detach().cpu().tolist()
        raise BCCandidateBatchError(
            'Native packaged remaining reactive-mass fraction is non-finite or outside [0, 1]',
            bad,
            'native_packaged_remaining_reactive_mass_invariant',
        )
    rem=rem.clamp(0.0,1.0)
    floor=float(config['numerics'].get('physicalFloors',{}).get('current_A',1e-15))
    compiler_standard=required_cpp_standard();standard_tag=compiler_standard.replace('+','p')
    time_axis=torch.arange(1,nsteps+1,device=device,dtype=volts.dtype)*dt
    if nsteps==configured_steps:
        time_axis[-1]=end
    out={
     'ignitionDelay_s':delay,'condensedPhaseIgnitionDelay_s':delay,'ignitionSucceeded':success,
     'areaAveragedUndecomposedFractionAt2s':evaluation_value('remainingReactiveMassFraction'),
     'areaAveragedUndecomposedFractionAtEvaluationTime':evaluation_value('remainingReactiveMassFraction'),
     'remainingReactiveMassFractionAt2s':evaluation_value('remainingReactiveMassFraction'),
     'remainingReactiveMassFractionAtEvaluationTime':evaluation_value('remainingReactiveMassFraction'),
     'meanGlobalProgressAt2s':evaluation_value('meanGlobalProgress'),'meanGlobalProgressAtEvaluationTime':evaluation_value('meanGlobalProgress'),
     'temperatureOnsetAreaFractionAt2s':evaluation_value('onsetAreaFraction'),
     'inputElectricalEnergyToIgnition_J':raw['ignition_energy'],'inputElectricalEnergyAt2s_J':evaluation_value('energy_J'),
     'inputElectricalEnergyAtEvaluationTime_J':evaluation_value('energy_J'),'inputElectricalEnergy_J':hist['energy_J'][-1],
     'peakMaximumTemperature_K':hist['maxTemperature_K'].amax(0),'peakCurrent_A':hist['current_A'].amax(0),
     'peakCurrentCongestion':hist['currentCongestion'].amax(0),
     'peakCurrentCongestionToEvaluationTime':evaluation_peak('currentCongestion'),
     'peakCurrentCongestionTo2s':evaluation_peak('currentCongestion'),
     'finalEffectiveResistance_ohm':(volts-float(config['electrical']['cathodeVoltage_V']))/
         hist['current_A'].gather(0,(steps.clamp_min(1)-1).reshape(1,-1)).squeeze(0).clamp_min(floor),
     'preContinuationEffectiveResistanceAtEvaluationTime_ohm':
         (volts-float(config['electrical']['cathodeVoltage_V']))/
         evaluation_value('current_A').clamp_min(floor),
     'maximumSpeciesLimiterFraction':raw['caps'][1],'maximumTemperatureCapFraction':raw['caps'][0],
     'maximumGasCapFraction':torch.zeros_like(volts),'maximumChemicalRateCapFraction':raw['caps'][2],
     'maximumThermalDiffusiveCFL':raw['caps'][4],
     'maximumThermalStabilityCFL':raw['caps'][3],
     'equation32ElectricalHeatRateAtEvaluationTime_W':evaluation_value('equation32ElectricalHeatRate_W'),
     'equation32ElectricalHeatEnergyAtEvaluationTime_J':evaluation_value('equation32ElectricalHeatEnergy_J'),
     'equation32ElectricalHeatEnergy_J':hist['equation32ElectricalHeatEnergy_J'][-1],
     'finalMeanChemicalProgress':hist['meanGlobalProgress'][-1],'finalGlobalProgress':hist['meanGlobalProgress'][-1],
     'finalAreaAveragedUndecomposedFraction':rem,'finalRemainingReactiveMassFraction':rem,
     'maximumAnodeCathodeCurrentMismatch':hist['anodeCathodeCurrentMismatch'].amax(0),
     'meanAnodeCathodeCurrentMismatch':hist['anodeCathodeCurrentMismatch'].mean(0),
     'finalAnodeCathodeCurrentMismatch':hist['anodeCathodeCurrentMismatch'][-1],
     # The compiled engine aborts immediately on any linear or nonlinear
     # Robin nonconvergence.  A returned lane has therefore converged at every
     # solve through its active pre-flame lifetime.
     'allElectricalLinearSolvesConverged':raw['linear_converged'],
     'allNonlinearRobinSolvesConverged':torch.ones_like(success,dtype=torch.bool),
     'surfaceContactModel':True,
     'propellantCellCount':int(masked[0].sum().item()),
     'electrodeMasksRemovePropellant':False,
     'contactCurrentIntegration':'sum_j_contact_times_dx_squared',
     'volumetricSourceConversion':'j_contact_divided_by_surface_layer_thickness',
     'solverDiagnostics':NativeDiagnostics(raw,device.type),'histories':hist,
     'time_s':torch.clamp(time_axis,max=end),'evaluationTime_s':evaluation,
     'modelStatus':'bc_global_corrected_surface_overlay_preflame_literature_nominal_uncalibrated',
     'postIgnitionClosure':'candidate_state_power_and_heat_frozen_at_first_condensed_onset',
     'preflameTermination':{
       'stopOnOnsetRequested':bool(stop_only),
       'candidateStatesFrozenAtFirstOnset':True,
       'allCandidatesReachedOnset':bool(torch.all(success).item()),
       'terminatedAfterAllCandidatesReachedOnset':bool(torch.all(success).item() and int(steps.max().item())<configured_steps),
       'executedSteps':int(steps.max().item()),'configuredSteps':configured_steps,
       'historyTailPolicy':'state_and_cumulative_hold_with_zero_instantaneous_tail' if not stop_only else 'partial_onset_trial'},
     'ignitionCriterion':{'type':'temperature_and_global_conversion_over_minimum_area','temperature_K':float(bc['onsetCriterion']['temperature_K']),
       'minimumGlobalProgress':float(bc['onsetCriterion']['minimum_progress']),'minimumAreaFraction':float(bc['onsetCriterion']['minimum_area_fraction']),
       'interpretation':'operational condensed-phase decomposition onset, not visible gas-flame ignition'},
     'globalReaction':{'equation':BC_GLOBAL_REACTION,'lpStoichiometricCoefficient':-1.45,'pvaRepeatStoichiometricCoefficient':-1.,
       'co2StoichiometricCoefficient':2.,'waterStoichiometricCoefficient':2.,'oxygenStoichiometricCoefficient':.4,'liClStoichiometricCoefficient':1.45,
       'scope':'reduced_global_chemistry_not_elementary_mechanism'},
     'kineticChannels':[asdict(c) for c in _validate_bc_config(config)],
     'stabilityDiagnostics':{'spacing_m':float(config['geometry']['domainSize_m'])/geometry.grid_size,'configured_dt_s':dt,
       'maximum_thermal_diffusive_CFL':raw['caps'][4],
       'maximum_total_explicit_thermal_CFL':raw['caps'][3],
       'allowed_maximum_total_explicit_thermal_CFL':float(numerical_quality.get('maximumExplicitThermalCFL',
                                                                                numerical_quality.get('maximumDiffusiveCFLFraction',1.0))),
       'thermal_stability_CFL_definition':'dt*(sum(harmonic_face_k)/dx^2 + h/L + 4*epsilon*sigma*max(T,Tambient)^3/L)/(rho*cp); fail_closed_above_limit'},
     'paperEquationUse':{'used':list(BC_PAPER_EQUATIONS_USED),'not_used_in_preflame':list(BC_PAPER_EQUATIONS_NOT_USED_IN_PREFLAME),'equation32_role':BC_EQUATION_32_ROLE},
     'nativeExecution':{'engine':f'{standard_tag}_cuda_fp64' if device.type=='cuda' else f'{standard_tag}_cpu_fp64','compiled_time_loop':True,
       'custom_cuda_kernels':device.type=='cuda','trialOnly':bool(stop_only),'validity_window':'through_first_onset' if stop_only else 'full_horizon',
       'inherited_geometry_semantics':'surface_contact_overlay_full_propellant_domain',
       'electrodeMasksRemovePropellant':False,
       'contactCurrentIntegration':'sum_j_contact_times_dx_squared',
       'volumetricSourceConversion':'j_contact_divided_by_surface_layer_thickness',
       'contactNormalConductionLength_m':float(config['interface']['contactNormalConductionLength_m']),
       'surfaceLayerThickness_m':float(config['geometry']['surfaceLayerThickness_m']),
       'compilerLanguageStandard':compiler_standard,'coreSourceLanguageFloor':'c++17','stepsExecutedPerCandidate':steps,
       'historyCoversFullHorizon':history_complete,'integrationAdvancedFullHorizon':advanced_complete,
       'completedFullHorizon':history_complete,
       'completedFullHorizonLegacyAliasDeprecated':'use_historyCoversFullHorizon_and_integrationAdvancedFullHorizon',
       'advancedThroughRequestedEndTime':advanced_complete,
       'stateEvolutionPolicy':'freeze_at_first_onset',
       'postOnsetHistorySemantics':'last_preflame_state_held_no_source_replay',
       'integrationStoppedAtOnset':success&~advanced_complete,
       'earlyStopped':bool(stop_only)&success&~advanced_complete,'requestedEndTime_s':end},
    }
    time_tolerance=_time_match_tolerance(end,evaluation)
    trial_evaluation_unavailable=(bool(stop_only)&success&(delay<evaluation-time_tolerance))
    unavailable=torch.full_like(volts,torch.nan)
    out['preflameTerminationRemainingReactiveMassFraction']=rem
    out['preflameTerminationMeanGlobalProgress']=hist['meanGlobalProgress'][-1]
    out['preflameTerminationMaximumTemperature_K']=hist['maxTemperature_K'][-1]
    out['maximumTemperatureAtEvaluationTime_K']=evaluation_value('maxTemperature_K')
    out['evaluationStateAvailable']=~trial_evaluation_unavailable
    out['evaluationStateTime_s']=torch.where(
        trial_evaluation_unavailable,unavailable,torch.full_like(volts,evaluation))
    out['objectiveEvaluationStateTime_s']=out['evaluationStateTime_s']
    if stop_only:
        for key in ('areaAveragedUndecomposedFractionAt2s','areaAveragedUndecomposedFractionAtEvaluationTime',
                    'remainingReactiveMassFractionAt2s','remainingReactiveMassFractionAtEvaluationTime',
                    'meanGlobalProgressAt2s','meanGlobalProgressAtEvaluationTime',
                    'temperatureOnsetAreaFractionAt2s','maximumTemperatureAtEvaluationTime_K'):
            out[key]=torch.where(trial_evaluation_unavailable,unavailable,out[key])
    if save_fields:
        out['finalFields']={'temperature_K':s[0],'potential_V':s[7],'cation_mol_per_m3':s[1],'anion_mol_per_m3':s[2],
                           'water_mol_per_m3':s[3],'mobileWater_mol_per_m3':s[3],
                           'generatedWaterProduct_mol_per_m3':s[8],
                           'electrochemicalLPConsumed_mol_per_m3':s[9],
                           'alphaChannel1':s[4],'alphaChannel2':s[5],'globalProgress':s[6],**inventory}
    ons=raw['onset_state'];oh=raw['onset_heat']
    onset_fields={
          'temperatureAtOnset_K':ons[0],'globalProgressAtOnset':ons[6],'alphaChannel1AtOnset':ons[4],'alphaChannel2AtOnset':ons[5],
          'cationAtOnset_mol_per_m3':ons[1],'anionAtOnset_mol_per_m3':ons[2],
          'mobileLPAtOnset_mol_per_m3':.5*(ons[1]+ons[2]),
          'mobileWaterAtOnset_mol_per_m3':ons[3],
          'generatedWaterProductAtOnset_mol_per_m3':ons[8],
          'electrochemicalLPConsumedAtOnset_mol_per_m3':ons[9],
          'pvaReactiveRepeatAtOnset_mol_per_m3':torch.clamp(
              torch.full_like(ons[6],composition.initial_pva_repeat_mol_per_m3)
              - min(composition.initial_lp_mol_per_m3/1.45,composition.initial_pva_repeat_mol_per_m3)*ons[6],min=0),
          'potentialAtOnset_V':ons[7],
          'xiMax_mol_per_m3':min(composition.initial_lp_mol_per_m3/1.45,composition.initial_pva_repeat_mol_per_m3),
          'molarMassLP_kg_per_mol':composition.molar_mass_lp_kg_per_mol,
          'molarMassPVARepeat_kg_per_mol':composition.molar_mass_pva_repeat_kg_per_mol,
          'initialReactiveMass_kg_per_m3':(
              composition.molar_mass_lp_kg_per_mol*composition.initial_lp_mol_per_m3
              + composition.molar_mass_pva_repeat_kg_per_mol*composition.initial_pva_repeat_mol_per_m3),
          'initialMobileLP_mol_per_m3':composition.initial_lp_mol_per_m3,
          'initialPVARepeat_mol_per_m3':composition.initial_pva_repeat_mol_per_m3,
          'initialMobileWater_mol_per_m3':composition.initial_water_mol_per_m3,
          'qJAtOnset_W_per_m3':oh[0],'qEchemAtOnset_W_per_m3':oh[1],'propellantMask':masked,'ignitionDelay_s':delay,
          'onsetSucceeded':success,'continuedElectricalHeating':False,
          'postOnsetElectricalPolicy':'recompute_required_no_preflame_replay',
          'preflameElectricalHistoryMayBeReplayed':False}
    out['onsetFields']=onset_fields
    if save_handoff:
        out['handoffFields']={
          'times_s':raw['snapshot_times'],'qJ_W_per_m3':raw['snapshot_qj'],'qEchem_W_per_m3':raw['snapshot_qe'],
          'temperatureHistory_K':raw['snapshot_T'],'globalProgressHistory':raw['snapshot_X'],
          **onset_fields}
    if not stop_only:
        complete_bc_evaluation_output(
            out,onset_fields,config,composition,volts.dtype
        )
    return out
