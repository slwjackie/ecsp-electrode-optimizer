"""Three pre-flame objectives from unmodified production metrics.

No PDE, onset criterion, time integration or voltage search is changed here.
A failed/censored denominator never becomes a favourable finite score.
"""
from __future__ import annotations
import copy, hashlib, json, math
import numpy as np

OBJECTIVES = ('ignition_delay_over_staggered',
              'minimum_voltage_over_staggered', 'current_congestion')
RAW_OBJECTIVES = ('ignitionDelay_s','minimumIgnitionVoltage_V','peakCurrentCongestionToEvaluationTime')
BASE_COMMIT = '99d80e33626c0753917f95c1ba8bf51ba0459ae0'


def strict_json(value):
    if isinstance(value,dict):return {str(k):strict_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [strict_json(v) for v in value]
    if isinstance(value,np.ndarray):return strict_json(value.tolist())
    if isinstance(value,np.generic):return strict_json(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def digest(value):
    return hashlib.sha256(json.dumps(strict_json(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def positive(value):
    try:return value is not None and math.isfinite(float(value)) and float(value)>0
    except (TypeError,ValueError):return False


def runtime_config(base,meta):
    """Per-pair geometry changes only. Keep physical/numerical blocks intact."""
    c=copy.deepcopy(base)
    L=float(meta['domain_mm']);n=int(meta['grid_size'])
    Aa=float(meta['anode_area_mm2']);Ac=float(meta['cathode_area_mm2'])
    if min(L,Aa,Ac)<=0 or abs(Aa-Ac)/max(Aa,Ac)>1e-6:
        raise ValueError('Candidate CAD must have equal positive electrode areas')
    g=c.setdefault('geometry',{})
    g.update(domain_mm=L,grid_size=n,minimum_gap_mm=3.,minimum_width_mm=2.,
             maximum_width_mm=L,target_area_fraction_per_polarity=(Aa+Ac)/(2*L*L),
             area_tolerance_fraction=.01,
             maximum_components_per_polarity=max(2,int(meta['anode_components']),int(meta['cathode_components'])),
             maximum_total_components=max(4,int(meta['anode_components'])+int(meta['cathode_components'])),
             disconnected_component_powering='implicit_3d_hidden_bus',
             surface_contact_model='overlay_on_full_propellant_domain')
    # Equal design/physics grids avoid a second, geometry-changing resize.
    c.setdefault('evaluator',{})['grid_size']=n
    for key in set(base)|set(c):
        if key not in {'geometry','evaluator'} and c[key]!=base[key]:
            raise AssertionError('Protected configuration changed: '+key)
    old=copy.deepcopy(base.get('evaluator',{}));new=copy.deepcopy(c['evaluator'])
    old.pop('grid_size',None);new.pop('grid_size',None)
    if old!=new:raise AssertionError('Evaluator numerics changed')
    return c


def context(meta,config_hash):
    return dict(domain_mm=float(meta['domain_mm']),grid_size=int(meta['grid_size']),
                anode_area_mm2=float(meta['anode_area_mm2']),cathode_area_mm2=float(meta['cathode_area_mm2']),
                electrode_area_fraction=float(meta['electrode_area_fraction']),
                minimum_gap_mm=3.,minimum_width_mm=2.,physics_config_hash=config_hash)


def assert_matched(candidate,baseline,rtol=.01):
    for key in ('domain_mm','grid_size','physics_config_hash','minimum_gap_mm','minimum_width_mm'):
        if candidate[key]!=baseline[key]:raise ValueError('Mismatched baseline '+key)
    for key in ('anode_area_mm2','cathode_area_mm2'):
        if abs(candidate[key]-baseline[key])/candidate[key]>rtol:
            raise ValueError('Mismatched actual baseline '+key)
    # Coverage alone is NEVER a cache/matching key.


def baseline_key(ctx,config,mask_hash):
    return digest(dict(context=ctx,config=config,baseline_mask_hash=mask_hash,
                       source_commit=BASE_COMMIT,schema='matched-three-v1'))


def metric_reasons(row,label):
    errors=[]
    if row.get('converged') is not True or row.get('physicsRejected',False):errors.append(label+':numerical_rejection')
    for key in ('allElectricalLinearSolvesConverged','allNonlinearRobinSolvesConverged'):
        if row.get(key) is not True:errors.append(label+':'+key)
    if row.get('external_numerical_valid') is not True:errors.append(label+':production_validity_not_confirmed')
    if row.get('ignitionSucceeded') is not True:errors.append(label+':no_ignition')
    if not positive(row.get('ignitionDelay_s')):errors.append(label+':invalid_ignition_delay')
    if row.get('minimumIgnitionVoltageSearchValid') is not True:errors.append(label+':invalid_voltage_search')
    for key in ('minimumIgnitionVoltageLeftCensored','minimumIgnitionVoltageRightCensored'):
        if row.get(key) is not False:errors.append(label+':'+key)
    if not positive(row.get('minimumIgnitionVoltage_V')):errors.append(label+':invalid_measured_voltage')
    return errors


def normalized_objectives(candidate,baseline,candidate_context,baseline_context):
    assert_matched(candidate_context,baseline_context)
    reasons=metric_reasons(candidate,'candidate')+metric_reasons(baseline,'baseline')
    J=candidate.get('peakCurrentCongestionToEvaluationTime',candidate.get('peakCurrentCongestion'))
    if not positive(J):reasons.append('candidate:invalid_current_congestion')
    result={'objective_names':list(OBJECTIVES),'objective_vector':None,
            'normalization_valid':not reasons,'reasons':reasons,
            'electrode_area_fraction':candidate_context['electrode_area_fraction'],
            'voltage_ratio_interpretation':'ratio of tested conservative upper ignition brackets, not exact continuous thresholds',
            'normalization_scope':'same physical domain, per-polarity area, grid and physics settings; not scale invariance'}
    if not reasons:
        result['objective_vector']=[float(candidate['ignitionDelay_s'])/float(baseline['ignitionDelay_s']),
          float(candidate['minimumIgnitionVoltage_V'])/float(baseline['minimumIgnitionVoltage_V']),float(J)]
    return result


def pareto_rows(rows):
    """Only fully valid, finite THREE-vectors participate. Raw data remain saved."""
    eligible=[r for r in rows if r.get('normalization_valid') and len(r.get('objective_vector',[]))==3]
    vals=np.asarray([r['objective_vector'] for r in eligible],float)
    if len(vals) and not np.isfinite(vals).all():raise ValueError('Nonfinite Pareto objective')
    for i,r in enumerate(eligible):
        r['pareto_front']=not bool(np.any(np.all(vals<=vals[i],axis=1)&np.any(vals<vals[i],axis=1)))
    return eligible
