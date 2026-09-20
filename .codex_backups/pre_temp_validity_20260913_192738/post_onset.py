"""Selectable BC post-onset backend and paired same-handoff comparisons.

Absence of post_onset keeps the original condensed_propagation backend.
The optional Reactive result never replaces a failed baseline (or vice versa).
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import copy
import csv
import hashlib
import json
import math
import numpy as np
from scipy.stats import rankdata, spearmanr
from .propagation import (PropagationConfigurationError,PropagationCandidateError,
                          PropagationCandidateInputError,run_condensed_propagation)

BACKENDS=("condensed_propagation","reactive_euler")
COMPARISON_METRICS=(
    "establishedTimeAfterOnset_s","finalUnreactedAreaFraction","finalRemainingReactiveMassFraction",
    "meanEffectiveRegressionVelocity_m_per_s","maximumEffectiveRegressionVelocity_m_per_s",
    "reactionFrontNonuniformity","maximumTemperatureDuringPropagation_K","finalMaximumTemperature_K",
    "initialMass_kg","finalMass_kg","finalMassBudgetResidual_kg","finalEnergyBudgetResidual_J",
    "integratedJouleHeat_J","integratedElectrochemicalHeat_J","integratedChemicalHeat_J","integratedHeatLoss_J")


def validate_post_onset_config(raw,*,for_optimization=False):
    if not isinstance(raw,Mapping):
        raise PropagationConfigurationError("post_onset must be a mapping")
    cfg=copy.deepcopy(dict(raw)); backend=str(cfg.get("backend","condensed_propagation"))
    if backend not in BACKENDS:
        raise PropagationConfigurationError(f"post_onset.backend must be one of {BACKENDS}")
    cfg["backend"]=backend
    compare=cfg.get("compare_backends",False)
    if not isinstance(compare,bool):
        raise PropagationConfigurationError("post_onset.compare_backends must be boolean")
    cfg["compare_backends"]=compare
    if not isinstance(cfg.get("allow_experimental_ranking",False),bool):
        raise PropagationConfigurationError("post_onset.allow_experimental_ranking must be boolean")
    if backend=="reactive_euler" or compare:
        reactive=cfg.get("reactive_euler")
        if not isinstance(reactive,Mapping) or "eos" not in reactive:
            raise PropagationConfigurationError("Reactive selection requires explicit post_onset.reactive_euler.eos")
        closure=reactive.get("caloric_closure","bc_cp_integral_plus_tait_cold_energy")
        if closure!="bc_cp_integral_plus_tait_cold_energy":
            raise PropagationConfigurationError("Unknown caloric closure; the coupled path cannot use legacy fixed cv")
    if for_optimization and backend=="reactive_euler" and not cfg.get("allow_experimental_ranking",False):
        raise PropagationConfigurationError(
            "Reactive rankings are not experimentally validated. Set post_onset.allow_experimental_ranking=true "
            "to request numerical-only provisional rankings, or keep condensed_propagation and compare_backends=true")
    if "execution" in cfg:
        from ecsp_reactive.condensed.tensor_solver import validate_execution_config
        cfg["execution"]=validate_execution_config(cfg["execution"])
    return cfg


def handoff_digest(handoff):
    h=hashlib.sha256()
    for k in sorted(handoff):
        h.update(k.encode())
        v=handoff[k]
        if isinstance(v,np.ndarray):
            h.update(str(v.dtype).encode());h.update(str(v.shape).encode());h.update(np.ascontiguousarray(v).tobytes())
        elif isinstance(v,np.generic):
            h.update(repr(v.item()).encode())
        else:
            h.update(repr(v).encode())
    return h.hexdigest()


def _json(path,payload):
    Path(path).write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8")


def _comparison(results,handoff_hash,out):
    payload={"sameOnsetSHA256":handoff_hash,"scope":"same_BC_state_same_duration_same_power_policy",
             "experimentalValidationPerformed":False,"backends":results,
             "frontDefinition":"same_weighted_two_channel_threshold; effective_regression_not_ablation",
             "energyAuditNote":"baseline_explicit_T_update_vs_Reactive_conservative_cp_integral_energy",
             "metrics":{}}
    b=results.get("condensed_propagation",{});r=results.get("reactive_euler",{})
    complete=all(results.get(name,{}).get("status")=="complete" for name in BACKENDS)
    payload["bothBackendsCompleted"]=complete
    rows=[]
    for key in COMPARISON_METRICS:
        lv=b.get(key);rv=r.get(key)
        valid=all(isinstance(v,(float,int)) and math.isfinite(v) for v in (lv,rv))
        delta=rv-lv if valid else None
        relative=(abs(delta)/max(abs(lv),abs(rv),1e-30)) if valid else None
        row={"metric":key,"condensed_propagation":lv,"reactive_euler":rv,
             "reactive_minus_condensed":delta,"symmetric_relative_difference":relative}
        rows.append(row);payload["metrics"][key]=row
    payload["established_time_censoring"]={name:results.get(name,{}).get("establishedTimeCensored") for name in BACKENDS}
    _json(out/"backend_comparison.json",payload)
    with (out/"backend_comparison.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return payload


def run_post_onset(handoff,propagation_config,bc_config,output_dir,*,post_onset_config=None,full_bc_config=None):
    cfg=validate_post_onset_config(post_onset_config or {})
    if "execution" in cfg:
        from .post_onset_batch import run_post_onset_batch
        result=run_post_onset_batch([handoff],propagation_config,bc_config,[output_dir],
                 post_onset_config=cfg,full_bc_config=full_bc_config)[0]
        if isinstance(result,PropagationCandidateError):raise result
        return result
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    selected=cfg["backend"]
    if cfg["compare_backends"] and bool(propagation_config.get("continued_electrical_heating",False)):
        raise PropagationConfigurationError("Paired comparison requires power-off on both: the preserved baseline has no recomputed electrical solver")
    if not bool(handoff.get("onsetSucceeded",False)):
        # Preserve the existing no-onset authorization/censoring contract. No
        # Reactive initial state is fabricated from an unsuccessful trial.
        m=run_condensed_propagation(handoff,propagation_config,bc_config,out/selected)
        m["postOnsetBackend"]=selected;m["propagationDirectory"]=str(out/selected)
        return m
    requested=list(BACKENDS) if cfg["compare_backends"] else [selected]
    results={}; primary_error=None; before=handoff_digest(handoff)
    for name in requested:
        try:
            if name=="condensed_propagation":
                result=run_condensed_propagation(copy.deepcopy(handoff),propagation_config,bc_config,out/name)
            else:
                from ecsp_reactive.condensed.solver import run_reactive_propagation
                result=run_reactive_propagation(copy.deepcopy(handoff),propagation_config,bc_config,out/name,
                                                reactive_config=cfg["reactive_euler"],full_bc_config=full_bc_config)
            results[name]=dict(result)
        except PropagationCandidateError as exc:
            results[name]={"status":"failed","errorType":type(exc).__name__,"message":str(exc)}
            (out/name).mkdir(exist_ok=True)
            _json(out/name/"PROPAGATION_FAILED.json",results[name])
            if name==selected: primary_error=exc
    if before!=handoff_digest(handoff):
        raise RuntimeError("Post-onset backend mutated the original handoff")
    pair=_comparison(results,before,out) if cfg["compare_backends"] else None
    if primary_error is not None:
        raise primary_error
    metrics=dict(results[selected]);metrics["postOnsetBackend"]=selected
    metrics["propagationDirectory"]=str(out/selected)
    metrics["postOnsetSameHandoffSHA256"]=before
    metrics["postOnsetBackendComparison"]=pair
    metrics["experimentalReactiveRankingRequested"]=selected=="reactive_euler"
    return metrics


def _nondominated_ranks(F):
    n=len(F); ranks=np.full(n,-1,int); pending=np.arange(n);level=0
    while pending.size:
        f=F[pending]
        dominated=np.zeros(len(pending),bool)
        for j,row in enumerate(f):
            dominated[j]=np.any(np.all(f<=row,axis=1)&np.any(f<row,axis=1))
        front=pending[~dominated];ranks[front]=level;pending=pending[dominated];level+=1
    return ranks


def write_backend_rank_comparison(individuals,output_dir):
    """Compare the same candidates, same eight objectives, no re-selection.

    Per-objective ranks/Pareto fronts are the primary comparison. Utopia-distance
    ranks also depend on the same explicitly chosen normalization/preference.
    """
    pairs=[]
    for ind in individuals:
        pair=ind.metrics.get("postOnsetBackendComparison")
        if pair and pair.get("bothBackendsCompleted"):
            pairs.append((ind,pair))
    output=Path(output_dir)
    if not pairs:
        _json(output/"backend_rank_comparison.json",{"status":"no_complete_pairs","pair_count":0})
        return
    matrices=[]
    for name in BACKENDS:
        rows=[]
        for ind,pair in pairs:
            m=pair["backends"][name]
            pre=ind.metrics["preflame_objective_vector"]
            rows.append([*pre,m["finalUnreactedAreaFraction"],m["establishedTimeAfterOnset_s"],
                         -m["meanEffectiveRegressionVelocity_m_per_s"],m["reactionFrontNonuniformity"]])
        matrices.append(np.asarray(rows,dtype=float))
    joined=np.vstack(matrices); lo=joined.min(axis=0);span=joined.max(axis=0)-lo;span=np.where(span>0,span,1.)
    scores=[np.linalg.norm((f-lo)/span,axis=1) for f in matrices]
    ranks=[rankdata(v,method="average") for v in scores]
    pareto=[_nondominated_ranks(f) for f in matrices]
    # One pair or all ties cannot support a rank-correlation statistic.
    rho=None
    if len(pairs)>1 and all(np.ptp(v)>0 for v in ranks):
        rho=float(spearmanr(*ranks).statistic)
    rows=[]
    for i,(ind,pair) in enumerate(pairs):
        rows.append({"geometry_id":ind.geometry_id,"condensed_utopia_rank":float(ranks[0][i]),
                     "reactive_utopia_rank":float(ranks[1][i]),"condensed_pareto_front":int(pareto[0][i]),
                     "reactive_pareto_front":int(pareto[1][i]),"rank_changed":bool(ranks[0][i]!=ranks[1][i])})
    with (output/"backend_rank_comparison.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    per_objective=[]
    for col in range(8):
        aa=rankdata(matrices[0][:,col]);bb=rankdata(matrices[1][:,col])
        corr=float(spearmanr(aa,bb).statistic) if len(aa)>1 and np.ptp(aa)>0 and np.ptp(bb)>0 else None
        per_objective.append({"objective_index":col,"spearman":corr,
                              "condensed_ranks":aa.tolist(),"reactive_ranks":bb.tolist()})
    _json(output/"backend_rank_comparison.json",{"status":"complete","pair_count":len(pairs),
            "spearman_utopia_rank":rho,"normalization":"joint_range_of_both_backends_same_8_objectives",
            "uncalibrated_model_comparison_not_experimental_validation":True,
            "baseline_staggered_excluded_from_optimization_ranks":True,
            "rows":rows,"per_objective":per_objective})
