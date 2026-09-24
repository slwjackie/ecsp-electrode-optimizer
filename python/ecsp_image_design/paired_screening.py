"""Same-context candidate/staggered screening, NOT cross-design ranking.

No solver imports. Missing/censored observations remain missing; thermal gain
is never substituted for ignition time or minimum ignition voltage.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping
import math

SCHEMA = "ecsp-paired-screening-v1"
J_KEY = "peakCurrentCongestionToEvaluationTime"


@dataclass(frozen=True)
class ScreeningPolicy:
    # These are decision margins, NOT calibrated physical constants.
    time_gain: float = 0.0
    voltage_gain: float = 0.0
    thermal_gain: float = 0.0
    congestion_gain: float = 0.0
    area_rtol: float = 0.01
    denominator_delta_temperature_K: float = 1e-9
    denominator_congestion: float = 1e-12
    comparison_epsilon: float = 1e-12

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid screening policy: {key}")
        if any(getattr(self, k) >= 1 for k in ("time_gain", "voltage_gain", "congestion_gain")):
            raise ValueError("Lower-is-better gain margins must be less than one")
        if self.area_rtol > 0.01:
            raise ValueError("Paired contact-area tolerance must not exceed 1%")


def finite(value: Any) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def positive(value: Any) -> bool:
    return finite(value) and value > 0


def pair_context_errors(a: Mapping, b: Mapping, policy: ScreeningPolicy) -> list[str]:
    errors = []
    for key in ("domain_mm", "grid_size", "minimum_gap_mm", "minimum_width_mm",
                "reference_voltage_V", "evaluation_time_s", "initial_temperature_K"):
        av, bv = a.get(key), b.get(key)
        if not positive(av) or not positive(bv):
            errors.append("missing_or_invalid_context:" + key)
        elif not math.isclose(av, bv, rel_tol=1e-12, abs_tol=1e-12):
            errors.append("mismatched_context:" + key)
    if not a.get("physics_config_hash") or a.get("physics_config_hash") != b.get("physics_config_hash"):
        errors.append("missing_or_mismatched_physics_config_hash")
    for key in ("anode_area_mm2", "cathode_area_mm2"):
        av, bv = a.get(key), b.get(key)
        if not positive(av) or not positive(bv):
            errors.append("missing_or_invalid_area:" + key)
        elif abs(av - bv) / av > policy.area_rtol + 1e-12:
            errors.append("mismatched_contact_area:" + key)
    for label, ctx in (("candidate", a), ("baseline", b)):
        if ctx.get("area_basis") != "solver_mask":
            errors.append(label + ":unverified_solver_mask_area")
        if all(positive(ctx.get(k)) for k in ("domain_mm", "anode_area_mm2", "cathode_area_mm2")):
            fe = (ctx["anode_area_mm2"] + ctx["cathode_area_mm2"]) / ctx["domain_mm"] ** 2
            if not finite(ctx.get("electrode_area_fraction")) or not math.isclose(fe, ctx["electrode_area_fraction"], rel_tol=1e-10, abs_tol=1e-12) or not 0 < fe < 1:
                errors.append(label + ":inconsistent_coverage")
    return errors


def numerical_errors(row: Mapping, label: str) -> list[str]:
    # Ignition and Vmin censoring are intentionally NOT numerical failures.
    errors = [label + ":" + key for key in
              ("external_numerical_valid", "converged", "allElectricalLinearSolvesConverged",
               "allNonlinearRobinSolvesConverged") if row.get(key) is not True]
    if row.get("physicsRejected", False):
        errors.append(label + ":physics_rejected")
    return errors


def voltage_state(row: Mapping) -> str:
    if row.get("minimumIgnitionVoltageSearchValid") is not True:
        return "invalid_or_missing_search"
    left = row.get("minimumIgnitionVoltageLeftCensored")
    right = row.get("minimumIgnitionVoltageRightCensored")
    if not isinstance(left, bool) or not isinstance(right, bool) or (left and right):
        return "invalid_censor_flags"
    if left:
        return "left_censored"
    if right:
        return "right_censored"
    return "bracketed" if positive(row.get("minimumIgnitionVoltage_V")) else "missing_voltage"


def voltage_interval(row: Mapping) -> tuple[float, float] | None:
    hi = row.get("minimumIgnitionVoltage_V")
    width = row.get("minimumIgnitionVoltageBracketWidth_V")
    if positive(hi) and finite(width) and 0 <= width < hi:
        return hi - width, hi
    return None


def screen_pair(record: Mapping[str, Any], policy: ScreeningPolicy | None = None) -> dict:
    """Return independent labels; never a score, rank, or fabricated ratio."""
    p = policy or ScreeningPolicy()
    c, b = record.get("candidate_raw", {}), record.get("baseline_raw", {})
    cc, bc = record.get("context", {}), record.get("baseline_context", {})
    out = dict(schema=SCHEMA, source_id=record.get("source_id"), screening_valid=False,
               comparison_state="unavailable", R_t=None, R_V=None, R_J=None, R_T=None,
               ratio_reasons={}, ignition_screen="unavailable", thermal_screen="unavailable",
               congestion_screen="unavailable", labels=[], reasons=[], policy=asdict(p),
               context=dict(cc), baseline_context=dict(bc),
               voltage_status={"candidate": voltage_state(c), "baseline": voltage_state(b)},
               voltage_comparison="unavailable", voltage_ratio_interval=None,
               evidence_level="nominal_model_screening_not_experimental_validation",
               cross_design_ranking=False, revalidation_domain_mm=25.0,
               metric_scope="R_J uses the existing peak-to-evaluation-time metric; onset may freeze a lane",
               warnings=["A larger R_T does not establish a smaller Vmin.",
                         "Ratios at different domains do not establish a universal topology order."])
    mismatch = pair_context_errors(cc, bc, p)
    if mismatch:
        out.update(comparison_state="invalid_pair", labels=["invalid_pair"], reasons=mismatch)
        return out
    errors = numerical_errors(c, "candidate") + numerical_errors(b, "baseline")
    if errors:
        out.update(comparison_state="numerically_invalid", labels=["numerically_invalid"], reasons=errors)
        return out
    ci, bi = c.get("ignitionSucceeded"), b.get("ignitionSucceeded")
    for label, row, ignited in (("candidate", c, ci), ("baseline", b, bi)):
        t = row.get("ignitionDelay_s")
        if not isinstance(ignited, bool) or (ignited and (not positive(t) or t > cc["evaluation_time_s"] + 1e-12)) or (ignited is False and finite(t)):
            errors.append(label + ":inconsistent_ignition_observation")
        if ignited is False and row.get("nativeExecution", {}).get("completedFullHorizon") is False:
            errors.append(label + ":incomplete_nonigniting_reference")
    if errors:
        out.update(comparison_state="incomplete_observation", labels=["incomplete_observation"], reasons=errors)
        return out
    out["screening_valid"] = True
    out["comparison_state"] = ("both_ignited" if ci and bi else "candidate_only_ignited" if ci
                               else "baseline_only_ignited" if bi else "neither_ignited")
    j_c, j_b = c.get(J_KEY), b.get(J_KEY)
    if finite(j_c) and j_c >= 0 and finite(j_b) and j_b > p.denominator_congestion:
        out["R_J"] = j_c / j_b
        out["congestion_screen"] = ("improved" if out["R_J"] < 1 - p.congestion_gain - p.comparison_epsilon
                                    else "worse" if out["R_J"] > 1 + p.congestion_gain + p.comparison_epsilon
                                    else "no_resolved_difference")
        if out["congestion_screen"] == "improved":
            out["labels"].append("congestion_superior_candidates")
    else:
        out["ratio_reasons"]["R_J"] = "missing_nonfinite_or_uninformative_baseline_congestion"
    if out["voltage_status"] == {"candidate": "bracketed", "baseline": "bracketed"}:
        out["R_V"] = c["minimumIgnitionVoltage_V"] / b["minimumIgnitionVoltage_V"]
        ic, ib = voltage_interval(c), voltage_interval(b)
        if ic is not None and ib is not None and ib[0] > 0:
            out["voltage_ratio_interval"] = [ic[0] / ib[1], ic[1] / ib[0]]
            out["voltage_comparison"] = ("lower_resolved" if ic[1] < (1 - p.voltage_gain) * ib[0] - p.comparison_epsilon
                                         else "higher_resolved" if ic[0] > (1 + p.voltage_gain) * ib[1] + p.comparison_epsilon
                                         else "overlapping_brackets")
        else:
            out["voltage_comparison"] = "point_ratio_only_brackets_missing"
    else:
        out["ratio_reasons"]["R_V"] = "censored_invalid_or_missing_search; no penalty substitution"
    if ci and bi:
        out["R_t"] = c["ignitionDelay_s"] / b["ignitionDelay_s"]
        faster = out["R_t"] < 1 - p.time_gain - p.comparison_epsilon
        lower = out["voltage_comparison"] == "lower_resolved"
        out["ignition_screen"] = "promising" if faster and lower else "tradeoff_or_unresolved"
        if faster and lower:
            out["labels"].append("ignition_superior_candidates")
        if out["R_t"] > 1 + p.time_gain + p.comparison_epsilon and out["voltage_comparison"] == "higher_resolved" and out["congestion_screen"] == "worse":
            out["labels"].append("baseline_dominated")
    elif ci:
        out["ignition_screen"] = "strong_promising_at_reference_condition"
        out["labels"].append("ignition_superior_candidates")
    elif bi:
        out["ignition_screen"] = "inferior_at_reference_condition"
        out["labels"].append("ignition_reject_at_reference_condition")
    else:
        out["ignition_screen"] = "not_demonstrated"
        t0 = cc["initial_temperature_K"]
        tc, tb = c.get("peakMaximumTemperature_K"), b.get("peakMaximumTemperature_K")
        if finite(tc) and finite(tb) and tc >= t0 and tb - t0 > p.denominator_delta_temperature_K:
            out["R_T"] = (tc - t0) / (tb - t0)
            out["thermal_screen"] = ("promising" if out["R_T"] > 1 + p.thermal_gain + p.comparison_epsilon
                                      else "inferior" if out["R_T"] < 1 - p.thermal_gain - p.comparison_epsilon
                                      else "no_resolved_difference")
            if out["thermal_screen"] == "promising":
                out["labels"].append("thermal_promising_candidates")
            elif out["thermal_screen"] == "inferior" and out["congestion_screen"] == "worse":
                out["labels"].append("fixed_condition_baseline_dominated")
        else:
            out["ratio_reasons"]["R_T"] = "missing_nonfinite_or_negligible_baseline_temperature_rise"
    if out["R_t"] is None:
        out["ratio_reasons"]["R_t"] = "both_reference_trials_must_ignite"
    if ci or bi:
        out["thermal_screen"] = "not_applicable_after_onset"
        out["ratio_reasons"]["R_T"] = "thermal screening is restricted to two nonigniting full-horizon trials"
    if not out["labels"]:
        out["labels"].append("inconclusive_or_no_resolved_gain")
    return out
