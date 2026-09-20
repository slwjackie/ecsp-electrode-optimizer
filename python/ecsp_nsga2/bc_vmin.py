"""Candidate-parallel finite-horizon voltage bracketing.

Only the *scheduler* is Python. Each callback integrates independent trials in
compiled FP64, with one voltage per candidate and a fresh initial state. This is
not a voltage ramp and never reuses a hot state from another trial.
"""
from __future__ import annotations
from typing import Callable, Mapping, Any, Sequence
import math
import numbers


def validate_voltage_search_contract(
    *,
    low_voltage: float,
    high_voltage: float,
    tolerance: float,
    max_iterations: int,
    invalid_penalty: float,
    censor_penalty: float,
) -> None:
    """Fail before a malformed search can publish a favourable failure score."""
    scalars = {
        "low_voltage": low_voltage,
        "high_voltage": high_voltage,
        "tolerance": tolerance,
        "invalid_penalty": invalid_penalty,
        "censor_penalty": censor_penalty,
    }
    if any(not math.isfinite(float(value)) for value in scalars.values()):
        raise ValueError("Voltage-search bounds, tolerance and penalties must be finite")
    if not 0.0 <= float(low_voltage) < float(high_voltage):
        raise ValueError("Voltage-search bounds must satisfy 0 <= low < high")
    if not float(tolerance) > 0.0:
        raise ValueError("Voltage-search tolerance must be positive")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, numbers.Integral)
        or int(max_iterations) < 1
    ):
        raise ValueError("Voltage-search maximum iterations must be an integer >= 1")
    if float(invalid_penalty) < 0.0 or float(censor_penalty) < 0.0:
        raise ValueError("Voltage-search invalid/censor penalties must be non-negative")
    if not math.isfinite(float(high_voltage) + float(invalid_penalty)) or not math.isfinite(
        float(high_voltage) + float(censor_penalty)
    ):
        raise ValueError("Voltage-search penalised objectives must remain finite")
    initial_width = float(high_voltage) - float(low_voltage)
    representable_resolution = max(
        math.ulp(float(low_voltage)), math.ulp(float(high_voltage))
    )
    if float(tolerance) < representable_resolution:
        raise ValueError(
            "Voltage-search tolerance is below the FP64 representable spacing "
            f"of the configured bracket; tolerance>={representable_resolution!r} is required"
        )
    if initial_width <= float(tolerance):
        required_iterations = 0
    else:
        # Subtract logarithms rather than forming width/tolerance, which can
        # overflow even when both configured operands are finite.  Correct
        # any boundary rounding with exact power-of-two scaling.
        required_iterations = max(
            0,
            int(
                math.ceil(
                    math.log2(initial_width) - math.log2(float(tolerance))
                )
            ),
        )
        while math.ldexp(initial_width, -required_iterations) > float(tolerance):
            required_iterations += 1
        while (
            required_iterations > 0
            and math.ldexp(initial_width, -(required_iterations - 1))
            <= float(tolerance)
        ):
            required_iterations -= 1
    if int(max_iterations) < required_iterations:
        raise ValueError(
            "Voltage-search maximum iterations cannot achieve the requested "
            f"tolerance; required>={required_iterations}"
        )


def _monotonicity_violation(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Return evidence when a tested lower voltage ignites but a higher one does not."""
    outcomes_by_voltage: dict[float, set[bool]] = {}
    for trial in trials:
        if not bool(trial.get("numerically_valid", False)):
            continue
        voltage = float(trial["voltage_V"])
        if math.isfinite(voltage):
            outcomes_by_voltage.setdefault(voltage, set()).add(
                bool(trial.get("ignited", False))
            )

    # Repeating a trial at exactly the same voltage is an especially strong
    # consistency check.  Detect it independently of callback/chronology order.
    for voltage, outcomes in sorted(outcomes_by_voltage.items()):
        if len(outcomes) > 1:
            return {
                "lowerIgnitingVoltage_V": voltage,
                "higherNonIgnitingVoltage_V": voltage,
                "sameVoltageContradiction": True,
            }

    igniting: list[float] = []
    for voltage, outcomes in sorted(outcomes_by_voltage.items()):
        ignited = True in outcomes
        if ignited:
            igniting.append(voltage)
        elif igniting:
            return {
                "lowerIgnitingVoltage_V": min(igniting),
                "higherNonIgnitingVoltage_V": voltage,
            }
    return None


def batched_voltage_search(
    reference_rows: Sequence[dict[str, Any]],
    run_trials: Callable[[list[int], list[float], str], list[dict[str, Any]]],
    valid: Callable[[Mapping[str, Any]], tuple[bool, str]],
    *, enabled: bool, low_voltage: float, high_voltage: float, tolerance: float,
    max_iterations: int, invalid_penalty: float, censor_penalty: float,
    verify_final: bool = False,
) -> None:
    """Augment rows in-place, preserving reference-run objectives and filenames.

    The bracket assumes monotonic onset feasibility over the requested voltage
    range. It reports a conservative tested upper bound, not an analytic global
    minimum. Numerical failures are invalid, NEVER physical non-ignition.
    """
    validate_voltage_search_contract(
        low_voltage=low_voltage,
        high_voltage=high_voltage,
        tolerance=tolerance,
        max_iterations=max_iterations,
        invalid_penalty=invalid_penalty,
        censor_penalty=censor_penalty,
    )
    states=[]
    for row in reference_rows:
        okay,why=valid(row)
        state=dict(low=low_voltage, high=high_voltage, iterations=0, active=False,
                   status='disabled_search_invalid', valid=False,
                   left=False, right=False, trials=[])
        if enabled:
            state['trials'].append(dict(voltage_V=high_voltage,role='reference_upper_bound',
                ignited=bool(row.get('ignitionSucceeded',False)),numerically_valid=okay,validity_reason=why,
                validity_window='full_horizon'))
            if not okay:state.update(status='invalid_reference_trial',valid=False)
            elif not row.get('ignitionSucceeded',False):state.update(
                status='right_censored_no_ignition_at_upper_bound',valid=True,
                right=True,low=high_voltage)
            else:state.update(active=True,valid=True,status='bracketed_converged')
        states.append(state)

    def apply(indices:list[int], volts:list[float], role:str):
        results=run_trials(indices,volts,role)
        if len(results)!=len(indices):raise RuntimeError('Trial callback lost candidate alignment')
        for idx,v,result in zip(indices,volts,results):
            st=states[idx];okay,why=valid(result);ignited=bool(result.get('ignitionSucceeded',False))
            native=result.get('nativeExecution',{})
            record=dict(voltage_V=float(v),role=role,ignited=ignited,
                        ignition_delay_s=result.get('ignitionDelay_s'),numerically_valid=okay,validity_reason=why,
                        validity_window=native.get('validity_window','full_horizon'),
                        steps_executed=native.get('stepsExecutedPerCandidate'),
                        early_stopped=native.get('earlyStopped',False))
            st['trials'].append(record)
            if role=='bisection':st['iterations']+=1
            if not okay:
                st.update(active=False,valid=False,status=f'invalid_{role}_trial');continue
            if role=='final_upper_full_horizon_verification':
                if not ignited:st.update(active=False,valid=False,status='inconsistent_final_upper_trial')
                continue
            if role=='lower_bound' and ignited:
                st.update(high=v,active=False,left=True,status='left_censored_lower_bound_ignites')
            elif ignited:st['high']=v
            else:st['low']=v

    indices=[i for i,s in enumerate(states) if s['active']]
    if indices:apply(indices,[low_voltage]*len(indices),'lower_bound')
    while True:
        indices=[i for i,s in enumerate(states) if s['active'] and
                 s['high']-s['low']>tolerance and s['iterations']<max_iterations]
        if not indices:break
        midpoints = [
            states[i]['low']
            + (states[i]['high'] - states[i]['low']) * 0.5
            for i in indices
        ]
        if any(
            midpoint <= states[index]['low']
            or midpoint >= states[index]['high']
            for index, midpoint in zip(indices, midpoints)
        ):
            raise RuntimeError(
                "Voltage-search midpoint failed to shrink a validated FP64 bracket"
            )
        apply(indices, midpoints, 'bisection')
    if enabled and verify_final:
        indices=[i for i,s in enumerate(states) if s['valid'] and not s['right'] and s['high']!=high_voltage]
        if indices:apply(indices,[states[i]['high'] for i in indices],'final_upper_full_horizon_verification')
    for st in states:
        evidence = _monotonicity_violation(st["trials"])
        st["monotonicity_evidence"] = evidence
        valid_voltages = {
            float(trial["voltage_V"])
            for trial in st["trials"]
            if bool(trial.get("numerically_valid", False))
            and math.isfinite(float(trial["voltage_V"]))
        }
        st["monotonicity_checked"] = bool(
            enabled and (evidence is not None or len(valid_voltages) >= 2)
        )
        if evidence is not None:
            st.update(
                active=False,
                valid=False,
                status="invalid_nonmonotonic_ignition_response",
            )
    for row,st in zip(reference_rows,states):
        width=0. if st['left'] else st['high']-st['low']
        if st['active'] and st['valid'] and width>tolerance:
            st.update(
                active=False,
                valid=False,
                status='invalid_iteration_budget_exhausted_before_tolerance',
            )
        physical=st['high'] if st['valid'] and not st['right'] else None
        objective=(high_voltage+invalid_penalty if not st['valid'] else
                   high_voltage+censor_penalty if st['right'] else st['high'])
        row.update(minimumIgnitionVoltageObjective_V=float(objective),minimumIgnitionVoltage_V=physical,
            minimumIgnitionVoltageSearchValid=st['valid'],minimumIgnitionVoltageSearchStatus=st['status'],
            minimumIgnitionVoltageLeftCensored=st['left'],minimumIgnitionVoltageRightCensored=st['right'],
            minimumIgnitionVoltageLowerNonIgnitingBound_V=None if st['left'] or not enabled else st['low'],
            minimumIgnitionVoltageUpperIgnitingBound_V=physical,
            minimumIgnitionVoltageBracketWidth_V=None if st['right'] or not st['valid'] else width,
            minimumIgnitionVoltageTargetTolerance_V=tolerance,
            minimumIgnitionVoltageBisectionIterations=st['iterations'],
            minimumIgnitionVoltageTrialCount=len(st['trials']),minimumIgnitionVoltageTrials=st['trials'],
            minimumIgnitionVoltageSearchAlgorithm='candidate_parallel_bisection_fresh_initial_states',
            minimumIgnitionVoltageMonotonicityAssumption=True,
            minimumIgnitionVoltageMonotonicityChecked=st['monotonicity_checked'],
            minimumIgnitionVoltageMonotonicityObserved=(
                st['monotonicity_evidence'] is None
                if st['monotonicity_checked']
                else None
            ),
            minimumIgnitionVoltageMonotonicityViolation=st['monotonicity_evidence'],
            minimumIgnitionVoltageFinalFullHorizonCheck=bool(enabled and verify_final and physical is not None),
            minimumIgnitionVoltageDefinition='tested conservative upper onset voltage within finite horizon; monotonic bracket assumption',
        )
