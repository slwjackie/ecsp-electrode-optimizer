from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp
import yaml

from ecsp_reactive.condensed.chemistry import (
    A1, A2, RHO, _CoordinateCertificationFailure,
)
from ecsp_reactive.condensed.handoff import BCReactiveHandoffAdapter
from ecsp_reactive.condensed.validation_cases import synthetic_condensed_case


ROOT = Path(__file__).resolve().parents[2]
COORDINATE_CONTROLS = dict(
    relative_tolerance=1.0e-7,
    absolute_tolerance=1.0e-10,
    temperature_tolerance_K=1.0e-4,
    maximum_depletion_iterations=40,
    maximum_local_refinements=16,
    maximum_reaction_coordinate_steps=64,
)


def _adapted(*, rates=(4.0, 1.5), heats=(0.0, 0.0),
             alpha=(0.1, 0.1), temperature=600.0, change=None):
    handoff, propagation, bc, reactive = synthetic_condensed_case(
        shape=(5, 5), rates=rates, heats=heats, alpha=alpha
    )
    handoff["temperatureAtOnset_K"][:] = temperature
    if change is not None:
        change(handoff, propagation, bc, reactive)
    return BCReactiveHandoffAdapter(propagation, bc, reactive).adapt(handoff)


def _coordinate(adapted, alpha0, temperature, duration, capacity=10.0,
                **control_overrides):
    controls = {**COORDINATE_CONTROLS, **control_overrides}
    sensible0 = float(adapted.thermo.heat.sensible_energy(temperature))
    return adapted.chemistry._solve_coupled_reaction_coordinate_cell(
        np.asarray(alpha0, dtype=np.float64), sensible0, temperature,
        duration, capacity, adapted.thermo, **controls
    )


def _radau_with_exact_channel_completion(
        adapted, alpha0, temperature0, duration):
    """Independent time-domain oracle with terminal completion events."""
    chemistry = adapted.chemistry
    alpha_reference = np.asarray(alpha0, dtype=np.float64)
    sensible_reference = float(
        adapted.thermo.heat.sensible_energy(temperature0)
    )
    alpha = alpha_reference.copy()
    elapsed = 0.0
    while elapsed < duration-64.0*np.finfo(float).eps*max(duration, 1.0):
        active = alpha < 1.0-1.0e-12
        if not np.any(active):
            break

        def rhs(_time, state):
            evaluation = np.minimum(state, np.nextafter(1.0, 0.0))
            sensible = sensible_reference+float(
                (evaluation-alpha_reference) @ chemistry.Q
            )
            temperature = float(adapted.thermo.heat.temperature(sensible))
            rates = chemistry._rates_from_alpha(
                evaluation[None, :], np.asarray([temperature])
            )[0]
            return np.where(active, rates, 0.0)

        active_indices = np.flatnonzero(active)
        events = []
        for channel_index in active_indices:
            def completion(_time, state, channel_index=int(channel_index)):
                return 1.0-state[channel_index]

            completion.terminal = True
            completion.direction = -1
            events.append(completion)
        solution = solve_ivp(
            rhs, (elapsed, duration), alpha, method="Radau",
            rtol=2.0e-11, atol=2.0e-13, events=events,
            max_step=duration/20.0,
        )
        assert solution.success
        next_elapsed = float(solution.t[-1])
        alpha = solution.y[:, -1]
        if next_elapsed >= duration-1.0e-14:
            elapsed = duration
            break
        completed = [
            int(active_indices[index])
            for index, values in enumerate(solution.t_events)
            if values.size
        ]
        assert completed and next_elapsed > elapsed
        alpha[completed] = 1.0
        elapsed = next_elapsed
    sensible = sensible_reference+float(
        (alpha-alpha_reference) @ chemistry.Q
    )
    return alpha, float(adapted.thermo.heat.temperature(sensible))


def _switching_case(*, swapped=False):
    decreasing = {
        "alpha_grid": [0.0, 0.1, 1.0],
        "activation_energy_J_per_mol": [0.0, 0.0, 0.0],
        "ln_Af_per_s": [math.log(4.0), math.log(4.0), math.log(0.5)],
        "heat_release_J_per_kg": 0.0,
    }
    constant = {
        "alpha_grid": [0.0, 1.0],
        "activation_energy_J_per_mol": [0.0, 0.0],
        "ln_Af_per_s": [math.log(1.5), math.log(1.5)],
        "heat_release_J_per_kg": 0.0,
    }

    def change(_handoff, _propagation, bc, _reactive):
        specifications = [decreasing, constant]
        if swapped:
            specifications.reverse()
            bc["kinetics"]["mass_conversion_weights"] = [1.0/3.0, 2.0/3.0]
        for channel, specification in zip(
                bc["kinetics"]["channels"], specifications):
            channel.update(copy.deepcopy(specification))

    return _adapted(change=change)


def _shared_inventory_stiff_case(*, swapped=False):
    channels = [
        {
            "alpha_grid": [0.0, 0.3, 0.65, 1.0],
            "activation_energy_J_per_mol": [
                60752.57709167853, 64042.73245563079,
                93550.51531785612, 112096.54780396441,
            ],
            "ln_Af_per_s": [
                13.619872636989093, 13.354648693058708,
                20.85971808792269, 23.22496209551369,
            ],
            "heat_release_J_per_kg": 130302.24250943401,
        },
        {
            "alpha_grid": [0.0, 0.3, 0.65, 1.0],
            "activation_energy_J_per_mol": [
                113170.64105899706, 91491.52460466264,
                42742.0932168346, 94640.19269046221,
            ],
            "ln_Af_per_s": [
                20.876023514938478, 19.862414311402752,
                8.799155310372413, 16.435364080705273,
            ],
            "heat_release_J_per_kg": 289086.1062147321,
        },
    ]
    weights = [2.0/3.0, 1.0/3.0]
    base_alpha = [0.2880679176765724, 0.18780234884152658]
    alpha = list(base_alpha)
    if swapped:
        channels.reverse()
        weights.reverse()
        alpha.reverse()

    def change(_handoff, _propagation, bc, _reactive):
        bc["thermal"]["heat_capacity"] = {
            "mode": "constant", "value": 2200.0,
        }
        bc["kinetics"]["mass_conversion_weights"] = weights
        if swapped:
            first = np.array(_handoff["alphaChannel1AtOnset"], copy=True)
            _handoff["alphaChannel1AtOnset"] = np.array(
                _handoff["alphaChannel2AtOnset"], copy=True
            )
            _handoff["alphaChannel2AtOnset"] = first
        _handoff["globalProgressAtOnset"] = (
            weights[0]*np.asarray(_handoff["alphaChannel1AtOnset"])
            +weights[1]*np.asarray(_handoff["alphaChannel2AtOnset"])
        )
        for channel, specification in zip(
                bc["kinetics"]["channels"], channels):
            channel.update(copy.deepcopy(specification))

    return (_adapted(
        rates=(1.0, 1.0), heats=(1.0, 1.0), alpha=tuple(base_alpha),
        temperature=654.3292090178273, change=change,
    ), np.asarray(alpha, dtype=np.float64))


def _radau_shared_inventory(adapted, alpha0, temperature0, duration,
                            capacity):
    chemistry = adapted.chemistry
    alpha0 = np.asarray(alpha0, dtype=np.float64)
    sensible0 = float(adapted.thermo.heat.sensible_energy(temperature0))

    def rhs(_time, alpha):
        sensible = sensible0+float((alpha-alpha0) @ chemistry.Q)
        temperature = float(adapted.thermo.heat.temperature(sensible))
        return chemistry._rates_from_alpha(
            alpha[None, :], np.asarray([temperature])
        )[0]

    def exhausted(_time, alpha):
        return capacity-float((alpha-alpha0) @ chemistry.weights)

    exhausted.terminal = True
    exhausted.direction = -1
    solution = solve_ivp(
        rhs, (0.0, duration), alpha0, method="Radau",
        rtol=2.0e-11, atol=2.0e-13, events=exhausted,
        max_step=duration/100.0,
    )
    assert solution.success
    alpha = solution.y[:, -1]
    sensible = sensible0+float((alpha-alpha0) @ chemistry.Q)
    event_time = (
        float(solution.t_events[0][0]) if solution.t_events[0].size
        else math.nan
    )
    return (alpha, float(adapted.thermo.heat.temperature(sensible)),
            event_time)


def _record_coordinate_drivers(adapted, solve):
    drivers = []
    original = adapted.chemistry._coupled_coordinate_dp54_step

    def recording_step(*args, **kwargs):
        drivers.append(int(args[2]))
        return original(*args, **kwargs)

    adapted.chemistry._coupled_coordinate_dp54_step = recording_step
    try:
        result = solve()
    finally:
        adapted.chemistry._coupled_coordinate_dp54_step = original
    return result, drivers


def test_dominance_switch_and_full_channel_permutation_covariance():
    base = _switching_case(swapped=False)
    swapped = _switching_case(swapped=True)
    alpha0 = np.asarray([0.1, 0.1])
    duration = 0.7
    base_result, base_drivers = _record_coordinate_drivers(
        base, lambda: _coordinate(base, alpha0, 600.0, duration)
    )
    swapped_result, swapped_drivers = _record_coordinate_drivers(
        swapped, lambda: _coordinate(swapped, alpha0, 600.0, duration)
    )
    reference_alpha, reference_temperature = (
        _radau_with_exact_channel_completion(base, alpha0, 600.0, duration)
    )

    assert base_drivers[0] == 0
    assert swapped_drivers[0] == 1
    assert base_result["driver_switches"] == 1
    assert swapped_result["driver_switches"] == 1
    np.testing.assert_allclose(
        base_result["alpha"], reference_alpha, rtol=0.0, atol=5.0e-8
    )
    assert base_result["temperature"] == pytest.approx(
        reference_temperature, abs=1.0e-10
    )
    np.testing.assert_allclose(
        swapped_result["alpha"][::-1], base_result["alpha"],
        rtol=0.0, atol=3.0e-12,
    )
    assert swapped_result["sensible"] == pytest.approx(
        base_result["sensible"], abs=1.0e-10
    )
    assert swapped_result["capacity_left"] == pytest.approx(
        base_result["capacity_left"], abs=2.0e-15
    )
    assert base_result["maximum_normalized_residual"] <= 1.0
    assert swapped_result["maximum_normalized_residual"] <= 1.0


def test_identical_equal_rate_channels_use_symmetric_coordinate_path():
    adapted = _adapted(rates=(2.0, 2.0))
    result = _coordinate(adapted, (0.1, 0.1), 600.0, 0.2)
    np.testing.assert_allclose(
        result["alpha"], [0.5, 0.5], rtol=0.0, atol=5.0e-8
    )
    assert result["alpha"][0] == pytest.approx(
        result["alpha"][1], abs=2.0e-15
    )


def test_nonidentical_equal_rate_tie_declines_without_index_bias():
    adapted = _adapted(rates=(2.0, 2.0), heats=(0.0, 1.0))
    with pytest.raises(
            _CoordinateCertificationFailure,
            match="ambiguous equal-rate dominant coordinate"):
        _coordinate(adapted, (0.1, 0.1), 600.0, 1.0)


def test_driver_knots_and_completion_handoff_to_one_active_are_exact():
    def change(_handoff, _propagation, bc, _reactive):
        bc["kinetics"]["channels"][0].update(
            alpha_grid=[0.0, 0.3, 0.6, 1.0],
            activation_energy_J_per_mol=[0.0]*4,
            ln_Af_per_s=[math.log(4.0)]*4,
        )

    adapted = _adapted(rates=(4.0, 1.0), change=change)
    result = _coordinate(adapted, (0.1, 0.1), 600.0, 0.3)
    np.testing.assert_allclose(
        result["alpha"], [1.0, 0.4], rtol=0.0, atol=3.0e-12
    )
    assert result["accepted_steps"] == 3
    assert result["one_active"] is not None
    assert result["one_active"]["reaches_bound"] is False
    assert result["maximum_normalized_residual"] <= 1.0


def test_driver_knot_overround_snap_refunds_capacity_exactly():
    start = 0.39409655507937175

    def change(_handoff, _propagation, bc, _reactive):
        bc["kinetics"]["channels"][0].update(
            alpha_grid=[0.0, 0.9, 1.0],
            activation_energy_J_per_mol=[0.0]*3,
            ln_Af_per_s=[math.log(4.0)]*3,
        )

    adapted = _adapted(
        rates=(4.0, 1.0), alpha=(start, 0.1), change=change
    )
    alpha0 = np.asarray([start, 0.1])
    capacity = 0.48
    result = _coordinate(
        adapted, alpha0, 600.0, 0.2, capacity=capacity,
        relative_tolerance=1.0e-12,
        absolute_tolerance=1.0e-15,
        temperature_tolerance_K=1.0e-9,
        maximum_depletion_iterations=60,
    )

    # start + (0.9-start) rounds to 0.9000000000000001.  Snapping that
    # endpoint down to the exact table knot must refund the signed one-ULP
    # progress correction instead of retaining it as consumed inventory.
    expected_capacity = capacity-float(
        (result["alpha"]-alpha0) @ adapted.chemistry.weights
    )
    np.testing.assert_allclose(
        result["alpha"], [1.0, 0.3], rtol=0.0, atol=3.0e-12
    )
    assert result["capacity_left"] == expected_capacity


@pytest.mark.parametrize("case", [
    "non_driver_completion", "non_driver_kinetic_knot", "inventory",
])
def test_non_driver_knots_completion_and_inventory_are_explicit_events(case):
    alpha0 = (0.1, 0.1)
    capacity = 10.0

    def change(_handoff, _propagation, bc, _reactive):
        if case == "non_driver_kinetic_knot":
            bc["kinetics"]["channels"][1].update(
                alpha_grid=[0.0, 0.2, 1.0],
                activation_energy_J_per_mol=[0.0]*3,
                ln_Af_per_s=[math.log(3.0)]*3,
            )
    if case == "non_driver_completion":
        alpha0 = (0.1, 0.98)
    if case == "inventory":
        capacity = 0.05
    adapted = _adapted(
        rates=(4.0, 3.0), alpha=alpha0, temperature=300.0,
        change=change,
    )
    result = _coordinate(
        adapted, alpha0, 300.0, 1.0, capacity=capacity,
        maximum_local_refinements=10,
    )
    if case == "inventory":
        event_time = capacity/float(
            np.asarray([4.0, 3.0]) @ adapted.chemistry.weights
        )
        expected = np.asarray(alpha0)+event_time*np.asarray([4.0, 3.0])
        np.testing.assert_allclose(result["alpha"], expected, atol=2.0e-7,
                                   rtol=0.0)
        assert result["depleted"]
        assert result["depletion_time"] == pytest.approx(
            event_time, abs=5.0e-8
        )
    else:
        reference_alpha, reference_temperature = (
            _radau_with_exact_channel_completion(
                adapted, np.asarray(alpha0), 300.0, 1.0
            )
        )
        np.testing.assert_allclose(
            result["alpha"], reference_alpha, atol=3.0e-7, rtol=0.0
        )
        assert result["temperature"] == pytest.approx(
            reference_temperature, abs=1.0e-8
        )
    assert result["maximum_normalized_residual"] <= 1.0


@pytest.mark.parametrize(
    ("case", "expected_message"),
    [
        ("exponent_clamp", "exponent clamp kink"),
        ("caloric_knot", "caloric knot"),
    ],
)
def test_nonmonotone_nonsmooth_events_retain_midpoint_fallback(
        case, expected_message):
    def change(_handoff, _propagation, bc, _reactive):
        if case == "exponent_clamp":
            bc["kinetics"]["channels"][0].update(
                activation_energy_J_per_mol=[0.0, 0.0],
                ln_Af_per_s=[50.0, 70.0],
            )
            bc["kinetics"]["channels"][1].update(
                activation_energy_J_per_mol=[0.0, 0.0],
                ln_Af_per_s=[49.0, 49.0],
            )
        else:
            bc["thermal"]["heat_capacity"] = {
                "mode": "table",
                "temperature_K": [250.0, 301.0, 1000.0],
                "values": [1000.0, 1000.0, 1000.0],
            }
            bc["kinetics"]["channels"][0][
                "heat_release_J_per_kg"
            ] = 1.0e6

    adapted = _adapted(
        rates=(4.0, 3.0), alpha=(0.1, 0.1), temperature=300.0,
        change=change,
    )
    with pytest.raises(_CoordinateCertificationFailure, match=expected_message):
        _coordinate(adapted, (0.1, 0.1), 300.0, 1.0)


def test_declined_coupled_coordinate_many_panel_midpoint_converges_to_radau(
        monkeypatch):
    gas_constant = 8.31446261815324

    def change(_handoff, _propagation, bc, _reactive):
        specifications = (
            (
                60000.0,
                math.log(2.5)+60000.0/(gas_constant*800.0),
                2.0e5,
            ),
            (
                80000.0,
                math.log(0.9)+80000.0/(gas_constant*800.0),
                3.0e5,
            ),
        )
        for channel, (activation, log_prefactor, heat) in zip(
                bc["kinetics"]["channels"], specifications):
            channel.update(
                activation_energy_J_per_mol=[activation, activation],
                ln_Af_per_s=[log_prefactor, log_prefactor],
                heat_release_J_per_kg=heat,
            )

    adapted = _adapted(
        rates=(1.0, 1.0), heats=(1.0, 1.0), alpha=(0.15, 0.1),
        temperature=800.0, change=change,
    )

    def decline_coordinate(*_args, **_kwargs):
        raise _CoordinateCertificationFailure("forced certification decline")

    monkeypatch.setattr(
        adapted.chemistry, "_solve_coupled_reaction_coordinate_cell",
        decline_coordinate,
    )
    duration = 0.02
    reference_alpha, reference_temperature = (
        _radau_with_exact_channel_completion(
            adapted, np.asarray([0.15, 0.1]), 800.0, duration
        )
    )

    def midpoint_solution(relative_tolerance, absolute_tolerance,
                          temperature_tolerance_K):
        state, diagnostics = adapted.chemistry.advance_local(
            adapted.U, adapted.thermo, duration,
            concentration_floor=0.0,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            temperature_tolerance_K=temperature_tolerance_K,
            maximum_corrector_iterations=8,
            maximum_depletion_iterations=40,
            maximum_local_refinements=16,
            maximum_reaction_coordinate_steps=64,
        )
        alpha = state[0, 0, A1:A2+1]/state[0, 0, RHO]
        temperature = float(adapted.thermo.primitive(state)[0, 0, 3])
        return alpha, temperature, diagnostics

    loose_alpha, loose_temperature, loose = midpoint_solution(
        1.0e-6, 1.0e-9, 1.0e-3
    )
    tight_alpha, tight_temperature, tight = midpoint_solution(
        1.0e-7, 1.0e-10, 1.0e-4
    )

    # Every failed embedded attempt reaches the accelerator, whose deliberate
    # certification decline must leave the chronological midpoint scheduler in
    # control.  The tight run exercises a genuinely many-panel fallback path.
    assert tight["coupled_reaction_coordinate_attempt_count"] > 0
    assert tight["coupled_reaction_coordinate_fallback_count"] == (
        tight["coupled_reaction_coordinate_attempt_count"]
    )
    assert tight["coupled_reaction_coordinate_cell_count"] == 0
    assert tight["maximum_accepted_panels_per_cell"] >= 8

    loose_alpha_error = float(np.max(np.abs(loose_alpha-reference_alpha)))
    tight_alpha_error = float(np.max(np.abs(tight_alpha-reference_alpha)))
    loose_temperature_error = abs(loose_temperature-reference_temperature)
    tight_temperature_error = abs(tight_temperature-reference_temperature)
    assert tight_alpha_error < 0.25*loose_alpha_error
    assert tight_temperature_error < 0.25*loose_temperature_error
    np.testing.assert_allclose(
        tight_alpha, reference_alpha, rtol=0.0, atol=2.0e-8
    )
    assert tight_temperature == pytest.approx(
        reference_temperature, abs=2.0e-6
    )


@pytest.mark.parametrize("swapped", [False, True])
def test_stiff_shared_inventory_event_matches_independent_radau_fast(swapped):
    adapted, alpha0 = _shared_inventory_stiff_case(swapped=swapped)
    duration = 0.2
    capacity = 0.1708788006988188
    result = _coordinate(
        adapted, alpha0, 654.3292090178273, duration,
        capacity=capacity, maximum_local_refinements=10,
        maximum_depletion_iterations=60,
    )
    reference_alpha, reference_temperature, reference_time = (
        _radau_shared_inventory(
            adapted, alpha0, 654.3292090178273, duration, capacity
        )
    )
    np.testing.assert_allclose(
        result["alpha"], reference_alpha, atol=8.0e-8, rtol=0.0
    )
    assert result["temperature"] == pytest.approx(
        reference_temperature, abs=5.0e-5
    )
    assert result["depletion_time"] == pytest.approx(
        reference_time, abs=5.0e-8
    )
    assert result["depleted"]
    assert 0.0 <= result["capacity_left"] <= (
        COORDINATE_CONTROLS["absolute_tolerance"]
        +COORDINATE_CONTROLS["relative_tolerance"]*capacity
    )
    assert result["accepted_steps"] <= 64
    assert result["rate_evaluations"] < 2000
    assert result["maximum_normalized_residual"] <= 1.0


def test_v009_hard_cell_matches_radau_and_keeps_exact_ledgers():
    production = yaml.safe_load(
        (ROOT / "config/nsga2_bc_reactive_m2cpu_200x3.yaml").read_text(
            encoding="utf-8"
        )
    )
    alpha0 = np.asarray([0.6320268989, 0.04700627128])
    temperature0 = 1188.6287369
    duration = 1.25e-4
    progress_capacity = 0.562979977

    def change(_handoff, _propagation, bc, _reactive):
        bc["kinetics"] = copy.deepcopy(production["bc_global"]["kinetics"])
        bc["thermal"] = copy.deepcopy(production["bc_global"]["thermal"])

    adapted = _adapted(
        rates=(1.0, 1.0), heats=(1.0, 1.0), alpha=tuple(alpha0),
        temperature=temperature0, change=change,
    )
    result = _coordinate(
        adapted, alpha0, temperature0, duration,
        capacity=progress_capacity, maximum_local_refinements=10,
    )
    reference_alpha, reference_temperature = (
        _radau_with_exact_channel_completion(
            adapted, alpha0, temperature0, duration
        )
    )
    sensible0 = float(
        adapted.thermo.heat.sensible_energy(temperature0)
    )
    expected_sensible = sensible0+float(
        (result["alpha"]-alpha0) @ adapted.chemistry.Q
    )
    expected_capacity = progress_capacity-float(
        (result["alpha"]-alpha0) @ adapted.chemistry.weights
    )

    np.testing.assert_allclose(
        result["alpha"], reference_alpha, rtol=0.0, atol=8.0e-8
    )
    assert result["temperature"] == pytest.approx(
        reference_temperature, abs=5.0e-5
    )
    assert result["sensible"] == pytest.approx(
        expected_sensible, rel=0.0, abs=5.0e-10
    )
    assert result["capacity_left"] == pytest.approx(
        expected_capacity, rel=0.0, abs=2.0e-15
    )
    assert np.all((result["alpha"] >= alpha0) & (result["alpha"] <= 1.0))
    assert result["alpha"][0] == 1.0
    assert result["one_active"] is not None
    assert result["accepted_steps"] == 7
    assert result["rejected_steps"] == 1
    assert result["maximum_depth"] == 1
    assert result["maximum_normalized_residual"] <= 1.0
    assert result["rate_evaluations"] < 1500
    assert result["coupled_rate_evaluations"] < 200
    assert result["rate_evaluations"] == (
        result["coupled_rate_evaluations"]
        + result["one_active"]["rate_evaluations"]
    )
