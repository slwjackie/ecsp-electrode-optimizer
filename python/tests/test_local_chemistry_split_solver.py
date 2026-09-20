from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp

from ecsp_nsga2.propagation import (
    PropagationCandidateNumericalError,
    PropagationConfigurationError,
    _kinetic_rate,
)
from ecsp_reactive.condensed.chemistry import (
    A1, A2, ANION, CATION, EC_LP, ENERGY, NCONS, PRODUCT_WATER,
    PVA, RHO, WATER,
    _CoordinateCertificationFailure,
)
from ecsp_reactive.condensed.handoff import BCReactiveHandoffAdapter
from ecsp_reactive.condensed.solver import BCReactiveSolver
from ecsp_reactive.condensed.validation_cases import synthetic_condensed_case


LOCAL_CONTROLS = dict(
    concentration_floor=0.0,
    relative_tolerance=1.0e-9,
    absolute_tolerance=1.0e-12,
    temperature_tolerance_K=1.0e-7,
    maximum_corrector_iterations=10,
    maximum_depletion_iterations=50,
    maximum_local_refinements=16,
)


def _adapted(*, alpha=(0.2, 0.2), temperature=700.0, change=None):
    handoff, propagation, bc, reactive = synthetic_condensed_case(
        shape=(5, 5), alpha=alpha
    )
    handoff["temperatureAtOnset_K"][:] = temperature
    if change is not None:
        change(handoff, propagation, bc, reactive)
    adapted = BCReactiveHandoffAdapter(
        propagation, bc, reactive
    ).adapt(handoff)
    return handoff, propagation, bc, reactive, adapted


def _local_solver(*, duration=1.0e-3, time_step=1.0e-3):
    handoff, propagation, bc, reactive = synthetic_condensed_case(
        shape=(5, 5), alpha=(0.2, 0.2)
    )
    propagation.update(
        duration_s=duration,
        time_step_s=time_step,
        snapshot_interval_s=duration,
    )
    bc["kinetics"]["maximum_rate_per_s"] = 1000.0
    reactive = {
        **reactive,
        "chemistry_integration_mode": "local_adaptive_thermochemical",
        "progress_log_interval_wall_s": 30.0,
    }
    adapted = BCReactiveHandoffAdapter(
        propagation, bc, reactive
    ).adapt(handoff)
    return BCReactiveSolver(adapted, propagation, bc, reactive)


def _v009_hard_coordinate_case():
    def change(_handoff, _propagation, bc, _reactive):
        bc["thermal"]["heat_capacity"] = {
            "mode": "table",
            "temperature_K": [298.15, 373.15, 473.15, 573.15, 773.15],
            "values": [2200.0, 2250.0, 2350.0, 2500.0, 2700.0],
        }
        bc["kinetics"]["mass_conversion_weights"] = [2.0/3.0, 1.0/3.0]
        grid = [value/10.0 for value in range(11)]
        bc["kinetics"]["channels"][0].update(
            alpha_grid=grid,
            activation_energy_J_per_mol=[
                85000, 85000, 91000, 98000, 105000, 112000,
                119000, 127000, 136000, 150000, 150000,
            ],
            ln_Af_per_s=[
                13.0, 13.0, 14.5, 15.5, 16.5, 18.0,
                19.0, 20.0, 21.0, 23.8, 23.8,
            ],
            heat_release_J_per_kg=881000.0,
        )
        bc["kinetics"]["channels"][1].update(
            alpha_grid=grid,
            activation_energy_J_per_mol=[
                150000, 150000, 145000, 140000, 140000, 150000,
                190000, 220000, 190000, 150000, 150000,
            ],
            ln_Af_per_s=[
                19.0, 19.0, 18.5, 18.0, 18.0, 20.0,
                26.0, 29.9, 26.0, 18.0, 18.0,
            ],
            heat_release_J_per_kg=1162000.0,
        )

    return _adapted(
        alpha=(0.6320268988661255, 0.04700627127685306),
        temperature=1188.6287368808814,
        change=change,
    )[-1]


def _radau_completion_reference(chemistry, thermo, alpha0, temperature0,
                                duration):
    sensible0 = float(thermo.heat.sensible_energy(temperature0))
    alpha = np.asarray(alpha0, dtype=np.float64).copy()
    elapsed = 0.0
    while elapsed < duration and np.any(alpha < 1.0-1.0e-13):
        active = np.flatnonzero(alpha < 1.0-1.0e-13)
        fixed = alpha.copy()

        def unpack(value):
            full = fixed.copy()
            full[active] = value
            return full

        def rhs(_time, value):
            full = unpack(value)
            sensible = sensible0+float((full-alpha0) @ chemistry.Q)
            temperature = float(thermo.heat.temperature(sensible))
            return chemistry._rates_from_alpha(
                full[None, :], np.asarray([temperature])
            )[0, active]

        events = []
        for local_index in range(active.size):
            def completion(_time, value, index=local_index):
                return 1.0-value[index]
            completion.terminal = True
            completion.direction = -1
            events.append(completion)
        reference = solve_ivp(
            rhs, (elapsed, duration), alpha[active], method="Radau",
            rtol=2.0e-11, atol=2.0e-13, events=events,
        )
        assert reference.success
        alpha = unpack(reference.y[:, -1])
        elapsed = float(reference.t[-1])
        hit = [index for index, values in enumerate(reference.t_events)
               if values.size]
        if not hit:
            break
        alpha[active[hit[0]]] = 1.0
        elapsed = math.nextafter(elapsed, duration)
    temperature = float(thermo.heat.temperature(
        sensible0+float((alpha-alpha0) @ chemistry.Q)
    ))
    return alpha, temperature


def _coordinate_controls(**changes):
    controls = {
        key: value for key, value in LOCAL_CONTROLS.items()
        if key not in {"concentration_floor", "maximum_corrector_iterations"}
    }
    controls.update(maximum_reaction_coordinate_steps=64)
    controls.update(changes)
    return controls


def test_v009_hard_cell_uses_dominant_coordinate_and_matches_radau():
    adapted = _v009_hard_coordinate_case()
    chemistry = adapted.chemistry
    alpha0 = np.asarray([0.6320268988661255, 0.04700627127685306])
    temperature0 = 1188.6287368808814
    duration = 1.25e-4
    capacity = 0.5629799769969657
    sensible0 = float(adapted.thermo.heat.sensible_energy(temperature0))
    controls = _coordinate_controls(
        relative_tolerance=1.0e-7,
        absolute_tolerance=1.0e-10,
        temperature_tolerance_K=1.0e-4,
        maximum_local_refinements=10,
    )
    reference_alpha, reference_temperature = _radau_completion_reference(
        chemistry, adapted.thermo, alpha0, temperature0, duration
    )
    solved = chemistry._solve_local_cells(
        alpha0[None, :], np.asarray([sensible0]),
        np.asarray([temperature0]), np.asarray([duration]),
        np.asarray([capacity]), adapted.thermo,
        maximum_corrector_iterations=8, **controls,
    )

    np.testing.assert_allclose(
        solved["alpha"][0], reference_alpha, rtol=0.0, atol=2.0e-8
    )
    assert solved["temperature"][0] == pytest.approx(
        reference_temperature, abs=1.0e-5
    )
    assert solved["alpha"][0, 0] == 1.0
    assert solved["coupled_reaction_coordinate_cell_count"][0] == 1
    assert solved["coupled_reaction_coordinate_fallback_count"][0] == 0
    assert solved["coupled_reaction_coordinate_accepted_steps"][0] <= 16
    assert solved["coupled_reaction_coordinate_driver_switches"][0] == 0
    assert solved["panel_attempt_count"][0] == 1
    assert ((solved["alpha"][0]-alpha0) @ chemistry.weights
            <= capacity+2.0e-15)


def test_dominant_coordinate_really_switches_and_is_channel_symmetric():
    def build(swapped):
        def change(_handoff, _propagation, bc, _reactive):
            descending = [math.log(100.0), math.log(0.1)]
            ascending = list(reversed(descending))
            tables = [descending, ascending]
            if swapped:
                tables.reverse()
                bc["kinetics"]["mass_conversion_weights"] = [1.0/3.0, 2.0/3.0]
            else:
                bc["kinetics"]["mass_conversion_weights"] = [2.0/3.0, 1.0/3.0]
            for channel, log_rates in zip(
                    bc["kinetics"]["channels"], tables):
                channel.update(
                    alpha_grid=[0.0, 1.0],
                    activation_energy_J_per_mol=[0.0, 0.0],
                    ln_Af_per_s=log_rates,
                    heat_release_J_per_kg=0.0,
                )
        return _adapted(
            alpha=(0.2, 0.2), temperature=600.0, change=change
        )[-1]

    answers = []
    for swapped in (False, True):
        adapted = build(swapped)
        chemistry = adapted.chemistry
        alpha0 = np.asarray([0.2, 0.2])
        sensible0 = float(adapted.thermo.heat.sensible_energy(600.0))
        answer = chemistry._solve_coupled_reaction_coordinate_cell(
            alpha0, sensible0, 600.0, 0.5, 10.0, adapted.thermo,
            **_coordinate_controls(),
        )
        reference_alpha, _ = _radau_completion_reference(
            chemistry, adapted.thermo, alpha0, 600.0, 0.5
        )
        np.testing.assert_allclose(
            answer["alpha"], reference_alpha, rtol=0.0, atol=2.0e-7
        )
        assert answer["driver_switches"] >= 1
        assert answer["accepted_steps"] <= 64
        answers.append(answer["alpha"])
    np.testing.assert_allclose(answers[0], answers[1][::-1], atol=2.0e-10)


def test_coupled_inventory_event_uses_actual_time_root_without_overshoot():
    adapted = _v009_hard_coordinate_case()
    chemistry = adapted.chemistry
    alpha0 = np.asarray([0.6320268988661255, 0.04700627127685306])
    temperature0 = 1188.6287368808814
    sensible0 = float(adapted.thermo.heat.sensible_energy(temperature0))
    capacity = 0.1
    solved = chemistry._solve_local_cells(
        alpha0[None, :], np.asarray([sensible0]),
        np.asarray([temperature0]), np.asarray([1.25e-4]),
        np.asarray([capacity]), adapted.thermo,
        maximum_corrector_iterations=8,
        **_coordinate_controls(maximum_local_refinements=10),
    )
    progress = float((solved["alpha"][0]-alpha0) @ chemistry.weights)
    assert solved["coupled_reaction_coordinate_attempt_count"][0] > 0
    assert solved["coupled_reaction_coordinate_fallback_count"][0] == 0
    assert solved["coupled_reaction_coordinate_cell_count"][0] == 1
    assert solved["depleted"][0]
    assert 0.0 < solved["depletion_time"][0] < 1.25e-4
    assert progress <= capacity
    assert capacity-progress <= 1.1e-8


def test_coordinate_step_guard_is_a_midpoint_fallback_not_rejection(
        monkeypatch):
    def change(_handoff, _propagation, bc, _reactive):
        gas_constant=8.31446261815324
        first_activation=60000.0
        bc["kinetics"]["channels"][0].update(
            alpha_grid=[0.0,0.21,1.0],
            activation_energy_J_per_mol=[first_activation]*3,
            ln_Af_per_s=[
                math.log(4.0)+first_activation/(gas_constant*600.0)
            ]*3,
            heat_release_J_per_kg=1.0e5,
        )
        second_activation=40000.0
        bc["kinetics"]["channels"][1].update(
            activation_energy_J_per_mol=[second_activation]*2,
            ln_Af_per_s=[
                second_activation/(gas_constant*600.0)
            ]*2,
            heat_release_J_per_kg=1.0e5,
        )

    adapted = _adapted(
        alpha=(0.2,0.2),temperature=600.0,change=change
    )[-1]
    chemistry = adapted.chemistry
    calls={"count":0}

    def exhausted_guard(*_args, **_kwargs):
        calls["count"]+=1
        raise _CoordinateCertificationFailure(
            "dominant-coordinate step guard requested midpoint fallback"
        )

    monkeypatch.setattr(
        chemistry,"_solve_coupled_reaction_coordinate_cell",exhausted_guard
    )
    alpha0 = np.asarray([0.2,0.2])
    temperature0 = 600.0
    sensible0 = float(adapted.thermo.heat.sensible_energy(temperature0))
    solved = chemistry._solve_local_cells(
        alpha0[None, :], np.asarray([sensible0]),
        np.asarray([temperature0]), np.asarray([1.0e-2]),
        np.asarray([0.8]), adapted.thermo,
        maximum_corrector_iterations=8,
        **_coordinate_controls(
            maximum_local_refinements=16,
            maximum_reaction_coordinate_steps=1,
        ),
    )
    assert calls["count"]>0
    assert solved["coupled_reaction_coordinate_cell_count"][0] == 0
    assert solved["coupled_reaction_coordinate_fallback_count"][0] > 0
    assert np.all(solved["alpha"][0]>alpha0)
    assert solved["panel_attempt_count"][0] > 1


def test_local_mode_rejects_zero_weight_channel_semantics_explicitly():
    def change(_handoff, _propagation, bc, _reactive):
        bc["kinetics"]["mass_conversion_weights"] = [1.0, 0.0]

    adapted = _adapted(change=change)[-1]
    with pytest.raises(
            PropagationConfigurationError,
            match="strictly positive mass_conversion_weights"):
        adapted.chemistry.advance_local(
            adapted.U, adapted.thermo, 1.0e-3, **LOCAL_CONTROLS
        )


def test_one_active_reaction_coordinate_matches_radau_for_either_channel():
    gas_constant = 8.31446261815324
    endpoints = []
    for active_channel in (0, 1):
        alpha = [1.0, 1.0]
        alpha[active_channel] = 0.2

        def change(_handoff, _propagation, bc, _reactive):
            bc["thermal"]["heat_capacity"] = {
                "mode": "table",
                "temperature_K": [500.0, 675.0, 900.0],
                "values": [1800.0, 2200.0, 2700.0],
            }
            activation = 55000.0
            reference_log_rate = (
                math.log(1.2)+activation/(gas_constant*650.0)
            )
            for channel in bc["kinetics"]["channels"]:
                channel.update(
                    alpha_grid=[0.0, 0.3, 0.6, 1.0],
                    activation_energy_J_per_mol=[activation]*4,
                    ln_Af_per_s=[
                        reference_log_rate,
                        reference_log_rate+0.4,
                        reference_log_rate-0.2,
                        reference_log_rate+0.5,
                    ],
                    heat_release_J_per_kg=5.0e5,
                )

        _, _, _, _, adapted = _adapted(
            alpha=tuple(alpha), temperature=650.0, change=change
        )
        dt = 0.08
        alpha_start = 0.2
        sensible_start = float(
            adapted.thermo.heat.sensible_energy(650.0)
        )

        def rhs(_time, value):
            sensible = (sensible_start
                        +adapted.chemistry.Q[active_channel]
                         *(value[0]-alpha_start))
            temperature = float(adapted.thermo.heat.temperature(sensible))
            return [_kinetic_rate(
                np.asarray(value[0]), np.asarray(temperature),
                adapted.chemistry.channels[active_channel],
                adapted.chemistry.R,
            )]

        reference = solve_ivp(
            rhs, (0.0, dt), [alpha_start], method="Radau",
            rtol=2.0e-12, atol=2.0e-14,
        )
        assert reference.success
        assert reference.y[0, -1] < 1.0
        after, diagnostics = adapted.chemistry.advance_local(
            adapted.U, adapted.thermo, dt, **LOCAL_CONTROLS
        )
        alpha_after = after[0, 0, A1:A2+1]/after[0, 0, RHO]
        inactive_channel = 1-active_channel
        assert alpha_after[inactive_channel] == 1.0
        assert alpha_after[active_channel] == pytest.approx(
            reference.y[0, -1], abs=2.0e-9
        )
        assert diagnostics["one_active_reaction_coordinate_cell_count"] == 25
        assert diagnostics["one_active_reaction_coordinate_rate_evaluation_count"] > 0
        assert diagnostics["local_chemistry_method"] == (
            "bulk_midpoint_with_dominant_driver_reaction_coordinate_and_"
            "error_controlled_midpoint_fallback"
        )
        assert diagnostics[
            "maximum_temperature_feedback_endpoint_temperature_correction_K"
        ] > 0.0
        endpoints.append(alpha_after[active_channel])

    assert endpoints[0] == pytest.approx(endpoints[1], abs=2.0e-12)


@pytest.mark.parametrize(
    "log_prefactor",
    ([62.0, 67.0, 72.0, 75.0],
     [-106.0, -101.0, -96.0, -91.0]),
)
def test_thermochemical_clock_explicitly_splits_clamp_crossing_and_matches_quad(
        log_prefactor):
    def change(_handoff, _propagation, bc, _reactive):
        bc["thermal"]["heat_capacity"] = {
            "mode": "table",
            "temperature_K": [400.0, 650.0, 900.0],
            "values": [1800.0, 2200.0, 2600.0],
        }
        bc["kinetics"]["channels"][0].update(
            alpha_grid=[0.0, 0.35, 0.7, 1.0],
            activation_energy_J_per_mol=[20000.0]*4,
            ln_Af_per_s=list(log_prefactor),
            heat_release_J_per_kg=5.0e5,
        )

    _, _, _, _, adapted = _adapted(
        alpha=(0.1, 1.0), temperature=600.0, change=change
    )
    chemistry = adapted.chemistry
    sensible0 = float(adapted.thermo.heat.sensible_energy(600.0))
    breaks, _ = chemistry._reaction_coordinate_breaks(
        0.1, 0.9, sensible0, 0, adapted.thermo.heat
    )

    def unclipped(alpha):
        temperature = float(adapted.thermo.heat.temperature(
            sensible0+chemistry.Q[0]*(alpha-0.1)
        ))
        channel = chemistry.channels[0]
        activation = np.interp(
            alpha, channel["alpha_grid"],
            channel["activation_energy_J_per_mol"],
        )
        log_a = np.interp(
            alpha, channel["alpha_grid"], channel["ln_Af_per_s"]
        )
        return log_a-activation/(chemistry.R*temperature)

    threshold = 60.0 if log_prefactor[0] > 0.0 else -100.0
    assert min(abs(unclipped(point)-threshold) for point in breaks) < 2.0e-12

    calculated = chemistry._reaction_coordinate_clock_path(
        0.1, 0.9, sensible0, 0, adapted.thermo.heat,
        alpha_tolerance=1.001e-9,
        temperature_tolerance_K=1.0e-6,
        maximum_local_refinements=16,
    )

    def inverse_rate(alpha):
        temperature = float(adapted.thermo.heat.temperature(
            sensible0+chemistry.Q[0]*(alpha-0.1)
        ))
        return 1.0/float(_kinetic_rate(
            np.asarray(alpha), np.asarray(temperature),
            chemistry.channels[0], chemistry.R,
        ))

    reference = quad(
        inverse_rate, 0.1, 0.9,
        points=list(breaks[1:-1]), epsabs=0.0, epsrel=2.0e-12,
        limit=500,
    )[0]
    assert calculated["clock"] == pytest.approx(reference, rel=2.0e-12)
    assert calculated["maximum_normalized_residual"] <= 0.125


@pytest.mark.parametrize(
    "retired_key",
    [
        "maximum_channel_increment",
        "maximum_chemistry_subcycles_per_pde_step",
        "maximum_chemical_rate_cap_fraction",
    ],
)
def test_local_solver_rejects_legacy_cap_or_subcycle_controls(retired_key):
    handoff, propagation, bc, reactive = synthetic_condensed_case(shape=(5, 5))
    reactive.update(
        chemistry_integration_mode="local_adaptive_thermochemical",
        **{retired_key: 1},
    )
    adapted = BCReactiveHandoffAdapter(propagation, bc, reactive).adapt(handoff)
    with pytest.raises(PropagationConfigurationError, match="retired legacy"):
        BCReactiveSolver(adapted, propagation, bc, reactive)


def test_subcycle_raw_is_retired_but_legacy_cap_remains_available():
    handoff, propagation, bc, reactive = synthetic_condensed_case(shape=(5, 5))
    retired = {**reactive, "chemistry_integration_mode": "subcycle_raw"}
    adapted = BCReactiveHandoffAdapter(propagation, bc, retired).adapt(handoff)
    with pytest.raises(PropagationConfigurationError, match="subcycle_raw is retired"):
        BCReactiveSolver(adapted, propagation, bc, retired)

    legacy = {
        **reactive,
        "chemistry_integration_mode": "legacy_cap",
        "maximum_channel_increment": 0.01,
        "maximum_chemical_rate_cap_fraction": 0.0,
    }
    adapted = BCReactiveHandoffAdapter(propagation, bc, legacy).adapt(handoff)
    solver = BCReactiveSolver(adapted, propagation, bc, legacy)
    assert solver.chemistry_integration_mode == "legacy_cap"
    assert solver.alpha_step == 0.01


def test_strang_split_order_exact_chemical_ledger_and_no_reaction_cfl(
        monkeypatch):
    solver = _local_solver(duration=0.1, time_step=0.1)
    before = solver.U.copy()
    calls = []
    increments = iter((2.0, 3.0))

    def chemistry(state, half_dt):
        calls.append(("C", half_dt))
        increment = next(increments)
        result = state.copy()
        result[..., ENERGY] += increment
        return result, np.full(state.shape[:-1], increment), {}

    def nonchemical(_time, state, dt, first_order=False):
        calls.append(("N", dt, first_order))
        result = state.copy()
        result[..., ENERGY] += 5.0
        zero = np.zeros(state.shape[:-1])
        fields = {
            "qJ_W_per_m3": zero, "qEchem_W_per_m3": zero,
            "qChem_W_per_m3": zero, "conduction_W_per_m3": zero,
            "loss_W_per_m3": zero,
        }
        transport = tuple({
            "mechanics_skipped_exact": True,
            "face_fallbacks": 0, "faces": 0, "hllc_fallbacks": 0,
        } for _ in range(3))
        return result, np.zeros(NCONS+5), transport, fields

    monkeypatch.setattr(solver, "_chemistry_half_step", chemistry)
    monkeypatch.setattr(solver, "_advance_nonchemical_ssprk", nonchemical)
    monkeypatch.setattr(solver, "_nonchemical_time_step_limit", lambda _u: math.inf)
    result, ledger, diagnostics, sources = solver.advance(0.0, before, 0.1)

    assert calls == [("C", 0.05), ("N", 0.1, False), ("C", 0.05)]
    np.testing.assert_allclose(result[..., ENERGY]-before[..., ENERGY], 10.0)
    expected_heat = (2.0+3.0)*np.prod(before.shape[:-1])*solver.vol
    assert ledger[NCONS+2] == pytest.approx(expected_heat)
    np.testing.assert_allclose(sources["qChem_W_per_m3"], 50.0)
    assert len(diagnostics["chemistry_half_steps"]) == 2

    monkeypatch.setattr(
        solver.chem, "raw_rates",
        lambda *_args, **_kwargs: pytest.fail(
            "local chemistry must not enter the reaction CFL"
        ),
    )
    monkeypatch.setattr(solver, "_nonchemical_time_step_limit", lambda _u: 0.25)
    assert solver.time_step(before, 0.1) == pytest.approx(0.1)


def test_second_chemistry_half_uses_transported_density_alpha_and_inventory(
        monkeypatch):
    solver = _local_solver(duration=0.1, time_step=0.1)
    before = solver.U.copy()
    real_chemistry = solver._chemistry_half_step
    chemistry_inputs = []

    def chemistry(state, half_dt):
        chemistry_inputs.append(state.copy())
        if len(chemistry_inputs) == 1:
            return state.copy(), np.zeros(state.shape[:-1]), {}
        return real_chemistry(state, half_dt)

    def nonchemical(_time, state, _dt, first_order=False):
        del first_order
        primitive = solver.thermo.primitive(state)
        # Increase density very slightly; the synthetic Tait law has a large
        # B, so a 10% density reduction would itself imply negative pressure.
        primitive[..., RHO] = 1000.001
        primitive[..., A1] = 0.998
        primitive[..., A2] = 0.997
        rho = primitive[..., RHO]
        progress = (solver.chem.weights[0]*primitive[..., A1]
                    +solver.chem.weights[1]*primitive[..., A2])
        extent = rho*solver.chem.xi_per_kg*progress
        primitive[..., CATION] = (
            rho*solver.chem.initial_lp_per_kg-1.45*extent
        )/rho
        primitive[..., ANION] = primitive[..., CATION]
        primitive[..., WATER] = solver.chem.initial_water_per_kg
        primitive[..., PVA] = (
            rho*solver.chem.initial_pva_per_kg-extent
        )/rho
        primitive[..., PRODUCT_WATER] = 2.0*extent/rho
        primitive[..., EC_LP] = 0.0
        result = solver.thermo.conservative(primitive)
        zero = np.zeros(state.shape[:-1])
        fields = {
            "qJ_W_per_m3": zero, "qEchem_W_per_m3": zero,
            "qChem_W_per_m3": zero, "conduction_W_per_m3": zero,
            "loss_W_per_m3": zero,
        }
        transport = tuple({
            "mechanics_skipped_exact": False,
            "face_fallbacks": 0, "faces": 1, "hllc_fallbacks": 0,
        } for _ in range(3))
        return result, np.zeros(NCONS+5), transport, fields

    monkeypatch.setattr(solver, "_chemistry_half_step", chemistry)
    monkeypatch.setattr(solver, "_advance_nonchemical_ssprk", nonchemical)
    monkeypatch.setattr(
        solver, "_nonchemical_time_step_limit", lambda _u: math.inf
    )
    after, _ledger, _diagnostics, _sources = solver.advance(
        0.0, before, 0.1
    )

    assert len(chemistry_inputs) == 2
    transported = chemistry_inputs[1]
    assert np.all(transported[..., RHO] == 1000.001)
    assert not np.array_equal(transported[..., A1:A2+1],
                              before[..., A1:A2+1])
    assert not np.array_equal(transported[..., PVA], before[..., PVA])
    rho = transported[..., RHO]
    start_alpha = transported[..., A1:A2+1]/rho[..., None]
    final_alpha = after[..., A1:A2+1]/rho[..., None]
    progress_increment = ((final_alpha-start_alpha)@solver.chem.weights)
    maximum_extent = np.minimum.reduce([
        0.5*(transported[..., CATION]+transported[..., ANION])/1.45,
        transported[..., PVA],
        transported[..., CATION]/1.45,
        transported[..., ANION]/1.45,
    ])
    transported_capacity = maximum_extent/(rho*solver.chem.xi_per_kg)
    np.testing.assert_allclose(
        progress_increment, transported_capacity, rtol=0.0, atol=2.0e-12
    )
    assert np.all(progress_increment <= transported_capacity+2.0e-12)
    extent_increment = rho*solver.chem.xi_per_kg*progress_increment
    np.testing.assert_allclose(
        transported[..., CATION]-after[..., CATION],
        1.45*extent_increment, rtol=0.0, atol=2.0e-10,
    )
    np.testing.assert_allclose(
        transported[..., PVA]-after[..., PVA],
        extent_increment, rtol=0.0, atol=2.0e-10,
    )
    assert np.min(after[..., CATION]) >= -1.0e-10
    assert np.min(after[..., PVA]) >= -1.0e-10


def test_post_first_half_stability_is_rechecked_before_nonchemical_step(
        monkeypatch):
    solver = _local_solver(duration=0.1, time_step=0.1)
    monkeypatch.setattr(
        solver, "_chemistry_half_step",
        lambda state, _dt: (state.copy(), np.zeros(state.shape[:-1]), {}),
    )
    monkeypatch.setattr(solver, "_nonchemical_time_step_limit", lambda _u: 0.04)
    monkeypatch.setattr(
        solver, "_advance_nonchemical_ssprk",
        lambda *_args, **_kwargs: pytest.fail(
            "nonchemical SSPRK must not run after a failed recheck"
        ),
    )
    with pytest.raises(PropagationCandidateNumericalError) as caught:
        solver.advance(0.0, solver.U, 0.1)
    assert caught.value.retry_reason == "post_chemistry_timestep_recheck"
    assert caught.value.retry_time_step_cap_s == pytest.approx(0.04)
    assert solver.post_chemistry_timestep_rechecks == 1
    assert solver.post_chemistry_timestep_recheck_rejections == 1


def _empty_success_contract(state):
    zero = np.zeros(state.shape[:-1])
    fields = {
        "qJ_W_per_m3": zero, "qEchem_W_per_m3": zero,
        "qChem_W_per_m3": zero, "conduction_W_per_m3": zero,
        "loss_W_per_m3": zero,
    }
    transport = tuple({
        "mechanics_skipped_exact": True,
        "face_fallbacks": 0, "faces": 0, "hllc_fallbacks": 0,
    } for _ in range(3))
    diagnostics = {
        "transport_stages": transport,
        "chemistry_half_steps": ({}, {}),
        "post_first_half_stability_safety_ratio": 1.0,
    }
    return state.copy(), np.zeros(NCONS+5), diagnostics, fields


def test_stability_retry_rolls_back_state_ledgers_arrival_and_electrical_warm_state(
        tmp_path, monkeypatch):
    solver = _local_solver(duration=1.0e-3, time_step=1.0e-3)
    initial = solver.U.copy()
    electrical = SimpleNamespace(
        potential=np.full(initial.shape[:-1], 3.0),
        maximum_mismatch=0.25,
        last_fields={"probe": np.asarray([1.0])},
        calls=0,
    )
    solver.electrical = electrical
    solver.progress_log_interval_wall_s = 1.0e-12
    attempted_dt = []

    def advance(_time, state, dt, first_order=False):
        attempted_dt.append(dt)
        electrical.calls += 1
        if len(attempted_dt) == 1:
            electrical.potential.fill(99.0)
            electrical.maximum_mismatch = 99.0
            electrical.last_fields = {"probe": np.asarray([99.0])}
            error = PropagationCandidateNumericalError("retry stability")
            error.retry_reason = "post_chemistry_timestep_recheck"
            error.retry_time_step_cap_s = 4.0e-4
            raise error
        if len(attempted_dt) == 2:
            np.testing.assert_array_equal(electrical.potential, 3.0)
            assert electrical.maximum_mismatch == 0.25
            np.testing.assert_array_equal(
                electrical.last_fields["probe"], [1.0]
            )
            np.testing.assert_array_equal(state, initial)
        electrical.potential.fill(7.0)
        electrical.maximum_mismatch = 0.5
        electrical.last_fields = {"probe": np.asarray([7.0])}
        return _empty_success_contract(state)

    printed = []

    def capture_print(*args, **kwargs):
        printed.append((args, kwargs))

    monkeypatch.setattr(solver, "advance", advance)
    monkeypatch.setattr("builtins.print", capture_print)
    metrics = solver.run(tmp_path)

    assert attempted_dt == pytest.approx([1.0e-3, 4.0e-4, 6.0e-4])
    np.testing.assert_array_equal(solver.U, initial)
    assert metrics["acceptedTimeSteps"] == 2
    assert metrics["rejectedTimeSteps"] == 1
    assert metrics["frontArrivalCoverageFraction"] == 0.0
    assert metrics["establishedTimeCensored"] is True
    assert metrics["integratedChemicalHeat_J"] == 0.0
    assert metrics["integratedJouleHeat_J"] == 0.0
    assert metrics["maximumAbsoluteMassBudgetResidual_kg"] == 0.0
    assert metrics["maximumAbsoluteEnergyBudgetResidual_J"] == 0.0
    assert metrics["maximumAbsoluteEnergyPlusChemicalBudgetResidual_J"] == 0.0
    assert metrics["maximumLocalInventoryResidual_mol_per_m3"] >= 0.0
    assert metrics["maximumAbsoluteIntegratedLPInventoryResidual_mol"] >= 0.0
    assert metrics["inheritedChemicalRateDiagnosticThreshold_per_s"] == 1000.0
    assert metrics[
        "maximumFractionAboveInheritedChemicalRateDiagnosticThreshold"
    ] == 0.0
    assert "not_the_same_physical_trajectory" in metrics[
        "chemistryTrajectoryProvenance"
    ]
    np.testing.assert_array_equal(electrical.potential, 7.0)
    assert any(
        json.loads(args[0])["trial_status"] == "rejected_retry_pending"
        and kwargs.get("flush") is True
        for args, kwargs in printed
        if args and isinstance(args[0], str)
    )


def test_local_chemistry_failure_is_terminal_and_does_not_halve_pde_dt(
        tmp_path, monkeypatch):
    solver = _local_solver(duration=1.0e-3, time_step=1.0e-3)
    initial = solver.U.copy()
    electrical = SimpleNamespace(
        potential=np.full(initial.shape[:-1], 4.0),
        maximum_mismatch=0.1,
        last_fields={"probe": np.asarray([2.0])},
        calls=0,
    )
    solver.electrical = electrical
    attempts = []

    def fail(_time, _state, dt, first_order=False):
        attempts.append((dt, first_order))
        electrical.calls += 1
        electrical.potential.fill(88.0)
        electrical.maximum_mismatch = 88.0
        electrical.last_fields = {"probe": np.asarray([88.0])}
        error = PropagationCandidateNumericalError(
            "configured panel-attempt work bound"
        )
        error.retry_reason = "local_chemistry"
        raise error

    monkeypatch.setattr(solver, "advance", fail)
    with pytest.raises(
            PropagationCandidateNumericalError,
            match="panel-attempt work bound"):
        solver.run(tmp_path)

    assert attempts == [(1.0e-3, False)]
    assert solver.rejected_steps == 1
    assert solver.step_number == 0
    np.testing.assert_array_equal(solver.U, initial)
    np.testing.assert_array_equal(electrical.potential, 4.0)
    assert electrical.maximum_mismatch == 0.1
    np.testing.assert_array_equal(electrical.last_fields["probe"], [2.0])
    assert not (tmp_path/"propagation_metrics.json").exists()
