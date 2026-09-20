from __future__ import annotations

import copy
import math

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp

from ecsp_nsga2.propagation import (
    PropagationCandidateNumericalError,
    PropagationConfigurationError,
)
from ecsp_reactive.condensed.chemistry import (
    A1, A2, ANION, CATION, EC_LP, ENERGY, PRODUCT_WATER, PVA, RHO, WATER,
    _CoordinateCertificationFailure,
)
from ecsp_reactive.condensed.handoff import BCReactiveHandoffAdapter
from ecsp_reactive.condensed.validation_cases import synthetic_condensed_case


DEFAULT_CONTROLS = dict(
    concentration_floor=0.0,
    relative_tolerance=1.0e-7,
    absolute_tolerance=1.0e-10,
    temperature_tolerance_K=1.0e-4,
    maximum_corrector_iterations=8,
    maximum_depletion_iterations=40,
    maximum_local_refinements=10,
)


def adapted_case(*, rates=(0.2, 0.1), heats=(1000.0, 2000.0),
                 alpha=(0.2, 0.2), temperature=300.0, change=None):
    handoff, propagation, bc, reactive = synthetic_condensed_case(
        shape=(5, 5), rates=rates, heats=heats, alpha=alpha
    )
    handoff["temperatureAtOnset_K"][:] = temperature
    if change is not None:
        change(handoff, propagation, bc, reactive)
    return BCReactiveHandoffAdapter(propagation, bc, reactive).adapt(handoff)


def test_fixed_beta_clock_crosses_knots_and_both_exponent_clamps():
    def change(_handoff, _propagation, bc, _reactive):
        bc["kinetics"]["channels"][0].update(
            alpha_grid=[0.0, 0.3, 0.6, 1.0],
            activation_energy_J_per_mol=[0.0, 0.0, 0.0, 0.0],
            ln_Af_per_s=[0.0, 3.0, 70.0, -130.0],
        )

    adapted = adapted_case(alpha=(0.25, 0.2), change=change)
    chemistry = adapted.chemistry
    channel = chemistry.channels[0]
    start = 0.25
    expected = 0.97

    def inverse_rate(value):
        log_rate = np.interp(
            value, channel["alpha_grid"], channel["ln_Af_per_s"]
        )
        return math.exp(-float(np.clip(log_rate, -100.0, 60.0)))

    # Independent adaptive quadrature includes the table knots and the upper
    # and lower exponent-clamp intersections.
    duration = quad(
        inverse_rate, start, expected,
        points=[0.3, 0.5552238805970149, 0.6, 0.62, 0.94],
        epsabs=0.0, epsrel=2.0e-13,
        limit=200,
    )[0]
    actual = chemistry._fixed_beta_flow_channel(
        np.asarray([start]), np.asarray([duration]),
        np.asarray([1.0/300.0]), 0,
    )[0]
    assert actual == pytest.approx(expected, rel=0.0, abs=3.0e-13)


def test_small_batch_scalar_clock_matches_vector_clock_randomized():
    def change(_handoff, _propagation, bc, _reactive):
        bc["kinetics"]["channels"][0].update(
            alpha_grid=[0.0, 0.3, 0.6, 1.0],
            activation_energy_J_per_mol=[0.0, 0.0, 0.0, 0.0],
            ln_Af_per_s=[0.0, 3.0, 70.0, -130.0],
        )

    adapted = adapted_case(alpha=(0.25, 0.2), change=change)
    chemistry = adapted.chemistry
    channel = chemistry.channels[0]
    random = np.random.default_rng(20260912)
    starts = random.uniform(0.0, 0.88, 64)
    targets = starts + random.uniform(0.0, 1.0, 64)*(1.0-starts)
    # Guarantee that several randomized trajectories cross every table knot
    # and both exponent-clamp intersections.
    starts[:8] = random.uniform(0.0, 0.2, 8)
    targets[:8] = random.uniform(0.95, 0.99, 8)
    beta = 1.0/random.uniform(250.0, 1800.0, 64)

    def inverse_rate(value):
        log_rate = np.interp(
            value, channel["alpha_grid"], channel["ln_Af_per_s"]
        )
        return math.exp(-float(np.clip(log_rate, -100.0, 60.0)))

    boundaries = (0.3, 0.5552238805970149, 0.6, 0.62, 0.94)
    durations = np.asarray([
        quad(
            inverse_rate, start, target,
            points=[point for point in boundaries if start < point < target],
            epsabs=0.0, epsrel=2.0e-13, limit=200,
        )[0]
        for start, target in zip(starts, targets)
    ])

    scalar = chemistry._fixed_beta_flow_channel(starts, durations, beta, 0)
    # The implementation uses the vector path only above 64 gathered cells.
    # Append an inert 65th cell, then compare the identical first 64 inputs.
    vector = chemistry._fixed_beta_flow_channel(
        np.append(starts, 0.5), np.append(durations, 0.0),
        np.append(beta, 1.0/300.0), 0,
    )[:64]
    np.testing.assert_allclose(scalar, vector, rtol=0.0, atol=4.0e-13)
    assert np.all(scalar[:8] > 0.94)
    assert np.isfinite(scalar).all()
    assert np.all((scalar >= starts) & (scalar <= 1.0))


def test_local_thermochemical_endpoint_matches_independent_dop853_reference():
    gas_constant = 8.31446261815324

    def change(_handoff, _propagation, bc, _reactive):
        specifications = (
            (60000.0, math.log(2.5)+60000.0/(gas_constant*800.0), 2.0e5),
            (80000.0, math.log(0.9)+80000.0/(gas_constant*800.0), 3.0e5),
        )
        for channel, (activation, log_prefactor, heat) in zip(
                bc["kinetics"]["channels"], specifications):
            channel.update(
                activation_energy_J_per_mol=[activation, activation],
                ln_Af_per_s=[log_prefactor, log_prefactor],
                heat_release_J_per_kg=heat,
            )

    adapted = adapted_case(
        alpha=(0.15, 0.1), temperature=800.0, change=change
    )
    chemistry = adapted.chemistry
    dt = 0.02
    after, diagnostics = chemistry.advance_local(
        adapted.U, adapted.thermo, dt, **{
            **DEFAULT_CONTROLS,
            "relative_tolerance": 2.0e-8,
            "absolute_tolerance": 2.0e-11,
            "temperature_tolerance_K": 2.0e-5,
        }
    )

    alpha0 = np.asarray([0.15, 0.1])
    sensible0 = float(adapted.thermo.heat.sensible_energy(800.0))

    def rhs(_time, alpha):
        sensible = sensible0 + float((alpha-alpha0) @ chemistry.Q)
        temperature = float(adapted.thermo.heat.temperature(sensible))
        return chemistry._rates_from_alpha(alpha[None, :], temperature)[0]

    reference = solve_ivp(
        rhs, (0.0, dt), alpha0, method="DOP853",
        rtol=2.0e-12, atol=2.0e-14,
    )
    assert reference.success
    alpha_after = after[0, 0, A1:A2+1]/after[0, 0, RHO]
    np.testing.assert_allclose(alpha_after, reference.y[:, -1], rtol=3.0e-7,
                               atol=3.0e-9)
    reference_sensible = sensible0 + float((reference.y[:, -1]-alpha0) @ chemistry.Q)
    reference_temperature = float(adapted.thermo.heat.temperature(reference_sensible))
    actual_temperature = float(adapted.thermo.primitive(after)[0, 0, 3])
    assert actual_temperature == pytest.approx(reference_temperature, abs=2.0e-4)
    assert diagnostics["maximum_normalized_embedded_residual"] <= 1.0
    assert diagnostics["endpoint_evaluation_count"] > 0
    assert diagnostics["evaluated_cell_count"] > 0


def test_extreme_stiff_minimum_timestep_is_not_treated_as_zero_time():
    gas_constant = 8.31446261815324
    initial_rates = (1.0e11, 5.0e10)
    activation = 8.0e4
    dt = 1.0e-14

    def change(_handoff, _propagation, bc, _reactive):
        for channel, rate in zip(bc["kinetics"]["channels"], initial_rates):
            channel.update(
                activation_energy_J_per_mol=[activation, activation],
                ln_Af_per_s=[
                    math.log(rate)+activation/(gas_constant*300.0)
                ]*2,
                heat_release_J_per_kg=1.0e6,
            )

    adapted = adapted_case(alpha=(0.1, 0.1), temperature=300.0, change=change)
    chemistry = adapted.chemistry
    alpha0 = np.asarray([0.1, 0.1])
    sensible0 = float(adapted.thermo.heat.sensible_energy(300.0))

    # Integrate in dimensionless time so the independent reference does not
    # itself need to resolve a 1e-14-second independent variable.
    def rhs(_scaled_time, alpha):
        sensible = sensible0+float((alpha-alpha0) @ chemistry.Q)
        temperature = float(adapted.thermo.heat.temperature(sensible))
        return dt*chemistry._rates_from_alpha(
            alpha[None, :], np.asarray([temperature])
        )[0]

    reference = solve_ivp(
        rhs, (0.0, 1.0), alpha0, method="DOP853",
        rtol=2.0e-13, atol=2.0e-15,
    )
    assert reference.success
    after, diagnostics = chemistry.advance_local(
        adapted.U, adapted.thermo, dt, **DEFAULT_CONTROLS
    )
    alpha_after = after[0, 0, A1:A2+1]/after[0, 0, RHO]

    # The initial raw increments alone are 1e-3 and 5e-4.  The old one-second
    # roundoff floor entered the coupled-coordinate shortcut, accepted no
    # steps, and returned alpha0 exactly; its scheduler floor could also stop
    # after only the first refined half-panel.
    assert np.all(alpha_after-alpha0 > np.asarray([1.0e-3, 5.0e-4]))
    np.testing.assert_allclose(
        alpha_after, reference.y[:, -1], rtol=0.0, atol=2.0e-8
    )
    assert diagnostics["coupled_reaction_coordinate_attempt_count"] > 0


def test_shared_inventory_stops_both_channels_at_one_nonovershooting_time():
    adapted = adapted_case(rates=(2.0, 1.0), heats=(1000.0, 2000.0))
    state = adapted.U.copy()
    state[..., CATION] = 14.5
    state[..., ANION] = 14.5
    state[..., PVA] = 10.0
    before = state.copy()
    controls = {
        **DEFAULT_CONTROLS,
        "relative_tolerance": 1.0e-10,
        "absolute_tolerance": 1.0e-12,
        "temperature_tolerance_K": 1.0e-7,
        "maximum_depletion_iterations": 50,
    }
    dt = 0.5
    after, diagnostics = adapted.chemistry.advance_local(
        state, adapted.thermo, dt, **controls
    )

    # rho*xi_per_kg == 100 mol/m3 and the shared capacity is 10 mol/m3,
    # hence weighted progress capacity is 0.1.  With constant rates, the
    # common event time is 0.1 / ((2/3)*2 + (1/3)*1) == 0.06 s.
    expected_time = 0.06
    delta = (after[..., A1:A2+1]-before[..., A1:A2+1]) / before[..., RHO, None]
    np.testing.assert_allclose(delta[..., 0], 2.0*expected_time,
                               rtol=0.0, atol=3.0e-10)
    np.testing.assert_allclose(delta[..., 1], expected_time,
                               rtol=0.0, atol=3.0e-10)
    assert diagnostics["minimum_inventory_depletion_time_fraction"] == pytest.approx(
        expected_time/dt, abs=3.0e-10
    )
    assert diagnostics["inventory_depleted_fraction"] == 1.0
    assert np.min(after[..., [CATION, ANION, PVA]]) >= 0.0
    assert np.max(after[..., PVA]) <= 2.0e-8

    d_rho_alpha = after[..., A1:A2+1]-before[..., A1:A2+1]
    expected_energy = d_rho_alpha @ adapted.chemistry.Q
    np.testing.assert_allclose(
        after[..., ENERGY]-before[..., ENERGY], expected_energy,
        rtol=2.0e-13, atol=2.0e-8,
    )
    dxi = adapted.chemistry.xi_per_kg*(d_rho_alpha @ adapted.chemistry.weights)
    np.testing.assert_allclose(after[..., CATION]-before[..., CATION], -1.45*dxi)
    np.testing.assert_allclose(after[..., ANION]-before[..., ANION], -1.45*dxi)
    np.testing.assert_allclose(after[..., PVA]-before[..., PVA], -dxi)
    np.testing.assert_allclose(
        after[..., PRODUCT_WATER]-before[..., PRODUCT_WATER], 2.0*dxi
    )
    np.testing.assert_array_equal(after[..., WATER], before[..., WATER])
    np.testing.assert_array_equal(after[..., EC_LP], before[..., EC_LP])


def test_empty_shared_inventory_allows_exactly_zero_reaction():
    adapted = adapted_case(rates=(2.0e4, 1.0e4), heats=(5.0e5, 8.0e5))
    state = adapted.U.copy()
    state[..., CATION] = 0.0
    state[..., ANION] = 0.0
    state[..., PVA] = 0.0
    after, diagnostics = adapted.chemistry.advance_local(
        state, adapted.thermo, 1.0e-3, **DEFAULT_CONTROLS
    )
    np.testing.assert_array_equal(after, state)
    assert diagnostics["inventory_depleted_fraction"] == 1.0
    assert diagnostics["minimum_inventory_depletion_time_fraction"] == 0.0
    assert diagnostics["endpoint_evaluation_count"] == 0


def test_one_active_coordinate_certification_failure_uses_midpoint_fallback(
        monkeypatch):
    adapted = adapted_case(
        rates=(0.2, 0.1), heats=(0.0, 0.0), alpha=(1.0, 0.2)
    )

    def decline_coordinate(*_args, **_kwargs):
        raise _CoordinateCertificationFailure("test certification decline")

    monkeypatch.setattr(
        adapted.chemistry, "_solve_one_active_reaction_coordinate_cell",
        decline_coordinate,
    )
    after, diagnostics = adapted.chemistry.advance_local(
        adapted.U, adapted.thermo, 1.0e-2, **DEFAULT_CONTROLS
    )
    alpha_after = after[0, 0, A1:A2+1]/after[0, 0, RHO]
    np.testing.assert_allclose(
        alpha_after, [1.0, 0.201], rtol=0.0, atol=2.0e-12
    )
    assert diagnostics["one_active_reaction_coordinate_cell_count"] == 0
    assert diagnostics["maximum_accepted_panels_per_cell"] > 0


def test_one_active_invalid_physical_state_remains_terminal(monkeypatch):
    adapted = adapted_case(
        rates=(0.2, 0.1), heats=(0.0, 0.0), alpha=(1.0, 0.2)
    )

    def fail_physical_state(*_args, **_kwargs):
        raise PropagationCandidateNumericalError("invalid clock integrand")

    monkeypatch.setattr(
        adapted.chemistry, "_solve_one_active_reaction_coordinate_cell",
        fail_physical_state,
    )
    with pytest.raises(
            PropagationCandidateNumericalError, match="invalid clock integrand"):
        adapted.chemistry.advance_local(
            adapted.U, adapted.thermo, 1.0e-2, **DEFAULT_CONTROLS
        )


def test_endpoint_diagnostics_do_not_change_a_valid_result():
    adapted = adapted_case(
        rates=(2.0, 1.0), heats=(1000.0, 2000.0),
        alpha=(0.1, 0.2), temperature=300.0,
    )
    before = adapted.U.copy()
    after, diagnostics = adapted.chemistry.advance_local(
        before, adapted.thermo, 0.05, **DEFAULT_CONTROLS
    )
    expected_cell = np.asarray([
        1000.0, 0.0, 0.0, 4270000.0, 200.0, 250.0,
        113.58333333333334, 113.58333333333334, 50.0,
        78.33333333333334, 43.33333333333333, 0.0,
    ])
    expected = np.broadcast_to(expected_cell, before.shape)
    np.testing.assert_array_equal(after, expected)
    np.testing.assert_array_equal(before, adapted.U)
    assert diagnostics["maximum_local_refinement_depth"] == 0
    assert diagnostics["inventory_depleted_fraction"] == 0.0


def test_nonphysical_endpoint_reports_fail_closed_diagnostics(monkeypatch):
    adapted = adapted_case(
        rates=(2.0, 1.0), heats=(0.0, 0.0),
        alpha=(0.1, 0.2), temperature=300.0,
    )
    state = adapted.U.copy()
    state[..., PVA] = 1.0
    original = adapted.chemistry._solve_local_cells

    def overconsume_cell_seven(*args, **kwargs):
        result = original(*args, **kwargs)
        result["alpha"] = result["alpha"].copy()
        result["alpha"][7] = np.asarray(args[0])[7]+np.asarray([0.1, 0.05])
        return result

    monkeypatch.setattr(
        adapted.chemistry, "_solve_local_cells", overconsume_cell_seven
    )
    with pytest.raises(PropagationCandidateNumericalError) as captured:
        adapted.chemistry.advance_local(
            state, adapted.thermo, 0.05, **DEFAULT_CONTROLS
        )

    message = str(captured.value)
    for field in (
        "nonfinite_state_count", "nonfinite_temperature_count",
        "alpha1_min", "alpha1_max", "alpha2_min", "alpha2_max",
        "cation_min", "anion_min", "pva_min",
        "final_temperature_min_K", "final_temperature_max_K",
        "thermo_tmin_K", "thermo_tmax_K",
        "maximum_alpha_bound_excess",
        "maximum_negative_inventory_violation",
        "offending_flattened_cell_indices",
    ):
        assert f"{field}=" in message
    assert "nonfinite_state_count=0" in message
    assert "nonfinite_temperature_count=0" in message
    assert "'pva_below_negative_tolerance': 7" in message


def test_rate_threshold_is_diagnostic_only():
    adapted = adapted_case(rates=(2.0e4, 3.0e3), heats=(1000.0, 2000.0))
    chemistry = adapted.chemistry
    chemistry.maximum_rate = 1000.0
    low_threshold_state, low = chemistry.advance_local(
        adapted.U, adapted.thermo, 1.0e-5, **DEFAULT_CONTROLS
    )
    chemistry.maximum_rate = 1.0e9
    high_threshold_state, high = chemistry.advance_local(
        adapted.U, adapted.thermo, 1.0e-5, **DEFAULT_CONTROLS
    )
    np.testing.assert_array_equal(low_threshold_state, high_threshold_state)
    assert low["chemical_rate_threshold_diagnostic_only"] is True
    assert low["chemical_rate_threshold_exceedance_fraction"] == 1.0
    assert high["chemical_rate_threshold_exceedance_fraction"] == 0.0


def test_refinement_is_cell_local_and_reports_feedback_correction():
    gas_constant = 8.31446261815324

    def change(_handoff, _propagation, bc, _reactive):
        for channel in bc["kinetics"]["channels"]:
            channel.update(
                activation_energy_J_per_mol=[150000.0, 150000.0],
                ln_Af_per_s=[25.0, 25.0],
                heat_release_J_per_kg=8.0e5,
            )

    adapted = adapted_case(alpha=(0.2, 0.2), temperature=450.0, change=change)
    primitive = adapted.thermo.primitive(adapted.U)
    primitive[2, 2, 3] = 1000.0
    state = adapted.thermo.conservative(primitive)
    after, diagnostics = adapted.chemistry.advance_local(
        state, adapted.thermo, 1.0e-4,
        **{**DEFAULT_CONTROLS, "maximum_local_refinements": 16},
    )
    assert np.isfinite(after).all()
    assert 0.0 < diagnostics["local_refined_cell_fraction"] < 1.0
    assert diagnostics["maximum_local_refinement_depth"] > 0
    assert diagnostics["maximum_temperature_feedback_alpha_correction"] > 0.0
    assert diagnostics["maximum_temperature_feedback_endpoint_temperature_correction_K"] > 0.0
    assert diagnostics["maximum_normalized_embedded_residual"] <= 1.0


@pytest.mark.parametrize("maximum_refinements", [0, 21])
def test_invalid_refinement_budget_fails_closed(maximum_refinements):
    adapted = adapted_case()
    with pytest.raises(PropagationConfigurationError):
        adapted.chemistry.advance_local(
            adapted.U, adapted.thermo, 1.0e-3,
            **{**DEFAULT_CONTROLS,
               "maximum_local_refinements": maximum_refinements},
        )


def test_material_negative_local_increment_is_not_accuracy_clamped(
        monkeypatch):
    adapted = adapted_case()
    original = adapted.chemistry._solve_local_cells

    def injected(*args, **kwargs):
        result = original(*args, **kwargs)
        result["alpha"] = np.asarray(args[0])-1.0e-11
        return result

    monkeypatch.setattr(adapted.chemistry, "_solve_local_cells", injected)
    with pytest.raises(
            PropagationCandidateNumericalError,
            match="invalid conversion"):
        adapted.chemistry.advance_local(
            adapted.U, adapted.thermo, 1.0e-3, **DEFAULT_CONTROLS
        )


def test_nonfinite_state_fails_closed_without_legacy_fallback():
    adapted = adapted_case()
    state = copy.deepcopy(adapted.U)
    state[0, 0, ENERGY] = np.nan
    with pytest.raises(PropagationCandidateNumericalError):
        adapted.chemistry.advance_local(
            state, adapted.thermo, 1.0e-3, **DEFAULT_CONTROLS
        )
