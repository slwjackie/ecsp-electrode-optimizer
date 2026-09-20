from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp

from ecsp_reactive.condensed.chemistry import (
    A1, A2, ANION, CATION, EC_LP, ENERGY, MX, MY, PRODUCT_WATER, PVA,
    RHO, WATER,
)
from ecsp_reactive.condensed.handoff import BCReactiveHandoffAdapter
from ecsp_reactive.condensed.solver import BCReactiveSolver
from ecsp_reactive.condensed.validation_cases import synthetic_condensed_case


CONTROLS = dict(
    concentration_floor=0.0,
    relative_tolerance=1.0e-7,
    absolute_tolerance=1.0e-10,
    temperature_tolerance_K=1.0e-4,
    maximum_corrector_iterations=8,
    maximum_depletion_iterations=40,
    maximum_local_refinements=16,
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


def test_constant_rates_are_exact_and_completed_state_is_stationary():
    adapted = adapted_case(
        rates=(2.0, 1.0), heats=(1000.0, 2000.0), alpha=(0.1, 0.2)
    )
    before = adapted.U.copy()
    after, diagnostics = adapted.chemistry.advance_local(
        before, adapted.thermo, 0.05, **CONTROLS
    )
    alpha_after = after[..., A1:A2+1]/after[..., RHO, None]
    np.testing.assert_allclose(alpha_after[..., 0], 0.2, atol=2.0e-14)
    np.testing.assert_allclose(alpha_after[..., 1], 0.25, atol=2.0e-14)
    np.testing.assert_array_equal(after[..., RHO], before[..., RHO])
    np.testing.assert_array_equal(after[..., MX:MY+1], before[..., MX:MY+1])
    reservoir_before = (
        before[..., ENERGY]-before[..., A1:A2+1] @ adapted.chemistry.Q
    )
    reservoir_after = (
        after[..., ENERGY]-after[..., A1:A2+1] @ adapted.chemistry.Q
    )
    np.testing.assert_allclose(
        reservoir_after, reservoir_before, rtol=0.0, atol=2.0e-9
    )
    assert diagnostics["maximum_local_refinement_depth"] == 0

    completed = adapted_case(
        rates=(2.0e4, 3.0e4), heats=(8.0e5, 9.0e5), alpha=(1.0, 1.0)
    )
    stationary, completed_diagnostics = completed.chemistry.advance_local(
        completed.U, completed.thermo, 0.01, **CONTROLS
    )
    np.testing.assert_array_equal(stationary, completed.U)
    assert completed_diagnostics["endpoint_evaluation_count"] == 0


@pytest.mark.parametrize("limiter", ["lp", "pva"])
def test_lp_and_pva_depletion_events_are_independently_enforced(limiter):
    adapted = adapted_case(rates=(2.0, 1.0), heats=(1000.0, 2000.0))
    state = adapted.U.copy()
    state[..., CATION] = state[..., ANION] = 145.0
    state[..., PVA] = 100.0
    if limiter == "lp":
        state[..., CATION] = state[..., ANION] = 14.5
    else:
        state[..., PVA] = 10.0
    before = state.copy()
    controls = {
        **CONTROLS,
        "relative_tolerance": 1.0e-10,
        "absolute_tolerance": 1.0e-12,
        "temperature_tolerance_K": 1.0e-7,
        "maximum_depletion_iterations": 50,
    }
    after, diagnostics = adapted.chemistry.advance_local(
        state, adapted.thermo, 0.5, **controls
    )

    delta = ((after[..., A1:A2+1]-before[..., A1:A2+1])
             / before[..., RHO, None])
    np.testing.assert_allclose(delta[..., 0], 0.12, atol=3.0e-10)
    np.testing.assert_allclose(delta[..., 1], 0.06, atol=3.0e-10)
    assert diagnostics["minimum_inventory_depletion_time_fraction"] \
        == pytest.approx(0.12, abs=3.0e-10)
    assert diagnostics["inventory_depleted_fraction"] == 1.0
    assert np.min(after[..., [CATION, ANION, PVA]]) >= 0.0
    if limiter == "lp":
        assert np.max(after[..., [CATION, ANION]]) <= 3.0e-8
        assert np.min(after[..., PVA]) > 80.0
    else:
        assert np.max(after[..., PVA]) <= 3.0e-8
        assert np.min(after[..., [CATION, ANION]]) > 100.0


def test_completed_faster_channel_leaves_headroom_for_shared_event():
    adapted = adapted_case(
        rates=(2.0, 1.0), heats=(1000.0, 2000.0), alpha=(1.0, 0.2)
    )
    state = adapted.U.copy()
    state[..., CATION] = state[..., ANION] = 14.5
    state[..., PVA] = 10.0
    before = state.copy()
    controls = {
        **CONTROLS,
        "relative_tolerance": 1.0e-10,
        "absolute_tolerance": 1.0e-12,
        "temperature_tolerance_K": 1.0e-7,
        "maximum_depletion_iterations": 50,
    }
    after, diagnostics = adapted.chemistry.advance_local(
        state, adapted.thermo, 0.5, **controls
    )
    alpha_after = after[..., A1:A2+1]/after[..., RHO, None]
    np.testing.assert_array_equal(alpha_after[..., 0], 1.0)
    np.testing.assert_allclose(alpha_after[..., 1], 0.5, atol=3.0e-10)
    assert diagnostics["minimum_inventory_depletion_time_fraction"] \
        == pytest.approx(0.6, abs=3.0e-10)
    np.testing.assert_allclose(after[..., CATION], after[..., ANION])
    np.testing.assert_array_equal(after[..., WATER], before[..., WATER])
    np.testing.assert_array_equal(after[..., EC_LP], before[..., EC_LP])
    extent = (before[..., PVA]-after[..., PVA])
    np.testing.assert_allclose(
        after[..., PRODUCT_WATER]-before[..., PRODUCT_WATER], 2.0*extent
    )


def test_self_heating_accelerates_and_tightening_tolerance_converges():
    gas_constant = 8.31446261815324

    def kinetics(heat_release):
        def change(_handoff, _propagation, bc, _reactive):
            for channel in bc["kinetics"]["channels"]:
                channel.update(
                    activation_energy_J_per_mol=[80000.0, 80000.0],
                    ln_Af_per_s=[
                        math.log(2.0)+80000.0/(gas_constant*700.0)
                    ]*2,
                    heat_release_J_per_kg=heat_release,
                )
        return change

    cold = adapted_case(
        alpha=(0.1, 0.1), temperature=700.0, change=kinetics(0.0)
    )
    cold_after, _ = cold.chemistry.advance_local(
        cold.U, cold.thermo, 0.02, **CONTROLS
    )
    cold_alpha = cold_after[0, 0, A1]/cold_after[0, 0, RHO]
    assert cold_alpha == pytest.approx(0.14, abs=2.0e-13)

    hot = adapted_case(
        alpha=(0.1, 0.1), temperature=700.0, change=kinetics(6.0e5)
    )
    alpha0 = np.asarray([0.1, 0.1])
    sensible0 = float(hot.thermo.heat.sensible_energy(700.0))

    def rhs(_time, alpha):
        sensible = sensible0+float((alpha-alpha0) @ hot.chemistry.Q)
        temperature = float(hot.thermo.heat.temperature(sensible))
        return hot.chemistry._rates_from_alpha(alpha[None, :], temperature)[0]

    reference = solve_ivp(
        rhs, (0.0, 0.02), alpha0, method="DOP853",
        rtol=1.0e-13, atol=1.0e-15,
    )
    assert reference.success
    loose_after, _ = hot.chemistry.advance_local(
        hot.U, hot.thermo, 0.02, **{
            **CONTROLS,
            "relative_tolerance": 1.0e-4,
            "absolute_tolerance": 1.0e-7,
            "temperature_tolerance_K": 1.0e-1,
        }
    )
    tight_after, tight_diagnostics = hot.chemistry.advance_local(
        hot.U, hot.thermo, 0.02, **CONTROLS
    )
    loose_alpha = loose_after[0, 0, A1:A2+1]/loose_after[0, 0, RHO]
    tight_alpha = tight_after[0, 0, A1:A2+1]/tight_after[0, 0, RHO]
    loose_error = float(np.max(np.abs(loose_alpha-reference.y[:, -1])))
    tight_error = float(np.max(np.abs(tight_alpha-reference.y[:, -1])))
    assert tight_alpha[0] > cold_alpha
    assert tight_error < 0.1*loose_error
    assert tight_diagnostics["maximum_temperature_rise_K"] > 0.0
    assert tight_diagnostics["maximum_caloric_inverse_residual_J_per_kg"] \
        < 1.0e-7


@pytest.mark.parametrize(
    ("endpoint_kind", "target", "duration_factor"),
    [("interior", 0.85, 1.0),
     ("completion", 1.0, 1.05),
     ("inventory", 0.73, 1.2)],
)
def test_one_active_reaction_coordinate_matches_knot_split_oracle(
        endpoint_kind, target, duration_factor):
    gas_constant = 8.31446261815324
    alpha_grid = np.asarray([0.0, 0.3, 0.55, 0.8, 1.0])
    activation = np.asarray([90000.0, 110000.0, 80000.0,
                             140000.0, 100000.0])
    rates_at_600_K = np.asarray([5.0, 20.0, 3.0, 50.0, 10.0])

    def change(_handoff, _propagation, bc, _reactive):
        bc["thermal"]["heat_capacity"] = {
            "mode": "table",
            "temperature_K": [500.0, 650.0, 720.0, 950.0],
            "values": [1400.0, 1800.0, 2300.0, 2600.0],
        }
        bc["kinetics"]["channels"][1].update(
            alpha_grid=alpha_grid.tolist(),
            activation_energy_J_per_mol=activation.tolist(),
            ln_Af_per_s=(
                np.log(rates_at_600_K)
                + activation/(gas_constant*600.0)
            ).tolist(),
            heat_release_J_per_kg=4.0e5,
        )

    adapted = adapted_case(
        alpha=(1.0, 0.15), temperature=600.0, change=change
    )
    chemistry = adapted.chemistry
    start = 0.15
    sensible0 = float(adapted.thermo.heat.sensible_energy(600.0))
    state = adapted.U.copy()
    state[..., CATION] = state[..., ANION] = 145.0
    state[..., PVA] = 100.0

    caloric_alpha = (
        start
        +(adapted.thermo.heat.sensible_energy(adapted.thermo.heat.grid)
          - sensible0)/chemistry.Q[1]
    )
    oracle_breaks = sorted(set(
        [float(value) for value in alpha_grid[1:-1]
         if start < value < target]
        + [float(value) for value in caloric_alpha
           if start < value < target]
    ))
    assert any(value in oracle_breaks for value in alpha_grid[1:-1])
    assert any(value in oracle_breaks for value in caloric_alpha)

    def inverse_rate(alpha):
        evaluation_alpha = min(float(alpha), math.nextafter(1.0, 0.0))
        sensible = sensible0+chemistry.Q[1]*(evaluation_alpha-start)
        temperature = float(adapted.thermo.heat.temperature(sensible))
        rate = chemistry._rates_from_alpha(
            np.asarray([[1.0, evaluation_alpha]]),
            np.asarray([temperature]),
        )[0, 1]
        return 1.0/float(rate)

    oracle_time = quad(
        inverse_rate, start, target, points=oracle_breaks,
        epsabs=1.0e-15, epsrel=2.0e-13, limit=500,
    )[0]
    if endpoint_kind == "inventory":
        progress_capacity = chemistry.weights[1]*(target-start)
        state[..., PVA] = (
            state[..., RHO]*chemistry.xi_per_kg*progress_capacity
        )
    before = state.copy()
    duration = duration_factor*oracle_time
    after, diagnostics = chemistry.advance_local(
        state, adapted.thermo, duration, **CONTROLS
    )

    alpha_after = after[..., A1:A2+1]/after[..., RHO, None]
    np.testing.assert_array_equal(alpha_after[..., 0], 1.0)
    np.testing.assert_allclose(alpha_after[..., 1], target, atol=2.0e-8)
    expected_temperature = float(adapted.thermo.heat.temperature(
        sensible0+chemistry.Q[1]*(target-start)
    ))
    np.testing.assert_allclose(
        adapted.thermo.primitive(after)[..., 3], expected_temperature,
        atol=8.0e-5,
    )
    assert diagnostics["one_active_reaction_coordinate_cell_count"] == 25
    assert diagnostics[
        "maximum_one_active_reaction_coordinate_normalized_quadrature_residual"
    ] <= 0.125
    if endpoint_kind == "inventory":
        assert diagnostics["inventory_depleted_fraction"] == 1.0
        assert diagnostics["minimum_inventory_depletion_time_fraction"] \
            == pytest.approx(1.0/duration_factor, rel=2.0e-11)
    else:
        assert diagnostics["inventory_depleted_fraction"] == 0.0

    d_rho_alpha = after[..., A1:A2+1]-before[..., A1:A2+1]
    np.testing.assert_allclose(
        after[..., ENERGY]-before[..., ENERGY],
        d_rho_alpha @ chemistry.Q, atol=2.0e-8,
    )
    extent = chemistry.xi_per_kg*(d_rho_alpha @ chemistry.weights)
    np.testing.assert_allclose(
        after[..., CATION]-before[..., CATION], -1.45*extent
    )
    np.testing.assert_allclose(
        after[..., ANION]-before[..., ANION], -1.45*extent
    )
    np.testing.assert_allclose(
        after[..., PVA]-before[..., PVA], -extent
    )
    np.testing.assert_allclose(
        after[..., PRODUCT_WATER]-before[..., PRODUCT_WATER], 2.0*extent
    )
    assert np.min(after[..., [CATION, ANION, PVA]]) >= 0.0


def test_full_strang_split_has_second_order_temporal_convergence():
    gas_constant = 8.31446261815324
    handoff, propagation, bc, reactive = synthetic_condensed_case(
        shape=(5, 5), rates=(1.0, 1.0), heats=(2.0e5, 3.0e5),
        alpha=(0.1, 0.1),
    )
    handoff["temperatureAtOnset_K"][:] = 550.0
    handoff["temperatureAtOnset_K"][2, 2] = 700.0
    for channel, rate, activation in zip(
            bc["kinetics"]["channels"], (15.0, 8.0),
            (50000.0, 70000.0)):
        channel.update(
            activation_energy_J_per_mol=[activation, activation],
            ln_Af_per_s=[
                math.log(rate)+activation/(gas_constant*600.0)
            ]*2,
        )
    # A synthetic conductivity makes the chemistry/conduction commutator
    # measurable on a short, inexpensive fixed spatial grid.
    bc["thermal"]["thermal_conductivity"] = 1000.0
    reactive.update(
        chemistry_integration_mode="local_adaptive_thermochemical",
        chemistry_relative_tolerance=1.0e-10,
        chemistry_absolute_tolerance=1.0e-13,
        chemistry_temperature_tolerance_K=1.0e-7,
        maximum_chemistry_corrector_iterations=12,
        maximum_chemistry_depletion_iterations=60,
        maximum_chemistry_local_refinements=18,
        boundary="reflective",
        riemann_solver="hllc",
        weno_epsilon=1.0e-6,
        cfl=0.35,
        thermal_cfl=0.7,
        stationary_mechanics_fast_path=True,
        maximum_time_steps=100000,
        maximum_step_retries=12,
        minimum_time_step_s=1.0e-14,
        mass_budget_relative_tolerance=1.0e-10,
        energy_budget_relative_tolerance=1.0e-9,
    )
    adapted = BCReactiveHandoffAdapter(
        propagation, bc, reactive
    ).adapt(handoff)
    horizon = 0.004

    def integrate(time_step):
        solver = BCReactiveSolver(
            adapted, propagation, bc, reactive,
            initialize_electrical=False,
        )
        state = adapted.U.copy()
        time = 0.0
        for _ in range(round(horizon/time_step)):
            state = solver.advance(time, state, time_step)[0]
            time += time_step
        return state

    coarse = integrate(1.0e-3)
    fine = integrate(5.0e-4)
    reference = integrate(1.25e-4)
    primitive_coarse = adapted.thermo.primitive(coarse)
    primitive_fine = adapted.thermo.primitive(fine)
    primitive_reference = adapted.thermo.primitive(reference)
    coarse_temperature_error = float(np.max(np.abs(
        primitive_coarse[..., 3]-primitive_reference[..., 3]
    )))
    fine_temperature_error = float(np.max(np.abs(
        primitive_fine[..., 3]-primitive_reference[..., 3]
    )))
    coarse_alpha_error = float(np.max(np.abs(
        primitive_coarse[..., A1:A2+1]-primitive_reference[..., A1:A2+1]
    )))
    fine_alpha_error = float(np.max(np.abs(
        primitive_fine[..., A1:A2+1]-primitive_reference[..., A1:A2+1]
    )))
    observed_temperature_order = math.log2(
        coarse_temperature_error/fine_temperature_error
    )
    observed_alpha_order = math.log2(coarse_alpha_error/fine_alpha_error)
    assert observed_temperature_order > 1.8
    assert observed_alpha_order > 1.8

    # Reflective conduction redistributes sensible energy; chemistry transfers
    # chemical reservoir energy into the same conserved total-energy field.
    initial_reservoir = np.sum(
        adapted.U[..., ENERGY]
        - adapted.U[..., A1:A2+1] @ adapted.chemistry.Q
    )
    final_reservoir = np.sum(
        reference[..., ENERGY]
        - reference[..., A1:A2+1] @ adapted.chemistry.Q
    )
    assert final_reservoir == pytest.approx(
        initial_reservoir, rel=2.0e-14, abs=2.0e-5
    )
    np.testing.assert_allclose(reference[..., CATION], reference[..., ANION])
    assert np.min(reference[..., [CATION, ANION, PVA]]) >= 0.0
