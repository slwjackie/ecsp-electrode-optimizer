from __future__ import annotations

import math

import numpy as np
import pytest

from ecsp_reactive.core import (
    REACTION_PROGRESS,
    TOTAL_ENERGY,
    conservative_to_primitive,
    flux_x,
    flux_y,
    physical_state_mask,
    primitive_to_conservative,
)
from ecsp_reactive.eos import IdealGasEOS, TaitEOS
from ecsp_reactive.numerics import (
    NUMERICAL_PROVENANCE,
    cfl_time_step,
    finite_volume_rhs,
    hll_flux,
    interface_fluxes,
    ssprk3_step,
    weno5_js_reconstruct,
)
from ecsp_reactive.provenance import ReactiveConfigurationError, ReactiveNumericalError
from ecsp_reactive.reaction import ArrheniusReaction, arrhenius_rate, reaction_source


@pytest.fixture
def ideal_gas() -> IdealGasEOS:
    # Verification-only nondimensional-style gas with R=1 in coherent SI
    # units.  The ideal-gas model is not attributed to the ECSP paper.
    return IdealGasEOS(gamma=1.4, gas_constant_J_per_kg_K=1.0)


@pytest.fixture
def tait() -> TaitEOS:
    return TaitEOS(
        rho0_kg_per_m3=1000.0,
        B_Pa=3.0e8,
        N=7.0,
        A_Pa=101325.0,
        cv_J_per_kg_K=2000.0,
        T_ref_K=300.0,
        e_ref_J_per_kg=125.0,
    )


def test_tait_eq3_pressure_and_sound_speed_finite_difference(tait: TaitEOS) -> None:
    rho = np.array([1000.0, 1001.0, 1010.0])
    expected_pressure = (
        tait.B_Pa * ((rho / tait.rho0_kg_per_m3) ** tait.N - 1.0) + tait.A_Pa
    )
    expected_c2 = (
        tait.B_Pa
        * tait.N
        / tait.rho0_kg_per_m3
        * (rho / tait.rho0_kg_per_m3) ** (tait.N - 1.0)
    )
    np.testing.assert_allclose(
        tait.pressure_from_density(rho), expected_pressure, rtol=2e-15
    )
    np.testing.assert_allclose(
        np.asarray(tait.sound_speed(rho)) ** 2, expected_c2, rtol=2e-15
    )

    rho_probe = 1000.0
    step = 1.0e-2
    dp_drho = (
        float(tait.pressure_from_density(rho_probe + step))
        - float(tait.pressure_from_density(rho_probe - step))
    ) / (2.0 * step)
    assert dp_drho == pytest.approx(float(tait.sound_speed(rho_probe)) ** 2, rel=2e-9)


def test_tait_caloric_roundtrip_and_cold_energy_identity(tait: TaitEOS) -> None:
    rho = np.array([999.99, 1000.0, 1001.0, 1010.0])
    temperature = np.array([280.0, 300.0, 700.0, 1600.0])
    internal = np.asarray(
        tait.specific_internal_energy_from_temperature(rho, temperature)
    )
    recovered = np.asarray(tait.temperature_from_rho_e(rho, internal))
    np.testing.assert_allclose(recovered, temperature, rtol=0.0, atol=2e-12)
    assert tait.metadata()["caloric_temperature_closure"] == "ASSUMED_NOT_FROM_PAPER"

    # The density-dependent cold energy is the integral required by
    # de/drho=p/rho^2, not an unrelated fitted proxy.
    rho_probe = 1001.0
    step = 1.0e-3
    de_drho = (
        float(tait.cold_specific_energy(rho_probe + step))
        - float(tait.cold_specific_energy(rho_probe - step))
    ) / (2.0 * step)
    expected = float(tait.pressure_from_density(rho_probe)) / rho_probe**2
    assert de_drho == pytest.approx(expected, rel=2e-8)

    # N=1 is mechanically valid in Eq. (3); its cold-energy integral has the
    # analytic logarithmic limit and must not be rejected by the caloric layer.
    linear_tait = TaitEOS(1000.0, 1.0e6, 1.0, 1.0e6, 2000.0, 300.0)
    linear_energy = linear_tait.specific_internal_energy_from_temperature(1001.0, 450.0)
    assert linear_tait.temperature_from_rho_e(1001.0, linear_energy) == pytest.approx(
        450.0
    )

    adjacent_tait = TaitEOS(
        1000.0,
        1.0e6,
        np.nextafter(1.0, 2.0),
        1.0e6,
        2000.0,
        300.0,
    )
    assert adjacent_tait.cold_specific_energy(1001.0) == pytest.approx(
        linear_tait.cold_specific_energy(1001.0), rel=3.0e-13, abs=1.0e-13
    )


def test_tait_rejects_nonpositive_pressure_and_missing_caloric_state(
    tait: TaitEOS,
) -> None:
    with pytest.raises(ReactiveNumericalError, match="non-positive pressure"):
        tait.pressure_from_density(990.0)
    with pytest.raises(ReactiveConfigurationError, match="temperature_K is required"):
        primitive_to_conservative(1000.0, 0.0, 0.0, 101325.0, 0.0, tait)


def test_ideal_gas_primitive_conservative_roundtrip_and_eq1_flux(
    ideal_gas: IdealGasEOS,
) -> None:
    state = primitive_to_conservative(
        density_kg_per_m3=2.0,
        velocity_x_m_per_s=3.0,
        velocity_y_m_per_s=-4.0,
        pressure_Pa=5.0,
        reaction_progress=0.25,
        eos=ideal_gas,
    )
    np.testing.assert_allclose(state, [2.0, 6.0, -8.0, 37.5, 0.5], rtol=0.0, atol=2e-14)
    primitive = conservative_to_primitive(state, ideal_gas)
    assert float(primitive.pressure_Pa) == pytest.approx(5.0)
    assert float(primitive.reaction_progress) == pytest.approx(0.25)
    assert float(primitive.temperature_K) == pytest.approx(2.5)
    np.testing.assert_allclose(
        flux_x(state, ideal_gas), [6.0, 23.0, -24.0, 127.5, 1.5], rtol=0.0, atol=2e-13
    )
    np.testing.assert_allclose(
        flux_y(state, ideal_gas),
        [-8.0, -24.0, 37.0, -170.0, -2.0],
        rtol=0.0,
        atol=2e-13,
    )


def test_conservative_conversion_fails_closed_without_clipping(
    ideal_gas: IdealGasEOS,
) -> None:
    good = primitive_to_conservative(1.0, 0.0, 0.0, 1.0, 0.5, ideal_gas)
    bad_density = good.copy()
    bad_density[0] = -1.0
    bad_progress = good.copy()
    bad_progress[REACTION_PROGRESS] = 1.1
    assert not bool(physical_state_mask(bad_density, ideal_gas))
    assert not bool(physical_state_mask(bad_progress, ideal_gas))
    with pytest.raises(ReactiveNumericalError, match="no floor was applied"):
        conservative_to_primitive(bad_density, ideal_gas)
    with pytest.raises(ReactiveNumericalError, match="no floor was applied"):
        conservative_to_primitive(bad_progress, ideal_gas)


def test_weno5_constant_preservation_and_smooth_fifth_order_convergence() -> None:
    constant = np.full((17, 3), [2.0, -1.0, 8.5], dtype=np.float64)
    left, right = weno5_js_reconstruct(constant, boundary="periodic")
    expected_constant_faces = np.repeat(constant[:1], constant.shape[0] + 1, axis=0)
    np.testing.assert_allclose(left, expected_constant_faces, rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(right, expected_constant_faces, rtol=0.0, atol=2e-15)

    errors = []
    for count in (20, 40, 80):
        dx = 1.0 / count
        center = (np.arange(count) + 0.5) * dx
        wave_number = 2.0 * np.pi
        # Exact finite-volume cell average of sin(2*pi*x).
        average = (
            np.cos(wave_number * (center - 0.5 * dx))
            - np.cos(wave_number * (center + 0.5 * dx))
        ) / (wave_number * dx)
        reconstructed, _ = weno5_js_reconstruct(average[:, None], boundary="periodic")
        exact = np.sin(wave_number * np.arange(count + 1) * dx)
        errors.append(float(np.mean(np.abs(reconstructed[:, 0] - exact))))
    observed_orders = [math.log(errors[i] / errors[i + 1], 2.0) for i in range(2)]
    assert min(observed_orders) > 4.5, (errors, observed_orders)
    assert (
        NUMERICAL_PROVENANCE["reconstruction"]["provenance"] == "ASSUMED_NOT_FROM_PAPER"
    )


def test_weno_weights_are_invariant_to_component_unit_rescaling() -> None:
    count = 37
    x = (np.arange(count, dtype=np.float64) + 0.5) / count
    waveform = (
        0.25
        + np.sin(2.0 * np.pi * x)
        + 0.15 * np.cos(6.0 * np.pi * x)
    )[:, None]
    reference_left, reference_right = weno5_js_reconstruct(
        waveform, boundary="periodic"
    )
    for factor in (1.0e-12, 1.0e12):
        scaled_left, scaled_right = weno5_js_reconstruct(
            factor * waveform, boundary="periodic"
        )
        np.testing.assert_allclose(
            scaled_left / factor, reference_left, rtol=3.0e-14, atol=3.0e-14
        )
        np.testing.assert_allclose(
            scaled_right / factor, reference_right, rtol=3.0e-14, atol=3.0e-14
        )


def test_weno_face_is_independent_of_values_outside_its_local_stencil() -> None:
    count = 60
    x = (np.arange(count, dtype=np.float64) + 0.5) / count
    field = np.column_stack(
        (
            np.sin(2.0 * np.pi * x),
            3.0 + np.cos(4.0 * np.pi * x),
        )
    )
    reference_left, reference_right = weno5_js_reconstruct(
        field, boundary="periodic"
    )
    changed = field.copy()
    changed[59, 0] = 1.0e12
    changed[59, 1] = -1.0e12
    changed_left, changed_right = weno5_js_reconstruct(
        changed, boundary="periodic"
    )
    for face in (10, 20, 30):
        np.testing.assert_array_equal(changed_left[face], reference_left[face])
        np.testing.assert_array_equal(changed_right[face], reference_right[face])


def test_weno_left_trace_does_not_use_opposite_side_sixth_cell() -> None:
    field = np.zeros((20, 1), dtype=np.float64)
    field[5:10, 0] = [0.0, 0.0, 1.0, 1.0, 1.0]
    reference_left, _ = weno5_js_reconstruct(field, boundary="periodic")
    field[10, 0] = 1.0e12  # outside the formal left stencil of face 8
    changed_left, _ = weno5_js_reconstruct(field, boundary="periodic")
    np.testing.assert_array_equal(changed_left[8], reference_left[8])


@pytest.mark.parametrize("epsilon", [1.0e-300, 1.0e160])
def test_weno_extreme_positive_epsilon_has_finite_normalized_weights(
    epsilon: float,
) -> None:
    values = np.ones((8, 1), dtype=np.float64)
    left, right = weno5_js_reconstruct(
        values, boundary="periodic", epsilon=epsilon
    )
    np.testing.assert_array_equal(left, np.ones((9, 1)))
    np.testing.assert_array_equal(right, np.ones((9, 1)))


def test_tait_finite_volume_rhs_is_reference_energy_gauge_covariant() -> None:
    count = 16
    x = (np.arange(count, dtype=np.float64) + 0.5) / count
    rho = 1000.0 + 0.15 * np.sin(2.0 * np.pi * x)
    velocity_x = 0.5 + 0.1 * np.cos(2.0 * np.pi * x)
    velocity_y = 0.03 * np.sin(4.0 * np.pi * x)
    temperature = 500.0 + 20.0 * np.cos(2.0 * np.pi * x)
    progress = 0.4 + 0.05 * np.sin(2.0 * np.pi * x)
    gauge = 1.0e6
    eos_zero = TaitEOS(1000.0, 3.0e8, 7.0, 3.0e8, 2000.0, 300.0, 0.0)
    eos_shifted = TaitEOS(
        1000.0, 3.0e8, 7.0, 3.0e8, 2000.0, 300.0, gauge
    )
    state_zero = primitive_to_conservative(
        rho,
        velocity_x,
        velocity_y,
        pressure_Pa=None,
        reaction_progress=progress,
        eos=eos_zero,
        temperature_K=temperature,
    )
    state_shifted = primitive_to_conservative(
        rho,
        velocity_x,
        velocity_y,
        pressure_Pa=None,
        reaction_progress=progress,
        eos=eos_shifted,
        temperature_K=temperature,
    )
    rhs_zero, diagnostics_zero = finite_volume_rhs(
        state_zero, 1.0 / count, eos_zero, boundary="periodic"
    )
    rhs_shifted, diagnostics_shifted = finite_volume_rhs(
        state_shifted, 1.0 / count, eos_shifted, boundary="periodic"
    )
    assert diagnostics_zero.positivity_face_fallback_count == 0
    assert diagnostics_shifted.positivity_face_fallback_count == 0
    residual = (
        rhs_shifted[..., TOTAL_ENERGY]
        - rhs_zero[..., TOTAL_ENERGY]
        - gauge * rhs_zero[..., 0]
    )
    scale = np.max(np.abs(rhs_shifted[..., TOTAL_ENERGY]))
    # The final subtraction itself loses several ulps because the gauge term is
    # O(1e6) times the mass residual; this is a roundoff-scaled covariance test.
    assert np.max(np.abs(residual)) <= 1.0e-11 * scale


def test_hll_consistency_and_face_positivity_fallback_is_measured(
    ideal_gas: IdealGasEOS,
) -> None:
    state = primitive_to_conservative(1.2, 2.0, -0.5, 3.0, 0.4, ideal_gas)
    np.testing.assert_allclose(
        hll_flux(state, state, ideal_gas, direction="x"), flux_x(state, ideal_gas)
    )

    # A deterministic, extreme but valid field whose component-wise WENO
    # reconstruction has nonphysical face combinations.  Local first-order
    # states must replace only those faces and the activation must be counted.
    rho = np.array(
        [
            3.26508884e-3,
            2.63550012e-2,
            6.42165236e1,
            3.11151727,
            3.67089408e-3,
            3.96973614e-1,
            7.48699924e-1,
            9.08727117e-3,
        ]
    )
    pressure = np.array(
        [
            7.52691846e1,
            8.11663767e-4,
            1.34842105e-1,
            1.36119850,
            2.78627619e-1,
            4.94753154,
            7.99286170e1,
            4.46825591e3,
        ]
    )
    velocity = np.array(
        [
            -8.63195345,
            5.94188828,
            7.84863987,
            -8.29117004,
            -19.94039666,
            18.93841099,
            -8.06395108,
            -7.44055992,
        ]
    )
    progress = np.array(
        [
            0.89171107,
            0.58516294,
            0.47130967,
            0.77327701,
            0.03034601,
            0.70696510,
            0.37424383,
            0.09085271,
        ]
    )
    field = primitive_to_conservative(rho, velocity, 0.0, pressure, progress, ideal_gas)
    numerical_flux, diagnostics = interface_fluxes(
        field, ideal_gas, boundary="periodic"
    )
    assert np.isfinite(numerical_flux).all()
    assert diagnostics.positivity_face_fallback_count > 0
    assert 0.0 < diagnostics.positivity_face_fallback_fraction <= 1.0
    assert diagnostics.positivity_face_fallback_max_abs_state_correction > 0.0


def test_ssprk3_third_order_for_nonautonomous_signature() -> None:
    errors = []
    for steps in (10, 20, 40):
        state = np.array([1.0], dtype=np.float64)
        time = 0.0
        dt = 1.0 / steps
        for _ in range(steps):
            state = ssprk3_step(state, time, dt, lambda _time, value: -value)
            time += dt
        errors.append(abs(float(state[0]) - math.exp(-1.0)))
    orders = [math.log(errors[i] / errors[i + 1], 2.0) for i in range(2)]
    assert min(orders) > 2.9, (errors, orders)


def test_arrhenius_rate_and_reaction_heat_progress_ratio(
    ideal_gas: IdealGasEOS,
) -> None:
    model = ArrheniusReaction(
        pre_exponential_s_inv=2.0e3,
        activation_energy_J_per_mol=4.0e4,
        reaction_order=1.5,
        heat_release_J_per_kg=3.0e6,
    )
    temperature = np.array([900.0, 1200.0])
    progress = np.array([0.2, 0.7])
    expected = (
        2.0e3
        * (1.0 - progress) ** 1.5
        * np.exp(-4.0e4 / (8.31446261815324 * temperature))
    )
    np.testing.assert_allclose(
        arrhenius_rate(temperature, progress, model), expected, rtol=2e-15
    )
    state = primitive_to_conservative(
        [1.0, 0.8], 0.0, 0.0, [1.0, 0.9], progress, ideal_gas
    )
    source = reaction_source(state, temperature, model)
    nonzero = source[..., REACTION_PROGRESS] > 0.0
    np.testing.assert_allclose(
        source[..., TOTAL_ENERGY][nonzero] / source[..., REACTION_PROGRESS][nonzero],
        model.heat_release_J_per_kg,
        rtol=2e-15,
    )
    assert model.metadata()["progress_rate_closure"] == "ASSUMED_NOT_FROM_PAPER"


def test_arrhenius_uses_explicit_gas_constant_and_signed_heat(
    ideal_gas: IdealGasEOS,
) -> None:
    common = dict(
        pre_exponential_s_inv=2.0,
        activation_energy_J_per_mol=1000.0,
        reaction_order=0.0,
        heat_release_J_per_kg=-25.0,
    )
    low_r = ArrheniusReaction(**common, gas_constant_J_per_mol_K=4.0)
    high_r = ArrheniusReaction(**common, gas_constant_J_per_mol_K=8.0)
    assert arrhenius_rate(300.0, 0.2, high_r) > arrhenius_rate(300.0, 0.2, low_r)
    state = primitive_to_conservative(1.0, 0.0, 0.0, 1.0, 0.2, ideal_gas)
    source = reaction_source(state, 300.0, high_r)
    assert source[TOTAL_ENERGY] / source[REACTION_PROGRESS] == pytest.approx(-25.0)
    assert arrhenius_rate(300.0, 1.0, high_r) == 0.0


def test_arrhenius_completion_tolerance_zeros_rate_without_changing_state(
    ideal_gas: IdealGasEOS,
) -> None:
    tolerance = 1.0e-6
    progress = 1.0 - 0.5 * tolerance
    model = ArrheniusReaction(2.0, 0.0, 0.0, 75.0)
    assert arrhenius_rate(300.0, progress, model) == pytest.approx(2.0)
    assert (
        arrhenius_rate(
            300.0,
            progress,
            model,
            completion_tolerance=tolerance,
        )
        == 0.0
    )
    state = primitive_to_conservative(
        1.0, 0.0, 0.0, 1.0, progress, ideal_gas
    )
    source = reaction_source(
        state,
        300.0,
        model,
        completion_tolerance=tolerance,
    )
    np.testing.assert_array_equal(source, 0.0)
    assert state[REACTION_PROGRESS] == pytest.approx(progress)
    with pytest.raises(ReactiveConfigurationError, match="completion_tolerance"):
        arrhenius_rate(
            300.0,
            progress,
            model,
            completion_tolerance=-1.0,
        )


@pytest.mark.parametrize(
    ("temperature", "progress"),
    [(0.0, 0.2), (float("nan"), 0.2), (1000.0, -0.1), (1000.0, 1.1)],
)
def test_arrhenius_rejects_invalid_inputs(temperature: float, progress: float) -> None:
    model = ArrheniusReaction(1.0, 0.0, 1.0, 0.0)
    with pytest.raises(ReactiveNumericalError):
        arrhenius_rate(temperature, progress, model)


def test_uniform_state_has_zero_periodic_flux_divergence(
    ideal_gas: IdealGasEOS,
) -> None:
    one = primitive_to_conservative(1.2, 3.0, -2.0, 2.5, 0.3, ideal_gas)
    field = np.repeat(one[None, :], 32, axis=0)
    rhs, diagnostics = finite_volume_rhs(field, 0.01, ideal_gas, boundary="periodic")
    np.testing.assert_allclose(rhs, 0.0, rtol=0.0, atol=0.0)
    assert diagnostics.positivity_face_fallback_count == 0


def test_periodic_sod_shock_tube_remains_positive_and_conservative(
    ideal_gas: IdealGasEOS,
) -> None:
    count = 120
    dx = 1.0 / count
    center = (np.arange(count) + 0.5) * dx
    high_state = (center >= 0.25) & (center < 0.75)
    density = np.where(high_state, 1.0, 0.125)
    pressure = np.where(high_state, 1.0, 0.1)
    state = primitive_to_conservative(density, 0.0, 0.0, pressure, 0.0, ideal_gas)
    initial_integral = np.sum(state, axis=0) * dx
    time = 0.0
    final_time = 0.04
    fallback_calls = 0

    while time < final_time:
        dt = min(cfl_time_step(state, ideal_gas, dx, cfl=0.2), final_time - time)

        def rhs(_time: float, current: np.ndarray) -> np.ndarray:
            nonlocal fallback_calls
            value, diagnostics = finite_volume_rhs(
                current, dx, ideal_gas, boundary="periodic"
            )
            fallback_calls += diagnostics.positivity_face_fallback_count
            return value

        state = ssprk3_step(
            state,
            time,
            dt,
            rhs,
            validator=lambda current: conservative_to_primitive(current, ideal_gas),
        )
        time += dt

    primitive = conservative_to_primitive(state, ideal_gas)
    assert np.min(primitive.density_kg_per_m3) > 0.0
    assert np.min(primitive.pressure_Pa) > 0.0
    assert np.min(primitive.temperature_K) > 0.0
    assert (
        fallback_calls >= 0
    )  # metric is always available, even when no fallback is needed.
    final_integral = np.sum(state, axis=0) * dx
    np.testing.assert_allclose(final_integral, initial_integral, rtol=0.0, atol=8e-14)


def test_explicit_reflective_boundary_has_zero_normal_mass_and_energy_flux(
    ideal_gas: IdealGasEOS,
) -> None:
    state = primitive_to_conservative(1.0, 2.0, 0.0, 1.0, 0.3, ideal_gas)
    field = np.repeat(state[None, :], 8, axis=0)
    face_flux, diagnostics = interface_fluxes(field, ideal_gas, boundary="reflective")
    assert diagnostics.boundary_type == "reflective"
    assert face_flux[0, 0] == pytest.approx(0.0, abs=2e-14)
    assert face_flux[-1, 0] == pytest.approx(0.0, abs=2e-14)
    assert face_flux[0, TOTAL_ENERGY] == pytest.approx(0.0, abs=2e-14)
    assert face_flux[-1, TOTAL_ENERGY] == pytest.approx(0.0, abs=2e-14)
