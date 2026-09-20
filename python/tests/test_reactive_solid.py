from __future__ import annotations

import numpy as np
import pytest

from ecsp_reactive.solid import (
    SolidThermalModel,
    advance_solid_ssprk3,
    laplacian_2d,
    solid_arrhenius_rate,
    solid_stable_time_step,
    solid_thermal_rhs,
)


def model(**changes: float) -> SolidThermalModel:
    values = {
        "density_kg_per_m3": 1000.0,
        "heat_capacity_J_per_kgK": 1000.0,
        "conductivity_W_per_mK": 0.0,
        "preexponential_per_s": 2.0,
        "activation_energy_J_per_mol": 0.0,
        "gas_constant_J_per_molK": 8.314,
        "heat_of_decomposition_J_per_kg": -100.0,
        "reaction_order": 0.0,
    }
    values.update(changes)
    return SolidThermalModel(**values)


def test_periodic_laplacian_conserves_integral() -> None:
    rng = np.random.default_rng(7)
    field = rng.normal(size=(17, 19))
    result = laplacian_2d(field, 0.2, 0.3, "PERIODIC")
    assert float(np.sum(result)) == pytest.approx(0.0, abs=2.0e-12)


def test_pure_periodic_heat_conduction_matches_sinusoidal_analytic_solution() -> None:
    cells = 32
    spacing = 1.0 / cells
    diffusivity = 0.01
    thermal = model(
        density_kg_per_m3=1.0,
        heat_capacity_J_per_kgK=1.0,
        conductivity_W_per_mK=diffusivity,
        preexponential_per_s=0.0,
        heat_of_decomposition_J_per_kg=0.0,
        reaction_order=1.0,
    )
    x = (np.arange(cells, dtype=np.float64) + 0.5) * spacing
    initial = np.broadcast_to(300.0 + np.sin(2.0 * np.pi * x), (cells, cells)).copy()
    alpha = np.zeros_like(initial)
    end_time = 0.01
    steps = 20
    dt = end_time / steps

    def rhs(t: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        temperature_rate, alpha_rate, _ = solid_thermal_rhs(
            t,
            a,
            thermal,
            spacing,
            spacing,
            0.0,
            boundary="PERIODIC",
        )
        return temperature_rate, alpha_rate

    temperature = initial
    for _ in range(steps):
        temperature, alpha = advance_solid_ssprk3(
            temperature,
            alpha,
            dt,
            rhs,
            minimum_temperature_K=1.0,
        )
    exact = np.broadcast_to(
        300.0
        + np.exp(-diffusivity * (2.0 * np.pi) ** 2 * end_time)
        * np.sin(2.0 * np.pi * x),
        (cells, cells),
    )
    np.testing.assert_allclose(temperature, exact, rtol=0.0, atol=1.3e-5)
    np.testing.assert_array_equal(alpha, 0.0)


def test_arrhenius_constant_rate_and_reaction_heat_are_consistent() -> None:
    temperature = np.full((8, 9), 300.0)
    alpha = np.zeros_like(temperature)
    rate = solid_arrhenius_rate(temperature, alpha, model())
    np.testing.assert_array_equal(rate, 2.0)
    dtemperature, dalpha, diagnostics = solid_thermal_rhs(
        temperature, alpha, model(), 1.0, 1.0, 0.0
    )
    np.testing.assert_array_equal(dalpha, 2.0)
    # -rho*Qr*r / (rho*cp) = -Qr*r/cp = 0.2 K/s.
    np.testing.assert_allclose(dtemperature, 0.2)
    assert diagnostics["reaction_heat_integral_W_per_m"] == pytest.approx(14_400_000.0)


def test_solid_completion_tolerance_zeros_rate_without_changing_alpha() -> None:
    tolerance = 1.0e-6
    alpha = np.full((8, 8), 1.0 - 0.5 * tolerance)
    temperature = np.full_like(alpha, 300.0)
    baseline = solid_arrhenius_rate(
        temperature, alpha, model(), progress_tolerance=0.0
    )
    terminal = solid_arrhenius_rate(
        temperature, alpha, model(), progress_tolerance=tolerance
    )
    np.testing.assert_array_equal(baseline, 2.0)
    np.testing.assert_array_equal(terminal, 0.0)
    np.testing.assert_array_equal(alpha, 1.0 - 0.5 * tolerance)


def test_ssprk3_integrates_constant_alpha_rate_without_clipping() -> None:
    temperature = np.full((8, 8), 300.0)
    alpha = np.zeros_like(temperature)

    def rhs(t: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros_like(t), np.full_like(a, 2.0)

    final_t, final_a = advance_solid_ssprk3(
        temperature,
        alpha,
        0.1,
        rhs,
        minimum_temperature_K=1.0,
    )
    np.testing.assert_allclose(final_t, temperature)
    np.testing.assert_allclose(final_a, 0.2)


def test_ssprk3_rejects_alpha_overshoot_instead_of_clipping() -> None:
    temperature = np.full((8, 8), 300.0)
    alpha = np.full_like(temperature, 0.99)

    def rhs(t: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros_like(t), np.full_like(a, 2.0)

    with pytest.raises(RuntimeError, match=r"left \[0,1\]"):
        advance_solid_ssprk3(
            temperature,
            alpha,
            0.1,
            rhs,
            minimum_temperature_K=1.0,
        )


def test_solid_rate_does_not_hide_roundoff_overshoot_with_clipping() -> None:
    temperature = np.full((8, 8), 300.0)
    alpha = np.full_like(temperature, 1.0 + 1.0e-14)
    with pytest.raises(RuntimeError, match=r"left \[0,1\]"):
        solid_arrhenius_rate(temperature, alpha, model(), progress_tolerance=1.0e-12)
    np.testing.assert_array_equal(
        solid_arrhenius_rate(temperature, np.ones_like(alpha), model()), 0.0
    )


@pytest.mark.parametrize(
    ("diffusion_safety", "progress_increment"),
    [(0.0, 0.1), (1.01, 0.1), (0.9, 0.0), (0.9, 1.01)],
)
def test_solid_stability_controls_are_validated(
    diffusion_safety: float, progress_increment: float
) -> None:
    with pytest.raises(ValueError):
        solid_stable_time_step(
            model(),
            0.1,
            0.1,
            1.0,
            diffusion_safety=diffusion_safety,
            maximum_progress_increment=progress_increment,
        )
