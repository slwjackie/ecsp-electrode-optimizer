from __future__ import annotations

import numpy as np
import pytest

from ecsp_reactive.electrical import (
    ElectricalSourceFields,
    SurfaceContactGeometry,
    canonical_bool_mask,
    joule_heating_sigma_e2,
    surface_heat_flux_to_volume,
)
from ecsp_reactive.provenance import (
    AssumptionRegistry,
    ReactiveConfigurationError,
    validate_model_mode,
)


def assumed(value: float, unit: str) -> dict[str, object]:
    return {
        "value": value,
        "unit": unit,
        "provenance": "ASSUMED_NOT_FROM_PAPER",
        "source": "synthetic unit-test input; not a paper value",
    }


def test_strict_paper_registry_fails_closed_on_assumed_quantity() -> None:
    registry = AssumptionRegistry(strict_paper=True)
    with pytest.raises(ReactiveConfigurationError, match="refuses assumed input"):
        registry.quantity("rho0", assumed(1000.0, "kg/m^3"), "kg/m^3", positive=True)


def test_assumption_registry_reports_runnable_nonpaper_input() -> None:
    registry = AssumptionRegistry(strict_paper=False)
    assert registry.quantity("rho0", assumed(1000.0, "kg/m^3"), "kg/m^3") == 1000.0
    registry.algorithm(
        "riemann_solver",
        {
            "choice": "HLL",
            "provenance": "ASSUMED_NOT_FROM_PAPER",
            "source": "paper does not identify a Riemann solver",
        },
    )
    report = registry.report()
    assert report["assumed_quantity_count"] == 1
    assert report["assumed_algorithm_count"] == 1


def test_model_modes_are_explicit() -> None:
    assert validate_model_mode("paper_faithful") == "paper_faithful"
    assert validate_model_mode("paper_reproduction") == "paper_faithful"
    assert validate_model_mode("ecsp_extended") == "ecsp_extended"
    with pytest.raises(ReactiveConfigurationError):
        validate_model_mode("paper_faithful_but_with_hidden_bv")


def test_noncanonical_ff_bool_is_canonicalised_to_raw_zero_one() -> None:
    raw = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    noncanonical = raw.view(np.bool_)
    assert set(noncanonical.view(np.uint8).ravel()) == {0, 255}
    result = canonical_bool_mask(noncanonical, "mask")
    assert result.flags.c_contiguous
    assert result.dtype == np.bool_
    assert set(result.view(np.uint8).ravel()) == {0, 1}


def test_numeric_mask_truth_is_evaluated_before_uint8_canonicalisation() -> None:
    raw = np.array([[0.0, 0.5, -0.5], [256.0, -256.0, 1.0]])
    result = canonical_bool_mask(raw, "mask")
    np.testing.assert_array_equal(
        result,
        [[False, True, True], [True, True, True]],
    )
    assert set(result.view(np.uint8).ravel()) == {0, 1}


@pytest.mark.parametrize(
    "raw",
    [
        np.array([[0.0, np.nan], [1.0, 0.0]]),
        np.array([[0.0, np.inf], [1.0, 0.0]]),
        np.array([[0.0 + 0.0j, 1.0 + 1.0j], [1.0, 0.0]]),
    ],
)
def test_nonfinite_or_complex_mask_is_rejected(raw: np.ndarray) -> None:
    with pytest.raises(ReactiveConfigurationError):
        canonical_bool_mask(raw, "mask")


def test_surface_contacts_are_overlays_on_full_propellant_domain() -> None:
    anode = np.zeros((6, 7), dtype=np.uint8)
    cathode = np.zeros_like(anode)
    anode[1:5, 1:3] = 255
    cathode[1:5, 4:6] = 255
    geometry = SurfaceContactGeometry.build(anode.view(np.bool_), cathode.view(np.bool_))
    assert geometry.propellant.all()
    assert geometry.propellant.sum() == 42
    assert geometry.anode_contact.sum() == 8
    assert geometry.cathode_contact.sum() == 8
    assert not np.any(geometry.anode_contact & geometry.cathode_contact)
    for mask in (
        geometry.propellant,
        geometry.anode_contact,
        geometry.cathode_contact,
    ):
        with pytest.raises(ValueError):
            mask.setflags(write=True)


def test_uniform_electric_field_has_exact_sigma_e_squared_heating() -> None:
    ny, nx = 7, 9
    dx, dy = 2.0e-4, 3.0e-4
    electric_field = 2.5e4
    x = np.arange(nx, dtype=np.float64) * dx
    potential = np.broadcast_to(-electric_field * x, (ny, nx)).copy()
    heat, diagnostics = joule_heating_sigma_e2(
        potential,
        3.2,
        dx,
        dy,
        gradient_boundary_scheme="FIRST_ORDER_ONE_SIDED",
    )
    expected = 3.2 * electric_field**2
    np.testing.assert_allclose(heat, expected, rtol=2e-15, atol=2e-6)
    assert diagnostics["joule_heat_min_W_per_m3"] == pytest.approx(expected)
    assert diagnostics["electric_gradient_boundary_scheme"] == (
        "FIRST_ORDER_ONE_SIDED"
    )


def test_surface_heat_is_applied_to_complete_footprint_not_perimeter() -> None:
    mask = np.zeros((9, 9), dtype=np.bool_)
    mask[2:7, 2:7] = True
    source = surface_heat_flux_to_volume(mask, 60.0, 3.0e-4)
    assert np.count_nonzero(source) == 25
    assert source[4, 4] == pytest.approx(2.0e5)  # interior, not perimeter
    assert source[0, 0] == 0.0


def test_surface_heat_volume_conversion_rejects_finite_input_overflow() -> None:
    mask = np.ones((3, 3), dtype=np.bool_)
    with pytest.raises(RuntimeError, match="overflowed"):
        surface_heat_flux_to_volume(mask, 1.0e308, 1.0e-300)


def test_electrical_source_sum_rejects_overflow() -> None:
    source = ElectricalSourceFields(
        joule_heat_W_per_m3=np.full((3, 3), 1.0e308),
        electrochemical_heat_W_per_m3=np.full((3, 3), 1.0e308),
        electric_potential_V=np.zeros((3, 3)),
        metadata={},
    )
    with np.errstate(over="ignore"):
        with pytest.raises(RuntimeError, match="non-finite"):
            _ = source.total_heat_W_per_m3
