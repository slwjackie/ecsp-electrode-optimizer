from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ecsp_v6.config import load_config
from ecsp_v6.physics.composition_model import build_composition
from ecsp_v6.physics.electrochem import (
    _implicit_bv_surface_contacts,
    _series_contact_slope,
    compute_current,
    initial_state,
    interface_reactions,
    solve_electrical_and_reaction,
    transport_fields,
)
from ecsp_v6.physics.geometry import GeometryBatch
from ecsp_v6.physics.numerics import harmonic_mean


ROOT = Path(__file__).resolve().parents[2]


def test_series_contact_slope_is_overflow_safe_and_invalid_inputs_propagate():
    kinetic = torch.tensor([1.0e308, 1.0e308, float("nan")], dtype=torch.float64)
    conductance = torch.tensor([1.0e308, 1.0e-100, 1.0], dtype=torch.float64)
    active = torch.ones(3, dtype=torch.bool)
    result = _series_contact_slope(kinetic, conductance, 1.0e-300, active)
    assert torch.isfinite(result[:2]).all()
    assert result[0].item() == pytest.approx(5.0e307)
    assert result[1].item() == pytest.approx(1.0e-100)
    assert torch.isnan(result[2])


def _surface_problem() -> tuple[GeometryBatch, dict, object, dict, dict, float]:
    config = load_config(ROOT / "config" / "default_lp_pva.yaml")
    config["interface"]["boundaryCouplingModel"] = "surface_overlay_bv"
    # The test exercises dimensional bookkeeping, not optional closures.
    config["coupled"]["usePaperMassTransferSaturation"] = False
    config["interface"]["includeActivationHeat"] = False
    config["interface"]["blocking"]["minimumActiveAreaFraction"] = 1.0
    config["interface"]["blocking"]["passivation"]["enabled"] = False
    config["interface"]["blocking"]["gasCoverage"]["enabled"] = False
    config["numerics"]["potentialSolver"]["methodStatic"] = "pcg"
    config["numerics"]["potentialSolver"]["methodCoupled"] = "pcg"
    config["numerics"]["potentialSolver"]["preconditioner"] = "jacobi"
    n = 9
    anode = torch.zeros((2, n, n), dtype=torch.bool)
    cathode = torch.zeros_like(anode)
    # Equal footprint counts but deliberately different perimeters.
    anode[0, 2:4, 1:3] = True
    cathode[0, 5:7, 6:8] = True
    anode[1, 1, 1:5] = True
    cathode[1, 7, 4:8] = True
    geometry = GeometryBatch(
        geometry_ids=["compact", "line"],
        anode=anode,
        cathode=cathode,
        fixed=torch.zeros_like(anode),
        propellant=torch.ones_like(anode),
        grid_size=n,
        domain_size_m=float(config["geometry"]["domainSize_m"]),
        minimum_gap_m=float(config["geometry"]["minimumElectrodeGap_m"]),
    )
    composition = build_composition(config)
    voltage = 260.0
    spacing = geometry.domain_size_m / (n - 1)
    state = initial_state(geometry, config, composition, voltage, torch.float64)
    transport = transport_fields(state, geometry, config, composition)
    return geometry, config, composition, state, transport, spacing


def test_equal_footprints_not_perimeters_give_equal_surface_current() -> None:
    geometry, config, composition, state, transport, spacing = _surface_problem()
    boundary = _implicit_bv_surface_contacts(
        state["potential"], state, transport, geometry, config, composition,
        260.0, spacing,
    )["reaction"]

    expected_a = (
        boundary["jAnode_A_per_m2"].sum(dim=(-2, -1)) * spacing**2
    )
    expected_c = (
        boundary["jCathode_A_per_m2"].sum(dim=(-2, -1)) * spacing**2
    )
    torch.testing.assert_close(boundary["rawAnodeCurrent_A"], expected_a)
    torch.testing.assert_close(boundary["rawCathodeCurrent_A"], expected_c)
    assert boundary["rawAnodeCurrent_A"][0].item() == pytest.approx(
        boundary["rawAnodeCurrent_A"][1].item(), rel=1e-12
    )
    assert boundary["rawCathodeCurrent_A"][0].item() == pytest.approx(
        boundary["rawCathodeCurrent_A"][1].item(), rel=1e-12
    )


def test_surface_iteration_defaults_do_not_inherit_legacy_budget() -> None:
    geometry, config, composition, state, transport, spacing = _surface_problem()
    robin = config["interface"]["nonlinearRobin"]
    robin.pop("localInterfaceMinimumIterations", None)
    robin.pop("localInterfaceMaximumIterations", None)

    reactions = []
    for legacy_budget in (1, 10_000):
        robin["localInterfaceIterations"] = legacy_budget
        reactions.append(
            _implicit_bv_surface_contacts(
                state["potential"],
                state,
                transport,
                geometry,
                config,
                composition,
                260.0,
                spacing,
            )["reaction"]
        )

    iterations = reactions[0]["maximumLocalRobinIterations"]
    assert bool(torch.all((iterations >= 8) & (iterations <= 60)))
    for name, expected in reactions[0].items():
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(
                reactions[1][name], expected, rtol=0.0, atol=0.0
            )


def test_surface_contact_newton_and_roundoff_match_native_iteration_contract() -> None:
    geometry, config, composition, state, transport, spacing = _surface_problem()
    potential = state["potential"].clone()
    potential[geometry.anode] = 250.0
    potential[geometry.cathode] = 10.0

    # alpha=0 with the forward-only law makes each channel constant.  The
    # safeguarded Newton step therefore has an analytic, exactly bracketed
    # target while one bisection iteration is deliberately insufficient.
    config["interface"]["useFullButlerVolmer"] = False
    for species in ("water", "lp"):
        for polarity in ("anode", "cathode"):
            channel = config["interface"][species][polarity]
            channel["equilibriumPotential_V"] = 0.0
            channel["exchangeCurrentDensity_A_per_m2"] = 1.0
            channel["chargeTransferCoefficient"] = 0.0
    robin = config["interface"]["nonlinearRobin"]
    robin["localInterfaceMinimumIterations"] = 1
    robin["localInterfaceMaximumIterations"] = 1
    robin["localRobinResidualTolerance_A_per_m2"] = 1e-300
    robin["localRobinRelativeResidualTolerance"] = 0.0
    robin["localRoundoffSafetyFactor"] = 2.0

    robin["localNewtonPolishIterations"] = 0
    without_newton = _implicit_bv_surface_contacts(
        potential,
        state,
        transport,
        geometry,
        config,
        composition,
        260.0,
        spacing,
    )["reaction"]
    assert bool(torch.all(without_newton["unresolvedLocalRobinFaceCount"] > 0))
    assert bool(torch.all(without_newton["maximumLocalRobinCombinedResidual"] > 1.0))

    robin["localNewtonPolishIterations"] = 1
    with_newton = _implicit_bv_surface_contacts(
        potential,
        state,
        transport,
        geometry,
        config,
        composition,
        260.0,
        spacing,
    )["reaction"]
    assert bool(torch.all(with_newton["unresolvedLocalRobinFaceCount"] == 0))
    assert bool(torch.all(with_newton["maximumLocalRobinCombinedResidual"] <= 1.0))
    assert bool(torch.all(with_newton["maximumLocalRobinRoundoffFloor_A_per_m2"] > 0.0))
    assert bool(torch.all(with_newton["roundoffLimitedLocalRobinFaceCount"] > 0))
    # The native diagnostic counts bisection iterations; Newton is a polish.
    assert with_newton["maximumLocalRobinIterations"].tolist() == [1, 1]

    robin["localRoundoffSafetyFactor"] = 4.0
    doubled_safety = _implicit_bv_surface_contacts(
        potential,
        state,
        transport,
        geometry,
        config,
        composition,
        260.0,
        spacing,
    )["reaction"]
    torch.testing.assert_close(
        doubled_safety["maximumLocalRobinRoundoffFloor_A_per_m2"],
        2.0 * with_newton["maximumLocalRobinRoundoffFloor_A_per_m2"],
        rtol=2e-15,
        atol=0.0,
    )


def test_surface_heat_and_species_sources_use_surface_layer_volume() -> None:
    geometry, config, composition, state, transport, spacing = _surface_problem()
    boundary = _implicit_bv_surface_contacts(
        state["potential"], state, transport, geometry, config, composition,
        260.0, spacing,
    )["reaction"]
    electrical = {
        "Jx": torch.zeros_like(state["potential"]),
        "Jy": torch.zeros_like(state["potential"]),
    }
    reaction = interface_reactions(
        state, electrical, transport, geometry, config, composition, 260.0,
        spacing, robin_boundary=boundary,
    )
    layer = float(config["geometry"]["surfaceLayerThickness_m"])
    volume_element = spacing**2 * layer
    faraday = float(config["transport"]["faradayConstant_C_per_mol"])
    channels = config["interface"]

    expected_heat = torch.zeros(geometry.batch_size, dtype=torch.float64)
    expected_salt = torch.zeros_like(expected_heat)
    expected_water = torch.zeros_like(expected_heat)
    for field, raw in (
        ("jWaterAnode_A_per_m2", channels["water"]["anode"]),
        ("jWaterCathode_A_per_m2", channels["water"]["cathode"]),
        ("jLPAnode_A_per_m2", channels["lp"]["anode"]),
        ("jLPCathode_A_per_m2", channels["lp"]["cathode"]),
    ):
        j_area = boundary[field].sum(dim=(-2, -1)) * spacing**2
        molar_electron_rate = j_area / (float(raw["electronNumber"]) * faraday)
        expected_heat += molar_electron_rate * float(raw["reactionEnthalpy_J_per_mol"])
        expected_salt += molar_electron_rate * float(
            raw.get("saltStoichiometry_mol_per_molElectron", 0.0)
        )
        expected_water += molar_electron_rate * float(
            raw.get("waterStoichiometry_mol_per_molElectron", 0.0)
        )

    actual_heat = reaction["qTotal_W_per_m3"].sum(dim=(-2, -1)) * volume_element
    actual_salt = reaction["saltSink_mol_per_m3_s"].sum(dim=(-2, -1)) * volume_element
    actual_water = reaction["waterSink_mol_per_m3_s"].sum(dim=(-2, -1)) * volume_element
    torch.testing.assert_close(actual_heat, expected_heat, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(actual_salt, expected_salt, rtol=1e-11, atol=1e-15)
    torch.testing.assert_close(actual_water, expected_water, rtol=1e-11, atol=1e-15)


def test_surface_contacts_do_not_pin_bulk_potential_to_metal_voltage() -> None:
    geometry, config, composition, state, _, _ = _surface_problem()
    midpoint = 0.5 * (
        260.0 + float(config["electrical"]["cathodeVoltage_V"])
    )
    contacts = geometry.anode | geometry.cathode
    torch.testing.assert_close(
        state["potential"][contacts],
        torch.full_like(state["potential"][contacts], midpoint),
    )
    assert bool(torch.all(state["cation"][contacts] > 0.0))
    assert bool(torch.all(state["water"][contacts] > 0.0))


def test_outer_ring_contact_cells_are_active_unknowns_and_balance_current() -> None:
    """Contacts touching the planform edge remain physical control volumes."""
    geometry, config, composition, _, _, spacing = _surface_problem()
    n = geometry.grid_size
    anode = torch.zeros((1, n, n), dtype=torch.bool)
    cathode = torch.zeros_like(anode)
    anode[0, 0, 1:5] = True
    cathode[0, -1, 4:8] = True
    edge_geometry = GeometryBatch(
        geometry_ids=["edge-contacts"],
        anode=anode,
        cathode=cathode,
        fixed=torch.zeros_like(anode),
        propellant=torch.ones_like(anode),
        grid_size=n,
        domain_size_m=geometry.domain_size_m,
        minimum_gap_m=geometry.minimum_gap_m,
    )
    state = initial_state(
        edge_geometry, config, composition, 260.0, torch.float64
    )
    transport = transport_fields(state, edge_geometry, config, composition)
    electrical, reaction, diagnostics = solve_electrical_and_reaction(
        state,
        transport,
        edge_geometry,
        config,
        composition,
        260.0,
        spacing,
        static=False,
    )

    assert bool(torch.all(diagnostics.converged))
    assert reaction["totalCurrent_A"].item() > 0.0
    assert reaction["currentBalanceMismatchBeforeCoupling"].item() <= float(
        config["interface"]["nonlinearRobin"]["currentBalanceTolerance"]
    )
    contacts = anode | cathode
    assert bool(torch.all(torch.isfinite(electrical["potential"][contacts])))
    assert not bool(torch.all(electrical["potential"][anode] == 260.0))
    assert not bool(
        torch.all(
            electrical["potential"][cathode]
            == float(config["electrical"]["cathodeVoltage_V"])
        )
    )


def test_direct_surface_solver_handles_outer_ring_control_volumes() -> None:
    """The SciPy reference path must not index outside an edge CV stencil."""
    geometry, config, composition, _, _, spacing = _surface_problem()
    n = geometry.grid_size
    anode = torch.zeros((1, n, n), dtype=torch.bool)
    cathode = torch.zeros_like(anode)
    anode[0, 0, 1:5] = True
    cathode[0, -1, 4:8] = True
    edge_geometry = GeometryBatch(
        geometry_ids=["edge-contacts-direct"],
        anode=anode,
        cathode=cathode,
        fixed=torch.zeros_like(anode),
        propellant=torch.ones_like(anode),
        grid_size=n,
        domain_size_m=geometry.domain_size_m,
        minimum_gap_m=geometry.minimum_gap_m,
    )
    config["numerics"]["potentialSolver"]["methodCoupled"] = "direct"
    state = initial_state(
        edge_geometry, config, composition, 260.0, torch.float64
    )
    transport = transport_fields(state, edge_geometry, config, composition)

    electrical, reaction, diagnostics = solve_electrical_and_reaction(
        state,
        transport,
        edge_geometry,
        config,
        composition,
        260.0,
        spacing,
        static=False,
    )

    assert bool(torch.all(diagnostics.converged))
    assert reaction["totalCurrent_A"].item() > 0.0
    assert bool(torch.all(torch.isfinite(electrical["potential"])))


def test_surface_joule_heat_equals_finite_volume_face_and_contact_power() -> None:
    geometry, config, composition, state, transport, spacing = _surface_problem()
    electrical, _, diagnostics = solve_electrical_and_reaction(
        state,
        transport,
        geometry,
        config,
        composition,
        260.0,
        spacing,
        static=False,
    )
    assert bool(torch.all(diagnostics.converged))
    potential = electrical["potential"]
    propellant = geometry.propellant
    valid_x = propellant[..., :, :-1] & propellant[..., :, 1:]
    valid_y = propellant[..., :-1, :] & propellant[..., 1:, :]
    sigma_x = harmonic_mean(
        transport["sigmaTotal"][..., :, :-1],
        transport["sigmaTotal"][..., :, 1:],
    )
    sigma_y = harmonic_mean(
        transport["sigmaTotal"][..., :-1, :],
        transport["sigmaTotal"][..., 1:, :],
    )
    ex_face = -(potential[..., :, 1:] - potential[..., :, :-1]) / spacing
    ey_face = -(potential[..., 1:, :] - potential[..., :-1, :]) / spacing
    expected_in_plane_density_sum = (
        torch.where(valid_x, sigma_x * ex_face.square(), 0.0).sum(dim=(-2, -1))
        + torch.where(valid_y, sigma_y * ey_face.square(), 0.0).sum(dim=(-2, -1))
    )
    actual_in_plane_density_sum = electrical[
        "inPlaneJouleHeat_W_per_m3"
    ].sum(dim=(-2, -1))
    torch.testing.assert_close(
        actual_in_plane_density_sum,
        expected_in_plane_density_sum,
        rtol=2e-13,
        atol=1e-12,
    )

    contact = electrical["contactNormalJouleHeat_W_per_m3"]
    torch.testing.assert_close(
        contact,
        electrical["contactNormalJouleHeatAnode_W_per_m3"]
        + electrical["contactNormalJouleHeatCathode_W_per_m3"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        electrical["jouleHeat_W_per_m3"],
        electrical["inPlaneJouleHeat_W_per_m3"] + contact,
        rtol=2e-15,
        atol=1e-12,
    )
    layer = float(config["geometry"]["surfaceLayerThickness_m"])
    cell_volume = spacing * spacing * layer
    integrated_contact_power = contact.sum(dim=(-2, -1)) * cell_volume
    assert bool(torch.all(integrated_contact_power > 0.0))


def test_total_j_dot_e_preserves_signed_diffusion_cross_term() -> None:
    geometry, config, _, state, _, _ = _surface_problem()
    n = 3
    propellant = torch.ones((1, n, n), dtype=torch.bool)
    small_geometry = GeometryBatch(
        geometry_ids=["signed-jdot-e"],
        anode=torch.zeros_like(propellant),
        cathode=torch.zeros_like(propellant),
        fixed=torch.zeros_like(propellant),
        propellant=propellant,
        grid_size=n,
        domain_size_m=2.0,
        minimum_gap_m=0.0,
    )
    potential = torch.arange(n, dtype=torch.float64)[None, None, :].expand(
        1, n, n
    ).clone()
    cation = (2.0 - torch.arange(n, dtype=torch.float64))[
        None, None, :
    ].expand_as(potential).clone()
    faraday = float(config["transport"]["faradayConstant_C_per_mol"])
    one = torch.ones_like(potential)
    zero = torch.zeros_like(potential)
    state = {**state, "cation": cation, "anion": one.clone()}
    transport = {
        "Dcation": one * (10.0 / faraday),
        "Danion": zero,
        "sigmaTotal": one,
        "sigmaIonic": one,
        "sigmaElectronic": zero,
    }
    config["electrical"]["jouleHeatModel"] = "total_j_dot_e"
    electrical = compute_current(
        potential, state, transport, small_geometry, config, spacing=1.0
    )

    # Six x-faces each have E=-1, J_ohm=-1, J_diff=+10, hence J·E=-9.
    assert electrical["jouleHeat_W_per_m3"].sum().item() == pytest.approx(
        -54.0, rel=2e-14, abs=1e-12
    )
    assert bool(torch.any(electrical["jouleHeat_W_per_m3"] < 0.0))
    torch.testing.assert_close(
        electrical["jouleHeat_W_per_m3"],
        electrical["electricalPowerDensityJdotE_W_per_m3"],
        rtol=0.0,
        atol=0.0,
    )


def test_surface_outer_nonconvergence_is_reported_even_if_raise_is_disabled() -> None:
    geometry, config, composition, state, transport, spacing = _surface_problem()
    robin = config["interface"]["nonlinearRobin"]
    robin["maximumIterations"] = 2
    robin["minimumIterationsCoupled"] = 2
    robin["potentialTolerance_V"] = 0.0
    robin["relativeReactionCurrentTolerance"] = 0.0
    robin["failOnNonConvergence"] = False
    _, reaction, linear_diagnostics = solve_electrical_and_reaction(
        state,
        transport,
        geometry,
        config,
        composition,
        260.0,
        spacing,
        static=False,
    )
    assert bool(torch.all(linear_diagnostics.converged))
    assert not bool(torch.any(reaction["nonlinearRobinConverged"]))


def test_surface_outer_iteration_freezes_each_converged_lane_and_matches_serial() -> None:
    geometry, config, composition, _, _, spacing = _surface_problem()
    # These two voltages converge on different outer iterations.  A converged
    # lane must not keep moving merely because another lane is still active.
    config["interface"]["nonlinearRobin"][
        "relativeReactionCurrentTolerance"
    ] = 0.2
    voltages = torch.tensor([260.0, 20.0], dtype=torch.float64)
    state = initial_state(
        geometry, config, composition, voltages, torch.float64
    )
    transport = transport_fields(state, geometry, config, composition)
    batch_electrical, batch_reaction, batch_linear = (
        solve_electrical_and_reaction(
            state,
            transport,
            geometry,
            config,
            composition,
            voltages,
            spacing,
            static=False,
        )
    )

    serial_results = []
    for index, voltage in enumerate(voltages.tolist()):
        single_geometry = GeometryBatch(
            geometry_ids=[geometry.geometry_ids[index]],
            anode=geometry.anode[index : index + 1],
            cathode=geometry.cathode[index : index + 1],
            fixed=geometry.fixed[index : index + 1],
            propellant=geometry.propellant[index : index + 1],
            grid_size=geometry.grid_size,
            domain_size_m=geometry.domain_size_m,
            minimum_gap_m=geometry.minimum_gap_m,
        )
        single_state = initial_state(
            single_geometry, config, composition, voltage, torch.float64
        )
        single_transport = transport_fields(
            single_state, single_geometry, config, composition
        )
        serial_results.append(
            solve_electrical_and_reaction(
                single_state,
                single_transport,
                single_geometry,
                config,
                composition,
                voltage,
                spacing,
                static=False,
            )
        )

    serial_iterations = torch.cat(
        [result[1]["nonlinearRobinIterations"] for result in serial_results]
    )
    assert serial_iterations[0].item() != serial_iterations[1].item()
    torch.testing.assert_close(
        batch_reaction["nonlinearRobinIterations"],
        serial_iterations,
        rtol=0.0,
        atol=0.0,
    )

    def assert_lane_mapping_matches(
        batch_values: dict, serial_values: dict, index: int
    ) -> None:
        assert batch_values.keys() == serial_values.keys()
        for name, batch_value in batch_values.items():
            serial_value = serial_values[name]
            if isinstance(batch_value, torch.Tensor):
                assert isinstance(serial_value, torch.Tensor), name
                assert batch_value.shape[0] == geometry.batch_size, name
                assert serial_value.shape[0] == 1, name
                if batch_value.is_floating_point():
                    torch.testing.assert_close(
                        batch_value[index],
                        serial_value[0],
                        rtol=2.0e-12,
                        atol=1.0e-15,
                        equal_nan=True,
                        msg=lambda message: f"{name}: {message}",
                    )
                else:
                    assert torch.equal(batch_value[index], serial_value[0]), name
            else:
                assert batch_value == serial_value, name

    for index, (serial_electrical, serial_reaction, serial_linear) in enumerate(
        serial_results
    ):
        assert_lane_mapping_matches(
            batch_electrical, serial_electrical, index
        )
        assert_lane_mapping_matches(batch_reaction, serial_reaction, index)
        assert batch_linear.method == serial_linear.method
        for name in (
            "iterations",
            "relative_residual",
            "converged",
            "restarts",
            "refinement_rounds",
            "fallback_used",
        ):
            batch_value = getattr(batch_linear, name)
            serial_value = getattr(serial_linear, name)
            if batch_value is None or serial_value is None:
                assert batch_value is None and serial_value is None, name
            elif batch_value.is_floating_point():
                torch.testing.assert_close(
                    batch_value[index],
                    serial_value[0],
                    rtol=2.0e-12,
                    atol=1.0e-15,
                    equal_nan=True,
                    msg=lambda message: f"{name}: {message}",
                )
            else:
                assert torch.equal(batch_value[index], serial_value[0]), name
