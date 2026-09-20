from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import pytest
import ecsp_reactive.solver as solver_module

from ecsp_reactive.configuration import ReactiveCaseConfig, load_reactive_config
from ecsp_reactive.core import REACTION_PROGRESS, RHO
from ecsp_reactive.electrical import ElectricalSourceFields
from ecsp_reactive.level_set import ReinitializationConfig
from ecsp_reactive.provenance import ReactiveConfigurationError, ReactiveNumericalError
from ecsp_reactive.solver import ElectricalStageRequest, ReactiveSolver


REPOSITORY = Path(__file__).resolve().parents[2]


def source_fields(
    shape: tuple[int, int],
    *,
    joule: float = 0.0,
    electrochemical: float = 0.0,
    provider: str = "PRESCRIBED_ARRAY_FIELDS",
) -> ElectricalSourceFields:
    return ElectricalSourceFields(
        joule_heat_W_per_m3=np.full(shape, joule, dtype=np.float64),
        electrochemical_heat_W_per_m3=np.full(
            shape, electrochemical, dtype=np.float64
        ),
        electric_potential_V=np.zeros(shape, dtype=np.float64),
        metadata={
            "test_source": "prescribed",
            "provider": provider,
            "provenance": "ASSUMED_NOT_FROM_PAPER",
            "source": "explicit synthetic solver-test field",
        },
    )


def small_config(**changes: object) -> ReactiveCaseConfig:
    base = load_reactive_config(
        REPOSITORY / "config" / "paper_faithful_assumed_v8_3_0.yaml"
    )
    values: dict[str, object] = {
        "nx": 8,
        "ny": 8,
        "lx_m": 8.0e-3,
        "ly_m": 8.0e-3,
        "interface_x_m": 4.0e-3,
        "end_time_s": 1.0e-8,
        "maximum_steps": 100,
        "tait_B_Pa": 100.0,
        "initial_u_m_per_s": 0.0,
        "initial_v_m_per_s": 0.0,
        "reaction_preexponential_per_s": 0.0,
        "reaction_activation_energy_J_per_mol": 0.0,
        "reaction_order": 1.0,
        "decomposition_preexponential_per_s": 0.0,
        "decomposition_activation_energy_J_per_mol": 0.0,
        "solid_conductivity_W_per_mK": 0.0,
        "decomposition_order": 1.0,
        "conductivity_S_per_m": None,
        "electric_field_x_V_per_m": None,
        "electric_field_y_V_per_m": None,
        "electrochemical_heat_W_per_m3": None,
        "electrical_source_provider": "PRESCRIBED_ARRAY_FIELDS",
        "algorithms": {**base.algorithms, "flow_boundary": "PERIODIC"},
    }
    values.update(changes)
    return replace(base, **values)


def extended_config(**changes: object) -> ReactiveCaseConfig:
    paper = small_config()
    values: dict[str, object] = {
        "model_mode": "ecsp_extended",
        "butler_volmer_enabled": True,
        "electrical_source_provider": "EXISTING_BC_GLOBAL_CALLBACK",
    }
    values.update(changes)
    return replace(paper, **values)


def uniform_config(**changes: object) -> ReactiveCaseConfig:
    array_case = small_config()
    values: dict[str, object] = {
        "electrical_source_provider": "CONFIG_UNIFORM_FIELD",
        "conductivity_S_per_m": 2.0,
        "electric_field_x_V_per_m": 3.0,
        "electric_field_y_V_per_m": -4.0,
        "electrochemical_heat_W_per_m3": 5.0,
    }
    values.update(changes)
    return replace(array_case, **values)


def surface_contacts(
    *, anode_width: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    anode = np.zeros((8, 8), dtype=np.uint8)
    cathode = np.zeros((8, 8), dtype=np.uint8)
    anode[1:7, 1 : 1 + anode_width] = 255
    cathode[1:7, 6:7] = 255
    return anode.view(np.bool_), cathode.view(np.bool_)


def test_electrical_provider_contracts_fail_closed_and_report_consumed_inputs() -> None:
    array_config = small_config()
    with pytest.raises(
        ReactiveConfigurationError, match="requires explicit ElectricalSourceFields"
    ):
        ReactiveSolver(array_config)

    with pytest.raises(ReactiveConfigurationError, match="identify as"):
        ReactiveSolver(
            array_config,
            prescribed_electrical=source_fields(
                (8, 8), provider="CONFIG_UNIFORM_FIELD"
            ),
        )

    uniform = uniform_config()
    with pytest.raises(ReactiveConfigurationError, match="rejects supplied"):
        ReactiveSolver(
            uniform, prescribed_electrical=source_fields((8, 8))
        )

    result = ReactiveSolver(uniform).run()
    consumed = result.metadata["consumed_electrical_inputs"]
    assert consumed["provider"] == "CONFIG_UNIFORM_FIELD"
    assert consumed["consumed_inputs"] == (
        "config.electrical.conductivity",
        "config.electrical.electric_field_x",
        "config.electrical.electric_field_y",
        "config.electrical.electrochemical_heat",
    )
    assert consumed["ignored_config_uniform_scalars"] == ()
    assert consumed["config_uniform_scalars_absent"] == ()
    assert result.metadata["provider_authentication_status"] == (
        "CONFIG_DECLARED_FIELD_NO_GEOMETRY_SOLVE"
    )
    assert result.metadata["electrical_source_stage_records"][0]["provider"] == (
        "CONFIG_UNIFORM_FIELD"
    )


def test_single_process_solver_rejects_declared_rowwise_mpi_runner() -> None:
    config = small_config()
    rowwise = replace(
        config,
        algorithms={**config.algorithms, "mpi_decomposition": "ROWWISE_WENO3_HALO"},
    )
    with pytest.raises(ReactiveConfigurationError, match="mpi_decomposition"):
        ReactiveSolver(
            rowwise, prescribed_electrical=source_fields((8, 8))
        )


def test_uniform_free_stream_and_separate_solid_state_are_preserved() -> None:
    config = small_config(initial_reaction_progress=1.0, initial_alpha=1.0)
    fields = source_fields((config.ny, config.nx))
    solver = ReactiveSolver(config, prescribed_electrical=fields)
    initial, initial_solid, initial_alpha, initial_phi = solver.initial_state()
    result = solver.run()

    np.testing.assert_allclose(result.conservative_state, initial, rtol=0.0, atol=2.0e-13)
    np.testing.assert_array_equal(result.solid_temperature_K, initial_solid)
    np.testing.assert_array_equal(result.solid_decomposition_alpha, initial_alpha)
    np.testing.assert_allclose(result.material_level_set_m, initial_phi, rtol=0.0, atol=2.0e-18)
    assert result.metadata["positivity_face_fallback_count"] == 0
    assert result.metadata["electrical_source_evaluation_count"] == 3
    assert result.metadata["electrical_callback_evaluation_count"] == 0
    assert result.metadata["provider_authentication_status"] == (
        "ARRAY_METADATA_SELF_ATTESTED_UNVERIFIED"
    )
    consumed = result.metadata["consumed_electrical_inputs"]
    assert consumed["provider"] == "PRESCRIBED_ARRAY_FIELDS"
    assert consumed["consumed_inputs"] == (
        "prescribed_electrical.joule_heat_W_per_m3",
        "prescribed_electrical.electrochemical_heat_W_per_m3",
        "prescribed_electrical.electric_potential_V",
        "prescribed_electrical.metadata",
    )
    assert consumed["config_uniform_scalars_absent"] == (
        "config.electrical.conductivity",
        "config.electrical.electric_field_x",
        "config.electrical.electric_field_y",
        "config.electrical.electrochemical_heat",
    )
    records = result.metadata["electrical_source_stage_records"]
    assert len(records) == 3
    assert records[0]["provenance"] == "ASSUMED_NOT_FROM_PAPER"
    assert len(records[0]["field_sha256"]["joule_heat_W_per_m3"]) == 64
    assert result.metadata["solid_eq2_coupling_status"].startswith("SEPARATE_DIAGNOSTIC")
    assert result.metadata["electrical_heat_destination"] == (
        "EULER_EQ1_ONLY_NO_DUPLICATION_IN_SOLID_EQ2"
    )
    front = result.metadata["species_front_tracking"]
    assert front["legacy_area_over_front_length_used"] is False
    assert front["thresholds"] == (0.3, 0.5, 0.7)
    assert len(front["history"]) == result.accepted_steps
    assert front["history"][0]["by_threshold"]["0.5"][
        "mean_positive_x_ray_speed_m_per_s"
    ] is None
    assert front["history_stride_steps"] == 1
    assert front["retained_sample_count"] == result.accepted_steps
    assert result.metadata["result_history_resource_preflight"][
        "projected_total_bytes"
    ] <= config.maximum_history_bytes


def test_deterministic_repeat_and_deterministic_npz_json(tmp_path: Path) -> None:
    config = small_config(end_time_s=1.0e-4)
    fields = source_fields((8, 8), joule=3.0, electrochemical=2.0)
    first = ReactiveSolver(config, prescribed_electrical=fields).run()
    second = ReactiveSolver(config, prescribed_electrical=fields).run()
    np.testing.assert_array_equal(first.conservative_state, second.conservative_state)
    np.testing.assert_array_equal(first.solid_temperature_K, second.solid_temperature_K)
    np.testing.assert_array_equal(first.material_level_set_m, second.material_level_set_m)
    assert first.summary() == second.summary()

    first_npz, first_json = first.save(tmp_path / "first")
    second_npz, second_json = second.save(tmp_path / "second")
    assert first_npz.read_bytes() == second_npz.read_bytes()
    assert first_json.read_bytes() == second_json.read_bytes()
    summary = json.loads(first_json.read_text(encoding="utf-8"))
    assert summary["accepted_steps"] == first.accepted_steps
    with np.load(first_npz, allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored["conservative_state"], first.conservative_state)

    with pytest.raises(TypeError):
        first.metadata["flow_progress_time_step_policy"][
            "maximum_progress_increment"
        ] = 0.9
    with pytest.raises(AttributeError):
        first.metadata["species_front_tracking"]["history"].append({})
    repeated_npz, repeated_json = first.save(tmp_path / "repeated")
    assert repeated_npz.read_bytes() == first_npz.read_bytes()
    assert repeated_json.read_bytes() == first_json.read_bytes()
    dotted_npz, dotted_json = first.save(tmp_path / "case.v1")
    assert dotted_npz.name == "case.v1.npz"
    assert dotted_json.name == "case.v1.json"
    for immutable_array in (
        first.conservative_state,
        first.solid_temperature_K,
        first.solid_decomposition_alpha,
        first.material_level_set_m,
        first.step_time_s,
        first.step_dt_s,
    ):
        with pytest.raises(ValueError):
            immutable_array.setflags(write=True)


def test_uniform_velocity_actually_transports_independent_material_level_set() -> None:
    baseline = small_config()
    velocity = 2.0
    end_time = 1.0e-5
    config = small_config(
        nx=12,
        lx_m=1.2e-2,
        interface_x_m=6.0e-3,
        initial_u_m_per_s=velocity,
        end_time_s=end_time,
        algorithms={**baseline.algorithms, "flow_boundary": "OUTFLOW"},
    )
    solver = ReactiveSolver(
        config, prescribed_electrical=source_fields((config.ny, config.nx))
    )
    _, _, _, initial_phi = solver.initial_state()
    result = solver.run()
    expected = initial_phi - velocity * end_time
    # Constant extrapolation affects WENO stencils near the outer boundary;
    # the central interface region follows the exact phi(x-ut) translation.
    np.testing.assert_allclose(
        result.material_level_set_m[:, 5:8], expected[:, 5:8], rtol=0.0, atol=1.0e-9
    )
    assert not np.array_equal(result.material_level_set_m, initial_phi)
    assert result.metadata["level_set_final_interior_area_m2"] > result.metadata[
        "level_set_initial_interior_area_m2"
    ]


def test_reaction_electrical_energy_and_progress_bookkeeping() -> None:
    reaction_rate = 2.0
    heat_per_progress = 50.0
    electrical_heat = 10.0
    end_time = 1.0e-5
    config = small_config(
        end_time_s=end_time,
        initial_reaction_progress=0.0,
        reaction_preexponential_per_s=reaction_rate,
        reaction_order=0.0,
        reaction_heat_J_per_kg=heat_per_progress,
    )
    result = ReactiveSolver(
        config,
        prescribed_electrical=source_fields((8, 8), joule=electrical_heat),
    ).run()
    area = config.lx_m * config.ly_m
    expected_progress_integral = (
        config.initial_rho_kg_per_m3 * reaction_rate * end_time * area
    )
    expected_reaction_energy = heat_per_progress * expected_progress_integral
    expected_electrical_energy = electrical_heat * end_time * area

    assert result.accepted_steps == 1
    assert result.conservation.reaction_progress_increment == pytest.approx(
        expected_progress_integral, rel=2.0e-13
    )
    assert result.conservation.reaction_heat_increment_J_per_m == pytest.approx(
        expected_reaction_energy, rel=2.0e-13
    )
    assert result.conservation.electrical_energy_increment_J_per_m == pytest.approx(
        expected_electrical_energy, rel=2.0e-13
    )
    assert abs(result.conservation.reaction_bookkeeping_residual_J_per_m) < 1.0e-18
    assert max(abs(value) for value in result.conservation.closure_residual) < 2.0e-14
    density = result.conservative_state[..., RHO]
    progress = result.conservative_state[..., REACTION_PROGRESS] / density
    np.testing.assert_allclose(progress, reaction_rate * end_time, rtol=0.0, atol=2.0e-16)
    # Eq. (2) receives no duplicated electric heat in this unresolved coupling.
    assert result.solid_budget.externally_coupled_heat_increment_J_per_m == 0.0


def test_reaction_rate_can_select_adaptive_time_step() -> None:
    config = small_config(
        end_time_s=8.0e-11,
        reaction_preexponential_per_s=1.0e9,
        reaction_activation_energy_J_per_mol=0.0,
        reaction_order=0.0,
        maximum_steps=10,
    )
    solver = ReactiveSolver(
        config, prescribed_electrical=source_fields((8, 8))
    )
    result = solver.run()
    assert result.steps[0].selected_time_step_limit == "flow_reaction"
    assert result.steps[0].flow_reaction_limit_s == pytest.approx(5.0e-11)
    assert np.max(
        result.conservative_state[..., REACTION_PROGRESS]
        / result.conservative_state[..., RHO]
    ) <= 1.0


def test_endothermic_flow_reaction_uses_temperature_headroom_limit() -> None:
    config = small_config(
        initial_temperature_K=1000.0,
        temperature_reject_below_K=250.0,
        end_time_s=1.0e-3,
        maximum_steps=200,
        tait_B_Pa=1.0e-12,
        reaction_preexponential_per_s=1.0e6,
        reaction_activation_energy_J_per_mol=80_000.0,
        reaction_order=0.0,
        reaction_heat_J_per_kg=-5.0e7,
    )
    solver = ReactiveSolver(
        config, prescribed_electrical=source_fields((8, 8))
    )
    result = solver.run()
    assert np.min(result.solid_temperature_K) > config.temperature_reject_below_K
    from ecsp_reactive.core import conservative_to_primitive

    primitive = conservative_to_primitive(result.conservative_state, solver.eos)
    assert np.min(primitive.temperature_K) > config.temperature_reject_below_K
    assert any(step.selected_time_step_limit == "flow_reaction" for step in result.steps)


def test_endothermic_solid_reaction_uses_temperature_headroom_limit() -> None:
    config = small_config(
        initial_temperature_K=1000.0,
        temperature_reject_below_K=250.0,
        end_time_s=1.0e-3,
        maximum_steps=200,
        tait_B_Pa=1.0e-12,
        decomposition_preexponential_per_s=1.0e6,
        decomposition_activation_energy_J_per_mol=80_000.0,
        decomposition_order=0.0,
        decomposition_heat_J_per_kg=5.0e7,
    )
    result = ReactiveSolver(
        config, prescribed_electrical=source_fields((8, 8))
    ).run()
    assert np.min(result.solid_temperature_K) > config.temperature_reject_below_K
    assert any(
        step.selected_time_step_limit == "solid_stability" for step in result.steps
    )


def test_subnormal_positive_reaction_rate_has_unbounded_not_invalid_limit() -> None:
    config = small_config(
        reaction_preexponential_per_s=1.0,
        reaction_activation_energy_J_per_mol=1_780_462.464315878,
        reaction_order=0.0,
    )
    solver = ReactiveSolver(
        config, prescribed_electrical=source_fields((8, 8))
    )
    state, solid_temperature, solid_alpha, _ = solver.initial_state()
    _, reaction_limit, _, _ = solver._time_limits(
        state, solid_temperature, solid_alpha
    )
    assert math.isinf(reaction_limit)


def test_negative_total_electrical_source_is_rejected() -> None:
    config = small_config()
    with pytest.raises(
        ReactiveNumericalError, match="must be non-negative"
    ) as captured:
        ReactiveSolver(
            config,
            prescribed_electrical=source_fields(
                (8, 8), joule=1.0, electrochemical=-2.0
            ),
        )
    assert captured.value.category == "negative_total_electrical_heat"


def test_extended_mode_requires_and_exercises_explicit_callback_each_stage() -> None:
    config = extended_config()
    anode, cathode = surface_contacts()

    def zero_callback(_: ElectricalStageRequest) -> ElectricalSourceFields:
        return source_fields(
            (8, 8), provider="EXISTING_BC_GLOBAL_CALLBACK"
        )

    with pytest.raises(ReactiveConfigurationError, match="requires explicit anode"):
        ReactiveSolver(config, electrical_callback=zero_callback)
    with pytest.raises(ReactiveConfigurationError, match="explicit electrical_callback"):
        ReactiveSolver(config, anode_contact=anode, cathode_contact=cathode)

    requests: list[tuple[float, str]] = []

    def callback(request: ElectricalStageRequest) -> ElectricalSourceFields:
        requests.append((request.time_s, request.stage_name))
        assert not request.conservative_state.flags.writeable
        return source_fields(
            (8, 8), joule=1.0, provider="EXISTING_BC_GLOBAL_CALLBACK"
        )

    result = ReactiveSolver(
        config,
        electrical_callback=callback,
        anode_contact=anode,
        cathode_contact=cathode,
    ).run()
    assert [name for _, name in requests] == ["stage_1", "stage_2", "stage_3"]
    assert result.metadata["electrical_callback_evaluation_count"] == 3
    assert result.conservation.electrical_energy_increment_J_per_m > 0.0
    assert result.metadata["provider_authentication_status"] == (
        "CALLBACK_SELF_ATTESTED_UNVERIFIED"
    )
    assert result.metadata["geometry_ranking_eligible"] is False


def test_extended_callback_shape_and_nonfinite_values_fail_closed() -> None:
    config = extended_config()
    anode, cathode = surface_contacts()

    def wrong_shape(_: ElectricalStageRequest) -> ElectricalSourceFields:
        return source_fields((7, 8))

    with pytest.raises(ReactiveConfigurationError, match="shape"):
        ReactiveSolver(
            config,
            electrical_callback=wrong_shape,
            anode_contact=anode,
            cathode_contact=cathode,
        ).run()

    def nonfinite(_: ElectricalStageRequest) -> ElectricalSourceFields:
        fields = source_fields((8, 8))
        fields.joule_heat_W_per_m3[0, 0] = np.inf
        return fields

    with pytest.raises(ReactiveNumericalError, match="infinity"):
        ReactiveSolver(
            config,
            electrical_callback=nonfinite,
            anode_contact=anode,
            cathode_contact=cathode,
        ).run()

    def mislabeled(_: ElectricalStageRequest) -> ElectricalSourceFields:
        return source_fields((8, 8), provider="PRESCRIBED_ARRAY_FIELDS")

    with pytest.raises(ReactiveConfigurationError, match="identify as"):
        ReactiveSolver(
            config,
            electrical_callback=mislabeled,
            anode_contact=anode,
            cathode_contact=cathode,
        ).run()


def test_paper_prescribed_mode_rejects_surface_masks_as_non_rankable() -> None:
    config = small_config()
    anode, cathode = surface_contacts()
    with pytest.raises(ReactiveConfigurationError, match="forbids electrode surface masks"):
        ReactiveSolver(
            config,
            prescribed_electrical=source_fields((8, 8)),
            anode_contact=anode,
            cathode_contact=cathode,
        )


def test_extended_surface_contacts_are_overlays_without_removing_state() -> None:
    config = extended_config()
    anode, cathode = surface_contacts()

    def callback(request: ElectricalStageRequest) -> ElectricalSourceFields:
        assert request.surface_geometry is not None
        assert request.surface_geometry.propellant.all()
        with pytest.raises(ValueError):
            request.surface_geometry.anode_contact.setflags(write=True)
        return source_fields(
            (8, 8), provider="EXISTING_BC_GLOBAL_CALLBACK"
        )

    solver = ReactiveSolver(
        config,
        electrical_callback=callback,
        anode_contact=anode,
        cathode_contact=cathode,
    )
    assert solver.surface_geometry is not None
    assert solver.surface_geometry.propellant.all()
    assert set(solver.surface_geometry.anode_contact.view(np.uint8).ravel()) == {0, 1}
    result = solver.run()
    density = result.conservative_state[..., RHO]
    assert np.all(density[solver.surface_geometry.anode_contact] > 0.0)
    assert np.all(density[solver.surface_geometry.cathode_contact] > 0.0)
    assert result.metadata["propellant_cell_count"] == 64
    assert result.metadata["anode_contact_cell_count"] == 12
    assert result.metadata["cathode_contact_cell_count"] == 6
    assert len(result.metadata["canonical_surface_geometry_sha256"]) == 64
    assert result.metadata["geometry_ranking_eligible"] is False


def test_geometry_sensitive_callback_contract_smoke_changes_source_not_ranking_gate() -> None:
    config = extended_config(end_time_s=1.0e-7)

    def geometry_sensitive_callback(
        request: ElectricalStageRequest,
    ) -> ElectricalSourceFields:
        assert request.surface_geometry is not None
        heat = request.surface_geometry.anode_contact.astype(np.float64) * 100.0
        fields = source_fields(
            heat.shape, provider="EXISTING_BC_GLOBAL_CALLBACK"
        )
        fields.joule_heat_W_per_m3[:] = heat
        return fields

    narrow_anode, narrow_cathode = surface_contacts(anode_width=1)
    wide_anode, wide_cathode = surface_contacts(anode_width=3)
    narrow = ReactiveSolver(
        config,
        electrical_callback=geometry_sensitive_callback,
        anode_contact=narrow_anode,
        cathode_contact=narrow_cathode,
    ).run()
    wide = ReactiveSolver(
        config,
        electrical_callback=geometry_sensitive_callback,
        anode_contact=wide_anode,
        cathode_contact=wide_cathode,
    ).run()
    assert wide.conservation.electrical_energy_increment_J_per_m == pytest.approx(
        3.0 * narrow.conservation.electrical_energy_increment_J_per_m,
        rel=2.0e-14,
    )
    assert not np.array_equal(wide.conservative_state, narrow.conservative_state)
    assert (
        wide.metadata["canonical_surface_geometry_sha256"]
        != narrow.metadata["canonical_surface_geometry_sha256"]
    )
    assert wide.metadata["surface_geometry_passed_to_callback"] is True
    assert wide.metadata["geometry_ranking_eligible"] is False
    record = wide.metadata["electrical_source_stage_records"][0]
    assert record["canonical_surface_geometry_sha256"] == wide.metadata[
        "canonical_surface_geometry_sha256"
    ]
    assert record["provider_authentication_status"] == (
        "CALLBACK_SELF_ATTESTED_UNVERIFIED"
    )


def test_explicit_reinitialization_is_counted_and_hidden_defaults_are_rejected() -> None:
    base = small_config()
    config = replace(
        base,
        reinitialization_interval_steps=1,
        algorithms={
            **base.algorithms,
            "level_set_reinitialization": "SUSSMAN_FIRST_ORDER",
        },
    )
    with pytest.raises(ReactiveConfigurationError, match="explicit enabled"):
        ReactiveSolver(config, prescribed_electrical=source_fields((8, 8)))
    reinitialization = ReinitializationConfig(
        enabled=True,
        pseudo_steps=2,
        pseudo_cfl=0.2,
        source="explicit synthetic solver regression setting",
    )
    result = ReactiveSolver(
        config,
        prescribed_electrical=source_fields((8, 8)),
        reinitialization=reinitialization,
    ).run()
    assert result.metadata["level_set_reinitialization_count"] == 1
    reinit_metadata = result.metadata["level_set_reinitialization_configuration"]
    assert reinit_metadata == {
        "enabled": True,
        "method": "SUSSMAN_FIRST_ORDER",
        "interval_steps": 1,
        "pseudo_steps": 2,
        "pseudo_cfl": 0.2,
        "narrow_band_half_width_m": None,
        "provenance": "ASSUMED_NOT_FROM_PAPER",
        "source": "explicit synthetic solver regression setting",
        "parameter_origin": "EXPLICIT_REQUIRED_SOLVER_ARGUMENT",
    }
    assert result.steps[0].material_level_set_reinitialized


def test_builtin_single_plane_level_set_rejects_periodic_x_boundary() -> None:
    base = small_config()
    with pytest.raises(ReactiveConfigurationError, match="single-plane"):
        replace(
            base,
            algorithms={**base.algorithms, "level_set_boundary_x": "PERIODIC"},
        )


def test_strict_paper_cannot_be_promoted_from_assumed_runtime_object() -> None:
    assumed = small_config()
    promoted = replace(assumed, strict_paper=True)
    with pytest.raises(
        ReactiveConfigurationError, match="Strict paper mode rejects stale"
    ):
        ReactiveSolver(
            promoted,
            prescribed_electrical=source_fields((8, 8)),
        )


def test_nonstrict_runtime_overrides_are_audited_with_effective_values() -> None:
    config = small_config(reaction_heat_J_per_kg=123.0)
    result = ReactiveSolver(
        config, prescribed_electrical=source_fields((8, 8))
    ).run()
    audit = result.metadata["runtime_configuration_provenance_audit"]
    assert audit["status"] == (
        "NONSTRICT_RUNTIME_OVERRIDES_EXPLICITLY_CLASSIFIED_AS_ASSUMED"
    )
    assert audit["overrides"]["reaction_heat_J_per_kg"][
        "effective_value"
    ] == 123.0
    assert audit["overrides"]["reaction_heat_J_per_kg"]["provenance"] == (
        "ASSUMED_NOT_FROM_PAPER"
    )
    assert audit["effective_runtime_configuration"][
        "reaction_heat_J_per_kg"
    ] == 123.0
    assert result.metadata["source_configuration_provenance"]["quantities"][
        "reaction.heat_release"
    ]["value"] != 123.0
    assert "provenance" not in result.metadata


def test_prescribed_nonfinite_source_is_rejected_before_any_step() -> None:
    config = small_config()
    fields = source_fields((8, 8))
    fields.electrochemical_heat_W_per_m3[2, 3] = np.nan
    with pytest.raises(ReactiveNumericalError, match="NaN or infinity"):
        ReactiveSolver(config, prescribed_electrical=fields)


def test_solver_owned_prescribed_source_metadata_and_arrays_are_not_mutable() -> None:
    solver = ReactiveSolver(
        small_config(), prescribed_electrical=source_fields((8, 8))
    )
    assert solver.prescribed_electrical is not None
    with pytest.raises(TypeError):
        solver.prescribed_electrical.metadata["source"] = "forged"  # type: ignore[index]
    with pytest.raises(TypeError):
        solver.prescribed_electrical.metadata["field_sha256"][
            "joule_heat_W_per_m3"
        ] = "forged"  # type: ignore[index]
    with pytest.raises(ValueError):
        solver.prescribed_electrical.joule_heat_W_per_m3.setflags(write=True)


def test_solid_remaining_progress_limits_timestep_without_clipping() -> None:
    config = small_config(
        end_time_s=6.0e-3,
        maximum_steps=20,
        tait_B_Pa=1.0e-12,
        initial_alpha=0.99,
        decomposition_preexponential_per_s=100.0,
        decomposition_activation_energy_J_per_mol=0.0,
        decomposition_order=1.0,
        decomposition_heat_J_per_kg=0.0,
    )
    result = ReactiveSolver(
        config, prescribed_electrical=source_fields((8, 8))
    ).run()
    assert result.steps[0].selected_time_step_limit == "solid_stability"
    assert result.steps[0].solid_stability_limit_s == pytest.approx(5.0e-3)
    assert np.all(result.solid_decomposition_alpha >= 0.0)
    assert np.all(result.solid_decomposition_alpha <= 1.0)
    assert result.metadata["solid_time_step_policy"][
        "remaining_progress_headroom"
    ] == config.solid_progress_headroom


@pytest.mark.parametrize("reaction_order", [0.0, 0.5])
def test_near_complete_flow_reaction_retries_transactionally(
    reaction_order: float,
) -> None:
    initial_progress = 0.99
    preexponential = 100.0
    initial_rate = preexponential * (1.0 - initial_progress) ** reaction_order
    initial_headroom_limit = (
        small_config().flow_progress_headroom
        * (1.0 - initial_progress)
        / initial_rate
    )
    end_time = 1.2 * initial_headroom_limit
    config = small_config(
        end_time_s=end_time,
        maximum_steps=100,
        tait_B_Pa=1.0e-12,
        initial_reaction_progress=initial_progress,
        reaction_preexponential_per_s=preexponential,
        reaction_activation_energy_J_per_mol=0.0,
        reaction_order=reaction_order,
    )
    result = ReactiveSolver(
        config, prescribed_electrical=source_fields((8, 8))
    ).run()
    repeated = ReactiveSolver(
        config, prescribed_electrical=source_fields((8, 8))
    ).run()
    progress = (
        result.conservative_state[..., REACTION_PROGRESS]
        / result.conservative_state[..., RHO]
    )
    if reaction_order == 0.0:
        expected = initial_progress + preexponential * end_time
        tolerance = 3.0e-15
    else:
        remaining_root = math.sqrt(1.0 - initial_progress) - 0.5 * preexponential * end_time
        expected = 1.0 - remaining_root**2
        tolerance = 2.0e-6
    np.testing.assert_allclose(progress, expected, rtol=0.0, atol=tolerance)
    assert result.final_time_s == end_time
    assert result.metadata["stage_retry_count"] >= 1
    assert result.metadata[
        "rejected_trial_electrical_source_evaluation_count"
    ] >= 1
    assert sum(step.stage_retry_count for step in result.steps) == result.metadata[
        "stage_retry_count"
    ]
    assert max(abs(value) for value in result.conservation.closure_residual) < 2.0e-12
    assert result.metadata["cell_state_clipping_count"] == 0
    assert np.all((progress >= 0.0) & (progress <= 1.0))
    np.testing.assert_array_equal(result.step_dt_s, repeated.step_dt_s)
    np.testing.assert_array_equal(
        result.conservative_state, repeated.conservative_state
    )
    retried = [step for step in result.steps if step.stage_retry_count > 0]
    assert retried
    assert all(step.selected_time_step_limit.startswith("rk_stage_retry_") for step in retried)
    assert all(step.proposed_time_step_limit != step.selected_time_step_limit for step in retried)


@pytest.mark.parametrize("reaction_order", [0.0, 0.5])
def test_near_complete_solid_reaction_retries_transactionally(
    reaction_order: float,
) -> None:
    initial_alpha = 0.99
    preexponential = 100.0
    initial_rate = preexponential * (1.0 - initial_alpha) ** reaction_order
    initial_headroom_limit = (
        small_config().solid_progress_headroom
        * (1.0 - initial_alpha)
        / initial_rate
    )
    end_time = 1.2 * initial_headroom_limit
    config = small_config(
        end_time_s=end_time,
        maximum_steps=100,
        tait_B_Pa=1.0e-12,
        initial_alpha=initial_alpha,
        decomposition_preexponential_per_s=preexponential,
        decomposition_activation_energy_J_per_mol=0.0,
        decomposition_order=reaction_order,
        decomposition_heat_J_per_kg=0.0,
    )
    result = ReactiveSolver(
        config, prescribed_electrical=source_fields((8, 8))
    ).run()
    repeated = ReactiveSolver(
        config, prescribed_electrical=source_fields((8, 8))
    ).run()
    if reaction_order == 0.0:
        expected = initial_alpha + preexponential * end_time
        tolerance = 3.0e-15
    else:
        remaining_root = math.sqrt(1.0 - initial_alpha) - 0.5 * preexponential * end_time
        expected = 1.0 - remaining_root**2
        tolerance = 2.0e-6
    np.testing.assert_allclose(
        result.solid_decomposition_alpha, expected, rtol=0.0, atol=tolerance
    )
    assert result.final_time_s == end_time
    assert result.metadata["stage_retry_count"] >= 1
    assert abs(result.solid_budget.alpha_closure_residual_m2) < 2.0e-18
    assert result.metadata["cell_state_clipping_count"] == 0
    assert np.all(
        (result.solid_decomposition_alpha >= 0.0)
        & (result.solid_decomposition_alpha <= 1.0)
    )
    np.testing.assert_array_equal(result.step_dt_s, repeated.step_dt_s)
    np.testing.assert_array_equal(
        result.solid_decomposition_alpha,
        repeated.solid_decomposition_alpha,
    )


def test_stage_retry_budget_exhaustion_fails_without_returning_partial_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = small_config(
        end_time_s=6.0e-5,
        maximum_steps=10,
        maximum_stage_retries=1,
        tait_B_Pa=1.0e-12,
        initial_reaction_progress=0.99,
        reaction_preexponential_per_s=100.0,
        reaction_activation_energy_J_per_mol=0.0,
        reaction_order=0.0,
    )

    def deliberately_insufficient_reduction(
        current_dt_s: float, _: ReactiveNumericalError
    ) -> float:
        return float(np.nextafter(current_dt_s, 0.0))

    monkeypatch.setattr(
        ReactiveSolver,
        "_reduced_retry_timestep",
        staticmethod(deliberately_insufficient_reduction),
    )
    with pytest.raises(ReactiveNumericalError) as captured:
        ReactiveSolver(
            config, prescribed_electrical=source_fields((8, 8))
        ).run()
    assert captured.value.category == "stage_retry_budget_exhausted"
    assert captured.value.diagnostics["maximum_stage_retries"] == 1


@pytest.mark.parametrize("reaction_order", [0.0, 0.5])
def test_flow_tolerance_terminal_runs_beyond_nominal_completion(
    reaction_order: float,
) -> None:
    initial_progress = 0.9
    preexponential = 1.0
    nominal_completion_time = (1.0 - initial_progress) ** (
        1.0 - reaction_order
    ) / (preexponential * (1.0 - reaction_order))
    end_time = nominal_completion_time + 0.01
    heat_per_progress = 75.0
    config = small_config(
        end_time_s=end_time,
        maximum_steps=250,
        tait_B_Pa=1.0e-12,
        initial_reaction_progress=initial_progress,
        reaction_preexponential_per_s=preexponential,
        reaction_activation_energy_J_per_mol=0.0,
        reaction_order=reaction_order,
        reaction_heat_J_per_kg=heat_per_progress,
    )
    first = ReactiveSolver(
        config, prescribed_electrical=source_fields((8, 8))
    ).run()
    second = ReactiveSolver(
        config, prescribed_electrical=source_fields((8, 8))
    ).run()
    progress = (
        first.conservative_state[..., REACTION_PROGRESS]
        / first.conservative_state[..., RHO]
    )
    remaining = 1.0 - progress
    terminal = first.metadata["completion_tolerance_rate_zeroing_flow"]
    assert first.final_time_s == pytest.approx(end_time, rel=0.0, abs=0.0)
    assert np.all(remaining > 0.0)
    assert np.max(remaining) <= config.progress_tolerance
    assert terminal["positive_remainder_terminal_cell_count"] == 64
    assert terminal["accepted_stage_zeroed_cell_evaluation_count"] > 0
    assert terminal["maximum_unreacted_remainder_in_terminal_cells"] == pytest.approx(
        float(np.max(remaining)), rel=0.0, abs=0.0
    )
    assert terminal["final_in_domain_terminal_remainder_inventory_kg_per_m"] > 0.0
    assert terminal[
        "final_in_domain_terminal_remainder_heat_magnitude_J_per_m"
    ] == pytest.approx(
        heat_per_progress
        * terminal["final_in_domain_terminal_remainder_inventory_kg_per_m"]
    )
    assert first.conservation.reaction_bookkeeping_residual_J_per_m == pytest.approx(
        0.0, abs=2.0e-15
    )
    assert first.metadata["reaction_rate_cap_application_count"] == 0
    assert first.metadata["completion_tolerance_terminal_policy"][
        "deliberate_terminal_cutoff"
    ] is True
    np.testing.assert_array_equal(first.step_dt_s, second.step_dt_s)
    np.testing.assert_array_equal(
        first.conservative_state, second.conservative_state
    )


@pytest.mark.parametrize("reaction_order", [0.0, 0.5])
def test_solid_tolerance_terminal_runs_beyond_nominal_completion(
    reaction_order: float,
) -> None:
    initial_alpha = 0.9
    preexponential = 1.0
    nominal_completion_time = (1.0 - initial_alpha) ** (
        1.0 - reaction_order
    ) / (preexponential * (1.0 - reaction_order))
    end_time = nominal_completion_time + 0.01
    decomposition_heat = 25.0
    config = small_config(
        end_time_s=end_time,
        maximum_steps=250,
        tait_B_Pa=1.0e-12,
        initial_alpha=initial_alpha,
        decomposition_preexponential_per_s=preexponential,
        decomposition_activation_energy_J_per_mol=0.0,
        decomposition_order=reaction_order,
        decomposition_heat_J_per_kg=decomposition_heat,
    )
    first = ReactiveSolver(
        config, prescribed_electrical=source_fields((8, 8))
    ).run()
    second = ReactiveSolver(
        config, prescribed_electrical=source_fields((8, 8))
    ).run()
    remaining = 1.0 - first.solid_decomposition_alpha
    terminal = first.metadata["completion_tolerance_rate_zeroing_solid"]
    assert first.final_time_s == pytest.approx(end_time, rel=0.0, abs=0.0)
    assert np.all(remaining > 0.0)
    assert np.max(remaining) <= config.progress_tolerance
    assert terminal["positive_remainder_terminal_cell_count"] == 64
    assert terminal["accepted_stage_zeroed_cell_evaluation_count"] > 0
    assert terminal["maximum_unreacted_remainder_in_terminal_cells"] == pytest.approx(
        float(np.max(remaining)), rel=0.0, abs=0.0
    )
    assert terminal["final_in_domain_terminal_remainder_inventory_kg_per_m"] > 0.0
    assert terminal[
        "final_in_domain_terminal_remainder_heat_magnitude_J_per_m"
    ] == pytest.approx(
        decomposition_heat
        * terminal["final_in_domain_terminal_remainder_inventory_kg_per_m"]
    )
    expected_decomposition_energy = (
        -config.solid_density_kg_per_m3
        * decomposition_heat
        * first.solid_budget.rate_increment_area_integral_m2
    )
    assert first.solid_budget.decomposition_heat_increment_J_per_m == pytest.approx(
        expected_decomposition_energy, rel=2.0e-13
    )
    assert first.metadata["cell_state_clipping_count"] == 0
    np.testing.assert_array_equal(first.step_dt_s, second.step_dt_s)
    np.testing.assert_array_equal(
        first.solid_decomposition_alpha,
        second.solid_decomposition_alpha,
    )


def test_face_fallback_maximum_correction_reaches_result_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = solver_module.finite_volume_rhs

    def instrumented(*args, **kwargs):
        rhs, diagnostics = original(*args, **kwargs)
        return rhs, replace(
            diagnostics,
            positivity_face_fallback_count=1,
            positivity_face_fallback_max_abs_state_correction=42.0,
        )

    monkeypatch.setattr(solver_module, "finite_volume_rhs", instrumented)
    config = small_config(initial_reaction_progress=1.0, initial_alpha=1.0)
    result = ReactiveSolver(
        config, prescribed_electrical=source_fields((8, 8))
    ).run()
    assert result.metadata["positivity_face_fallback_count"] == 6
    assert result.metadata[
        "positivity_face_fallback_max_abs_state_correction"
    ] == 42.0
    assert result.steps[0].positivity_face_fallback_count == 6
    assert result.steps[0].positivity_face_fallback_max_abs_state_correction == 42.0
    assert result.metadata["cell_state_clipping_count"] == 0


def test_budget_scales_cells_before_reduction_and_never_serializes_infinity() -> None:
    config = small_config(end_time_s=1.0e-20)
    source = 1.0e307
    result = ReactiveSolver(
        config,
        prescribed_electrical=source_fields((8, 8), joule=source),
    ).run()
    expected = source * config.end_time_s * config.lx_m * config.ly_m
    assert math.isfinite(result.conservation.electrical_energy_increment_J_per_m)
    assert result.conservation.electrical_energy_increment_J_per_m == pytest.approx(
        expected, rel=2.0e-15
    )
    assert all(math.isfinite(value) for value in result.conservation.closure_residual)
    summary = result.summary()
    encoded = json.dumps(summary, allow_nan=False)
    assert "Infinity" not in encoded
    first_step = summary["steps"][0]
    assert first_step["flow_reaction_limit_s"] is None
    assert first_step["flow_reaction_limit_status"] == "UNBOUNDED"


def test_result_history_resource_preflight_fails_before_allocating_run_state() -> None:
    config = small_config(maximum_history_bytes=10_000)
    solver = ReactiveSolver(config, prescribed_electrical=source_fields((8, 8)))
    with pytest.raises(
        ReactiveNumericalError, match="result-history memory budget"
    ) as captured:
        solver.run()
    assert captured.value.category == "projected_result_history_exceeds_limit"
    assert captured.value.diagnostics["projected_total_bytes"] > 10_000


def test_prescribed_metadata_is_included_in_history_preflight() -> None:
    config = small_config(maximum_steps=1, maximum_history_bytes=25_000)
    solver = ReactiveSolver(config, prescribed_electrical=source_fields((8, 8)))
    with pytest.raises(ReactiveNumericalError) as captured:
        solver.run()
    assert captured.value.category == "projected_result_history_exceeds_limit"
    assert (
        captured.value.diagnostics["electrical_stage_metadata_charge_basis"]
        == "PRESCRIBED_COMPLETE_STAGE_RECORD_MEASURED_DEEP_PLUS_JSON"
    )


def test_grid_working_set_preflight_fails_before_grid_allocation() -> None:
    config = small_config(
        nx=1_000_000_000,
        maximum_working_set_bytes=512 * 1024**2,
    )
    with pytest.raises(
        ReactiveNumericalError, match="before grid allocation"
    ) as captured:
        ReactiveSolver(config, prescribed_electrical=source_fields((8, 8)))
    assert captured.value.category == "projected_working_set_exceeds_limit"
    assert captured.value.diagnostics["cell_count"] == 8_000_000_000


def test_callback_metadata_is_runtime_bounded_by_history_resource_limit() -> None:
    config = extended_config(maximum_history_bytes=4_000_000)
    anode, cathode = surface_contacts()

    def oversized_metadata(_: ElectricalStageRequest) -> ElectricalSourceFields:
        fields = source_fields(
            (8, 8), provider="EXISTING_BC_GLOBAL_CALLBACK"
        )
        fields.metadata["oversized_payload"] = "x" * 2_100_000
        return fields

    with pytest.raises(
        ReactiveNumericalError, match="metadata record exceeds"
    ) as captured:
        ReactiveSolver(
            config,
            electrical_callback=oversized_metadata,
            anode_contact=anode,
            cathode_contact=cathode,
        ).run()
    assert captured.value.category == (
        "electrical_metadata_record_exceeds_history_limit"
    )
