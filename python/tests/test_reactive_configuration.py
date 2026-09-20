from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import copy
import math

import pytest
import yaml

from ecsp_reactive.configuration import load_reactive_config, parse_reactive_config
from ecsp_reactive.provenance import ReactiveConfigurationError


ROOT = Path(__file__).resolve().parents[2]


def test_assumed_paper_case_is_runnable_but_self_identifies_assumptions() -> None:
    config = load_reactive_config(ROOT / "config/paper_faithful_assumed_v8_3_0.yaml")
    assert config.model_mode == "paper_faithful"
    assert config.strict_paper is False
    assert config.nx == config.ny == 50
    assert config.butler_volmer_enabled is False
    assert config.electrical_source_provider == "CONFIG_UNIFORM_FIELD"
    assert config.algorithms["mpi_decomposition"] == "SERIAL_SINGLE_PROCESS"
    assert config.provenance_report["assumed_quantity_count"] > 0
    assert config.provenance_report["assumed_algorithm_count"] > 0
    assert config.front_thresholds == (0.3, 0.5, 0.7)
    assert config.solid_progress_headroom == 0.5
    assert config.level_set_weno_relative_epsilon == 1.0e-12
    assert config.level_set_weno_absolute_epsilon_m2 == 1.0e-40
    assert config.maximum_stage_retries == 32
    assert config.front_history_stride_steps == 1
    assert config.maximum_history_bytes == 268435456
    assert config.algorithms["flow_boundary"] == "OUTFLOW"
    assert config.algorithms["solid_boundary"] == "OUTFLOW"
    assert config.algorithms["level_set_boundary_x"] == "OUTFLOW"
    assert config.algorithms["level_set_boundary_y"] == "OUTFLOW"
    assert (
        config.algorithms["electrical_gradient_boundary"]
        == "FIRST_ORDER_ONE_SIDED"
    )
    retry_quantity = config.provenance_report["quantities"][
        "time.maximum_stage_retries"
    ]
    assert retry_quantity["unit"] == "attempts"
    assert retry_quantity["provenance"] == "ASSUMED_NOT_FROM_PAPER"
    stride_quantity = config.provenance_report["quantities"][
        "output.front_history_stride_steps"
    ]
    memory_quantity = config.provenance_report["quantities"][
        "output.maximum_history_bytes"
    ]
    assert stride_quantity["unit"] == "steps"
    assert stride_quantity["provenance"] == "ASSUMED_NOT_FROM_PAPER"
    assert memory_quantity["unit"] == "bytes"
    assert memory_quantity["provenance"] == "ASSUMED_NOT_FROM_PAPER"
    assert config.provenance_report["not_consumed_inputs"] == {}
    tolerance_quantity = config.provenance_report["quantities"][
        "safety.progress_tolerance"
    ]
    assert tolerance_quantity["provenance"] == "ASSUMED_NOT_FROM_PAPER"
    assert "terminal chemistry cutoff" in tolerance_quantity["source"]


def test_strict_paper_template_fails_closed_at_first_unresolved_input() -> None:
    with pytest.raises(ReactiveConfigurationError, match="UNRESOLVED"):
        load_reactive_config(ROOT / "config/paper_faithful_required_inputs_v8_3_0.yaml")


def test_extended_config_is_explicit_inherited_nonpaper_mode() -> None:
    config = load_reactive_config(ROOT / "config/ecsp_extended_reactive_v8_3_0.yaml")
    assert config.model_mode == "ecsp_extended"
    assert config.butler_volmer_enabled is True
    assert config.electrical_source_provider == "EXISTING_BC_GLOBAL_CALLBACK"
    assert config.conductivity_S_per_m is None
    assert config.electric_field_x_V_per_m is None
    assert config.electric_field_y_V_per_m is None
    assert config.electrochemical_heat_W_per_m3 is None
    assert set(config.provenance_report["not_consumed_inputs"]) == {
        "electrical.conductivity",
        "electrical.electric_field_x",
        "electrical.electric_field_y",
        "electrical.electrochemical_heat",
    }
    assert "electrical.conductivity" not in config.provenance_report["quantities"]
    assert config.provenance_report["algorithms"]["electrical_source_provider"][
        "provenance"
    ] == "ASSUMED_NOT_FROM_PAPER"
    for name in (
        "solid_boundary",
        "level_set_boundary_x",
        "level_set_boundary_y",
        "electrical_gradient_boundary",
    ):
        assert name in config.algorithms
        assert config.provenance_report["algorithms"][name]["provenance"] == (
            "ASSUMED_NOT_FROM_PAPER"
        )


def test_inherited_quantity_and_algorithm_records_are_atomic(tmp_path: Path) -> None:
    base = ROOT / "config" / "paper_faithful_assumed_v8_3_0.yaml"
    quantity_child = tmp_path / "quantity.yaml"
    quantity_child.write_text(
        f"extends: {base}\nreaction:\n  heat_release:\n    value: 123\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ReactiveConfigurationError, match="Missing required configuration key"
    ):
        load_reactive_config(quantity_child)

    algorithm_child = tmp_path / "algorithm.yaml"
    algorithm_child.write_text(
        f"extends: {base}\nnumerics:\n  algorithms:\n    flow_boundary:\n"
        "      choice: PERIODIC\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ReactiveConfigurationError, match="Missing required configuration key"
    ):
        load_reactive_config(algorithm_child)


def test_quantity_boolean_is_rejected_instead_of_becoming_one() -> None:
    raw = yaml.safe_load(
        (ROOT / "config/paper_faithful_assumed_v8_3_0.yaml").read_text(
            encoding="utf-8"
        )
    )
    modified = copy.deepcopy(raw)
    modified["grid"]["nx"]["value"] = True
    with pytest.raises(ReactiveConfigurationError, match="not a boolean"):
        parse_reactive_config(modified)


def test_reinitialization_algorithm_cannot_enable_when_interval_is_zero() -> None:
    raw = yaml.safe_load(
        (ROOT / "config/paper_faithful_assumed_v8_3_0.yaml").read_text(
            encoding="utf-8"
        )
    )
    modified = copy.deepcopy(raw)
    modified["numerics"]["algorithms"]["level_set_reinitialization"][
        "choice"
    ] = "SUSSMAN_FIRST_ORDER"
    with pytest.raises(
        ReactiveConfigurationError,
        match=r"interval=0 requires .*level_set_reinitialization=NONE",
    ):
        parse_reactive_config(modified)


def test_reinitialization_interval_cannot_run_when_algorithm_is_none() -> None:
    raw = yaml.safe_load(
        (ROOT / "config/paper_faithful_assumed_v8_3_0.yaml").read_text(
            encoding="utf-8"
        )
    )
    modified = copy.deepcopy(raw)
    modified["level_set"]["reinitialization_interval"]["value"] = 5
    with pytest.raises(
        ReactiveConfigurationError,
        match=(
            r"interval=5 requires .*level_set_reinitialization="
            r"SUSSMAN_FIRST_ORDER, received NONE"
        ),
    ):
        parse_reactive_config(modified)


@pytest.mark.parametrize(
    ("value", "message"),
    [(0, "must be positive"), (1.5, "must be an exact integer")],
)
def test_maximum_stage_retries_is_an_exact_positive_integer(
    value: float, message: str
) -> None:
    raw = yaml.safe_load(
        (ROOT / "config/paper_faithful_assumed_v8_3_0.yaml").read_text(
            encoding="utf-8"
        )
    )
    modified = copy.deepcopy(raw)
    modified["time"]["maximum_stage_retries"]["value"] = value
    with pytest.raises(ReactiveConfigurationError, match=message):
        parse_reactive_config(modified)


def test_strict_template_marks_stage_retry_budget_unresolved() -> None:
    raw = yaml.safe_load(
        (ROOT / "config/paper_faithful_required_inputs_v8_3_0.yaml").read_text(
            encoding="utf-8"
        )
    )
    retry = raw["time"]["maximum_stage_retries"]
    assert retry["unit"] == "attempts"
    assert retry["provenance"] == "UNRESOLVED"

    output = raw["output"]
    assert output["front_history_stride_steps"]["unit"] == "steps"
    assert output["front_history_stride_steps"]["provenance"] == "UNRESOLVED"
    assert output["maximum_history_bytes"]["unit"] == "bytes"
    assert output["maximum_history_bytes"]["provenance"] == "UNRESOLVED"

    algorithms = raw["numerics"]["algorithms"]
    for name in (
        "flow_boundary",
        "solid_boundary",
        "level_set_boundary_x",
        "level_set_boundary_y",
        "electrical_gradient_boundary",
    ):
        assert algorithms[name]["provenance"] == "UNRESOLVED"
    assert algorithms["electrical_gradient_boundary"]["choice"] == (
        "FIRST_ORDER_ONE_SIDED"
    )
    assert algorithms["mpi_decomposition"]["choice"] == "SERIAL_SINGLE_PROCESS"
    assert raw["electrical"]["source_provider"]["choice"] == (
        "PRESCRIBED_ARRAY_FIELDS"
    )
    assert raw["electrical"]["source_provider"]["provenance"] == "UNRESOLVED"


@pytest.mark.parametrize(
    ("entry", "value", "message"),
    [
        ("front_history_stride_steps", 0, "must be positive"),
        ("front_history_stride_steps", 1.5, "must be an exact integer"),
        ("maximum_history_bytes", 0, "must be positive"),
        ("maximum_history_bytes", 1.5, "must be an exact integer"),
    ],
)
def test_output_resource_limits_are_exact_positive_integers(
    entry: str, value: float, message: str
) -> None:
    raw = yaml.safe_load(
        (ROOT / "config/paper_faithful_assumed_v8_3_0.yaml").read_text(
            encoding="utf-8"
        )
    )
    modified = copy.deepcopy(raw)
    modified["output"][entry]["value"] = value
    with pytest.raises(ReactiveConfigurationError, match=message):
        parse_reactive_config(modified)


def test_typo_in_extended_child_is_not_masked_by_inherited_value(
    tmp_path: Path,
) -> None:
    child = {
        "extends": str(ROOT / "config/paper_faithful_assumed_v8_3_0.yaml"),
        "reaction": {
            "preexponentail": {
                "value": 2.0e5,
                "unit": "1/s",
                "provenance": "ASSUMED_NOT_FROM_PAPER",
                "source": "deliberate misspelling for fail-closed regression",
            }
        },
    }
    path = tmp_path / "typo_child.yaml"
    path.write_text(yaml.safe_dump(child), encoding="utf-8")
    with pytest.raises(ReactiveConfigurationError, match=r"reaction\.preexponentail"):
        load_reactive_config(path)


def test_arbitrary_top_level_key_is_rejected() -> None:
    raw = yaml.safe_load(
        (ROOT / "config/paper_faithful_assumed_v8_3_0.yaml").read_text(
            encoding="utf-8"
        )
    )
    modified = copy.deepcopy(raw)
    modified["unexpected_root_option"] = True
    with pytest.raises(
        ReactiveConfigurationError,
        match=r"root\.unexpected_root_option",
    ):
        parse_reactive_config(modified)


def test_non_string_configuration_key_is_rejected_as_configuration_error() -> None:
    raw = yaml.safe_load(
        (ROOT / "config/paper_faithful_assumed_v8_3_0.yaml").read_text(
            encoding="utf-8"
        )
    )
    modified = copy.deepcopy(raw)
    modified[7] = "must not reach mixed-type sorting"
    with pytest.raises(
        ReactiveConfigurationError,
        match=r"non-string configuration key.*7.*int",
    ):
        parse_reactive_config(modified)


def test_quantity_object_extra_key_is_rejected() -> None:
    raw = yaml.safe_load(
        (ROOT / "config/paper_faithful_assumed_v8_3_0.yaml").read_text(
            encoding="utf-8"
        )
    )
    modified = copy.deepcopy(raw)
    modified["grid"]["nx"]["comment_typo"] = "must not be ignored"
    with pytest.raises(
        ReactiveConfigurationError,
        match=r"grid\.nx\.comment_typo",
    ):
        parse_reactive_config(modified)


def test_duplicate_key_in_main_yaml_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate_main.yaml"
    path.write_text(
        "schema: ecsp.paper-reactive/v1\n"
        "schema: ecsp.paper-reactive/v1\n",
        encoding="utf-8",
    )
    with pytest.raises(ReactiveConfigurationError, match="Duplicate YAML mapping key"):
        load_reactive_config(path)


def test_duplicate_key_in_extended_base_yaml_is_rejected(tmp_path: Path) -> None:
    base = tmp_path / "duplicate_base.yaml"
    base.write_text(
        "schema: ecsp.paper-reactive/v1\n"
        "schema: ecsp.paper-reactive/v1\n",
        encoding="utf-8",
    )
    child = tmp_path / "child.yaml"
    child.write_text("extends: duplicate_base.yaml\n", encoding="utf-8")
    with pytest.raises(ReactiveConfigurationError, match="Duplicate YAML mapping key"):
        load_reactive_config(child)


@pytest.mark.parametrize(
    ("name", "unsupported"),
    [
        ("solid_boundary", "MIRROR"),
        ("level_set_boundary_x", "MIRROR"),
        ("level_set_boundary_y", "MIRROR"),
        ("electrical_gradient_boundary", "PERIODIC"),
    ],
)
def test_independent_boundary_algorithms_reject_unsupported_choices(
    name: str,
    unsupported: str,
) -> None:
    raw = yaml.safe_load(
        (ROOT / "config/paper_faithful_assumed_v8_3_0.yaml").read_text(
            encoding="utf-8"
        )
    )
    modified = copy.deepcopy(raw)
    modified["numerics"]["algorithms"][name]["choice"] = unsupported
    with pytest.raises(ReactiveConfigurationError, match=rf"Unsupported {name}="):
        parse_reactive_config(modified)


@pytest.mark.parametrize(
    "missing_name",
    [
        "conductivity",
        "electric_field_x",
        "electric_field_y",
        "electrochemical_heat",
    ],
)
def test_config_uniform_field_requires_every_scalar_quantity(
    missing_name: str,
) -> None:
    raw = yaml.safe_load(
        (ROOT / "config/paper_faithful_assumed_v8_3_0.yaml").read_text(
            encoding="utf-8"
        )
    )
    del raw["electrical"][missing_name]
    with pytest.raises(
        ReactiveConfigurationError,
        match=rf"Missing required configuration entry electrical\.{missing_name}",
    ):
        parse_reactive_config(raw)


def test_prescribed_array_provider_allows_omitting_uniform_scalars() -> None:
    raw = yaml.safe_load(
        (ROOT / "config/paper_faithful_assumed_v8_3_0.yaml").read_text(
            encoding="utf-8"
        )
    )
    electrical = raw["electrical"]
    electrical["source_provider"] = {
        "choice": "PRESCRIBED_ARRAY_FIELDS",
        "provenance": "MEASURED",
        "source": "synthetic measured-array provider contract test",
    }
    for name in (
        "conductivity",
        "electric_field_x",
        "electric_field_y",
        "electrochemical_heat",
    ):
        del electrical[name]
    config = parse_reactive_config(raw)
    assert config.electrical_source_provider == "PRESCRIBED_ARRAY_FIELDS"
    assert config.conductivity_S_per_m is None
    assert config.electric_field_x_V_per_m is None
    assert config.electric_field_y_V_per_m is None
    assert config.electrochemical_heat_W_per_m3 is None
    assert config.provenance_report["not_consumed_inputs"] == {}


def test_prescribed_array_provider_marks_present_uniform_scalars_not_consumed() -> None:
    raw = yaml.safe_load(
        (ROOT / "config/paper_faithful_assumed_v8_3_0.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["electrical"]["source_provider"] = {
        "choice": "PRESCRIBED_ARRAY_FIELDS",
        "provenance": "MEASURED",
        "source": "synthetic measured-array provider contract test",
    }
    config = parse_reactive_config(raw)
    ignored = config.provenance_report["not_consumed_inputs"]
    assert set(ignored) == {
        "electrical.conductivity",
        "electrical.electric_field_x",
        "electrical.electric_field_y",
        "electrical.electrochemical_heat",
    }
    assert all(row["status"] == "IGNORED_NOT_CONSUMED" for row in ignored.values())
    assert config.conductivity_S_per_m is None


def test_unknown_electrical_source_provider_fails_closed() -> None:
    raw = yaml.safe_load(
        (ROOT / "config/paper_faithful_assumed_v8_3_0.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["electrical"]["source_provider"]["choice"] = "TYPO_FIELDS"
    with pytest.raises(
        ReactiveConfigurationError,
        match="Unsupported electrical source provider",
    ):
        parse_reactive_config(raw)


def test_mpi_algorithm_supports_serial_and_future_rowwise_decomposition() -> None:
    raw = yaml.safe_load(
        (ROOT / "config/paper_faithful_assumed_v8_3_0.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["numerics"]["algorithms"]["mpi_decomposition"]["choice"] = (
        "ROWWISE_WENO3_HALO"
    )
    config = parse_reactive_config(raw)
    assert config.algorithms["mpi_decomposition"] == "ROWWISE_WENO3_HALO"


def test_unknown_mpi_algorithm_fails_closed() -> None:
    raw = yaml.safe_load(
        (ROOT / "config/paper_faithful_assumed_v8_3_0.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["numerics"]["algorithms"]["mpi_decomposition"]["choice"] = "AUTO_MPI"
    with pytest.raises(
        ReactiveConfigurationError,
        match="Unsupported mpi_decomposition",
    ):
        parse_reactive_config(raw)


@pytest.mark.parametrize("value", [1.0, 2.0, math.nan])
def test_progress_tolerance_must_be_finite_and_strictly_below_one(
    value: float,
) -> None:
    raw = yaml.safe_load(
        (ROOT / "config/paper_faithful_assumed_v8_3_0.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw["safety"]["progress_tolerance"]["value"] = value
    with pytest.raises(ReactiveConfigurationError, match="progress_tolerance"):
        parse_reactive_config(raw)


def test_configuration_nested_state_is_detached_and_recursively_immutable() -> None:
    raw = yaml.safe_load(
        (ROOT / "config/paper_faithful_assumed_v8_3_0.yaml").read_text(
            encoding="utf-8"
        )
    )
    config = parse_reactive_config(raw)
    raw["grid"]["nx"]["value"] = 999
    assert config.raw["grid"]["nx"]["value"] == 50

    with pytest.raises(TypeError):
        config.algorithms["flow_boundary"] = "PERIODIC"  # type: ignore[index]
    with pytest.raises(TypeError):
        config.provenance_report["quantities"]["grid.nx"]["value"] = 999  # type: ignore[index]
    with pytest.raises(TypeError):
        config.raw["grid"]["nx"]["value"] = 999  # type: ignore[index]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"nx": 8.0}, "nx must be an exact integer"),
        ({"nx": 7}, "at least 8 cells"),
        ({"end_time_s": 0.0}, "end_time_s must be positive"),
        ({"initial_u_m_per_s": math.nan}, "initial_u_m_per_s must be finite"),
        ({"cfl": 1.1}, "cfl must not exceed 1"),
        ({"maximum_steps": 0}, "maximum_steps must be positive"),
        ({"maximum_stage_retries": 0}, "maximum_stage_retries must be positive"),
        ({"front_history_stride_steps": 0}, "front_history_stride_steps must be positive"),
        ({"maximum_history_bytes": 0}, "maximum_history_bytes must be positive"),
        (
            {"maximum_flow_progress_increment": 1.1},
            "maximum_flow_progress_increment must not exceed 1",
        ),
        ({"initial_reaction_progress": 1.1}, "must be in \\[0,1\\]"),
        ({"interface_x_m": 0.0}, "interface must lie inside"),
        ({"front_thresholds": (0.5, 0.5)}, "front thresholds must be unique"),
        ({"progress_tolerance": 1.0}, "progress_tolerance must satisfy"),
    ],
)
def test_dataclass_replace_cannot_bypass_numeric_runtime_contract(
    changes: dict[str, object],
    message: str,
) -> None:
    base = load_reactive_config(
        ROOT / "config/paper_faithful_assumed_v8_3_0.yaml"
    )
    with pytest.raises(ReactiveConfigurationError, match=message):
        replace(base, **changes)


def test_dataclass_replace_cannot_bypass_algorithm_or_reinit_contract() -> None:
    base = load_reactive_config(
        ROOT / "config/paper_faithful_assumed_v8_3_0.yaml"
    )
    with pytest.raises(ReactiveConfigurationError, match="Unsupported flow_boundary"):
        replace(base, algorithms={**base.algorithms, "flow_boundary": "TYPO"})
    with pytest.raises(
        ReactiveConfigurationError,
        match="Inconsistent level-set reinitialization",
    ):
        replace(base, reinitialization_interval_steps=1)


def test_dataclass_replace_enforces_provider_specific_optional_scalars() -> None:
    base = load_reactive_config(
        ROOT / "config/paper_faithful_assumed_v8_3_0.yaml"
    )
    with pytest.raises(
        ReactiveConfigurationError,
        match="PRESCRIBED_ARRAY_FIELDS requires uniform electrical scalar fields to be None",
    ):
        replace(base, electrical_source_provider="PRESCRIBED_ARRAY_FIELDS")

    arrays = replace(
        base,
        electrical_source_provider="PRESCRIBED_ARRAY_FIELDS",
        conductivity_S_per_m=None,
        electric_field_x_V_per_m=None,
        electric_field_y_V_per_m=None,
        electrochemical_heat_W_per_m3=None,
    )
    assert arrays.conductivity_S_per_m is None
    with pytest.raises(
        ReactiveConfigurationError,
        match="CONFIG_UNIFORM_FIELD requires conductivity_S_per_m",
    ):
        replace(arrays, electrical_source_provider="CONFIG_UNIFORM_FIELD")
