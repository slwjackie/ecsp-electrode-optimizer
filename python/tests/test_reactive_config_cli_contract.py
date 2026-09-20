from __future__ import annotations

from dataclasses import replace
import copy
import json
from pathlib import Path
import sys

import pytest
import yaml

from ecsp_reactive.cli import _parser, main
from ecsp_reactive.configuration import load_reactive_config, parse_reactive_config
from ecsp_reactive.provenance import ReactiveConfigurationError


ROOT = Path(__file__).resolve().parents[2]
ASSUMED_CONFIG = ROOT / "config" / "paper_faithful_assumed_v8_3_0.yaml"


def _raw_config() -> dict[str, object]:
    loaded = yaml.safe_load(ASSUMED_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return copy.deepcopy(loaded)


def test_working_set_limit_is_required_exact_and_runtime_validated() -> None:
    config = load_reactive_config(ASSUMED_CONFIG)
    assert config.maximum_working_set_bytes == 2 * 1024**3
    record = config.provenance_report["quantities"][
        "output.maximum_working_set_bytes"
    ]
    assert record["value"] == 2 * 1024**3
    assert record["unit"] == "bytes"
    assert record["provenance"] == "ASSUMED_NOT_FROM_PAPER"

    missing = _raw_config()
    del missing["output"]["maximum_working_set_bytes"]  # type: ignore[index]
    with pytest.raises(
        ReactiveConfigurationError,
        match=r"Missing required configuration entry output\.maximum_working_set_bytes",
    ):
        parse_reactive_config(missing)

    with pytest.raises(
        ReactiveConfigurationError,
        match="maximum_working_set_bytes must be positive",
    ):
        replace(config, maximum_working_set_bytes=0)


def test_integer_quantities_preserve_large_yaml_integer_tokens_exactly() -> None:
    raw = _raw_config()
    exact = 2**53 + 1
    raw["output"]["maximum_history_bytes"]["value"] = exact  # type: ignore[index]
    config = parse_reactive_config(raw)
    assert config.maximum_history_bytes == exact
    assert config.provenance_report["quantities"][
        "output.maximum_history_bytes"
    ]["value"] == exact


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (True, "not a boolean"),
        (1.25, "must be an exact integer"),
        (float(2**54), "exceeds the exact integer range"),
        (sys.maxsize + 1, "exceeds the platform integer range"),
    ],
)
def test_integer_quantities_fail_closed_on_ambiguous_or_overflowing_values(
    value: object,
    message: str,
) -> None:
    raw = _raw_config()
    raw["output"]["maximum_working_set_bytes"]["value"] = value  # type: ignore[index]
    with pytest.raises(ReactiveConfigurationError, match=message):
        parse_reactive_config(raw)


@pytest.mark.parametrize(
    ("field", "member", "value", "message"),
    [
        ("electric_field_x", "unit", "kV/m", "unit must be 'V/m'"),
        ("electric_field_y", "value", float("nan"), "must be finite"),
        ("electrochemical_heat", "provenance", "INVENTED", "Invalid value/provenance"),
        ("electrochemical_heat", "source", "   ", "requires a human-readable source"),
        ("electrochemical_heat", "source", 123, "human-readable string source"),
        ("conductivity", "value", -1.0, "must be non-negative"),
    ],
)
def test_nonuniform_provider_still_validates_ignored_uniform_records(
    field: str,
    member: str,
    value: object,
    message: str,
) -> None:
    raw = _raw_config()
    raw["electrical"]["source_provider"] = {  # type: ignore[index]
        "choice": "PRESCRIBED_ARRAY_FIELDS",
        "provenance": "MEASURED",
        "source": "explicit array-provider contract test",
    }
    raw["electrical"][field][member] = value  # type: ignore[index]
    with pytest.raises(ReactiveConfigurationError, match=message):
        parse_reactive_config(raw)


def test_malformed_yaml_is_a_typed_configuration_error(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("schema: [unterminated\n", encoding="utf-8")
    with pytest.raises(ReactiveConfigurationError, match="Invalid YAML configuration"):
        load_reactive_config(malformed)


def test_cli_missing_config_is_structured_expected_oserror(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = main(
        [
            str(tmp_path / "missing.yaml"),
            "--output-prefix",
            str(tmp_path / "result"),
        ]
    )
    assert status == 2
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "failed_closed"
    assert error["error_type"] == "FileNotFoundError"


def test_cli_has_no_unreachable_surface_mask_options() -> None:
    option_strings = {
        option
        for action in _parser()._actions
        for option in action.option_strings
    }
    assert "--anode-mask" not in option_strings
    assert "--cathode-mask" not in option_strings


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--reinit-pseudo-steps", "3"),
        ("--reinit-pseudo-cfl", "0.2"),
        ("--reinit-band-m", "0.001"),
    ],
)
def test_cli_rejects_reinitialization_arguments_when_interval_is_zero(
    option: str,
    value: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = main(
        [
            str(ASSUMED_CONFIG),
            "--output-prefix",
            str(tmp_path / "result"),
            option,
            value,
        ]
    )
    assert status == 2
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "failed_closed"
    assert "interval is zero" in error["message"]
