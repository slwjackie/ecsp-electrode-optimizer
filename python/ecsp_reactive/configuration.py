"""Validated configuration for the paper-reactive reference implementation.

There are deliberately no hidden numerical defaults.  Every dimensional value
is read with an exact SI unit and a provenance tag.  A configuration can be a
runnable exploratory case or a strict paper preflight, but an unresolved value
never reaches the solver.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields
import math
from numbers import Integral, Real
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import yaml
from yaml.constructor import ConstructorError
from yaml.resolver import BaseResolver

from .provenance import (
    AssumptionRegistry,
    Quantity,
    ReactiveConfigurationError,
    validate_model_mode,
)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys at every mapping depth."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    if not isinstance(node, yaml.nodes.MappingNode):
        raise ConstructorError(
            None,
            None,
            f"expected a mapping node, received {node.id}",
            node.start_mark,
        )
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ReactiveConfigurationError(
                "Duplicate YAML mapping key "
                f"{key!r} at line {key_node.start_mark.line + 1}, "
                f"column {key_node.start_mark.column + 1}"
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_unique_yaml(stream: Any) -> Any:
    """Load trusted configuration syntax without accepting last-key-wins YAML."""

    return yaml.load(stream, Loader=_UniqueKeySafeLoader)


_QUANTITY_KEYS = frozenset({"value", "unit", "provenance", "source"})
_ALGORITHM_KEYS = frozenset({"choice", "provenance", "source"})
_REQUIRED_ALGORITHMS = frozenset(
    {
        "spatial_reconstruction",
        "riemann_solver",
        "time_integrator",
        "source_coupling",
        "flow_boundary",
        "solid_boundary",
        "level_set_boundary_x",
        "level_set_boundary_y",
        "electrical_gradient_boundary",
        "material_interface",
        "level_set_gradient",
        "level_set_reinitialization",
        "mpi_decomposition",
    }
)
_SUPPORTED_ALGORITHMS: Mapping[str, frozenset[str]] = {
    "spatial_reconstruction": frozenset({"WENO5_JS"}),
    "riemann_solver": frozenset({"HLL"}),
    "time_integrator": frozenset({"SSPRK33"}),
    "source_coupling": frozenset({"SSPRK_STAGE_COUPLED"}),
    "flow_boundary": frozenset({"OUTFLOW", "PERIODIC"}),
    "solid_boundary": frozenset({"OUTFLOW", "PERIODIC"}),
    "level_set_boundary_x": frozenset({"OUTFLOW", "PERIODIC"}),
    "level_set_boundary_y": frozenset({"OUTFLOW", "PERIODIC"}),
    "electrical_gradient_boundary": frozenset({"FIRST_ORDER_ONE_SIDED"}),
    "material_interface": frozenset(
        {"PASSIVE_SINGLE_EOS", "GHOST_FLUID_EXTERNAL"}
    ),
    "level_set_gradient": frozenset({"WENO5_JS_UPWIND"}),
    "level_set_reinitialization": frozenset(
        {"NONE", "SUSSMAN_FIRST_ORDER"}
    ),
    "mpi_decomposition": frozenset(
        {"SERIAL_SINGLE_PROCESS", "ROWWISE_WENO3_HALO"}
    ),
}
_SUPPORTED_SOURCE_PROVIDERS = frozenset(
    {
        "CONFIG_UNIFORM_FIELD",
        "PRESCRIBED_ARRAY_FIELDS",
        "EXISTING_BC_GLOBAL_CALLBACK",
    }
)
_SECTION_QUANTITIES: Mapping[str, frozenset[str]] = {
    "grid": frozenset({"nx", "ny", "lx", "ly"}),
    "time": frozenset(
        {
            "end",
            "cfl",
            "maximum_steps",
            "maximum_stage_retries",
            "maximum_flow_progress_increment",
            "flow_progress_headroom",
            "maximum_solid_progress_increment",
            "solid_progress_headroom",
            "solid_diffusion_safety",
        }
    ),
    "initial": frozenset(
        {"rho", "u", "v", "temperature", "reaction_progress", "alpha"}
    ),
    "eos": frozenset(
        {
            "rho0",
            "A",
            "B",
            "N",
            "cv",
            "reference_temperature",
            "reference_energy",
        }
    ),
    "reaction": frozenset(
        {
            "preexponential",
            "activation_energy",
            "gas_constant",
            "heat_release",
            "order",
        }
    ),
    "solid_thermal": frozenset(
        {
            "density",
            "heat_capacity",
            "conductivity",
            "preexponential",
            "activation_energy",
            "heat_of_decomposition",
            "order",
        }
    ),
    "level_set": frozenset({"initial_interface_x", "reinitialization_interval"}),
    "safety": frozenset(
        {
            "minimum_density",
            "minimum_pressure",
            "minimum_temperature",
            "progress_tolerance",
        }
    ),
    "output": frozenset(
        {
            "front_history_stride_steps",
            "maximum_history_bytes",
            "maximum_working_set_bytes",
        }
    ),
}
_SECTION_KEYS: Mapping[str, frozenset[str]] = {
    **_SECTION_QUANTITIES,
    "electrical": frozenset(
        {
            "conductivity",
            "electric_field_x",
            "electric_field_y",
            "electrochemical_heat",
            "butler_volmer_enabled",
            "source_provider",
        }
    ),
    "front": frozenset({"thresholds"}),
    "numerics": frozenset(
        {
            "weno_epsilon",
            "level_set_weno_relative_epsilon",
            "level_set_weno_absolute_epsilon",
            "algorithms",
        }
    ),
}
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema",
        "model_mode",
        "strict_paper",
        "description",
        *_SECTION_KEYS,
        # These two documentation-only lists occur in the deliberately
        # incomplete strict preflight template and never enter a calculation.
        "unresolved_required_inputs",
        "paper_dimensional_conflicts",
    }
)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReactiveConfigurationError(f"{path} must be an object")
    return value


def _validate_key_set(
    value: Mapping[str, Any],
    allowed: frozenset[str],
    path: str,
    *,
    require_all: bool = False,
) -> None:
    non_string = [key for key in value if not isinstance(key, str)]
    if non_string:
        rendered = [f"{key!r} ({type(key).__name__})" for key in non_string]
        raise ReactiveConfigurationError(
            f"{path or 'root'} contains non-string configuration key(s): "
            f"{rendered}; all configuration keys must be strings"
        )
    keys = set(value)
    unknown = sorted(keys - allowed)
    if unknown:
        qualified = [f"{path}.{key}" if path else key for key in unknown]
        raise ReactiveConfigurationError(
            f"Unknown configuration key(s): {qualified}"
        )
    if require_all:
        missing = sorted(allowed - keys)
        if missing:
            qualified = [f"{path}.{key}" if path else key for key in missing]
            raise ReactiveConfigurationError(
                f"Missing required configuration key(s): {qualified}"
            )


def _validate_configuration_keys(root: Mapping[str, Any]) -> None:
    """Reject every unknown key after inheritance has been merged.

    Top-level/section completeness remains with the normal parser so the
    intentionally incomplete strict template can fail at its first UNRESOLVED
    quantity.  Whenever a quantity or algorithm object is present, however,
    its internal key set is exact and cannot carry ignored metadata/typos.
    """

    _validate_key_set(root, _TOP_LEVEL_KEYS, "root")
    for section_name, allowed_keys in _SECTION_KEYS.items():
        if section_name not in root:
            continue
        section = _mapping(root[section_name], section_name)
        _validate_key_set(section, allowed_keys, section_name)

    for section_name, quantity_names in _SECTION_QUANTITIES.items():
        if section_name not in root:
            continue
        section = _mapping(root[section_name], section_name)
        for quantity_name in quantity_names:
            if quantity_name in section:
                quantity_path = f"{section_name}.{quantity_name}"
                quantity = _mapping(section[quantity_name], quantity_path)
                _validate_key_set(
                    quantity,
                    _QUANTITY_KEYS,
                    quantity_path,
                    require_all=True,
                )

    if "electrical" in root:
        electrical = _mapping(root["electrical"], "electrical")
        for quantity_name in (
            "conductivity",
            "electric_field_x",
            "electric_field_y",
            "electrochemical_heat",
        ):
            if quantity_name in electrical:
                quantity_path = f"electrical.{quantity_name}"
                _validate_key_set(
                    _mapping(electrical[quantity_name], quantity_path),
                    _QUANTITY_KEYS,
                    quantity_path,
                    require_all=True,
                )
        if "source_provider" in electrical:
            _validate_key_set(
                _mapping(electrical["source_provider"], "electrical.source_provider"),
                _ALGORITHM_KEYS,
                "electrical.source_provider",
                require_all=True,
            )

    if "front" in root:
        front = _mapping(root["front"], "front")
        if "thresholds" in front:
            thresholds = front["thresholds"]
            if not isinstance(thresholds, list):
                raise ReactiveConfigurationError("front.thresholds must be a list")
            for index, raw_quantity in enumerate(thresholds):
                path = f"front.thresholds[{index}]"
                _validate_key_set(
                    _mapping(raw_quantity, path),
                    _QUANTITY_KEYS,
                    path,
                    require_all=True,
                )

    if "numerics" in root:
        numerics = _mapping(root["numerics"], "numerics")
        for quantity_name in (
            "weno_epsilon",
            "level_set_weno_relative_epsilon",
            "level_set_weno_absolute_epsilon",
        ):
            if quantity_name in numerics:
                quantity_path = f"numerics.{quantity_name}"
                _validate_key_set(
                    _mapping(numerics[quantity_name], quantity_path),
                    _QUANTITY_KEYS,
                    quantity_path,
                    require_all=True,
                )
        if "algorithms" in numerics:
            algorithms = _mapping(numerics["algorithms"], "numerics.algorithms")
            _validate_key_set(
                algorithms,
                _REQUIRED_ALGORITHMS,
                "numerics.algorithms",
                require_all=True,
            )
            for algorithm_name in _REQUIRED_ALGORITHMS:
                path = f"numerics.algorithms.{algorithm_name}"
                _validate_key_set(
                    _mapping(algorithms[algorithm_name], path),
                    _ALGORITHM_KEYS,
                    path,
                    require_all=True,
                )


def _child(root: Mapping[str, Any], path: str) -> Any:
    current: Any = root
    for part in path.split("."):
        current = _mapping(current, path).get(part)
        if current is None:
            raise ReactiveConfigurationError(f"Missing required configuration entry {path}")
    return current


def _boolean(root: Mapping[str, Any], path: str) -> bool:
    value = _child(root, path)
    if not isinstance(value, bool):
        raise ReactiveConfigurationError(f"{path} must be true or false")
    return value


def _deep_copy_freeze(value: Any) -> Any:
    """Recursively detach and freeze configuration-owned container state."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                copy.deepcopy(key): _deep_copy_freeze(child)
                for key, child in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_copy_freeze(child) for child in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_copy_freeze(child) for child in value)
    return copy.deepcopy(value)


def _runtime_real(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ReactiveConfigurationError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ReactiveConfigurationError(f"{name} must be finite")
    return converted


def _runtime_integer(name: str, value: Any, *, positive: bool = True) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ReactiveConfigurationError(f"{name} must be an exact integer")
    if positive and value <= 0:
        raise ReactiveConfigurationError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class ReactiveCaseConfig:
    source_path: str
    model_mode: str
    strict_paper: bool
    nx: int
    ny: int
    lx_m: float
    ly_m: float
    end_time_s: float
    cfl: float
    maximum_steps: int
    maximum_stage_retries: int
    maximum_flow_progress_increment: float
    flow_progress_headroom: float
    maximum_solid_progress_increment: float
    solid_progress_headroom: float
    solid_diffusion_safety: float
    weno_epsilon: float
    level_set_weno_relative_epsilon: float
    level_set_weno_absolute_epsilon_m2: float
    initial_rho_kg_per_m3: float
    initial_u_m_per_s: float
    initial_v_m_per_s: float
    initial_temperature_K: float
    initial_reaction_progress: float
    initial_alpha: float
    tait_rho0_kg_per_m3: float
    tait_A_Pa: float
    tait_B_Pa: float
    tait_N: float
    caloric_cv_J_per_kgK: float
    caloric_reference_temperature_K: float
    caloric_reference_energy_J_per_kg: float
    reaction_preexponential_per_s: float
    reaction_activation_energy_J_per_mol: float
    gas_constant_J_per_molK: float
    reaction_heat_J_per_kg: float
    reaction_order: float
    solid_density_kg_per_m3: float
    solid_heat_capacity_J_per_kgK: float
    solid_conductivity_W_per_mK: float
    decomposition_preexponential_per_s: float
    decomposition_activation_energy_J_per_mol: float
    decomposition_heat_J_per_kg: float
    decomposition_order: float
    conductivity_S_per_m: float | None
    electric_field_x_V_per_m: float | None
    electric_field_y_V_per_m: float | None
    electrochemical_heat_W_per_m3: float | None
    interface_x_m: float
    reinitialization_interval_steps: int
    front_thresholds: tuple[float, ...]
    front_history_stride_steps: int
    maximum_history_bytes: int
    maximum_working_set_bytes: int
    density_reject_below_kg_per_m3: float
    pressure_reject_below_Pa: float
    temperature_reject_below_K: float
    progress_tolerance: float
    butler_volmer_enabled: bool
    electrical_source_provider: str
    algorithms: Mapping[str, str]
    provenance_report: Mapping[str, Any]
    raw: Mapping[str, Any]

    def __post_init__(self) -> None:
        self.validate_runtime_contract()
        object.__setattr__(
            self,
            "front_thresholds",
            tuple(float(value) for value in self.front_thresholds),
        )
        object.__setattr__(self, "algorithms", _deep_copy_freeze(self.algorithms))
        object.__setattr__(
            self,
            "provenance_report",
            _deep_copy_freeze(self.provenance_report),
        )
        object.__setattr__(self, "raw", _deep_copy_freeze(self.raw))

    def validate_runtime_contract(self) -> None:
        """Validate invariants even for direct construction/dataclasses.replace."""

        if not isinstance(self.source_path, str) or not self.source_path:
            raise ReactiveConfigurationError("source_path must be a nonempty string")
        if self.model_mode not in {"paper_faithful", "ecsp_extended"}:
            raise ReactiveConfigurationError(
                "model_mode must be 'paper_faithful' or 'ecsp_extended'"
            )
        if not isinstance(self.strict_paper, bool):
            raise ReactiveConfigurationError("strict_paper must be true or false")
        if not isinstance(self.butler_volmer_enabled, bool):
            raise ReactiveConfigurationError(
                "butler_volmer_enabled must be true or false"
            )

        for name in (
            "nx",
            "ny",
            "maximum_steps",
            "maximum_stage_retries",
            "front_history_stride_steps",
            "maximum_history_bytes",
            "maximum_working_set_bytes",
        ):
            _runtime_integer(name, getattr(self, name))
        _runtime_integer(
            "reinitialization_interval_steps",
            self.reinitialization_interval_steps,
            positive=False,
        )
        if self.nx < 8 or self.ny < 8:
            raise ReactiveConfigurationError(
                "WENO5 reference runs require at least 8 cells per axis"
            )
        if self.reinitialization_interval_steps < 0:
            raise ReactiveConfigurationError(
                "Level-set reinitialization interval cannot be negative"
            )

        positive_reals = (
            "lx_m",
            "ly_m",
            "end_time_s",
            "cfl",
            "maximum_flow_progress_increment",
            "flow_progress_headroom",
            "maximum_solid_progress_increment",
            "solid_progress_headroom",
            "solid_diffusion_safety",
            "weno_epsilon",
            "level_set_weno_absolute_epsilon_m2",
            "initial_rho_kg_per_m3",
            "initial_temperature_K",
            "tait_rho0_kg_per_m3",
            "tait_B_Pa",
            "tait_N",
            "caloric_cv_J_per_kgK",
            "caloric_reference_temperature_K",
            "gas_constant_J_per_molK",
            "solid_density_kg_per_m3",
            "solid_heat_capacity_J_per_kgK",
            "density_reject_below_kg_per_m3",
            "temperature_reject_below_K",
        )
        for name in positive_reals:
            if _runtime_real(name, getattr(self, name)) <= 0.0:
                raise ReactiveConfigurationError(f"{name} must be positive")

        nonnegative_reals = (
            "level_set_weno_relative_epsilon",
            "initial_reaction_progress",
            "initial_alpha",
            "reaction_preexponential_per_s",
            "reaction_activation_energy_J_per_mol",
            "reaction_order",
            "solid_conductivity_W_per_mK",
            "decomposition_preexponential_per_s",
            "decomposition_activation_energy_J_per_mol",
            "decomposition_order",
            "density_reject_below_kg_per_m3",
            "pressure_reject_below_Pa",
            "temperature_reject_below_K",
            "progress_tolerance",
        )
        for name in nonnegative_reals:
            if _runtime_real(name, getattr(self, name)) < 0.0:
                raise ReactiveConfigurationError(f"{name} must be non-negative")

        unrestricted_reals = (
            "initial_u_m_per_s",
            "initial_v_m_per_s",
            "tait_A_Pa",
            "caloric_reference_energy_J_per_kg",
            "reaction_heat_J_per_kg",
            "decomposition_heat_J_per_kg",
            "interface_x_m",
        )
        for name in unrestricted_reals:
            _runtime_real(name, getattr(self, name))

        if self.cfl > 1.0:
            raise ReactiveConfigurationError("cfl must not exceed 1")
        for name in (
            "maximum_flow_progress_increment",
            "flow_progress_headroom",
            "maximum_solid_progress_increment",
            "solid_progress_headroom",
            "solid_diffusion_safety",
        ):
            if getattr(self, name) > 1.0:
                raise ReactiveConfigurationError(f"{name} must not exceed 1")
        if self.initial_reaction_progress > 1.0 or self.initial_alpha > 1.0:
            raise ReactiveConfigurationError(
                "Initial reaction progress and alpha must be in [0,1]"
            )
        if self.progress_tolerance >= 1.0:
            raise ReactiveConfigurationError(
                "progress_tolerance must satisfy 0 <= tolerance < 1"
            )
        if not 0.0 < self.interface_x_m < self.lx_m:
            raise ReactiveConfigurationError(
                "Initial level-set interface must lie inside the domain"
            )

        if isinstance(self.front_thresholds, (str, bytes)) or not isinstance(
            self.front_thresholds, Sequence
        ):
            raise ReactiveConfigurationError("front_thresholds must be a sequence")
        thresholds = tuple(
            _runtime_real(f"front_thresholds[{index}]", value)
            for index, value in enumerate(self.front_thresholds)
        )
        if not thresholds:
            raise ReactiveConfigurationError("front_thresholds must be nonempty")
        if any(not 0.0 < value < 1.0 for value in thresholds) or len(
            set(thresholds)
        ) != len(thresholds):
            raise ReactiveConfigurationError(
                "front thresholds must be unique values strictly inside (0,1)"
            )

        if not isinstance(self.algorithms, Mapping):
            raise ReactiveConfigurationError("algorithms must be a mapping")
        _validate_key_set(
            self.algorithms,
            _REQUIRED_ALGORITHMS,
            "algorithms",
            require_all=True,
        )
        for name, choices in _SUPPORTED_ALGORITHMS.items():
            choice = self.algorithms[name]
            if not isinstance(choice, str) or choice not in choices:
                raise ReactiveConfigurationError(
                    f"Unsupported {name}={choice!r}; supported choices are "
                    f"{sorted(choices)}"
                )
        if self.algorithms["level_set_boundary_x"] == "PERIODIC":
            raise ReactiveConfigurationError(
                "level_set_boundary_x=PERIODIC is unavailable with the built-in "
                "single-plane level-set initializer; a periodic paired-interface "
                "initializer has not been implemented"
            )

        expected_reinitialization = (
            "NONE"
            if self.reinitialization_interval_steps == 0
            else "SUSSMAN_FIRST_ORDER"
        )
        if (
            self.algorithms["level_set_reinitialization"]
            != expected_reinitialization
        ):
            raise ReactiveConfigurationError(
                "Inconsistent level-set reinitialization configuration: "
                f"interval={self.reinitialization_interval_steps} requires "
                "level_set_reinitialization="
                f"{expected_reinitialization}"
            )

        if self.electrical_source_provider not in _SUPPORTED_SOURCE_PROVIDERS:
            raise ReactiveConfigurationError("Unsupported electrical source provider")
        if self.model_mode == "paper_faithful":
            if self.butler_volmer_enabled:
                raise ReactiveConfigurationError(
                    "paper_faithful mode forbids Butler-Volmer"
                )
            if self.electrical_source_provider not in {
                "CONFIG_UNIFORM_FIELD",
                "PRESCRIBED_ARRAY_FIELDS",
            }:
                raise ReactiveConfigurationError(
                    "paper_faithful mode requires a prescribed electrical provider"
                )
        else:
            if not self.butler_volmer_enabled:
                raise ReactiveConfigurationError(
                    "ecsp_extended mode must enable Butler-Volmer"
                )
            if self.electrical_source_provider != "EXISTING_BC_GLOBAL_CALLBACK":
                raise ReactiveConfigurationError(
                    "ecsp_extended mode requires EXISTING_BC_GLOBAL_CALLBACK"
                )

        electrical_values = {
            "conductivity_S_per_m": self.conductivity_S_per_m,
            "electric_field_x_V_per_m": self.electric_field_x_V_per_m,
            "electric_field_y_V_per_m": self.electric_field_y_V_per_m,
            "electrochemical_heat_W_per_m3": self.electrochemical_heat_W_per_m3,
        }
        if self.electrical_source_provider == "CONFIG_UNIFORM_FIELD":
            for name, value in electrical_values.items():
                if value is None:
                    raise ReactiveConfigurationError(
                        f"CONFIG_UNIFORM_FIELD requires {name}"
                    )
                numeric = _runtime_real(name, value)
                if name == "conductivity_S_per_m" and numeric < 0.0:
                    raise ReactiveConfigurationError(
                        "conductivity_S_per_m must be non-negative"
                    )
        elif any(value is not None for value in electrical_values.values()):
            raise ReactiveConfigurationError(
                f"{self.electrical_source_provider} requires uniform electrical "
                "scalar fields to be None because they are not consumed"
            )

        if not isinstance(self.provenance_report, Mapping):
            raise ReactiveConfigurationError("provenance_report must be a mapping")
        if not isinstance(self.raw, Mapping):
            raise ReactiveConfigurationError("raw must be a mapping")

    def runtime_configuration_snapshot(self) -> dict[str, Any]:
        """Return the effective computational inputs, excluding source records."""

        excluded = {"source_path", "provenance_report", "raw"}
        snapshot: dict[str, Any] = {}
        for item in fields(self):
            if item.name in excluded:
                continue
            value = getattr(self, item.name)
            if isinstance(value, Mapping):
                snapshot[item.name] = {
                    str(key): value[key] for key in sorted(value)
                }
            elif isinstance(value, tuple):
                snapshot[item.name] = list(value)
            else:
                snapshot[item.name] = value
        return snapshot

    def runtime_provenance_audit(self) -> dict[str, Any]:
        """Compare effective values with the immutable parser-time snapshot.

        ``dataclasses.replace`` remains useful for non-strict manufactured
        tests, but it is not a provenance-bearing configuration API.  Such
        changes are therefore exposed as assumed runtime overrides.  Strict
        paper runs reject any override or any assumed source record.
        """

        report = self.provenance_report
        parsed = report.get("parsed_runtime_configuration")
        if not isinstance(parsed, Mapping):
            raise ReactiveConfigurationError(
                "Configuration provenance is missing its parser-time runtime snapshot"
            )
        current = self.runtime_configuration_snapshot()
        override_names = sorted(
            key
            for key in set(parsed) | set(current)
            if parsed.get(key) != current.get(key)
        )
        overrides = {
            name: {
                "parsed_value": parsed.get(name),
                "effective_value": current.get(name),
                "provenance": "ASSUMED_NOT_FROM_PAPER",
                "source": (
                    "non-provenance-bearing runtime object override; use a complete "
                    "YAML value/unit/provenance/source or choice/provenance/source "
                    "record for traceable physical work"
                ),
            }
            for name in override_names
        }
        report_strict = report.get("strict_paper")
        if not isinstance(report_strict, bool):
            raise ReactiveConfigurationError(
                "Configuration provenance strict_paper flag is missing or invalid"
            )
        assumed_quantity_count = report.get("assumed_quantity_count")
        assumed_algorithm_count = report.get("assumed_algorithm_count")
        if not isinstance(assumed_quantity_count, int) or not isinstance(
            assumed_algorithm_count, int
        ):
            raise ReactiveConfigurationError(
                "Configuration provenance assumption counts are missing or invalid"
            )
        if self.strict_paper and (
            not report_strict
            or override_names
            or assumed_quantity_count > 0
            or assumed_algorithm_count > 0
        ):
            raise ReactiveConfigurationError(
                "Strict paper mode rejects stale, assumed, or non-provenance-bearing "
                "runtime configuration overrides"
            )
        return {
            "status": (
                "MATCHES_PARSED_PROVENANCE"
                if not override_names and report_strict == self.strict_paper
                else "NONSTRICT_RUNTIME_OVERRIDES_EXPLICITLY_CLASSIFIED_AS_ASSUMED"
            ),
            "parser_report_strict_paper": report_strict,
            "effective_strict_paper": self.strict_paper,
            "override_count": len(overrides),
            "overrides": overrides,
            "effective_runtime_configuration": current,
            "ranking_eligible": False,
        }

    @property
    def dx_m(self) -> float:
        return self.lx_m / self.nx

    @property
    def dy_m(self) -> float:
        return self.ly_m / self.ny


def parse_reactive_config(raw: Mapping[str, Any], source_path: str = "<mapping>") -> ReactiveCaseConfig:
    root = _mapping(raw, "root")
    if root.get("schema") != "ecsp.paper-reactive/v1":
        raise ReactiveConfigurationError("schema must be 'ecsp.paper-reactive/v1'")
    _validate_configuration_keys(root)
    mode = validate_model_mode(str(root.get("model_mode", "")))
    strict_paper = _boolean(root, "strict_paper")
    registry = AssumptionRegistry(strict_paper=strict_paper)

    def q(path: str, unit: str, *, positive: bool = False, nonnegative: bool = False) -> float:
        return registry.quantity(
            path,
            _mapping(_child(root, path), path),
            unit,
            positive=positive,
            nonnegative=nonnegative,
        )

    def integer(path: str, unit: str, *, positive: bool = True) -> int:
        """Parse a resource/count quantity without a lossy float round trip."""

        item = _mapping(_child(root, path), path)
        raw_value = item.get("value")
        if isinstance(raw_value, bool):
            raise ReactiveConfigurationError(
                f"{path} value must be numeric, not a boolean"
            )
        if not isinstance(raw_value, Real):
            raise ReactiveConfigurationError(f"{path} must be an exact integer")
        if isinstance(raw_value, Integral):
            converted = int(raw_value)
        else:
            try:
                floating = float(raw_value)
            except (OverflowError, TypeError, ValueError) as exc:
                raise ReactiveConfigurationError(
                    f"{path} must be a finite exact integer"
                ) from exc
            if not math.isfinite(floating) or not floating.is_integer():
                raise ReactiveConfigurationError(f"{path} must be an exact integer")
            # A YAML floating-point token beyond this range cannot uniquely
            # identify adjacent integer values. Require an integer token there.
            if abs(floating) > 2**53:
                raise ReactiveConfigurationError(
                    f"{path} floating-point value exceeds the exact integer range; "
                    "write it as an integer token"
                )
            converted = int(floating)
        if converted < -sys.maxsize - 1 or converted > sys.maxsize:
            raise ReactiveConfigurationError(
                f"{path} exceeds the platform integer range"
            )
        if positive and converted <= 0:
            raise ReactiveConfigurationError(f"{path} must be positive")

        # Validate the complete unit/provenance/source contract and register
        # it, but retain the exact integer in both runtime state and report.
        q(path, unit, positive=positive)
        parsed = registry.quantities[path]
        registry.quantities[path] = Quantity(
            value=converted,
            unit=parsed.unit,
            provenance=parsed.provenance,
            source=parsed.source,
        )
        return converted

    nx = integer("grid.nx", "cells")
    ny = integer("grid.ny", "cells")
    if min(nx, ny) < 8:
        raise ReactiveConfigurationError("WENO5 reference runs require at least 8 cells per axis")
    lx_m = q("grid.lx", "m", positive=True)
    ly_m = q("grid.ly", "m", positive=True)
    end_time_s = q("time.end", "s", positive=True)
    cfl = q("time.cfl", "1", positive=True)
    if cfl > 1.0:
        raise ReactiveConfigurationError("time.cfl must not exceed 1")
    maximum_steps = integer("time.maximum_steps", "steps")
    maximum_stage_retries = integer("time.maximum_stage_retries", "attempts")
    front_history_stride_steps = integer(
        "output.front_history_stride_steps", "steps"
    )
    maximum_history_bytes = integer("output.maximum_history_bytes", "bytes")
    maximum_working_set_bytes = integer(
        "output.maximum_working_set_bytes", "bytes"
    )
    maximum_flow_progress_increment = q(
        "time.maximum_flow_progress_increment", "1", positive=True
    )
    flow_progress_headroom = q("time.flow_progress_headroom", "1", positive=True)
    maximum_solid_progress_increment = q(
        "time.maximum_solid_progress_increment", "1", positive=True
    )
    solid_progress_headroom = q("time.solid_progress_headroom", "1", positive=True)
    solid_diffusion_safety = q("time.solid_diffusion_safety", "1", positive=True)
    weno_epsilon = q("numerics.weno_epsilon", "1", positive=True)
    level_set_weno_relative_epsilon = q(
        "numerics.level_set_weno_relative_epsilon", "1", nonnegative=True
    )
    level_set_weno_absolute_epsilon_m2 = q(
        "numerics.level_set_weno_absolute_epsilon", "m^2", positive=True
    )
    if any(
        value > 1.0
        for value in (
            maximum_flow_progress_increment,
            flow_progress_headroom,
            maximum_solid_progress_increment,
            solid_progress_headroom,
            solid_diffusion_safety,
        )
    ):
        raise ReactiveConfigurationError("Progress and diffusion safety factors must not exceed 1")

    algorithms_raw = _mapping(_child(root, "numerics.algorithms"), "numerics.algorithms")
    required_algorithms = (
        "spatial_reconstruction",
        "riemann_solver",
        "time_integrator",
        "source_coupling",
        "flow_boundary",
        "solid_boundary",
        "level_set_boundary_x",
        "level_set_boundary_y",
        "electrical_gradient_boundary",
        "material_interface",
        "level_set_gradient",
        "level_set_reinitialization",
        "mpi_decomposition",
    )
    algorithms = {
        name: registry.algorithm(name, _mapping(algorithms_raw.get(name), name))
        for name in required_algorithms
    }
    for name, choice in algorithms.items():
        if choice not in _SUPPORTED_ALGORITHMS[name]:
            raise ReactiveConfigurationError(
                f"Unsupported {name}={choice!r}; supported choices are "
                f"{sorted(_SUPPORTED_ALGORITHMS[name])}"
            )

    thresholds_raw = _child(root, "front.thresholds")
    if not isinstance(thresholds_raw, list) or not thresholds_raw:
        raise ReactiveConfigurationError("front.thresholds must be a nonempty list")
    thresholds = tuple(
        registry.quantity(
            f"front.thresholds[{index}]",
            _mapping(value, f"front.thresholds[{index}]"),
            "1",
        )
        for index, value in enumerate(thresholds_raw)
    )
    if any(not 0.0 < value < 1.0 for value in thresholds) or len(set(thresholds)) != len(thresholds):
        raise ReactiveConfigurationError("front thresholds must be unique values strictly inside (0,1)")

    bv_enabled = _boolean(root, "electrical.butler_volmer_enabled")
    source_provider = registry.algorithm(
        "electrical_source_provider",
        _mapping(_child(root, "electrical.source_provider"), "electrical.source_provider"),
    )
    if source_provider not in _SUPPORTED_SOURCE_PROVIDERS:
        raise ReactiveConfigurationError("Unsupported electrical source provider")
    if mode == "paper_faithful" and bv_enabled:
        raise ReactiveConfigurationError(
            "paper_faithful mode forbids Butler-Volmer because it is not specified by the paper"
        )
    if mode == "ecsp_extended" and not bv_enabled:
        raise ReactiveConfigurationError(
            "ecsp_extended mode must explicitly enable its non-paper Butler-Volmer extension"
        )
    if mode == "paper_faithful" and source_provider not in {
        "CONFIG_UNIFORM_FIELD",
        "PRESCRIBED_ARRAY_FIELDS",
    }:
        raise ReactiveConfigurationError(
            "paper_faithful mode requires CONFIG_UNIFORM_FIELD or "
            "PRESCRIBED_ARRAY_FIELDS, not the BV callback"
        )
    if mode == "ecsp_extended" and source_provider != "EXISTING_BC_GLOBAL_CALLBACK":
        raise ReactiveConfigurationError(
            "ecsp_extended mode requires the existing BC-global callback boundary"
        )

    electrical_scalar_specs = {
        "conductivity": ("S/m", True),
        "electric_field_x": ("V/m", False),
        "electric_field_y": ("V/m", False),
        "electrochemical_heat": ("W/m^3", False),
    }
    not_consumed_inputs: dict[str, dict[str, Any]] = {}
    if source_provider == "CONFIG_UNIFORM_FIELD":
        conductivity_S_per_m = q(
            "electrical.conductivity", "S/m", nonnegative=True
        )
        electric_field_x_V_per_m = q("electrical.electric_field_x", "V/m")
        electric_field_y_V_per_m = q("electrical.electric_field_y", "V/m")
        electrochemical_heat_W_per_m3 = q(
            "electrical.electrochemical_heat", "W/m^3"
        )
    else:
        conductivity_S_per_m = None
        electric_field_x_V_per_m = None
        electric_field_y_V_per_m = None
        electrochemical_heat_W_per_m3 = None
        electrical_section = _mapping(_child(root, "electrical"), "electrical")
        for quantity_name, (expected_unit, require_nonnegative) in (
            electrical_scalar_specs.items()
        ):
            if quantity_name not in electrical_section:
                continue
            path = f"electrical.{quantity_name}"
            item = _mapping(
                electrical_section[quantity_name],
                path,
            )
            # These values are deliberately not registered as consumed model
            # inputs, but malformed/stale records must still fail closed.
            if not isinstance(item.get("source"), str):
                raise ReactiveConfigurationError(
                    f"{path} requires a human-readable string source"
                )
            validated = Quantity.parse(path, item, expected_unit)
            if require_nonnegative and validated.value < 0.0:
                raise ReactiveConfigurationError(f"{path} must be non-negative")
            not_consumed_inputs[path] = {
                "status": "IGNORED_NOT_CONSUMED",
                "provider": source_provider,
                "configured_value": validated.value,
                "configured_unit": validated.unit,
                "expected_uniform_field_unit": expected_unit,
                "configured_provenance": validated.provenance.value,
                "configured_source": validated.source,
                "reason": (
                    f"{source_provider} obtains electrical fields outside the "
                    "uniform scalar configuration"
                ),
            }

    reinitialization_interval_steps = integer(
        "level_set.reinitialization_interval", "steps", positive=False
    )
    if reinitialization_interval_steps < 0:
        raise ReactiveConfigurationError(
            "Level-set reinitialization interval cannot be negative"
        )
    reinitialization_algorithm = algorithms["level_set_reinitialization"]
    expected_reinitialization_algorithm = (
        "NONE"
        if reinitialization_interval_steps == 0
        else "SUSSMAN_FIRST_ORDER"
    )
    if reinitialization_algorithm != expected_reinitialization_algorithm:
        raise ReactiveConfigurationError(
            "Inconsistent level-set reinitialization configuration: "
            f"interval={reinitialization_interval_steps} requires "
            f"numerics.algorithms.level_set_reinitialization="
            f"{expected_reinitialization_algorithm}, received "
            f"{reinitialization_algorithm}"
        )

    config = ReactiveCaseConfig(
        source_path=source_path,
        model_mode=mode,
        strict_paper=strict_paper,
        nx=nx,
        ny=ny,
        lx_m=lx_m,
        ly_m=ly_m,
        end_time_s=end_time_s,
        cfl=cfl,
        maximum_steps=maximum_steps,
        maximum_stage_retries=maximum_stage_retries,
        maximum_flow_progress_increment=maximum_flow_progress_increment,
        flow_progress_headroom=flow_progress_headroom,
        maximum_solid_progress_increment=maximum_solid_progress_increment,
        solid_progress_headroom=solid_progress_headroom,
        solid_diffusion_safety=solid_diffusion_safety,
        weno_epsilon=weno_epsilon,
        level_set_weno_relative_epsilon=level_set_weno_relative_epsilon,
        level_set_weno_absolute_epsilon_m2=level_set_weno_absolute_epsilon_m2,
        initial_rho_kg_per_m3=q("initial.rho", "kg/m^3", positive=True),
        initial_u_m_per_s=q("initial.u", "m/s"),
        initial_v_m_per_s=q("initial.v", "m/s"),
        initial_temperature_K=q("initial.temperature", "K", positive=True),
        initial_reaction_progress=q("initial.reaction_progress", "1", nonnegative=True),
        initial_alpha=q("initial.alpha", "1", nonnegative=True),
        tait_rho0_kg_per_m3=q("eos.rho0", "kg/m^3", positive=True),
        tait_A_Pa=q("eos.A", "Pa"),
        tait_B_Pa=q("eos.B", "Pa", positive=True),
        tait_N=q("eos.N", "1", positive=True),
        caloric_cv_J_per_kgK=q("eos.cv", "J/(kg*K)", positive=True),
        caloric_reference_temperature_K=q("eos.reference_temperature", "K", positive=True),
        caloric_reference_energy_J_per_kg=q("eos.reference_energy", "J/kg"),
        reaction_preexponential_per_s=q("reaction.preexponential", "1/s", nonnegative=True),
        reaction_activation_energy_J_per_mol=q("reaction.activation_energy", "J/mol", nonnegative=True),
        gas_constant_J_per_molK=q("reaction.gas_constant", "J/(mol*K)", positive=True),
        reaction_heat_J_per_kg=q("reaction.heat_release", "J/kg"),
        reaction_order=q("reaction.order", "1", nonnegative=True),
        solid_density_kg_per_m3=q("solid_thermal.density", "kg/m^3", positive=True),
        solid_heat_capacity_J_per_kgK=q("solid_thermal.heat_capacity", "J/(kg*K)", positive=True),
        solid_conductivity_W_per_mK=q("solid_thermal.conductivity", "W/(m*K)", nonnegative=True),
        decomposition_preexponential_per_s=q("solid_thermal.preexponential", "1/s", nonnegative=True),
        decomposition_activation_energy_J_per_mol=q(
            "solid_thermal.activation_energy", "J/mol", nonnegative=True
        ),
        decomposition_heat_J_per_kg=q("solid_thermal.heat_of_decomposition", "J/kg"),
        decomposition_order=q("solid_thermal.order", "1", nonnegative=True),
        conductivity_S_per_m=conductivity_S_per_m,
        electric_field_x_V_per_m=electric_field_x_V_per_m,
        electric_field_y_V_per_m=electric_field_y_V_per_m,
        electrochemical_heat_W_per_m3=electrochemical_heat_W_per_m3,
        interface_x_m=q("level_set.initial_interface_x", "m", nonnegative=True),
        reinitialization_interval_steps=reinitialization_interval_steps,
        front_thresholds=thresholds,
        front_history_stride_steps=front_history_stride_steps,
        maximum_history_bytes=maximum_history_bytes,
        maximum_working_set_bytes=maximum_working_set_bytes,
        density_reject_below_kg_per_m3=q("safety.minimum_density", "kg/m^3", positive=True),
        pressure_reject_below_Pa=q("safety.minimum_pressure", "Pa", nonnegative=True),
        temperature_reject_below_K=q("safety.minimum_temperature", "K", positive=True),
        progress_tolerance=q("safety.progress_tolerance", "1", nonnegative=True),
        butler_volmer_enabled=bv_enabled,
        electrical_source_provider=source_provider,
        algorithms=algorithms,
        provenance_report={},
        raw=dict(root),
    )
    if config.interface_x_m <= 0.0 or config.interface_x_m >= config.lx_m:
        raise ReactiveConfigurationError("Initial level-set interface must lie inside the domain")
    if config.initial_reaction_progress > 1.0 or config.initial_alpha > 1.0:
        raise ReactiveConfigurationError("Initial reaction progress and alpha must be in [0,1]")
    provenance_report = registry.report()
    provenance_report["not_consumed_inputs"] = dict(sorted(not_consumed_inputs.items()))
    provenance_report["parsed_runtime_configuration"] = (
        config.runtime_configuration_snapshot()
    )
    object.__setattr__(
        config,
        "provenance_report",
        _deep_copy_freeze(provenance_report),
    )
    return config


def load_reactive_config(path: str | Path) -> ReactiveCaseConfig:
    source = Path(path).expanduser().resolve(strict=True)
    try:
        with source.open("r", encoding="utf-8") as stream:
            raw = _load_unique_yaml(stream)
    except ReactiveConfigurationError:
        raise
    except (yaml.YAMLError, UnicodeError) as exc:
        raise ReactiveConfigurationError(
            f"Invalid YAML configuration in {source}: {exc}"
        ) from exc
    raw = dict(_mapping(raw, "root"))
    extends = raw.pop("extends", None)
    if extends is not None:
        base_path = (source.parent / str(extends)).resolve(strict=True)
        try:
            with base_path.open("r", encoding="utf-8") as stream:
                base_raw = _load_unique_yaml(stream)
        except ReactiveConfigurationError:
            raise
        except (yaml.YAMLError, UnicodeError) as exc:
            raise ReactiveConfigurationError(
                f"Invalid extended-base YAML configuration in {base_path}: {exc}"
            ) from exc
        base = dict(_mapping(base_raw, "extended base"))
        if "extends" in base:
            raise ReactiveConfigurationError("Nested/cyclic reactive configuration extension is forbidden")

        def merge(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
            result = dict(left)
            for key, value in right.items():
                if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
                    left_value = _mapping(result[key], key)
                    record_keys = _QUANTITY_KEYS | _ALGORITHM_KEYS
                    if set(left_value) & record_keys or set(value) & record_keys:
                        # Quantity and algorithm provenance records are atomic.
                        # Replacing only ``value``/``choice`` must not inherit an
                        # unrelated unit, source, or provenance from the parent.
                        result[key] = dict(value)
                    else:
                        result[key] = merge(left_value, value)
                else:
                    result[key] = value
            return result

        raw = merge(base, raw)
    return parse_reactive_config(raw, source_path=str(source))
