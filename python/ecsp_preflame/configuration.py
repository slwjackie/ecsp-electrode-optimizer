"""Canonical preflame configuration names and explicit legacy input aliases.

Aliases only translate names; they never change numerical coefficients.  An
ambiguous configuration is rejected instead of choosing one physical model.
The old Python optimizer package is intentionally not an import alias.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping


LEGACY_BACKENDS = {
    "bc_global_preflame": "preflame_torch",
    "bc_global_torch": "preflame_torch",
    "paper_bc_global": "preflame_torch",
    "bc_global_preflame_propagation": "preflame_torch_propagation",
    "bc_global_native": "preflame_cpp_cuda",
    "bc_native_cpu": "preflame_cpp_cpu",
    "bc_native_cuda": "preflame_cuda",
    "bc_global_native_hybrid": "preflame_cpp_cuda_hybrid",
    "bc_native_cpu_pool": "preflame_cpp_cpu_pool",
    "bc_global_native_cpu_pool": "preflame_cpp_cpu_pool",
}


def is_preflame_model_status(value: Any) -> bool:
    """Recognize the same physical model in Torch, compiled and saved results."""
    return str(value or "").startswith((
        "electrochemical_thermal_decomposition_",
        "preflame_model_", "preflame_torch_", "bc_global_",
    ))


def _rename_aliases(config: dict[str, Any], canonical: str, aliases: tuple[str, ...]) -> None:
    present = [key for key in (canonical, *aliases) if key in config]
    if not present:
        return
    value = config[present[0]]
    for key in present[1:]:
        if config[key] != value:
            raise ValueError(f"Conflicting configuration aliases: {present[0]} and {key}")
    config[canonical] = value
    for key in aliases:
        config.pop(key, None)


def normalize_preflame_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a workflow configuration, accepting pre-refactor YAML key names."""
    result = copy.deepcopy(dict(config))
    _rename_aliases(result, "preflame_model", ("bc_global", "bcGlobal", "preflameModel"))
    _rename_aliases(result, "evaluation", ("optimization",))
    if isinstance(result.get("evaluator"), Mapping):
        result["evaluator"] = normalize_evaluator_config(result["evaluator"])
    return result


def normalize_evaluator_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the adapter options used by both direct and factory callers."""
    result = copy.deepcopy(dict(config))
    _rename_aliases(result, "cpp_cuda", ("native",))
    if "backend" in result:
        backend = str(result["backend"]).lower()
        result["backend"] = LEGACY_BACKENDS.get(backend, backend)
    if isinstance(result.get("physics_config"), Mapping):
        result["physics_config"] = normalize_preflame_config(result["physics_config"])
    return result


def normalize_resolved_preflame_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read a saved pre-refactor solver config when continuing an onset state."""
    result = copy.deepcopy(dict(config))
    _rename_aliases(result, "preflameModel", ("bcGlobal",))
    return result
