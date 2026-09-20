"""Strict provenance and failure types for the paper-reactive solver.

The 2024 ECSP paper gives four governing equations but omits enough material,
kinetic and numerical data that a unique reproduction is impossible.  This
module makes that incompleteness executable: every dimensional input carries a
provenance tag and strict paper mode refuses assumed values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping


class Provenance(str, Enum):
    PAPER = "PAPER"
    CITED_REFERENCE = "CITED_REFERENCE"
    MEASURED = "MEASURED"
    CALIBRATED = "CALIBRATED"
    ASSUMED_NOT_FROM_PAPER = "ASSUMED_NOT_FROM_PAPER"
    UNRESOLVED = "UNRESOLVED"


class ReactiveConfigurationError(ValueError):
    """The requested model is incomplete, inconsistent, or mislabeled."""


class ReactiveNumericalError(RuntimeError):
    """A numerical step failed without silently clipping the physical state."""

    def __init__(self, category: str, message: str, diagnostics: Mapping[str, Any] | None = None):
        if not category:
            raise ValueError("ReactiveNumericalError requires a category")
        super().__init__(message)
        self.category = str(category)
        self.diagnostics = dict(diagnostics or {})


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str
    provenance: Provenance
    source: str

    @classmethod
    def parse(cls, name: str, raw: Mapping[str, Any], expected_unit: str) -> "Quantity":
        if not isinstance(raw, Mapping):
            raise ReactiveConfigurationError(
                f"{name} must be an object with value/unit/provenance/source"
            )
        missing = [key for key in ("value", "unit", "provenance", "source") if key not in raw]
        if missing:
            raise ReactiveConfigurationError(f"{name} is missing {missing}")
        if isinstance(raw["value"], bool):
            raise ReactiveConfigurationError(
                f"{name} value must be numeric, not a boolean"
            )
        try:
            value = float(raw["value"])
            provenance = Provenance(str(raw["provenance"]))
        except (TypeError, ValueError) as exc:
            raise ReactiveConfigurationError(f"Invalid value/provenance for {name}") from exc
        unit = str(raw["unit"])
        source = str(raw["source"]).strip()
        if unit != expected_unit:
            raise ReactiveConfigurationError(
                f"{name} unit must be {expected_unit!r}, received {unit!r}; implicit unit conversion is forbidden"
            )
        if not math.isfinite(value):
            raise ReactiveConfigurationError(f"{name} must be finite")
        if not source:
            raise ReactiveConfigurationError(f"{name} requires a human-readable source")
        if provenance is Provenance.UNRESOLVED:
            raise ReactiveConfigurationError(
                f"{name} is UNRESOLVED and cannot be used in a numerical run"
            )
        return cls(value=value, unit=unit, provenance=provenance, source=source)


@dataclass
class AssumptionRegistry:
    strict_paper: bool
    quantities: dict[str, Quantity] = field(default_factory=dict)
    algorithms: dict[str, dict[str, str]] = field(default_factory=dict)

    def quantity(
        self,
        name: str,
        raw: Mapping[str, Any],
        expected_unit: str,
        *,
        positive: bool = False,
        nonnegative: bool = False,
    ) -> float:
        item = Quantity.parse(name, raw, expected_unit)
        if positive and item.value <= 0.0:
            raise ReactiveConfigurationError(f"{name} must be positive")
        if nonnegative and item.value < 0.0:
            raise ReactiveConfigurationError(f"{name} must be non-negative")
        if self.strict_paper and item.provenance is Provenance.ASSUMED_NOT_FROM_PAPER:
            raise ReactiveConfigurationError(
                f"Strict paper mode refuses assumed input {name}; provide a measured, paper, cited-reference, or calibrated value"
            )
        self.quantities[name] = item
        return item.value

    def algorithm(self, name: str, raw: Mapping[str, Any]) -> str:
        if not isinstance(raw, Mapping):
            raise ReactiveConfigurationError(
                f"Algorithm {name} must contain choice/provenance/source"
            )
        choice = str(raw.get("choice", "")).strip()
        source = str(raw.get("source", "")).strip()
        try:
            provenance = Provenance(str(raw.get("provenance", "")))
        except ValueError as exc:
            raise ReactiveConfigurationError(f"Invalid provenance for algorithm {name}") from exc
        if not choice or not source:
            raise ReactiveConfigurationError(f"Algorithm {name} requires choice and source")
        if provenance is Provenance.UNRESOLVED:
            raise ReactiveConfigurationError(
                f"Algorithm {name} is UNRESOLVED and cannot be used in a numerical run"
            )
        if self.strict_paper and provenance is Provenance.ASSUMED_NOT_FROM_PAPER:
            raise ReactiveConfigurationError(
                f"Strict paper mode refuses assumed algorithm {name}={choice}"
            )
        self.algorithms[name] = {
            "choice": choice,
            "provenance": provenance.value,
            "source": source,
        }
        return choice

    def report(self) -> dict[str, Any]:
        return {
            "strict_paper": self.strict_paper,
            "quantities": {
                name: {
                    "value": item.value,
                    "unit": item.unit,
                    "provenance": item.provenance.value,
                    "source": item.source,
                }
                for name, item in sorted(self.quantities.items())
            },
            "algorithms": dict(sorted(self.algorithms.items())),
            "assumed_quantity_count": sum(
                item.provenance is Provenance.ASSUMED_NOT_FROM_PAPER
                for item in self.quantities.values()
            ),
            "assumed_algorithm_count": sum(
                row["provenance"] == Provenance.ASSUMED_NOT_FROM_PAPER.value
                for row in self.algorithms.values()
            ),
        }


def validate_model_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode == "paper_reproduction":
        mode = "paper_faithful"
    allowed = {"paper_faithful", "ecsp_extended"}
    if mode not in allowed:
        raise ReactiveConfigurationError(
            f"reactive model mode must be one of {sorted(allowed)}, received {value!r}"
        )
    return mode
