"""Command-line entry point for the CPU FP64 paper-reactive reference run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

from .configuration import load_reactive_config
from .level_set import ReinitializationConfig
from .provenance import ReactiveConfigurationError, ReactiveNumericalError
from .solver import ReactiveSolver


def _finite_error_json(value: object) -> object:
    """Best-effort diagnostic sanitizer that cannot hide the original error."""

    active: set[int] = set()

    def convert(item: object) -> object:
        if isinstance(item, np.ndarray):
            return convert(item.tolist())
        if isinstance(item, np.generic):
            return convert(item.item())
        if isinstance(item, float):
            if math.isfinite(item):
                return item
            label = "NAN" if math.isnan(item) else ("POSITIVE_INFINITY" if item > 0 else "NEGATIVE_INFINITY")
            return {"status": f"NONFINITE_{label}", "value": None}
        if isinstance(item, dict):
            identifier = id(item)
            if identifier in active:
                return {"status": "CYCLIC_DIAGNOSTIC_MAPPING"}
            active.add(identifier)
            try:
                output: dict[str, object] = {}
                for index, (key, child) in enumerate(item.items()):
                    safe_key = (
                        key
                        if isinstance(key, str)
                        else f"<non_string_key_{index}_{type(key).__name__}>"
                    )
                    output[safe_key] = convert(child)
                return {key: output[key] for key in sorted(output)}
            finally:
                active.remove(identifier)
        if isinstance(item, (tuple, list)):
            identifier = id(item)
            if identifier in active:
                return [{"status": "CYCLIC_DIAGNOSTIC_SEQUENCE"}]
            active.add(identifier)
            try:
                return [convert(child) for child in item]
            finally:
                active.remove(identifier)
        if isinstance(item, (str, int, bool)) or item is None:
            return item
        return {"status": "UNSERIALIZABLE_DIAGNOSTIC", "type": type(item).__name__}

    try:
        return convert(value)
    except Exception as exc:  # diagnostics must never replace the physics error
        return {
            "status": "DIAGNOSTIC_SANITIZATION_FAILED",
            "sanitizer_error_type": type(exc).__name__,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the deterministic single-process CPU FP64 reactive-Euler/Tait/"
            "solid/level-set reference solver. ECSP-extended electrical callbacks "
            "must be connected through the Python API and are never fabricated by this CLI."
        )
    )
    parser.add_argument("config", type=Path, help="Validated ecsp.paper-reactive/v1 YAML")
    parser.add_argument(
        "--output-prefix",
        required=True,
        type=Path,
        help="Output path without extension; .npz and .json are written atomically",
    )
    parser.add_argument(
        "--reinit-pseudo-steps",
        type=int,
        help="Required explicit pseudo-step count when the YAML reinitialization interval is nonzero",
    )
    parser.add_argument(
        "--reinit-pseudo-cfl",
        type=float,
        help="Required explicit pseudo-time CFL when reinitialization is enabled",
    )
    parser.add_argument("--reinit-band-m", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        config = load_reactive_config(arguments.config)
        if config.model_mode == "ecsp_extended":
            raise ReactiveConfigurationError(
                "ecsp_extended cannot run from the standalone CLI because it requires an "
                "explicit in-process electrical callback. Use ReactiveSolver(..., "
                "electrical_callback=...) so the corrected BV provider is auditable."
            )
        reinitialization_arguments = (
            arguments.reinit_pseudo_steps,
            arguments.reinit_pseudo_cfl,
            arguments.reinit_band_m,
        )
        if config.reinitialization_interval_steps == 0 and any(
            value is not None for value in reinitialization_arguments
        ):
            raise ReactiveConfigurationError(
                "Reinitialization CLI arguments are forbidden when the YAML "
                "reinitialization interval is zero"
            )
        reinitialization = None
        if config.reinitialization_interval_steps > 0:
            if (
                arguments.reinit_pseudo_steps is None
                or arguments.reinit_pseudo_cfl is None
            ):
                raise ReactiveConfigurationError(
                    "A nonzero YAML reinitialization interval requires "
                    "--reinit-pseudo-steps and --reinit-pseudo-cfl; no hidden "
                    "defaults are used"
                )
            reinitialization = ReinitializationConfig(
                enabled=True,
                pseudo_steps=arguments.reinit_pseudo_steps,
                pseudo_cfl=arguments.reinit_pseudo_cfl,
                narrow_band_half_width_m=arguments.reinit_band_m,
                source="explicit CLI arguments; algorithm/frequency absent from paper",
            )
        solver = ReactiveSolver(
            config,
            reinitialization=reinitialization,
        )
        result = solver.run()
        npz_path, json_path = result.save(arguments.output_prefix)
        message = {
            "status": "completed",
            "model_mode": config.model_mode,
            "final_time_s": result.final_time_s,
            "accepted_steps": result.accepted_steps,
            "npz": str(npz_path.resolve()),
            "json": str(json_path.resolve()),
            "paper_reproduction_verdict": result.metadata[
                "paper_reproduction_verdict"
            ],
        }
        print(json.dumps(message, sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 0
    except (ReactiveConfigurationError, ReactiveNumericalError, OSError) as exc:
        error = {
            "status": "failed_closed",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        if isinstance(exc, ReactiveNumericalError):
            error["category"] = exc.category
            error["diagnostics"] = _finite_error_json(exc.diagnostics)
        print(
            json.dumps(error, sort_keys=True, ensure_ascii=False, allow_nan=False),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
