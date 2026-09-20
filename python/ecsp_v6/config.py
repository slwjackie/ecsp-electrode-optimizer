from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


COMPATIBILITY_PROFILES: dict[str, dict[str, Any]] = {
    "uploaded_v63": {
        "electrical": {
            "jouleHeatModel": "legacy_magnitude_E_times_J",
        },
        "interface": {
            "currentDirectionModel": "legacy_magnitude",
            "boundaryCouplingModel": "legacy_posthoc",
        },
    },
}


def apply_compatibility_profile(config: dict[str, Any]) -> dict[str, Any]:
    """Atomically apply a named historical-physics compatibility profile.

    Historical regression must not depend on remembering several independent
    switches.  A single profile changes the coupled set of legacy options
    together, while the default/current profile leaves the configuration
    untouched.
    """
    project = config.setdefault("project", {})
    name = str(project.get("compatibilityProfile", "current") or "current").strip().lower()
    if name in {"", "current", "none", "default"}:
        project["compatibilityProfile"] = "current"
        return config
    if name not in COMPATIBILITY_PROFILES:
        raise ValueError(
            f"Unknown project.compatibilityProfile={name!r}; "
            f"available={sorted(COMPATIBILITY_PROFILES)}"
        )
    resolved = deep_merge(config, COMPATIBILITY_PROFILES[name])
    resolved.setdefault("project", {})["compatibilityProfile"] = name
    return resolved


_SCIENTIFIC_NUMBER = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?\d+$"
)


def _normalize_yaml_scalars(value: Any) -> Any:
    """Normalize PyYAML scientific-notation strings to real floats.

    PyYAML 6 follows YAML 1.1 resolution rules and may parse ``5.0e9``
    (without an explicit exponent sign) as a string.  That is dangerous in a
    resolved JSON config and in any code path that does not call ``float``
    immediately.  Convert only strings that are unambiguously complete
    scientific-notation numerals; calibration placeholders and ordinary text
    remain untouched.
    """
    if isinstance(value, dict):
        return {key: _normalize_yaml_scalars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_yaml_scalars(item) for item in value]
    if isinstance(value, str) and _SCIENTIFIC_NUMBER.fullmatch(value.strip()):
        return float(value)
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top level of {path} must be a mapping")
    return _normalize_yaml_scalars(data)


def resolve_manufacturing_component_limits(
    config_or_manufacturability: Mapping[str, Any],
) -> tuple[int, int]:
    """Resolve component caps from one authoritative fallback policy.

    The helper accepts either the complete configuration or an extracted
    ``manufacturability`` mapping.  Every generator, sanitizer, projection,
    codec and solver-grid audit must call this function rather than restating
    local defaults.

    ``maximumComponentsPerPolarity`` defaults to 16 and
    ``maximumTotalComponents`` defaults to twice that resolved per-polarity cap.
    """
    nested = config_or_manufacturability.get("manufacturability")
    manufacturing = nested if isinstance(nested, Mapping) else config_or_manufacturability
    maximum_per = int(manufacturing.get("maximumComponentsPerPolarity", 16))
    maximum_total = int(
        manufacturing.get("maximumTotalComponents", 2 * maximum_per)
    )
    return maximum_per, maximum_total


def resolve_bootstrap_component_bounds(config: dict[str, Any]) -> tuple[int, int]:
    """Return the configured bootstrap component-count interval.

    ``bootstrapMaximumComponents`` has one authoritative default: the resolved
    manufacturability ``maximumTotalComponents``.  Keeping this resolution in
    one helper prevents generators, bootstrap schedulers, and config validation
    from silently advertising different reachable design spaces when a derived
    config omits the optional key.
    """
    _, maximum_total = resolve_manufacturing_component_limits(config)
    cad = config.get("cadPrimitive", {})
    minimum = int(cad.get("bootstrapMinimumComponents", 2))
    maximum = int(cad.get("bootstrapMaximumComponents", maximum_total))
    return minimum, maximum


def load_config(base_path: Path, override_paths: list[Path] | None = None) -> dict[str, Any]:
    config = load_yaml(base_path)
    for path in override_paths or []:
        config = deep_merge(config, load_yaml(path))
    config = apply_compatibility_profile(config)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = [
        "project",
        "composition",
        "geometry",
        "manufacturability",
        "electrical",
        "transport",
        "interface",
        "coupled",
        "multiFidelity",
        "reducedCFD",
        "numericalConvergence",
        "numerics",
    ]
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"Missing config sections: {missing}")
    backend = str(config["project"].get("backend", "python_gpu"))
    supported_backends = {
        "python_gpu",
        "python_cpu",
        "python_debug",
        "cpp_fp64_cpu",
    }
    if backend not in supported_backends:
        raise ValueError(
            "project.backend must be one of: "
            + ", ".join(sorted(supported_backends))
        )
    masses = config["composition"]["masses_g"]
    total = sum(float(value) for value in masses.values())
    if abs(total - 35.0) > 1e-6:
        raise ValueError(f"Recipe masses must sum to 35.000 g; found {total:.6f} g")
    retained_water = float(config["composition"].get("retained_water_fraction", 1.0))
    if not (0.0 <= retained_water <= 1.0):
        raise ValueError("composition.retained_water_fraction must lie in [0, 1]")
    water_basis = str(
        config["composition"].get(
            "retained_water_fraction_basis", "unspecified_assumption"
        )
    )
    if bool(config["composition"].get("requireMeasuredRetainedWater", False)):
        measurement_id = config["composition"].get("retained_water_measurement_id")
        if water_basis != "measured_cure_mass_balance" or not str(
            measurement_id or ""
        ).strip():
            raise ValueError(
                "Measured retained water is required: set "
                "retained_water_fraction_basis=measured_cure_mass_balance and "
                "provide retained_water_measurement_id."
            )
    fractions = [
        float(config["activeLearning"]["paretoFraction"]),
        float(config["activeLearning"]["randomFraction"]),
        float(config["activeLearning"].get("noveltyFraction", 0.0)),
        float(config["activeLearning"]["ahcpsValidationFraction"]),
    ]
    if any((not math.isfinite(value)) or value < 0.0 or value > 1.0 for value in fractions):
        raise ValueError(
            "Each active-learning selection fraction must be finite and lie in [0, 1]"
        )
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(
            "Active-learning fractions (paretoFraction + randomFraction + "
            "noveltyFraction + ahcpsValidationFraction) must sum to 1"
        )
    active_cfg = config["activeLearning"]
    coupled_per_iteration = int(active_cfg.get("coupledPerIteration", 0))
    maximum_iterations = int(active_cfg.get("maximumIterations", 0))
    initial_iterations = int(active_cfg.get("initialIterations", 1))
    patience = int(active_cfg.get("patience", 1))
    minimum_improvement = float(active_cfg.get("minimumImprovement", 0.0))
    if coupled_per_iteration < 1:
        raise ValueError("activeLearning.coupledPerIteration must be >= 1")
    if maximum_iterations < 1:
        raise ValueError("activeLearning.maximumIterations must be >= 1")
    if not (1 <= initial_iterations <= maximum_iterations):
        raise ValueError(
            "activeLearning.initialIterations must lie in [1, maximumIterations]"
        )
    if patience < 1:
        raise ValueError("activeLearning.patience must be >= 1")
    if not math.isfinite(minimum_improvement) or minimum_improvement < 0.0:
        raise ValueError("activeLearning.minimumImprovement must be finite and >= 0")
    final_weights = active_cfg.get("finalWeights", {})
    if not isinstance(final_weights, dict) or not final_weights:
        raise ValueError("activeLearning.finalWeights must be a non-empty mapping")
    numeric_weights = [float(value) for value in final_weights.values()]
    if any((not math.isfinite(value)) or value < 0.0 for value in numeric_weights):
        raise ValueError("activeLearning.finalWeights must be finite and non-negative")
    if sum(numeric_weights) <= 0.0:
        raise ValueError("activeLearning.finalWeights must contain positive total weight")
    if int(config["activeLearning"].get("minimumDiffusionCount", 0)) < 0:
        raise ValueError("activeLearning.minimumDiffusionCount must be >= 0")
    selection_pool_mode = str(
        config["activeLearning"].get("selectionPoolMode", "generated_only")
    ).strip().lower()
    if selection_pool_mode not in {"generated_only", "mixed_sources"}:
        raise ValueError(
            "activeLearning.selectionPoolMode must be generated_only or mixed_sources"
        )
    if selection_pool_mode == "mixed_sources":
        raise ValueError(
            "activeLearning.selectionPoolMode=mixed_sources is reserved but is "
            "not wired into run_ai_design_loop; use generated_only until the "
            "loop actually constructs a mixed bootstrap/generated candidate pool"
        )
    if (
        selection_pool_mode == "generated_only"
        and int(config["activeLearning"].get("minimumDiffusionCount", 0)) != 0
    ):
        raise ValueError(
            "minimumDiffusionCount is redundant for selectionPoolMode=generated_only; "
            "set it to 0, or use mixed_sources when the selection table really contains "
            "both bootstrap and diffusion candidates."
        )
    lower_progress_clip = config["activeLearning"].get(
        "progressScoreLowerClip", None
    )
    upper_progress_clip = config["activeLearning"].get(
        "progressScoreUpperClip", 2.0
    )
    if lower_progress_clip is not None and not math.isfinite(
        float(lower_progress_clip)
    ):
        raise ValueError("activeLearning.progressScoreLowerClip must be finite or null")
    if upper_progress_clip is not None and not math.isfinite(
        float(upper_progress_clip)
    ):
        raise ValueError("activeLearning.progressScoreUpperClip must be finite or null")
    if (
        lower_progress_clip is not None
        and upper_progress_clip is not None
        and float(lower_progress_clip) >= float(upper_progress_clip)
    ):
        raise ValueError(
            "activeLearning.progressScoreLowerClip must be smaller than "
            "progressScoreUpperClip"
        )
    multifidelity = config["multiFidelity"]
    if bool(multifidelity.get("enabled", False)):
        coupled_fraction = float(multifidelity["coupledFractionOfElectrical"])
        cfd_fraction = float(multifidelity["reducedCfdFractionOfElectrical"])
        if not (0.10 <= coupled_fraction <= 0.20):
            raise ValueError(
                "multiFidelity.coupledFractionOfElectrical must lie in [0.10, 0.20]"
            )
        if not (0.03 <= cfd_fraction <= 0.05):
            raise ValueError(
                "multiFidelity.reducedCfdFractionOfElectrical must lie in [0.03, 0.05]"
            )
        if cfd_fraction > coupled_fraction:
            raise ValueError("Reduced-CFD fraction cannot exceed the coupled fraction")
        for key in (
            "coupledMinimumCount",
            "coupledMaximumCount",
            "reducedCfdMinimumCount",
            "reducedCfdMaximumCount",
        ):
            if int(multifidelity.get(key, 0)) < 0:
                raise ValueError(f"multiFidelity.{key} must be >= 0")
        if not (0.0 < float(multifidelity.get("initialTopShare", 0.85)) <= 1.0):
            raise ValueError("multiFidelity.initialTopShare must lie in (0, 1]")
        final_cfd_weights = multifidelity.get("finalCfdWeights", {})
        if final_cfd_weights and sum(float(value) for value in final_cfd_weights.values()) <= 0:
            raise ValueError("multiFidelity.finalCfdWeights must contain positive total weight")
    electrical_cfg = config.get("electrical", {})
    joule_model = str(electrical_cfg.get("jouleHeatModel", "conductive_sigma_E2")).lower()
    if joule_model not in {
        "conductive_sigma_e2", "conductive", "sigma_e2", "jcond_dot_e",
        "total_j_dot_e", "j_dot_e", "total",
        "legacy_magnitude_e_times_j", "legacy_magnitude", "e_times_jmag",
    }:
        raise ValueError("Unsupported electrical.jouleHeatModel")

    reduced = config["reducedCFD"]
    if bool(reduced.get("enabled", False)):
        if str(reduced.get("handoffMode", "first_local_ignition_or_final")) not in {
            "first_local_ignition_or_final",
            "final",
        }:
            raise ValueError(
                "reducedCFD.handoffMode must be first_local_ignition_or_final or final"
            )
        if int(reduced["gridSize"]) < 9:
            raise ValueError("reducedCFD.gridSize must be at least 9")
        if float(reduced["gasLayerHeight_m"]) <= 0:
            raise ValueError("reducedCFD.gasLayerHeight_m must be positive")
        if float(reduced["timeStep_s"]) <= 0 or float(reduced["endTime_s"]) <= 0:
            raise ValueError("reducedCFD timeStep_s and endTime_s must be positive")
        if int(reduced["maximumSteps"]) < 1:
            raise ValueError("reducedCFD.maximumSteps must be >= 1")
        pressure_cfg = reduced["pressureSolver"]
        if not (0.0 < float(pressure_cfg["omega"]) < 2.0):
            raise ValueError("reducedCFD.pressureSolver.omega must lie in (0, 2)")
        if int(pressure_cfg["maximumIterations"]) < 1:
            raise ValueError("reducedCFD.pressureSolver.maximumIterations must be >= 1")
        if float(pressure_cfg["relativeTolerance"]) <= 0:
            raise ValueError("reducedCFD.pressureSolver.relativeTolerance must be positive")
    interface_cfg = config["interface"]
    current_direction_model = str(interface_cfg.get("currentDirectionModel", "signed_normal")).lower()
    if current_direction_model not in {
        "signed_normal", "directional", "signed_j_dot_n",
        "legacy_magnitude", "magnitude_legacy", "absolute_components",
    }:
        raise ValueError("Unsupported interface.currentDirectionModel")
    nernst_cfg = interface_cfg.get("nernst", {})
    if bool(nernst_cfg.get("enabled", False)):
        if float(nernst_cfg.get("activityFloor", 0.0)) <= 0:
            raise ValueError("interface.nernst.activityFloor must be positive")
        if float(nernst_cfg.get("maximumAbsoluteShift_V", 0.0)) <= 0:
            raise ValueError("interface.nernst.maximumAbsoluteShift_V must be positive")
    blocking_cfg = interface_cfg.get("blocking", {})
    active_floor = float(blocking_cfg.get("minimumActiveAreaFraction", 0.02))
    if not (0.0 < active_floor <= 1.0):
        raise ValueError("interface.blocking.minimumActiveAreaFraction must lie in (0,1]")
    numerics_cfg = config["numerics"]
    potential_solver_cfg = numerics_cfg["potentialSolver"]
    preconditioner = str(potential_solver_cfg.get("preconditioner", "multigrid")).lower()
    allowed_preconditioners = {
        "jacobi", "diagonal", "identity", "none",
        "multigrid", "mg", "geometric_multigrid",
    }
    if preconditioner not in allowed_preconditioners:
        raise ValueError("Unsupported numerics.potentialSolver.preconditioner")

    allowed_potential_methods = {
        "auto", "pcg", "bicgstab", "bicg", "nonsymmetric_krylov",
        "rb_sor", "sor", "matlab_parity", "direct", "direct_cpu",
        "scipy_direct",
    }
    for key in ("method", "methodStatic", "methodCoupled"):
        if key in potential_solver_cfg and str(potential_solver_cfg[key]).lower() not in allowed_potential_methods:
            raise ValueError(
                f"Unsupported potential solver {key}={potential_solver_cfg[key]!r}; "
                f"allowed={sorted(allowed_potential_methods)}"
            )
    for key in ("maximumIterationsStatic", "maximumIterationsCoupled"):
        if int(potential_solver_cfg.get(key, 0)) < 1:
            raise ValueError(f"potentialSolver.{key} must be >= 1")
    for key in ("relativeToleranceStatic", "relativeToleranceCoupled", "absoluteTolerance"):
        value = float(potential_solver_cfg.get(key, 0.0))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"potentialSolver.{key} must be positive and finite")

    for stage_key in ("methodStatic", "methodCoupled"):
        requested = str(
            potential_solver_cfg.get(stage_key, potential_solver_cfg.get("method", "auto"))
        ).lower()
        if requested == "pcg":
            if preconditioner not in {"jacobi", "diagonal", "identity", "none"}:
                raise ValueError(
                    f"{stage_key}=pcg requires a symmetric Jacobi/identity preconditioner; "
                    "the legacy multigrid V-cycle must not be silently routed to BiCGStab"
                )
            if not bool(potential_solver_cfg.get("symmetricEquilibration", True)):
                raise ValueError("PCG requires potentialSolver.symmetricEquilibration=true")
            if not bool(potential_solver_cfg.get("correctionForm", True)):
                raise ValueError("PCG requires potentialSolver.correctionForm=true")

    physics_device = str(numerics_cfg.get("physicsDevice", "auto")).lower()
    physics_dtype = str(numerics_cfg.get("physicsDtype", "float64")).lower()
    if physics_device == "mps" and physics_dtype in {"float64", "double", "fp64"}:
        raise ValueError("MPS requires numerics.physicsDtype=float32")
    floors = numerics_cfg.get("physicalFloors", {})
    for name, raw in floors.items():
        value = float(raw)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"numerics.physicalFloors.{name} must be positive and finite")

    # Focused no-F NSGA-II releases retain the mature v7.7.2 physics
    # dictionaries for numerical compatibility, but remove the legacy
    # DDPM/active-learning/CFD runtime modules.  Stop validation after the
    # active electrochemical/solid model has been checked so startup does not
    # import archived modules that are deliberately absent from this package.
    model_status = str(config.get("project", {}).get("modelStatus", "")).lower()
    focused_no_f = (
        "no_flame_progress" in model_status
        or "no_f" in model_status
        or "condensed_phase_nominal_unvalidated" in model_status
    )
    if focused_no_f:
        ignition = config.get("condensedIgnition", {})
        metrics = config.get("condensedPhaseMetrics", {})
        onset = float(ignition.get("decompositionOnsetTemperature_K", 0.0))
        evaluation_time = float(metrics.get("evaluationTime_s", 0.0))
        end_time = float(config.get("coupled", {}).get("endTime_s", 0.0))
        if onset <= 0.0 or not math.isfinite(onset):
            raise ValueError(
                "condensedIgnition.decompositionOnsetTemperature_K must be positive and finite"
            )
        if evaluation_time <= 0.0 or evaluation_time > end_time + 1e-12:
            raise ValueError(
                "condensedPhaseMetrics.evaluationTime_s must lie in (0, coupled.endTime_s]"
            )
        if bool(config.get("reducedCFD", {}).get("enabled", False)):
            raise ValueError("reducedCFD must remain disabled in the focused no-F package")
        if bool(config.get("multiFidelity", {}).get("enabled", False)):
            raise ValueError("multiFidelity must remain disabled in the focused no-F package")
        return
    convergence_cfg = config.get("numericalConvergence", {}).get("reducedCFD", {})
    if bool(convergence_cfg.get("enabled", False)):
        grids = [int(value) for value in convergence_cfg.get("gridSizes", [])]
        if len(grids) < 3 or any(value < 9 for value in grids):
            raise ValueError("numericalConvergence.reducedCFD.gridSizes must contain at least three grids >= 9")
        if sorted(grids) != grids:
            raise ValueError("numericalConvergence.reducedCFD.gridSizes must be increasing")
    ddpm_cfg = config.get("ddpm", {})
    dropout_probability = float(ddpm_cfg.get("conditionDropoutProbability", 0.12))
    if not (0.0 <= dropout_probability < 1.0):
        raise ValueError("ddpm.conditionDropoutProbability must lie in [0, 1)")
    if float(ddpm_cfg.get("guidanceScale", 2.0)) <= 0:
        raise ValueError("ddpm.guidanceScale must be positive")
    dynamic_share = float(ddpm_cfg.get("dynamicParentTargetShare", 0.0))
    if not (0.0 <= dynamic_share < 1.0):
        raise ValueError("ddpm.dynamicParentTargetShare must lie in [0, 1)")
    if int(ddpm_cfg.get("dynamicParentShareRampIterations", 1)) < 1:
        raise ValueError(
            "ddpm.dynamicParentShareRampIterations must be >= 1"
        )
    if "conditionArchiveMaximumSize" in ddpm_cfg:
        raise ValueError(
            "ddpm.conditionArchiveMaximumSize is deprecated because it incorrectly "
            "counted the immutable base library. Use "
            "ddpm.conditionArchiveDynamicMaximum instead."
        )
    for key in (
        "conditionArchiveGeneratedPerIteration",
        "conditionArchiveDynamicMaximum",
        "conditionArchiveEliteMaximum",
    ):
        if int(ddpm_cfg.get(key, 0)) < 0:
            raise ValueError(f"ddpm.{key} must be >= 0")
    # Validate decode/novelty/synthesis blocks eagerly so misconfigurations
    # fail at startup, not mid-run.  Lazy imports avoid module cycles.
    from ecsp_v6.condition_synthesis import (  # noqa: PLC0415
        resolve_condition_projection,
        resolve_synthesis_weights,
    )
    from ecsp_v6.decode import resolve_decode_config  # noqa: PLC0415
    from ecsp_v6.novelty import resolve_novelty_config  # noqa: PLC0415

    resolve_decode_config(config)
    resolve_novelty_config(config)
    resolve_synthesis_weights(ddpm_cfg)
    resolve_condition_projection(ddpm_cfg)
    guidance = ddpm_cfg.get("guidanceScaleByStrategy") or {}
    allowed_guidance = {"perturb", "interpolate", "extrapolate", "prior", "unconditional"}
    unknown_guidance = set(guidance) - allowed_guidance
    if unknown_guidance:
        raise ValueError(
            f"Unknown ddpm.guidanceScaleByStrategy keys: {sorted(unknown_guidance)}"
        )
    if any(float(value) <= 0 for value in guidance.values()):
        raise ValueError("All strategy-specific guidance scales must be positive")
    for key in (
        "sequenceTokenCrossEntropyWeight",
        "sequenceSlotCrossEntropyWeight",
        "sequencePolarityCrossEntropyWeight",
        "sequenceCountCrossEntropyWeight",
        "sequencePolarityClassWeightMultiplier",
        "sequencePolarityCountLossWeight",
        "sequenceSlotClassPseudoCount",
    ):
        if float(ddpm_cfg.get(key, 1.0)) < 0.0:
            raise ValueError(f"ddpm.{key} must be non-negative")
    count_weight_power = float(
        ddpm_cfg.get("sequenceCountClassWeightPower", 0.25)
    )
    maximum_count_weight = float(
        ddpm_cfg.get("sequenceMaximumCountClassWeight", 4.0)
    )
    count_temperature = float(
        ddpm_cfg.get("sequenceCountSamplingTemperature", 1.0)
    )
    count_prior_sigma = float(
        ddpm_cfg.get("componentConditionPriorSigmaComponents", 1.5)
    )
    if count_weight_power < 0.0:
        raise ValueError("ddpm.sequenceCountClassWeightPower must be non-negative")
    if maximum_count_weight <= 0.0:
        raise ValueError("ddpm.sequenceMaximumCountClassWeight must be positive")
    if int(ddpm_cfg.get("sequenceMinimumCountClassExamples", 2)) < 1:
        raise ValueError("ddpm.sequenceMinimumCountClassExamples must be >= 1")
    if not math.isfinite(count_temperature) or count_temperature <= 0.0:
        raise ValueError("ddpm.sequenceCountSamplingTemperature must be positive and finite")
    if not math.isfinite(count_prior_sigma) or count_prior_sigma <= 0.0:
        raise ValueError("ddpm.componentConditionPriorSigmaComponents must be positive and finite")
    active_slot_threshold = float(
        ddpm_cfg.get("sequenceActiveSlotLogitThreshold", 0.75)
    )
    if not math.isfinite(active_slot_threshold):
        raise ValueError("ddpm.sequenceActiveSlotLogitThreshold must be finite")
    from ecsp_v6.eligibility import resolve_optimization_roles  # noqa: PLC0415
    from ecsp_v6.quality import resolve_numerical_quality_config  # noqa: PLC0415

    resolve_optimization_roles(config)
    resolve_numerical_quality_config(config)
    generator_mode = str(
        config.get("geometry", {}).get("generatorMode", "cad_sequence_diffusion")
    ).lower()
    allowed_generators = {
        "cad_sequence_diffusion",
        "cad_quantized_ar_latent_diffusion",
        "cad_autoregressive_latent_diffusion",
    }
    if generator_mode not in allowed_generators:
        raise ValueError(
            "Production geometry.generatorMode must be cad_sequence_diffusion or "
            "cad_quantized_ar_latent_diffusion; the 30-family pixel-DDPM path is "
            "archived under legacy_reference."
        )
    representation = str(ddpm_cfg.get("representation", "cad_sequence")).lower()
    latent_mode = generator_mode in {
        "cad_quantized_ar_latent_diffusion",
        "cad_autoregressive_latent_diffusion",
    } or representation in {
        "quantized_ar_latent",
        "cad_quantized_autoregressive_latent_diffusion_v1",
    }
    if latent_mode:
        if int(ddpm_cfg.get("parameterQuantizationLevels", 256)) < 8:
            raise ValueError("ddpm.parameterQuantizationLevels must be >= 8")
        hidden = int(ddpm_cfg.get("arHiddenDimension", 192))
        heads = int(ddpm_cfg.get("arAttentionHeads", 6))
        if hidden < 8 or heads < 1 or hidden % heads:
            raise ValueError(
                "ddpm.arHiddenDimension must be positive and divisible by arAttentionHeads"
            )
        for key in (
            "arEncoderLayers",
            "arDecoderLayers",
            "latentDimension",
            "latentDenoiserHiddenDimension",
            "latentDenoiserResidualBlocks",
            "latentDiffusionTimesteps",
        ):
            if int(ddpm_cfg.get(key, 0)) < 1:
                raise ValueError(f"ddpm.{key} must be >= 1")
        top_p = float(ddpm_cfg.get("autoregressiveTopP", 0.95))
        if not (0.0 < top_p <= 1.0):
            raise ValueError("ddpm.autoregressiveTopP must lie in (0, 1]")
        if float(ddpm_cfg.get("autoregressiveTemperature", 0.85)) <= 0.0:
            raise ValueError("ddpm.autoregressiveTemperature must be positive")
        for key in (
            "autoencoderEpochs",
            "autoencoderFineTuneEpochs",
            "latentDiffusionEpochs",
            "latentDiffusionFineTuneEpochs",
            "latentAuxiliaryEpochs",
            "latentAuxiliaryFineTuneEpochs",
        ):
            if int(ddpm_cfg.get(key, 0)) < 0:
                raise ValueError(f"ddpm.{key} must be >= 0")
        if int(ddpm_cfg.get("latentAuxiliaryHiddenDimension", 256)) < 1:
            raise ValueError("ddpm.latentAuxiliaryHiddenDimension must be >= 1")
        if int(ddpm_cfg.get("latentValidityNegativeMaximum", 0)) < 0:
            raise ValueError("ddpm.latentValidityNegativeMaximum must be >= 0")
        for key in (
            "latentValidityGuidanceScale",
            "latentPhysicsGuidanceScale",
            "latentGuidanceStepSize",
            "latentGuidanceMaximumGradientNorm",
        ):
            if float(ddpm_cfg.get(key, 0.0)) < 0.0:
                raise ValueError(f"ddpm.{key} must be nonnegative")
        guidance_start = float(
            ddpm_cfg.get("latentAuxiliaryGuidanceStartFraction", 0.75)
        )
        if not (0.0 <= guidance_start <= 1.0):
            raise ValueError(
                "ddpm.latentAuxiliaryGuidanceStartFraction must lie in [0,1]"
            )
        augmentation = ddpm_cfg.get("syntheticAugmentation") or {}
        if int(augmentation.get("count", 0)) < 0 or int(augmentation.get("fineTuneCount", 0)) < 0:
            raise ValueError("ddpm.syntheticAugmentation counts must be >= 0")
        performance = ddpm_cfg.get("performanceConditioning") or {}
        if bool(performance.get("enabled", False)):
            targets = performance.get("targets") or []
            if not targets:
                raise ValueError("Enabled ddpm.performanceConditioning requires target names")
            if int(performance.get("minimumLabeledSamples", 1)) < 1:
                raise ValueError(
                    "ddpm.performanceConditioning.minimumLabeledSamples must be >= 1"
                )
            fraction = float(performance.get("generationTargetFraction", 0.0))
            if not (0.0 <= fraction <= 1.0):
                raise ValueError(
                    "ddpm.performanceConditioning.generationTargetFraction must lie in [0,1]"
                )
    if generator_mode in allowed_generators:
        cad = config.get("cadPrimitive", {})
        if int(cad.get("bootstrapCount", 0)) < 3:
            raise ValueError("cadPrimitive.bootstrapCount must be at least 3")
        minimum_segments = int(cad.get("minimumSegmentsPerComponent", 1))
        maximum_segments = int(cad.get("maximumSegmentsPerComponent", 6))
        if minimum_segments < 1 or maximum_segments < minimum_segments:
            raise ValueError("Invalid CAD primitive segment-count bounds")
        if float(cad.get("minimumWidthFraction", 0.0)) <= 0:
            raise ValueError("cadPrimitive.minimumWidthFraction must be positive")
        if float(cad.get("maximumWidthFraction", 0.0)) <= float(
            cad.get("minimumWidthFraction", 0.0)
        ):
            raise ValueError("CAD primitive width bounds are inverted")
        bootstrap_maximum_width = float(
            cad.get("bootstrapMaximumWidthFraction", cad.get("maximumWidthFraction", 0.0))
        )
        representable_maximum_width = float(
            cad.get(
                "maximumRepresentableWidthFraction",
                cad.get("maximumWidthFraction", 0.0),
            )
        )
        if bootstrap_maximum_width < float(cad.get("minimumWidthFraction", 0.0)):
            raise ValueError(
                "cadPrimitive.bootstrapMaximumWidthFraction must be at least "
                "minimumWidthFraction"
            )
        if representable_maximum_width < bootstrap_maximum_width:
            raise ValueError(
                "cadPrimitive.maximumRepresentableWidthFraction must be at "
                "least bootstrapMaximumWidthFraction"
            )
        if representable_maximum_width > 1.0:
            raise ValueError(
                "cadPrimitive.maximumRepresentableWidthFraction must not exceed 1"
            )
        if float(cad.get("minimumFilletRadiusFraction", 0.0)) <= 0:
            raise ValueError("cadPrimitive.minimumFilletRadiusFraction must be positive")
        if not (0.0 < float(cad.get("minimumProgramMaskIoU", 0.82)) <= 1.0):
            raise ValueError("cadPrimitive.minimumProgramMaskIoU must lie in (0,1]")
        minimum_acceptance = float(
            cad.get("minimumGeneratedAcceptanceFraction", 0.0)
        )
        if not (0.0 <= minimum_acceptance <= 1.0):
            raise ValueError(
                "cadPrimitive.minimumGeneratedAcceptanceFraction must lie in [0,1]"
            )
        if int(cad.get("minimumSamplesBeforeAcceptanceGuard", 1)) < 1:
            raise ValueError(
                "cadPrimitive.minimumSamplesBeforeAcceptanceGuard must be >= 1"
            )
        if int(cad.get("componentRasterMinimumPixels", 1)) < 1:
            raise ValueError("cadPrimitive.componentRasterMinimumPixels must be >= 1")
        if int(cad.get("samePolarityComponentMoatPixels", 0)) < 0:
            raise ValueError(
                "cadPrimitive.samePolarityComponentMoatPixels must be >= 0"
            )
        if int(cad.get("componentClearanceRepairIterations", 1)) < 1:
            raise ValueError(
                "cadPrimitive.componentClearanceRepairIterations must be >= 1"
            )
        repair_mode = str(
            cad.get("componentClearanceRepairMode", "constructive_transform_pack_v2")
        ).strip().lower()
        if repair_mode not in {
            "legacy",
            "legacy_translation",
            "constructive_transform_pack_v2",
        }:
            raise ValueError(
                "cadPrimitive.componentClearanceRepairMode must be legacy or "
                "constructive_transform_pack_v2"
            )
        if int(cad.get("componentPackingRestarts", 1)) < 1:
            raise ValueError("cadPrimitive.componentPackingRestarts must be >= 1")
        if int(cad.get("componentPackingCandidates", 8)) < 8:
            raise ValueError("cadPrimitive.componentPackingCandidates must be >= 8")
        if int(cad.get("componentPackingRotationSteps", 0)) < 0:
            raise ValueError("cadPrimitive.componentPackingRotationSteps must be >= 0")
        if int(cad.get("componentPackingMaximumShrinkSteps", 0)) < 0:
            raise ValueError("cadPrimitive.componentPackingMaximumShrinkSteps must be >= 0")
        packing_shrink = float(cad.get("componentPackingShrinkFactor", 0.86))
        if not (0.5 <= packing_shrink < 1.0):
            raise ValueError(
                "cadPrimitive.componentPackingShrinkFactor must lie in [0.5,1)"
            )
        quota = int(cad.get("bootstrapReachabilityQuotaPerComponentCount", 0))
        if quota < 0:
            raise ValueError(
                "cadPrimitive.bootstrapReachabilityQuotaPerComponentCount must be >= 0"
            )
        reachability = float(
            cad.get("bootstrapReachabilityMinimumAcceptanceFraction", 0.0)
        )
        if not (0.0 <= reachability <= 1.0):
            raise ValueError(
                "cadPrimitive.bootstrapReachabilityMinimumAcceptanceFraction must lie in [0,1]"
            )
        if str(cad.get("bootstrapReachabilityAction", "raise")) not in {
            "raise", "warn", "record"
        }:
            raise ValueError(
                "cadPrimitive.bootstrapReachabilityAction must be raise, warn, or record"
            )
        clearance_relaxation = float(
            cad.get("componentClearanceRepairRelaxation", 0.65)
        )
        if not (0.0 < clearance_relaxation <= 1.0):
            raise ValueError(
                "cadPrimitive.componentClearanceRepairRelaxation must lie in (0,1]"
            )
        if not latent_mode:
            if str(ddpm_cfg.get("representation", "cad_sequence")) != "cad_sequence":
                raise ValueError(
                    "Direct CAD sequence mode requires ddpm.representation: cad_sequence"
                )
            hidden = int(ddpm_cfg.get("sequenceHiddenDimension", 192))
            heads = int(ddpm_cfg.get("sequenceAttentionHeads", 6))
            if hidden < 16 or heads < 1 or hidden % heads:
                raise ValueError(
                    "ddpm.sequenceHiddenDimension must be divisible by sequenceAttentionHeads"
                )
    if float(config["manufacturability"].get("minimumGapPixels", 0)) < 2:
        raise ValueError("minimumGapPixels must be at least 2 to retain a propellant layer")
    if float(config["geometry"].get("minimumElectrodeGap_m", 0)) <= 0:
        raise ValueError("geometry.minimumElectrodeGap_m must be positive")
    if float(config["manufacturability"].get("minimumGap_m", 0)) <= 0:
        raise ValueError("manufacturability.minimumGap_m must be positive")
    connectivity_mode = str(
        config["manufacturability"].get(
            "electrodeConnectivityMode", "implicit_3d_bus"
        )
    )
    if connectivity_mode not in {"implicit_3d_bus", "boundary_bus", "single_component"}:
        raise ValueError(
            "manufacturability.electrodeConnectivityMode must be "
            "implicit_3d_bus, boundary_bus, or single_component"
        )
    if connectivity_mode == "implicit_3d_bus" and not bool(
        config["manufacturability"].get("implicit3DBusDocumented", False)
    ):
        raise ValueError(
            "implicit_3d_bus requires implicit3DBusDocumented=true so the physical "
            "power-feed assumption is explicit"
        )
    for key in ("anodeBusBoundary", "cathodeBusBoundary"):
        if str(config["manufacturability"].get(key, "left")) not in {
            "left", "right", "top", "bottom"
        }:
            raise ValueError(f"Unsupported manufacturability.{key}")
    if int(config["manufacturability"].get("busContactPixels", 1)) < 1:
        raise ValueError("manufacturability.busContactPixels must be >= 1")
    connect_components = bool(
        config["manufacturability"].get("connectSamePolarityComponents", False)
    )
    maximum_components, _ = resolve_manufacturing_component_limits(config)
    if connectivity_mode == "single_component" and maximum_components != 1:
        raise ValueError(
            "single_component mode requires "
            "manufacturability.maximumComponentsPerPolarity=1"
        )
    if connectivity_mode == "single_component" and not connect_components:
        raise ValueError(
            "single_component mode requires "
            "manufacturability.connectSamePolarityComponents=true so generation "
            "projects same-polarity islands to one network"
        )
    bridge_width = int(
        config["manufacturability"].get(
            "connectivityBridgeWidthPixels",
            math.ceil(float(config["manufacturability"].get("minimumWidthPixels", 1))),
        )
    )
    if bridge_width < int(
        math.ceil(float(config["manufacturability"].get("minimumWidthPixels", 1)))
    ):
        raise ValueError(
            "manufacturability.connectivityBridgeWidthPixels must be at least "
            "minimumWidthPixels"
        )
    if float(
        config["manufacturability"].get(
            "maximumConnectivityAreaGrowthFraction", 2.0
        )
    ) < 0:
        raise ValueError(
            "manufacturability.maximumConnectivityAreaGrowthFraction must be >= 0"
        )
    if int(
        config["manufacturability"].get(
            "maximumConnectivityBridgePathPixels", 0
        )
    ) < 0:
        raise ValueError(
            "manufacturability.maximumConnectivityBridgePathPixels must be >= 0"
        )

    manufacturing = config["manufacturability"]
    maximum_per, maximum_total = resolve_manufacturing_component_limits(config)
    if maximum_per < 1:
        raise ValueError(
            "manufacturability.maximumComponentsPerPolarity must be >= 1"
        )
    if maximum_total < 2:
        raise ValueError(
            "manufacturability.maximumTotalComponents must be >= 2"
        )
    if maximum_total > 2 * maximum_per:
        raise ValueError(
            "manufacturability.maximumTotalComponents cannot exceed twice "
            "maximumComponentsPerPolarity"
        )
    cad = config.get("cadPrimitive", {})
    bootstrap_minimum, bootstrap_maximum = resolve_bootstrap_component_bounds(
        config
    )
    if bootstrap_minimum < 2:
        raise ValueError("cadPrimitive.bootstrapMinimumComponents must be >= 2")
    if bootstrap_maximum < bootstrap_minimum:
        raise ValueError(
            "cadPrimitive.bootstrapMaximumComponents must be >= "
            "bootstrapMinimumComponents"
        )
    if bootstrap_maximum > maximum_total:
        raise ValueError(
            "cadPrimitive.bootstrapMaximumComponents cannot exceed "
            "manufacturability.maximumTotalComponents; silent clipping would "
            "misstate the audited design space"
        )
    target_total_area = float(
        manufacturing.get("targetTotalElectrodeAreaFraction", 0.0)
    )
    if bool(manufacturing.get("enforceFixedTotalElectrodeArea", False)):
        if not (0.0 < target_total_area < 1.0):
            raise ValueError(
                "targetTotalElectrodeAreaFraction must lie in (0, 1) when fixed area is enabled"
            )
        target_share = float(
            manufacturing.get("targetAnodeShareOfElectrodeArea", 0.5)
        )
        if not (0.0 < target_share < 1.0):
            raise ValueError(
                "targetAnodeShareOfElectrodeArea must lie in (0, 1)"
            )
        if target_total_area > float(
            manufacturing.get("maximumTotalElectrodeAreaFraction", 1.0)
        ) + 1e-12:
            raise ValueError(
                "targetTotalElectrodeAreaFraction exceeds maximumTotalElectrodeAreaFraction"
            )
        minimum_area = float(
            manufacturing.get("minimumAreaFractionPerPolarity", 0.0)
        )
        if bool(manufacturing.get("enforceEqualPolarityArea", True)):
            if 0.5 * target_total_area + 1e-12 < minimum_area:
                raise ValueError(
                    "fixed equal-polarity target is below minimumAreaFractionPerPolarity"
                )
        if int(manufacturing.get("fixedAreaTolerancePixels", 0)) < 0:
            raise ValueError("fixedAreaTolerancePixels must be >= 0")
        if int(manufacturing.get("maximumFixedAreaBoundaryEdits", 0)) < 0:
            raise ValueError("maximumFixedAreaBoundaryEdits must be >= 0")
        if int(manufacturing.get("maximumFixedAreaBoundaryEditsAfterResize", 0)) < 0:
            raise ValueError("maximumFixedAreaBoundaryEditsAfterResize must be >= 0")
        resize_area_tolerance = float(
            manufacturing.get("resizeFixedAreaRelativeTolerance", 0.0)
        )
        if not (0.0 <= resize_area_tolerance <= 1.0):
            raise ValueError("resizeFixedAreaRelativeTolerance must lie in [0, 1]")

    minimum_corner_pixels = int(manufacturing.get("minimumCornerRadiusPixels", 0))
    minimum_corner_m = float(manufacturing.get("minimumCornerRadius_m", 0.0))
    if minimum_corner_pixels < 0 or minimum_corner_m < 0:
        raise ValueError("minimum corner radius values must be non-negative")
    if not (0.0 <= float(
        manufacturing.get("maximumCornerRoundingChangeFraction", 0.08)
    ) <= 1.0):
        raise ValueError(
            "maximumCornerRoundingChangeFraction must lie in [0, 1]"
        )
    if not (0.0 <= float(
        manufacturing.get(
            "maximumCornerRoundingChangeFractionAfterResize",
            manufacturing.get("maximumCornerRoundingChangeFraction", 0.08),
        )
    ) <= 1.0):
        raise ValueError(
            "maximumCornerRoundingChangeFractionAfterResize must lie in [0, 1]"
        )
    minimum_width = int(math.ceil(float(manufacturing.get("minimumWidthPixels", 1))))
    if minimum_width < 1:
        raise ValueError("minimumWidthPixels must be >= 1")
    physics_dtype = str(config["numerics"].get("physicsDtype", "float64")).lower()
    if physics_dtype not in {"float64", "double", "fp64", "float32", "single", "fp32"}:
        raise ValueError(f"Unsupported numerics.physicsDtype: {physics_dtype}")
    solver = config["numerics"].get("potentialSolver", {})
    allowed_solvers = {
        "auto",
        "pcg",
        "bicgstab",
        "bicg",
        "nonsymmetric_krylov",
        "rb_sor",
        "sor",
        "matlab_parity",
        "direct",
        "direct_cpu",
        "scipy_direct",
    }
    for key in ("method", "methodStatic", "methodCoupled"):
        if key in solver and str(solver[key]).lower() not in allowed_solvers:
            raise ValueError(
                f"Unsupported potential solver {key}={solver[key]!r}; "
                f"allowed={sorted(allowed_solvers)}"
            )
    if int(solver.get("directMaximumUnknowns", 500_000)) < 1:
        raise ValueError("potentialSolver.directMaximumUnknowns must be positive")
    if int(config["electrical"]["gridSize"]) < 17 or int(config["coupled"]["gridSize"]) < 17:
        raise ValueError("Electrical and coupled grids must be at least 17x17")
    if str(config.get("geometry", {}).get("generatorMode", "cad_sequence_diffusion")).lower() == "legacy_pixel_ddpm":
        if int(config["geometry"]["maskSize"]) != int(config["ddpm"]["imageSize"]):
            raise ValueError("geometry.maskSize and ddpm.imageSize must match in legacy pixel mode")
    if int(config["electrical"]["gridSize"]) % 2 == 0 or int(config["coupled"]["gridSize"]) % 2 == 0:
        raise ValueError("Use odd electrical/coupled grid sizes so the domain center is represented")
    if int(config["numerics"].get("speciesFixedSubsteps", 1)) < 1:
        raise ValueError("numerics.speciesFixedSubsteps must be at least 1")
    if int(config["numerics"].get("speciesMaximumSubsteps", 1)) < int(config["numerics"].get("speciesFixedSubsteps", 1)):
        raise ValueError("speciesMaximumSubsteps must be >= speciesFixedSubsteps")
    failure_cfg = config["numerics"].get("failureIsolation", {})
    maximum_failure_fraction = float(
        failure_cfg.get("maximumFailureFraction", 1.0)
    )
    if not (0.0 <= maximum_failure_fraction <= 1.0):
        raise ValueError(
            "numerics.failureIsolation.maximumFailureFraction must lie in [0,1]"
        )
    if int(failure_cfg.get("minimumAttemptsBeforeFailureFractionGuard", 1)) < 1:
        raise ValueError(
            "numerics.failureIsolation.minimumAttemptsBeforeFailureFractionGuard "
            "must be >= 1"
        )
    if str(failure_cfg.get("failureFractionAction", "raise")).lower() not in {
        "raise",
        "warn",
    }:
        raise ValueError(
            "numerics.failureIsolation.failureFractionAction must be raise or warn"
        )
    compatible_projection_limit = float(
        config.get("reducedCFD", {}).get("quality", {}).get(
            "maximumCompatibleProjectionResidualL2", float("inf")
        )
    )
    if compatible_projection_limit < 0.0 or math.isnan(
        compatible_projection_limit
    ):
        raise ValueError(
            "reducedCFD.quality.maximumCompatibleProjectionResidualL2 must be "
            "non-negative"
        )
    if float(config["coupled"]["timeStep_s"]) <= 0:
        raise ValueError("coupled.timeStep_s must be positive")
    coupled_dt = float(config["coupled"]["timeStep_s"])
    coupled_end = float(config["coupled"].get("endTime_s", 0.0))
    if coupled_end <= 0.0:
        raise ValueError("coupled.endTime_s must be positive")
    coupled_steps = coupled_end / coupled_dt
    if not math.isclose(
        coupled_steps,
        round(coupled_steps),
        rel_tol=0.0,
        abs_tol=1.0e-10 * max(1.0, abs(coupled_steps)),
    ):
        raise ValueError(
            "coupled.endTime_s must be an integer multiple of "
            "coupled.timeStep_s because the coupled integrator uses a constant dt"
        )
    electrical_interval = float(
        config["coupled"].get("electricalUpdateInterval_s", coupled_dt)
    )
    if electrical_interval <= 0.0:
        raise ValueError("coupled.electricalUpdateInterval_s must be positive")
    interval_steps = electrical_interval / coupled_dt
    if not math.isclose(
        interval_steps,
        round(interval_steps),
        rel_tol=0.0,
        abs_tol=1.0e-10 * max(1.0, abs(interval_steps)),
    ):
        raise ValueError(
            "coupled.electricalUpdateInterval_s must be an integer multiple of "
            "coupled.timeStep_s"
        )

    screening = config.get("screening", {})
    if int(screening.get("graphGridSize", 0)) < 3:
        raise ValueError("screening.graphGridSize must be at least 3")
    heat_times = screening.get("heatKernelTimes_s", [])
    if not isinstance(heat_times, (list, tuple)) or not heat_times:
        raise ValueError("screening.heatKernelTimes_s must be a non-empty list")
    if any((not math.isfinite(float(value))) or float(value) <= 0.0 for value in heat_times):
        raise ValueError("screening.heatKernelTimes_s values must be finite and positive")
    if int(screening.get("heatKernelImplicitSteps", 0)) < 1:
        raise ValueError("screening.heatKernelImplicitSteps must be >= 1")
    target_fraction = float(screening.get("percolationTargetFraction", 0.0))
    if not (0.0 < target_fraction <= 1.0):
        raise ValueError("screening.percolationTargetFraction must lie in (0,1]")
    if float(screening.get("percolationActivationThreshold", 0.0)) <= 0.0:
        raise ValueError("screening.percolationActivationThreshold must be positive")
    if float(screening.get("percolationMaximumMultiplier", 0.0)) <= 0.0:
        raise ValueError("screening.percolationMaximumMultiplier must be positive")
    for key in (
        "percolationBinarySearchIterations",
        "percolationMaximumSteps",
        "pcgMaximumIterations",
    ):
        if int(screening.get(key, 0)) < 1:
            raise ValueError(f"screening.{key} must be >= 1")
    if float(screening.get("laplacianRegularization", 0.0)) <= 0.0:
        raise ValueError("screening.laplacianRegularization must be positive")
    if float(screening.get("pcgTolerance", 0.0)) <= 0.0:
        raise ValueError("screening.pcgTolerance must be positive")
    if config["project"].get("requireCalibratedParameters", False):
        text = json.dumps(config)
        if "<" in text or config["project"].get("calibrationStatus") != "calibrated":
            raise ValueError(
                "Calibrated execution requested, but placeholders/calibration status remain"
            )


def save_resolved_json(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
