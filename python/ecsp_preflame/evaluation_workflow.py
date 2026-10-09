"""Shared evaluation and post-onset services for fixed electrode candidates.

This module constructs physics evaluators and preserves their validity/handoff
contracts. It does not generate offspring or run an evolutionary optimizer.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence
import csv
import copy
import json
import math

import numpy as np

from .baselines import (
    generate_area_matched_staggered, strict_json_dump,
    write_comparison_image, write_objective_comparison_csv,
)
from .evaluator import canonicalise_metrics, create_evaluator, EvaluatorError
from .geometry import GeometryLimits, RasterizedGeometry, save_geometry
from .candidate_ranking import CandidateEvaluation
from .configuration import normalize_preflame_config, is_preflame_model_status

OBJECTIVE_NAMES = (
    "ignition_delay_s",
    "area_undecomposed_fraction_at_evaluation_time",
    "minimum_ignition_voltage_V",
    "current_congestion",
)

PROPAGATION_OBJECTIVE_NAMES = (
    "final_unreacted_area_fraction",
    "established_time_after_onset_s",
    "negative_mean_regression_velocity_m_per_s",
    "reaction_front_nonuniformity",
)

REFINED_OBJECTIVE_NAMES = OBJECTIVE_NAMES + PROPAGATION_OBJECTIVE_NAMES


def _json_safe(value: Any) -> Any:
    """Convert NumPy scalars/non-finite floats into strict JSON values."""
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class EvaluationWorkflow:
    """Evaluate supplied candidates and their optional post-onset continuation."""

    def __init__(
        self,
        package_root: Path,
        config: Mapping[str, Any],
        workdir: Path,
        allow_debug_physics: bool = False,
    ) -> None:
        self.package_root = Path(package_root).resolve()
        self.config = normalize_preflame_config(config)
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        project = self.config.get("project", {})
        self.opt = dict(self.config.get("evaluation", self.config.get("optimization", {})))
        # Keep metric ordering explicit for fixed-candidate reports.
        self.opt["objectives_minimise"] = list(OBJECTIVE_NAMES)
        self.config["evaluation"] = self.opt
        self.end_time_s = float(self.config.get("physics", {}).get("end_time_s", 2.0))
        self.no_ignition_penalty_s = float(self.opt.get("no_ignition_penalty_s", 2.0))
        self.no_ignition_constraint_violation = float(
            self.opt.get("no_ignition_constraint_violation", 1.0)
        )
        self.minimum_no_ignition_violation = float(
            self.opt.get("minimum_no_ignition_violation", 1.0e-6)
        )
        self.vmin_cfg = dict(self.config.get("minimum_ignition_voltage_search", {}))
        self.reference_voltage_V = float(
            self.config.get("physics", {}).get("voltage_V", 260.0)
        )
        self.vmin_upper_bound_V = float(
            self.vmin_cfg.get("upper_bound_V", self.reference_voltage_V)
            if self.vmin_cfg.get("upper_bound_V", self.reference_voltage_V) is not None
            else self.reference_voltage_V
        )
        self.vmin_invalid_search_constraint_violation = float(
            self.opt.get("vmin_invalid_search_constraint_violation", 1.0)
        )
        if (
            not math.isfinite(self.end_time_s)
            or self.end_time_s <= 0.0
            or not math.isfinite(self.no_ignition_penalty_s)
            or self.no_ignition_penalty_s < 0.0
            or not math.isfinite(
                self.no_ignition_constraint_violation
            )
            or self.no_ignition_constraint_violation < 0.0
            or not math.isfinite(self.minimum_no_ignition_violation)
            or self.minimum_no_ignition_violation <= 1.0e-12
            or not math.isfinite(
                self.vmin_invalid_search_constraint_violation
            )
            or self.vmin_invalid_search_constraint_violation <= 1.0e-12
            or not math.isfinite(
                self.end_time_s + self.no_ignition_penalty_s
            )
        ):
            raise EvaluatorError(
                "Invalid evaluation failure-penalty contract: time/penalty "
                "values must be finite and non-negative, and minimum "
                "non-ignition/Vmin violations must exceed the "
                "feasibility tolerance"
            )
        ignition_cfg = dict(self.config.get("condensed_ignition", {}))
        self.ignition_onset_temperature_K = float(
            ignition_cfg.get("onset_temperature_K", 622.15)
        )
        self.batch_size = max(1, int(self.opt.get("physics_batch_size", 32)))
        self.limits = GeometryLimits(**self.config.get("geometry", {}))
        evaluator_cfg = dict(self.config.get("evaluator", {}))
        evaluator_cfg.setdefault("end_time_s", self.end_time_s)
        evaluator_cfg.setdefault("device", project.get("device", "auto"))
        evaluator_cfg.setdefault("voltage_V", self.config.get("physics", {}).get("voltage_V", 260.0))
        evaluator_cfg["physics_config"] = self.config
        self.evaluator = create_evaluator(
            self.package_root,
            evaluator_cfg,
            self.workdir / "adapter",
            allow_debug_physics,
        )
        evaluator_physics_cfg = getattr(self.evaluator, "config", {})
        if isinstance(evaluator_physics_cfg, Mapping):
            resolved_bc = evaluator_physics_cfg.get("preflameModel", {})
            if isinstance(resolved_bc, Mapping) and resolved_bc:
                resolved_end_time = float(resolved_bc["endTime_s"])
                if not math.isclose(
                    self.end_time_s,
                    resolved_end_time,
                    rel_tol=0.0,
                    abs_tol=64.0
                    * np.finfo(float).eps
                    * max(1.0, abs(self.end_time_s), abs(resolved_end_time)),
                ):
                    raise EvaluatorError(
                        "physics.end_time_s and preflame_model.endTime_s must agree"
                    )
                self.end_time_s = resolved_end_time
        self.initial_temperature_K = float(
            evaluator_physics_cfg.get("thermal", {}).get("initialTemperature_K", 298.15)
            if isinstance(evaluator_physics_cfg, Mapping)
            else 298.15
        )
        self.propagation_cfg = dict(self.config.get("propagation_refinement", {}))
        self.propagation_enabled = bool(self.propagation_cfg.get("enabled", False))
        from .post_onset import validate_post_onset_config
        self.post_onset_cfg = validate_post_onset_config(
            self.config.get("post_onset", {}), for_optimization=self.propagation_enabled
        )
        if self.propagation_enabled and not hasattr(self.evaluator, "evaluate_handoff"):
            raise EvaluatorError(
                "propagation_refinement requires an evaluator exposing evaluate_handoff; "
                "use evaluator.backend=preflame_torch"
            )
        self._save_effective_config()


    def _save_effective_config(self) -> None:
        data = copy.deepcopy(self.config)
        data["resolved_geometry_limits"] = asdict(self.limits)
        data["workflow_invariants"] = {
            "candidate_source": "caller_supplied_fixed_geometries",
            "evolutionary_optimization_enabled": False,
            "openfoam_enabled": False,
            "gas_phase_cfd_enabled": False,
            "objectives": list(OBJECTIVE_NAMES),
            "electrode_masks": "surface_contact_labels_do_not_remove_propellant",
            "post_onset_condensed_propagation_enabled": self.propagation_enabled,
            "post_onset_backend": self.post_onset_cfg["backend"],
            "post_onset_compare_backends": self.post_onset_cfg["compare_backends"],
        }
        (self.workdir / "effective_config.json").write_text(
            json.dumps(_json_safe(data), indent=2, allow_nan=False), encoding="utf-8"
        )

    @staticmethod
    def _metadata(ind: CandidateEvaluation, raster: RasterizedGeometry, generation: int) -> dict[str, Any]:
        return {
            "geometry_id": ind.geometry_id,
            "topology_id": ind.topology_id,
            "generation": generation,
            "source_role": ind.source_role,
            "genome": ind.genome,
            "geometry_descriptors": raster.descriptors,
            "geometry_constraint_violation": raster.constraint_violation,
            "geometry_violation_details": raster.violation_details,
            "intended_anode_components": len(ind.genome["anode"]["components"]),
            "intended_cathode_components": len(ind.genome["cathode"]["components"]),
            "surface_contact_model": True,
            "hidden_bus_assumed": True,
        }


    def _constraint_penalty_objectives(self, violation: float) -> np.ndarray:
        factor = 1.0 + max(float(violation), 0.0)
        return np.asarray(
            [
                self.end_time_s + self.no_ignition_penalty_s * factor,
                1.0 + factor,
                self.vmin_upper_bound_V + 100.0 * factor,
                1.0e9 * factor,
            ],
            dtype=float,
        )


    def _numerical_violation(self, metrics: Mapping[str, Any]) -> float:
        thresholds = self.opt.get("numerical_cap_thresholds", {})
        aliases = {
            "temperature": ("maximumTemperatureCapFraction", "maximum_temperature_cap_fraction"),
            "species": ("maximumSpeciesLimiterFraction", "maximum_species_limiter_fraction"),
            "gas": ("maximumGasCapFraction", "maximum_gas_cap_fraction"),
            "chemical_rate": ("maximumChemicalRateCapFraction", "maximum_chemical_rate_cap_fraction"),
        }
        lower = {str(k).lower(): v for k, v in metrics.items()}
        total = 0.0
        for name, keys in aliases.items():
            limit = float(thresholds.get(name, 0.02))
            if not np.isfinite(limit) or not 0.0 <= limit <= 1.0:
                raise ValueError(
                    f"evaluation.numerical_cap_thresholds.{name} must lie "
                    "in [0, 1]"
                )
            value = None
            for key in keys:
                if key in metrics:
                    value = metrics[key]
                    break
                if key.lower() in lower:
                    value = lower[key.lower()]
                    break
            if value is None:
                total += 100.0
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                numeric_value = float("nan")
            if not np.isfinite(numeric_value) or not 0.0 <= numeric_value <= 1.0:
                total += 100.0
                continue
            total += max(0.0, numeric_value - limit) / max(limit, 1e-12)
        converged = metrics.get("converged", metrics.get("solverConverged", False))
        if not bool(converged):
            total += 100.0
        return total


    def _apply_raw_metrics(self, ind: CandidateEvaluation, raw: Mapping[str, Any]) -> None:
        """Canonicalise one evaluator row and apply the production constraints."""
        canonical = canonicalise_metrics(
            raw,
            end_time_s=self.end_time_s,
            no_ignition_penalty_s=self.no_ignition_penalty_s,
        )
        numerical_violation = self._numerical_violation(canonical)
        ignition_success = bool(canonical.get("ignition_success", False))
        require_ignition = bool(
            self.opt.get("require_ignition_for_feasibility", True)
        )
        if ignition_success or not require_ignition:
            ignition_violation = 0.0
            temperature_shortfall = 0.0
        elif bool(self.opt.get("continuous_ignition_constraint", True)):
            preflame_model = is_preflame_model_status(canonical.get("modelStatus"))
            qualified_fraction = canonical.get("temperatureOnsetAreaFractionAt2s")
            required_fraction = canonical.get("ignitionMinimumAreaFraction")
            if (
                preflame_model
                and qualified_fraction is not None
                and required_fraction is not None
                and np.isfinite(float(qualified_fraction))
                and np.isfinite(float(required_fraction))
                and float(required_fraction) > 0.0
            ):
                # The electrochemical-thermal-decomposition onset criterion is conjunctive (temperature AND
                # global conversion over a minimum area).  Ranking infeasible
                # designs by peak temperature alone would incorrectly treat a
                # hot but chemically unreacted candidate as nearly feasible.
                temperature_shortfall = float(
                    np.clip(
                        (float(required_fraction) - float(qualified_fraction))
                        / float(required_fraction),
                        0.0,
                        1.0,
                    )
                )
            else:
                peak_temperature = canonical.get("peakMaximumTemperature_K")
                if peak_temperature is None or not np.isfinite(float(peak_temperature)):
                    temperature_shortfall = self.no_ignition_constraint_violation
                else:
                    denominator = max(
                        self.ignition_onset_temperature_K - self.initial_temperature_K,
                        1.0e-12,
                    )
                    temperature_shortfall = float(
                        np.clip(
                            (self.ignition_onset_temperature_K - float(peak_temperature))
                            / denominator,
                            0.0,
                            1.0,
                        )
                    )
            ignition_violation = max(
                temperature_shortfall,
                self.minimum_no_ignition_violation,
            )
        else:
            temperature_shortfall = self.no_ignition_constraint_violation
            # A configured zero severity must never turn a required-but-
            # missing ignition event into an feasible candidate.
            ignition_violation = max(
                temperature_shortfall,
                self.minimum_no_ignition_violation,
            )

        vmin_search_valid = bool(
            canonical.get("minimumIgnitionVoltageSearchValid", False)
        )
        vmin_search_violation = (
            0.0 if vmin_search_valid else self.vmin_invalid_search_constraint_violation
        )

        ind.constraint_violation += (
            numerical_violation + ignition_violation + vmin_search_violation
        )
        ind.metrics.update(canonical)
        ind.metrics["numerical_constraint_violation"] = numerical_violation
        ind.metrics["ignition_constraint_violation"] = ignition_violation
        ind.metrics["vmin_search_constraint_violation"] = vmin_search_violation
        ind.metrics["ignition_temperature_shortfall_fraction"] = temperature_shortfall
        ind.metrics["ignition_temperature_margin_K"] = (
            float(canonical.get("peakMaximumTemperature_K", float("nan")))
            - self.ignition_onset_temperature_K
            if canonical.get("peakMaximumTemperature_K") is not None
            else float("nan")
        )
        ind.objectives = np.asarray(canonical["objective_vector"], dtype=float)
        if numerical_violation > 0:
            ind.objectives = ind.objectives + self._constraint_penalty_objectives(
                numerical_violation
            )


    def _evaluate_population(
        self,
        population: Sequence[CandidateEvaluation],
        rasters: Mapping[str, RasterizedGeometry],
        generation: int,
    ) -> None:
        gen_dir = self.workdir / f"generation_{generation:03d}"
        geometry_dir = gen_dir / "geometries"
        physics_dir = gen_dir / "physics"
        geometry_dir.mkdir(parents=True, exist_ok=True)
        physics_dir.mkdir(parents=True, exist_ok=True)

        pending: list[tuple[CandidateEvaluation, RasterizedGeometry]] = []
        for ind in population:
            raster = rasters[ind.geometry_id]
            save_geometry(ind.genome, raster, geometry_dir)
            ind.metrics["geometry_descriptors"] = raster.descriptors
            ind.metrics["geometry_violation_details"] = raster.violation_details
            ind.constraint_violation = max(ind.constraint_violation, raster.constraint_violation)
            if ind.constraint_violation > 1e-12:
                ind.objectives = self._constraint_penalty_objectives(ind.constraint_violation)
                ind.metrics["physics_skipped"] = "manufacturability_constraint"
            else:
                pending.append((ind, raster))

        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            payload = []
            for ind, raster in batch:
                out = physics_dir / ind.geometry_id
                out.mkdir(parents=True, exist_ok=True)
                payload.append(
                    (
                        raster.anode_mask,
                        raster.cathode_mask,
                        self._metadata(ind, raster, generation),
                        out,
                    )
                )
            if hasattr(self.evaluator, "evaluate_batch"):
                raw_rows = self.evaluator.evaluate_batch(payload)
            else:
                raw_rows = [
                    self.evaluator.evaluate(a, c, m, o) for a, c, m, o in payload
                ]
            if len(raw_rows) != len(batch):
                raise EvaluatorError(
                    f"Evaluator returned {len(raw_rows)} rows for a batch of {len(batch)}"
                )
            for (ind, _), raw in zip(batch, raw_rows):
                self._apply_raw_metrics(ind, raw)

            processed = min(start + len(batch), len(pending))
            rejected = sum(
                bool(ind.metrics.get("physicsRejected", False))
                or float(ind.metrics.get("numerical_constraint_violation", 0.0)) >= 100.0
                for ind, _ in pending[:processed]
            )
            successful = processed - rejected
            elapsed = time.time() - self.start_time
            print(
                f"[generation {generation:03d}] physics {processed}/{len(pending)} "
                f"successful={successful} rejected={rejected} elapsed={elapsed:.1f}s",
                flush=True,
            )

        self._write_population_csv(population, gen_dir / "population_evaluated.csv")


    @staticmethod
    def _write_population_csv(
        population: Sequence[CandidateEvaluation],
        path: Path,
        objective_names: Sequence[str] = OBJECTIVE_NAMES,
    ) -> None:
        rows: list[dict[str, Any]] = []
        for ind in population:
            row: dict[str, Any] = {
                "geometry_id": ind.geometry_id,
                "topology_id": ind.topology_id,
                "source_role": ind.source_role,
                "rank": ind.rank,
                "crowding_distance": ind.crowding_distance,
                "constraint_violation": ind.constraint_violation,
            }
            if ind.objectives is not None:
                row.update(dict(zip(objective_names, [float(x) for x in ind.objectives])))
            for k, v in ind.metrics.items():
                if isinstance(v, (str, int, float, bool, np.number)) or v is None:
                    row[k] = v
            rows.append(row)
        keys = sorted({k for r in rows for k in r})
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


    def _area_matched_staggered_enabled(self) -> bool:
        cfg = self.config.get("baselines", {}).get("area_matched_staggered", {})
        return bool(cfg.get("enabled", True)) if isinstance(cfg, Mapping) else True


    def _recommended_area_match_target(
        self, recommendation: CandidateEvaluation | None
    ) -> tuple[float, str]:
        """Return the per-polarity area to match and an auditable source label."""
        if recommendation is not None:
            metric_pairs = (
                ("anodeContactAreaFraction", "cathodeContactAreaFraction", "recommended_physics_grid_contact_area"),
                ("anode_contact_area_fraction", "cathode_contact_area_fraction", "recommended_physics_grid_contact_area"),
            )
            for anode_key, cathode_key, source in metric_pairs:
                av = recommendation.metrics.get(anode_key)
                cv = recommendation.metrics.get(cathode_key)
                if av is not None and cv is not None:
                    avf, cvf = float(av), float(cv)
                    if np.isfinite(avf) and np.isfinite(cvf) and avf > 0.0 and cvf > 0.0:
                        return 0.5 * (avf + cvf), source
            descriptors = recommendation.metrics.get("geometry_descriptors", {})
            if isinstance(descriptors, Mapping):
                av = descriptors.get("anode_area_fraction")
                cv = descriptors.get("cathode_area_fraction")
                if av is not None and cv is not None:
                    avf, cvf = float(av), float(cv)
                    if np.isfinite(avf) and np.isfinite(cvf) and avf > 0.0 and cvf > 0.0:
                        return 0.5 * (avf + cvf), "recommended_design_grid_contact_area"
        return (
            float(self.limits.target_area_fraction_per_polarity),
            "configured_target_area_fraction_per_polarity",
        )


    def _evaluate_area_matched_staggered(
        self, final_dir: Path, recommendation: CandidateEvaluation | None = None
    ) -> CandidateEvaluation | None:
        """Evaluate one fair staggered reference for supplied candidate results.

        The reference is evaluated separately from the fixed candidate set. It uses the same
        domain, per-polarity contact area, width bounds, minimum gap, voltage,
        time horizon, physics grid and C++ FP64 evaluator as the AI design.  Its
        contact mask is the requested A-C-A-C array of two top-fed anode and two
        bottom-fed cathode fingers; the hidden buses are outside the propellant
        contact plane and therefore contribute zero mask area.
        """
        if not self._area_matched_staggered_enabled():
            return None
        cfg = self.config.get("baselines", {}).get("area_matched_staggered", {})
        if not isinstance(cfg, Mapping):
            cfg = {}
        physics_grid_size = int(getattr(self.evaluator, "grid_size", self.limits.grid_size))
        area_match_target, area_match_source = self._recommended_area_match_target(
            recommendation
        )
        genome, raster, parameters = generate_area_matched_staggered(
            self.limits,
            physics_grid_size=physics_grid_size,
            target_area_fraction_per_polarity=area_match_target,
            fingers_per_polarity=int(cfg.get("fingers_per_polarity", 2)),
            target_interdigitation_overlap_fraction=float(
                cfg.get("target_interdigitation_overlap_fraction", 0.50)
            ),
            minimum_interdigitation_overlap_fraction=float(
                cfg.get("minimum_interdigitation_overlap_fraction", 0.45)
            ),
            maximum_gap_safety_pixels=int(
                cfg.get("maximum_gap_safety_pixels", 8)
            ),
        )
        baseline_dir = final_dir / "area_matched_staggered"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        save_geometry(genome, raster, baseline_dir)
        individual = CandidateEvaluation(
            geometry_id=str(genome["geometry_id"]),
            genome=genome,
            topology_id=str(genome["topology_id"]),
            source_role="fixed_reference_area_matched_staggered",
            constraint_violation=float(raster.constraint_violation),
        )
        individual.metrics["geometry_descriptors"] = raster.descriptors
        individual.metrics["geometry_violation_details"] = raster.violation_details
        individual.metrics["baseline_parameters"] = parameters.__dict__
        individual.metrics["area_match_target_per_polarity"] = area_match_target
        individual.metrics["area_match_source"] = area_match_source
        individual.metrics["included_in_candidate_ranking"] = False
        individual.metrics["included_in_nsga2_population"] = False  # legacy report alias
        individual.metrics["included_in_final_pareto_front"] = False
        individual.metrics["selection_role"] = "fixed_reference_only"
        metadata = self._metadata(individual, raster, generation=-1)
        metadata.update(
            {
                "baseline_type": "area_matched_staggered",
                "baseline_layout_revision": "v7.9.2_hidden_bus_vertical_2a2c",
                "included_in_candidate_ranking": False,
                "included_in_nsga2_population": False,  # legacy report alias
                "included_in_final_pareto_front": False,
                # This fixed reference intentionally has two visible
                # contact components per polarity even when the AI search was
                # restricted to 1+1.  Each same-polarity pair is one electrical
                # terminal through a hidden bus outside the propellant plane.
                "allow_hidden_bus_reference_component_override": True,
                "reference_maximum_components_per_polarity": 2,
                "reference_maximum_total_components": 4,
                "reference_target_area_fraction_per_polarity": area_match_target,
                "hidden_bus_in_contact_mask": False,
                "hidden_bus_contact_area_fraction": 0.0,
            }
        )
        physics_dir = baseline_dir / "physics"
        physics_dir.mkdir(parents=True, exist_ok=True)
        raw = self.evaluator.evaluate(
            raster.anode_mask, raster.cathode_mask, metadata, physics_dir
        )
        self._apply_raw_metrics(individual, raw)
        self._write_population_csv(
            [individual], final_dir / "area_matched_staggered_metrics.csv"
        )
        strict_json_dump(
            final_dir / "area_matched_staggered.json",
            {
                "baseline_type": "area_matched_staggered",
                "baseline_layout_revision": "v7.9.2_hidden_bus_vertical_2a2c",
                "selection_role": "fixed_reference_only",
                "included_in_candidate_ranking": False,
                "included_in_nsga2_population": False,  # legacy report alias
                "included_in_final_pareto_front": False,
                "geometry_id": individual.geometry_id,
                "topology_id": individual.topology_id,
                "objectives": dict(
                    zip(OBJECTIVE_NAMES, [float(x) for x in individual.objectives])
                ),
                "constraint_violation": float(individual.constraint_violation),
                "ignition_success": bool(
                    individual.metrics.get("ignition_success", False)
                ),
                "geometry_descriptors": raster.descriptors,
                "baseline_parameters": parameters.__dict__,
                "area_match_target_per_polarity": area_match_target,
                "area_match_source": area_match_source,
                "metrics": individual.metrics,
                "fairness_contract": {
                    "domain_mm": self.limits.domain_mm,
                    "configured_target_area_fraction_per_polarity": self.limits.target_area_fraction_per_polarity,
                    "matched_area_fraction_per_polarity": area_match_target,
                    "area_match_source": area_match_source,
                    "minimum_gap_mm": self.limits.minimum_gap_mm,
                    "minimum_width_mm": self.limits.minimum_width_mm,
                    "maximum_width_mm": self.limits.maximum_width_mm,
                    "voltage_V": float(self.config.get("physics", {}).get("voltage_V", 260.0)),
                    "end_time_s": self.end_time_s,
                    "physics_grid_size": physics_grid_size,
                    "physics_backend": individual.metrics.get("backend"),
                    "visible_contact_components": {
                        "anode": 2,
                        "cathode": 2,
                        "total": 4,
                    },
                    "electrical_terminals": {"anode": 1, "cathode": 1},
                    "hidden_bus_in_contact_mask": False,
                    "hidden_bus_contact_area_fraction": 0.0,
                    "horizontal_contact_order": [
                        "anode",
                        "cathode",
                        "anode",
                        "cathode",
                    ],
                },
            },
        )
        source_png = baseline_dir / f"{individual.geometry_id}.png"
        source_npz = baseline_dir / f"{individual.geometry_id}.npz"
        if source_png.is_file():
            (final_dir / "area_matched_staggered.png").write_bytes(
                source_png.read_bytes()
            )
        if source_npz.is_file():
            (final_dir / "area_matched_staggered.npz").write_bytes(
                source_npz.read_bytes()
            )
        return individual


    @staticmethod
    def _comparison_record(individual: CandidateEvaluation) -> dict[str, Any]:
        # Reference reports must show physical/raw objectives,
        # not the large evaluation penalty vector used internally for infeasible
        # candidates.  Validity is reported separately.
        raw = individual.metrics.get("objective_vector")
        if raw is not None and len(raw) == len(OBJECTIVE_NAMES):
            values = [float(x) for x in raw]
            raw_unpenalized = True
        else:
            values = [float(x) for x in individual.objectives]
            raw_unpenalized = False
        numerical_violation = float(
            individual.metrics.get("numerical_constraint_violation", 0.0)
        )
        vmin_search_valid = bool(
            individual.metrics.get("minimumIgnitionVoltageSearchValid", False)
        )
        return {
            "geometry_id": individual.geometry_id,
            "topology_id": individual.topology_id,
            "objectives": dict(zip(OBJECTIVE_NAMES, values)),
            "objectives_are_raw_unpenalized": raw_unpenalized,
            "constraint_violation": float(individual.constraint_violation),
            "numerical_constraint_violation": numerical_violation,
            "numerically_valid": numerical_violation <= 1.0e-12,
            "vmin_search_valid": vmin_search_valid,
            "comparison_metric_valid": (
                numerical_violation <= 1.0e-12 and vmin_search_valid
            ),
            "ignition_success": bool(individual.metrics.get("ignition_success", False)),
        }


    def _write_area_matched_staggered_comparison(
        self,
        recommendation: CandidateEvaluation,
        baseline: CandidateEvaluation,
        final_dir: Path,
    ) -> None:
        recommended_record = self._comparison_record(recommendation)
        baseline_record = self._comparison_record(baseline)
        improvement: dict[str, float | None] = {}
        for name in OBJECTIVE_NAMES:
            ai_value = float(recommended_record["objectives"][name])
            baseline_value = float(baseline_record["objectives"][name])
            if np.isfinite(ai_value) and np.isfinite(baseline_value) and abs(baseline_value) > 1.0e-30:
                improvement[name] = 100.0 * (baseline_value - ai_value) / abs(baseline_value)
            else:
                improvement[name] = None

        write_objective_comparison_csv(
            final_dir / "recommended_vs_area_matched_staggered.csv",
            recommended_record,
            baseline_record,
            OBJECTIVE_NAMES,
        )
        strict_json_dump(
            final_dir / "recommended_vs_area_matched_staggered.json",
            {
                "comparison_scope": "fixed_reference_only",
                "baseline_excluded_from_candidate_ranking": True,
                "baseline_excluded_from_nsga2_and_pareto_selection": True,  # legacy report alias
                "lower_is_better_for_all_objectives": True,
                "comparison_uses_raw_unpenalized_objectives": True,
                "comparison_numerically_valid": bool(
                    recommended_record.get("comparison_metric_valid", False)
                    and baseline_record.get("comparison_metric_valid", False)
                ),
                "recommended_ai": recommended_record,
                "area_matched_staggered": baseline_record,
                "ai_improvement_percent_relative_to_staggered": improvement,
                "interpretation": (
                    "Positive improvement means the recommended AI design is lower/better; "
                    "negative improvement means the area-matched staggered reference is lower/better. "
                    "When comparison_numerically_valid is false, values are raw diagnostics and must not be treated as validated performance."
                ),
            },
        )
        write_comparison_image(
            final_dir / "recommended_vs_area_matched_staggered.png",
            final_dir / "recommended_design.png",
            final_dir / "area_matched_staggered.png",
            recommended_record,
            baseline_record,
            OBJECTIVE_NAMES,
        )


    def _record_baseline_failure(self, final_dir: Path, exc: Exception) -> None:
        strict_json_dump(
            final_dir / "AREA_MATCHED_STAGGERED_FAILED.json",
            {
                "status": "area_matched_staggered_failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "effect_on_candidate_evaluation": "none; baseline is a separate fixed reference",
            },
        )
        print(
            f"[baseline] area-matched staggered failed: {type(exc).__name__}: {exc}",
            flush=True,
        )


    def _resolved_propagation_config(self) -> dict[str, Any]:
        cfg = copy.deepcopy(self.propagation_cfg)
        evaluator_cfg = getattr(self.evaluator, "config", {})
        evaluator_geometry = (
            evaluator_cfg.get("geometry", {})
            if isinstance(evaluator_cfg, Mapping)
            else {}
        )
        thermal = (
            evaluator_cfg.get("preflameModel", {}).get("thermal", {})
            if isinstance(evaluator_cfg, Mapping)
            else {}
        )
        composition = getattr(self.evaluator, "composition", None)
        density = thermal.get("density_kg_per_m3")
        if density is None:
            density = getattr(composition, "density_kg_per_m3", None)
        if density is None:
            raise EvaluatorError("Propagation refinement requires a resolved condensed density")
        if "domainSize_m" not in evaluator_geometry:
            raise EvaluatorError(
                "Propagation refinement requires evaluator geometry.domainSize_m"
            )
        evaluator_domain = float(evaluator_geometry["domainSize_m"])
        design_domain = float(self.limits.domain_mm) * 1.0e-3
        if not math.isclose(
            evaluator_domain,
            design_domain,
            rel_tol=0.0,
            abs_tol=64.0 * np.finfo(float).eps * max(1.0, abs(evaluator_domain)),
        ):
            raise EvaluatorError(
                "Design-mask and pre-flame physics domain sizes disagree"
            )
        cfg["domain_size_m"] = evaluator_domain
        cfg["density_kg_per_m3"] = float(density)
        cfg["surface_layer_thickness_m"] = float(
            evaluator_geometry.get("surfaceLayerThickness_m", 1.0e-3)
        )
        numerical_quality = (
            evaluator_cfg.get("numericalQuality", {})
            if isinstance(evaluator_cfg, Mapping)
            else {}
        )
        if not isinstance(numerical_quality, Mapping):
            numerical_quality = {}
        cfg.setdefault(
            "maximum_thermal_stability_cfl",
            float(
                numerical_quality.get(
                    "maximumExplicitThermalCFL",
                    numerical_quality.get("maximumDiffusiveCFLFraction", 1.0),
                )
            ),
        )
        transport = (
            evaluator_cfg.get("transport", {})
            if isinstance(evaluator_cfg, Mapping)
            else {}
        )
        try:
            gas_constant = float(transport["gasConstant_J_per_molK"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluatorError(
                "Propagation refinement requires the resolved pre-flame gas constant"
            ) from exc
        if not math.isfinite(gas_constant) or gas_constant <= 0.0:
            raise EvaluatorError(
                "Resolved pre-flame gas constant must be finite and positive"
            )
        cfg["gas_constant_J_per_molK"] = gas_constant
        cfg.setdefault("maximum_temperature_cap_fraction", 0.0)
        evaluation = (
            evaluator_cfg.get("evaluation", evaluator_cfg.get("optimization", {}))
            if isinstance(evaluator_cfg, Mapping)
            else {}
        )
        cap_thresholds = (
            evaluation.get("numerical_cap_thresholds", {})
            if isinstance(evaluation, Mapping)
            else {}
        )
        if not isinstance(cap_thresholds, Mapping):
            cap_thresholds = {}
        cfg.setdefault(
            "maximum_chemical_rate_cap_fraction",
            float(cap_thresholds.get("chemical_rate", 0.0)),
        )
        return cfg


    def _propagation_metadata(
        self,
        ind: CandidateEvaluation,
        raster: RasterizedGeometry,
        *,
        baseline: bool = False,
    ) -> dict[str, Any]:
        metadata = self._metadata(ind, raster, generation=-2)
        metadata.update(
            {
                "selection_role": ind.metrics.get(
                    "propagation_selection_role",
                    "fixed_candidate_post_onset_refinement",
                ),
                "propagation_refinement": True,
                "propagation_scope": "post_onset_condensed_phase_not_gas_cfd",
            }
        )
        if baseline:
            metadata.update(
                {
                    "baseline_type": "area_matched_staggered",
                    "baseline_layout_revision": "v7.9.2_hidden_bus_vertical_2a2c",
                    "included_in_candidate_ranking": False,
                    "included_in_nsga2_population": False,  # legacy report alias
                    "included_in_final_pareto_front": False,
                    "allow_hidden_bus_reference_component_override": True,
                    "reference_maximum_components_per_polarity": 2,
                    "reference_maximum_total_components": 4,
                    "reference_target_area_fraction_per_polarity": float(
                        ind.metrics.get(
                            "area_match_target_per_polarity",
                            self.limits.target_area_fraction_per_polarity,
                        )
                    ),
                    "hidden_bus_in_contact_mask": False,
                    "hidden_bus_contact_area_fraction": 0.0,
                }
            )
        return metadata


    @staticmethod
    def _validate_handoff_reevaluation(
        ind: CandidateEvaluation, handoff_result: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Require the full-field rerun to reproduce the selected candidate."""
        rerun = handoff_result.get("metrics", {})
        comparisons = {
            "ignition_delay_s": (
                ind.metrics.get("ignition_delay_s"),
                rerun.get("ignitionDelay_s"),
            ),
            "remaining_reactive_mass_fraction": (
                ind.metrics.get("area_undecomposed_fraction_at_evaluation_time"),
                rerun.get("remainingReactiveMassFractionAtEvaluationTime"),
            ),
            "current_congestion": (
                ind.metrics.get("current_congestion"),
                rerun.get(
                    "peakCurrentCongestionToEvaluationTime",
                    rerun.get("peakCurrentCongestion"),
                ),
            ),
        }
        tolerances = {
            "ignition_delay_s": (1.0e-12, 0.0),
            "remaining_reactive_mass_fraction": (1.0e-8, 2.0e-6),
            "current_congestion": (2.0e-4, 5.0e-4),
        }
        deltas: dict[str, float | None] = {}
        inconsistent: list[str] = []
        for key, (reference, repeated) in comparisons.items():
            try:
                a = float(reference)
                b = float(repeated)
            except (TypeError, ValueError):
                deltas[key] = None
                inconsistent.append(f"{key}:missing_or_non_numeric")
                continue
            if not np.isfinite(a) or not np.isfinite(b):
                deltas[key] = None
                inconsistent.append(f"{key}:nonfinite")
                continue
            delta = abs(a - b)
            deltas[key] = delta
            absolute, relative = tolerances[key]
            allowed = absolute + relative * max(abs(a), abs(b))
            if delta > allowed:
                inconsistent.append(
                    f"{key}:delta={delta:.17g}>allowed={allowed:.17g}"
                )
        if inconsistent:
            raise EvaluatorError(
                "Pre-flame handoff re-evaluation is inconsistent with the "
                "optimized candidate metrics: " + "; ".join(inconsistent)
            )
        return {
            "preflameHandoffReevaluationAbsoluteDeltas": deltas,
            "preflameHandoffReevaluationTolerances": {
                key: {"absolute": absolute, "relative": relative}
                for key, (absolute, relative) in tolerances.items()
            },
            "preflameHandoffReevaluationConsistent": True,
        }


    def _run_propagation_refinement(
        self,
        ind: CandidateEvaluation,
        raster: RasterizedGeometry,
        output_dir: Path,
        *,
        baseline: bool = False,
        handoff_result: Mapping[str, Any] | None = None,
        precomputed_result: Any = None,
    ) -> dict[str, Any]:
        from .propagation import (
            PropagationCandidateError,
            PropagationCandidateInputError,
            ConfiguredModelTemperatureRangeExceeded,
            candidate_failure_payload,
            PropagationConfigurationError,
            run_condensed_propagation,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        metadata = self._propagation_metadata(ind, raster, baseline=baseline)
        try:
            if handoff_result is None:
                handoff_result = self.evaluator.evaluate_handoff(
                    raster.anode_mask,
                    raster.cathode_mask,
                    metadata,
                    output_dir / "bc_handoff",
                )
            if bool(handoff_result.get("handoffPreparationFailed", False)):
                raise PropagationCandidateInputError(
                    "electrochemical-thermal-decomposition propagation handoff preparation failed for this "
                    "candidate: "
                    + str(
                        handoff_result.get(
                            "handoffPreparationFailure", "unknown candidate failure"
                        )
                    )
                )
            handoff = handoff_result["handoff"]
            preflame_model_config = getattr(self.evaluator, "config", {}).get("preflameModel", {})
            if not preflame_model_config:
                raise PropagationConfigurationError(
                    "electrochemical-thermal-decomposition evaluator did not expose resolved preflameModel configuration"
                )
            if bool(handoff.get("onsetSucceeded", False)):
                try:
                    consistency = self._validate_handoff_reevaluation(
                        ind, handoff_result
                    )
                except EvaluatorError as exc:
                    raise PropagationCandidateInputError(str(exc)) from exc
            else:
                consistency = {"preflameHandoffReevaluationConsistent": None}
            post_cfg = getattr(self, "post_onset_cfg", {"backend": "condensed_propagation"})
            if precomputed_result is not None:
                if isinstance(precomputed_result, PropagationCandidateError):
                    raise precomputed_result
                metrics = dict(precomputed_result)
            elif post_cfg.get("backend", "condensed_propagation") == "condensed_propagation" and not post_cfg.get("compare_backends", False):
                # Preserve the original baseline call and exception boundary.
                metrics = run_condensed_propagation(
                    handoff, self._resolved_propagation_config(), preflame_model_config,
                    output_dir / "condensed_propagation",
                )
            else:
                from .post_onset import run_post_onset
                metrics = run_post_onset(
                    handoff,
                    self._resolved_propagation_config(),
                    preflame_model_config,
                    output_dir,
                    post_onset_config=post_cfg,
                    full_preflame_config=getattr(self.evaluator, "config", {}),
                )
            metrics = dict(metrics)
            metrics["propagationSucceeded"] = bool(
                metrics.get("status") == "complete"
                and metrics.get("onsetSucceeded", False)
                and metrics.get("postOnsetModelValid", True)
                and metrics.get("postOnsetValidityConstraintViolation", 0.0) == 0.0
            )
            if metrics["propagationSucceeded"]:
                metrics.setdefault("postOnsetModelValid", True)
                metrics.setdefault("postOnsetValidityConstraintViolation", 0.0)
                metrics.setdefault("postOnsetValidityReason", "")
            metrics["bcHandoffDirectory"] = str(output_dir / "bc_handoff")
            metrics.setdefault("propagationDirectory", str(output_dir / post_cfg.get("backend", "condensed_propagation")))
            metrics.update(consistency)
        except PropagationCandidateError as exc:
            failure = {
                "status": "propagation_failed",
                "propagationSucceeded": False,
                "errorType": type(exc).__name__,
                "message": str(exc),
                "finalUnreactedAreaFraction": 1.0,
                "meanEffectiveRegressionVelocity_m_per_s": 0.0,
                "maximumEffectiveRegressionVelocity_m_per_s": 0.0,
                "establishedTimeAfterOnset_s": float(self.propagation_cfg.get("duration_s", 1.0)),
                "reactionFrontNonuniformity": 1.0e6,
                "finalMeanGlobalProgress": 0.0,
                "finalMaximumTemperature_K": float("nan"),
                "onsetSucceeded": False,
                "continuedElectricalHeating": bool(
                    self.propagation_cfg.get("continued_electrical_heating", True)
                ),
            }
            if isinstance(exc, ConfiguredModelTemperatureRangeExceeded):
                failure.update(candidate_failure_payload(exc))
                # Post-onset validity says nothing adverse about the already
                # successful pre-flame onset or the candidate's geometry.
                failure["onsetSucceeded"] = bool(
                    handoff_result["handoff"].get("onsetSucceeded", False)
                )
            (output_dir / "PROPAGATION_FAILED.json").write_text(
                json.dumps(_json_safe(failure), indent=2, allow_nan=False),
                encoding="utf-8",
            )
            metrics = failure
        ind.metrics.update(metrics)
        return metrics


    @staticmethod
    def _propagation_comparison_record(ind: CandidateEvaluation) -> dict[str, Any]:
        return {
            "geometry_id": ind.geometry_id,
            "topology_id": ind.topology_id,
            "final_unreacted_area_fraction": float(
                ind.metrics.get("finalUnreactedAreaFraction", 1.0)
            ),
            "established_time_after_onset_s": float(
                ind.metrics.get("establishedTimeAfterOnset_s", float("nan"))
            ),
            "mean_regression_velocity_m_per_s": float(
                ind.metrics.get("meanEffectiveRegressionVelocity_m_per_s", 0.0)
            ),
            "reaction_front_nonuniformity": float(
                ind.metrics.get("reactionFrontNonuniformity", float("nan"))
            ),
            "propagation_succeeded": bool(
                ind.metrics.get("propagationSucceeded", False)
            ),
        }


    def _write_propagation_comparison(
        self,
        recommendation: CandidateEvaluation,
        baseline: CandidateEvaluation,
        final_dir: Path,
    ) -> None:
        ai = self._propagation_comparison_record(recommendation)
        ref = self._propagation_comparison_record(baseline)
        rows = []
        directions = {
            "final_unreacted_area_fraction": "minimise",
            "established_time_after_onset_s": "minimise",
            "mean_regression_velocity_m_per_s": "maximise",
            "reaction_front_nonuniformity": "minimise",
        }
        for name, direction in directions.items():
            ai_v = float(ai[name])
            ref_v = float(ref[name])
            if np.isfinite(ai_v) and np.isfinite(ref_v):
                delta = ai_v - ref_v
                ai_better = ai_v <= ref_v if direction == "minimise" else ai_v >= ref_v
            else:
                delta = float("nan")
                ai_better = False
            rows.append(
                {
                    "metric": name,
                    "direction": direction,
                    "recommended_ai": ai_v,
                    "area_matched_staggered": ref_v,
                    "ai_minus_staggered": delta,
                    "recommended_ai_better_or_equal": ai_better,
                }
            )
        with (final_dir / "recommended_vs_area_matched_staggered_propagation.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        strict_json_dump(
            final_dir / "recommended_vs_area_matched_staggered_propagation.json",
            {
                "scope": "post_onset_condensed_phase_reaction_progress_and_level_set_not_gas_cfd",
                "baseline_excluded_from_candidate_ranking": True,
                "baseline_excluded_from_optimization": True,  # legacy report alias
                "recommended_ai": ai,
                "area_matched_staggered": ref,
                "directions": directions,
                "comparison": rows,
            },
        )


    @staticmethod
    def _load_persisted_baseline_raster(
        final_dir: Path,
        baseline: CandidateEvaluation,
    ) -> RasterizedGeometry:
        baseline_npz = final_dir / "area_matched_staggered.npz"
        if not baseline_npz.is_file():
            raise RuntimeError(
                "Area-matched staggered masks were not persisted for propagation refinement"
            )
        with np.load(baseline_npz, allow_pickle=False) as data:
            return RasterizedGeometry(
                anode_mask=np.asarray(data["anode_mask"], dtype=bool),
                cathode_mask=np.asarray(data["cathode_mask"], dtype=bool),
                descriptors=dict(baseline.metrics.get("geometry_descriptors", {})),
                constraint_violation=float(baseline.constraint_violation),
                violation_details=dict(
                    baseline.metrics.get("geometry_violation_details", {})
                ),
            )
