from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence
import csv
import copy
import json
import math
import random
import time

import numpy as np

from .baselines import (
    BaselineGeometryError,
    generate_area_matched_staggered,
    strict_json_dump,
    write_comparison_image,
    write_objective_comparison_csv,
)
from .diversity import farthest_point_selection, quota_preserving_mating_pool
from .evaluator import canonicalise_metrics, create_evaluator, EvaluatorError
from .geometry import (
    GeometryLimits,
    RasterizedGeometry,
    crossover_genomes,
    instantiate_variant,
    make_topology_templates,
    mutate_genome,
    rasterize_and_validate,
    save_geometry,
    topology_signature,
)
from .nsga2 import (
    Individual,
    binary_tournament,
    environmental_selection,
    pareto_front,
    rank_and_crowd,
    select_knee_by_utopia_distance,
)


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


class NSGA2ElectricalSolidWorkflow:
    """No-F condensed-phase grammar/NSGA-II closed loop.

    OpenFOAM, gas-phase CFD and the empirical flame-progress variable are
    deliberately absent. Every valid candidate is evaluated by the same
    electrochemical-solid thermal/decomposition model. The 70+20+10 subset is
    a reproduction/mating quota, not a destructive physics-evaluation filter.
    """

    def __init__(
        self,
        package_root: Path,
        config: Mapping[str, Any],
        workdir: Path,
        allow_debug_physics: bool = False,
    ) -> None:
        self.package_root = Path(package_root).resolve()
        self.config = dict(config)
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        project = self.config.get("project", {})
        self.seed = int(project.get("seed", 20260827))
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)
        self.opt = dict(self.config.get("optimization", {}))
        # v7.9.4 owns the active objective contract.  Normalise legacy custom
        # YAMLs so old energy-objective audit entries cannot contradict the
        # minimum-ignition-voltage metric actually used by NSGA-II.
        self.opt["objectives_minimise"] = list(OBJECTIVE_NAMES)
        self.config["optimization"] = self.opt
        self.population_size = int(self.opt.get("population_size", 1000))
        self.generations = int(self.opt.get("generations", 5))
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
                "Invalid optimization failure-penalty contract: time/penalty "
                "values must be finite and non-negative, and minimum "
                "non-ignition/Vmin violations must exceed the NSGA-II "
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
            resolved_bc = evaluator_physics_cfg.get("bcGlobal", {})
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
                        "physics.end_time_s and bc_global.endTime_s must agree"
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
                "use evaluator.backend=bc_global_preflame"
            )
        self.start_time = time.time()
        self._save_effective_config()

    def _save_effective_config(self) -> None:
        data = copy.deepcopy(self.config)
        data["resolved_geometry_limits"] = asdict(self.limits)
        data["workflow_invariants"] = {
            "openfoam_enabled": False,
            "gas_phase_cfd_enabled": False,
            "all_feasible_candidates_get_condensed_phase_physics": True,
            "bootstrap_all_topology_quotas_must_be_unique_and_feasible": True,
            "bootstrap_physics_grid_revalidation_before_any_candidate_PDE": True,
            "bootstrap_invalid_padding_allowed": False,
            "empirical_surface_progress_variable_present": False,
            "objectives": list(OBJECTIVE_NAMES),
            "mating_pool_quota": "70 performance + 20 topology coverage + 10 max-diversity",
            "final_recommendation": "minimum normalised distance to Pareto utopia point",
            "propellant_domain": f"full_{self.limits.domain_mm:g}x{self.limits.domain_mm:g}_mm_surface_layer_all_cells",
            "electrode_masks": "surface_contact_labels_do_not_remove_propellant",
            "component_limits": f"Na<={self.limits.maximum_components_per_polarity}, Nc<={self.limits.maximum_components_per_polarity}, Na+Nc<={self.limits.maximum_total_components}",
            "disconnected_component_connection": "implicit_out_of_plane_backside_bus",
            "post_optimization_baseline": "area_matched_staggered_excluded_from_nsga2_and_pareto",
            "minimum_ignition_voltage_objective": "bracketed_numerically_valid_threshold_within_configured_voltage_interval",
            "other_objectives_reference_voltage_V": self.reference_voltage_V,
            "bc_global_preflame_enabled": str(self.config.get("evaluator", {}).get("backend", "")).lower().startswith("bc_global"),
            "post_onset_condensed_propagation_enabled": self.propagation_enabled,
            "propagation_scope": "reaction_progress_and_level_set_condensed_phase_not_gas_cfd" if self.propagation_enabled else "disabled",
            "post_onset_backend": self.post_onset_cfg["backend"],
            "post_onset_compare_backends": self.post_onset_cfg["compare_backends"],
            "propagation_selection": "preflame_pareto_plus_near_pareto_max_diversity" if self.propagation_enabled else "disabled",
        }
        (self.workdir / "effective_config.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _metadata(ind: Individual, raster: RasterizedGeometry, generation: int) -> dict[str, Any]:
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

    def _initial_population(self) -> tuple[list[Individual], dict[str, RasterizedGeometry]]:
        from .bootstrap import build_bootstrap, resolved_physics_grid
        grid_size = int(getattr(self.evaluator, "grid_size",
                         resolved_physics_grid(self.config, self.package_root)))
        entries, report = build_bootstrap(
            self.config, physics_grid_size=grid_size,
            audit_dir=self.workdir / "geometry_bootstrap",
            cache_dir=self.config.get("_bootstrap_cache_directory"),
        )
        population: list[Individual] = []
        rasters: dict[str, RasterizedGeometry] = {}
        for genome, raster in entries:
            gid = str(genome["geometry_id"])
            ind = Individual(geometry_id=gid, genome=genome,
                             topology_id=str(genome["topology_id"]),
                             source_role="bootstrap_20x50")
            ind.metrics["geometry_descriptors"] = raster.descriptors
            population.append(ind)
            rasters[gid] = raster
        return population, rasters

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
                    f"optimization.numerical_cap_thresholds.{name} must lie "
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

    def _apply_raw_metrics(self, ind: Individual, raw: Mapping[str, Any]) -> None:
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
            bc_model = str(canonical.get("modelStatus", "")).startswith("bc_global_")
            qualified_fraction = canonical.get("temperatureOnsetAreaFractionAt2s")
            required_fraction = canonical.get("ignitionMinimumAreaFraction")
            if (
                bc_model
                and qualified_fraction is not None
                and required_fraction is not None
                and np.isfinite(float(qualified_fraction))
                and np.isfinite(float(required_fraction))
                and float(required_fraction) > 0.0
            ):
                # The B/C onset criterion is conjunctive (temperature AND
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
            # missing ignition event into an NSGA-II-feasible candidate.
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
        population: Sequence[Individual],
        rasters: Mapping[str, RasterizedGeometry],
        generation: int,
    ) -> None:
        gen_dir = self.workdir / f"generation_{generation:03d}"
        geometry_dir = gen_dir / "geometries"
        physics_dir = gen_dir / "physics"
        geometry_dir.mkdir(parents=True, exist_ok=True)
        physics_dir.mkdir(parents=True, exist_ok=True)

        pending: list[tuple[Individual, RasterizedGeometry]] = []
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
        population: Sequence[Individual],
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

    def _make_child(
        self,
        mating_pool: Sequence[Individual],
        generation: int,
        child_index: int,
        mode: str,
    ) -> tuple[Individual, RasterizedGeometry]:
        best: tuple[float, dict[str, Any], RasterizedGeometry] | None = None
        max_attempts = int(self.opt.get("offspring_generation_attempts", 40))
        for attempt in range(max_attempts):
            if mode == "crossover":
                p1 = binary_tournament(mating_pool, self.rng)
                p2 = binary_tournament(mating_pool, self.rng)
                genome = crossover_genomes(p1.genome, p2.genome, self.rng, self.limits)
                genome = mutate_genome(genome, self.rng, self.limits, mutation_strength=0.55)
                topology_id = f"X{generation:03d}_{topology_signature(genome)[:8]}"
                role = "crossover"
            elif mode == "mutation":
                p = binary_tournament(mating_pool, self.rng)
                genome = mutate_genome(p.genome, self.rng, self.limits, mutation_strength=1.0)
                topology_id = f"M{generation:03d}_{topology_signature(genome)[:8]}"
                role = "mutation"
            elif mode == "new_grammar":
                template = make_topology_templates(1, self.seed + generation * 100000 + child_index * 97 + attempt, self.limits)[0]
                genome = instantiate_variant(
                    template,
                    variant_index=attempt,
                    seed=self.seed + generation * 1000003 + child_index,
                    limits=self.limits,
                )
                topology_id = f"N{generation:03d}_{topology_signature(genome)[:8]}"
                role = "new_grammar_immigrant"
            else:
                raise ValueError(mode)
            gid = f"G{generation:03d}_{role}_{child_index:05d}"
            genome["geometry_id"] = gid
            genome["topology_id"] = topology_id
            raster = rasterize_and_validate(genome, self.limits)
            genome = raster.fitted_genome or genome
            item = (raster.constraint_violation, genome, raster)
            if best is None or item[0] < best[0]:
                best = item
            if raster.constraint_violation <= 1e-12:
                ind = Individual(gid, genome, topology_id=topology_id, source_role=role)
                ind.metrics["geometry_descriptors"] = raster.descriptors
                return ind, raster
        assert best is not None
        violation, genome, raster = best
        gid = str(genome["geometry_id"])
        ind = Individual(
            gid,
            genome,
            topology_id=str(genome["topology_id"]),
            source_role=f"{mode}_constrained",
            constraint_violation=float(violation),
        )
        ind.metrics["geometry_descriptors"] = raster.descriptors
        return ind, raster

    def _offspring(
        self,
        population: Sequence[Individual],
        generation: int,
    ) -> tuple[list[Individual], dict[str, RasterizedGeometry], list[Individual]]:
        rank_and_crowd(population)
        mating_pool = quota_preserving_mating_pool(
            population,
            performance_count=int(self.opt.get("mating_performance_count", 70)),
            topology_count=int(self.opt.get("mating_topology_count", 20)),
            diversity_count=int(self.opt.get("mating_diversity_count", 10)),
        )
        self._write_population_csv(
            mating_pool,
            self.workdir / f"generation_{generation:03d}" / "mating_pool_70_20_10.csv",
        )
        fractions = self.opt.get(
            "offspring_fractions", {"crossover": 0.60, "mutation": 0.25, "new_grammar": 0.15}
        )
        n_cross = int(round(self.population_size * float(fractions.get("crossover", 0.60))))
        n_mut = int(round(self.population_size * float(fractions.get("mutation", 0.25))))
        n_new = self.population_size - n_cross - n_mut
        modes = ["crossover"] * n_cross + ["mutation"] * n_mut + ["new_grammar"] * n_new
        self.rng.shuffle(modes)
        children: list[Individual] = []
        rasters: dict[str, RasterizedGeometry] = {}
        for i, mode in enumerate(modes):
            ind, raster = self._make_child(mating_pool, generation + 1, i, mode)
            children.append(ind)
            rasters[ind.geometry_id] = raster
        return children, rasters, mating_pool

    def _hypervolume_proxy(self, front: Sequence[Individual]) -> float:
        # Dimensionless monitoring metric: inverse mean normalised utopia distance.
        if not front:
            return 0.0
        x = np.asarray([i.objectives for i in front], dtype=float)
        lo = np.min(x, axis=0)
        hi = np.max(x, axis=0)
        span = np.where(hi > lo, hi - lo, 1.0)
        z = (x - lo) / span
        return float(np.mean(1.0 / (1.0 + np.linalg.norm(z, axis=1))))

    def _save_generation_summary(self, population: Sequence[Individual], generation: int) -> float:
        rank_and_crowd(population)
        front = pareto_front(population)
        directory = self.workdir / f"generation_{generation:03d}"
        self._write_population_csv(population, directory / "population_selected.csv")
        self._write_population_csv(front, directory / "pareto_front.csv")
        proxy = self._hypervolume_proxy(front)
        summary = {
            "generation": generation,
            "population_size": len(population),
            "feasible_count": sum(i.constraint_violation <= 1e-12 for i in population),
            "pareto_count": len(front),
            "hypervolume_proxy": proxy,
            "elapsed_s": time.time() - self.start_time,
        }
        (directory / "generation_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return proxy

    def _area_matched_staggered_enabled(self) -> bool:
        cfg = self.config.get("baselines", {}).get("area_matched_staggered", {})
        return bool(cfg.get("enabled", True)) if isinstance(cfg, Mapping) else True

    def _recommended_area_match_target(
        self, recommendation: Individual | None
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
        self, final_dir: Path, recommendation: Individual | None = None
    ) -> Individual | None:
        """Evaluate one fair staggered reference after optimisation only.

        The reference is never inserted into the NSGA-II population, mating
        pool, Pareto front or utopia-distance recommendation. It uses the same
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
        individual = Individual(
            geometry_id=str(genome["geometry_id"]),
            genome=genome,
            topology_id=str(genome["topology_id"]),
            source_role="post_optimization_area_matched_staggered_reference",
            constraint_violation=float(raster.constraint_violation),
        )
        individual.metrics["geometry_descriptors"] = raster.descriptors
        individual.metrics["geometry_violation_details"] = raster.violation_details
        individual.metrics["baseline_parameters"] = parameters.__dict__
        individual.metrics["area_match_target_per_polarity"] = area_match_target
        individual.metrics["area_match_source"] = area_match_source
        individual.metrics["included_in_nsga2_population"] = False
        individual.metrics["included_in_final_pareto_front"] = False
        individual.metrics["selection_role"] = "post_optimization_reference_only"
        metadata = self._metadata(individual, raster, generation=-1)
        metadata.update(
            {
                "baseline_type": "area_matched_staggered",
                "baseline_layout_revision": "v7.9.2_hidden_bus_vertical_2a2c",
                "included_in_nsga2_population": False,
                "included_in_final_pareto_front": False,
                # This post-optimization reference intentionally has two visible
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
                "selection_role": "post_optimization_reference_only",
                "included_in_nsga2_population": False,
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
    def _comparison_record(individual: Individual) -> dict[str, Any]:
        # Post-optimization reference reports must show physical/raw objectives,
        # not the large NSGA-II penalty vector used internally for infeasible
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
        recommendation: Individual,
        baseline: Individual,
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
                "comparison_scope": "post_optimization_reference_only",
                "baseline_excluded_from_nsga2_and_pareto_selection": True,
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
                "effect_on_nsga2": "none; baseline is post-optimization only",
            },
        )
        print(
            f"[baseline] area-matched staggered failed: {type(exc).__name__}: {exc}",
            flush=True,
        )

    def _select_propagation_candidates(
        self,
        population: Sequence[Individual],
        front: Sequence[Individual],
    ) -> list[Individual]:
        """Select the full/capped pre-flame Pareto set plus near-Pareto diversity.

        The DOCX contract asks that the propagation refinement not be a second
        arbitrary performance filter.  We therefore retain the pre-flame
        Pareto front (crowding-preserving cap only when necessary), then add
        10--20 % near-Pareto/max-min geometry-diverse insurance candidates.
        """
        cfg = self.propagation_cfg.get("selection", {})
        if not isinstance(cfg, Mapping):
            cfg = {}
        rank_and_crowd(population)
        maximum_pareto = max(1, int(cfg.get("maximum_pareto_candidates", len(front))))
        ordered_front = sorted(
            front,
            key=lambda ind: (-float(ind.crowding_distance), ind.geometry_id),
        )
        pareto_selected = list(ordered_front[:maximum_pareto])
        selected_ids = {ind.geometry_id for ind in pareto_selected}

        fraction = float(cfg.get("additional_fraction", 0.15))
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("propagation_refinement.selection.additional_fraction must be in [0,1]")
        minimum_additional = max(0, int(cfg.get("minimum_additional_candidates", 0)))
        requested_additional = max(
            minimum_additional,
            int(math.ceil(max(len(pareto_selected), 1) * fraction)),
        )
        maximum_total = max(
            len(pareto_selected),
            int(cfg.get("maximum_total_candidates", len(pareto_selected) + requested_additional)),
        )
        requested_additional = min(
            requested_additional,
            max(maximum_total - len(pareto_selected), 0),
        )
        maximum_near_rank = max(1, int(cfg.get("maximum_near_pareto_rank", 3)))
        feasible_remaining = [
            ind
            for ind in population
            if ind.constraint_violation <= 1.0e-12
            and ind.objectives is not None
            and ind.geometry_id not in selected_ids
        ]
        near_pool = [ind for ind in feasible_remaining if ind.rank <= maximum_near_rank]
        diverse = farthest_point_selection(
            near_pool,
            pareto_selected,
            requested_additional,
        )
        if len(diverse) < requested_additional:
            chosen = {ind.geometry_id for ind in diverse}
            fallback = [
                ind for ind in feasible_remaining if ind.geometry_id not in chosen
            ]
            diverse.extend(
                farthest_point_selection(
                    fallback,
                    pareto_selected + diverse,
                    requested_additional - len(diverse),
                )
            )

        for ind in pareto_selected:
            ind.metrics["propagation_selection_role"] = "preflame_pareto"
        for ind in diverse:
            ind.metrics["propagation_selection_role"] = "near_pareto_max_geometry_diversity"
        selected = pareto_selected + diverse
        for index, ind in enumerate(selected):
            ind.metrics["propagation_selection_index"] = index
            ind.metrics["propagation_selected"] = True
        return selected

    def _resolved_propagation_config(self) -> dict[str, Any]:
        cfg = copy.deepcopy(self.propagation_cfg)
        evaluator_cfg = getattr(self.evaluator, "config", {})
        evaluator_geometry = (
            evaluator_cfg.get("geometry", {})
            if isinstance(evaluator_cfg, Mapping)
            else {}
        )
        thermal = (
            evaluator_cfg.get("bcGlobal", {}).get("thermal", {})
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
        optimization = (
            evaluator_cfg.get("optimization", {})
            if isinstance(evaluator_cfg, Mapping)
            else {}
        )
        cap_thresholds = (
            optimization.get("numerical_cap_thresholds", {})
            if isinstance(optimization, Mapping)
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
        ind: Individual,
        raster: RasterizedGeometry,
        *,
        baseline: bool = False,
    ) -> dict[str, Any]:
        metadata = self._metadata(ind, raster, generation=-2)
        metadata.update(
            {
                "selection_role": ind.metrics.get(
                    "propagation_selection_role",
                    "post_optimization_propagation_refinement",
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
                    "included_in_nsga2_population": False,
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
        ind: Individual, handoff_result: Mapping[str, Any]
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
        ind: Individual,
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
                    "B/C propagation handoff preparation failed for this "
                    "candidate: "
                    + str(
                        handoff_result.get(
                            "handoffPreparationFailure", "unknown candidate failure"
                        )
                    )
                )
            handoff = handoff_result["handoff"]
            bc_config = getattr(self.evaluator, "config", {}).get("bcGlobal", {})
            if not bc_config:
                raise PropagationConfigurationError(
                    "B/C evaluator did not expose resolved bcGlobal configuration"
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
                    handoff, self._resolved_propagation_config(), bc_config,
                    output_dir / "condensed_propagation",
                )
            else:
                from .post_onset import run_post_onset
                metrics = run_post_onset(
                    handoff,
                    self._resolved_propagation_config(),
                    bc_config,
                    output_dir,
                    post_onset_config=post_cfg,
                    full_bc_config=getattr(self.evaluator, "config", {}),
                )
            metrics = dict(metrics)
            metrics["propagationSucceeded"] = bool(
                metrics.get("status") == "complete"
                and metrics.get("onsetSucceeded", False)
            )
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
            (output_dir / "PROPAGATION_FAILED.json").write_text(
                json.dumps(_json_safe(failure), indent=2, allow_nan=False),
                encoding="utf-8",
            )
            metrics = failure
        ind.metrics.update(metrics)
        return metrics

    @staticmethod
    def _propagation_comparison_record(ind: Individual) -> dict[str, Any]:
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
        recommendation: Individual,
        baseline: Individual,
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
                "baseline_excluded_from_optimization": True,
                "recommended_ai": ai,
                "area_matched_staggered": ref,
                "directions": directions,
                "comparison": rows,
            },
        )

    def _copy_recommendation_geometry(
        self, recommendation: Individual, final_dir: Path
    ) -> None:
        source_png = next(
            self.workdir.glob(
                f"generation_*/geometries/{recommendation.geometry_id}.png"
            ),
            None,
        )
        source_npz = next(
            self.workdir.glob(
                f"generation_*/geometries/{recommendation.geometry_id}.npz"
            ),
            None,
        )
        if source_png:
            (final_dir / "recommended_design.png").write_bytes(source_png.read_bytes())
        if source_npz:
            (final_dir / "recommended_design.npz").write_bytes(source_npz.read_bytes())

    @staticmethod
    def _load_persisted_baseline_raster(
        final_dir: Path,
        baseline: Individual,
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

    def _finalise_with_propagation(
        self,
        population: Sequence[Individual],
        front: Sequence[Individual],
        final_dir: Path,
    ) -> Individual:
        from .propagation import final_refinement_objectives

        preflame_recommendation = select_knee_by_utopia_distance(front)
        self._write_population_csv(
            front,
            final_dir / "preflame_final_pareto_designs.csv",
            objective_names=OBJECTIVE_NAMES,
        )
        self._write_population_csv(
            [preflame_recommendation],
            final_dir / "preflame_recommended_design_metrics.csv",
            objective_names=OBJECTIVE_NAMES,
        )
        strict_json_dump(
            final_dir / "preflame_recommended_design.json",
            {
                "selection_rule": "minimum normalised distance to the four-objective pre-flame Pareto utopia point",
                "geometry_id": preflame_recommendation.geometry_id,
                "topology_id": preflame_recommendation.topology_id,
                "objectives": dict(
                    zip(OBJECTIVE_NAMES, [float(x) for x in preflame_recommendation.objectives])
                ),
                "role": "pre_refinement_reference_only",
            },
        )

        candidates = self._select_propagation_candidates(population, front)
        self._write_population_csv(
            candidates,
            final_dir / "propagation_selection.csv",
            objective_names=OBJECTIVE_NAMES,
        )
        strict_json_dump(
            final_dir / "propagation_selection_policy.json",
            {
                "policy": "preflame Pareto plus near-Pareto/max-min geometry diversity",
                "configured_selection": self.propagation_cfg.get("selection", {}),
                "selected_count": len(candidates),
                "selected_geometry_ids": [ind.geometry_id for ind in candidates],
                "staggered_handled_separately": True,
            },
        )

        refined: list[Individual] = []
        propagation_root = final_dir / "propagation_candidates"
        prepared: list[tuple[Individual, Individual, RasterizedGeometry, Path]] = []
        handoff_payloads = []
        for source in candidates:
            raster = rasterize_and_validate(source.genome, self.limits)
            clone = copy.deepcopy(source)
            candidate_dir = propagation_root / clone.geometry_id
            prepared.append((source, clone, raster, candidate_dir))
            handoff_payloads.append(
                (
                    raster.anode_mask,
                    raster.cathode_mask,
                    self._propagation_metadata(clone, raster),
                    candidate_dir / "bc_handoff",
                )
            )

        if hasattr(self.evaluator, "evaluate_handoff_batch"):
            print(
                f"[propagation] preparing B/C handoff fields for {len(prepared)} candidates in batched mode",
                flush=True,
            )
            handoff_results = self.evaluator.evaluate_handoff_batch(handoff_payloads)
        else:
            handoff_results = [
                self.evaluator.evaluate_handoff(a, c, m, o)
                for a, c, m, o in handoff_payloads
            ]

        execution_cfg = self.propagation_cfg.get("execution", {})
        if not isinstance(execution_cfg, Mapping):
            execution_cfg = {}
        parallel_cases = max(1, int(execution_cfg.get("parallel_cases", 1)))

        def refine_one(
            index: int,
            item: tuple[Individual, Individual, RasterizedGeometry, Path],
            prepared_handoff: Mapping[str, Any],
            precomputed_result: Any = None,
        ) -> Individual:
            source, clone, raster, candidate_dir = item
            print(
                f"[propagation] candidate {index + 1}/{len(prepared)} {source.geometry_id}",
                flush=True,
            )
            metrics = self._run_propagation_refinement(
                clone,
                raster,
                candidate_dir,
                handoff_result=prepared_handoff,
                precomputed_result=precomputed_result,
            )
            if not bool(metrics.get("propagationSucceeded", False)):
                clone.constraint_violation += 100.0
            clone.objectives = final_refinement_objectives(source.objectives, metrics)
            clone.metrics["preflame_objective_vector"] = [
                float(x) for x in source.objectives
            ]
            clone.metrics["refined_objective_vector"] = [
                float(x) for x in clone.objectives
            ]
            clone.metrics["final_refinement_objective_names"] = list(
                REFINED_OBJECTIVE_NAMES
            )
            return clone

        if "execution" in self.post_onset_cfg:
            from .post_onset_batch import run_post_onset_batch
            from .propagation import PropagationCandidateError, PropagationCandidateInputError
            precomputed = [None]*len(prepared)
            ready = []
            for i, (item, result) in enumerate(zip(prepared, handoff_results)):
                try:
                    if result.get("handoffPreparationFailed", False):
                        raise PropagationCandidateInputError(str(result.get("handoffPreparationFailure", "handoff failed")))
                    if result["handoff"].get("onsetSucceeded",False):
                        try:
                            self._validate_handoff_reevaluation(item[1], result)
                        except EvaluatorError as exc:
                            raise PropagationCandidateInputError(str(exc)) from exc
                    ready.append(i)
                except PropagationCandidateError as exc:
                    precomputed[i]=exc
            if ready:
                batch_results=run_post_onset_batch(
                    [handoff_results[i]["handoff"] for i in ready],
                    self._resolved_propagation_config(),
                    getattr(self.evaluator,"config",{}).get("bcGlobal",{}),
                    [prepared[i][3] for i in ready],
                    post_onset_config=self.post_onset_cfg,
                    full_bc_config=getattr(self.evaluator,"config",{}),
                )
                for i,result in zip(ready,batch_results):precomputed[i]=result
            refined=[refine_one(i,item,handoff_results[i],precomputed[i]) for i,item in enumerate(prepared)]
        elif parallel_cases > 1 and len(prepared) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(parallel_cases, len(prepared))) as pool:
                futures = [
                    pool.submit(refine_one, i, item, handoff_results[i])
                    for i, item in enumerate(prepared)
                ]
                # Preserve deterministic candidate ordering independently of
                # worker completion order.
                refined = [future.result() for future in futures]
        else:
            refined = [
                refine_one(i, item, handoff_results[i])
                for i, item in enumerate(prepared)
            ]

        if self.post_onset_cfg.get("compare_backends", False):
            from .post_onset import write_backend_rank_comparison
            write_backend_rank_comparison(refined, final_dir)
        refined_front = pareto_front(refined)
        if not refined_front:
            strict_json_dump(
                final_dir / "NO_SUCCESSFUL_PROPAGATION_REFINEMENT.json",
                {
                    "status": "no_successful_propagation_refinement",
                    "selected_count": len(candidates),
                    "message": "All selected candidates failed propagation refinement; no final recommendation was fabricated.",
                },
            )
            raise RuntimeError(
                "All selected B/C candidates failed post-onset condensed propagation refinement"
            )
        recommendation = select_knee_by_utopia_distance(refined_front)
        self._write_population_csv(
            refined,
            final_dir / "all_propagation_refined_designs.csv",
            objective_names=REFINED_OBJECTIVE_NAMES,
        )
        self._write_population_csv(
            refined_front,
            final_dir / "final_pareto_designs.csv",
            objective_names=REFINED_OBJECTIVE_NAMES,
        )
        self._write_population_csv(
            [recommendation],
            final_dir / "recommended_design_metrics.csv",
            objective_names=REFINED_OBJECTIVE_NAMES,
        )
        strict_json_dump(
            final_dir / "recommended_design.json",
            {
                "selection_rule": "minimum Euclidean distance to the normalised eight-objective post-refinement Pareto utopia point",
                "geometry_id": recommendation.geometry_id,
                "topology_id": recommendation.topology_id,
                "preflame_objectives": dict(
                    zip(OBJECTIVE_NAMES, [float(x) for x in recommendation.objectives[:4]])
                ),
                "propagation_objectives": dict(
                    zip(PROPAGATION_OBJECTIVE_NAMES, [float(x) for x in recommendation.objectives[4:]])
                ),
                "all_minimisation_objectives": dict(
                    zip(REFINED_OBJECTIVE_NAMES, [float(x) for x in recommendation.objectives])
                ),
                "constraint_violation": float(recommendation.constraint_violation),
                "metrics": recommendation.metrics,
                "genome": recommendation.genome,
                "model_scope": "B/C pre-flame plus post-onset condensed reaction propagation/regression; no gas-phase CFD or visible-flame claim",
                "chemistry_scope": "reduced global LP/PVA chemistry with conversion-dependent two-channel kinetics, not an elementary mechanism",
                "handoff": (
                    "authorised full-field onset state: T, Xg, both channel progresses, "
                    "reactant/product inventories, electrochemical LP consumption, "
                    "potential, initial-inventory scalars, qJ and qEchem; post-onset "
                    "electrical sources are not replayed; ignition coordinate is not prescribed"
                ),
            },
        )
        self._copy_recommendation_geometry(recommendation, final_dir)

        baseline: Individual | None = None
        try:
            baseline = self._evaluate_area_matched_staggered(
                final_dir, recommendation=recommendation
            )
        except Exception as exc:
            self._record_baseline_failure(final_dir, exc)
        if baseline is not None:
            self._write_area_matched_staggered_comparison(
                recommendation, baseline, final_dir
            )
            baseline_raster = self._load_persisted_baseline_raster(
                final_dir, baseline
            )
            baseline_metrics = self._run_propagation_refinement(
                baseline,
                baseline_raster,
                final_dir / "area_matched_staggered_propagation",
                baseline=True,
            )
            self._write_population_csv(
                [baseline],
                final_dir / "area_matched_staggered_propagation_metrics.csv",
                objective_names=OBJECTIVE_NAMES,
            )
            strict_json_dump(
                final_dir / "area_matched_staggered_propagation.json",
                {
                    "baseline_type": "area_matched_staggered",
                    "included_in_optimization": False,
                    "preflame_objectives": self._comparison_record(baseline)["objectives"],
                    "propagation_metrics": baseline_metrics,
                },
            )
            self._write_propagation_comparison(recommendation, baseline, final_dir)
        return recommendation

    def _finalise(self, population: Sequence[Individual]) -> Individual:
        rank_and_crowd(population)
        front = pareto_front(population)
        final_dir = self.workdir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        if not front:
            # A reference can still be informative when no AI candidate is
            # feasible, but it remains separate from optimisation.
            try:
                baseline = self._evaluate_area_matched_staggered(final_dir)
                if baseline is not None and self.propagation_enabled:
                    baseline_raster = self._load_persisted_baseline_raster(
                        final_dir, baseline
                    )
                    baseline_metrics = self._run_propagation_refinement(
                        baseline,
                        baseline_raster,
                        final_dir / "area_matched_staggered_propagation",
                        baseline=True,
                    )
                    strict_json_dump(
                        final_dir / "area_matched_staggered_propagation.json",
                        {
                            "baseline_type": "area_matched_staggered",
                            "included_in_optimization": False,
                            "preflame_objectives": self._comparison_record(baseline)["objectives"],
                            "propagation_metrics": baseline_metrics,
                        },
                    )
            except Exception as exc:
                self._record_baseline_failure(final_dir, exc)
            ignited = sum(bool(i.metrics.get("ignition_success", False)) for i in population)
            feasible = sum(i.constraint_violation <= 1e-12 for i in population)
            diagnostic = {
                "status": "no_feasible_igniting_design",
                "population_size": len(population),
                "ignition_success_count": ignited,
                "feasible_count": feasible,
                "message": (
                    "No candidate satisfied the condensed-phase ignition and numerical/manufacturing "
                    "constraints. No recommendation was fabricated. Inspect generation CSV files, "
                    "physics_rejection.txt files, calibration, voltage, and the 2 s horizon."
                ),
            }
            (final_dir / "NO_FEASIBLE_IGNITING_DESIGN.json").write_text(
                json.dumps(diagnostic, indent=2), encoding="utf-8"
            )
            raise RuntimeError(diagnostic["message"])

        if self.propagation_enabled:
            return self._finalise_with_propagation(population, front, final_dir)

        # Freeze the AI recommendation before creating/evaluating the external
        # baseline. This makes the non-interference contract explicit.
        recommendation = select_knee_by_utopia_distance(front)
        self._write_population_csv(front, final_dir / "final_pareto_designs.csv")
        self._write_population_csv([recommendation], final_dir / "recommended_design_metrics.csv")
        (final_dir / "recommended_design.json").write_text(
            json.dumps(
                _json_safe({
                    "selection_rule": "minimum Euclidean distance to the normalised Pareto utopia point",
                    "geometry_id": recommendation.geometry_id,
                    "topology_id": recommendation.topology_id,
                    "objectives": dict(zip(OBJECTIVE_NAMES, [float(x) for x in recommendation.objectives])),
                    "constraint_violation": recommendation.constraint_violation,
                    "metrics": recommendation.metrics,
                    "genome": recommendation.genome,
                    "caveat": (
                        "Objectives 1, 2 and 4 are evaluated at the fixed reference voltage. "
                        "Objective 3 is the bracketed minimum applied voltage that reaches the same "
                        "condensed-phase ignition criterion within the finite horizon. Energy-to-ignition "
                        "is retained as a diagnostic only. No flame-progress variable, burned-area claim, "
                        "gas-phase CFD, or gas-flame stability claim is made."
                    ),
                }),
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        source_png = next(self.workdir.glob(f"generation_*/geometries/{recommendation.geometry_id}.png"), None)
        source_npz = next(self.workdir.glob(f"generation_*/geometries/{recommendation.geometry_id}.npz"), None)
        if source_png:
            (final_dir / "recommended_design.png").write_bytes(source_png.read_bytes())
        if source_npz:
            (final_dir / "recommended_design.npz").write_bytes(source_npz.read_bytes())

        baseline: Individual | None = None
        try:
            baseline = self._evaluate_area_matched_staggered(
                final_dir, recommendation=recommendation
            )
        except Exception as exc:
            self._record_baseline_failure(final_dir, exc)
        if baseline is not None:
            self._write_area_matched_staggered_comparison(
                recommendation, baseline, final_dir
            )
        return recommendation

    def run(self) -> Individual:
        population, rasters = self._initial_population()
        self._evaluate_population(population, rasters, generation=0)
        previous_proxy: float | None = None
        stagnant = 0
        patience = int(self.opt.get("early_stop_patience", 2))
        min_improvement = float(self.opt.get("early_stop_min_improvement", 1e-3))
        self._save_generation_summary(population, 0)

        for generation in range(self.generations - 1):
            children, child_rasters, _ = self._offspring(population, generation)
            self._evaluate_population(children, child_rasters, generation + 1)
            population = environmental_selection(
                list(population) + list(children), self.population_size
            )
            proxy = self._save_generation_summary(population, generation + 1)
            if previous_proxy is not None and proxy - previous_proxy < min_improvement:
                stagnant += 1
            else:
                stagnant = 0
            previous_proxy = proxy
            if stagnant >= patience:
                (self.workdir / "EARLY_STOPPED.txt").write_text(
                    f"Stopped after generation {generation + 1}: proxy improvement below {min_improvement} "
                    f"for {patience} generations.\n",
                    encoding="utf-8",
                )
                break
        recommendation = self._finalise(population)
        (self.workdir / "RUN_COMPLETE.json").write_text(
            json.dumps(
                {
                    "recommended_geometry_id": recommendation.geometry_id,
                    "elapsed_s": time.time() - self.start_time,
                    "openfoam_used": False,
                    "model_scope": (
                        "bc_global_preflame_plus_post_onset_condensed_reaction_propagation"
                        if self.propagation_enabled
                        else "condensed_phase_no_empirical_surface_progress"
                    ),
                    "bc_global_preflame_used": str(
                        self.config.get("evaluator", {}).get("backend", "")
                    ).lower().startswith("bc_global"),
                    "post_onset_condensed_propagation_used": (
                        self.propagation_enabled and (
                            self.post_onset_cfg["backend"] == "condensed_propagation"
                            or self.post_onset_cfg.get("compare_backends", False)
                        )
                    ),
                    "post_onset_backend": self.post_onset_cfg["backend"] if self.propagation_enabled else "disabled",
                    "post_onset_reactive_euler_used": (
                        self.propagation_enabled and (
                            self.post_onset_cfg["backend"] == "reactive_euler"
                            or self.post_onset_cfg.get("compare_backends", False)
                        )
                    ),
                    "reactive_ranking_experimentally_validated": False,
                    "gas_phase_cfd_used": False,
                    "area_matched_staggered_evaluated": bool(
                        (self.workdir / "final" / "area_matched_staggered.json").is_file()
                    ),
                    "area_matched_staggered_included_in_optimization": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return recommendation
