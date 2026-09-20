"""Explicit DOE adapters around the unchanged production physics services.

No NSGA-II workflow is constructed or run. The few production workflow methods
used here contain the existing handoff, model-validity and comparison contracts.
Candidate generation, selection and result caching belong to the DOE caller.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ecsp_nsga2.baselines import generate_area_matched_staggered
from ecsp_nsga2.evaluator import EvaluatorError, canonicalise_metrics, create_evaluator
from ecsp_nsga2.geometry import GeometryLimits, RasterizedGeometry, save_geometry
from ecsp_nsga2.nsga2 import Individual, rank_and_crowd
from ecsp_nsga2.workflow import (
    NSGA2ElectricalSolidWorkflow as ProductionWorkflow,
    OBJECTIVE_NAMES,
    REFINED_OBJECTIVE_NAMES,
    _json_safe,
)
from .storage import digest, write_csv, write_json


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_csv(path, rows)


def _mask_hash(anode: np.ndarray, cathode: np.ndarray) -> str:
    digest = hashlib.sha256()
    for mask in (anode, cathode):
        array = np.asarray(mask, dtype=np.uint8)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _component_pair(metadata: Mapping[str, Any]) -> tuple[int, int]:
    pair = metadata.get("component_pair")
    if isinstance(pair, str):
        pair = {"1A1C": (1, 1), "2A2C": (2, 2)}.get(pair)
    if pair is None and "intended_anode_components" in metadata:
        pair = (metadata["intended_anode_components"], metadata["intended_cathode_components"])
    if pair is None or tuple(pair) not in ((1, 1), (2, 2)):
        raise EvaluatorError("DOE physics metadata requires symmetric component_pair 1A1C or 2A2C")
    return tuple(int(count) for count in pair)


def _pack_cached_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Store strict JSON while retaining the distinction between null and NaN.

    Human-facing reports use JSON null for nonfinite diagnostics. Resume data
    additionally records their paths so production comparison code receives
    the original numeric values, including signed infinities, on a restart.
    """
    nonfinite = []

    def encode(value, path):
        if isinstance(value, Mapping):
            return {str(key): encode(item, path + [str(key)]) for key, item in value.items()}
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return [encode(item, path + [index]) for index, item in enumerate(value)]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            nonfinite.append({"path": path, "value": str(value)})
            return None
        return value

    return {"metrics": encode(metrics, []), "nonfinite_metric_values": nonfinite}


def _unpack_cached_metrics(cached: Mapping[str, Any]) -> dict[str, Any]:
    metrics = copy.deepcopy(cached["metrics"])
    for entry in cached.get("nonfinite_metric_values", []):
        if entry["value"] not in ("nan", "inf", "-inf") or not entry["path"]:
            raise ValueError("Invalid nonfinite value in physics result checkpoint")
        container = metrics
        for key in entry["path"][:-1]:
            container = container[key]
        container[entry["path"][-1]] = float(entry["value"])
    return metrics


class ProductionPhysicsAdapter:
    """Reuse B/C evaluation and post-onset services without NSGA orchestration.

    ``evaluator`` is an explicit injection point for small, solver-free tests.
    Production callers omit it and receive the configured production evaluator.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        package_root: Path,
        run_root: Path,
        *,
        evaluator: Any = None,
    ) -> None:
        self.config = copy.deepcopy(dict(config))
        self.package_root = Path(package_root).resolve()
        self.workdir = Path(run_root).resolve()
        self.limits = GeometryLimits(**self.config.get("geometry", {}))
        self.opt = dict(self.config.get("optimization", {}))
        self.end_time_s = float(self.config.get("physics", {}).get("end_time_s", 2.0))
        self.no_ignition_penalty_s = float(self.opt.get("no_ignition_penalty_s", 2.0))
        self.propagation_cfg = copy.deepcopy(self.config.get("propagation_refinement", {}))
        from ecsp_nsga2.post_onset import validate_post_onset_config
        self.post_onset_cfg = validate_post_onset_config(
            self.config.get("post_onset", {}),
            for_optimization=bool(self.propagation_cfg.get("enabled", False)),
        )
        if evaluator is None:
            evaluator_cfg = dict(self.config.get("evaluator", {}))
            backend = str(evaluator_cfg.get("backend", "")).lower()
            if backend not in {
                "bc_global_preflame", "bc_global_torch", "paper_bc_global",
                "bc_global_preflame_propagation", "bc_global_native",
                "bc_native_cpu", "bc_native_cuda", "bc_global_native_hybrid",
                "bc_native_cpu_pool", "bc_global_native_cpu_pool",
            }:
                raise EvaluatorError("PHIDL DOE requires an existing production B/C evaluator")
            evaluator_cfg.setdefault("end_time_s", self.end_time_s)
            evaluator_cfg.setdefault("device", self.config.get("project", {}).get("device", "auto"))
            evaluator_cfg.setdefault("voltage_V", self.config.get("physics", {}).get("voltage_V", 260.0))
            evaluator_cfg["physics_config"] = self.config
            evaluator = create_evaluator(
                self.package_root, evaluator_cfg, self.workdir / "adapter", False,
            )
        self.evaluator = evaluator
        resolved_bc = getattr(evaluator, "config", {}).get("bcGlobal", {})
        if "endTime_s" in resolved_bc and not np.isclose(
            float(resolved_bc["endTime_s"]), self.end_time_s,
            rtol=0.0, atol=64.0 * np.finfo(float).eps * max(1.0, self.end_time_s),
        ):
            raise EvaluatorError("physics.end_time_s and bc_global.endTime_s must agree")
        self.config_hash = hashlib.sha256(
            json.dumps(_json_safe(self.config), sort_keys=True, allow_nan=False).encode("utf-8")
        ).hexdigest()

    def evaluate(
        self,
        anode_mask: np.ndarray,
        cathode_mask: np.ndarray,
        metadata: Mapping[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        """Make exactly one production call and retain its objective contract."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        metadata = dict(metadata)
        pair = _component_pair(metadata)
        # These explicit fields activate production post-resize geometry checks.
        metadata["intended_anode_components"], metadata["intended_cathode_components"] = pair
        metadata.setdefault("surface_contact_model", True)
        metadata.setdefault("hidden_bus_assumed", True)
        raw = self.evaluator.evaluate(anode_mask, cathode_mask, metadata, Path(output_dir))
        metrics = canonicalise_metrics(raw, self.end_time_s, self.no_ignition_penalty_s)
        violation = ProductionWorkflow._numerical_violation(self, metrics)
        reasons = []
        if violation > 1.0e-12 or bool(metrics.get("physicsRejected", False)):
            reasons.append("production_numerical_rejection")
        if not bool(metrics.get("minimumIgnitionVoltageSearchValid", False)):
            reasons.append("invalid_minimum_ignition_voltage_search")
        metrics["numerical_constraint_violation"] = violation
        # A numerically valid non-igniting result retains the production finite
        # no-ignition penalty and is still a completed, selectable DOE result.
        metrics["physics_success"] = not reasons
        metrics["doe_evaluation_status"] = "failed" if reasons else "success"
        metrics["doe_failure_reason"] = ";".join(reasons)
        return metrics

    @staticmethod
    def _individual(record: Mapping[str, Any]) -> tuple[Individual, RasterizedGeometry]:
        metadata = copy.deepcopy(dict(record["metadata"]))
        metrics = copy.deepcopy(dict(record["metrics"]))
        pair = _component_pair(metadata)
        genome = copy.deepcopy(metadata.get("genome", metadata))
        for polarity, count in zip(("anode", "cathode"), pair):
            genome.setdefault(polarity, {"components": [{} for _ in range(int(count))]})
        ind = Individual(
            geometry_id=str(metadata["geometry_id"]),
            topology_id=str(metadata["topology_id"]),
            genome=genome,
            objectives=np.asarray([metrics[name] for name in OBJECTIVE_NAMES], dtype=float),
            source_role=str(metadata.get("source_role", "phidl_doe_preflame_selected")),
            metrics=metrics,
            constraint_violation=float(metrics.get("numerical_constraint_violation", 0.0)),
        )
        raster = RasterizedGeometry(
            np.asarray(record["anode_mask"], dtype=bool),
            np.asarray(record["cathode_mask"], dtype=bool),
            dict(metadata.get("geometry_descriptors", metrics.get("geometry_descriptors", {}))),
            float(metadata.get("geometry_constraint_violation", 0.0)),
            dict(metadata.get("geometry_violation_details", {})),
        )
        ind.metrics.setdefault("geometry_descriptors", raster.descriptors)
        return ind, raster

    def evaluate_baseline(self, final_dir: Path) -> tuple[Individual, RasterizedGeometry]:
        """Generate the existing external reference and resume its pre-flame result."""
        final_dir = Path(final_dir)
        final_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.config.get("baselines", {}).get("area_matched_staggered", {})
        genome, raster, parameters = generate_area_matched_staggered(
            self.limits,
            physics_grid_size=int(getattr(self.evaluator, "grid_size", self.limits.grid_size)),
            target_area_fraction_per_polarity=self.limits.target_area_fraction_per_polarity,
            **{key: cfg[key] for key in (
                "fingers_per_polarity", "target_interdigitation_overlap_fraction",
                "minimum_interdigitation_overlap_fraction", "maximum_gap_safety_pixels",
            ) if key in cfg},
        )
        geometry_hash = _mask_hash(raster.anode_mask, raster.cathode_mask)
        genome.update(
            geometry_hash=geometry_hash, config_hash=self.config_hash,
            component_pair=[2, 2], geometry_descriptors=raster.descriptors,
            geometry_violation_details=raster.violation_details,
            selection_role="post_optimization_reference_only",
        )
        directory = final_dir / "area_matched_staggered"
        save_geometry(genome, raster, directory)
        for suffix in ("json", "png", "npz"):
            (final_dir / f"area_matched_staggered.{suffix}").write_bytes(
                (directory / f"{genome['geometry_id']}.{suffix}").read_bytes()
            )
        ind = Individual(
            geometry_id=str(genome["geometry_id"]), topology_id=str(genome["topology_id"]),
            genome=genome, source_role="post_optimization_area_matched_staggered_reference",
        )
        metadata = self._propagation_metadata(ind, raster, baseline=True)
        metadata["propagation_refinement"] = False
        metadata["selection_role"] = "post_optimization_reference_only"
        cache_path = directory / "preflame_result.json"
        cache = json.loads(cache_path.read_text()) if cache_path.is_file() else {}
        if cache.get("config_hash") == self.config_hash and cache.get("geometry_hash") == geometry_hash and cache.get("completed") is True:
            metrics = _unpack_cached_metrics(cache)
        else:
            metrics = self.evaluate(raster.anode_mask, raster.cathode_mask, metadata, directory / "physics")
            write_json(cache_path, dict(
                completed=True, geometry_hash=geometry_hash,
                config_hash=self.config_hash, **_pack_cached_metrics(metrics),
            ))
        ind.metrics.update(metrics)
        ind.metrics.update(
            geometry_descriptors=raster.descriptors,
            geometry_violation_details=raster.violation_details,
            baseline_parameters=parameters.__dict__,
            area_match_target_per_polarity=self.limits.target_area_fraction_per_polarity,
            area_match_source="configured_target_area_fraction_per_polarity",
            included_in_final_pareto_front=False,
        )
        ind.objectives = np.asarray([metrics[name] for name in OBJECTIVE_NAMES], dtype=float)
        ind.constraint_violation = float(metrics.get("numerical_constraint_violation", 0.0))
        ProductionWorkflow._write_population_csv([ind], final_dir / "area_matched_staggered_metrics.csv")
        return ind, ProductionWorkflow._load_persisted_baseline_raster(final_dir, ind)

    def write_preflame_comparison(self, selected, baseline, final_dir: Path) -> None:
        ref = ProductionWorkflow._comparison_record(baseline[0])
        rows = []
        for record in selected:
            ind, _ = self._individual(record)
            ai = ProductionWorkflow._comparison_record(ind)
            for name in OBJECTIVE_NAMES:
                value, reference = float(ai["objectives"][name]), float(ref["objectives"][name])
                rows.append(dict(
                    geometry_id=ind.geometry_id, topology_id=ind.topology_id, objective=name,
                    candidate=value, area_matched_staggered=reference,
                    candidate_minus_staggered=value-reference,
                    improvement_percent=(100.0*(reference-value)/abs(reference) if abs(reference) > 1e-30 else None),
                    comparison_valid=bool(ai["comparison_metric_valid"] and ref["comparison_metric_valid"]),
                ))
        _write_rows(Path(final_dir) / "preflame_top20_vs_staggered.csv", rows)

    # Deliberately narrow delegation: these are existing production services,
    # not copied equations, comparison calculations or optimizer entry points.
    _metadata = staticmethod(ProductionWorkflow._metadata)
    _validate_handoff_reevaluation = staticmethod(ProductionWorkflow._validate_handoff_reevaluation)
    _propagation_comparison_record = staticmethod(ProductionWorkflow._propagation_comparison_record)

    def _propagation_metadata(self, ind, raster, *, baseline=False):
        return ProductionWorkflow._propagation_metadata(self, ind, raster, baseline=baseline)

    def _resolved_propagation_config(self):
        return ProductionWorkflow._resolved_propagation_config(self)

    def refine_selected(self, selected, baseline, final_dir: Path) -> list[dict[str, Any]]:
        """Refine this frozen selection and one reference, never promote failures.

        The complete DOE pool is intentionally absent from this interface.
        Existing candidate-local model-validity handling is used unchanged for
        both roles, including the configured 2500 K validity rule.
        """
        from ecsp_nsga2.propagation import final_refinement_objectives

        final_dir = Path(final_dir)
        selection_path = final_dir / "preflame_top20.csv"
        frozen = selection_path.read_bytes()
        with selection_path.open(newline="", encoding="utf-8") as stream:
            frozen_ids = [row["geometry_id"] for row in csv.DictReader(stream)]
        selected_ids = [str(record["metadata"]["geometry_id"]) for record in selected]
        if frozen_ids != selected_ids or len(set(selected_ids)) != len(selected_ids):
            raise ValueError("Post-onset input must exactly match the immutable pre-flame selection")
        if baseline[0].geometry_id in selected_ids:
            raise ValueError("The external staggered reference cannot be part of DOE selection")
        prepared = [self._individual(record) for record in selected]
        prepared.append(copy.deepcopy(baseline))
        rows, valid, comparison_rows = [], [], []
        for index, (ind, raster) in enumerate(prepared):
            is_baseline = index == len(selected)
            directory = final_dir / "propagation" / ("area_matched_staggered" if is_baseline else ind.geometry_id)
            identity = digest({
                "geometry_id": ind.geometry_id, "config_hash": self.config_hash,
                "geometry_hash": _mask_hash(raster.anode_mask, raster.cathode_mask),
                "preflame_objectives": ind.objectives.tolist(), "baseline": is_baseline,
            })
            checkpoint = directory / "result_checkpoint.json"
            cached = json.loads(checkpoint.read_text()) if checkpoint.is_file() else {}
            if cached and cached.get("identity") != identity:
                raise RuntimeError(f"Persisted post-onset identity mismatch: {checkpoint}")
            reused = cached.get("status") == "completed"
            if reused:
                metrics = _unpack_cached_metrics(cached)
                ind.metrics.update(metrics)
            else:
                write_json(checkpoint, {"identity": identity, "status": "running"})
                metrics = ProductionWorkflow._run_propagation_refinement(
                    self, ind, raster, directory, baseline=is_baseline,
                )
                # A model-domain rejection is also a completed attempt. Resume
                # must preserve it rather than rerun or replace the candidate.
                write_json(checkpoint, {
                    "identity": identity, "status": "completed", **_pack_cached_metrics(metrics),
                })
            write_json(directory / "metrics.json", metrics)
            row = {"geometry_id": ind.geometry_id, "topology_id": ind.topology_id,
                   "baseline": is_baseline, "reused_post_onset_result": reused, **metrics}
            rows.append(row)
            # Persist attempted results incrementally even if a later candidate
            # raises a fatal configuration error. This cannot rewrite selection.
            _write_rows(final_dir / "propagation_metrics.csv", rows)
            eligible = (bool(metrics.get("propagationSucceeded", False))
                        and bool(metrics.get("postOnsetModelValid", True))
                        and metrics.get("postOnsetValidityConstraintViolation", 0.0) == 0.0)
            if eligible and not is_baseline:
                ind.metrics["preflame_objective_vector"] = ind.objectives.tolist()
                ind.objectives = final_refinement_objectives(ind.objectives, metrics)
                valid.append(ind)
        for ind, _ in prepared[:-1]:
            directory = final_dir / "propagation" / ind.geometry_id
            ProductionWorkflow._write_propagation_comparison(self, ind, prepared[-1][0], directory)
            with (directory / "recommended_vs_area_matched_staggered_propagation.csv").open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    comparison_rows.append({
                        "geometry_id": ind.geometry_id, "topology_id": ind.topology_id,
                        "candidate_post_onset_valid": bool(ind.metrics.get("propagationSucceeded", False)),
                        "baseline_post_onset_valid": bool(prepared[-1][0].metrics.get("propagationSucceeded", False)),
                        **row,
                    })
        rank_and_crowd(valid)
        valid.sort(key=lambda ind: (ind.rank, -ind.crowding_distance, ind.geometry_id))
        ProductionWorkflow._write_population_csv(
            valid, final_dir / "post_onset_valid_refined_ranking.csv", REFINED_OBJECTIVE_NAMES,
        )
        _write_rows(final_dir / "propagation_comparison.csv", comparison_rows)
        _write_rows(final_dir / "propagation_rejections.csv", [
            row for row in rows if not row.get("propagationSucceeded", False)
        ])
        if selection_path.read_bytes() != frozen:
            raise RuntimeError("The immutable pre-flame selection changed during refinement")
        return rows
