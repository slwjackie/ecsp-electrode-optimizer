"""Resumable 100x3 -> 30x10 -> 20 DOE orchestration.

All physics is delegated; this module contains no field equations, geometry
repair, optimizer, crossover or mutation. Tests inject a counting evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Mapping
import copy
import hashlib
import traceback

import numpy as np
import yaml

from ecsp_nsga2.geometry import GeometryLimits
from ecsp_nsga2.bootstrap import resolved_physics_grid
from .selection import OBJECTIVES, rank_designs, select_final, select_topologies
from .storage import (RejectionLog, atomic_bytes, canonical_json, contact_sheet, digest,
                      load_record, read_json, run_lock, runtime_versions, save_record,
                      source_fingerprint, write_csv, write_json)


@dataclass(frozen=True)
class DOEOptions:
    topology_count: int = 100
    stage1_variants: int = 3
    selected_topologies: int = 30
    stage2_variants: int = 10
    final_count: int = 20

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.selected_topologies > self.topology_count:
            raise ValueError("selected_topologies exceeds topology_count")
        if self.final_count > self.topology_count * self.stage1_variants + self.selected_topologies * self.stage2_variants:
            raise ValueError("final_count exceeds total candidate count")


def derived_seed(seed, *parts):
    return int(digest([int(seed), *parts])[:8], 16)


class DOEWorkflow:
    def __init__(self, package_root, config, run_root, *, options=None,
                 physics=None, library_generator=None, variant_generator=None):
        self.package_root = Path(package_root).resolve()
        self.run_root = Path(run_root).resolve()
        self.config = copy.deepcopy(dict(config))
        self.geometry_options = dict(self.config.get("phidl_doe", {}))
        option_values = {k: self.geometry_options[k] for k in DOEOptions.__dataclass_fields__ if k in self.geometry_options}
        self.options = options or DOEOptions(**option_values)
        self.geometry_options.update(asdict(self.options))
        self.config["phidl_doe"] = self.geometry_options
        self.seed = int(self.config["project"]["seed"])
        self.limits = GeometryLimits(**self.config["geometry"])
        self.physics_grid_size = resolved_physics_grid(self.config, self.package_root)
        self.physics = physics
        self.library_generator = library_generator
        self.variant_generator = variant_generator
        base = self.config.get("evaluator", {}).get("base_config")
        base_path = (self.package_root / base).resolve() if base else None
        if base_path and not base_path.is_file():
            raise FileNotFoundError(f"Missing evaluator base_config: {base_path}")
        self.identity = {"config": self.config, "resolved_geometry_limits": asdict(self.limits),
                         "base_config_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest() if base_path else None,
                         "code_sha256": source_fingerprint(self.package_root), "runtime": runtime_versions()}
        self.config_hash = digest(self.identity)

    def _initialise(self, resume):
        self.run_root.mkdir(parents=True, exist_ok=True)
        manifest = self.run_root / "run_identity.json"
        if manifest.exists():
            if not resume:
                raise RuntimeError("Run directory already initialized; use --resume")
            if read_json(manifest).get("config_hash") != self.config_hash:
                raise RuntimeError("Resume config/code/dependency fingerprint mismatch; use a new run directory")
        else:
            if resume:
                raise RuntimeError("Cannot resume: run_identity.json is missing")
            write_json(manifest, {"config_hash": self.config_hash, **self.identity}, immutable=True)
        atomic_bytes(self.run_root / "config_snapshot.yaml", yaml.safe_dump(self.config, sort_keys=True).encode(), immutable=True)

    def _physics(self):
        if self.physics is None:
            from .physics import ProductionPhysicsAdapter
            self.physics = ProductionPhysicsAdapter(self.config, self.package_root, self.run_root)
        return self.physics

    def _library(self):
        folder = self.run_root / "topology_library"
        specs_path = folder / "topology_specs.json"
        if specs_path.exists():
            specs = read_json(specs_path)
        else:
            if self.library_generator is None:
                from .topology import generate_topology_library
                self.library_generator = generate_topology_library
            specs = self.library_generator(self.limits, self.options.topology_count,
                derived_seed(self.seed, "topologies"), self.physics_grid_size, self.geometry_options,
                rejection_callback=RejectionLog(folder / "generation_rejections.csv"))
            if len(specs) != self.options.topology_count:
                raise RuntimeError("Topology library did not fill its valid unique quota")
            write_json(specs_path, specs, immutable=True)
        if len(specs) != self.options.topology_count:
            raise RuntimeError("Persisted topology count mismatch")
        summary = [{key: spec.get(key) for key in
                    ("topology_id", "topology_graph_signature", "component_pair",
                     "placement_style", "shape_family", "interdigitated_finger_count",
                     "interdigitated_orientation_degrees", "maximum_representative_iou")}
                   for spec in specs]
        write_csv(folder / "topology_summary.csv", summary)
        return specs

    def _stage(self, number, specs, variants, excluded=()):
        folder = self.run_root / f"stage{number}"
        manifest = folder / "geometry_manifest.json"
        if manifest.exists():
            records = []
            for item in read_json(manifest):
                path = self.run_root / item["path"]
                if hashlib.sha256(path.with_suffix(".json").read_bytes()).hexdigest() != item["metadata_sha256"]:
                    raise RuntimeError(f"Persisted geometry metadata hash mismatch: {path}")
                records.append(load_record(path))
            if len(records) != len(specs) * variants:
                raise RuntimeError(f"Stage {number} manifest has wrong candidate count")
            return records
        if self.variant_generator is None:
            from .sampling import generate_variants
            self.variant_generator = generate_variants
        exclusions = {}
        old_hashes = {r["metadata"]["geometry_hash"] for r in excluded}
        for record in excluded:
            meta = record["metadata"]
            exclusions.setdefault(meta["topology_id"], []).append(meta["parameter_vector"])
        rejected = RejectionLog(folder / "generation_rejections.csv")
        records = []
        for spec in specs:
            topology_id = spec["topology_id"]
            print(f"[DOE] stage{number} {topology_id}: generating {variants} valid variants", flush=True)
            candidates = self.variant_generator(spec, self.limits, variants,
                derived_seed(self.seed, f"stage{number}", topology_id), self.physics_grid_size,
                self.geometry_options, excluded_parameters=exclusions.get(topology_id, ()),
                rejection_callback=rejected)
            if len(candidates) != variants:
                raise RuntimeError(f"{topology_id}: valid variant quota incomplete")
            for index, candidate in enumerate(candidates):
                geometry_id = f"S{number}_{topology_id}_V{index:03d}"
                record = save_record(folder / "geometries" / geometry_id, candidate, geometry_id, number)
                if record["metadata"]["geometry_hash"] in old_hashes:
                    raise RuntimeError(f"Repeated geometry across DOE stages: {geometry_id}")
                old_hashes.add(record["metadata"]["geometry_hash"])
                records.append(record)
        paths = [{"path": str(Path(r["artifact_path"]).relative_to(self.run_root)),
                  "metadata_sha256": hashlib.sha256(Path(r["artifact_path"]).with_suffix(".json").read_bytes()).hexdigest()}
                 for r in records]
        write_json(manifest, paths, immutable=True)
        return records

    def _stage_reports(self, number, records, specs):
        folder = self.run_root / f"stage{number}"
        write_csv(folder / "parameter_samples.csv", [{k: r["metadata"].get(k) for k in
            ("geometry_id", "topology_id", "topology_graph_signature", "component_pair",
             "placement_style", "shape_family", "interdigitation", "parameter_vector",
             "generator_seed", "geometry_hash")} for r in records])
        contact_sheet([Path(r["artifact_path"]).with_suffix(".png") for r in records], folder / "contact_sheet.png")
        if number == 1:
            write_csv(folder / "topologies.csv", specs)
            first = {}
            for r in records:
                first.setdefault(r["metadata"]["topology_id"], Path(r["artifact_path"]).with_suffix(".png"))
            # Representatives used for the IoU check are persisted by the geometry
            # library when available; otherwise instantiate those exact parameters.
            representative_paths = []
            for spec in specs:
                params = spec.get("representative_parameters")
                if params is None:
                    representative_paths.append(first[spec["topology_id"]])
                    continue
                from .geometry import realize
                candidate = realize(spec, params, self.limits, self.physics_grid_size, self.geometry_options)
                path = self.run_root / "topology_library" / "representatives" / spec["topology_id"]
                save_record(path, candidate, spec["topology_id"], 0)
                representative_paths.append(path.with_suffix(".png"))
            contact_sheet(representative_paths, self.run_root / "topology_library" / "topology_contact_sheet.png")

    def evaluate_stage(self, number, records, *, retry_failed=False):
        folder = self.run_root / f"stage{number}"
        rows, failed = [], []
        for index, record in enumerate(records):
            metadata = record["metadata"]
            output = folder / "physics" / metadata["geometry_id"]
            checkpoint = output / "evaluation.json"
            key = {"geometry_hash": metadata["geometry_hash"], "physics_mask_hash": metadata["physics_mask_hash"],
                   "config_hash": self.config_hash}
            result = read_json(checkpoint) if checkpoint.exists() else None
            if result is not None and any(result.get(k) != v for k, v in key.items()):
                raise RuntimeError(f"Cached physics identity mismatch: {checkpoint}")
            if result is not None and result["status"] == "running":
                result = None
            if result is None or (retry_failed and result["status"] != "success"):
                output.mkdir(parents=True, exist_ok=True)
                write_json(checkpoint, {**key, "status": "running", "geometry_id": metadata["geometry_id"]})
                print(f"[DOE] stage{number} physics {index + 1}/{len(records)} {metadata['geometry_id']}", flush=True)
                try:
                    metrics = self._physics().evaluate(record["anode_mask"], record["cathode_mask"], metadata, output)
                    if not metrics.get("physics_success", True) or metrics.get("doe_evaluation_status", "success") != "success":
                        raise PhysicsResultFailure(metrics)
                    if not all(np.isfinite(float(metrics[name])) for name in OBJECTIVES):
                        raise PhysicsResultFailure(metrics, "nonfinite_objective")
                    result = {**key, "geometry_id": metadata["geometry_id"], "status": "success", "metrics": metrics}
                except Exception as exc:
                    result = {**key, "geometry_id": metadata["geometry_id"], "status": "failed",
                              "error": str(exc), "exception_type": type(exc).__name__,
                              "metrics": getattr(exc, "metrics", {}), "traceback": traceback.format_exc()}
                write_json(checkpoint, result)
            record["metrics"] = result.get("metrics", {})
            row = {k: metadata.get(k) for k in
                   ("geometry_id", "topology_id", "topology_graph_signature", "component_pair",
                    "placement_style", "shape_family", "interdigitation", "parameter_vector",
                    "generator_seed", "geometry_hash")}
            row.update(config_hash=self.config_hash, status=result["status"], error=result.get("error", ""))
            row.update({name: result.get("metrics", {}).get(name) for name in OBJECTIVES})
            rows.append(row)
            write_csv(folder / "preflame_metrics.csv", rows)
            if result["status"] != "success":
                failed.append(metadata["geometry_id"])
        if failed:
            raise RuntimeError(f"Stage {number}: {len(failed)} failed physics results; inspect evaluation.json and resume with --retry-failed. No selection performed.")
        ranked = rank_designs(rows)
        write_csv(folder / "preflame_metrics.csv", ranked)
        return ranked

    def run(self, *, resume=False, execute_physics=False, postflame=False, retry_failed=False):
        if postflame and not execute_physics:
            raise ValueError("Post-flame requires execute_physics")
        with run_lock(self.run_root):
            self._initialise(resume)
            specs = self._library()
            stage1 = self._stage(1, specs, self.options.stage1_variants)
            self._stage_reports(1, stage1, specs)
            if not execute_physics:
                return {"phase": "stage1_geometry", "candidate_count": len(stage1)}
            rows1 = self.evaluate_stage(1, stage1, retry_failed=retry_failed)
            topologies = select_topologies(rows1, self.options.selected_topologies)
            write_csv(self.run_root / "stage1/top30_topologies.csv", topologies, immutable=True)
            spec_by_id = {spec["topology_id"]: spec for spec in specs}
            stage2_specs = [spec_by_id[row["topology_id"]] for row in topologies]
            stage2 = self._stage(2, stage2_specs, self.options.stage2_variants, excluded=stage1)
            self._stage_reports(2, stage2, stage2_specs)
            rows2 = self.evaluate_stage(2, stage2, retry_failed=retry_failed)
            all_rows = rank_designs(rows1 + rows2)
            write_csv(self.run_root / "all_600_preflame_metrics.csv", all_rows)
            final_rows = select_final(all_rows, self.options.final_count)
            final = self.run_root / "final"
            write_csv(final / "preflame_top20.csv", final_rows, immutable=True)
            all_records = {r["metadata"]["geometry_id"]: r for r in stage1 + stage2}
            selected = [all_records[row["geometry_id"]] for row in final_rows]
            for record in selected:
                for suffix in (".png", ".json", ".npz"):
                    source = Path(record["artifact_path"]).with_suffix(suffix)
                    atomic_bytes(final / "preflame_top20" / source.name, source.read_bytes(), immutable=True)
            adapter = self._physics()
            baseline = adapter.evaluate_baseline(final)
            adapter.write_preflame_comparison(selected, baseline, final)
            if postflame:
                adapter.refine_selected(selected, baseline, final)
            return {"phase": "postflame" if postflame else "preflame_complete",
                    "candidate_count": len(all_rows), "selected_count": len(selected),
                    "baseline_count": 1, "run_root": str(self.run_root)}


class PhysicsResultFailure(RuntimeError):
    def __init__(self, metrics, reason=None):
        self.metrics = metrics
        super().__init__(reason or metrics.get("doe_failure_reason", "production_physics_failed"))
