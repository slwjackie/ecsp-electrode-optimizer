#!/usr/bin/env python3
"""Eight fixed full-height strip cases, independent of the 148-topology workflow.

Only the geometry contract is specialized. Native B/C integration, voltage
search, numerical validity, and field export are inherited without changes.
No legacy files, factory registrations, or module globals are patched.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import yaml

SCHEMA = "full_height_strip_doe_v1"
BASE_COMMIT = "563ffbff7ee1cfbf46a81ff946aac69383b4a461"
UPSTREAM_BLOBS = {
    "python/ecsp_nsga2/evaluator.py": "eb10af045493006097a6c54b2666d96e9d8bac7b",
    "python/ecsp_nsga2/bc_native.py": "3af0b82bd2d0a0e90b36623844157202a28d4718",
}
# id: (anode width, cathode width, EDGE gap, analysis groups)
CASES = {
    "SP_G1": (2, 2, 1, ("spacing",)),
    "BASE_G2_W2": (2, 2, 2, ("spacing", "width")),
    "SP_G4": (2, 2, 4, ("spacing",)),
    "WD_W1": (1, 1, 2, ("width",)),
    "WD_W4": (4, 4, 2, ("width",)),
    "AR_1TO2": (2, 4, 2, ("area_ratio",)),
    "AR_1TO1": (3, 3, 2, ("area_ratio",)),
    "AR_2TO1": (4, 2, 2, ("area_ratio",)),
}


def clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8",
                                     delete=False) as f:
        temporary = Path(f.name)
        json.dump(clean(value), f, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8",
                                     newline="", delete=False) as f:
        temporary = Path(f.name)
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(clean(row))
    os.replace(temporary, path)


def integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class Study:
    grid_size: int = 200
    replicates: int = 5
    seed: int = 20261005
    batch_size: int = 4
    save_representative_fields: bool = True
    reference_voltage_V: float = 260.0
    end_time_s: float = 2.0

    def __post_init__(self):
        n = integer(self.grid_size, "grid_size", 100)
        if n % 50:
            raise ValueError("Cell-centred grid_size must be a multiple of 50; "
                             "200 gives dx=25/200=0.125 mm and exact half-mm boundaries")
        integer(self.replicates, "replicates", 5)
        integer(self.seed, "seed", 0)
        integer(self.batch_size, "batch_size", 1)
        if type(self.save_representative_fields) is not bool:
            raise ValueError("save_representative_fields must be boolean")
        for name in ("reference_voltage_V", "end_time_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @classmethod
    def load(cls, path: Path) -> "Study":
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or set(document) != {"study"}:
            raise ValueError("Expected exactly one study block")
        values = dict(document["study"])
        for name in ("domain_mm", "electrode_length_mm"):
            if values.pop(name, None) != 25.0:
                raise ValueError(f"{name} must be exactly 25 mm (not thickness)")
        if values.pop("schema", None) != SCHEMA:
            raise ValueError("Unknown study schema")
        return cls(**values)

    def record(self) -> dict:
        return dict(schema=SCHEMA, domain_mm=25.0, electrode_length_mm=25.0,
                    **self.__dict__, cases=CASES)


def make_case(case_id: str, study: Study) -> tuple[np.ndarray, np.ndarray, dict]:
    wa, wc, gap, groups = CASES[case_id]
    n, L = study.grid_size, 25.0
    h = L / n
    x0 = (L - wa - gap - wc) / 2.0
    edges = [x0, x0 + wa, x0 + wa + gap, x0 + wa + gap + wc]
    columns = [round(x / h) for x in edges]
    if any(abs(x / h - i) > 1e-10 for x, i in zip(edges, columns)):
        raise ValueError("Requested geometry is not exactly cell aligned")
    a, c = np.zeros((n, n), bool), np.zeros((n, n), bool)
    a[:, columns[0]:columns[1]] = True
    c[:, columns[2]:columns[3]] = True
    metadata = dict(
        geometry_id=case_id, geometry_schema=SCHEMA,
        source_role="electrode_parameter_study", study_groups=list(groups),
        domain_mm=L, grid_size=n, cell_size_mm=h,
        electrode_length_mm=L, anode_width_mm=wa, cathode_width_mm=wc,
        gap_mm=gap, anode_area_mm2=wa * L, cathode_area_mm2=wc * L,
        total_electrode_area_mm2=(wa + wc) * L,
        anode_to_cathode_area_ratio=wa / wc,
        target_anode_area_fraction=wa / L, target_cathode_area_fraction=wc / L,
        electrode_area_fraction=(wa + wc) / L,
        intended_anode_components=1, intended_cathode_components=1,
        surface_contact_model=True, hidden_bus_assumed=True,
        electrode_masks_remove_propellant=False, propellant_domain_area_fraction=1.0,
        electrode_bounds_mm=[edges[:2], edges[2:]],
        assembly_alignment="centred_outer_bounds", y_bounds_mm=[0.0, L],
    )
    return a, c, metadata


def validate_item(a: np.ndarray, c: np.ndarray, metadata: Mapping,
                  study: Study) -> dict:
    sid = metadata.get("geometry_id")
    if sid not in CASES:
        raise ValueError("Not one of the eight authorized strip cases")
    expected_a, expected_c, expected = make_case(sid, study)
    for key, value in expected.items():
        if clean(metadata.get(key)) != clean(value):
            raise ValueError(f"{sid}: metadata mismatch: {key}")
    for label, actual, nominal in (("anode", a, expected_a), ("cathode", c, expected_c)):
        if not isinstance(actual, np.ndarray) or actual.dtype != np.bool_:
            raise ValueError(f"{sid}: {label} must be a boolean mask")
        if actual.shape != nominal.shape or not np.array_equal(actual, nominal):
            raise ValueError(f"{sid}: {label} differs from the exact physical rectangle")
    if metadata.get("baseline_type") or metadata.get("allow_hidden_bus_reference_component_override"):
        raise ValueError("Reference metadata is not accepted in this workflow")
    h = 25.0 / study.grid_size
    measured_a, measured_c = float(a.sum() * h * h), float(c.sum() * h * h)
    if not (math.isclose(measured_a, expected["anode_area_mm2"], abs_tol=1e-10)
            and math.isclose(measured_c, expected["cathode_area_mm2"], abs_tol=1e-10)):
        raise ValueError("Raster/CAD area mismatch")
    return dict(geometry_id=sid, passed=True, grid_size=study.grid_size,
                cell_size_mm=h, anode_area_mm2=measured_a, cathode_area_mm2=measured_c,
                gap_mm=expected["gap_mm"], full_height=True, full_propellant=True)


def generate(study: Study, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists() or (out / "library").exists():
        raise ValueError("Geometry output already exists; use audit/run --resume")
    records = []
    for sid in CASES:
        a, c, meta = make_case(sid, study)
        folder = out / "library" / sid
        folder.mkdir(parents=True)
        np.savez_compressed(folder / "mask.npz", anode=a, cathode=c,
                            propellant=np.ones_like(a), domain_mm=25.0)
        write_json(folder / "metadata.json", meta)
        records.append(dict(geometry_id=sid, mask_sha256=file_sha(folder / "mask.npz"),
                            metadata_sha256=file_sha(folder / "metadata.json")))
    manifest = dict(schema=SCHEMA, study=study.record(), study_hash=digest(study.record()),
                    cases=records, case_count=8, pde_executed=False)
    write_json(out / "manifest.json", manifest)
    experiment = [dict(case_id=sid, replicate=r, valid="", ignited="",
                       ignition_delay_s="", observation_window_s=study.end_time_s,
                       notes="") for r in range(1, study.replicates + 1) for sid in CASES]
    columns = ["case_id", "replicate", "valid", "ignited", "ignition_delay_s",
               "observation_window_s", "notes"]
    write_csv(out / "experiment_template.csv", experiment, columns)
    rng = np.random.default_rng(study.seed)
    schedule = []
    for r in range(1, study.replicates + 1):
        for sid in rng.permutation(list(CASES)):
            schedule.append(dict(run_order=len(schedule)+1, block=r, case_id=str(sid), replicate=r))
    write_csv(out / "experiment_schedule.csv", schedule,
              ["run_order", "block", "case_id", "replicate"])
    audit(study, out)
    return manifest


def load_library(study: Study, out: Path) -> list[tuple]:
    manifest = read_json(out / "manifest.json")
    if manifest.get("study_hash") != digest(study.record()):
        raise ValueError("Study differs from saved geometry; use a new output directory")
    records = manifest["cases"]
    if [r["geometry_id"] for r in records] != list(CASES) or len(records) != 8:
        raise ValueError("Geometry library must contain the exact eight IDs")
    items = []
    for record in records:
        sid = record["geometry_id"]
        folder = out / "library" / sid
        for filename, key in (("mask.npz", "mask_sha256"), ("metadata.json", "metadata_sha256")):
            if file_sha(folder / filename) != record[key]:
                raise ValueError(f"{sid}: changed {filename}")
        meta = read_json(folder / "metadata.json")
        with np.load(folder / "mask.npz", allow_pickle=False) as data:
            a, c, p = data["anode"].copy(), data["cathode"].copy(), data["propellant"]
            if p.dtype != np.bool_ or p.shape != a.shape or not p.all() or float(data["domain_mm"]) != 25.0:
                raise ValueError("Full-propellant overlay contract violated")
        validate_item(a, c, meta, study)
        items.append((a, c, meta, out / "cases" / sid))
    return items


def audit(study: Study, out: Path) -> dict:
    rows = [validate_item(a, c, m, study) for a, c, m, _ in load_library(study, out)]
    result = dict(case_count=len(rows), all_passed=True, pde_executed=False, cases=rows)
    write_json(out / "geometry_audit.json", result)
    return result


def verify_upstream(root: Path) -> None:
    for relative, expected in UPSTREAM_BLOBS.items():
        data = (root / relative).read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if actual != expected:
            raise ValueError(f"Unsupported upstream revision: {relative}. "
                             "Do not reset your changes; review adapter compatibility first.")


def source_hashes(root: Path) -> dict:
    prefixes = ("cpp", "config", "data/image_design", "python/ecsp_nsga2",
                "python/ecsp_v6", "python/ecsp_native", "python/ecsp_image_design",
                "python/ecsp_reactive", "python/ecsp_cpp", "python/ecsp_cuda")
    suffixes = {".py", ".cpp", ".cc", ".cu", ".cuh", ".h", ".hpp", ".yaml", ".yml", ".json", ".csv", ".npz"}
    return {p.relative_to(root).as_posix(): file_sha(p)
            for prefix in prefixes for p in sorted((root / prefix).rglob("*"))
            if p.is_file() and not p.is_symlink() and p.suffix in suffixes
            and "__pycache__" not in p.parts}


def runtime_config(base: Mapping, study: Study, device: str) -> dict:
    if device not in {"cpu", "cuda"}:
        raise ValueError("Use an explicit cpu or cuda device")
    cfg = copy.deepcopy(dict(base))
    expected = [(cfg["physics"]["voltage_V"], study.reference_voltage_V),
                (cfg["evaluator"]["voltage_V"], study.reference_voltage_V),
                (cfg["minimum_ignition_voltage_search"]["upper_bound_V"], study.reference_voltage_V),
                (cfg["physics"]["end_time_s"], study.end_time_s),
                (cfg["physics"]["metric_evaluation_time_s"], study.end_time_s),
                (cfg["evaluator"]["end_time_s"], study.end_time_s),
                (cfg["bc_global"]["endTime_s"], study.end_time_s),
                (cfg["bc_global"]["evaluationTime_s"], study.end_time_s)]
    if any(not math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12) for a, b in expected):
        raise ValueError("Study voltage/time must match all base physics declarations; "
                         "no silent physics overrides are allowed")
    if cfg["minimum_ignition_voltage_search"]["enabled"] is not True:
        raise ValueError("The production B/C evaluator requires its Vmin search")
    cfg["geometry"].update(domain_mm=25.0, grid_size=study.grid_size,
        minimum_gap_mm=1.0, minimum_width_mm=1.0, maximum_width_mm=4.0,
        maximum_components_per_polarity=1, maximum_total_components=2,
        surface_contact_model="overlay_on_full_propellant_domain")
    ev = cfg["evaluator"]
    ev.update(backend="bc_global_native", device=device, grid_size=study.grid_size,
              internal_batch_size=study.batch_size,
              save_representative_fields=study.save_representative_fields)
    ev.setdefault("native", {})["cpu_workers"] = 0
    ev.setdefault("base_overrides", {}).setdefault("numerics", {})["physicsDevice"] = device
    # Existing optimization and reference settings are NOT invoked by this runner.
    for key in cfg:
        if key not in {"geometry", "evaluator"} and cfg[key] != base[key]:
            raise AssertionError(f"Protected physics configuration changed: {key}")
    return cfg


class FullHeightStripGeometryMixin:
    """Replace only the equal-area library geometry policy, NOT the physics.

    Exact reconstruction validates position, area, width, gap and connectivity
    more strictly than a tolerance gate. The original low-level tensor builder
    retains its overlap/contact/gap checks. No dummy masks reach the solver.
    """
    def _build_geometry_batch(self, items):
        import torch
        from ecsp_nsga2.evaluator import DirectCondensedV772NoFEvaluator
        study = Study(grid_size=self.grid_size)
        if not math.isclose(self.domain_size_m, 0.025, rel_tol=0, abs_tol=1e-14):
            raise ValueError("The strip adapter requires a 25 mm domain")
        for a, c, metadata, _ in items:
            validate_item(a, c, metadata, study)
        geometry = DirectCondensedV772NoFEvaluator._build_geometry_batch(self, items)
        if list(geometry.geometry_ids) != [item[2]["geometry_id"] for item in items]:
            raise ValueError("Geometry ID alignment changed")
        for i, (a, c, _, _) in enumerate(items):
            if not (np.array_equal(geometry.anode[i].detach().cpu().numpy(), a)
                    and np.array_equal(geometry.cathode[i].detach().cpu().numpy(), c)):
                raise ValueError("Production tensor builder resized or changed a DOE mask")
        # Identical surface-overlay semantics to BCGlobalPreflameEvaluator.
        geometry.fixed = torch.zeros_like(geometry.anode)
        geometry.propellant = torch.ones_like(geometry.anode)
        return geometry


def create_study_evaluator(root: Path, cfg: dict, folder: Path):
    verify_upstream(root)
    from ecsp_nsga2.bc_native import NativeBCGlobalEvaluator

    class StripDOEEvaluator(FullHeightStripGeometryMixin, NativeBCGlobalEvaluator):
        pass

    adapter = copy.deepcopy(cfg["evaluator"])
    adapter["physics_config"] = copy.deepcopy(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    return StripDOEEvaluator(root, adapter, folder)


def classify(row: Mapping, valid: bool) -> str:
    if row.get("physicsRejected") or not valid:
        return "numerically_invalid"
    if row.get("ignitionSucceeded") is True:
        return "condensed_onset_reached"
    return "no_condensed_onset_within_horizon"


METRICS = ["ignitionSucceeded", "ignitionDelay_s", "minimumIgnitionVoltage_V",
    "minimumIgnitionVoltageSearchValid", "minimumIgnitionVoltageSearchStatus",
    "minimumIgnitionVoltageLeftCensored", "minimumIgnitionVoltageRightCensored",
    "peakCurrentCongestionToEvaluationTime", "inputElectricalEnergyToIgnition_J",
    "inputElectricalEnergyAtEvaluationTime_J", "peakMaximumTemperature_K", "peakCurrent_A",
    "evaluationStateTime_s", "objectiveEvaluationStateTime_s", "postOnsetContinuationApplied",
    "areaAveragedUndecomposedFractionAtEvaluationTime", "converged",
    "allElectricalLinearSolvesConverged", "allNonlinearRobinSolvesConverged", "physicsRejected"]


def summarize_simulation(out: Path) -> list[dict]:
    rows = []
    for sid in CASES:
        path = out / "cases" / sid / "result.json"
        if not path.exists():
            continue
        result = read_json(path)
        meta = result["geometry"]
        row = {k: meta[k] for k in ("geometry_id", "anode_width_mm", "cathode_width_mm",
            "gap_mm", "electrode_length_mm", "anode_area_mm2", "cathode_area_mm2",
            "total_electrode_area_mm2", "anode_to_cathode_area_ratio")}
        row.update(study_groups=";".join(meta["study_groups"]), status=result["status"],
                   numerical_valid=result["numerical_valid"], reason=result.get("reason", ""))
        raw = result.get("raw", {})
        row.update({name: raw.get(name) for name in METRICS})
        # Optimizer penalty values remain in raw JSON, never published as observations.
        if not result["numerical_valid"]:
            for name in METRICS:
                if name not in {"converged", "allElectricalLinearSolvesConverged",
                                "allNonlinearRobinSolvesConverged", "physicsRejected"}:
                    row[name] = None
        elif row["ignitionSucceeded"] is not True:
            row["ignitionDelay_s"] = None
            row["inputElectricalEnergyToIgnition_J"] = None
        if raw.get("minimumIgnitionVoltageSearchValid") is not True:
            row["minimumIgnitionVoltage_V"] = None
        rows.append(row)
    columns = list(rows[0]) if rows else ["geometry_id", "status"]
    write_csv(out / "parameter_study_summary.csv", rows, columns)
    for group in ("spacing", "width", "area_ratio"):
        write_csv(out / f"{group}_summary.csv",
                  [r for r in rows if group in r["study_groups"].split(";")], columns)
    return rows


def evaluate(study: Study, root: Path, physics_path: Path, out: Path, *,
             device: str, only: list[str] | None = None, resume: bool = False,
             retry_failed: bool = False) -> dict:
    import fcntl  # Linux/A100 and macOS; automatically released after process death.
    root, out = root.resolve(), out.resolve()
    if out == root or any(out.is_relative_to(root / p) for p in ("python", "config", "cpp", "data", "tools", ".git")):
        raise ValueError("Output must not be placed in a source directory; use runs/...")
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another process is using this output directory") from exc
        if retry_failed and not resume:
            raise ValueError("--retry-failed requires --resume")
        ids = list(CASES) if only is None else list(only)
        if not ids or len(ids) != len(set(ids)) or set(ids) - set(CASES):
            raise ValueError("--only must contain distinct known case IDs")
        if not (out / "manifest.json").exists():
            generate(study, out)
        audit(study, out)
        items = load_library(study, out)
        base = yaml.safe_load(physics_path.read_text(encoding="utf-8"))
        cfg = runtime_config(base, study, device)
        before = source_hashes(root)
        signature = digest(dict(study=study.record(), config=cfg, sources=before,
                                runner=file_sha(Path(__file__).resolve()), device=device))
        run_path = out / "run_manifest.json"
        if run_path.exists():
            old = read_json(run_path)
            if not resume or old.get("input_signature") != signature:
                raise ValueError("Existing run or changed inputs: use identical --resume inputs or a new directory")
        else:
            write_json(run_path, dict(input_signature=signature, source_base_commit=BASE_COMMIT,
                resolved_request=cfg, original_source_hashes=before,
                geometry_mode=SCHEMA, comparison_mode="raw_within_study_only"))
        pending = []
        for item in items:
            sid, folder = item[2]["geometry_id"], item[3]
            if sid not in ids:
                continue
            saved_path = folder / "result.json"
            if saved_path.exists():
                saved = read_json(saved_path)
                if saved.get("input_signature") != signature:
                    raise ValueError(f"{sid}: changed result provenance")
                if not (retry_failed and saved["status"] == "execution_failed"):
                    continue
            pending.append(item)
        evaluator = None
        started = time.perf_counter()
        try:
            if pending:
                evaluator = create_study_evaluator(root, cfg, out / "adapter")
                if (evaluator.grid_size != study.grid_size
                        or not math.isclose(evaluator.domain_size_m, .025, rel_tol=0, abs_tol=1e-14)
                        or str(evaluator.device).split(":")[0] != device
                        or not math.isclose(evaluator.voltage, study.reference_voltage_V)):
                    raise ValueError("Resolved evaluator geometry/device/voltage mismatch")
                write_json(out / "resolved_physics_config.json", evaluator.config)
                for start in range(0, len(pending), study.batch_size):
                    chunk = pending[start:start + study.batch_size]
                    try:
                        raw_rows = evaluator.evaluate_batch(chunk)
                        if len(raw_rows) != len(chunk):
                            raise RuntimeError("Evaluator returned the wrong number of cases")
                        if [r.get("geometry_id") for r in raw_rows] != [i[2]["geometry_id"] for i in chunk]:
                            raise RuntimeError("Evaluator lost case alignment")
                        for item, raw in zip(chunk, raw_rows):
                            valid, reason = evaluator._trial_valid(raw)
                            if raw.get("physicsRejected", False):
                                reason = "physics_rejected"
                            elif not (raw.get("allElectricalLinearSolvesConverged") is True
                                      and raw.get("allNonlinearRobinSolvesConverged") is True):
                                reason = "cumulative_solver_convergence_not_confirmed"
                            valid = bool(valid and not raw.get("physicsRejected", False)
                                and raw.get("allElectricalLinearSolvesConverged") is True
                                and raw.get("allNonlinearRobinSolvesConverged") is True)
                            result = dict(input_signature=signature, geometry=item[2], raw=raw,
                                status=classify(raw, valid), numerical_valid=valid, reason=str(reason))
                            write_json(item[3] / "result.json", result)
                    except Exception as exc:
                        for item in chunk:
                            write_json(item[3] / "result.json", dict(input_signature=signature,
                                geometry=item[2], raw={}, status="execution_failed", numerical_valid=False,
                                reason=str(exc), exception_type=type(exc).__name__, traceback=traceback.format_exc()))
                        summarize_simulation(out)
                        raise
                    summarize_simulation(out)
        finally:
            if evaluator is not None and hasattr(evaluator, "close"):
                evaluator.close()
            after = source_hashes(root)
            write_json(out / "source_preservation.json", dict(unchanged=before == after,
                before=before, after=after))
            if before != after:
                raise RuntimeError("Existing sources changed during the study")
        rows = summarize_simulation(out)
        report = dict(requested_ids=ids, available_results=len(rows),
                      elapsed_s=time.perf_counter()-started, original_sources_unchanged=True,
                      statuses={r["geometry_id"]: r["status"] for r in rows})
        write_json(out / "RUN_FINISHED.json", report)
        return report


def summarize_experiment(source: Path, destination: Path) -> list[dict]:
    """Unsuccessful ignitions are censored, never substituted by the time limit."""
    with source.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    seen, groups = set(), {sid: [] for sid in CASES}
    def flag(value: str) -> bool:
        value = value.strip().lower()
        if value in {"true", "1"}: return True
        if value in {"false", "0"}: return False
        raise ValueError("Use true/false or 1/0 for completed experiment flags")
    for row in rows:
        sid = row["case_id"]
        if sid not in CASES: raise ValueError("Unknown experiment case")
        rep = int(row["replicate"])
        if rep < 1 or (sid, rep) in seen: raise ValueError("Duplicate or invalid replicate")
        seen.add((sid, rep))
        if not row["valid"].strip():
            if row["ignited"].strip() or row["ignition_delay_s"].strip():
                raise ValueError("An entered observation requires an explicit validity flag")
            continue
        valid = flag(row["valid"])
        if not valid:
            groups[sid].append((False, False, None))
            continue
        ignited = flag(row["ignited"])
        horizon = float(row["observation_window_s"])
        if not math.isfinite(horizon) or horizon <= 0: raise ValueError("Invalid observation window")
        delay = float(row["ignition_delay_s"]) if ignited else None
        if ignited and (not math.isfinite(delay) or delay <= 0 or delay > horizon):
            raise ValueError("Ignition delay must be positive and within the observation window")
        if not ignited and row["ignition_delay_s"].strip():
            raise ValueError("Do not replace a nonignition by the time limit")
        groups[sid].append((True, ignited, delay))
    result = []
    for sid, records in groups.items():
        valid = [r for r in records if r[0]]
        delays = [r[2] for r in valid if r[1]]
        result.append(dict(case_id=sid, n_completed=len(records), n_valid=len(valid),
            n_excluded=len(records)-len(valid), n_ignited=len(delays),
            n_right_censored=len(valid)-len(delays),
            ignition_fraction=len(delays)/len(valid) if valid else None,
            mean_delay_among_ignited_s=float(np.mean(delays)) if delays else None,
            sample_sd_among_ignited_s=float(np.std(delays, ddof=1)) if len(delays)>1 else None))
    write_csv(destination, result, list(result[0]))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["generate", "audit", "run", "summarize-experiment"])
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--study", type=Path, default=Path("studies/electrode_parameter_study_25mm.yaml"))
    parser.add_argument("--config", type=Path, default=Path("config/nsga2_bc_reactive_a100_cpu8_poweroff_200x3.yaml"))
    parser.add_argument("--out", type=Path, default=Path("runs/electrode_parameter_study_25mm"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    def resolve(path): return path.resolve() if path.is_absolute() else root / path
    try:
        if args.action == "summarize-experiment":
            if args.input is None or args.output is None:
                raise ValueError("--input and --output are required")
            result = summarize_experiment(resolve(args.input), resolve(args.output))
        else:
            study, out = Study.load(resolve(args.study)), resolve(args.out)
            if args.action == "generate": result = generate(study, out)
            elif args.action == "audit": result = audit(study, out)
            else:
                result = evaluate(study, root, resolve(args.config), out,
                    device=args.device, only=args.only, resume=args.resume, retry_failed=args.retry_failed)
        print(json.dumps(clean(result), indent=2, allow_nan=False))
        if args.action == "run" and any(v in {"execution_failed", "numerically_invalid"}
                                       for v in result["statuses"].values()):
            return 2
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        print(f"[strip-doe] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
