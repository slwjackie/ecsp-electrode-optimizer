#!/usr/bin/env python3
"""Independent eight-case, full-height strip study. All legacy files stay unchanged.

Only geometry-batch construction is specialized. The native B/C integration,
reference metrics, numerical validity checks and voltage search are inherited.
No topology fitting, optimization, staggered reference or post-onset runner is
called. The inherited reference calculation may include its existing fixed-time
condensed continuation; that is not a gas-flame propagation calculation.
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
import statistics
import sys
import tempfile
import traceback
from typing import Any, Mapping

import numpy as np
import yaml

BASE_REVISION = "563ffbff7ee1cfbf46a81ff946aac69383b4a461"
SCHEMA = "full_height_strip_study_v1"
# id: (anode width, cathode width, edge-to-edge gap, analysis groups)
CASES = {
    "SP_G1": (2.0, 2.0, 1.0, ("spacing",)),
    "BASE_G2_W2": (2.0, 2.0, 2.0, ("spacing", "width")),
    "SP_G4": (2.0, 2.0, 4.0, ("spacing",)),
    "WD_W1": (1.0, 1.0, 2.0, ("width",)),
    "WD_W4": (4.0, 4.0, 2.0, ("width",)),
    "AR_1TO2": (2.0, 4.0, 2.0, ("area_ratio",)),
    "AR_1TO1": (3.0, 3.0, 2.0, ("area_ratio",)),
    "AR_2TO1": (4.0, 2.0, 2.0, ("area_ratio",)),
}
PHYSICS_DEFAULT = "config/nsga2_bc_reactive_a100_cpu8_poweroff_200x3.yaml"


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


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def validate_spec(spec: Mapping[str, Any]) -> dict:
    """A deliberately fixed study; reject silent changes of its physical design."""
    expected_keys = {"schema", "domain_mm", "electrode_length_mm", "grid_size",
                     "placement", "experimental_replicates", "cases"}
    if set(spec) != expected_keys or spec["schema"] != SCHEMA:
        raise ValueError("Unknown study schema/keys")
    if spec["domain_mm"] != 25.0 or spec["electrode_length_mm"] != 25.0:
        raise ValueError("This study requires a 25 x 25 mm domain and 25 mm contact length")
    n = spec["grid_size"]
    # Cell-centred dx=L/N, NOT L/(N-1). Half-mm edges must lie on cell faces.
    if isinstance(n, bool) or not isinstance(n, int) or n < 100 or n % 50:
        raise ValueError("grid_size must be a multiple of 50, >=100 (recommended 200)")
    if spec["placement"] != "centered_assembly":
        raise ValueError("Only centered_assembly placement is supported")
    reps = spec["experimental_replicates"]
    if isinstance(reps, bool) or not isinstance(reps, int) or reps < 5:
        raise ValueError("experimental_replicates must be an integer >=5")
    rows = spec["cases"]
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("Exactly eight unique cases are required")
    found = {}
    for row in rows:
        if set(row) != {"id", "anode_width_mm", "cathode_width_mm", "gap_mm", "groups"}:
            raise ValueError("Invalid case keys")
        sid = row["id"]
        if sid not in CASES or sid in found:
            raise ValueError("Unknown/duplicate case ID")
        wa, wc, gap, groups = CASES[sid]
        if (row["anode_width_mm"], row["cathode_width_mm"], row["gap_mm"],
            tuple(row["groups"])) != (wa, wc, gap, groups):
            raise ValueError(f"{sid}: dimensions/groups differ from the requested study")
        found[sid] = row
    result = copy.deepcopy(dict(spec))
    result["cases"] = [found[sid] for sid in CASES]
    return result


def make_case(sid: str, n: int) -> tuple[np.ndarray, np.ndarray, dict]:
    wa, wc, gap, groups = CASES[sid]
    L = 25.0
    if isinstance(n, bool) or not isinstance(n, int) or n < 100 or n % 50:
        raise ValueError("Grid cannot represent every design edge exactly")
    h = L / n
    left = (L - wa - gap - wc) / 2
    edges = (left, left + wa, left + wa + gap, left + wa + gap + wc)
    indices = [round(x / h) for x in edges]
    if any(not math.isclose(i * h, x, abs_tol=1e-12, rel_tol=0)
           for i, x in zip(indices, edges)):
        raise ValueError("Off-grid contact edge; no rounding or refitting is permitted")
    a = np.zeros((n, n), dtype=bool)
    c = np.zeros_like(a)
    a[:, indices[0]:indices[1]] = True
    c[:, indices[2]:indices[3]] = True
    meta = dict(schema=SCHEMA, geometry_id=sid, source_role="independent_parameter_study",
                groups=list(groups), domain_mm=L, grid_size=n, grid_spacing_mm=h,
                grid_convention="cell_centred_L_over_N", electrode_length_mm=L,
                anode_width_mm=wa, cathode_width_mm=wc, gap_mm=gap,
                placement="centered_assembly", anode_bounds_mm=[edges[0], 0., edges[1], L],
                cathode_bounds_mm=[edges[2], 0., edges[3], L],
                anode_area_mm2=L * wa, cathode_area_mm2=L * wc,
                total_electrode_area_mm2=L * (wa + wc),
                anode_to_cathode_area_ratio=wa / wc,
                target_anode_area_fraction=wa / L, target_cathode_area_fraction=wc / L,
                actual_anode_area_mm2=float(a.sum() * h * h),
                actual_cathode_area_mm2=float(c.sum() * h * h),
                intended_anode_components=1, intended_cathode_components=1,
                surface_contact_model=True, hidden_bus_assumed=True,
                propellant_domain_area_mm2=L * L, electrode_masks_remove_propellant=False)
    return a, c, meta


def validate_item(a: np.ndarray, c: np.ndarray, meta: Mapping, n: int) -> None:
    """Exact manufacturing contract replaces equal-area topology constraints.

    Check dimensions, polarity assignment, area, gap, connectivity and full-height
    contact together by equality to the analytically rasterised registered case.
    Never repair masks, resize them, loosen an area tolerance or drop metadata.
    """
    sid = meta.get("geometry_id")
    if sid not in CASES:
        raise ValueError("Unregistered parameter-study case")
    ea, ec, em = make_case(sid, n)
    if any(meta.get(k) != v for k, v in em.items()):
        raise ValueError(f"{sid}: missing or inconsistent geometry metadata")
    if (a.dtype != np.bool_ or c.dtype != np.bool_ or
        not np.array_equal(a, ea) or not np.array_equal(c, ec)):
        raise ValueError(f"{sid}: mask differs from the exact full-height rectangle contract")


def geometry_svg(meta: Mapping) -> str:
    # An engineering vector drawing from exact coordinates, not a generated image.
    rects = []
    for polarity in ("anode", "cathode"):
        x0, y0, x1, y1 = meta[f"{polarity}_bounds_mm"]
        rects.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" '
                     f'fill="none" stroke="black" stroke-width="0.08"/>'
                     f'<text x="{(x0+x1)/2}" y="12.5" text-anchor="middle" '
                     f'font-size="0.9">{polarity[0].upper()}</text>')
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="25mm" height="25mm" '
            'viewBox="0 0 25 25"><rect width="25" height="25" fill="white" '
            'stroke="black" stroke-width="0.08"/>' + ''.join(rects) + '</svg>\n')


def generate(spec: Mapping, out: Path) -> None:
    spec = validate_spec(spec)
    if out.exists():
        raise FileExistsError("Output already exists; use a new path. No files were overwritten")
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=out.name + ".preparing-", dir=out.parent))
    records = []
    for sid in CASES:
        a, c, meta = make_case(sid, spec["grid_size"])
        folder = staging / "library" / sid
        folder.mkdir(parents=True)
        np.savez_compressed(folder / "mask.npz", anode=a, cathode=c,
                            propellant=np.ones_like(a), domain_mm=25.0)
        (folder / "geometry.svg").write_text(geometry_svg(meta), encoding="utf-8")
        meta["mask_file_sha256"] = file_hash(folder / "mask.npz")
        write_json(folder / "metadata.json", meta)
        records.append(meta)
    write_json(staging / "study.json", spec)
    write_json(staging / "geometry_manifest.json", records)
    template = [dict(case_id=sid, replicate=k, batch_id="", ignition_observed="",
                     ignition_delay_s="", observation_window_s="", notes="")
                for sid in CASES for k in range(1, spec["experimental_replicates"] + 1)]
    write_csv(staging / "experiment_measurements_template.csv", template)
    write_json(staging / "GENERATED.json", dict(schema=SCHEMA, cases=8, pde_executed=False))
    audit(staging)
    # Rename atomically; a concurrent creator is not allowed to have its files replaced.
    if out.exists():
        raise FileExistsError("Output was created concurrently; staging was retained")
    staging.rename(out)


def audit(out: Path) -> tuple[dict, list[tuple]]:
    spec = validate_spec(read_json(out / "study.json"))
    manifest = read_json(out / "geometry_manifest.json")
    if [m["geometry_id"] for m in manifest] != list(CASES):
        raise ValueError("Manifest IDs/order changed")
    if {p.name for p in (out / "library").iterdir() if p.is_dir()} != set(CASES):
        raise ValueError("Unexpected/missing geometry directory")
    items = []
    for record in manifest:
        folder = out / "library" / record["geometry_id"]
        meta = read_json(folder / "metadata.json")
        if meta != record or file_hash(folder / "mask.npz") != meta["mask_file_sha256"]:
            raise ValueError("Stored geometry checksum/metadata mismatch")
        with np.load(folder / "mask.npz", allow_pickle=False) as z:
            a, c = z["anode"].copy(), z["cathode"].copy()
            if (z["propellant"].shape != a.shape or z["propellant"].dtype != np.bool_
                or not z["propellant"].all() or float(z["domain_mm"]) != 25.0):
                raise ValueError("Full-propellant overlay contract violated")
        validate_item(a, c, meta, spec["grid_size"])
        items.append((a, c, meta, out / "cases" / meta["geometry_id"]))
    return spec, items


def runtime_config(base: Mapping, n: int, device: str, batch_size: int) -> dict:
    if device not in {"cpu", "cuda"} or batch_size < 1:
        raise ValueError("Use an explicit cpu/cuda device and a positive batch size")
    cfg = copy.deepcopy(dict(base))
    cfg.setdefault("geometry", {}).update(domain_mm=25., grid_size=n,
        minimum_gap_mm=1., minimum_width_mm=1., maximum_components_per_polarity=1,
        maximum_total_components=2, surface_contact_model="overlay_on_full_propellant_domain")
    adapter = cfg.setdefault("evaluator", {})
    adapter.update(backend="bc_global_native", device=device, grid_size=n,
                   internal_batch_size=batch_size, save_representative_fields=True)
    adapter.setdefault("native", {})["cpu_workers"] = 0
    numerics = adapter.setdefault("base_overrides", {}).setdefault("numerics", {})
    numerics["physicsDevice"] = device
    if str(numerics.get("physicsDtype", "float64")).lower() != "float64":
        raise ValueError("Use a production FP64 config; this runner does not change precision")
    # All other top-level blocks, including chemistry, numerics and onset/Vmin
    # semantics, remain verbatim. Their optimization/baseline flags are inert:
    # none of those orchestrators is instantiated by this independent runner.
    return cfg


def build_geometry(items, n: int, device, minimum_gap_m: float, geometry_type):
    """Make the same GeometryBatch type consumed by the original native engine."""
    import torch
    if not items:
        raise ValueError("Empty geometry batch")
    ids, anodes, cathodes = [], [], []
    for a, c, meta, _ in items:
        validate_item(a, c, meta, n)
        if meta["gap_mm"] / 1000 + 1e-12 < minimum_gap_m:
            raise ValueError("Case gap is below the evaluator's requested minimum")
        ids.append(meta["geometry_id"])
        anodes.append(a)
        cathodes.append(c)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case in one physics batch")
    at = torch.as_tensor(np.stack(anodes), dtype=torch.bool, device=device)
    ct = torch.as_tensor(np.stack(cathodes), dtype=torch.bool, device=device)
    return geometry_type(geometry_ids=ids, anode=at, cathode=ct,
                         fixed=torch.zeros_like(at), propellant=torch.ones_like(at),
                         grid_size=n, domain_size_m=0.025, minimum_gap_m=minimum_gap_m)


def create_study_evaluator(root: Path, cfg: Mapping, workdir: Path):
    """Additive subclass: override geometry only, never monkey-patch shared code."""
    from ecsp_nsga2.bc_native import NativeBCGlobalEvaluator
    from ecsp_nsga2.evaluator import BCCandidateGeometryError
    from ecsp_v6.physics.geometry import GeometryBatch

    class FullHeightStripEvaluator(NativeBCGlobalEvaluator):
        def _build_geometry_batch(self, items):
            if not math.isclose(self.domain_size_m, 0.025, abs_tol=1e-15, rel_tol=0):
                raise RuntimeError("Resolved physics domain is not 25 mm")
            try:
                return build_geometry(items, self.grid_size, self.device,
                                      self.minimum_gap_m, GeometryBatch)
            except (ValueError, KeyError, TypeError) as exc:
                raise BCCandidateGeometryError(str(exc)) from exc

    adapter = copy.deepcopy(cfg["evaluator"])
    adapter["physics_config"] = copy.deepcopy(dict(cfg))
    evaluator = FullHeightStripEvaluator(root, adapter, workdir)
    if evaluator.grid_size != cfg["geometry"]["grid_size"]:
        raise RuntimeError("Resolved grid differs from the exact raster")
    return evaluator


def source_fence(root: Path) -> dict[str, str]:
    """Hash source/config inputs, not compiled binaries or simulation outputs."""
    suffixes = {".py", ".cpp", ".cc", ".h", ".hpp", ".cu", ".cuh", ".yaml", ".yml", ".json", ".csv"}
    result = {}
    for prefix in ("python", "cpp", "config", "data/image_design"):
        for path in sorted((root / prefix).rglob("*")):
            if path.is_file() and path.suffix in suffixes and "__pycache__" not in path.parts:
                result[path.relative_to(root).as_posix()] = file_hash(path)
    if not result:
        raise ValueError("No source inputs found at project root")
    return result


SUMMARY_KEYS = ("ignitionSucceeded", "ignitionDelay_s", "minimumIgnitionVoltage_V",
    "minimumIgnitionVoltageSearchValid", "minimumIgnitionVoltageSearchStatus",
    "minimumIgnitionVoltageLeftCensored", "minimumIgnitionVoltageRightCensored",
    "peakCurrentCongestionToEvaluationTime", "inputElectricalEnergyToIgnition_J",
    "inputElectricalEnergyAtEvaluationTime_J", "peakMaximumTemperature_K",
    "areaAveragedUndecomposedFractionAtEvaluationTime", "converged",
    "allElectricalLinearSolvesConverged", "allNonlinearRobinSolvesConverged",
    "physicsRejected", "external_numerical_valid", "external_numerical_reason")


def summary_rows(records: list[dict]) -> list[dict]:
    rows = []
    for rec in records:
        m, raw = rec["geometry"], rec["raw"]
        good = raw.get("external_numerical_valid") is True
        state = ("numerically_invalid" if not good else
                 "condensed_onset" if raw.get("ignitionSucceeded") is True else
                 "no_onset_within_horizon")
        row = {k: m[k] for k in ("geometry_id", "anode_width_mm", "cathode_width_mm", "gap_mm",
            "anode_area_mm2", "cathode_area_mm2", "total_electrode_area_mm2",
            "anode_to_cathode_area_ratio", "grid_size", "grid_spacing_mm")}
        row.update(groups=";".join(m["groups"]), result_state=state)
        row.update({k: raw.get(k) for k in SUMMARY_KEYS})
        # Preserve every original raw field in result.json. Do not present
        # numerical-failure penalties or no-onset sentinels as a measured delay.
        if not good or raw.get("ignitionSucceeded") is not True:
            row["ignitionDelay_s"] = None
        if (not good or raw.get("minimumIgnitionVoltageSearchValid") is not True
            or raw.get("minimumIgnitionVoltageRightCensored") is True):
            row["minimumIgnitionVoltage_V"] = None
        rows.append(row)
    return rows


def evaluate(root: Path, out: Path, physics_path: Path, device: str,
             batch_size: int, resume: bool, factory=create_study_evaluator) -> list[dict]:
    spec, items = audit(out)
    if any(out == root / p or (root / p) in out.parents for p in ("python", "cpp", "config", "data")):
        raise ValueError("Simulation output must not be inside source/data directories")
    base = yaml.safe_load(physics_path.read_text(encoding="utf-8"))
    cfg = runtime_config(base, spec["grid_size"], device, batch_size)
    fence = source_fence(root)
    identity = digest(dict(schema=SCHEMA, study=spec, runtime=cfg, sources=fence,
                           geometry=read_json(out / "geometry_manifest.json")))
    run_path = out / "run_identity.json"
    if run_path.exists():
        if not resume or read_json(run_path)["signature"] != identity:
            raise ValueError("Existing/changed run: use --resume with identical inputs or a new output")
    elif (out / "cases").exists():
        raise ValueError("Orphaned case outputs without run identity; use a new output")
    else:
        write_json(run_path, dict(signature=identity, base_revision=BASE_REVISION,
                   physics_config=str(physics_path), device=device, schema=SCHEMA))
    write_json(out / "requested_runtime_config.json", cfg)
    write_json(out / "sources_before.json", fence)
    records, pending = [], []
    for item in items:
        result_path = item[3] / "result.json"
        if result_path.exists():
            rec = read_json(result_path)
            if rec["signature"] != identity or rec["geometry"] != item[2]:
                raise ValueError("Cached result does not match geometry/configuration")
            if rec["raw"].get("geometry_id") != item[2]["geometry_id"]:
                raise ValueError("Cached result ID mismatch")
            records.append(rec)
        else:
            pending.append(item)
    evaluator = None
    try:
        if pending:
            evaluator = factory(root, cfg, out / "adapter")
            write_json(out / "resolved_physics_config.json", evaluator.config)
            for start in range(0, len(pending), batch_size):
                chunk = pending[start:start + batch_size]
                raw_rows = evaluator.evaluate_batch(chunk)
                if len(raw_rows) != len(chunk):
                    raise RuntimeError("Production evaluator lost case alignment")
                for item, row in zip(chunk, raw_rows):
                    if row.get("geometry_id") != item[2]["geometry_id"]:
                        raise RuntimeError("Production result ID mismatch")
                    valid, reason = evaluator._trial_valid(row)
                    row = dict(row, external_numerical_valid=bool(valid),
                               external_numerical_reason=str(reason))
                    rec = dict(signature=identity, geometry=item[2], raw=clean(row))
                    write_json(item[3] / "result.json", rec)
                    records.append(rec)
                rows = summary_rows(records)
                write_csv(out / "parameter_study_summary.csv", rows)
    except Exception as exc:
        write_json(out / "execution_failure.json", dict(signature=identity,
                   error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc()))
        raise  # Infrastructure failures must not look like physical nonignition.
    finally:
        if evaluator is not None and hasattr(evaluator, "close"):
            evaluator.close()
        after = source_fence(root)
        write_json(out / "sources_after.json", after)
        if after != fence:
            raise RuntimeError("Protected source/config inputs changed during execution")
    ordered = {r["geometry"]["geometry_id"]: r for r in records}
    rows = summary_rows([ordered[sid] for sid in CASES])
    write_csv(out / "parameter_study_summary.csv", rows)
    for group in ("spacing", "width", "area_ratio"):
        write_csv(out / f"{group}_summary.csv", [r for r in rows if group in r["groups"].split(";")])
    write_json(out / "RUN_FINISHED.json", dict(cases=len(rows), sources_unchanged=True,
               numerically_invalid=sum(r["result_state"] == "numerically_invalid" for r in rows),
               comparison="raw_independent_cases_no_staggered", deterministic_repetitions=1))
    return rows


def summarize_experiments(source: Path, destination: Path) -> None:
    """Sample mean/SD of successful trials; nonignition remains right-censored."""
    if source.resolve() == destination.resolve():
        raise ValueError("Experimental input and summary output must be different files")
    groups = {sid: {"delays": [], "failed": 0} for sid in CASES}
    seen = set()
    with source.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            sid, rep = row["case_id"], row["replicate"]
            if sid not in CASES or not rep.isdigit() or int(rep) < 1:
                raise ValueError("Invalid experimental case/replicate")
            key = (sid, int(rep))
            if key in seen:
                raise ValueError("Duplicate experimental replicate")
            seen.add(key)
            status = row["ignition_observed"].strip().lower()
            value = row["ignition_delay_s"].strip()
            if not status:
                if value:
                    raise ValueError("Delay given without ignition status")
                continue
            window = float(row["observation_window_s"])
            if not math.isfinite(window) or window <= 0:
                raise ValueError("A finite positive observation window is required")
            if status in {"true", "1", "yes"}:
                delay = float(value)
                if not math.isfinite(delay) or not 0 < delay <= window:
                    raise ValueError("Invalid ignition delay")
                groups[sid]["delays"].append(delay)
            elif status in {"false", "0", "no"}:
                if value:
                    raise ValueError("Do not replace a censored delay with zero/window")
                groups[sid]["failed"] += 1
            else:
                raise ValueError("ignition_observed must be true/false or blank")
    output = []
    for sid, g in groups.items():
        values = g["delays"]
        total = len(values) + g["failed"]
        output.append(dict(case_id=sid, n_observed=total, n_ignited=len(values),
            n_no_ignition=g["failed"], ignition_fraction=len(values) / total if total else None,
            mean_delay_ignited_only_s=statistics.mean(values) if values else None,
            sample_sd_delay_ignited_only_s=statistics.stdev(values) if len(values) >= 2 else None))
    write_csv(destination, output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "audit", "evaluate", "summarize-experiments"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--study-config", type=Path, default=Path("config/electrode_parameter_study_25mm.yaml"))
    parser.add_argument("--physics-config", type=Path, default=Path(PHYSICS_DEFAULT))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    def relative(path):
        return path.resolve() if path.is_absolute() else (root / path).resolve()
    if args.out is None:
        parser.error("--out is required")
    out = relative(args.out)
    if args.command == "generate":
        spec = yaml.safe_load(relative(args.study_config).read_text(encoding="utf-8"))
        generate(spec, out)
        print(f"Generated 8 exact full-height cases: {out}")
    elif args.command == "audit":
        spec, items = audit(out)
        print(json.dumps(dict(cases=len(items), grid_size=spec["grid_size"],
                              spacing_mm=25 / spec["grid_size"], pde_executed=False)))
    elif args.command == "evaluate":
        rows = evaluate(root, out, relative(args.physics_config), args.device,
                        args.batch_size, args.resume)
        bad = sum(r["result_state"] == "numerically_invalid" for r in rows)
        print(f"Completed {len(rows)} cases; numerically invalid: {bad}")
        return 2 if bad else 0
    else:
        if args.input is None:
            parser.error("summarize-experiments requires --input")
        summarize_experiments(relative(args.input), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
