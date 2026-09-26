"""Fixed-library preflame orchestration. Original geometry/physics are read-only.

Different physical domains never share one evaluator. CUDA execution is explicit;
this avoids the CPU-first hybrid scheduler consuming both members of a pair.
"""
from __future__ import annotations

import copy
import csv
import dataclasses
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import time
import traceback
import zipfile

import numpy as np
import yaml

from .objectives import digest, runtime_config, strict_json
from .paired_screening import ScreeningPolicy, pair_context_errors, screen_pair

FIVE_IDS = ("E058", "E114", "R038", "R050", "R091")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(strict_json(value), indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def protected_hashes(root):
    root = Path(root)
    prefixes = ("cpp", "python/ecsp_native", "python/ecsp_nsga2", "python/ecsp_v6",
                "python/ecsp_cuda", "python/ecsp_cpp", "python/ecsp_reactive", "config")
    return {p.relative_to(root).as_posix(): file_hash(p) for prefix in prefixes
            for p in sorted((root / prefix).rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in
            {".py", ".cpp", ".cc", ".h", ".hpp", ".cu", ".yaml", ".yml", ".json", ".csv"}}


def case_ids(library, only_ids=None, expected_count=None):
    library = Path(library)
    ids = sorted(d.name for d in library.iterdir() if d.is_dir() and (d / "metadata.json").is_file())
    if only_ids is not None:
        wanted = list(only_ids)
        if len(wanted) != len(set(wanted)) or set(wanted) - set(ids):
            raise ValueError("Duplicate or missing requested IDs")
        ids = sorted(wanted)
    if not ids or any(not re.fullmatch(r"[A-Za-z0-9_-]+", sid) for sid in ids):
        raise ValueError("Empty library or unsafe geometry ID")
    if expected_count is not None and len(ids) != expected_count:
        raise ValueError(f"Expected {expected_count} fixed geometries, found {len(ids)}; nothing was evaluated")
    return ids


def load_case(path):
    """Verify the actual masks/CAD; never repair, resize, or redesign a candidate."""
    import shapely
    from .five_topologies import render, gap_guard, signature, mask_signature, first_erosion_width, rolling_open
    path = Path(path)
    m = read_json(path / "metadata.json")
    if m.get("status") != "geometry_accepted" or m["source_id"] != path.name:
        raise ValueError("Unaccepted geometry or ID mismatch")
    if m.get("mask_file_sha256") and m["mask_file_sha256"] != file_hash(path / "mask.npz"):
        raise ValueError("Stored mask checksum mismatch")
    with np.load(path / "mask.npz", allow_pickle=False) as f:
        a, c = np.asarray(f["anode"], bool), np.asarray(f["cathode"], bool)
        if not f["propellant"].all() or not np.isclose(float(f["domain_mm"]), m["domain_mm"]):
            raise ValueError("Full-propellant/domain contract mismatch")
    n, L = int(m["grid_size"]), float(m["domain_mm"])
    if a.shape != (n, n) or c.shape != (n, n) or not a.any() or not c.any() or (a & c).any():
        raise ValueError("Invalid electrode masks")
    master = read_json(path / "master.json")
    shapes = [shapely.from_geojson(master[k]) for k in ("anode", "cathode")]
    pa, pc = shapes
    if not shapely.box(0, 0, L, L).covers(pa.union(pc)) or pa.distance(pc) < 3.0 - 1e-10:
        raise ValueError("CAD outside domain or gap below 3 mm")
    if abs(pa.area - pc.area) / max(pa.area, pc.area) > 1e-6 or gap_guard(a, c, L / n) < 3.0 - 1e-10:
        raise ValueError("CAD area equality or raster gap failure")
    for mask, shape, polarity in zip((a, c), shapes, ("anode", "cathode")):
        sg = mask_signature(mask)
        if sg["components4"] != sg["components8"] or sg["components8"] != m[polarity + "_components"] or sg["holes"] != sum(signature(shape)):
            raise ValueError("CAD/raster topology mismatch")
        if not np.array_equal(render(shape, n, L), mask):
            raise ValueError("Stored CAD and solver mask disagree")
        if abs(mask.mean() * L * L - shape.area) / shape.area > 0.01 + 1e-12:
            raise ValueError("CAD/raster contact area error exceeds 1%")
        if first_erosion_width(shape) < 1.998 or shape.difference(rolling_open(shape, 1.0)).area / shape.area >= 0.002:
            raise ValueError("Existing 2 mm feature-width audit failed")
    return m, a, c


def audit_library(library, only_ids=None, expected_count=None):
    ids = case_ids(library, only_ids, expected_count)
    records = []
    for sid in ids:
        m, a, c = load_case(Path(library) / sid)
        records.append(dict(source_id=sid, domain_mm=m["domain_mm"], grid_size=m["grid_size"],
                            anode_area_mm2=float(a.mean() * m["domain_mm"] ** 2),
                            cathode_area_mm2=float(c.mean() * m["domain_mm"] ** 2)))
    return {"count": len(ids), "ids": ids, "geometry_records": records, "pde_executed": False}


def prepare_library(root, base_archive, out, five_library=None):
    """143 byte-identical v5 geometries + the five unchanged main fitters."""
    root, out = Path(root).resolve(), Path(out).resolve()
    catalog = read_json(root / "data/image_design/paired_catalogue_148.json")
    if file_hash(base_archive) not in catalog["accepted_base_archive_sha256"]:
        raise ValueError("Unrecognised archive. Use the supplied frozen-143 bundle or original v5 ZIP")
    if out.exists():
        raise ValueError("Output exists; use a new directory, never overwrite a geometry library")
    staging = out.with_name(out.name + ".preparing")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        lib = staging / "library"
        with zipfile.ZipFile(base_archive) as z:
            members = z.namelist()
            for sid in catalog["frozen_143_ids"]:
                for name in ("metadata.json", "mask.npz", "master.json"):
                    hits = [x for x in members if x.endswith(f"library/{sid}/{name}")]
                    if len(hits) != 1:
                        raise ValueError(f"Missing/ambiguous frozen file: {sid}/{name}")
                    data = z.read(hits[0])
                    dest = lib / sid / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(data)
        if five_library is None:
            # Existing source remains unchanged. Only these five are generated.
            import run_five_topology_preflame as five
            refs = five.load_refs(root / "data/image_design/five_topology_references.json")
            five.generate(refs, staging / "five_generated")
            five_library = staging / "five_generated/library"
        for sid in FIVE_IDS:
            src = Path(five_library) / sid
            load_case(src)
            (lib / sid).mkdir(parents=True, exist_ok=False)
            for name in ("metadata.json", "mask.npz", "master.json"):
                shutil.copy2(src / name, lib / sid / name)
        if case_ids(lib, expected_count=148) != sorted(catalog["frozen_143_ids"] + list(FIVE_IDS)):
            raise ValueError("Catalogue ID mismatch")
        provenance = {sid: {name: file_hash(lib / sid / name) for name in
                            ("metadata.json", "mask.npz", "master.json")} for sid in case_ids(lib)}
        write_json(staging / "library_checksums.json", provenance)
        write_json(staging / "library_provenance.json", dict(base_archive_sha256=file_hash(base_archive),
                    unchanged_frozen_geometries=143, current_special_geometries=5, total=148,
                    candidates_resized=False, full_geometry_audit_required_before_evaluation=True))
        os.replace(staging, out)
    except BaseException:
        # Keep staging as evidence; no original archive or run is removed.
        raise
    return out / "library"


def make_context(m, a, c, resolved, voltage, settings_hash):
    L = float(m["domain_mm"])
    bc = resolved["bcGlobal"]
    return dict(domain_mm=L, grid_size=int(m["grid_size"]),
                anode_area_mm2=float(a.mean() * L * L), cathode_area_mm2=float(c.mean() * L * L),
                electrode_area_fraction=float((a.mean() + c.mean()) * L * L / L ** 2),
                area_basis="solver_mask", minimum_gap_mm=3.0, minimum_width_mm=2.0,
                reference_voltage_V=float(voltage), evaluation_time_s=float(bc["evaluationTime_s"]),
                initial_temperature_K=float(bc["thermal"]["initialTemperature_K"]),
                physics_config_hash=settings_hash)


def metadata(m, baseline=False):
    return dict(geometry_id=m["source_id"] + ("_staggered" if baseline else ""),
                topology_id="independent_area_matched_staggered" if baseline else m["source_id"],
                intended_anode_components=2 if baseline else m["anode_components"],
                intended_cathode_components=2 if baseline else m["cathode_components"],
                surface_contact_model=True, hidden_bus_assumed=True, domain_mm=m["domain_mm"],
                electrode_area_fraction=m["electrode_area_fraction"],
                source_role="external_reference_only" if baseline else "fixed_library_screening",
                baseline_type="area_matched_staggered" if baseline else None)


def _production_dependencies():
    from ecsp_nsga2.evaluator import create_evaluator
    from ecsp_nsga2.baselines import generate_area_matched_staggered
    from ecsp_nsga2.geometry import GeometryLimits
    return create_evaluator, generate_area_matched_staggered, GeometryLimits


def evaluate_library(root, config_path, library, out, *, only_ids=None, expected_count=None,
                     device="cuda", pcg_max_iterations=None, resume=False, retry_failed=False,
                     policy=None):
    """Sequential independent pairs, explicit GPU, checkpointed; no NSGA-II.

    Whole-pair exceptions are stored as execution failures, never as nonignition.
    No hidden retry with changed tolerances/physics, no cross-context batching.
    """
    root, library, out = Path(root).resolve(), Path(library).resolve(), Path(out).resolve()
    policy = policy or ScreeningPolicy()
    ids = case_ids(library, only_ids, expected_count)
    base = yaml.safe_load(Path(config_path).read_text())
    if device not in ("cuda", "cpu"):
        raise ValueError("Explicit cuda or cpu device is required")
    fence = protected_hashes(root)
    # Input signature excludes classification policy so old raw results can be reclassified.
    identity = digest(dict(base=base, protected=fence, device=device, pcg=pcg_max_iterations))
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "requested_config.json", base)
    write_json(out / "protected_before.json", fence)
    create, baseline_generator, Limits = _production_dependencies()
    reports = []
    try:
        for sid in ids:
            case = out / "preflame" / sid
            inputs = {name: file_hash(library / sid / name) for name in ("mask.npz", "metadata.json", "master.json")}
            key = digest(dict(run=identity, source_id=sid, inputs=inputs))
            checkpoint = case / "raw_pair.json"
            failure = case / "failure.json"
            if checkpoint.exists():
                saved = read_json(checkpoint)
                if not resume or saved.get("input_signature") != key:
                    raise ValueError(f"{sid}: existing result or changed inputs; use --resume with identical inputs or a new output directory")
                report = screen_pair(saved, policy)
                reports.append(report)
                write_json(case / "screening.json", report)
                print(f"[paired] {sid} cached -> {report['comparison_state']}", flush=True)
                write_reports(out, reports)
                continue
            if failure.exists() and not (resume and retry_failed):
                raise ValueError(f"{sid}: previous execution failure; use --resume --retry-failed or a new directory")
            t = time.perf_counter()
            evaluator = None
            print(f"[paired] {sid} start; device={device}", flush=True)
            try:
                m, a, c = load_case(library / sid)
                cfg = runtime_config(base, m)
                # Runtime geometry and routing only. Original config files are untouched.
                cfg["geometry"]["target_area_fraction_per_polarity"] = float((a.mean() + c.mean()) / 2)
                cfg["evaluator"]["backend"] = "bc_global_native"
                cfg["evaluator"]["device"] = device
                cfg["evaluator"].setdefault("native", {})["cpu_workers"] = 0
                numerics = cfg["evaluator"].setdefault("base_overrides", {}).setdefault("numerics", {})
                numerics["physicsDevice"] = device
                if pcg_max_iterations is not None:
                    if isinstance(pcg_max_iterations, bool) or int(pcg_max_iterations) != pcg_max_iterations or pcg_max_iterations < 1:
                        raise ValueError("PCG iteration budget must be a positive integer")
                    numerics.setdefault("potentialSolver", {})["maximumIterationsCoupled"] = pcg_max_iterations
                fields = {f.name for f in dataclasses.fields(Limits)}
                limits = Limits(**{k: v for k, v in cfg["geometry"].items() if k in fields})
                options = cfg.get("baselines", {}).get("area_matched_staggered", {})
                opts = {k: options[k] for k in ("fingers_per_polarity", "target_interdigitation_overlap_fraction",
                        "minimum_interdigitation_overlap_fraction", "maximum_gap_safety_pixels") if k in options}
                print(f"[paired] {sid} constructing same-context staggered", flush=True)
                baseline_started = time.perf_counter()
                _, raster, params = baseline_generator(limits, physics_grid_size=m["grid_size"],
                    target_area_fraction_per_polarity=cfg["geometry"]["target_area_fraction_per_polarity"], **opts)
                ba, bca = np.asarray(raster.anode_mask, bool), np.asarray(raster.cathode_mask, bool)
                print(
                    f"[paired] {sid} staggered ready; elapsed="
                    f"{time.perf_counter() - baseline_started:.3f}s",
                    flush=True,
                )
                if ba.shape != a.shape or bca.shape != c.shape:
                    raise ValueError("Baseline grid differs")
                from .five_topologies import gap_guard
                if gap_guard(ba, bca, m["domain_mm"] / m["grid_size"]) < 3.0 - 1e-10:
                    raise ValueError("Baseline raster gap failure")
                for x, y in ((a, ba), (c, bca)):
                    if not y.any() or abs(float(x.mean()) - float(y.mean())) / float(x.mean()) > policy.area_rtol + 1e-12:
                        raise ValueError("Candidate and baseline actual contact areas differ by more than 1%")
                adapter = copy.deepcopy(cfg["evaluator"])
                adapter["physics_config"] = copy.deepcopy(cfg)
                # Capture only representative reference-run fields. Vmin trials
                # remain scalar-only and no time-history field stack is retained.
                adapter["save_representative_fields"] = True
                write_json(case / "runtime_config.json", cfg)
                write_json(case / "baseline_parameters.json", dataclasses.asdict(params))
                np.savez_compressed(case / "staggered_mask.npz", anode=ba, cathode=bca, domain_mm=m["domain_mm"])
                evaluator = create(root, adapter, case / "adapter", False)
                resolved = evaluator.config
                if not np.isclose(float(resolved["geometry"]["domainSize_m"]) * 1000, m["domain_mm"]):
                    raise ValueError("Resolved physics domain differs from CAD domain")
                if int(evaluator.grid_size) != m["grid_size"] or str(evaluator.device).split(":")[0] != device:
                    raise ValueError("Resolved evaluator grid/device differs")
                write_json(case / "resolved_physics_config.json", resolved)
                cc = make_context(m, a, c, resolved, evaluator.voltage, digest(resolved))
                bc = make_context(m, ba, bca, resolved, evaluator.voltage, digest(resolved))
                errors = pair_context_errors(cc, bc, policy)
                if errors:
                    raise ValueError("; ".join(errors))
                print(f"[paired] {sid} PDE begin; N={m['grid_size']} L={m['domain_mm']} mm", flush=True)
                raw = evaluator.evaluate_batch([(a, c, metadata(m), case / "candidate"),
                                                (ba, bca, metadata(m, True), case / "staggered")])
                if len(raw) != 2:
                    raise ValueError("Lost candidate/baseline alignment")
                cand, stag = [dict(row) for row in raw]
                for row, expected_id in ((cand, sid), (stag, sid + "_staggered")):
                    if row.get("geometry_id") != expected_id:
                        raise ValueError("Wrong returned geometry ID")
                    ok, reason = evaluator._trial_valid(row)
                    row["external_numerical_valid"] = bool(ok)
                    row["external_numerical_reason"] = str(reason)
                saved = dict(source_id=sid, context=cc, baseline_context=bc, candidate_raw=cand,
                             baseline_raw=stag, input_signature=key, elapsed_s=time.perf_counter() - t)
                write_json(checkpoint, saved)
                report = screen_pair(strict_json(saved), policy)
                write_json(case / "result.json", {**saved, "screening": report})
                write_json(case / "screening.json", report)
                if failure.exists():
                    failure.rename(case / f"previous_failure_{time.time_ns()}.json")
                print(f"[paired] {sid} done -> {report['comparison_state']} ({saved['elapsed_s']:.1f}s)", flush=True)
            except Exception as exc:
                # Do not interpret infrastructure/PCG errors as physical nonignition.
                report = dict(source_id=sid, screening_valid=False, comparison_state="execution_failed",
                              labels=["execution_failed"], reasons=[str(exc)], cross_design_ranking=False)
                write_json(failure, dict(input_signature=key, error_type=type(exc).__name__,
                                        error=str(exc), traceback=traceback.format_exc()))
                print(f"[paired] {sid} execution_failed: {exc}", flush=True)
            finally:
                if evaluator is not None and hasattr(evaluator, "close"):
                    evaluator.close()
            reports.append(report)
            write_reports(out, reports)
    finally:
        after = protected_hashes(root)
        write_json(out / "protected_after.json", after)
        if after != fence:
            raise RuntimeError("Protected solver/config sources changed during run")
    write_json(out / "RUN_FINISHED.json", dict(requested=len(ids), processed=len(reports),
                execution_failures=sum(x["comparison_state"] == "execution_failed" for x in reports),
                cross_design_ranking=False, physics_sources_unchanged=True))
    return reports


def write_reports(out, reports):
    out = Path(out)
    rows = sorted(reports, key=lambda x: x["source_id"])
    if len(rows) != len({x["source_id"] for x in rows}):
        raise ValueError("Duplicate geometry IDs in screening report")
    write_json(out / "screening_summary.json", rows)
    groups = {}
    for row in rows:
        for label in row["labels"]:
            groups.setdefault(label, []).append(row["source_id"])
    write_json(out / "screening_groups.json", groups)
    cols = ["source_id", "screening_valid", "comparison_state", "ignition_screen", "thermal_screen",
            "congestion_screen", "R_t", "R_V", "R_J", "R_T", "voltage_comparison", "labels", "reasons"]
    with (out / "screening_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row[k], ensure_ascii=False) if isinstance(row.get(k), (dict, list)) else row.get(k) for k in cols})
    shortlist = []
    for row in rows:
        aims = [x for x in row["labels"] if x in ("ignition_superior_candidates", "thermal_promising_candidates", "congestion_superior_candidates")]
        if aims:
            shortlist.append(dict(source_id=row["source_id"], purposes=aims, target_domain_mm=25,
                 minimum_width_mm=2, minimum_gap_mm=3, status="requires_redesign_and_revalidation",
                 feasibility_not_guaranteed=True, source_context=row.get("context")))
    write_json(out / "fixed_25mm_revalidation_manifest.json", shortlist)
    head = "<html><meta charset='utf-8'><title>ECSP paired screening</title><h1>Paired screening — no cross-design rank</h1><p>Each row is compared only with its own matched staggered. Labels may overlap. Thermal gain is not a Vmin estimate. ID order is not performance order.</p><table border='1' cellpadding='6'><tr>"
    head += "".join(f"<th>{html.escape(k)}</th>" for k in cols) + "</tr>"
    for row in rows:
        head += "<tr>" + "".join(f"<td>{html.escape(str(row.get(k, '')))}</td>" for k in cols) + "</tr>"
    (out / "screening_report.html").write_text(head + "</table></html>", encoding="utf-8")


def enrich_legacy_record(record, result_path):
    """Recover missing observational context from SAVED artifacts, not new PDEs.

    A legacy context contains candidate CAD area; measure its saved solver mask
    before calling it an actual-area-matched pair. Missing evidence stays invalid.
    """
    r = copy.deepcopy(record)
    case = Path(result_path).resolve().parent
    if r.get("context", {}).get("area_basis") == "solver_mask":
        return r
    cc, bc = r.setdefault("context", {}), r.setdefault("baseline_context", {})
    candidates = [case / "resolved_physics_config.json", case / "adapter/primary/resolved_physics_config.json",
                  case / "adapter/resolved_physics_config.json"]
    resolved_path = next((p for p in candidates if p.is_file()), None)
    if resolved_path is None:
        return r
    resolved = read_json(resolved_path)
    sid = r["source_id"]
    # Old five-topology and old image-library layouts are both supported.
    masks = [case.parents[1] / "library" / sid / "mask.npz",
             case.parent / "library" / sid / "mask.npz"]
    mask_path = next((p for p in masks if p.is_file()), None)
    if mask_path is not None:
        with np.load(mask_path, allow_pickle=False) as f:
            L = float(f["domain_mm"])
            a, c = np.asarray(f["anode"], bool), np.asarray(f["cathode"], bool)
        if np.isclose(L, cc.get("domain_mm", -1)) and a.shape == (cc.get("grid_size"),) * 2 and c.shape == a.shape:
            cc.update(anode_area_mm2=float(a.mean() * L * L), cathode_area_mm2=float(c.mean() * L * L), area_basis="solver_mask")
    bm = case / "staggered_mask.npz"
    if not bm.exists():
        bm = case / "independent_staggered.npz"
    params = case / "baseline_parameters.json"
    if bm.is_file():
        with np.load(bm, allow_pickle=False) as f:
            L = float(f["domain_mm"])
            ba, bca = np.asarray(f["anode"], bool), np.asarray(f["cathode"], bool)
        if np.isclose(L, bc.get("domain_mm", -1)) and ba.shape == (bc.get("grid_size"),) * 2 and bca.shape == ba.shape:
            bc.update(anode_area_mm2=float(ba.mean() * L * L), cathode_area_mm2=float(bca.mean() * L * L), area_basis="solver_mask")
    elif params.is_file():
        bp = read_json(params)
        frac = bp.get("physics_area_fraction_per_polarity")
        if isinstance(frac, (int, float)) and 0 < frac < 0.5 and bp.get("physics_grid_size") == bc.get("grid_size"):
            expected = frac * bc["domain_mm"] ** 2
            if all(np.isclose(bc.get(k, -1), expected) for k in ("anode_area_mm2", "cathode_area_mm2")):
                bc["area_basis"] = "solver_mask"
    saved_bc = resolved.get("bcGlobal", {})
    try:
        t0 = float(saved_bc["thermal"]["initialTemperature_K"])
        end = float(saved_bc["evaluationTime_s"])
        voltage = float(resolved["coupled"]["voltage_V"])
        resolved_domain = float(resolved["geometry"]["domainSize_m"]) * 1000
    except (KeyError, TypeError, ValueError):
        return r
    for ctx in (cc, bc):
        if not np.isclose(ctx.get("domain_mm", -1), resolved_domain):
            ctx["physics_config_hash"] = "resolved_domain_mismatch_" + str(ctx.get("domain_mm"))
            ctx["area_basis"] = "unverified"
            continue
        ctx.update(initial_temperature_K=t0, evaluation_time_s=end, reference_voltage_V=voltage)
        if all(k in ctx for k in ("anode_area_mm2", "cathode_area_mm2", "domain_mm")):
            ctx["electrode_area_fraction"] = (ctx["anode_area_mm2"] + ctx["cathode_area_mm2"]) / ctx["domain_mm"] ** 2
    r["context_evidence"] = dict(resolved_config=str(resolved_path), resolved_config_sha256=file_hash(resolved_path),
                                 candidate_mask=str(mask_path) if mask_path else None,
                                 baseline_artifact=str(bm if bm.exists() else params))
    return r


def reclassify(paths, out, policy=None):
    """Read-only raw-results import. No evaluator imports, no solver invocation."""
    results = []
    sources = []
    files = sorted({p.resolve() for path in paths for p in
                    ([Path(path)] if Path(path).is_file() else Path(path).rglob("result.json"))})
    if not files:
        raise ValueError("No result.json files found")
    for path in files:
        raw = read_json(path)
        if "candidate_raw" not in raw or "baseline_raw" not in raw:
            raise ValueError(f"Not a paired raw result: {path}")
        record = enrich_legacy_record(raw, path)
        report = screen_pair(record, policy)
        report["raw_result_path"] = str(path)
        report["raw_result_sha256"] = file_hash(path)
        report["context_evidence"] = record.get("context_evidence")
        results.append(report)
        sources.append(dict(path=str(path), sha256=file_hash(path)))
    Path(out).mkdir(parents=True, exist_ok=True)
    write_reports(out, results)
    write_json(Path(out) / "reclassification_provenance.json", dict(sources=sources, pde_executed=False,
                                                                  original_results_modified=False))
    return results
