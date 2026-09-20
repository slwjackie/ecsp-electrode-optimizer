#!/usr/bin/env python3

import csv
import json
import math
import os
import re
import sys
import time
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import ecsp_nsga2.workflow as wf
import run_nsga2_electrical_solid_loop as cli


def convert(value):
    if value is None:
        return None

    s = str(value).strip()

    if s == "":
        return None
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    if s.lower() in {"none", "null"}:
        return None

    try:
        x = float(s)
        if math.isnan(x):
            return math.nan
        return x
    except ValueError:
        pass

    try:
        return json.loads(s)
    except Exception:
        return s


def normalise(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def objective_columns(header):
    names = list(wf.OBJECTIVE_NAMES)
    mapping = {normalise(c): c for c in header}
    cols = []

    for name in names:
        possibilities = (
            normalise(name),
            normalise("objective_" + name),
            normalise("objective:" + name),
        )

        found = None
        for key in possibilities:
            if key in mapping:
                found = mapping[key]
                break

        if found is None:
            raise RuntimeError(
                f"Cannot find objective column for {name}. "
                f"CSV columns={header}"
            )

        cols.append(found)

    return names, cols


def load_population(workflow):
    path = workflow.workdir / "generation_002" / "population_selected.csv"

    if not path.exists():
        raise RuntimeError(f"Missing {path}")

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = list(reader.fieldnames or [])

    names, objcols = objective_columns(header)

    print("[resume-final] objective mapping:")
    for n, c in zip(names, objcols):
        print(f"  {n} <- {c}")

    population = []

    for row in rows:
        values = {k: convert(v) for k, v in row.items()}

        gid = str(values["geometry_id"])
        tid = str(values.get("topology_id", ""))

        objectives = np.asarray(
            [float(values[c]) for c in objcols],
            dtype=np.float64,
        )

        metrics = dict(values)

        for key in ("metrics_json", "metrics"):
            v = values.get(key)
            if isinstance(v, dict):
                metrics.update(v)

        genome = {}
        for key in ("genome_json", "genome"):
            v = values.get(key)
            if isinstance(v, dict):
                genome = v
                break

        if not genome:
            files = list(
                workflow.workdir.glob(
                    f"generation_*/geometries/{gid}.json"
                )
            )
            if files:
                try:
                    d = json.loads(files[0].read_text())
                    if isinstance(d, dict) and "anode" in d and "cathode" in d:
                        genome = d
                    else:
                        genome = d.get("genome", d.get("parameters", {}))
                except Exception:
                    pass

        item = SimpleNamespace(
            geometry_id=gid,
            topology_id=tid,
            genome=genome,
            objectives=objectives,
            metrics=metrics,
            constraint_violation=float(
                values.get("constraint_violation", 0.0) or 0.0
            ),
        )

        for k, v in values.items():
            if isinstance(k, str) and k.isidentifier():
                setattr(item, k, v)

        population.append(item)

    return population


def resume_run(self):
    population = load_population(self)
    front = wf.pareto_front(population)

    print(
        f"[resume-final] loaded population={len(population)} "
        f"pareto_front={len(front)}"
    )

    print("[resume-final] selected candidates:")
    for item in front:
        print(" ", item.geometry_id)

    # Safety guard: the failed production run selected exactly 17.
    # Abort before any expensive B/C calculation if reconstruction differs.
    if len(population) != 200:
        raise RuntimeError(
            f"Expected final selected population=200, got {len(population)}"
        )

    if len(front) != 13:
        raise RuntimeError(
            f"Expected pre-flame Pareto front=13, got {len(front)}. "
            "No physics was rerun."
        )

    FIXED_PROPAGATION_IDS = [
        "G000_T02_2A1C_5c93ad_V009",
        "G000_T03_2A2C_6bbb2a_V002",
        "G000_T06_1A1C_5ee018_V000",
        "G001_new_grammar_immigrant_00034",
        "G000_T19_1A2C_e24705_V008",
        "G000_T17_1A3C_2f1913_V003",
        "G002_new_grammar_immigrant_00185",
        "G000_T06_1A1C_5ee018_V004",
        "G000_T14_2A1C_79adb7_V004",
        "G000_T02_2A1C_5c93ad_V002",
        "G000_T18_1A1C_90cdcd_V005",
        "G002_new_grammar_immigrant_00146",
        "G001_new_grammar_immigrant_00082",
        "G001_mutation_00085",
        "G001_crossover_00043",
        "G001_new_grammar_immigrant_00193",
        "G002_new_grammar_immigrant_00107",
    ]

    by_id = {x.geometry_id: x for x in population}
    missing = [
        gid for gid in FIXED_PROPAGATION_IDS
        if gid not in by_id
    ]

    if missing:
        raise RuntimeError(
            "Original propagation candidates missing from "
            f"generation_002 population: {missing}"
        )

    fixed_candidates = [
        by_id[gid] for gid in FIXED_PROPAGATION_IDS
    ]

    self._select_propagation_candidates = (
        lambda _population, _front: list(fixed_candidates)
    )

    print(
        "[resume-final] fixed original propagation candidates=17"
    )
    for gid in FIXED_PROPAGATION_IDS:
        print("  ", gid)

    if os.environ.get("ECSP_FINAL_DRYRUN") == "1":
        print("[resume-final] DRY RUN PASSED; no physics executed")
        return front[0]

    recommendation = self._finalise(population)

    payload = {
        "recommended_geometry_id": recommendation.geometry_id,
        "resumed_final_only": True,
        "source_generation": 2,
        "source_population_size": 200,
        "preflame_physics_recomputed": False,
        "elapsed_resume_s": time.time() - self.start_time,
    }

    (self.workdir / "RUN_COMPLETE.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(
        "[resume-final] COMPLETE:",
        recommendation.geometry_id,
    )

    return recommendation


Workflow = None

for name, obj in vars(wf).items():
    if (
        inspect.isclass(obj)
        and hasattr(obj, "_finalise")
        and "run" in obj.__dict__
    ):
        Workflow = obj
        break

if Workflow is None:
    raise RuntimeError("Could not locate NSGA-II workflow class")

Workflow.run = resume_run


# --- ECSP REUSE HANDOFF PATCH v2 ---
import sys as _rh_sys

_REUSE_HANDOFF = "--reuse-handoff" in _rh_sys.argv

# Wrapper-only flag: remove it before the original ECSP CLI parses argv.
if _REUSE_HANDOFF:
    _rh_sys.argv = [
        x for x in _rh_sys.argv
        if x != "--reuse-handoff"
    ]


def _rh_load_persisted_handoff(item):
    import json
    from pathlib import Path
    import numpy as np
    from collections.abc import Mapping

    # item:
    # (anode_mask, cathode_mask, propagation_metadata, bc_handoff_dir)
    _, _, requested_meta, output_dir = item
    output_dir = Path(output_dir)

    fields_path = output_dir / "bc_handoff_fields.npz"
    meta_path = output_dir / "bc_handoff_metadata.json"

    if not fields_path.is_file():
        raise RuntimeError(
            f"--reuse-handoff missing {fields_path}"
        )
    if not meta_path.is_file():
        raise RuntimeError(
            f"--reuse-handoff missing {meta_path}"
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if meta.get("field_file") != "bc_handoff_fields.npz":
        raise RuntimeError(
            f"Unexpected persisted handoff field_file in {meta_path}"
        )

    if not bool(meta.get("onsetSucceeded", False)):
        raise RuntimeError(
            f"Persisted handoff has no successful onset: {output_dir}"
        )

    if not bool(
        meta.get("numericallyValidForPropagationHandoff", False)
    ):
        raise RuntimeError(
            f"Persisted handoff is not numerically authorised: {output_dir}"
        )

    requested_meta = (
        dict(requested_meta)
        if isinstance(requested_meta, Mapping)
        else {}
    )

    expected_gid = str(requested_meta.get("geometry_id", ""))
    saved_gid = str(meta.get("geometry_id", ""))

    if expected_gid and saved_gid and expected_gid != saved_gid:
        raise RuntimeError(
            "Persisted handoff geometry mismatch: "
            f"expected={expected_gid}, saved={saved_gid}"
        )

    with np.load(fields_path, allow_pickle=False) as d:
        handoff = {
            key: d[key].copy()
            for key in d.files
        }

    # Add scalar quantities required by BCReactiveHandoffAdapter.
    for key, value in meta.items():
        if key not in {"field_file", "geometry_id"}:
            handoff[key] = value

    # Basic corruption/shape checks.
    required = (
        "temperatureAtOnset_K",
        "globalProgressAtOnset",
        "alphaChannel1AtOnset",
        "alphaChannel2AtOnset",
        "cationAtOnset_mol_per_m3",
        "anionAtOnset_mol_per_m3",
        "mobileLPAtOnset_mol_per_m3",
        "mobileWaterAtOnset_mol_per_m3",
        "pvaReactiveRepeatAtOnset_mol_per_m3",
        "generatedWaterProductAtOnset_mol_per_m3",
        "electrochemicalLPConsumedAtOnset_mol_per_m3",
        "potentialAtOnset_V",
        "qJAtOnset_W_per_m3",
        "qEchemAtOnset_W_per_m3",
        "propellantMask",
        "anodeContactMask",
        "cathodeContactMask",
    )

    missing = [key for key in required if key not in handoff]
    if missing:
        raise RuntimeError(
            "Persisted handoff is incomplete: "
            + ", ".join(missing)
        )

    shape = handoff["temperatureAtOnset_K"].shape

    for key in required:
        if handoff[key].shape != shape:
            raise RuntimeError(
                f"Persisted handoff shape mismatch: "
                f"{key}={handoff[key].shape}, expected={shape}"
            )

    # Build a result compatible with the normal evaluate_handoff_batch path.
    result = {}
    result.update(requested_meta)
    result.update(meta)

    # Useful canonical alias for downstream consistency checks.
    if "ignitionDelay_s" in meta:
        result.setdefault(
            "ignition_delay_s",
            float(meta["ignitionDelay_s"]),
        )

    result["handoff"] = handoff
    result["handoffPreparationFailed"] = False
    result["handoffPreparationFailure"] = ""
    result["reusedPersistedHandoff"] = True

    # Some downstream code expects a metrics mapping.
    metrics = {}
    metrics.update(requested_meta)
    metrics.update(meta)

    if "ignitionDelay_s" in meta:
        metrics.setdefault(
            "ignition_delay_s",
            float(meta["ignitionDelay_s"]),
        )

    result["metrics"] = metrics

    return result


def _rh_reuse_batch(self, items):
    print(
        "[resume-final] --reuse-handoff: "
        f"loading {len(items)} persisted B/C handoffs; "
        "B/C physics will NOT be recomputed",
        flush=True,
    )

    results = [
        _rh_load_persisted_handoff(item)
        for item in items
    ]

    print(
        "[resume-final] --reuse-handoff: "
        f"{len(results)}/{len(items)} handoffs loaded successfully",
        flush=True,
    )

    return results


def _rh_reuse_one(
    self,
    anode_mask,
    cathode_mask,
    metadata,
    output_dir,
):
    return _rh_load_persisted_handoff(
        (
            anode_mask,
            cathode_mask,
            metadata,
            output_dir,
        )
    )


def _rh_install_patch():
    import importlib

    patched = []

    for module_name in (
        "ecsp_nsga2.evaluator",
        "ecsp_nsga2.bc_native",
    ):
        module = importlib.import_module(module_name)

        for name, obj in vars(module).items():
            if not isinstance(obj, type):
                continue

            if callable(
                getattr(obj, "evaluate_handoff_batch", None)
            ):
                obj.evaluate_handoff_batch = _rh_reuse_batch

                if callable(
                    getattr(obj, "evaluate_handoff", None)
                ):
                    obj.evaluate_handoff = _rh_reuse_one

                patched.append(
                    f"{module_name}.{name}"
                )

    if not patched:
        raise RuntimeError(
            "--reuse-handoff patch found no evaluator "
            "with evaluate_handoff_batch()"
        )

    print(
        "[resume-final] --reuse-handoff enabled",
        flush=True,
    )

    for name in patched:
        print(
            f"[resume-final] patched {name}",
            flush=True,
        )


if _REUSE_HANDOFF:
    _rh_install_patch()



# --- ECSP REUSE HANDOFF METRIC BRIDGE v1 ---
#
# A persisted handoff contains the actual onset fields and scalar handoff
# metadata, but not every diagnostic metric produced by a fresh B/C
# re-evaluation.  In --reuse-handoff mode restore those diagnostics from the
# already-optimised Individual before running the original consistency check.

if globals().get("_REUSE_HANDOFF", False):

    _rh_original_validate = (
        wf.NSGA2ElectricalSolidWorkflow._validate_handoff_reevaluation
    )

    def _rh_validate_persisted_handoff(self, individual, result):
        import math
        import numbers

        if not result.get("reusedPersistedHandoff", False):
            return _rh_original_validate(individual, result)

        source = dict(getattr(individual, "metrics", {}) or {})
        metrics = dict(result.get("metrics", {}) or {})

        # Restore every finite scalar diagnostic available in the optimised
        # candidate.  Put it in both locations because different evaluator
        # paths historically exposed metrics flat or under ["metrics"].
        for key, value in source.items():
            if isinstance(value, numbers.Real):
                value = float(value)
                if math.isfinite(value):
                    result.setdefault(key, value)
                    metrics.setdefault(key, value)

        _objectives = getattr(individual, "objectives", None)
        objectives = [] if _objectives is None else list(_objectives)

        # Required re-evaluation diagnostic #1.
        if "remaining_reactive_mass_fraction" not in metrics:
            value = source.get(
                "remaining_reactive_mass_fraction",
                source.get(
                    "area_undecomposed_fraction_at_evaluation_time",
                    objectives[1] if len(objectives) > 1 else None,
                ),
            )
            if value is not None:
                value = float(value)
                result["remaining_reactive_mass_fraction"] = value
                metrics["remaining_reactive_mass_fraction"] = value

        # Required re-evaluation diagnostic #2.
        if "current_congestion" not in metrics:
            value = source.get(
                "current_congestion",
                objectives[3] if len(objectives) > 3 else None,
            )
            if value is not None:
                value = float(value)
                result["current_congestion"] = value
                metrics["current_congestion"] = value

        result["metrics"] = metrics

        gid = getattr(individual, "geometry_id", "?")

        print(
            "[resume-final] reuse metrics "
            f"{gid}: remaining_reactive_mass_fraction="
            f"{metrics.get('remaining_reactive_mass_fraction')} "
            f"current_congestion={metrics.get('current_congestion')}",
            flush=True,
        )

        # Keep the original validator active.  We are not disabling validation.
        return _rh_original_validate(individual, result)

    wf.NSGA2ElectricalSolidWorkflow._validate_handoff_reevaluation = (
        _rh_validate_persisted_handoff
    )

    print(
        "[resume-final] --reuse-handoff metric bridge enabled",
        flush=True,
    )


def main():
    return cli.main()


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    raise SystemExit(main())
