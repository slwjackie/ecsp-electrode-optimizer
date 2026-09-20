import os
import json
import copy
from pathlib import Path

import numpy as np
import yaml

from ecsp_nsga2.post_onset_batch import run_post_onset_batch



def main():
    run = Path(os.environ["RUN"])
    cfg_path = Path(os.environ["CFG"])
    gid = os.environ["ID"]
    out = Path(os.environ["OUT"])
    horizon = float(os.environ["HORIZON"])

    cfg = yaml.safe_load(cfg_path.read_text())


    # Find the propagation/refinement configuration.
    matches = []

    def walk(obj, path="root"):
        if isinstance(obj, dict):
            if "duration_s" in obj and "time_step_s" in obj:
                matches.append((path, obj))
            for key, value in obj.items():
                walk(value, f"{path}.{key}")
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                walk(value, f"{path}[{i}]")


    walk(cfg)

    matches = [
        item for item in matches
        if "post_onset" not in item[0].lower()
    ]

    if not matches:
        raise RuntimeError("Could not locate propagation configuration")

    matches.sort(
        key=lambda item: (
            "propagation" not in item[0].lower(),
            "refinement" not in item[0].lower(),
            len(item[0]),
        )
    )

    prop_path, propagation = matches[0]
    propagation = copy.deepcopy(propagation)
    propagation["duration_s"] = horizon

    print(f"[debug] candidate={gid}", flush=True)
    print(f"[debug] propagation config={prop_path}", flush=True)
    print(f"[debug] horizon={horizon}s", flush=True)


    # Locate bcGlobal.
    bc = cfg["bc_global"]
    print("[debug] bcGlobal config=root.bc_global", flush=True)

    post = copy.deepcopy(cfg["post_onset"])
    post.setdefault("execution", {})["cpu_workers"] = 1


    # Load the persisted handoff.
    hdir = (
        run
        / "final"
        / "propagation_candidates"
        / gid
        / "bc_handoff"
    )

    meta_path = hdir / "bc_handoff_metadata.json"
    field_path = hdir / "bc_handoff_fields.npz"

    if not meta_path.exists():
        raise RuntimeError(f"Missing metadata: {meta_path}")

    if not field_path.exists():
        raise RuntimeError(f"Missing fields: {field_path}")

    metadata = json.loads(meta_path.read_text())

    with np.load(field_path, allow_pickle=False) as z:
        fields = {key: z[key].copy() for key in z.files}

    handoff = dict(metadata)
    handoff.update(fields)

    print(
        f"[debug] loaded persisted handoff: {len(fields)} field arrays",
        flush=True,
    )


    # Never write into the production candidate directory.
    case_out = out / gid
    case_out.mkdir(parents=True, exist_ok=True)

    results = run_post_onset_batch(
        [handoff],
        propagation,
        bc,
        [case_out],
        post_onset_config=post,
        full_bc_config=cfg,
    )

    result = results[0]

    print("\n========== RESULT ==========", flush=True)
    print(f"type={type(result).__name__}", flush=True)

    if isinstance(result, BaseException):
        print(f"ERROR TYPE: {type(result).__name__}", flush=True)
        print(f"MESSAGE: {result}", flush=True)
    else:
        print(result, flush=True)


if __name__ == "__main__":
    main()
