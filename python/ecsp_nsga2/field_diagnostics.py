"""Representative pre-flame field capture for paired screening.

This module is diagnostics/output only.  It does not advance the PDE, change
the potential solve, alter material properties, or invent a current proxy.
Current-density fields are reconstructed from the native solver's saved state
and potential with the production B/C transport closure and compute_current().
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(value), indent=2, allow_nan=False), encoding="utf-8")


def _lane(tensor, index: int):
    import torch
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Representative field source must be a torch.Tensor")
    if tensor.ndim < 3 or index >= tensor.shape[0]:
        raise ValueError("Representative field tensor has no requested batch lane")
    return tensor[index:index + 1]


def _make_geometry(item, grid_size: int, domain_size_m: float, device):
    import torch
    from ecsp_v6.physics.geometry import GeometryBatch
    from ecsp_v6.physics.numerics import resize_nearest_numpy

    a_raw, c_raw, metadata, _ = item
    a = resize_nearest_numpy(np.asarray(a_raw, dtype=bool), grid_size)
    c = resize_nearest_numpy(np.asarray(c_raw, dtype=bool), grid_size)
    anode = torch.as_tensor(a, device=device, dtype=torch.bool).unsqueeze(0)
    cathode = torch.as_tensor(c, device=device, dtype=torch.bool).unsqueeze(0)
    # B/C surface-overlay semantics: the entire 2-D domain remains propellant;
    # electrode masks are top-surface labels rather than removed/fixed cells.
    fixed = torch.zeros_like(anode)
    propellant = torch.ones_like(anode)
    return GeometryBatch(
        geometry_ids=[str(metadata.get("geometry_id"))],
        anode=anode,
        cathode=cathode,
        fixed=fixed,
        propellant=propellant,
        grid_size=grid_size,
        domain_size_m=domain_size_m,
        minimum_gap_m=0.0,
    ), a, c


def _state_from_final(fields: Mapping[str, Any], index: int) -> dict[str, Any]:
    return {
        "temperature": _lane(fields["temperature_K"], index),
        "cation": _lane(fields["cation_mol_per_m3"], index),
        "anion": _lane(fields["anion_mol_per_m3"], index),
        "water": _lane(fields["water_mol_per_m3"], index),
        "globalProgress": _lane(fields["globalProgress"], index),
        "potential": _lane(fields["potential_V"], index),
    }


def _state_from_onset(fields: Mapping[str, Any], index: int) -> dict[str, Any]:
    return {
        "temperature": _lane(fields["temperatureAtOnset_K"], index),
        "cation": _lane(fields["cationAtOnset_mol_per_m3"], index),
        "anion": _lane(fields["anionAtOnset_mol_per_m3"], index),
        "water": _lane(fields["mobileWaterAtOnset_mol_per_m3"], index),
        "globalProgress": _lane(fields["globalProgressAtOnset"], index),
        "potential": _lane(fields["potentialAtOnset_V"], index),
    }


def _numpy2(tensor) -> np.ndarray:
    value = tensor.detach().cpu().numpy()
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2:
        raise ValueError(f"Expected a 2-D representative field, got {value.shape}")
    return np.asarray(value, dtype=np.float64)


def _field_metrics(
    temperature: np.ndarray,
    current: np.ndarray,
    progress: np.ndarray,
    initial_temperature_K: float,
) -> dict[str, Any]:
    t = temperature[np.isfinite(temperature)]
    j = current[np.isfinite(current) & (current >= 0.0)]
    x = progress[np.isfinite(progress)]
    if t.size == 0 or j.size == 0 or x.size == 0:
        raise ValueError("Representative fields contain no finite data")
    p05, p95 = np.percentile(t, [5.0, 95.0])
    rise = t - float(initial_temperature_K)
    mean_rise = float(np.mean(rise))
    std_rise = float(np.std(rise))
    mean_t = float(np.mean(t))
    std_t = float(np.std(t))
    mean_j = float(np.mean(j))
    std_j = float(np.std(j))
    return {
        "temperatureMean_K": mean_t,
        "temperatureStd_K": std_t,
        "temperatureCV": std_t / mean_t if mean_t > 0.0 else None,
        "temperatureP05_K": float(p05),
        "temperatureP95_K": float(p95),
        "temperatureP95MinusP05_K": float(p95 - p05),
        "temperatureRiseMean_K": mean_rise,
        "temperatureRiseStd_K": std_rise,
        "temperatureRiseCV": (
            std_rise / mean_rise if mean_rise > 1.0e-12 else None
        ),
        "currentDensityMean_A_per_m2": mean_j,
        "currentDensityStd_A_per_m2": std_j,
        "currentDensityCV": std_j / mean_j if mean_j > 1.0e-30 else None,
        "currentDensityP99_A_per_m2": float(np.percentile(j, 99.0)),
        "currentDensityMaximum_A_per_m2": float(np.max(j)),
        "globalProgressMean": float(np.mean(x)),
        "globalProgressMaximum": float(np.max(x)),
    }


def _palette(values: np.ndarray, kind: str) -> np.ndarray:
    v = np.clip(values, 0.0, 1.0)
    if kind == "temperature":
        stops = np.array([[0, 20, 80], [0, 170, 220], [250, 230, 50], [220, 30, 20]], float)
    elif kind == "current":
        stops = np.array([[0, 0, 0], [30, 40, 160], [0, 220, 220], [255, 240, 60], [255, 255, 255]], float)
    elif kind == "potential":
        stops = np.array([[20, 50, 180], [245, 245, 245], [190, 30, 30]], float)
    else:
        stops = np.array([[0, 0, 0], [80, 30, 150], [220, 80, 100], [255, 235, 60]], float)
    pos = v * (len(stops) - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(stops) - 1)
    w = (pos - lo)[..., None]
    return np.clip(stops[lo] * (1.0 - w) + stops[hi] * w, 0, 255).astype(np.uint8)


def _render_scalar(
    path: Path,
    array: np.ndarray,
    title: str,
    kind: str,
    *,
    fixed_range: tuple[float, float] | None = None,
    log_nonnegative: bool = False,
) -> None:
    raw = np.asarray(array, dtype=np.float64)
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        raise ValueError(f"{title}: no finite pixels")
    raw_min, raw_max = float(np.min(finite)), float(np.max(finite))
    display = raw.copy()
    transform_note = ""
    if log_nonnegative:
        display = np.log1p(np.maximum(display, 0.0))
        finite_display = display[np.isfinite(display)]
        vmin, vmax = 0.0, float(np.max(finite_display))
        transform_note = " | display=log1p"
    elif fixed_range is not None:
        vmin, vmax = map(float, fixed_range)
    else:
        vmin, vmax = raw_min, raw_max
    if not math.isfinite(vmax - vmin) or vmax <= vmin:
        vmax = vmin + max(1.0, abs(vmin) * 1.0e-12)
    norm = np.clip((display - vmin) / (vmax - vmin), 0.0, 1.0)
    rgb = _palette(norm, kind)
    image = Image.fromarray(rgb, "RGB")
    max_side = 1100
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize(
            (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
            Image.Resampling.BILINEAR,
        )
    header, footer = 46, 42
    canvas = Image.new("RGB", (image.width, image.height + header + footer), "white")
    canvas.paste(image, (0, header))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill="black", font=font)
    draw.text(
        (8, image.height + header + 8),
        f"raw min={raw_min:.6g}, max={raw_max:.6g}{transform_note}",
        fill="black",
        font=font,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _render_geometry(path: Path, anode: np.ndarray, cathode: np.ndarray, title: str) -> None:
    h, w = anode.shape
    rgb = np.full((h, w, 3), 255, dtype=np.uint8)
    rgb[anode] = np.array([220, 45, 45], dtype=np.uint8)
    rgb[cathode] = np.array([45, 85, 220], dtype=np.uint8)
    image = Image.fromarray(rgb, "RGB")
    max_side = 1100
    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        image = image.resize(
            (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
            Image.Resampling.NEAREST,
        )
    canvas = Image.new("RGB", (image.width, image.height + 46), "white")
    canvas.paste(image, (0, 46))
    ImageDraw.Draw(canvas).text((8, 8), title, fill="black", font=ImageFont.load_default())
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _overview(path: Path, image_paths: Sequence[Path]) -> None:
    images = [Image.open(p).convert("RGB") for p in image_paths]
    tile_w, tile_h = 520, 520
    canvas = Image.new("RGB", (3 * tile_w, 2 * tile_h), "white")
    for i, image in enumerate(images[:6]):
        image.thumbnail((tile_w - 10, tile_h - 10), Image.Resampling.LANCZOS)
        x = (i % 3) * tile_w + (tile_w - image.width) // 2
        y = (i // 3) * tile_h + (tile_h - image.height) // 2
        canvas.paste(image, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def write_snapshot_artifacts(
    folder: Path,
    tag: str,
    *,
    temperature: np.ndarray,
    current_x: np.ndarray,
    current_y: np.ndarray,
    current_magnitude: np.ndarray,
    potential: np.ndarray,
    progress: np.ndarray,
    joule_heat: np.ndarray,
    anode: np.ndarray,
    cathode: np.ndarray,
    domain_mm: float,
    time_s: float,
    initial_temperature_K: float,
    semantics: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    n = int(temperature.shape[0])
    if any(np.asarray(x).shape != (n, n) for x in (
        current_x, current_y, current_magnitude, potential, progress, joule_heat, anode, cathode
    )):
        raise ValueError("Representative snapshot fields must share one square grid")
    x_mm = (np.arange(n, dtype=np.float64) + 0.5) * float(domain_mm) / n
    npz_name = f"fields_{tag}.npz"
    np.savez_compressed(
        folder / npz_name,
        temperature_K=np.asarray(temperature, np.float32),
        current_density_x_A_per_m2=np.asarray(current_x, np.float32),
        current_density_y_A_per_m2=np.asarray(current_y, np.float32),
        current_density_magnitude_A_per_m2=np.asarray(current_magnitude, np.float32),
        potential_V=np.asarray(potential, np.float32),
        global_progress=np.asarray(progress, np.float32),
        joule_heat_W_per_m3=np.asarray(joule_heat, np.float32),
        anode_mask=np.asarray(anode, np.uint8),
        cathode_mask=np.asarray(cathode, np.uint8),
        x_mm=x_mm.astype(np.float32),
        y_mm=x_mm.astype(np.float32),
        time_s=np.asarray(float(time_s), np.float64),
        domain_mm=np.asarray(float(domain_mm), np.float64),
    )

    names = {
        "geometry": f"geometry_{tag}.png",
        "temperature": f"temperature_{tag}.png",
        "current": f"current_{tag}.png",
        "potential": f"potential_{tag}.png",
        "progress": f"progress_{tag}.png",
    }
    _render_geometry(folder / names["geometry"], anode, cathode, f"geometry | t={time_s:.6g} s")
    _render_scalar(folder / names["temperature"], temperature, f"temperature [K] | t={time_s:.6g} s", "temperature")
    _render_scalar(
        folder / names["current"],
        current_magnitude,
        f"|J| [A/m^2] | t={time_s:.6g} s",
        "current",
        log_nonnegative=True,
    )
    _render_scalar(folder / names["potential"], potential, f"potential [V] | t={time_s:.6g} s", "potential")
    _render_scalar(
        folder / names["progress"],
        progress,
        f"global progress [-] | t={time_s:.6g} s",
        "progress",
        fixed_range=(0.0, 1.0),
    )
    overview_name = "overview_panel.png" if tag == "eval" else f"overview_{tag}_panel.png"
    _overview(
        folder / overview_name,
        [folder / names[k] for k in ("geometry", "temperature", "current", "potential", "progress")],
    )

    metrics = _field_metrics(
        np.asarray(temperature, np.float64),
        np.asarray(current_magnitude, np.float64),
        np.asarray(progress, np.float64),
        float(initial_temperature_K),
    )
    artifact = {
        "npz": npz_name,
        "images": {**names, "overview": overview_name},
        "time_s": float(time_s),
        "state_semantics": semantics,
        "solver_storage_dtype": "float64",
        "archive_storage_dtype": "float32",
        "grid_size": n,
        "domain_mm": float(domain_mm),
        "current_density_definition": (
            "production compute_current() evaluated from the saved native state/potential; "
            "includes configured conductive and diffusion-current terms"
        ),
    }
    return artifact, metrics


def _diagnostic_arrays(state, geometry, config, composition):
    import torch
    from ecsp_v6.physics.bc_global import bc_transport_fields
    from ecsp_v6.physics.electrochem import compute_current

    spacing = float(geometry.domain_size_m) / int(geometry.grid_size)
    with torch.inference_mode():
        transport = bc_transport_fields(state, geometry, config, composition)
        electrical = compute_current(
            state["potential"], state, transport, geometry, config, spacing
        )
    return {
        "temperature": _numpy2(state["temperature"]),
        "potential": _numpy2(state["potential"]),
        "progress": _numpy2(state["globalProgress"]),
        "current_x": _numpy2(electrical["Jx"]),
        "current_y": _numpy2(electrical["Jy"]),
        "current_magnitude": _numpy2(electrical["Jmagnitude"]),
        "joule_heat": _numpy2(electrical["jouleHeat_W_per_m3"]),
    }


def save_representative_field_artifacts(
    items,
    rows,
    output: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    composition,
    grid_size: int,
    domain_size_m: float,
) -> list[dict[str, Any]]:
    final_fields = output.get("finalFields")
    onset_fields = output.get("onsetFields")
    if not isinstance(final_fields, Mapping):
        raise RuntimeError("Representative field capture requested but native finalFields are missing")
    initial_temperature = float(config["bcGlobal"]["thermal"]["initialTemperature_K"])
    domain_mm = float(domain_size_m) * 1000.0
    records = []

    for i, (item, row) in enumerate(zip(items, rows)):
        folder = Path(item[3])
        geometry, anode, cathode = _make_geometry(
            item, int(grid_size), float(domain_size_m), _lane(final_fields["temperature_K"], i).device
        )
        artifacts: dict[str, Any] = {}
        metrics: dict[str, Any] = {}

        eval_state = _state_from_final(final_fields, i)
        eval_arrays = _diagnostic_arrays(eval_state, geometry, dict(config), composition)
        ignited = bool(row.get("ignitionSucceeded", False))
        eval_semantics = (
            "preflame_state_frozen_at_first_onset_and_held_to_evaluation_time"
            if ignited
            else "full_horizon_preflame_state_at_evaluation_time"
        )
        eval_artifact, eval_metrics = write_snapshot_artifacts(
            folder,
            "eval",
            **eval_arrays,
            anode=anode,
            cathode=cathode,
            domain_mm=domain_mm,
            time_s=float(row.get("evaluationTime_s", config["bcGlobal"]["evaluationTime_s"])),
            initial_temperature_K=initial_temperature,
            semantics=eval_semantics,
        )
        artifacts["evaluation"] = eval_artifact
        metrics["evaluation"] = eval_metrics

        if ignited:
            if not isinstance(onset_fields, Mapping):
                raise RuntimeError("Ignited reference has no onsetFields")
            onset_state = _state_from_onset(onset_fields, i)
            onset_arrays = _diagnostic_arrays(onset_state, geometry, dict(config), composition)
            onset_time = row.get("ignitionDelay_s")
            if onset_time is None or not math.isfinite(float(onset_time)):
                raise RuntimeError("Ignited reference has no finite ignitionDelay_s")
            onset_artifact, onset_metrics = write_snapshot_artifacts(
                folder,
                "onset",
                **onset_arrays,
                anode=anode,
                cathode=cathode,
                domain_mm=domain_mm,
                time_s=float(onset_time),
                initial_temperature_K=initial_temperature,
                semantics="first_condensed_phase_onset_state",
            )
            artifacts["onset"] = onset_artifact
            metrics["onset"] = onset_metrics

        manifest = {
            "schema": "ecsp-representative-fields-v1",
            "geometry_id": str(item[2].get("geometry_id")),
            "snapshots": artifacts,
            "metrics": metrics,
            "full_time_history_saved": False,
            "physics_or_solver_modified_by_capture": False,
        }
        _write_json(folder / "field_manifest.json", manifest)
        flat = {
            "representativeFieldArtifacts": artifacts,
            "representativeFieldMetrics": metrics,
            "representativeFieldSchema": manifest["schema"],
        }
        if "evaluation" in metrics:
            em = metrics["evaluation"]
            flat.update({
                "temperatureStdAtEvaluationTime_K": em["temperatureStd_K"],
                "temperatureP95MinusP05AtEvaluationTime_K": em["temperatureP95MinusP05_K"],
                "temperatureRiseCVAtEvaluationTime": em["temperatureRiseCV"],
                "currentDensityCVAtEvaluationTime": em["currentDensityCV"],
                "currentDensityP99AtEvaluationTime_A_per_m2": em["currentDensityP99_A_per_m2"],
            })
        records.append(flat)
    return records
