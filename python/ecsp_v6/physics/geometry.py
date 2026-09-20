from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from ..config import resolve_manufacturing_component_limits
from ..manufacturability import (
    corner_radius_diagnostics,
    fixed_area_targets,
    project_geometry_constraints,
)
from ..maskio import load_geometry
from .numerics import resize_nearest_numpy


@dataclass
class GeometryBatch:
    geometry_ids: list[str]
    anode: torch.Tensor
    cathode: torch.Tensor
    fixed: torch.Tensor
    propellant: torch.Tensor
    grid_size: int
    domain_size_m: float
    minimum_gap_m: float

    @property
    def batch_size(self) -> int:
        return len(self.geometry_ids)


def _direct_contact(anode: np.ndarray, cathode: np.ndarray) -> int:
    neighborhood = ndimage.binary_dilation(anode, structure=np.ones((3, 3), dtype=bool))
    return int(np.count_nonzero(neighborhood & cathode))


def _component_count(mask: np.ndarray) -> int:
    return int(
        ndimage.label(
            np.asarray(mask, dtype=bool),
            structure=np.ones((3, 3), dtype=int),
        )[1]
    )


def _minimum_clear_distance_m(anode: np.ndarray, cathode: np.ndarray, domain_size_m: float) -> float:
    if not np.any(anode) or not np.any(cathode):
        return 0.0
    rows, columns = anode.shape
    dy = domain_size_m / rows
    dx = domain_size_m / columns
    support = ndimage.binary_dilation(
        anode, structure=np.ones((3, 3), dtype=bool)
    )
    distance = ndimage.distance_transform_edt(~support, sampling=(dy, dx))
    return float(np.min(distance[cathode]))


def load_and_resize_geometry(
    path: Path,
    grid_size: int,
    domain_size_m: float,
    minimum_gap_m: float,
    reject_resize_shorts: bool,
    manufacturability_cfg: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | bool]]:
    anode_raw, cathode_raw = load_geometry(path)
    anode = resize_nearest_numpy(anode_raw, grid_size)
    cathode = resize_nearest_numpy(cathode_raw, grid_size)
    cfg = dict(manufacturability_cfg or {})
    spacing_m = domain_size_m / grid_size
    resize_projection_diagnostics: dict[str, object] = {
        "resize_constraint_projection_applied": False,
        "resize_constraint_projection_success": True,
    }
    if bool(cfg.get("projectConstraintsAfterResize", True)):
        resize_cfg = dict(cfg)
        resize_cfg["minimumGapPixels"] = max(
            2, int(math.ceil(minimum_gap_m / max(spacing_m, 1e-15) - 1e-12))
        )
        physical_corner = float(cfg.get("minimumCornerRadius_m", 0.0))
        resize_cfg["minimumCornerRadiusPixels"] = (
            int(math.ceil(physical_corner / max(spacing_m, 1e-15) - 1e-12))
            if physical_corner > 0
            else int(cfg.get("minimumCornerRadiusPixels", 0))
        )
        resize_cfg["maximumFixedAreaBoundaryEdits"] = int(
            cfg.get("maximumFixedAreaBoundaryEditsAfterResize", 2048)
        )
        resize_corner_tolerance = float(
            cfg.get(
                "maximumCornerRoundingChangeFractionAfterResize",
                cfg.get("maximumCornerRoundingChangeFraction", 0.01),
            )
        )
        corner_radius_hard_after_resize = bool(
            cfg.get("enforceCornerRadiusAfterResize", False)
        )
        # The source design mask is the manufacturing geometry.  Re-rasterizing
        # that geometry on a Cartesian solver grid introduces staircase pixels;
        # applying the same corner filter again is therefore an audit, not a
        # second physical fillet operation.  Keep the audit threshold in a
        # separate config, while allowing the resize projection to satisfy the
        # area/component/gap constraints even when the staircase audit warns.
        corner_audit_cfg = dict(resize_cfg)
        corner_audit_cfg["maximumCornerRoundingChangeFraction"] = (
            resize_corner_tolerance
        )
        resize_cfg["maximumCornerRoundingChangeFraction"] = (
            resize_corner_tolerance if corner_radius_hard_after_resize else 1.0
        )
        resize_cfg["fastFixedAreaProjection"] = True
        anode, cathode, resize_projection_diagnostics = (
            project_geometry_constraints(anode, cathode, resize_cfg)
        )
        effective_cfg = resize_cfg
    else:
        effective_cfg = cfg
    overlap = int(np.count_nonzero(anode & cathode))
    if overlap:
        raise ValueError(f"Anode/cathode overlap after resize: {path}")
    if not np.any(anode) or not np.any(cathode):
        raise ValueError(f"Both polarities must remain nonempty after resize: {path}")
    direct_contact = _direct_contact(anode, cathode)
    minimum_distance_m = _minimum_clear_distance_m(anode, cathode, domain_size_m)
    anode_components = _component_count(anode)
    cathode_components = _component_count(cathode)
    total_components = anode_components + cathode_components
    maximum_per, maximum_total = resolve_manufacturing_component_limits(
        effective_cfg
    )
    component_limit_safe = bool(
        anode_components <= maximum_per
        and cathode_components <= maximum_per
        and total_components <= maximum_total
    )
    short_unsafe = direct_contact > 0 or (
        minimum_gap_m > 0
        and minimum_distance_m
        < minimum_gap_m
        - 10 * np.finfo(float).eps * max(minimum_gap_m, 1.0)
    )

    target = fixed_area_targets(anode.shape, effective_cfg)
    target_a = int(target["target_anode_pixels"])
    target_c = int(target["target_cathode_pixels"])
    actual_a = int(np.count_nonzero(anode))
    actual_c = int(np.count_nonzero(cathode))
    relative_tolerance = float(effective_cfg.get("resizeFixedAreaRelativeTolerance", 0.02))
    absolute_tolerance = max(2, int(math.ceil(max(target_a, target_c, 1) * relative_tolerance)))
    fixed_area_safe = bool(
        not bool(effective_cfg.get("enforceFixedTotalElectrodeArea", False))
        or (
            abs(actual_a - target_a) <= absolute_tolerance
            and abs(actual_c - target_c) <= absolute_tolerance
        )
    )
    corner_audit_cfg = locals().get("corner_audit_cfg", effective_cfg)
    corner_radius_hard_after_resize = bool(
        cfg.get("enforceCornerRadiusAfterResize", False)
    )
    anode_corner = corner_radius_diagnostics(
        anode, corner_audit_cfg, pixel_spacing_m=spacing_m
    )
    cathode_corner = corner_radius_diagnostics(
        cathode, corner_audit_cfg, pixel_spacing_m=spacing_m
    )
    # Always report the independent solver-grid staircase audit.  It is a hard
    # rejection only when explicitly enabled; the 64/128 design-mask corner
    # constraint remains hard in the manufacturing gate.
    corner_safe = bool(
        anode_corner["corner_radius_satisfied"]
        and cathode_corner["corner_radius_satisfied"]
    )
    anode_corner_change = float(anode_corner["corner_rounding_change_fraction"])
    cathode_corner_change = float(cathode_corner["corner_rounding_change_fraction"])
    resize_projection_safe = bool(
        resize_projection_diagnostics.get(
            "constraint_projection_success",
            resize_projection_diagnostics.get("resize_constraint_projection_success", True),
        )
    )
    unsafe = (
        short_unsafe
        or not component_limit_safe
        or not fixed_area_safe
        or (corner_radius_hard_after_resize and not corner_safe)
        or not resize_projection_safe
    )
    if reject_resize_shorts and unsafe:
        raise ValueError(
            f"Unsafe or constraint-violating geometry after resize: {path}; "
            f"direct_contact={direct_contact}, "
            f"minimum_center_distance_m={minimum_distance_m:.8g}, "
            f"required={minimum_gap_m:.8g}, "
            f"anode_components={anode_components}, "
            f"cathode_components={cathode_components}, "
            f"maximum_total_components={maximum_total}, "
            f"area_pixels=({actual_a},{actual_c}), "
            f"target_pixels=({target_a},{target_c}), "
            f"corner_safe={corner_safe}"
        )
    return anode, cathode, {
        "overlap_pixels": overlap,
        "direct_contact_pixels": direct_contact,
        "minimum_center_distance_m": minimum_distance_m,
        "short_circuit_free_after_resize": not short_unsafe,
        "anode_components_after_resize": anode_components,
        "cathode_components_after_resize": cathode_components,
        "total_components_after_resize": total_components,
        "component_limit_safe_after_resize": component_limit_safe,
        "anode_area_pixels_after_resize": actual_a,
        "cathode_area_pixels_after_resize": actual_c,
        "target_anode_area_pixels_after_resize": target_a,
        "target_cathode_area_pixels_after_resize": target_c,
        "fixed_area_safe_after_resize": fixed_area_safe,
        "anode_corner_change_fraction_after_resize": anode_corner_change,
        "cathode_corner_change_fraction_after_resize": cathode_corner_change,
        "corner_radius_safe_after_resize": corner_safe,
        "corner_radius_enforced_after_resize": corner_radius_hard_after_resize,
        "single_component_per_polarity_after_resize": (
            anode_components == 1 and cathode_components == 1
        ),
        **{
            f"resize_{key}": value
            for key, value in resize_projection_diagnostics.items()
        },
    }


def load_geometry_batch(
    rows,
    *,
    grid_size: int,
    domain_size_m: float,
    minimum_gap_m: float,
    reject_resize_shorts: bool,
    device: torch.device,
    manufacturability_cfg: dict | None = None,
) -> tuple[GeometryBatch, list[dict]]:
    anodes: list[np.ndarray] = []
    cathodes: list[np.ndarray] = []
    diagnostics: list[dict] = []
    ids: list[str] = []
    for _, row in rows.iterrows():
        path_value = row.get("npz_path", row.get("mat_path"))
        path = Path(str(path_value))
        anode, cathode, diag = load_and_resize_geometry(
            path,
            grid_size,
            domain_size_m,
            minimum_gap_m,
            reject_resize_shorts,
            manufacturability_cfg=manufacturability_cfg,
        )
        ids.append(str(row["geometry_id"]))
        anodes.append(anode)
        cathodes.append(cathode)
        diagnostics.append(diag)
    anode_t = torch.as_tensor(np.stack(anodes), device=device, dtype=torch.bool)
    cathode_t = torch.as_tensor(np.stack(cathodes), device=device, dtype=torch.bool)
    # Keep the shared loader's historical embedded-electrode semantics for
    # legacy/non-B/C profiles.  The corrected B/C evaluator deliberately
    # overrides these two fields after loading so only that production family
    # uses full-domain propellant plus separate surface-contact masks.
    fixed = anode_t | cathode_t
    return GeometryBatch(
        geometry_ids=ids,
        anode=anode_t,
        cathode=cathode_t,
        fixed=fixed,
        propellant=~fixed,
        grid_size=grid_size,
        domain_size_m=domain_size_m,
        minimum_gap_m=minimum_gap_m,
    ), diagnostics
