from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import csv
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .geometry import (
    GeometryLimits,
    RasterizedGeometry,
    minimum_clear_gap_pixels,
)


class BaselineGeometryError(RuntimeError):
    """Raised when a fair area-matched reference cannot be constructed."""


@dataclass(frozen=True)
class AreaMatchedStaggeredParameters:
    """Discrete geometry selected for the hidden-bus 2+2 staggered reference.

    Only the four vertical contact fingers belong to the surface-contact masks.
    The two fingers of each polarity are electrically joined by an ideal hidden
    bus outside the propellant-contact plane, so the bus contributes exactly
    zero contact area and never appears in the masks passed to the physics
    solver.
    """

    fingers_per_polarity: int
    common_finger_width_px: int
    common_finger_length_px: int
    horizontal_gap_px: int
    left_margin_px: int
    right_margin_px: int
    design_grid_size: int
    physics_grid_size: int
    target_area_fraction_per_polarity: float
    design_spacing_mm: float
    physics_spacing_mm: float
    common_finger_width_mm: float
    common_finger_length_mm: float
    design_minimum_gap_mm: float
    physics_minimum_gap_mm: float
    design_area_fraction_per_polarity: float
    physics_area_fraction_per_polarity: float
    design_vertical_overlap_fraction: float
    physics_vertical_overlap_fraction: float
    hidden_bus_contact_area_fraction: float = 0.0


def _nearest_resize(mask: np.ndarray, rows: int) -> np.ndarray:
    """Match the package's nearest-neighbour physics-grid resize exactly."""
    from ecsp_v6.physics.numerics import resize_nearest_numpy

    return np.asarray(resize_nearest_numpy(np.asarray(mask, dtype=bool), rows), dtype=bool)


def _component_count(mask: np.ndarray) -> int:
    _, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    return int(count)


def _component_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return component boxes as ``(y0, y1_exclusive, x0, x1_exclusive)``."""
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    boxes: list[tuple[int, int, int, int]] = []
    for label in range(1, int(count) + 1):
        yy, xx = np.nonzero(labels == label)
        if yy.size == 0:
            continue
        boxes.append((int(yy.min()), int(yy.max()) + 1, int(xx.min()), int(xx.max()) + 1))
    boxes.sort(key=lambda box: box[2])
    return boxes


def _minimum_gap_mm(
    anode: np.ndarray, cathode: np.ndarray, spacing_mm: float
) -> float:
    return minimum_clear_gap_pixels(anode, cathode) * float(spacing_mm)


def _perimeter(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask & ~ndimage.binary_erosion(mask)))


def _build_hidden_bus_vertical_staggered_masks(
    *,
    grid_size: int,
    finger_width_px: int,
    finger_length_px: int,
    horizontal_gap_px: int,
    fingers_per_polarity: int = 2,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Build the requested top-view staggered pattern.

    The visible/contact order from left to right is ``A, C, A, C``.  Anode
    fingers start at the top boundary and have exactly equal lengths.  Cathode
    fingers start at the bottom boundary and have exactly equal lengths.  No
    horizontal bus is rasterised: the buses are ideal out-of-plane/backside
    conductors and therefore have no propellant contact area.
    """

    n = int(grid_size)
    width = int(finger_width_px)
    length = int(finger_length_px)
    gap = int(horizontal_gap_px)
    count = int(fingers_per_polarity)
    if count != 2:
        raise BaselineGeometryError(
            "The v7.9.2 reference is fixed to exactly two anode and two cathode "
            "surface-contact fingers."
        )
    if min(n, width, length, gap) <= 0 or length > n:
        raise BaselineGeometryError("invalid hidden-bus staggered dimensions")

    total_fingers = 2 * count
    used_width = total_fingers * width + (total_fingers - 1) * gap
    if used_width > n:
        raise BaselineGeometryError("four staggered fingers do not fit horizontally")
    remaining = n - used_width
    left_margin = remaining // 2
    right_margin = remaining - left_margin

    anode = np.zeros((n, n), dtype=bool)
    cathode = np.zeros((n, n), dtype=bool)
    x_starts = [left_margin + slot * (width + gap) for slot in range(total_fingers)]
    for slot, x0 in enumerate(x_starts):
        if slot % 2 == 0:
            anode[0:length, x0 : x0 + width] = True
        else:
            cathode[n - length : n, x0 : x0 + width] = True
    return anode, cathode, left_margin, right_margin


def _equal_rectangular_fingers(
    mask: np.ndarray,
    *,
    expected_count: int,
    boundary: str,
) -> bool:
    boxes = _component_boxes(mask)
    if len(boxes) != int(expected_count):
        return False
    dimensions = {(y1 - y0, x1 - x0) for y0, y1, x0, x1 in boxes}
    if len(dimensions) != 1:
        return False
    for y0, y1, x0, x1 in boxes:
        component = mask[y0:y1, x0:x1]
        if not np.all(component):
            return False
        if boundary == "top" and y0 != 0:
            return False
        if boundary == "bottom" and y1 != mask.shape[0]:
            return False
    return True


def _vertical_overlap_fraction(anode: np.ndarray, cathode: np.ndarray) -> float:
    a_rows = np.flatnonzero(np.any(anode, axis=1))
    c_rows = np.flatnonzero(np.any(cathode, axis=1))
    if a_rows.size == 0 or c_rows.size == 0:
        return 0.0
    first = max(int(a_rows.min()), int(c_rows.min()))
    last = min(int(a_rows.max()), int(c_rows.max()))
    overlap_rows = max(0, last - first + 1)
    return float(overlap_rows) / float(anode.shape[0])


def generate_area_matched_staggered(
    limits: GeometryLimits,
    *,
    physics_grid_size: int,
    target_area_fraction_per_polarity: float | None = None,
    fingers_per_polarity: int = 2,
    target_interdigitation_overlap_fraction: float = 0.50,
    minimum_interdigitation_overlap_fraction: float = 0.45,
    maximum_gap_safety_pixels: int = 8,
) -> tuple[dict[str, Any], RasterizedGeometry, AreaMatchedStaggeredParameters]:
    """Create the requested area-matched hidden-bus 2+2 staggered reference.

    This reference is intentionally *not* a connected surface comb.  Its four
    visible rectangles are the only regions touching propellant.  The two
    anode fingers share one hidden terminal bus outside the contact plane, and
    the two cathode fingers share another.  The masks therefore contain two
    connected components per polarity while representing one electrical
    terminal per polarity.

    Geometry selection uses no ECSP objective.  A deterministic integer search
    chooses a common width and common length for all four fingers, an A-C-A-C
    horizontal ordering, and a safe horizontal gap.  Candidates must satisfy
    area, gap, equal-length/equal-width and deep-interdigitation checks on both
    the design and physics grids.
    """

    count = int(fingers_per_polarity)
    if count != 2:
        raise BaselineGeometryError(
            "area-matched hidden-bus staggered requires fingers_per_polarity=2"
        )
    target = float(
        limits.target_area_fraction_per_polarity
        if target_area_fraction_per_polarity is None
        else target_area_fraction_per_polarity
    )
    if not 0.0 < target < 0.5:
        raise BaselineGeometryError(
            "target_area_fraction_per_polarity must lie strictly between 0 and 0.5"
        )
    target_overlap = float(target_interdigitation_overlap_fraction)
    minimum_overlap = float(minimum_interdigitation_overlap_fraction)
    if not 0.0 <= minimum_overlap <= target_overlap <= 1.0:
        raise BaselineGeometryError(
            "interdigitation overlap fractions must satisfy "
            "0 <= minimum <= target <= 1"
        )

    n = int(limits.grid_size)
    p = int(physics_grid_size)
    if n < 9 or p < 9:
        raise BaselineGeometryError("design and physics grids must be at least 9x9")
    design_dx = float(limits.domain_mm) / n
    physics_dx = float(limits.domain_mm) / p
    minimum_width_px = max(1, int(math.ceil(float(limits.minimum_width_mm) / design_dx)))
    maximum_width_px = int(math.floor(float(limits.maximum_width_mm) / design_dx))
    if maximum_width_px < minimum_width_px:
        raise BaselineGeometryError(
            "no raster width satisfies the configured minimum/maximum electrode width"
        )
    nominal_gap_px = max(1, int(math.ceil(float(limits.minimum_gap_mm) / design_dx)))
    minimum_side_margin_px = max(0, int(math.ceil(float(limits.margin_mm) / design_dx)))
    target_pixels_per_polarity = target * n * n

    candidates: list[
        tuple[
            tuple[float, ...],
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            AreaMatchedStaggeredParameters,
        ]
    ] = []

    for width_px in range(minimum_width_px, maximum_width_px + 1):
        required_length = target_pixels_per_polarity / max(count * width_px, 1)
        length_options = {
            int(math.floor(required_length)) + offset
            for offset in range(-3, 4)
        }
        length_options.update(
            {
                int(round(required_length)),
                int(math.ceil(required_length)),
            }
        )
        for length_px in sorted(length_options):
            if length_px <= 0 or length_px > n:
                continue
            for gap_px in range(
                nominal_gap_px,
                nominal_gap_px + max(1, int(maximum_gap_safety_pixels)) + 1,
            ):
                used_width = 4 * width_px + 3 * gap_px
                if used_width > n:
                    continue
                remaining = n - used_width
                left_margin = remaining // 2
                right_margin = remaining - left_margin
                if min(left_margin, right_margin) < minimum_side_margin_px:
                    continue

                anode, cathode, left_margin, right_margin = (
                    _build_hidden_bus_vertical_staggered_masks(
                        grid_size=n,
                        finger_width_px=width_px,
                        finger_length_px=length_px,
                        horizontal_gap_px=gap_px,
                        fingers_per_polarity=count,
                    )
                )
                if np.any(anode & cathode):
                    continue
                if np.any(
                    ndimage.binary_dilation(
                        anode, structure=np.ones((3, 3), dtype=bool)
                    )
                    & cathode
                ):
                    continue
                if not _equal_rectangular_fingers(
                    anode, expected_count=count, boundary="top"
                ):
                    continue
                if not _equal_rectangular_fingers(
                    cathode, expected_count=count, boundary="bottom"
                ):
                    continue

                design_gap = _minimum_gap_mm(anode, cathode, design_dx)
                design_area = float(anode.mean())
                design_area_error = abs(design_area - target) / target
                design_overlap = _vertical_overlap_fraction(anode, cathode)
                if design_gap + 1.0e-12 < limits.minimum_gap_mm:
                    continue
                if design_area_error > limits.area_tolerance_fraction + 1.0e-12:
                    continue
                if design_overlap + 1.0e-12 < minimum_overlap:
                    continue

                physics_anode = _nearest_resize(anode, p)
                physics_cathode = _nearest_resize(cathode, p)
                if np.any(physics_anode & physics_cathode):
                    continue
                if np.any(
                    ndimage.binary_dilation(
                        physics_anode, structure=np.ones((3, 3), dtype=bool)
                    )
                    & physics_cathode
                ):
                    continue
                if not _equal_rectangular_fingers(
                    physics_anode, expected_count=count, boundary="top"
                ):
                    continue
                if not _equal_rectangular_fingers(
                    physics_cathode, expected_count=count, boundary="bottom"
                ):
                    continue
                # Require all four resized contact fingers to remain identical.
                physics_dims = [
                    (y1 - y0, x1 - x0)
                    for y0, y1, x0, x1 in (
                        _component_boxes(physics_anode) + _component_boxes(physics_cathode)
                    )
                ]
                if len(set(physics_dims)) != 1:
                    continue

                physics_gap = _minimum_gap_mm(
                    physics_anode, physics_cathode, physics_dx
                )
                physics_area = float(physics_anode.mean())
                physics_area_error = abs(physics_area - target) / target
                physics_overlap = _vertical_overlap_fraction(
                    physics_anode, physics_cathode
                )
                if physics_gap + 1.0e-12 < limits.minimum_gap_mm:
                    continue
                if physics_area_error > limits.area_tolerance_fraction + 1.0e-12:
                    continue
                if physics_overlap + 1.0e-12 < minimum_overlap:
                    continue

                parameters = AreaMatchedStaggeredParameters(
                    fingers_per_polarity=count,
                    common_finger_width_px=width_px,
                    common_finger_length_px=length_px,
                    horizontal_gap_px=gap_px,
                    left_margin_px=left_margin,
                    right_margin_px=right_margin,
                    design_grid_size=n,
                    physics_grid_size=p,
                    target_area_fraction_per_polarity=target,
                    design_spacing_mm=design_dx,
                    physics_spacing_mm=physics_dx,
                    common_finger_width_mm=width_px * design_dx,
                    common_finger_length_mm=length_px * design_dx,
                    design_minimum_gap_mm=design_gap,
                    physics_minimum_gap_mm=physics_gap,
                    design_area_fraction_per_polarity=design_area,
                    physics_area_fraction_per_polarity=physics_area,
                    design_vertical_overlap_fraction=design_overlap,
                    physics_vertical_overlap_fraction=physics_overlap,
                    hidden_bus_contact_area_fraction=0.0,
                )
                # Area matching is primary.  Among comparably area-matched masks,
                # select the target deep overlap, smallest safe gap excess and the
                # most symmetric side margins.  No physics objective is consulted.
                maximum_area_error = float(
                    max(design_area_error, physics_area_error)
                )
                overlap_error = float(
                    abs(0.5 * (design_overlap + physics_overlap) - target_overlap)
                )
                score = (
                    # Preserve a genuinely area-matched reference while preventing
                    # tiny raster-area differences from driving the fingers far
                    # away from the requested ~50% deep interdigitation.
                    maximum_area_error + 0.05 * overlap_error,
                    maximum_area_error,
                    overlap_error,
                    float(physics_gap - limits.minimum_gap_mm),
                    float(abs(left_margin - right_margin)),
                    float(-length_px),
                    float(width_px),
                )
                candidates.append(
                    (
                        score,
                        anode,
                        cathode,
                        physics_anode,
                        physics_cathode,
                        parameters,
                    )
                )

    if not candidates:
        minimum_width_needed = target * float(limits.domain_mm) / 2.0
        raise BaselineGeometryError(
            "No two-anode/two-cathode hidden-bus staggered geometry satisfies "
            "the current area, width, gap, side-margin and deep-overlap constraints "
            "on both grids. The four contact fingers are fixed by the requested "
            "reference topology and are never replaced by a connected comb. "
            f"domain={limits.domain_mm:g} mm, target={target:g} per polarity, "
            f"width=[{limits.minimum_width_mm:g}, {limits.maximum_width_mm:g}] mm, "
            f"gap>={limits.minimum_gap_mm:g} mm, minimum overlap={minimum_overlap:g}. "
            "As a continuous lower-bound check, each of two full-height fingers "
            f"would need width about {minimum_width_needed:g} mm."
        )

    candidates.sort(key=lambda item: item[0])
    _, anode, cathode, physics_anode, physics_cathode, params = candidates[0]
    area_error = abs(float(anode.mean()) - target) / target
    imbalance = abs(float(anode.mean()) - float(cathode.mean())) / max(
        float(anode.mean() + cathode.mean()), 1.0e-12
    )
    yy, xx = np.nonzero(anode | cathode)
    spatial_dispersion = (
        float((np.std(xx) + np.std(yy)) / (2.0 * n)) if len(xx) else 0.0
    )

    violation_details = {
        "overlap_pixels": 0.0,
        "direct_contact_pixels": 0.0,
        "minimum_gap_shortfall_mm": 0.0,
        "anode_component_excess": 0.0,
        "cathode_component_excess": 0.0,
        "total_component_excess": 0.0,
        "anode_component_count_mismatch": 0.0,
        "cathode_component_count_mismatch": 0.0,
        "branch_count_excess": 0.0,
        "branch_depth_excess": 0.0,
        "minimum_width_shortfall_mm": 0.0,
        "maximum_width_excess_mm": 0.0,
        "area_relative_error_excess": max(
            0.0, area_error - limits.area_tolerance_fraction
        ),
        "polarity_area_imbalance_excess": max(
            0.0, imbalance - limits.area_tolerance_fraction
        ),
        "minimum_interdigitation_overlap_shortfall": max(
            0.0,
            minimum_overlap
            - min(
                params.design_vertical_overlap_fraction,
                params.physics_vertical_overlap_fraction,
            ),
        ),
        "hidden_bus_contact_pixels": 0.0,
    }
    descriptors = {
        "n_line": 4.0,
        "n_arc": 0.0,
        "n_branch": 0.0,
        "branch_depth": 0.0,
        "segment_tree_depth": 1.0,
        "n_components": 4.0,
        "anode_components": 2.0,
        "cathode_components": 2.0,
        "intended_anode_components": 2.0,
        "intended_cathode_components": 2.0,
        "electrical_terminal_components": 2.0,
        "electrical_terminal_components_per_polarity": 1.0,
        "length_mean": float(params.common_finger_length_mm),
        "length_std": 0.0,
        "arc_radius_mean": 0.0,
        "orientation_entropy": 0.0,
        "minimum_gap_mm": float(params.design_minimum_gap_mm),
        "perimeter_px": float(_perimeter(anode) + _perimeter(cathode)),
        "spatial_dispersion": spatial_dispersion,
        "anode_area_fraction": float(anode.mean()),
        "cathode_area_fraction": float(cathode.mean()),
        "total_contact_area_fraction": float(anode.mean() + cathode.mean()),
        "area_imbalance_fraction": imbalance,
        "anode_width_calibration_scale": 1.0,
        "cathode_width_calibration_scale": 1.0,
        "minimum_effective_width_mm": float(params.common_finger_width_mm),
        "maximum_effective_width_mm": float(params.common_finger_width_mm),
        "propellant_domain_area_fraction": 1.0,
        "baseline_fingers_per_polarity": 2.0,
        "baseline_common_finger_width_mm": float(params.common_finger_width_mm),
        "baseline_common_finger_length_mm": float(params.common_finger_length_mm),
        "baseline_vertical_overlap_fraction": float(
            params.design_vertical_overlap_fraction
        ),
        "baseline_hidden_bus_contact_area_fraction": 0.0,
        "physics_preview_anode_area_fraction": float(physics_anode.mean()),
        "physics_preview_cathode_area_fraction": float(physics_cathode.mean()),
        "physics_preview_minimum_gap_mm": float(params.physics_minimum_gap_mm),
        "physics_preview_vertical_overlap_fraction": float(
            params.physics_vertical_overlap_fraction
        ),
    }

    geometry_id = "AREA_MATCHED_STAGGERED"
    anode_boxes = _component_boxes(anode)
    cathode_boxes = _component_boxes(cathode)

    def component_record(
        polarity: str,
        box: tuple[int, int, int, int],
        index: int,
        boundary: str,
    ) -> dict[str, Any]:
        y0, y1, x0, x1 = box
        return {
            "type": "VERTICAL_CONTACT_FINGER",
            "polarity": polarity,
            "finger_index": index,
            "boundary_origin": boundary,
            "x_start_px": x0,
            "x_stop_px_exclusive": x1,
            "y_start_px": y0,
            "y_stop_px_exclusive": y1,
            "width_mm": params.common_finger_width_mm,
            "length_mm": params.common_finger_length_mm,
            "hidden_bus_id": f"{polarity}_hidden_bus",
            "hidden_bus_in_contact_mask": False,
            "children": [],
        }

    genome: dict[str, Any] = {
        "geometry_id": geometry_id,
        "topology_id": "BASELINE_HIDDEN_BUS_STAGGERED_2A2C",
        "baseline_type": "area_matched_staggered",
        "baseline_layout_revision": "v7.9.2_hidden_bus_vertical_2a2c",
        "source_role": "post_optimization_reference_only",
        "included_in_nsga2_population": False,
        "selection_rule": (
            "fixed A-C-A-C vertical contact order; exactly two equal anode fingers "
            "from the top and two equal cathode fingers from the bottom; hidden "
            "buses outside the propellant contact plane; geometry-only calibration "
            "of common width/length to match the recommended contact area"
        ),
        "domain_mm": float(limits.domain_mm),
        "target_area_fraction_per_polarity": target,
        "minimum_gap_mm": float(limits.minimum_gap_mm),
        "minimum_width_mm": float(limits.minimum_width_mm),
        "maximum_width_mm": float(limits.maximum_width_mm),
        "parameters": params.__dict__,
        "anode": {
            "components": [
                component_record("anode", box, index, "top")
                for index, box in enumerate(anode_boxes)
            ]
        },
        "cathode": {
            "components": [
                component_record("cathode", box, index, "bottom")
                for index, box in enumerate(cathode_boxes)
            ]
        },
        "electrode_mask_semantics": "surface_contact_overlay",
        "visible_contact_components_per_polarity": 2,
        "electrical_terminal_components_per_polarity": 1,
        "hidden_bus_assumption": (
            "same-polarity fingers are electrically connected by an ideal "
            "out-of-plane/backside bus outside the propellant-contact mask"
        ),
        "hidden_bus_in_contact_mask": False,
        "hidden_bus_contact_area_fraction": 0.0,
        "hidden_bus_pixels_in_contact_mask": 0,
        "horizontal_contact_order": ["anode", "cathode", "anode", "cathode"],
        "equal_length_within_each_polarity": True,
        "equal_width_for_all_four_fingers": True,
        "deep_interdigitation": True,
    }
    raster = RasterizedGeometry(
        anode_mask=anode,
        cathode_mask=cathode,
        descriptors=descriptors,
        constraint_violation=0.0,
        violation_details=violation_details,
    )
    return genome, raster, params

def write_objective_comparison_csv(
    path: Path,
    recommended: Mapping[str, Any],
    baseline: Mapping[str, Any],
    objective_names: tuple[str, ...],
) -> None:
    rows = []
    for role, data in (("recommended_ai", recommended), ("area_matched_staggered", baseline)):
        row = {
            "role": role,
            "geometry_id": data.get("geometry_id"),
            "topology_id": data.get("topology_id"),
            "constraint_violation": data.get("constraint_violation"),
            "ignition_success": data.get("ignition_success"),
        }
        objectives = data.get("objectives", {})
        if isinstance(objectives, Mapping):
            for name in objective_names:
                row[name] = objectives.get(name)
        rows.append(row)
    keys = list(rows[0].keys()) + [
        key for key in rows[1] if key not in rows[0]
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def write_comparison_image(
    output_path: Path,
    recommended_png: Path,
    baseline_png: Path,
    recommended: Mapping[str, Any],
    baseline: Mapping[str, Any],
    objective_names: tuple[str, ...],
) -> None:
    """Write a compact side-by-side geometry and scalar-metric comparison."""
    if not recommended_png.is_file() or not baseline_png.is_file():
        return
    left = Image.open(recommended_png).convert("RGB").resize((420, 420))
    right = Image.open(baseline_png).convert("RGB").resize((420, 420))
    canvas = Image.new("RGB", (940, 650), "white")
    canvas.paste(left, (35, 75))
    canvas.paste(right, (485, 75))
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(28)
    header_font = _load_font(21)
    body_font = _load_font(17)
    draw.text((35, 20), "Recommended AI vs Area-Matched Staggered", fill="black", font=title_font)
    draw.text((150, 48), "Recommended AI", fill="black", font=header_font)
    draw.text((565, 48), "Area-matched staggered", fill="black", font=header_font)

    y = 520
    rec_obj = recommended.get("objectives", {})
    base_obj = baseline.get("objectives", {})
    labels = {
        "ignition_delay_s": "Ignition delay [s]",
        "area_undecomposed_fraction_at_evaluation_time": (
            "Undecomposed at configured evaluation time [-]"
        ),
        "area_undecomposed_fraction_at_2s": (
            "Undecomposed at configured evaluation time [-] (deprecated At2s alias)"
        ),
        "minimum_ignition_voltage_V": "Minimum ignition voltage [V]",
        "current_congestion": "Current congestion [-]",
    }
    for name in objective_names:
        rv = rec_obj.get(name) if isinstance(rec_obj, Mapping) else None
        bv = base_obj.get(name) if isinstance(base_obj, Mapping) else None
        rtext = "n/a" if rv is None else f"{float(rv):.6g}"
        btext = "n/a" if bv is None else f"{float(bv):.6g}"
        draw.text((45, y), labels.get(name, name), fill="black", font=body_font)
        draw.text((420, y), rtext, fill="black", font=body_font, anchor="ra")
        draw.text((870, y), btext, fill="black", font=body_font, anchor="ra")
        y += 28
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def strict_json_dump(path: Path, data: Mapping[str, Any]) -> None:
    def clean(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(data), indent=2, allow_nan=False), encoding="utf-8")
