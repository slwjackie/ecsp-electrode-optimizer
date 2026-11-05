"""Equal-area inclined contacts on a genuine 12 mm wide x 25 mm high domain.

The horizontal contact width is selectable as 2 or 4 mm for both polarities.
Areas remain constant within each width group; changing width changes area.
The masks contain only the rectangular propellant domain, without padding.
"""
from __future__ import annotations

import copy
import math
from numbers import Real
import re
from typing import Any, Mapping

import numpy as np

RECTANGULAR_SCHEMA = "rectangular_inclined_strip_study_v1"
DOMAIN_WIDTH_MM = 12.0
DOMAIN_HEIGHT_MM = 25.0
_CASE_KEYS = {"id", "anode_width_mm", "cathode_width_mm", "gap_mm",
              "inclination_deg", "groups"}


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _grid(n: int) -> tuple[float, int]:
    if isinstance(n, bool) or not isinstance(n, int) or n < 100 or n % 50:
        raise ValueError("grid_size (rows) must be a multiple of 50, >=100 (recommended 200)")
    h = DOMAIN_HEIGHT_MM / n
    columns = round(DOMAIN_WIDTH_MM / h)
    if not math.isclose(columns * h, DOMAIN_WIDTH_MM, abs_tol=1e-12, rel_tol=0):
        raise ValueError("Rectangular width must be exactly representable with square cells")
    return h, columns


def _case(row: Mapping[str, Any], n: int) -> tuple[dict, list[int]]:
    h, _ = _grid(n)
    if not isinstance(row, Mapping) or set(row) != _CASE_KEYS:
        raise ValueError("Invalid rectangular inclination case keys")
    sid = row["id"]
    if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,79}", sid):
        raise ValueError("Rectangular case ID must be a safe, nonempty identifier")
    if row["groups"] != ["inclination"]:
        raise ValueError("Rectangular cases require groups: [inclination]")
    values = {key: _number(row[key], key) for key in
              ("anode_width_mm", "cathode_width_mm", "gap_mm", "inclination_deg")}
    wa, wc, gap, theta = (values[key] for key in
                         ("anode_width_mm", "cathode_width_mm", "gap_mm", "inclination_deg"))
    if wa not in {2., 4.} or wc != wa:
        raise ValueError("Both electrodes must have the same selectable horizontal width: 2 or 4 mm")
    if gap != DOMAIN_WIDTH_MM - wa - wc:
        raise ValueError("Edge-anchored central gap must equal 12 - anode_width - cathode_width")
    if not 0 <= theta < 90:
        raise ValueError("inclination_deg must satisfy 0 <= angle < 90")
    if wa * math.cos(math.radians(theta)) < 1.0 - 1e-12:
        raise ValueError("Inclined arm normal width must be at least the study's 1 mm minimum")
    if gap - DOMAIN_HEIGHT_MM * math.tan(math.radians(theta)) < 1.0 - 1e-12:
        raise ValueError("Inclination leaves less than the study's 1 mm analytical end gap")
    edges = [0., wa, DOMAIN_WIDTH_MM - wc, DOMAIN_WIDTH_MM]
    indices = [round(edge / h) for edge in edges]
    if any(not math.isclose(index * h, edge, abs_tol=1e-12, rel_tol=0)
           for index, edge in zip(indices, edges)):
        raise ValueError("Central contact edges must lie on cell faces")
    return dict(id=sid, **values, groups=["inclination"]), indices


def make_rectangular_case(row: Mapping[str, Any], n: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Rasterize fixed-area mirrored contacts using equal dx and dy on Ny x Nx."""
    case, indices = _case(row, n)
    h, columns_count = _grid(n)
    wa, wc, gap, theta = (case[key] for key in
                         ("anode_width_mm", "cathode_width_mm", "gap_mm", "inclination_deg"))
    slope = math.tan(math.radians(theta))
    half = DOMAIN_HEIGHT_MM / 2
    delta = half * slope
    distances = np.abs(np.arange(n, dtype=float) - (n - 1) / 2) * h
    ideal_displacements = distances * slope
    shifts = np.floor(ideal_displacements / h + 0.5).astype(np.int64)
    a_start, a_end = indices[0] + shifts, indices[1] + shifts
    c_start, c_end = indices[2] - shifts, indices[3] - shifts
    gap_cells = c_start - a_end
    if np.any(gap_cells < 1):
        raise ValueError("Inclined raster needs an empty gap cell; refine grid or reduce angle")
    if (np.any(a_start < 0) or np.any(c_start < 0) or
        np.any(a_end > columns_count) or np.any(c_end > columns_count)):
        raise ValueError("Inclined raster leaves the rectangular domain")
    if np.any(np.abs(np.diff(shifts)) >= min(indices[1] - indices[0], indices[3] - indices[2])):
        raise ValueError("Inclined raster loses four-connectivity; refine grid or increase width")
    columns = np.arange(columns_count)[None, :]
    a = (columns >= a_start[:, None]) & (columns < a_end[:, None])
    c = (columns >= c_start[:, None]) & (columns < c_end[:, None])

    def polygon(x0: float, x1: float, displacement: float) -> list[list[float]]:
        return [[x0 + displacement, 0.], [x1 + displacement, 0.],
                [x1, half], [x1 + displacement, DOMAIN_HEIGHT_MM],
                [x0 + displacement, DOMAIN_HEIGHT_MM], [x0, half]]

    end_gap = gap - 2 * delta
    raster_gap = float(np.min(gap_cells) * h)
    if raster_gap < 1.0 - 1e-12:
        raise ValueError("Inclined raster leaves less than the study's 1 mm minimum gap")
    error = float(np.max(np.abs(shifts * h - ideal_displacements)) + slope * h / 2)
    meta = dict(schema=RECTANGULAR_SCHEMA, geometry_id=case["id"],
        source_role="independent_parameter_study", groups=["inclination"],
        design_case=copy.deepcopy(case),
        # Legacy scalar keys refer to the vertical extent/row count only.
        domain_mm=DOMAIN_HEIGHT_MM, domain_scalar_reference="vertical_extent",
        domain_width_mm=DOMAIN_WIDTH_MM, domain_height_mm=DOMAIN_HEIGHT_MM,
        grid_size=n, grid_rows=n, grid_columns=columns_count,
        grid_spacing_mm=h, grid_convention="cell_centred_equal_spacing_H_over_Ny",
        electrode_length_mm=DOMAIN_HEIGHT_MM, vertical_extent_mm=DOMAIN_HEIGHT_MM,
        anode_width_mm=wa, cathode_width_mm=wc, gap_mm=gap,
        inclination_deg=theta, inclination_reference="vertical_arm_direction",
        central_gap_mm=gap, end_gap_mm=end_gap,
        minimum_edge_gap_mm=end_gap, raster_minimum_gap_mm=raster_gap,
        end_displacement_mm=delta,
        anode_normal_width_mm=wa * math.cos(math.radians(theta)),
        cathode_normal_width_mm=wc * math.cos(math.radians(theta)),
        electrode_centerline_length_mm=DOMAIN_HEIGHT_MM / math.cos(math.radians(theta)),
        width_convention="horizontal_cross_section; normal width applies away from central bend",
        placement="edge_anchored_pair", layout="mirrored_inward_chevrons",
        anode_polygon_mm=polygon(0., wa, delta),
        cathode_polygon_mm=polygon(DOMAIN_WIDTH_MM - wc, DOMAIN_WIDTH_MM, -delta),
        anode_area_mm2=DOMAIN_HEIGHT_MM * wa, cathode_area_mm2=DOMAIN_HEIGHT_MM * wc,
        total_electrode_area_mm2=DOMAIN_HEIGHT_MM * (wa + wc),
        anode_to_cathode_area_ratio=wa / wc,
        target_anode_area_fraction=wa / DOMAIN_WIDTH_MM,
        target_cathode_area_fraction=wc / DOMAIN_WIDTH_MM,
        actual_anode_area_mm2=float(a.sum() * h * h),
        actual_cathode_area_mm2=float(c.sum() * h * h),
        rasterization="nearest_cell_row_translation_preserving_each_polarity_area",
        raster_max_boundary_error_mm=error,
        inclined_edges_exact_on_grid=(theta == 0),
        intended_anode_components=1, intended_cathode_components=1,
        surface_contact_model=True, hidden_bus_assumed=True,
        propellant_domain_area_mm2=DOMAIN_WIDTH_MM * DOMAIN_HEIGHT_MM,
        electrode_masks_remove_propellant=False)
    return a, c, meta


def validate_rectangular_spec(spec: Mapping[str, Any]) -> dict:
    """Validate either width group or both, each anchored by its own 0-degree control."""
    keys = {"schema", "domain_width_mm", "domain_height_mm", "electrode_length_mm",
            "grid_size", "placement", "experimental_replicates", "cases"}
    if not isinstance(spec, Mapping) or set(spec) != keys or spec["schema"] != RECTANGULAR_SCHEMA:
        raise ValueError("Unknown rectangular inclination study schema/keys")
    if (_number(spec["domain_width_mm"], "domain_width_mm") != DOMAIN_WIDTH_MM or
        _number(spec["domain_height_mm"], "domain_height_mm") != DOMAIN_HEIGHT_MM or
        _number(spec["electrode_length_mm"], "electrode_length_mm") != DOMAIN_HEIGHT_MM):
        raise ValueError("Rectangular study requires width 12 mm, height 25 mm and vertical extent 25 mm")
    _grid(spec["grid_size"])
    if spec["placement"] != "edge_anchored_pair":
        raise ValueError("Only edge_anchored_pair placement is supported")
    reps = spec["experimental_replicates"]
    if isinstance(reps, bool) or not isinstance(reps, int) or reps < 5:
        raise ValueError("experimental_replicates must be an integer >=5")
    rows = spec["cases"]
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("Rectangular study needs straight controls and at least one inclined case")
    canonical, masks = [], set()
    for row in rows:
        a, c, meta = make_rectangular_case(row, spec["grid_size"])
        key = (np.packbits(a).tobytes(), np.packbits(c).tobytes())
        if key in masks:
            raise ValueError("Rectangular cases produce identical raster masks; refine grid or increase angle separation")
        masks.add(key)
        canonical.append(meta["design_case"])
    if len({row["id"] for row in canonical}) != len(canonical):
        raise ValueError("Duplicate rectangular case ID")
    for width in {row["anode_width_mm"] for row in canonical}:
        angles = [row["inclination_deg"] for row in canonical if row["anode_width_mm"] == width]
        if angles.count(0.) != 1:
            raise ValueError("Each selected width needs exactly one 0 degree straight control")
    if not any(row["inclination_deg"] > 0 for row in canonical):
        raise ValueError("Rectangular study needs at least one inclined case")
    result = copy.deepcopy(dict(spec))
    result["cases"] = canonical
    return result
