"""Area-preserving chevron contacts for the opt-in 25 mm inclination study.

``width`` is horizontal width, not width normal to an inclined arm. Keeping it
fixed preserves contact area (25 * width) while the central edge gap is fixed.
The analytical drawing and the cell mask are separately described: inclined
edges cannot in general be represented exactly by a Cartesian Boolean mask.
"""
from __future__ import annotations

import copy
import math
from numbers import Real
import re
from typing import Any, Mapping

import numpy as np

INCLINED_SCHEMA = "inclined_strip_study_v1"
DOMAIN_MM = 25.0
_CASE_KEYS = {"id", "anode_width_mm", "cathode_width_mm", "gap_mm",
              "inclination_deg", "groups"}


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _grid(n: int) -> float:
    if isinstance(n, bool) or not isinstance(n, int) or n < 100 or n % 50:
        raise ValueError("grid_size must be a multiple of 50, >=100 (recommended 200)")
    return DOMAIN_MM / n


def _case(row: Mapping[str, Any], n: int) -> tuple[dict, list[int]]:
    h = _grid(n)
    if not isinstance(row, Mapping) or set(row) != _CASE_KEYS:
        raise ValueError("Invalid inclination case keys")
    sid = row["id"]
    if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,79}", sid):
        raise ValueError("Inclination case ID must be a safe, nonempty identifier")
    if row["groups"] != ["inclination"]:
        raise ValueError("Inclination cases require groups: [inclination]")
    values = {key: _number(row[key], key) for key in
              ("anode_width_mm", "cathode_width_mm", "gap_mm", "inclination_deg")}
    wa, wc, gap, theta = (values[key] for key in
                         ("anode_width_mm", "cathode_width_mm", "gap_mm", "inclination_deg"))
    if wa <= 0 or wc <= 0 or gap <= 0 or wa + wc + gap > DOMAIN_MM:
        raise ValueError("Positive widths/gap must fit inside the 25 mm domain")
    if not 0 <= theta < 90:
        raise ValueError("inclination_deg must satisfy 0 <= angle < 90")
    if min(wa, wc) * math.cos(math.radians(theta)) < 1.0 - 1e-12:
        raise ValueError("Inclined arm normal width must be at least the study's 1 mm minimum")
    if gap - DOMAIN_MM * math.tan(math.radians(theta)) < 1.0 - 1e-12:
        raise ValueError("Inclination leaves less than the study's 1 mm analytical end gap")
    left = (DOMAIN_MM - wa - wc - gap) / 2
    edges = [left, left + wa, left + wa + gap, left + wa + gap + wc]
    indices = [round(edge / h) for edge in edges]
    if any(not math.isclose(index * h, edge, abs_tol=1e-12, rel_tol=0)
           for index, edge in zip(indices, edges)):
        raise ValueError("Central contact edges must lie on cell faces; use another grid or dimensions")
    if indices[1] == indices[0] or indices[3] == indices[2]:
        raise ValueError("Electrode widths must occupy at least one cell")
    return dict(id=sid, **values, groups=["inclination"]), indices


def make_inclined_case(row: Mapping[str, Any], n: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Rasterize two mirrored bent strips without changing either contact area.

    Every row retains its exact integer width. A common nonnegative rounded
    displacement moves the anode right and cathode left, ensuring left/right
    symmetry for equal widths even when displacement is exactly half a cell.
    """
    case, indices = _case(row, n)
    h = _grid(n)
    wa, wc, gap, theta = (case[key] for key in
                         ("anode_width_mm", "cathode_width_mm", "gap_mm", "inclination_deg"))
    slope = math.tan(math.radians(theta))
    half = DOMAIN_MM / 2
    delta = half * slope
    # This expression gives identical floating-point distances for mirrored rows.
    distances = np.abs(np.arange(n, dtype=float) - (n - 1) / 2) * h
    ideal_displacements = distances * slope
    shifts = np.floor(ideal_displacements / h + 0.5).astype(np.int64)
    a_start, a_end = indices[0] + shifts, indices[1] + shifts
    c_start, c_end = indices[2] - shifts, indices[3] - shifts
    gap_cells = c_start - a_end
    if np.any(gap_cells < 1):
        raise ValueError("Inclined raster needs at least one empty gap cell; refine grid or reduce angle")
    if (np.any(a_start < 0) or np.any(c_start < 0) or
        np.any(a_end > n) or np.any(c_end > n)):
        raise ValueError("Inclined raster leaves the domain")
    # Diagonal-only contacts are not connected in the solver's four-neighbour grid.
    if np.any(np.abs(np.diff(shifts)) >= min(indices[1] - indices[0], indices[3] - indices[2])):
        raise ValueError("Inclined raster loses four-connectivity; refine grid or increase width")
    columns = np.arange(n)[None, :]
    a = (columns >= a_start[:, None]) & (columns < a_end[:, None])
    c = (columns >= c_start[:, None]) & (columns < c_end[:, None])
    left = (DOMAIN_MM - wa - wc - gap) / 2
    edges = [left, left + wa, left + wa + gap, left + wa + gap + wc]

    def polygon(x0: float, x1: float, displacement: float) -> list[list[float]]:
        return [[x0 + displacement, 0.], [x1 + displacement, 0.],
                [x1, half], [x1 + displacement, DOMAIN_MM],
                [x0 + displacement, DOMAIN_MM], [x0, half]]

    end_gap = gap - 2 * delta
    raster_gap = float(np.min(gap_cells) * h)
    if raster_gap < 1.0 - 1e-12:
        raise ValueError("Inclined raster leaves less than the study's 1 mm minimum gap")
    # Bound the horizontal error over complete cell rows, not only row centres.
    error = float(np.max(np.abs(shifts * h - ideal_displacements)) + slope * h / 2)
    meta = dict(schema=INCLINED_SCHEMA, geometry_id=case["id"],
        source_role="independent_parameter_study", groups=["inclination"],
        design_case=copy.deepcopy(case), domain_mm=DOMAIN_MM, grid_size=n,
        grid_spacing_mm=h, grid_convention="cell_centred_L_over_N",
        electrode_length_mm=DOMAIN_MM, vertical_extent_mm=DOMAIN_MM,
        anode_width_mm=wa, cathode_width_mm=wc, gap_mm=gap,
        inclination_deg=theta, inclination_reference="vertical_arm_direction",
        central_gap_mm=gap, end_gap_mm=end_gap,
        minimum_edge_gap_mm=end_gap, raster_minimum_gap_mm=raster_gap,
        end_displacement_mm=delta,
        anode_normal_width_mm=wa * math.cos(math.radians(theta)),
        cathode_normal_width_mm=wc * math.cos(math.radians(theta)),
        electrode_centerline_length_mm=DOMAIN_MM / math.cos(math.radians(theta)),
        width_convention="horizontal_cross_section; normal width applies away from central bend",
        placement="centered_assembly", layout="mirrored_inward_chevrons",
        anode_polygon_mm=polygon(edges[0], edges[1], delta),
        cathode_polygon_mm=polygon(edges[2], edges[3], -delta),
        anode_area_mm2=DOMAIN_MM * wa, cathode_area_mm2=DOMAIN_MM * wc,
        total_electrode_area_mm2=DOMAIN_MM * (wa + wc),
        anode_to_cathode_area_ratio=wa / wc,
        target_anode_area_fraction=wa / DOMAIN_MM,
        target_cathode_area_fraction=wc / DOMAIN_MM,
        actual_anode_area_mm2=float(a.sum() * h * h),
        actual_cathode_area_mm2=float(c.sum() * h * h),
        rasterization="nearest_cell_row_translation_preserving_each_polarity_area",
        raster_max_boundary_error_mm=error,
        inclined_edges_exact_on_grid=(theta == 0),
        intended_anode_components=1, intended_cathode_components=1,
        surface_contact_model=True, hidden_bus_assumed=True,
        propellant_domain_area_mm2=DOMAIN_MM**2,
        electrode_masks_remove_propellant=False)
    return a, c, meta


def validate_inclined_spec(spec: Mapping[str, Any]) -> dict:
    """Validate a matched-area sweep with one straight control and bent variants."""
    keys = {"schema", "domain_mm", "electrode_length_mm", "grid_size",
            "placement", "experimental_replicates", "cases"}
    if not isinstance(spec, Mapping) or set(spec) != keys or spec["schema"] != INCLINED_SCHEMA:
        raise ValueError("Unknown inclination study schema/keys")
    if (_number(spec["domain_mm"], "domain_mm") != DOMAIN_MM or
        _number(spec["electrode_length_mm"], "electrode_length_mm") != DOMAIN_MM):
        raise ValueError("Inclination study requires 25 x 25 mm domain and 25 mm vertical extent")
    _grid(spec["grid_size"])
    if spec["placement"] != "centered_assembly":
        raise ValueError("Only centered_assembly placement is supported")
    reps = spec["experimental_replicates"]
    if isinstance(reps, bool) or not isinstance(reps, int) or reps < 5:
        raise ValueError("experimental_replicates must be an integer >=5")
    rows = spec["cases"]
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("Inclination study needs a straight control and at least one inclined case")
    canonical = []
    masks = set()
    for row in rows:
        a, c, meta = make_inclined_case(row, spec["grid_size"])
        key = (np.packbits(a).tobytes(), np.packbits(c).tobytes())
        if key in masks:
            raise ValueError("Inclination cases produce identical raster masks; reduce grid spacing or increase angle separation")
        masks.add(key)
        canonical.append(meta["design_case"])
    if len({row["id"] for row in canonical}) != len(canonical):
        raise ValueError("Duplicate inclination case ID")
    if len({row["inclination_deg"] for row in canonical}) != len(canonical):
        raise ValueError("Duplicate inclination angle")
    if sum(row["inclination_deg"] == 0 for row in canonical) != 1:
        raise ValueError("Exactly one 0 degree straight control is required")
    dimensions = {(row["anode_width_mm"], row["cathode_width_mm"], row["gap_mm"])
                  for row in canonical}
    if len(dimensions) != 1:
        raise ValueError("All inclination cases must preserve widths, contact areas and central gap")
    result = copy.deepcopy(dict(spec))
    result["cases"] = canonical
    return result
