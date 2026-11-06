"""Vary central separation while holding both endpoint gaps and widths fixed.

The propellant is 12 mm wide and 25 mm high. Each electrode has a fixed
2 mm horizontal cross-section; its contact area is always 50 mm². The two
2 mm endpoint gaps fix the endpoint footprints at x=[3,5] and [7,9] mm.
Only the centre vertices move. Inclination is a derived quantity, not input.
"""
from __future__ import annotations

import copy
import math
from numbers import Real
import re
from typing import Any, Mapping

import numpy as np


CENTRAL_GAP_SCHEMA = "fixed_end_gap_central_sweep_v1"
DOMAIN_WIDTH_MM = 12.0
DOMAIN_HEIGHT_MM = 25.0
ELECTRODE_WIDTH_MM = 2.0
END_GAP_MM = 2.0
CENTRAL_GAPS_MM = (2.0, 4.0, 6.0, 8.0)
PLACEMENT = "fixed_end_gap_mirrored_pair"
_CASE_KEYS = {"id", "central_gap_mm", "groups"}


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _grid(n: int) -> tuple[float, int]:
    if isinstance(n, bool) or not isinstance(n, int) or n < 100 or n % 50:
        raise ValueError("grid_size (rows) must be a multiple of 50, >=100 (recommended 200)")
    h = DOMAIN_HEIGHT_MM / n
    nx = round(DOMAIN_WIDTH_MM / h)
    if not math.isclose(nx * h, DOMAIN_WIDTH_MM, abs_tol=1e-12, rel_tol=0.):
        raise ValueError("Rectangular width must be exactly representable with square cells")
    return h, nx


def _case(row: Mapping[str, Any], n: int) -> tuple[dict, list[float], list[int]]:
    h, _ = _grid(n)
    if not isinstance(row, Mapping) or set(row) != _CASE_KEYS:
        raise ValueError("Central-gap cases require only id, central_gap_mm and groups")
    sid = row["id"]
    if not isinstance(sid, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,79}", sid):
        raise ValueError("Central-gap case ID must be a safe, nonempty identifier")
    if row["groups"] != ["central_gap"]:
        raise ValueError("Central-gap cases require groups: [central_gap]")
    gap = _number(row["central_gap_mm"], "central_gap_mm")
    if gap not in CENTRAL_GAPS_MM:
        raise ValueError("central_gap_mm must be one of 2, 4, 6 or 8 mm")
    centre_x = DOMAIN_WIDTH_MM / 2
    edges = [centre_x - gap / 2 - ELECTRODE_WIDTH_MM, centre_x - gap / 2,
             centre_x + gap / 2, centre_x + gap / 2 + ELECTRODE_WIDTH_MM]
    indices = [round(edge / h) for edge in edges]
    if any(not math.isclose(index * h, edge, abs_tol=1e-12, rel_tol=0.)
           for index, edge in zip(indices, edges)):
        raise ValueError("Central contact edges must lie on cell faces")
    return dict(id=sid, central_gap_mm=gap, groups=["central_gap"]), edges, indices


def make_central_gap_case(row: Mapping[str, Any], n: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Rasterize a requested centre gap without changing area or endpoint positions."""
    case, edges, indices = _case(row, n)
    h, nx = _grid(n)
    gap = case["central_gap_mm"]
    half = DOMAIN_HEIGHT_MM / 2
    displacement = (gap - END_GAP_MM) / 2
    slope = (gap - END_GAP_MM) / DOMAIN_HEIGHT_MM
    theta = math.degrees(math.atan(slope))
    distance_to_centre = np.abs(np.arange(n, dtype=float) - (n - 1) / 2) * h
    ideal_shift = distance_to_centre * slope
    shifts = np.floor(ideal_shift / h + 0.5).astype(np.int64)
    a_start, a_end = indices[0] + shifts, indices[1] + shifts
    c_start, c_end = indices[2] - shifts, indices[3] - shifts
    gap_cells = c_start - a_end
    if (np.any(a_start < 0) or np.any(c_start < 0) or
            np.any(a_end > nx) or np.any(c_end > nx)):
        raise ValueError("Central-gap raster leaves the rectangular domain")
    if np.any(np.abs(np.diff(shifts)) >= min(indices[1] - indices[0], indices[3] - indices[2])):
        raise ValueError("Central-gap raster loses four-connectivity; refine the grid")
    raster_gap = float(gap_cells.min() * h)
    if raster_gap < END_GAP_MM - 1e-12:
        raise ValueError("Central-gap raster leaves less than the fixed 2 mm minimum gap")
    columns = np.arange(nx)[None, :]
    anode = (columns >= a_start[:, None]) & (columns < a_end[:, None])
    cathode = (columns >= c_start[:, None]) & (columns < c_end[:, None])

    def polygon(x0: float, x1: float, delta: float) -> list[list[float]]:
        return [[x0 + delta, 0.], [x1 + delta, 0.], [x1, half],
                [x1 + delta, DOMAIN_HEIGHT_MM], [x0 + delta, DOMAIN_HEIGHT_MM],
                [x0, half]]

    error = float(np.max(np.abs(shifts * h - ideal_shift)) + slope * h / 2)
    normal_width = ELECTRODE_WIDTH_MM / math.sqrt(1 + slope * slope)
    area = DOMAIN_HEIGHT_MM * ELECTRODE_WIDTH_MM
    meta = dict(
        schema=CENTRAL_GAP_SCHEMA, geometry_id=case["id"],
        source_role="independent_parameter_study", groups=["central_gap"],
        design_case=copy.deepcopy(case),
        domain_mm=DOMAIN_HEIGHT_MM, domain_scalar_reference="vertical_extent",
        domain_width_mm=DOMAIN_WIDTH_MM, domain_height_mm=DOMAIN_HEIGHT_MM,
        grid_size=n, grid_rows=n, grid_columns=nx, grid_spacing_mm=h,
        grid_convention="cell_centred_equal_spacing_H_over_Ny",
        electrode_length_mm=DOMAIN_HEIGHT_MM, vertical_extent_mm=DOMAIN_HEIGHT_MM,
        anode_width_mm=ELECTRODE_WIDTH_MM, cathode_width_mm=ELECTRODE_WIDTH_MM,
        gap_mm=gap, central_gap_mm=gap, end_gap_mm=END_GAP_MM,
        inclination_deg=theta, inclination_reference="vertical_arm_direction",
        inclination_is_derived=True, independent_parameter="central_gap_mm",
        minimum_edge_gap_mm=END_GAP_MM, raster_minimum_gap_mm=raster_gap,
        end_displacement_mm=displacement,
        central_displacement_from_straight_mm=displacement,
        central_side_clearance_mm=edges[0],
        end_side_clearance_mm=(DOMAIN_WIDTH_MM - 2 * ELECTRODE_WIDTH_MM - END_GAP_MM) / 2,
        central_outer_side_clearance_mm=edges[0],
        end_outer_side_clearance_mm=(DOMAIN_WIDTH_MM - 2 * ELECTRODE_WIDTH_MM - END_GAP_MM) / 2,
        anode_normal_width_mm=normal_width, cathode_normal_width_mm=normal_width,
        electrode_centerline_length_mm=DOMAIN_HEIGHT_MM * math.sqrt(1 + slope * slope),
        width_convention="horizontal_cross_section; normal width applies away from central bend",
        placement=PLACEMENT, layout="mirrored_inward_chevrons",
        anode_polygon_mm=polygon(edges[0], edges[1], displacement),
        cathode_polygon_mm=polygon(edges[2], edges[3], -displacement),
        anode_area_mm2=area, cathode_area_mm2=area, total_electrode_area_mm2=2 * area,
        anode_to_cathode_area_ratio=1.,
        target_anode_area_fraction=ELECTRODE_WIDTH_MM / DOMAIN_WIDTH_MM,
        target_cathode_area_fraction=ELECTRODE_WIDTH_MM / DOMAIN_WIDTH_MM,
        actual_anode_area_mm2=float(anode.sum() * h * h),
        actual_cathode_area_mm2=float(cathode.sum() * h * h),
        rasterization="nearest_cell_row_translation_preserving_each_polarity_area",
        raster_max_boundary_error_mm=error, inclined_edges_exact_on_grid=(gap == END_GAP_MM),
        intended_anode_components=1, intended_cathode_components=1,
        surface_contact_model=True, hidden_bus_assumed=True,
        propellant_domain_area_mm2=DOMAIN_WIDTH_MM * DOMAIN_HEIGHT_MM,
        electrode_masks_remove_propellant=False,
    )
    return anode, cathode, meta


def validate_central_gap_spec(spec: Mapping[str, Any]) -> dict:
    """Validate a nonempty selection from the four fixed-end central-gap cases."""
    keys = {"schema", "domain_width_mm", "domain_height_mm", "electrode_length_mm",
            "electrode_width_mm", "end_gap_mm", "grid_size", "placement",
            "experimental_replicates", "cases"}
    if not isinstance(spec, Mapping) or set(spec) != keys or spec["schema"] != CENTRAL_GAP_SCHEMA:
        raise ValueError("Unknown fixed-end central-gap study schema/keys")
    fixed = {"domain_width_mm": DOMAIN_WIDTH_MM, "domain_height_mm": DOMAIN_HEIGHT_MM,
             "electrode_length_mm": DOMAIN_HEIGHT_MM, "electrode_width_mm": ELECTRODE_WIDTH_MM,
             "end_gap_mm": END_GAP_MM}
    for key, value in fixed.items():
        if _number(spec[key], key) != value:
            raise ValueError(f"Fixed-end central-gap study requires {key} = {value:g}")
    _grid(spec["grid_size"])
    if spec["placement"] != PLACEMENT:
        raise ValueError(f"Only {PLACEMENT} placement is supported")
    reps = spec["experimental_replicates"]
    if isinstance(reps, bool) or not isinstance(reps, int) or reps < 5:
        raise ValueError("experimental_replicates must be an integer >=5")
    if not isinstance(spec["cases"], list) or not spec["cases"]:
        raise ValueError("Central-gap study needs at least one requested case")
    canonical, ids, gaps = [], set(), set()
    for row in spec["cases"]:
        _, _, meta = make_central_gap_case(row, spec["grid_size"])
        case = meta["design_case"]
        if case["id"] in ids:
            raise ValueError("Duplicate central-gap case ID")
        if case["central_gap_mm"] in gaps:
            raise ValueError("Duplicate central gap produces an identical raster mask")
        ids.add(case["id"])
        gaps.add(case["central_gap_mm"])
        canonical.append(case)
    result = copy.deepcopy(dict(spec))
    result["cases"] = canonical
    return result
