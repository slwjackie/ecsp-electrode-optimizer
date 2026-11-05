"""Plan angle requests without sending impossible electrode contacts to physics.

The planner dispatches to the square or 12 x 25 mm rectangular rasterizer.
``inclination_limits`` is a separate analytical calculation; raster validation
is performed for every generated case.
"""
from __future__ import annotations

import copy
import math
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np

from electrode_inclination_geometry import make_inclined_case, validate_inclined_spec
from electrode_rectangular_geometry import (
    RECTANGULAR_SCHEMA, make_rectangular_case, validate_rectangular_spec,
)

DEFAULT_INCLINATION_ANGLES = tuple(range(0, 31, 5))


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def inclination_limits(domain_width_mm: float, domain_height_mm: float,
                       anode_width_mm: float, cathode_width_mm: float,
                       central_gap_mm: float, minimum_gap_mm: float = 1.,
                       minimum_normal_width_mm: float = 1.) -> dict:
    """Continuous angle limits for fixed-horizontal-width, inward chevrons.

    The full projected height is H, so the end gap is g - H*tan(theta),
    while an arm's normal width is w*cos(theta). The returned maximum is
    inclusive for the specified positive minimum gap/width. A Cartesian mask
    can have a stricter admissibility limit and needs its own validation.
    """
    names = ("domain_width_mm", "domain_height_mm", "anode_width_mm",
             "cathode_width_mm", "central_gap_mm", "minimum_gap_mm",
             "minimum_normal_width_mm")
    values = dict(zip(names, (_finite_number(value, name) for name, value in zip(names,
        (domain_width_mm, domain_height_mm, anode_width_mm, cathode_width_mm,
         central_gap_mm, minimum_gap_mm, minimum_normal_width_mm)))))
    if any(value <= 0 for value in values.values()):
        raise ValueError("Domain dimensions, widths, central gap and minima must be positive")
    W, H, wa, wc, gap, gmin, wmin = (values[name] for name in names)
    if wa + wc + gap > W:
        raise ValueError("Electrode widths and central gap do not fit inside the domain width")
    if gap < gmin or min(wa, wc) < wmin:
        raise ValueError("Even the straight control violates the minimum gap or normal width")
    gap_limit = math.degrees(math.atan((gap - gmin) / H))
    a_limit = math.degrees(math.acos(wmin / wa))
    c_limit = math.degrees(math.acos(wmin / wc))
    return dict(**values, maximum_inclination_deg=min(gap_limit, a_limit, c_limit),
                gap_limit_deg=gap_limit, anode_width_limit_deg=a_limit,
                cathode_width_limit_deg=c_limit,
                contact_angle_deg=math.degrees(math.atan(gap / H)),
                contact_angle_is_exclusive=True,
                geometry_only=True, raster_checked=False)


def _case_id(angle: float) -> str:
    if angle.is_integer():
        return f"ANGLE_{int(angle):02d}"
    # repr is round-trip exact, so distinct floats cannot alias one case ID.
    token = repr(angle).replace(".", "p").replace("-", "m").replace("+", "p")
    return f"ANGLE_{token}"


def plan_inclination_sweep(template: Mapping[str, Any],
                           angles: Sequence[float] | None = None) -> tuple[dict, list[dict]]:
    """Return a valid study and a report including every feasible/rejected request.

    A 0-degree control is inserted automatically. Duplicate requested values
    are deduplicated; distinct angles with identical raster masks are reported
    as rejected. Bad numeric input is an error, not a rejected physical design.
    At least one nonzero angle must survive, as required by the study contract.
    """
    rectangular = isinstance(template, Mapping) and template.get("schema") == RECTANGULAR_SCHEMA
    validator = validate_rectangular_spec if rectangular else validate_inclined_spec
    make_case = make_rectangular_case if rectangular else make_inclined_case
    result = validator(template)
    requested = DEFAULT_INCLINATION_ANGLES if angles is None else angles
    if isinstance(requested, (str, bytes, Mapping)):
        raise ValueError("angles must be a sequence of finite numbers")
    try:
        requested = list(requested)
    except TypeError as exc:
        raise ValueError("angles must be a sequence of finite numbers") from exc
    if not requested:
        raise ValueError("At least one nonzero inclination angle must be requested")
    numeric = [_finite_number(angle, "inclination_deg") for angle in requested]
    if any(not 0 <= angle < 90 for angle in numeric):
        raise ValueError("inclination_deg must satisfy 0 <= angle < 90")
    numeric = sorted(set(numeric) | {0.})
    controls = [row for row in result["cases"] if row["inclination_deg"] == 0]
    controls.sort(key=lambda row: (row["anode_width_mm"], row["cathode_width_mm"]))
    domain_width = result["domain_width_mm"] if rectangular else result["domain_mm"]
    domain_height = result["domain_height_mm"] if rectangular else result["electrode_length_mm"]
    accepted, reports = [], []
    for control in controls:
        wa, wc, gap = (control[key] for key in
                       ("anode_width_mm", "cathode_width_mm", "gap_mm"))
        limits = inclination_limits(domain_width, domain_height, wa, wc, gap)
        masks = set()
        for angle in numeric:
            row = copy.deepcopy(control)
            sid = f"W{wa:g}_{_case_id(angle)}" if rectangular else _case_id(angle)
            row.update(id=sid, inclination_deg=angle)
            report = dict(inclination_deg=angle, geometry_id=row["id"],
                          status="rejected", reason="", central_gap_mm=gap,
                          end_gap_mm=gap - domain_height * math.tan(math.radians(angle)),
                          anode_normal_width_mm=wa * math.cos(math.radians(angle)),
                          cathode_normal_width_mm=wc * math.cos(math.radians(angle)),
                          maximum_analytical_inclination_deg=limits["maximum_inclination_deg"],
                          raster_minimum_gap_mm=None)
            if rectangular:
                report.update(anode_width_mm=wa, cathode_width_mm=wc,
                              domain_width_mm=domain_width, domain_height_mm=domain_height)
            try:
                a, c, meta = make_case(row, result["grid_size"])
                key = (np.packbits(a).tobytes(), np.packbits(c).tobytes())
                if key in masks:
                    raise ValueError("Identical raster mask to an accepted angle; reduce grid spacing or increase angle separation")
                masks.add(key)
            except ValueError as exc:
                report["reason"] = str(exc)
            else:
                accepted.append(meta["design_case"])
                report.update(status="accepted", raster_minimum_gap_mm=meta["raster_minimum_gap_mm"])
            reports.append(report)
    if not any(row["inclination_deg"] > 0 for row in accepted):
        raise ValueError("No feasible nonzero inclination angle remains; inspect dimensions, angle range and grid spacing")
    result["cases"] = accepted
    return validator(result), reports
