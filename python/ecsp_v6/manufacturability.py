from __future__ import annotations

import math
from collections import deque

import numpy as np
from scipy import ndimage

from .config import resolve_manufacturing_component_limits

_EIGHT_CONNECTED = np.ones((3, 3), dtype=int)


def _component_sizes(mask: np.ndarray) -> list[int]:
    labels, count = ndimage.label(mask, structure=_EIGHT_CONNECTED)
    return [int(np.count_nonzero(labels == index)) for index in range(1, count + 1)]


def _minimum_gap(anode: np.ndarray, cathode: np.ndarray) -> float:
    if not np.any(anode) or not np.any(cathode):
        return 0.0
    anode_support = ndimage.binary_dilation(anode, structure=_EIGHT_CONNECTED)
    distance = ndimage.distance_transform_edt(~anode_support)
    return float(np.min(distance[cathode]))


def _minimum_width_proxy(mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    distance = ndimage.distance_transform_edt(mask)
    return float(2 * np.percentile(distance[mask], 10))


def _direct_contact_pixels(anode: np.ndarray, cathode: np.ndarray) -> int:
    # Includes edge and corner contact; a valid design must retain propellant
    # between polarities, not merely avoid exact pixel overlap.
    neighborhood = ndimage.binary_dilation(anode, structure=_EIGHT_CONNECTED, iterations=1)
    return int(np.count_nonzero(neighborhood & cathode))




def _boundary_zone(shape: tuple[int, int], boundary: str, contact_pixels: int) -> np.ndarray:
    zone = np.zeros(shape, dtype=bool)
    width = max(1, int(contact_pixels))
    if boundary == "left":
        zone[:, :width] = True
    elif boundary == "right":
        zone[:, -width:] = True
    elif boundary == "top":
        zone[-width:, :] = True
    elif boundary == "bottom":
        zone[:width, :] = True
    else:
        raise ValueError(f"Unsupported bus boundary: {boundary}")
    return zone


def _all_components_touch_boundary(
    mask: np.ndarray, boundary: str, contact_pixels: int
) -> bool:
    labels, count = ndimage.label(mask, structure=_EIGHT_CONNECTED)
    if count == 0:
        return False
    zone = _boundary_zone(mask.shape, boundary, contact_pixels)
    return all(np.any((labels == index) & zone) for index in range(1, count + 1))


def _connectivity_diagnostics(
    anode: np.ndarray, cathode: np.ndarray, cfg: dict, a_components: int, c_components: int
) -> dict:
    """Make the electrical-bus assumption explicit and test it when possible."""
    mode = str(cfg.get("electrodeConnectivityMode", "implicit_3d_bus"))
    documented = bool(cfg.get("implicit3DBusDocumented", False))
    contact_pixels = int(cfg.get("busContactPixels", 1))
    if mode == "implicit_3d_bus":
        a_connected = documented
        c_connected = documented
        assumption = (
            "all 2D islands are connected through a propellant-noncontacting "
            "depth-direction/3D bus"
        )
    elif mode == "boundary_bus":
        a_boundary = str(cfg.get("anodeBusBoundary", "left"))
        c_boundary = str(cfg.get("cathodeBusBoundary", "right"))
        a_connected = _all_components_touch_boundary(anode, a_boundary, contact_pixels)
        c_connected = _all_components_touch_boundary(cathode, c_boundary, contact_pixels)
        assumption = f"all components touch configured 2D buses ({a_boundary}/{c_boundary})"
    elif mode == "single_component":
        a_connected = a_components == 1
        c_connected = c_components == 1
        assumption = "one connected component per polarity"
    else:
        raise ValueError(f"Unsupported electrodeConnectivityMode: {mode}")
    return {
        "electrode_connectivity_mode": mode,
        "implicit_3d_bus_documented": documented,
        "anode_bus_connected": bool(a_connected),
        "cathode_bus_connected": bool(c_connected),
        "bus_connectivity_assumption": assumption,
    }

def _remove_small_components(mask: np.ndarray, minimum_pixels: int) -> np.ndarray:
    labels, count = ndimage.label(mask, structure=_EIGHT_CONNECTED)
    if count == 0:
        return np.asarray(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= max(1, minimum_pixels)
    keep[0] = False
    return keep[labels]


def _limit_components(mask: np.ndarray, maximum_components: int) -> np.ndarray:
    labels, count = ndimage.label(mask, structure=_EIGHT_CONNECTED)
    if count <= maximum_components:
        return mask.astype(bool)
    sizes = [(index, int(np.count_nonzero(labels == index))) for index in range(1, count + 1)]
    keep_labels = {index for index, _ in sorted(sizes, key=lambda item: item[1], reverse=True)[:maximum_components]}
    return np.isin(labels, list(keep_labels))



def _limit_total_components(
    anode: np.ndarray,
    cathode: np.ndarray,
    maximum_total_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the globally largest components while retaining both polarities."""
    anode = np.asarray(anode, dtype=bool)
    cathode = np.asarray(cathode, dtype=bool)
    max_total = max(2, int(maximum_total_components))
    a_labels, a_count = ndimage.label(anode, structure=_EIGHT_CONNECTED)
    c_labels, c_count = ndimage.label(cathode, structure=_EIGHT_CONNECTED)
    if a_count + c_count <= max_total:
        return anode.copy(), cathode.copy()

    records: list[tuple[int, str, int]] = []
    for index in range(1, a_count + 1):
        records.append((int(np.count_nonzero(a_labels == index)), "a", index))
    for index in range(1, c_count + 1):
        records.append((int(np.count_nonzero(c_labels == index)), "c", index))

    keep_a: set[int] = set()
    keep_c: set[int] = set()
    if a_count:
        keep_a.add(max((r for r in records if r[1] == "a"), key=lambda r: r[0])[2])
    if c_count:
        keep_c.add(max((r for r in records if r[1] == "c"), key=lambda r: r[0])[2])

    remaining = sorted(records, key=lambda r: r[0], reverse=True)
    for _, polarity, label in remaining:
        if len(keep_a) + len(keep_c) >= max_total:
            break
        if polarity == "a":
            keep_a.add(label)
        else:
            keep_c.add(label)
    return np.isin(a_labels, list(keep_a)), np.isin(c_labels, list(keep_c))


def _disk_structure(radius_pixels: int) -> np.ndarray:
    radius = max(0, int(radius_pixels))
    if radius == 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def _round_mask(mask: np.ndarray, radius_pixels: int) -> np.ndarray:
    """Round raster corners without deleting narrow electrode branches.

    A disk opening is a valid rolling-ball test, but it erases every branch
    narrower than twice the requested radius.  That destroyed rail/finger
    topology in the fixed-area library.  Instead, diffuse the binary signed
    indicator over a radius-scaled Gaussian kernel and threshold at 0.5.  The
    operation removes square convex corner pixels and fills matching concave
    notches while preserving long narrow branches.  ``corner_radius_diagnostics``
    audits how much a candidate would still change under the same operator.
    This is a grid manufacturing proxy, not an exact CAD fillet construction.
    """
    current = np.asarray(mask, dtype=bool).copy()
    radius = max(0, int(radius_pixels))
    if radius == 0 or not np.any(current):
        return current
    sigma = max(0.8, float(radius))
    smoothed = ndimage.gaussian_filter(
        current.astype(np.float64), sigma=sigma, mode="constant", cval=0.0
    )
    rounded = smoothed >= 0.5
    # A corner operation may not remove an entire polarity.  Returning the
    # original mask lets the subsequent diagnostic reject it explicitly.
    return (rounded if np.any(rounded) else current).astype(bool)


def _corner_radius_pixels(
    shape: tuple[int, int],
    cfg: dict,
    pixel_spacing_m: float | None = None,
) -> int:
    # Prefer the physical manufacturing radius at every resolution.  When the
    # caller does not provide the spacing (base/DDPM mask paths), derive it
    # from the configured physical domain so 64x64 and 128x128 masks represent
    # the same corner radius instead of the same pixel count.
    if pixel_spacing_m is None or pixel_spacing_m <= 0:
        domain_size_m = float(cfg.get("domainSize_m", 0.0))
        if domain_size_m > 0:
            pixel_spacing_m = domain_size_m / max(shape)
    if pixel_spacing_m is not None and pixel_spacing_m > 0:
        physical = float(cfg.get("minimumCornerRadius_m", 0.0))
        if physical > 0:
            return int(math.ceil(physical / pixel_spacing_m - 1e-12))
    return max(0, int(cfg.get("minimumCornerRadiusPixels", 0)))


def corner_radius_diagnostics(
    mask: np.ndarray,
    cfg: dict,
    *,
    pixel_spacing_m: float | None = None,
) -> dict[str, float | int | bool]:
    mask = np.asarray(mask, dtype=bool)
    radius = _corner_radius_pixels(mask.shape, cfg, pixel_spacing_m)
    if radius <= 0 or not np.any(mask):
        return {
            "minimum_corner_radius_pixels": radius,
            "minimum_corner_radius_m": (
                float(radius * pixel_spacing_m)
                if pixel_spacing_m is not None
                else float(cfg.get("minimumCornerRadius_m", 0.0))
            ),
            "corner_rounding_change_fraction": 0.0,
            "corner_radius_satisfied": bool(np.any(mask)),
        }
    rounded = _round_mask(mask, radius)
    changed = int(np.count_nonzero(mask ^ rounded))
    denominator = max(int(np.count_nonzero(mask)), 1)
    change_fraction = changed / denominator
    tolerance = float(cfg.get("maximumCornerRoundingChangeFraction", 0.08))
    return {
        "minimum_corner_radius_pixels": radius,
        "minimum_corner_radius_m": (
            float(radius * pixel_spacing_m)
            if pixel_spacing_m is not None
            else float(cfg.get("minimumCornerRadius_m", 0.0))
        ),
        "corner_rounding_change_fraction": float(change_fraction),
        "corner_radius_satisfied": bool(change_fraction <= tolerance),
    }


def _fixed_area_targets(
    shape: tuple[int, int], cfg: dict
) -> tuple[int, int, int]:
    pixels = int(np.prod(shape))
    fraction = float(cfg.get("targetTotalElectrodeAreaFraction", 0.0))
    share = float(cfg.get("targetAnodeShareOfElectrodeArea", 0.5))
    if fraction <= 0:
        return 0, 0, 0
    if bool(cfg.get("enforceEqualPolarityArea", True)):
        each = max(1, int(round(0.5 * fraction * pixels)))
        return 2 * each, each, each
    total_target = max(2, int(round(fraction * pixels)))
    anode_target = min(total_target - 1, max(1, int(round(share * total_target))))
    return total_target, anode_target, total_target - anode_target


def fixed_area_targets(
    shape: tuple[int, int], cfg: dict
) -> dict[str, int | float]:
    total, anode, cathode = _fixed_area_targets(shape, cfg)
    return {
        "target_total_electrode_pixels": total,
        "target_anode_pixels": anode,
        "target_cathode_pixels": cathode,
        "target_total_electrode_area_fraction": (
            total / float(np.prod(shape)) if total else 0.0
        ),
    }


def _allowed_area(
    shape: tuple[int, int], opposite: np.ndarray, cfg: dict
) -> np.ndarray:
    allowed = np.ones(shape, dtype=bool)
    border = max(0, int(cfg.get("minimumBorderMarginPixels", 0)))
    if border:
        allowed[:border, :] = False
        allowed[-border:, :] = False
        allowed[:, :border] = False
        allowed[:, -border:] = False
    opposite = np.asarray(opposite, dtype=bool)
    if np.any(opposite):
        distance = ndimage.distance_transform_edt(~opposite)
        allowed &= distance >= float(cfg["minimumGapPixels"])
    return allowed


def _signed_distance_score(mask: np.ndarray, radius_pixels: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    inside = ndimage.distance_transform_edt(mask)
    outside = ndimage.distance_transform_edt(~mask)
    score = inside - outside
    sigma = max(0.5, float(radius_pixels) * 0.75)
    return ndimage.gaussian_filter(score, sigma=sigma)


def _top_k_mask(score: np.ndarray, allowed: np.ndarray, count: int) -> np.ndarray:
    indices = np.flatnonzero(allowed)
    if count <= 0 or len(indices) < count:
        return np.zeros_like(allowed, dtype=bool)
    values = score.ravel()[indices]
    order = np.lexsort((indices, -values))
    selected = indices[order[:count]]
    result = np.zeros_like(allowed, dtype=bool)
    result.ravel()[selected] = True
    return result


def _exact_boundary_adjustment(
    mask: np.ndarray,
    allowed: np.ndarray,
    score: np.ndarray,
    target_pixels: int,
) -> tuple[np.ndarray, int, bool]:
    """Reach the exact pixel target with one vectorized boundary edit pass."""
    current = np.asarray(mask, dtype=bool).copy()
    area = int(np.count_nonzero(current))
    edits = abs(area - int(target_pixels))
    if area < target_pixels:
        missing = target_pixels - area
        candidates = np.flatnonzero(allowed & ~current)
        if len(candidates) < missing:
            return current, edits, False
        distance = ndimage.distance_transform_edt(~current).ravel()[candidates]
        values = score.ravel()[candidates]
        # Grow nearest to the existing electrode first, then prefer high score.
        order = np.lexsort((candidates, -values, distance))
        current.ravel()[candidates[order[:missing]]] = True
    elif area > target_pixels:
        excess = area - target_pixels
        candidates = np.flatnonzero(current)
        if len(candidates) <= excess:
            return current, edits, False
        interior_distance = ndimage.distance_transform_edt(current).ravel()[candidates]
        values = score.ravel()[candidates]
        # Remove the shallowest boundary pixels first, then low-score pixels.
        order = np.lexsort((candidates, values, interior_distance))
        current.ravel()[candidates[order[:excess]]] = False
    return current, edits, int(np.count_nonzero(current)) == target_pixels


def _project_one_polarity_area(
    mask: np.ndarray,
    opposite: np.ndarray,
    target_pixels: int,
    cfg: dict,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    original = np.asarray(mask, dtype=bool)
    allowed = _allowed_area(original.shape, opposite, cfg)
    if target_pixels <= 0 or int(np.count_nonzero(allowed)) < target_pixels:
        return original.copy(), {
            "success": False,
            "failure_reason": "insufficient_allowed_pixels",
            "target_pixels": int(target_pixels),
            "actual_pixels": int(np.count_nonzero(original)),
            "exact_area_edits": 0,
        }
    radius = _corner_radius_pixels(original.shape, cfg)
    rounded_seed = _round_mask(original, radius)
    if not np.any(rounded_seed):
        rounded_seed = original.copy()
    score = _signed_distance_score(rounded_seed, radius)
    score[~allowed] = -np.inf

    # Fast path used after grid resize: source masks are already rounded and
    # fixed-area, so one smooth level-set selection plus exact correction is
    # sufficient and avoids dozens of morphology searches per candidate.
    if bool(cfg.get("fastFixedAreaProjection", False)):
        candidate = _top_k_mask(score, allowed, target_pixels)
        rounded_candidate = _round_mask(candidate, radius) & allowed
        projected, edits, exact_ok = _exact_boundary_adjustment(
            rounded_candidate, allowed, score, target_pixels
        )
        maximum_edits = int(cfg.get("maximumFixedAreaBoundaryEdits", 192))
        corner_adjustment = float(
            np.count_nonzero(projected ^ rounded_candidate)
            / max(int(target_pixels), 1)
        )
        corner_tolerance = float(
            cfg.get("maximumCornerRoundingChangeFraction", 0.08)
        )
        corner_ok = bool(corner_adjustment <= corner_tolerance)
        success = bool(
            exact_ok
            and edits <= maximum_edits
            and int(np.count_nonzero(projected)) == target_pixels
            and corner_ok
        )
        reasons: list[str] = []
        if not exact_ok:
            reasons.append("exact_area_adjustment_failed")
        if edits > maximum_edits:
            reasons.append("excessive_exact_area_boundary_edits")
        if not corner_ok:
            reasons.append("corner_radius_projection_failed")
        return projected.astype(bool), {
            "success": success,
            "failure_reason": ";".join(reasons),
            "target_pixels": int(target_pixels),
            "actual_pixels": int(np.count_nonzero(projected)),
            "exact_area_edits": int(edits),
            "minimum_corner_radius_pixels": int(radius),
            "minimum_corner_radius_m": float(
                cfg.get("minimumCornerRadius_m", 0.0)
            ),
            "corner_projection_adjustment_fraction": corner_adjustment,
            "corner_rounding_change_fraction": corner_adjustment,
            "corner_radius_satisfied": corner_ok,
        }

    # Search a pre-rounding area whose stable rounded result is closest to the
    # exact target, then perform only a small frontier correction.
    low = max(1, int(target_pixels * 0.45))
    high = min(int(np.count_nonzero(allowed)), int(target_pixels * 1.8) + 32)
    best_mask: np.ndarray | None = None
    best_difference = math.inf
    best_count = target_pixels
    examined: set[int] = set()

    def consider(count: int) -> tuple[int, int]:
        nonlocal best_mask, best_difference, best_count
        count = int(np.clip(count, 1, np.count_nonzero(allowed)))
        if count in examined:
            area = int(np.count_nonzero(best_mask)) if best_mask is not None else 0
            return count, area
        examined.add(count)
        candidate = _top_k_mask(score, allowed, count)
        candidate = _round_mask(candidate, radius) & allowed
        area = int(np.count_nonzero(candidate))
        difference = abs(area - target_pixels)
        if difference < best_difference:
            best_difference = difference
            best_mask = candidate
            best_count = count
        return count, area

    for _ in range(18):
        if low > high:
            break
        middle = (low + high) // 2
        _, area = consider(middle)
        if area < target_pixels:
            low = middle + 1
        elif area > target_pixels:
            high = middle - 1
        else:
            break
    for count in range(max(1, best_count - 20), min(int(np.count_nonzero(allowed)), best_count + 20) + 1):
        consider(count)
    if best_mask is None:
        return original.copy(), {
            "success": False,
            "failure_reason": "area_search_failed",
            "target_pixels": int(target_pixels),
            "actual_pixels": int(np.count_nonzero(original)),
            "exact_area_edits": 0,
        }

    projected, edits, exact_ok = _exact_boundary_adjustment(
        best_mask, allowed, score, target_pixels
    )
    maximum_edits = int(cfg.get("maximumFixedAreaBoundaryEdits", 32))
    corner_adjustment = float(
        np.count_nonzero(projected ^ best_mask) / max(int(target_pixels), 1)
    )
    corner_tolerance = float(
        cfg.get("maximumCornerRoundingChangeFraction", 0.08)
    )
    corner_ok = bool(corner_adjustment <= corner_tolerance)
    success = bool(
        exact_ok
        and edits <= maximum_edits
        and int(np.count_nonzero(projected)) == target_pixels
        and corner_ok
    )
    reasons: list[str] = []
    if not exact_ok:
        reasons.append("exact_area_adjustment_failed")
    if edits > maximum_edits:
        reasons.append("excessive_exact_area_boundary_edits")
    if not corner_ok:
        reasons.append("corner_radius_projection_failed")
    return projected.astype(bool), {
        "success": success,
        "failure_reason": ";".join(reasons),
        "target_pixels": int(target_pixels),
        "actual_pixels": int(np.count_nonzero(projected)),
        "exact_area_edits": int(edits),
        "minimum_corner_radius_pixels": int(radius),
        "minimum_corner_radius_m": float(cfg.get("minimumCornerRadius_m", 0.0)),
        "corner_projection_adjustment_fraction": corner_adjustment,
        "corner_rounding_change_fraction": corner_adjustment,
        "corner_radius_satisfied": corner_ok,
    }


def project_geometry_constraints(
    anode: np.ndarray,
    cathode: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Project a geometry to fixed area, component and corner constraints."""
    anode0 = np.asarray(anode, dtype=bool).copy()
    cathode0 = np.asarray(cathode, dtype=bool).copy()
    anode0, cathode0 = _cut_clearance_moat(
        anode0, cathode0, float(cfg["minimumGapPixels"])
    )
    minimum_object = int(cfg["minimumObjectPixels"])
    anode0 = _remove_small_components(anode0, minimum_object)
    cathode0 = _remove_small_components(cathode0, minimum_object)
    maximum_per, maximum_total = resolve_manufacturing_component_limits(cfg)
    anode0 = _limit_components(anode0, maximum_per)
    cathode0 = _limit_components(cathode0, maximum_per)
    anode0, cathode0 = _limit_total_components(anode0, cathode0, maximum_total)

    fixed_area_enabled = bool(cfg.get("enforceFixedTotalElectrodeArea", False))
    _, target_anode, target_cathode = _fixed_area_targets(anode0.shape, cfg)
    orders = ("anode_first", "cathode_first")
    candidates: list[tuple[float, np.ndarray, np.ndarray, dict[str, object]]] = []

    for order in orders:
        a = anode0.copy()
        c = cathode0.copy()
        a_diag: dict[str, object] = {}
        c_diag: dict[str, object] = {}
        success = True
        failure_reasons: list[str] = []
        for _ in range(3):
            cycle_reasons: list[str] = []
            a = _limit_components(a, maximum_per)
            c = _limit_components(c, maximum_per)
            a, c = _limit_total_components(a, c, maximum_total)
            if fixed_area_enabled:
                if order == "anode_first":
                    a, a_diag = _project_one_polarity_area(a, c, target_anode, cfg)
                    c, c_diag = _project_one_polarity_area(c, a, target_cathode, cfg)
                else:
                    c, c_diag = _project_one_polarity_area(c, a, target_cathode, cfg)
                    a, a_diag = _project_one_polarity_area(a, c, target_anode, cfg)
                if not bool(a_diag.get("success", False)):
                    cycle_reasons.append(f"anode:{a_diag.get('failure_reason', '')}")
                if not bool(c_diag.get("success", False)):
                    cycle_reasons.append(f"cathode:{c_diag.get('failure_reason', '')}")
            else:
                radius = _corner_radius_pixels(a.shape, cfg)
                a = _round_mask(a, radius)
                c = _round_mask(c, radius)
            a_count = _component_count(a)
            c_count = _component_count(c)
            gap = _minimum_gap(a, c)
            area_ok = (
                not fixed_area_enabled
                or (
                    int(np.count_nonzero(a)) == target_anode
                    and int(np.count_nonzero(c)) == target_cathode
                )
            )
            corner_a = corner_radius_diagnostics(a, cfg)
            corner_c = corner_radius_diagnostics(c, cfg)
            corner_a_ok = bool(
                a_diag.get(
                    "corner_radius_satisfied",
                    corner_a["corner_radius_satisfied"],
                )
            )
            corner_c_ok = bool(
                c_diag.get(
                    "corner_radius_satisfied",
                    corner_c["corner_radius_satisfied"],
                )
            )
            if (
                area_ok
                and a_count <= maximum_per
                and c_count <= maximum_per
                and a_count + c_count <= maximum_total
                and gap >= float(cfg["minimumGapPixels"])
                and corner_a_ok
                and corner_c_ok
                and not cycle_reasons
            ):
                failure_reasons = []
                break
            failure_reasons = cycle_reasons
        else:
            success = False

        a_count = _component_count(a)
        c_count = _component_count(c)
        area_error = abs(int(np.count_nonzero(a)) - target_anode) + abs(
            int(np.count_nonzero(c)) - target_cathode
        )
        if a_count + c_count > maximum_total:
            failure_reasons.append("maximum_total_components_exceeded")
        if a_count > maximum_per or c_count > maximum_per:
            failure_reasons.append("maximum_components_per_polarity_exceeded")
        if _minimum_gap(a, c) < float(cfg["minimumGapPixels"]):
            failure_reasons.append("gap_too_small_after_projection")
        if fixed_area_enabled and area_error:
            failure_reasons.append("fixed_area_target_missed")
        corner_a = corner_radius_diagnostics(a, cfg)
        corner_c = corner_radius_diagnostics(c, cfg)
        corner_a_ok = bool(
            a_diag.get(
                "corner_radius_satisfied",
                corner_a["corner_radius_satisfied"],
            )
        )
        corner_c_ok = bool(
            c_diag.get(
                "corner_radius_satisfied",
                corner_c["corner_radius_satisfied"],
            )
        )
        if not corner_a_ok:
            failure_reasons.append("anode_corner_radius_failed")
        if not corner_c_ok:
            failure_reasons.append("cathode_corner_radius_failed")
        success = bool(success and not failure_reasons)
        modification = (
            np.count_nonzero(a ^ anode0) + np.count_nonzero(c ^ cathode0)
        ) / max(np.count_nonzero(anode0) + np.count_nonzero(cathode0), 1)
        diag: dict[str, object] = {
            "constraint_projection_applied": True,
            "constraint_projection_success": success,
            "constraint_projection_order": order,
            "constraint_projection_failure_reason": ";".join(dict.fromkeys(failure_reasons)),
            "target_anode_pixels": int(target_anode),
            "target_cathode_pixels": int(target_cathode),
            "target_total_electrode_pixels": int(target_anode + target_cathode),
            "actual_anode_pixels": int(np.count_nonzero(a)),
            "actual_cathode_pixels": int(np.count_nonzero(c)),
            "actual_total_electrode_pixels": int(np.count_nonzero(a) + np.count_nonzero(c)),
            "fixed_area_error_pixels": int(area_error),
            "anode_components_after_projection": int(a_count),
            "cathode_components_after_projection": int(c_count),
            "total_components_after_projection": int(a_count + c_count),
            "anode_corner_rounding_change_fraction": float(
                a_diag.get(
                    "corner_projection_adjustment_fraction",
                    corner_a["corner_rounding_change_fraction"],
                )
            ),
            "cathode_corner_rounding_change_fraction": float(
                c_diag.get(
                    "corner_projection_adjustment_fraction",
                    corner_c["corner_rounding_change_fraction"],
                )
            ),
            "anode_corner_radius_satisfied": corner_a_ok,
            "cathode_corner_radius_satisfied": corner_c_ok,
            "minimum_corner_radius_pixels": int(
                max(
                    int(a_diag.get("minimum_corner_radius_pixels", corner_a["minimum_corner_radius_pixels"])),
                    int(c_diag.get("minimum_corner_radius_pixels", corner_c["minimum_corner_radius_pixels"])),
                )
            ),
            "minimum_corner_radius_m": float(cfg.get("minimumCornerRadius_m", 0.0)),
            "constraint_projection_modification_fraction": float(modification),
            "anode_exact_area_edits": int(a_diag.get("exact_area_edits", 0)),
            "cathode_exact_area_edits": int(c_diag.get("exact_area_edits", 0)),
        }
        candidates.append((float(modification + 1000 * area_error + (0 if success else 1e6)), a, c, diag))

    candidates.sort(key=lambda item: item[0])
    _, anode_out, cathode_out, diagnostics = candidates[0]
    return anode_out.astype(bool), cathode_out.astype(bool), diagnostics



def _component_count(mask: np.ndarray) -> int:
    return int(ndimage.label(mask, structure=_EIGHT_CONNECTED)[1])


def _bridge_structure(width_pixels: int) -> np.ndarray:
    width = max(1, int(width_pixels))
    return np.ones((width, width), dtype=bool)


def _shortest_safe_path(
    source: np.ndarray,
    target: np.ndarray,
    traversable: np.ndarray,
) -> np.ndarray | None:
    """Return an 8-neighbour shortest path from source to target.

    The grid is tiny (normally 64x64), so a multi-source breadth-first search
    is faster and easier to audit than constructing a sparse graph.  Only
    source-boundary pixels seed the queue.  The returned mask includes one
    source pixel and one target pixel.
    """
    source = np.asarray(source, dtype=bool)
    target = np.asarray(target, dtype=bool)
    traversable = np.asarray(traversable, dtype=bool)
    if not np.any(source) or not np.any(target):
        return None

    rows, columns = source.shape
    visited = np.zeros_like(source, dtype=bool)
    predecessor = np.full(rows * columns, -1, dtype=np.int32)
    queue: deque[tuple[int, int]] = deque()
    boundary = source & ~ndimage.binary_erosion(
        source, structure=_EIGHT_CONNECTED, border_value=0
    )
    ys, xs = np.nonzero(boundary)
    for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
        visited[y, x] = True
        queue.append((y, x))

    neighbours = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),            (0, 1),
        (1, -1),  (1, 0),   (1, 1),
    )
    found: tuple[int, int] | None = None
    while queue:
        y, x = queue.popleft()
        if target[y, x]:
            found = (y, x)
            break
        parent_index = y * columns + x
        for dy, dx in neighbours:
            ny, nx = y + dy, x + dx
            if (
                0 <= ny < rows
                and 0 <= nx < columns
                and not visited[ny, nx]
                and traversable[ny, nx]
            ):
                visited[ny, nx] = True
                predecessor[ny * columns + nx] = parent_index
                queue.append((ny, nx))

    if found is None:
        return None

    path = np.zeros_like(source, dtype=bool)
    index = found[0] * columns + found[1]
    while index >= 0:
        y, x = divmod(int(index), columns)
        path[y, x] = True
        if source[y, x]:
            return path
        index = int(predecessor[index])
    return None


def _connect_one_polarity(
    mask: np.ndarray,
    opposite: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, dict]:
    """Join all same-polarity components using shortest safe metal bridges."""
    original = np.asarray(mask, dtype=bool).copy()
    opposite = np.asarray(opposite, dtype=bool)
    before_components = _component_count(original)
    diagnostics = {
        "components_before": before_components,
        "components_after": before_components,
        "bridge_pixels_added": 0,
        "bridge_path_pixels": 0,
        "success": before_components == 1,
        "failure_reason": "" if before_components == 1 else "not_attempted",
    }
    if before_components == 0:
        diagnostics.update(success=False, failure_reason="empty_polarity")
        return original, diagnostics
    if before_components == 1:
        return original, diagnostics

    minimum_gap = float(cfg["minimumGapPixels"])
    minimum_width = max(1, int(math.ceil(float(cfg["minimumWidthPixels"]))))
    bridge_width = max(
        minimum_width,
        int(cfg.get("connectivityBridgeWidthPixels", minimum_width)),
    )
    bridge_structure = _bridge_structure(bridge_width)
    border = max(0, int(cfg.get("minimumBorderMarginPixels", 0)))
    maximum_growth = float(cfg.get("maximumConnectivityAreaGrowthFraction", 2.0))
    maximum_path = int(cfg.get("maximumConnectivityBridgePathPixels", 0))

    safe_pixels = np.ones_like(original, dtype=bool)
    if np.any(opposite):
        safe_pixels &= (
            ndimage.distance_transform_edt(~opposite) >= minimum_gap
        )
    if border > 0:
        safe_pixels[:border, :] = False
        safe_pixels[-border:, :] = False
        safe_pixels[:, :border] = False
        safe_pixels[:, -border:] = False

    # A centre pixel is routable only when the complete bridge cross-section
    # can fit inside the polarity-safe region. Existing electrode pixels are
    # always traversable so a route can leave and enter each component.
    centre_safe = ndimage.binary_erosion(
        safe_pixels, structure=bridge_structure, border_value=0
    )
    current = original.copy()
    original_pixels = max(1, int(np.count_nonzero(original)))
    total_added = 0
    total_path = 0

    for _ in range(before_components - 1):
        labels, count = ndimage.label(current, structure=_EIGHT_CONNECTED)
        if count <= 1:
            break
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        main_label = int(np.argmax(sizes))
        source = labels == main_label
        target = current & ~source
        path = _shortest_safe_path(source, target, centre_safe | current)
        if path is None:
            diagnostics.update(
                components_after=count,
                bridge_pixels_added=total_added,
                bridge_path_pixels=total_path,
                success=False,
                failure_reason="no_safe_same_polarity_path",
            )
            return original, diagnostics

        path_pixels = int(np.count_nonzero(path & ~current))
        if maximum_path > 0 and path_pixels > maximum_path:
            diagnostics.update(
                components_after=count,
                bridge_pixels_added=total_added,
                bridge_path_pixels=total_path,
                success=False,
                failure_reason="bridge_path_too_long",
            )
            return original, diagnostics

        bridge = ndimage.binary_dilation(path, structure=bridge_structure)
        bridge &= safe_pixels
        candidate = current | bridge
        new_count = _component_count(candidate)
        if new_count >= count:
            diagnostics.update(
                components_after=count,
                bridge_pixels_added=total_added,
                bridge_path_pixels=total_path,
                success=False,
                failure_reason="bridge_did_not_reduce_components",
            )
            return original, diagnostics

        added = int(np.count_nonzero(candidate & ~current))
        if (total_added + added) / original_pixels > maximum_growth:
            diagnostics.update(
                components_after=count,
                bridge_pixels_added=total_added,
                bridge_path_pixels=total_path,
                success=False,
                failure_reason="connectivity_area_growth_exceeded",
            )
            return original, diagnostics
        current = candidate
        total_added += added
        total_path += path_pixels

    after_components = _component_count(current)
    success = after_components == 1
    if success and np.any(opposite):
        success = (
            _direct_contact_pixels(current, opposite) == 0
            and _minimum_gap(current, opposite) >= minimum_gap
        )
    diagnostics.update(
        components_after=after_components,
        bridge_pixels_added=total_added,
        bridge_path_pixels=total_path,
        success=bool(success),
        failure_reason="" if success else "final_connectivity_or_clearance_failed",
    )
    return (current if success else original), diagnostics


def connect_same_polarity_networks(
    anode: np.ndarray,
    cathode: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Project both polarities to one connected component without shorts.

    Both routing orders are tried because the first polarity's new bridges
    become obstacles for the second.  The successful order that adds the
    fewest pixels is retained.  If neither order succeeds, the input masks are
    returned unchanged and the caller can reject the candidate.
    """
    anode = np.asarray(anode, dtype=bool).copy()
    cathode = np.asarray(cathode, dtype=bool).copy()
    before_a = _component_count(anode)
    before_c = _component_count(cathode)
    base = {
        "connectivity_projection_applied": True,
        "connectivity_projection_success": False,
        "connectivity_projection_order": "",
        "connectivity_failure_reason": "",
        "connectivity_anode_components_before": before_a,
        "connectivity_cathode_components_before": before_c,
        "connectivity_anode_components_after": before_a,
        "connectivity_cathode_components_after": before_c,
        "connectivity_anode_bridge_pixels_added": 0,
        "connectivity_cathode_bridge_pixels_added": 0,
        "connectivity_bridge_pixels_added": 0,
        "connectivity_bridge_path_pixels": 0,
        "connectivity_area_growth_fraction": 0.0,
    }
    if before_a == 0 or before_c == 0:
        base["connectivity_failure_reason"] = "empty_polarity"
        return anode, cathode, base
    if before_a == 1 and before_c == 1:
        base["connectivity_projection_success"] = True
        base["connectivity_projection_order"] = "already_connected"
        return anode, cathode, base

    candidates: list[tuple[int, int, np.ndarray, np.ndarray, str, dict, dict]] = []
    failure_reasons: list[str] = []
    for order in ("anode_then_cathode", "cathode_then_anode"):
        projected_a = anode.copy()
        projected_c = cathode.copy()
        if order == "anode_then_cathode":
            projected_a, diag_a = _connect_one_polarity(projected_a, projected_c, cfg)
            if not diag_a["success"]:
                failure_reasons.append(f"{order}:anode:{diag_a['failure_reason']}")
                continue
            projected_c, diag_c = _connect_one_polarity(projected_c, projected_a, cfg)
        else:
            projected_c, diag_c = _connect_one_polarity(projected_c, projected_a, cfg)
            if not diag_c["success"]:
                failure_reasons.append(f"{order}:cathode:{diag_c['failure_reason']}")
                continue
            projected_a, diag_a = _connect_one_polarity(projected_a, projected_c, cfg)

        if not diag_a["success"] or not diag_c["success"]:
            failed = diag_a if not diag_a["success"] else diag_c
            polarity = "anode" if not diag_a["success"] else "cathode"
            failure_reasons.append(f"{order}:{polarity}:{failed['failure_reason']}")
            continue
        if _direct_contact_pixels(projected_a, projected_c) != 0:
            failure_reasons.append(f"{order}:direct_contact_after_projection")
            continue
        if _minimum_gap(projected_a, projected_c) < float(cfg["minimumGapPixels"]):
            failure_reasons.append(f"{order}:gap_after_projection")
            continue

        added = int(diag_a["bridge_pixels_added"] + diag_c["bridge_pixels_added"])
        path = int(diag_a["bridge_path_pixels"] + diag_c["bridge_path_pixels"])
        candidates.append((added, path, projected_a, projected_c, order, diag_a, diag_c))

    if not candidates:
        base["connectivity_failure_reason"] = ";".join(failure_reasons)
        return anode, cathode, base

    candidates.sort(key=lambda item: (item[0], item[1], item[4]))
    added, path, projected_a, projected_c, order, diag_a, diag_c = candidates[0]
    before_total = max(1, int(np.count_nonzero(anode) + np.count_nonzero(cathode)))
    base.update(
        connectivity_projection_success=True,
        connectivity_projection_order=order,
        connectivity_failure_reason="",
        connectivity_anode_components_after=_component_count(projected_a),
        connectivity_cathode_components_after=_component_count(projected_c),
        connectivity_anode_bridge_pixels_added=int(diag_a["bridge_pixels_added"]),
        connectivity_cathode_bridge_pixels_added=int(diag_c["bridge_pixels_added"]),
        connectivity_bridge_pixels_added=added,
        connectivity_bridge_path_pixels=path,
        connectivity_area_growth_fraction=float(added / before_total),
    )
    return projected_a, projected_c, base


def _cut_clearance_moat(anode: np.ndarray, cathode: np.ndarray, minimum_gap_pixels: float) -> tuple[np.ndarray, np.ndarray]:
    anode = np.asarray(anode, dtype=bool).copy()
    cathode = np.asarray(cathode, dtype=bool).copy()

    # Any uncertain class becomes propellant; never randomly assign it to a
    # polarity because that can create a metallic bridge.
    overlap = anode & cathode
    anode[overlap] = False
    cathode[overlap] = False

    for _ in range(12):
        if not np.any(anode) or not np.any(cathode):
            break
        distance_to_cathode = ndimage.distance_transform_edt(~cathode)
        distance_to_anode = ndimage.distance_transform_edt(~anode)
        conflict_anode = anode & (distance_to_cathode < minimum_gap_pixels)
        conflict_cathode = cathode & (distance_to_anode < minimum_gap_pixels)
        if not np.any(conflict_anode) and not np.any(conflict_cathode):
            break
        # Symmetric removal avoids systematic anode/cathode area bias.
        anode[conflict_anode] = False
        cathode[conflict_cathode] = False
    return anode, cathode


def _mask_iou(before: np.ndarray, after: np.ndarray) -> float:
    union = int(np.count_nonzero(before | after))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(before & after) / union)


def repair(anode: np.ndarray, cathode: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Repair a DDPM mask and project it to the configured manufacturing set.

    Ambiguous pixels become propellant and tiny islands are removed.  The
    default v7.1.4 path then enforces equal fixed polarity areas, a global
    component cap and a corner-radius proxy.  Optional same-polarity bridge
    routing remains available for ``single_component`` mode, but it is disabled
    when multiple externally bus-connected islands are permitted.
    """
    anode = np.asarray(anode, dtype=bool).copy()
    cathode = np.asarray(cathode, dtype=bool).copy()
    anode_before = anode.copy()
    cathode_before = cathode.copy()
    before_overlap = int(np.count_nonzero(anode & cathode))
    before_contact = _direct_contact_pixels(anode, cathode)

    anode, cathode = _cut_clearance_moat(
        anode, cathode, float(cfg["minimumGapPixels"])
    )
    minimum_object = int(cfg["minimumObjectPixels"])
    anode = _remove_small_components(anode, minimum_object)
    cathode = _remove_small_components(cathode, minimum_object)

    connection_enabled = bool(cfg.get("connectSamePolarityComponents", False))
    if connection_enabled:
        anode, cathode, connection_diagnostics = connect_same_polarity_networks(
            anode, cathode, cfg
        )
    else:
        maximum_components, maximum_total_components = (
            resolve_manufacturing_component_limits(cfg)
        )
        anode = _limit_components(anode, maximum_components)
        cathode = _limit_components(cathode, maximum_components)
        anode, cathode = _limit_total_components(
            anode, cathode, maximum_total_components
        )
        anode, cathode = _cut_clearance_moat(
            anode, cathode, float(cfg["minimumGapPixels"])
        )
        connection_diagnostics = {
            "connectivity_projection_applied": False,
            "connectivity_projection_success": True,
            "connectivity_projection_order": "disabled",
            "connectivity_failure_reason": "",
            "connectivity_anode_components_before": _component_count(anode),
            "connectivity_cathode_components_before": _component_count(cathode),
            "connectivity_anode_components_after": _component_count(anode),
            "connectivity_cathode_components_after": _component_count(cathode),
            "connectivity_anode_bridge_pixels_added": 0,
            "connectivity_cathode_bridge_pixels_added": 0,
            "connectivity_bridge_pixels_added": 0,
            "connectivity_bridge_path_pixels": 0,
            "connectivity_area_growth_fraction": 0.0,
        }

    anode, cathode, constraint_diagnostics = project_geometry_constraints(
        anode, cathode, cfg
    )

    iou_anode = _mask_iou(anode_before, anode)
    iou_cathode = _mask_iou(cathode_before, cathode)
    before_total = int(
        np.count_nonzero(anode_before) + np.count_nonzero(cathode_before)
    )
    after_total = int(np.count_nonzero(anode) + np.count_nonzero(cathode))
    pixels_added = int(
        np.count_nonzero(anode & ~anode_before)
        + np.count_nonzero(cathode & ~cathode_before)
    )
    pixels_removed = int(
        np.count_nonzero(anode_before & ~anode)
        + np.count_nonzero(cathode_before & ~cathode)
    )
    area_change = (
        (before_total - after_total) / before_total if before_total else 0.0
    )
    modification_fraction = (
        (pixels_added + pixels_removed) / before_total if before_total else 0.0
    )
    minimum_iou = float(cfg.get("minimumRepairIoU", 0.70))
    enforce = bool(cfg.get("rejectExcessiveRepair", True))
    connectivity_unresolved = bool(
        connection_enabled
        and not connection_diagnostics["connectivity_projection_success"]
    )
    constraint_unresolved = not bool(
        constraint_diagnostics.get("constraint_projection_success", False)
    )
    diagnostics = {
        "repair_overlap_pixels_removed": before_overlap,
        "repair_direct_contact_pixels_before": before_contact,
        "repair_direct_contact_pixels_after": _direct_contact_pixels(anode, cathode),
        "repair_iou_anode": iou_anode,
        "repair_iou_cathode": iou_cathode,
        "repair_iou_min": min(iou_anode, iou_cathode),
        # Kept for compatibility: positive means net removal, negative net growth.
        "repair_area_change_fraction": float(area_change),
        "repair_area_growth_fraction": (
            float(pixels_added / before_total) if before_total else 0.0
        ),
        "repair_modification_fraction": float(modification_fraction),
        "repair_pixels_added": pixels_added,
        "repair_pixels_removed": pixels_removed,
        "repair_connectivity_unresolved": connectivity_unresolved,
        "repair_constraint_projection_unresolved": constraint_unresolved,
        "repair_distortion_exceeded": bool(
            enforce and min(iou_anode, iou_cathode) < minimum_iou
        ),
        "repair_applied": True,
        **connection_diagnostics,
        **constraint_diagnostics,
    }
    return anode.astype(bool), cathode.astype(bool), diagnostics

def evaluate(
    anode: np.ndarray,
    cathode: np.ndarray,
    cfg: dict,
    projection_diagnostics: dict | None = None,
) -> dict:
    anode = np.asarray(anode, dtype=bool)
    cathode = np.asarray(cathode, dtype=bool)
    total = anode.size
    a_frac = float(np.mean(anode))
    c_frac = float(np.mean(cathode))
    overlap = int(np.count_nonzero(anode & cathode))
    direct_contact = _direct_contact_pixels(anode, cathode)
    a_sizes = _component_sizes(anode)
    c_sizes = _component_sizes(cathode)
    connectivity = _connectivity_diagnostics(
        anode, cathode, cfg, len(a_sizes), len(c_sizes)
    )
    border = int(cfg["minimumBorderMarginPixels"])
    border_mask = np.zeros_like(anode, dtype=bool)
    if border > 0:
        border_mask[:border, :] = True
        border_mask[-border:, :] = True
        border_mask[:, :border] = True
        border_mask[:, -border:] = True

    reasons: list[str] = []
    if overlap:
        reasons.append("polarity_overlap")
    if direct_contact:
        reasons.append("polarity_direct_contact")
    if min(a_frac, c_frac) < float(cfg["minimumAreaFractionPerPolarity"]):
        reasons.append("insufficient_electrode_area")
    if a_frac + c_frac > float(cfg["maximumTotalElectrodeAreaFraction"]):
        reasons.append("excessive_total_electrode_area")
    imbalance = abs(a_frac - c_frac) / max(a_frac + c_frac, 1e-12)
    if imbalance > float(cfg["maximumAreaImbalanceFraction"]):
        reasons.append("polarity_area_imbalance")
    target_total, target_anode, target_cathode = _fixed_area_targets(anode.shape, cfg)
    area_tolerance = int(cfg.get("fixedAreaTolerancePixels", 0))
    anode_area_error = abs(int(np.count_nonzero(anode)) - target_anode) if target_total else 0
    cathode_area_error = abs(int(np.count_nonzero(cathode)) - target_cathode) if target_total else 0
    fixed_area_satisfied = bool(
        not bool(cfg.get("enforceFixedTotalElectrodeArea", False))
        or (anode_area_error <= area_tolerance and cathode_area_error <= area_tolerance)
    )
    if not fixed_area_satisfied:
        reasons.append("fixed_electrode_area_mismatch")
    maximum_per, maximum_total = resolve_manufacturing_component_limits(cfg)
    if len(a_sizes) > maximum_per or len(c_sizes) > maximum_per:
        reasons.append("too_many_components_per_polarity")
    if len(a_sizes) + len(c_sizes) > maximum_total:
        reasons.append("too_many_total_components")
    minimum_object = int(cfg["minimumObjectPixels"])
    if (a_sizes and min(a_sizes) < minimum_object) or (c_sizes and min(c_sizes) < minimum_object):
        reasons.append("small_island")

    if not connectivity["anode_bus_connected"]:
        reasons.append("anode_bus_disconnected")
    if not connectivity["cathode_bus_connected"]:
        reasons.append("cathode_bus_disconnected")
    if (
        connectivity["electrode_connectivity_mode"] == "implicit_3d_bus"
        and not connectivity["implicit_3d_bus_documented"]
    ):
        reasons.append("undocumented_implicit_3d_bus")

    gap = _minimum_gap(anode, cathode)
    if gap < float(cfg["minimumGapPixels"]):
        reasons.append("gap_too_small")

    n = anode.shape[0]
    domain_size_m = float(cfg.get("domainSize_m", 0.0))
    pixel_spacing_m = domain_size_m / n if domain_size_m > 0 else math.nan
    gap_m = gap * pixel_spacing_m if math.isfinite(pixel_spacing_m) else math.nan
    minimum_gap_m = float(cfg.get("minimumGap_m", 0.0))
    if minimum_gap_m > 0 and (not math.isfinite(gap_m) or gap_m < minimum_gap_m):
        reasons.append("physical_gap_too_small")

    min_width = min(_minimum_width_proxy(anode), _minimum_width_proxy(cathode))
    if min_width < float(cfg["minimumWidthPixels"]):
        reasons.append("width_too_small")
    anode_corner = corner_radius_diagnostics(anode, cfg)
    cathode_corner = corner_radius_diagnostics(cathode, cfg)
    projection = projection_diagnostics or {}
    projection_valid = bool(
        projection.get("constraint_projection_success", False)
    )
    if projection_valid:
        anode_corner_ok = bool(
            projection.get(
                "anode_corner_radius_satisfied",
                anode_corner["corner_radius_satisfied"],
            )
        )
        cathode_corner_ok = bool(
            projection.get(
                "cathode_corner_radius_satisfied",
                cathode_corner["corner_radius_satisfied"],
            )
        )
        anode_corner_change = float(
            projection.get(
                "anode_corner_rounding_change_fraction",
                anode_corner["corner_rounding_change_fraction"],
            )
        )
        cathode_corner_change = float(
            projection.get(
                "cathode_corner_rounding_change_fraction",
                cathode_corner["corner_rounding_change_fraction"],
            )
        )
        corner_audit_source = "constraint_projection"
    else:
        anode_corner_ok = bool(anode_corner["corner_radius_satisfied"])
        cathode_corner_ok = bool(cathode_corner["corner_radius_satisfied"])
        anode_corner_change = float(
            anode_corner["corner_rounding_change_fraction"]
        )
        cathode_corner_change = float(
            cathode_corner["corner_rounding_change_fraction"]
        )
        corner_audit_source = "independent_mask_audit"
    if not anode_corner_ok:
        reasons.append("anode_corner_radius_too_small")
    if not cathode_corner_ok:
        reasons.append("cathode_corner_radius_too_small")
    if border > 0 and np.any((anode | cathode) & border_mask):
        reasons.append("border_margin_violation")

    return {
        "manufacturable": not reasons,
        "reasons": ";".join(dict.fromkeys(reasons)),
        "short_circuit_free": overlap == 0 and direct_contact == 0 and gap >= float(cfg["minimumGapPixels"]),
        "anode_area_fraction": a_frac,
        "cathode_area_fraction": c_frac,
        "total_electrode_area_fraction": a_frac + c_frac,
        "area_imbalance_fraction": imbalance,
        "target_total_electrode_pixels": int(target_total),
        "target_anode_pixels": int(target_anode),
        "target_cathode_pixels": int(target_cathode),
        "anode_area_error_pixels": int(anode_area_error),
        "cathode_area_error_pixels": int(cathode_area_error),
        "fixed_area_satisfied": fixed_area_satisfied,
        "minimum_gap_pixels": gap,
        "minimum_gap_m": gap_m,
        "minimum_propellant_gap_layers": max(int(math.floor(gap)) - 1, 0),
        "minimum_width_proxy_pixels": min_width,
        "anode_components": len(a_sizes),
        "cathode_components": len(c_sizes),
        "total_components": len(a_sizes) + len(c_sizes),
        "maximum_total_components": int(maximum_total),
        "component_limit_satisfied": bool(
            len(a_sizes) <= maximum_per
            and len(c_sizes) <= maximum_per
            and len(a_sizes) + len(c_sizes) <= maximum_total
        ),
        "single_component_per_polarity": len(a_sizes) == 1 and len(c_sizes) == 1,
        "minimum_corner_radius_pixels": int(
            max(
                anode_corner["minimum_corner_radius_pixels"],
                cathode_corner["minimum_corner_radius_pixels"],
            )
        ),
        "minimum_corner_radius_m": float(cfg.get("minimumCornerRadius_m", 0.0)),
        "anode_corner_rounding_change_fraction": anode_corner_change,
        "cathode_corner_rounding_change_fraction": cathode_corner_change,
        "corner_radius_satisfied": bool(
            anode_corner_ok and cathode_corner_ok
        ),
        "corner_radius_audit_source": corner_audit_source,
        "overlap_pixels": overlap,
        "direct_contact_pixels": direct_contact,
        **connectivity,
        "total_pixels": total,
    }
