"""Subcell Species-front tracking on a fixed normal sampling frame.

The routines here measure the motion of a reaction Species/progress isovalue.
They neither create nor update the transported material level set.  Regression
is computed from isovalue intersections on the *same* physical rays and in the
same unit-normal directions at both times; it is therefore distinct from the
legacy consumed-area divided by front-length diagnostic.

The paper does not identify a unique Species threshold.  The standard
``(0.3, 0.5, 0.7)`` sensitivity set is consequently labelled
``ASSUMED_NOT_FROM_PAPER`` in every result.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import numpy as np

from .provenance import Provenance, ReactiveConfigurationError, ReactiveNumericalError


ASSUMED_SPECIES_THRESHOLD_SET = (0.3, 0.5, 0.7)
ASSUMED_SPECIES_THRESHOLD_PROVENANCE = Provenance.ASSUMED_NOT_FROM_PAPER
FRONT_TRACKING_METHOD = "fixed_normal_ray_linear_subcell_interpolation"
FRONT_TRACKING_METHOD_PROVENANCE = Provenance.ASSUMED_NOT_FROM_PAPER


def _immutable_float_array(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    return np.frombuffer(result.tobytes(order="C"), dtype=np.float64).reshape(
        result.shape
    )


def _immutable_int_array(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=np.int64, copy=True, order="C")
    return np.frombuffer(result.tobytes(order="C"), dtype=np.int64).reshape(
        result.shape
    )


def _immutable_bool_array(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=bool, copy=True, order="C")
    return np.frombuffer(result.view(np.uint8).tobytes(), dtype=np.bool_).reshape(
        result.shape
    )


@dataclass(frozen=True)
class NormalSamplingFrame:
    """Fixed coordinates and normals used at every tracked time.

    For ray ``r`` and signed sample offset ``s``, the physical sampling point is
    ``anchor_points_xy_m[r] + s * unit_normals_xy[r]``.
    """

    anchor_points_xy_m: np.ndarray
    unit_normals_xy: np.ndarray
    signed_sample_offsets_m: np.ndarray

    def __post_init__(self) -> None:
        anchors = np.asarray(self.anchor_points_xy_m, dtype=np.float64)
        normals = np.asarray(self.unit_normals_xy, dtype=np.float64)
        offsets = np.asarray(self.signed_sample_offsets_m, dtype=np.float64)
        if anchors.ndim != 2 or anchors.shape[1:] != (2,) or anchors.shape[0] == 0:
            raise ReactiveConfigurationError(
                "front sampling anchors must have non-empty shape (ray_count, 2)"
            )
        if normals.shape != anchors.shape:
            raise ReactiveConfigurationError(
                "front sampling unit normals must have the same shape as anchors"
            )
        if offsets.ndim != 1 or offsets.size < 2:
            raise ReactiveConfigurationError(
                "front sampling offsets must be a one-dimensional array with at least two values"
            )
        if not (
            np.all(np.isfinite(anchors))
            and np.all(np.isfinite(normals))
            and np.all(np.isfinite(offsets))
        ):
            raise ReactiveConfigurationError("front sampling frame must be finite")
        normal_magnitudes = np.linalg.norm(normals, axis=1)
        if np.any(normal_magnitudes <= 0.0):
            raise ReactiveConfigurationError("front sampling normals must be non-zero")
        normals = normals / normal_magnitudes[:, None]
        if np.any(np.diff(offsets) <= 0.0):
            raise ReactiveConfigurationError(
                "front sampling offsets must be strictly increasing"
            )
        object.__setattr__(self, "anchor_points_xy_m", _immutable_float_array(anchors))
        object.__setattr__(self, "unit_normals_xy", _immutable_float_array(normals))
        object.__setattr__(
            self, "signed_sample_offsets_m", _immutable_float_array(offsets)
        )

    @property
    def ray_count(self) -> int:
        return int(self.anchor_points_xy_m.shape[0])

    def coordinates_xy_m(self) -> np.ndarray:
        return self.anchor_points_xy_m[:, None, :] + (
            self.signed_sample_offsets_m[None, :, None]
            * self.unit_normals_xy[:, None, :]
        )


@dataclass(frozen=True)
class IsofrontDiagnostics:
    ray_count: int
    unique_intersection_ray_count: int
    absent_intersection_ray_count: int
    multiple_intersection_ray_count: int
    threshold_plateau_ray_count: int
    out_of_grid_sample_count: int


@dataclass(frozen=True)
class SpeciesIsofrontSnapshot:
    """Subcell threshold locations on a fixed normal sampling frame."""

    species_threshold: float
    threshold_provenance: Provenance
    normal_iso_position_m: np.ndarray
    isofront_coordinates_xy_m: np.ndarray
    intersection_count_by_ray: np.ndarray
    unique_intersection_mask: np.ndarray
    threshold_plateau_mask: np.ndarray
    sampling_frame: NormalSamplingFrame
    diagnostics: IsofrontDiagnostics
    tracking_method: str = FRONT_TRACKING_METHOD
    tracking_method_provenance: Provenance = FRONT_TRACKING_METHOD_PROVENANCE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "normal_iso_position_m",
            _immutable_float_array(self.normal_iso_position_m),
        )
        object.__setattr__(
            self,
            "isofront_coordinates_xy_m",
            _immutable_float_array(self.isofront_coordinates_xy_m),
        )
        object.__setattr__(
            self,
            "intersection_count_by_ray",
            _immutable_int_array(self.intersection_count_by_ray),
        )
        object.__setattr__(
            self,
            "unique_intersection_mask",
            _immutable_bool_array(self.unique_intersection_mask),
        )
        object.__setattr__(
            self,
            "threshold_plateau_mask",
            _immutable_bool_array(self.threshold_plateau_mask),
        )


@dataclass(frozen=True)
class FrontTransitionDiagnostics:
    persistent_unique_ray_count: int
    front_birth_ray_count: int
    front_death_ray_count: int
    unique_to_multiple_branching_candidate_ray_count: int
    multiple_to_unique_merging_candidate_ray_count: int
    previous_multiple_intersection_ray_count: int
    current_multiple_intersection_ray_count: int
    topology_ambiguous_ray_count: int
    no_comparable_intersection_ray_count: int


@dataclass(frozen=True)
class NormalRegressionResult:
    """Local and aggregate signed motion along the fixed unit normals."""

    species_threshold: float
    threshold_provenance: Provenance
    elapsed_time_s: float
    local_normal_displacement_m: np.ndarray
    local_normal_regression_speed_m_s: np.ndarray
    comparable_ray_mask: np.ndarray
    mean_normal_regression_speed_m_s: float
    median_normal_regression_speed_m_s: float
    normal_regression_speed_standard_deviation_m_s: float
    previous_snapshot: SpeciesIsofrontSnapshot
    current_snapshot: SpeciesIsofrontSnapshot
    diagnostics: FrontTransitionDiagnostics
    tracking_method: str = FRONT_TRACKING_METHOD
    tracking_method_provenance: Provenance = FRONT_TRACKING_METHOD_PROVENANCE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "local_normal_displacement_m",
            _immutable_float_array(self.local_normal_displacement_m),
        )
        object.__setattr__(
            self,
            "local_normal_regression_speed_m_s",
            _immutable_float_array(self.local_normal_regression_speed_m_s),
        )
        object.__setattr__(
            self,
            "comparable_ray_mask",
            _immutable_bool_array(self.comparable_ray_mask),
        )


@dataclass(frozen=True)
class ThresholdSensitivityResult:
    threshold_provenance: Provenance
    by_threshold: Mapping[float, NormalRegressionResult]
    finite_mean_speed_count: int
    mean_speed_minimum_m_s: float
    mean_speed_maximum_m_s: float
    mean_speed_span_m_s: float
    maximum_absolute_deviation_from_threshold_0p5_m_s: float


def axis_aligned_sampling_frame(
    shape: tuple[int, int],
    *,
    dx_m: float,
    dy_m: float | None = None,
    grid_first_cell_xy_m: tuple[float, float] = (0.0, 0.0),
    normal_axis: str = "x",
    normal_sign: int = 1,
) -> NormalSamplingFrame:
    """Create one full-domain normal ray per row or column of a Cartesian grid."""

    if len(shape) != 2 or min(shape) < 2:
        raise ReactiveConfigurationError(
            "front grid shape must contain two axes of size >= 2"
        )
    dx = float(dx_m)
    dy = dx if dy_m is None else float(dy_m)
    if not math.isfinite(dx) or not math.isfinite(dy) or dx <= 0.0 or dy <= 0.0:
        raise ReactiveConfigurationError(
            "front grid spacing must be positive and finite"
        )
    if normal_axis not in {"x", "y"}:
        raise ReactiveConfigurationError("normal_axis must be 'x' or 'y'")
    if normal_sign not in {-1, 1}:
        raise ReactiveConfigurationError("normal_sign must equal -1 or +1")
    first_x, first_y = map(float, grid_first_cell_xy_m)
    if not math.isfinite(first_x) or not math.isfinite(first_y):
        raise ReactiveConfigurationError("front grid origin must be finite")
    rows, columns = shape
    if normal_axis == "x":
        y_coordinates = first_y + np.arange(rows, dtype=np.float64) * dy
        anchor_x = first_x if normal_sign > 0 else first_x + (columns - 1) * dx
        anchors = np.column_stack((np.full(rows, anchor_x), y_coordinates))
        normals = np.tile(np.array((float(normal_sign), 0.0)), (rows, 1))
        offsets = np.arange(columns, dtype=np.float64) * dx
    else:
        x_coordinates = first_x + np.arange(columns, dtype=np.float64) * dx
        anchor_y = first_y if normal_sign > 0 else first_y + (rows - 1) * dy
        anchors = np.column_stack((x_coordinates, np.full(columns, anchor_y)))
        normals = np.tile(np.array((0.0, float(normal_sign))), (columns, 1))
        offsets = np.arange(rows, dtype=np.float64) * dy
    return NormalSamplingFrame(anchors, normals, offsets)


def _validated_species_progress(values: np.ndarray) -> np.ndarray:
    progress = np.asarray(values, dtype=np.float64)
    if progress.ndim != 2 or min(progress.shape) < 2:
        raise ReactiveConfigurationError(
            "reaction_species_progress must be a two-dimensional array with axes >= 2"
        )
    if not np.all(np.isfinite(progress)):
        raise ReactiveNumericalError(
            "nonfinite_reaction_species",
            "reaction Species/progress contains NaN or infinity",
        )
    if np.any(progress < 0.0) or np.any(progress > 1.0):
        raise ReactiveNumericalError(
            "reaction_species_out_of_bounds",
            "reaction Species/progress must remain in the closed interval [0, 1]",
            {
                "minimum": float(np.min(progress)),
                "maximum": float(np.max(progress)),
            },
        )
    return progress


def _validated_threshold(value: float) -> float:
    threshold = float(value)
    if not math.isfinite(threshold) or threshold <= 0.0 or threshold >= 1.0:
        raise ReactiveConfigurationError(
            "Species threshold must lie strictly inside (0, 1)"
        )
    return threshold


def _bilinear_sample_cell_centred(
    field: np.ndarray,
    coordinates_xy_m: np.ndarray,
    *,
    dx_m: float,
    dy_m: float,
    grid_first_cell_xy_m: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    first_x, first_y = grid_first_cell_xy_m
    flat_coordinates = np.asarray(coordinates_xy_m, dtype=np.float64).reshape(-1, 2)
    fractional_x = (flat_coordinates[:, 0] - first_x) / dx_m
    fractional_y = (flat_coordinates[:, 1] - first_y) / dy_m
    rows, columns = field.shape
    coordinate_tolerance = 64.0 * np.finfo(np.float64).eps * max(rows, columns)
    valid = (
        (fractional_x >= -coordinate_tolerance)
        & (fractional_x <= columns - 1 + coordinate_tolerance)
        & (fractional_y >= -coordinate_tolerance)
        & (fractional_y <= rows - 1 + coordinate_tolerance)
    )
    x = np.clip(fractional_x, 0.0, columns - 1.0)
    y = np.clip(fractional_y, 0.0, rows - 1.0)
    column_0 = np.minimum(np.floor(x).astype(np.int64), columns - 2)
    row_0 = np.minimum(np.floor(y).astype(np.int64), rows - 2)
    fraction_x = x - column_0
    fraction_y = y - row_0
    lower = (1.0 - fraction_x) * field[row_0, column_0] + fraction_x * field[
        row_0, column_0 + 1
    ]
    upper = (1.0 - fraction_x) * field[row_0 + 1, column_0] + fraction_x * field[
        row_0 + 1, column_0 + 1
    ]
    samples = (1.0 - fraction_y) * lower + fraction_y * upper
    samples = np.where(valid, samples, np.nan)
    return samples.reshape(coordinates_xy_m.shape[:-1]), valid.reshape(
        coordinates_xy_m.shape[:-1]
    )


def _ray_threshold_intersections(
    offsets: np.ndarray,
    samples: np.ndarray,
    threshold: float,
) -> tuple[list[float], bool]:
    finite_samples = samples[np.isfinite(samples)]
    if finite_samples.size == 0:
        return [], False
    scale = max(1.0, abs(threshold), float(np.max(np.abs(finite_samples))))
    value_tolerance = 64.0 * np.finfo(np.float64).eps * scale
    offset_tolerance = (
        64.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(offsets))))
    )
    intersections: list[float] = []
    plateau = False
    for index in range(offsets.size - 1):
        value_0 = samples[index]
        value_1 = samples[index + 1]
        if not math.isfinite(value_0) or not math.isfinite(value_1):
            continue
        delta_0 = value_0 - threshold
        delta_1 = value_1 - threshold
        equal_0 = abs(delta_0) <= value_tolerance
        equal_1 = abs(delta_1) <= value_tolerance
        if equal_0 and equal_1:
            plateau = True
            intersections.extend((float(offsets[index]), float(offsets[index + 1])))
            continue
        if equal_0:
            intersections.append(float(offsets[index]))
        if equal_1:
            intersections.append(float(offsets[index + 1]))
        if delta_0 * delta_1 < 0.0:
            fraction = -delta_0 / (delta_1 - delta_0)
            intersections.append(
                float(offsets[index] + fraction * (offsets[index + 1] - offsets[index]))
            )
    intersections.sort()
    deduplicated: list[float] = []
    for value in intersections:
        if not deduplicated or abs(value - deduplicated[-1]) > offset_tolerance:
            deduplicated.append(value)
    return deduplicated, plateau


def extract_species_isofront(
    reaction_species_progress: np.ndarray,
    sampling_frame: NormalSamplingFrame,
    *,
    species_threshold: float,
    dx_m: float,
    dy_m: float | None = None,
    grid_first_cell_xy_m: tuple[float, float] = (0.0, 0.0),
    threshold_provenance: Provenance = Provenance.ASSUMED_NOT_FROM_PAPER,
) -> SpeciesIsofrontSnapshot:
    """Locate a Species isovalue by linear interpolation between ray samples."""

    progress = _validated_species_progress(reaction_species_progress)
    threshold = _validated_threshold(species_threshold)
    try:
        provenance = Provenance(threshold_provenance)
    except ValueError as exc:
        raise ReactiveConfigurationError(
            "Invalid Species threshold provenance"
        ) from exc
    dx = float(dx_m)
    dy = dx if dy_m is None else float(dy_m)
    if not math.isfinite(dx) or not math.isfinite(dy) or dx <= 0.0 or dy <= 0.0:
        raise ReactiveConfigurationError(
            "front grid spacing must be positive and finite"
        )
    first_xy = tuple(map(float, grid_first_cell_xy_m))
    if len(first_xy) != 2 or not all(map(math.isfinite, first_xy)):
        raise ReactiveConfigurationError(
            "front grid first-cell coordinate must be finite"
        )

    coordinates = sampling_frame.coordinates_xy_m()
    samples, valid_samples = _bilinear_sample_cell_centred(
        progress,
        coordinates,
        dx_m=dx,
        dy_m=dy,
        grid_first_cell_xy_m=first_xy,
    )
    ray_count = sampling_frame.ray_count
    positions = np.full(ray_count, np.nan, dtype=np.float64)
    counts = np.zeros(ray_count, dtype=np.int64)
    plateau_mask = np.zeros(ray_count, dtype=bool)
    for ray in range(ray_count):
        intersections, plateau = _ray_threshold_intersections(
            sampling_frame.signed_sample_offsets_m,
            samples[ray],
            threshold,
        )
        plateau_mask[ray] = plateau
        counts[ray] = max(len(intersections), 2 if plateau else 0)
        if len(intersections) == 1 and not plateau:
            positions[ray] = intersections[0]
    unique = (counts == 1) & ~plateau_mask
    isofront_coordinates = np.full((ray_count, 2), np.nan, dtype=np.float64)
    isofront_coordinates[unique] = (
        sampling_frame.anchor_points_xy_m[unique]
        + positions[unique, None] * sampling_frame.unit_normals_xy[unique]
    )
    multiple = (counts > 1) | plateau_mask
    diagnostics = IsofrontDiagnostics(
        ray_count=ray_count,
        unique_intersection_ray_count=int(np.count_nonzero(unique)),
        absent_intersection_ray_count=int(np.count_nonzero(counts == 0)),
        multiple_intersection_ray_count=int(np.count_nonzero(multiple)),
        threshold_plateau_ray_count=int(np.count_nonzero(plateau_mask)),
        out_of_grid_sample_count=int(np.count_nonzero(~valid_samples)),
    )
    return SpeciesIsofrontSnapshot(
        species_threshold=threshold,
        threshold_provenance=provenance,
        normal_iso_position_m=positions,
        isofront_coordinates_xy_m=isofront_coordinates,
        intersection_count_by_ray=counts,
        unique_intersection_mask=unique,
        threshold_plateau_mask=plateau_mask,
        sampling_frame=sampling_frame,
        diagnostics=diagnostics,
    )


def _same_sampling_frame(left: NormalSamplingFrame, right: NormalSamplingFrame) -> bool:
    return (
        np.array_equal(left.anchor_points_xy_m, right.anchor_points_xy_m)
        and np.array_equal(left.unit_normals_xy, right.unit_normals_xy)
        and np.array_equal(left.signed_sample_offsets_m, right.signed_sample_offsets_m)
    )


def measure_normal_regression(
    previous_snapshot: SpeciesIsofrontSnapshot,
    current_snapshot: SpeciesIsofrontSnapshot,
    *,
    elapsed_time_s: float,
) -> NormalRegressionResult:
    """Measure signed isofront displacement using identical rays and normals."""

    dt = float(elapsed_time_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ReactiveConfigurationError(
            "front elapsed_time_s must be positive and finite"
        )
    if previous_snapshot.species_threshold != current_snapshot.species_threshold:
        raise ReactiveConfigurationError(
            "front snapshots must use the same Species threshold"
        )
    if (
        previous_snapshot.threshold_provenance
        is not current_snapshot.threshold_provenance
    ):
        raise ReactiveConfigurationError(
            "front snapshots must use the same threshold provenance"
        )
    if not _same_sampling_frame(
        previous_snapshot.sampling_frame, current_snapshot.sampling_frame
    ):
        raise ReactiveConfigurationError(
            "front displacement requires identical physical coordinates and unit normals"
        )

    previous_unique = previous_snapshot.unique_intersection_mask
    current_unique = current_snapshot.unique_intersection_mask
    comparable = previous_unique & current_unique
    displacement = np.full(comparable.shape, np.nan, dtype=np.float64)
    displacement[comparable] = (
        current_snapshot.normal_iso_position_m[comparable]
        - previous_snapshot.normal_iso_position_m[comparable]
    )
    speeds = displacement / dt
    if np.any(comparable):
        finite_speeds = speeds[comparable]
        mean_speed = float(np.mean(finite_speeds))
        median_speed = float(np.median(finite_speeds))
        standard_deviation = float(np.std(finite_speeds))
    else:
        mean_speed = math.nan
        median_speed = math.nan
        standard_deviation = math.nan

    previous_multiple = previous_snapshot.intersection_count_by_ray > 1
    current_multiple = current_snapshot.intersection_count_by_ray > 1
    previous_absent = previous_snapshot.intersection_count_by_ray == 0
    current_absent = current_snapshot.intersection_count_by_ray == 0
    births = previous_absent & current_unique
    deaths = previous_unique & current_absent
    branching_candidates = previous_unique & current_multiple
    merging_candidates = previous_multiple & current_unique
    ambiguous = previous_multiple | current_multiple
    neither_comparable = ~comparable
    diagnostics = FrontTransitionDiagnostics(
        persistent_unique_ray_count=int(np.count_nonzero(comparable)),
        front_birth_ray_count=int(np.count_nonzero(births)),
        front_death_ray_count=int(np.count_nonzero(deaths)),
        unique_to_multiple_branching_candidate_ray_count=int(
            np.count_nonzero(branching_candidates)
        ),
        multiple_to_unique_merging_candidate_ray_count=int(
            np.count_nonzero(merging_candidates)
        ),
        previous_multiple_intersection_ray_count=int(
            np.count_nonzero(previous_multiple)
        ),
        current_multiple_intersection_ray_count=int(np.count_nonzero(current_multiple)),
        topology_ambiguous_ray_count=int(np.count_nonzero(ambiguous)),
        no_comparable_intersection_ray_count=int(np.count_nonzero(neither_comparable)),
    )
    return NormalRegressionResult(
        species_threshold=previous_snapshot.species_threshold,
        threshold_provenance=previous_snapshot.threshold_provenance,
        elapsed_time_s=dt,
        local_normal_displacement_m=displacement,
        local_normal_regression_speed_m_s=speeds,
        comparable_ray_mask=comparable,
        mean_normal_regression_speed_m_s=mean_speed,
        median_normal_regression_speed_m_s=median_speed,
        normal_regression_speed_standard_deviation_m_s=standard_deviation,
        previous_snapshot=previous_snapshot,
        current_snapshot=current_snapshot,
        diagnostics=diagnostics,
    )


def track_species_isofront_motion(
    previous_reaction_species_progress: np.ndarray,
    current_reaction_species_progress: np.ndarray,
    sampling_frame: NormalSamplingFrame,
    *,
    elapsed_time_s: float,
    species_threshold: float,
    dx_m: float,
    dy_m: float | None = None,
    grid_first_cell_xy_m: tuple[float, float] = (0.0, 0.0),
    threshold_provenance: Provenance = Provenance.ASSUMED_NOT_FROM_PAPER,
) -> NormalRegressionResult:
    """Extract both time levels on one frame and calculate their normal motion."""

    previous = extract_species_isofront(
        previous_reaction_species_progress,
        sampling_frame,
        species_threshold=species_threshold,
        dx_m=dx_m,
        dy_m=dy_m,
        grid_first_cell_xy_m=grid_first_cell_xy_m,
        threshold_provenance=threshold_provenance,
    )
    current = extract_species_isofront(
        current_reaction_species_progress,
        sampling_frame,
        species_threshold=species_threshold,
        dx_m=dx_m,
        dy_m=dy_m,
        grid_first_cell_xy_m=grid_first_cell_xy_m,
        threshold_provenance=threshold_provenance,
    )
    return measure_normal_regression(previous, current, elapsed_time_s=elapsed_time_s)


def evaluate_species_threshold_sensitivity(
    previous_reaction_species_progress: np.ndarray,
    current_reaction_species_progress: np.ndarray,
    sampling_frame: NormalSamplingFrame,
    *,
    elapsed_time_s: float,
    dx_m: float,
    dy_m: float | None = None,
    grid_first_cell_xy_m: tuple[float, float] = (0.0, 0.0),
    species_thresholds: Iterable[float] = ASSUMED_SPECIES_THRESHOLD_SET,
) -> ThresholdSensitivityResult:
    """Evaluate the documented non-paper threshold set on identical rays."""

    thresholds = tuple(_validated_threshold(value) for value in species_thresholds)
    if not thresholds:
        raise ReactiveConfigurationError(
            "Species threshold sensitivity set cannot be empty"
        )
    if len(set(thresholds)) != len(thresholds):
        raise ReactiveConfigurationError(
            "Species threshold sensitivity values must be unique"
        )
    results = {
        threshold: track_species_isofront_motion(
            previous_reaction_species_progress,
            current_reaction_species_progress,
            sampling_frame,
            elapsed_time_s=elapsed_time_s,
            species_threshold=threshold,
            dx_m=dx_m,
            dy_m=dy_m,
            grid_first_cell_xy_m=grid_first_cell_xy_m,
            threshold_provenance=ASSUMED_SPECIES_THRESHOLD_PROVENANCE,
        )
        for threshold in thresholds
    }
    finite_means = np.array(
        [
            result.mean_normal_regression_speed_m_s
            for result in results.values()
            if math.isfinite(result.mean_normal_regression_speed_m_s)
        ],
        dtype=np.float64,
    )
    if finite_means.size:
        minimum = float(np.min(finite_means))
        maximum = float(np.max(finite_means))
        span = maximum - minimum
    else:
        minimum = maximum = span = math.nan
    reference = results.get(0.5)
    if reference is None or not math.isfinite(
        reference.mean_normal_regression_speed_m_s
    ):
        maximum_deviation = math.nan
    else:
        reference_speed = reference.mean_normal_regression_speed_m_s
        maximum_deviation = float(
            max(
                (
                    abs(value.mean_normal_regression_speed_m_s - reference_speed)
                    for value in results.values()
                    if math.isfinite(value.mean_normal_regression_speed_m_s)
                ),
                default=math.nan,
            )
        )
    return ThresholdSensitivityResult(
        threshold_provenance=ASSUMED_SPECIES_THRESHOLD_PROVENANCE,
        by_threshold=results,
        finite_mean_speed_count=int(finite_means.size),
        mean_speed_minimum_m_s=minimum,
        mean_speed_maximum_m_s=maximum,
        mean_speed_span_m_s=span,
        maximum_absolute_deviation_from_threshold_0p5_m_s=maximum_deviation,
    )


__all__ = [
    "ASSUMED_SPECIES_THRESHOLD_PROVENANCE",
    "ASSUMED_SPECIES_THRESHOLD_SET",
    "FrontTransitionDiagnostics",
    "FRONT_TRACKING_METHOD",
    "FRONT_TRACKING_METHOD_PROVENANCE",
    "IsofrontDiagnostics",
    "NormalRegressionResult",
    "NormalSamplingFrame",
    "SpeciesIsofrontSnapshot",
    "ThresholdSensitivityResult",
    "axis_aligned_sampling_frame",
    "evaluate_species_threshold_sensitivity",
    "extract_species_isofront",
    "measure_normal_regression",
    "track_species_isofront_motion",
]
