"""Transported material level set for the paper-reactive solver.

This module deliberately does *not* accept reaction progress, thermal
decomposition progress, or an alpha field.  ``material_level_set`` represents
the material boundary and is advanced only by the supplied two-dimensional
velocity field,

    phi_t + u phi_x + v phi_y = 0.

The sign convention is fixed throughout the module: ``phi < 0`` is material
interior, ``phi == 0`` is the boundary, and ``phi > 0`` is exterior.

The paper states WENO spatial differencing and third-order Runge--Kutta time
integration, but does not fully identify a WENO variant or a reinitialisation
algorithm.  The implementation below therefore labels WENO5-JS and optional
PDE reinitialisation as ``ASSUMED_NOT_FROM_PAPER`` instead of presenting those
choices as paper data.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from .provenance import Provenance, ReactiveConfigurationError, ReactiveNumericalError


BoundaryKind = Literal["periodic", "outflow"]

MATERIAL_LEVEL_SET_SIGN_CONVENTION = (
    "phi<0 material_interior; phi=0 material_boundary; phi>0 material_exterior"
)
WENO_VARIANT = "WENO5_JS_UPWIND"
WENO_VARIANT_PROVENANCE = Provenance.ASSUMED_NOT_FROM_PAPER
SSP_RK_VARIANT = "SSPRK33"
SSP_RK_VARIANT_PROVENANCE = Provenance.ASSUMED_NOT_FROM_PAPER
BOUNDARY_CLOSURE_PROVENANCE = Provenance.ASSUMED_NOT_FROM_PAPER
ZERO_CONTOUR_AREA_METRIC = "piecewise_linear_cell_centre_mesh_clipping"
ZERO_CONTOUR_AREA_METRIC_PROVENANCE = Provenance.ASSUMED_NOT_FROM_PAPER
DEFAULT_WENO_RELATIVE_EPSILON = 1.0e-12
DEFAULT_WENO_ABSOLUTE_EPSILON_M2 = 1.0e-40


@dataclass(frozen=True)
class BoundaryCondition2D:
    """Boundary closure used to populate WENO ghost cells.

    ``outflow`` is constant (zero normal-gradient) extrapolation.  This is a
    numerical boundary closure, not a physical far-field model.
    """

    x: BoundaryKind = "outflow"
    y: BoundaryKind = "outflow"

    def __post_init__(self) -> None:
        for axis, value in (("x", self.x), ("y", self.y)):
            if value not in {"periodic", "outflow"}:
                raise ReactiveConfigurationError(
                    f"material level-set {axis} boundary must be periodic or outflow"
                )


@dataclass(frozen=True)
class ReinitializationConfig:
    """Explicit, non-paper option for signed-distance PDE reinitialisation.

    The zero contour of the *transported* field is the initial condition for
    the pseudo-time Hamilton--Jacobi solve.  It is never reconstructed from a
    reaction-progress threshold.
    """

    enabled: bool = False
    provenance: Provenance = Provenance.ASSUMED_NOT_FROM_PAPER
    method: str = "SUSSMAN_FIRST_ORDER"
    pseudo_steps: int = 60
    pseudo_cfl: float = 0.3
    narrow_band_half_width_m: float | None = None
    source: str = "Numerical option not specified in the ECSP paper"

    def validate(self) -> None:
        if not self.enabled:
            return
        try:
            provenance = Provenance(self.provenance)
        except ValueError as exc:
            raise ReactiveConfigurationError(
                "Invalid material level-set reinitialisation provenance"
            ) from exc
        if provenance is not Provenance.ASSUMED_NOT_FROM_PAPER:
            raise ReactiveConfigurationError(
                "The selected level-set reinitialisation is not specified by the paper; "
                "it must be tagged ASSUMED_NOT_FROM_PAPER"
            )
        if self.method != "SUSSMAN_FIRST_ORDER":
            raise ReactiveConfigurationError(
                "Only the explicitly documented SUSSMAN_FIRST_ORDER option is implemented"
            )
        if not isinstance(self.pseudo_steps, int) or self.pseudo_steps <= 0:
            raise ReactiveConfigurationError(
                "reinitialisation pseudo_steps must be positive"
            )
        if (
            not math.isfinite(self.pseudo_cfl)
            or self.pseudo_cfl <= 0.0
            or self.pseudo_cfl > 0.5
        ):
            raise ReactiveConfigurationError(
                "reinitialisation pseudo_cfl must lie in (0, 0.5]"
            )
        if self.narrow_band_half_width_m is not None and (
            not math.isfinite(self.narrow_band_half_width_m)
            or self.narrow_band_half_width_m <= 0.0
        ):
            raise ReactiveConfigurationError(
                "reinitialisation narrow-band half-width must be positive and finite"
            )
        if not str(self.source).strip():
            raise ReactiveConfigurationError(
                "reinitialisation requires a provenance source"
            )


@dataclass(frozen=True)
class MaterialLevelSet:
    """Cell-centred material level-set state, independent of reaction fields."""

    values_m: np.ndarray
    dx_m: float
    dy_m: float

    def __post_init__(self) -> None:
        values = _validated_phi(self.values_m, minimum_axis_size=2)
        dx, dy = _validated_spacing(self.dx_m, self.dy_m)
        owned = np.array(values, dtype=np.float64, copy=True, order="C")
        values = np.frombuffer(
            owned.tobytes(order="C"), dtype=np.float64
        ).reshape(owned.shape)
        object.__setattr__(self, "values_m", values)
        object.__setattr__(self, "dx_m", dx)
        object.__setattr__(self, "dy_m", dy)

    @property
    def interior_mask(self) -> np.ndarray:
        """Return cells whose centres lie strictly inside the material."""

        return self.values_m < 0.0

    @property
    def exterior_mask(self) -> np.ndarray:
        return self.values_m > 0.0


@dataclass(frozen=True)
class LevelSetStepDiagnostics:
    spatial_scheme: str
    spatial_scheme_provenance: Provenance
    time_integrator: str
    time_integrator_provenance: Provenance
    boundary_x: str
    boundary_y: str
    boundary_closure_provenance: Provenance
    weno_relative_epsilon: float
    weno_absolute_epsilon_m2: float
    cfl_number: float
    reinitialized: bool
    reinitialization_provenance: Provenance | None
    transported_zero_contour_area_m2: float
    output_zero_contour_area_m2: float
    reinitialization_zero_contour_area_drift_m2: float
    reinitialization_zero_contour_relative_drift: float
    sign_changed_cell_count_during_reinitialization: int
    signed_distance_rms_error_before: float
    signed_distance_rms_error_after: float
    zero_contour_area_metric: str
    zero_contour_area_metric_provenance: Provenance


@dataclass(frozen=True)
class LevelSetStepResult:
    material_level_set: MaterialLevelSet
    diagnostics: LevelSetStepDiagnostics


def _validated_spacing(dx_m: float, dy_m: float | None) -> tuple[float, float]:
    dx = float(dx_m)
    dy = dx if dy_m is None else float(dy_m)
    if not math.isfinite(dx) or not math.isfinite(dy) or dx <= 0.0 or dy <= 0.0:
        raise ReactiveConfigurationError(
            "level-set grid spacing must be positive and finite"
        )
    return dx, dy


def _validated_phi(values: np.ndarray, *, minimum_axis_size: int) -> np.ndarray:
    phi = np.asarray(values, dtype=np.float64)
    if phi.ndim != 2:
        raise ReactiveConfigurationError(
            "material_level_set must be a two-dimensional array"
        )
    if min(phi.shape) < minimum_axis_size:
        raise ReactiveConfigurationError(
            f"material_level_set axes must each contain at least {minimum_axis_size} cells"
        )
    if not np.all(np.isfinite(phi)):
        raise ReactiveNumericalError(
            "nonfinite_material_level_set",
            "material_level_set contains NaN or infinity",
        )
    return phi


def _coerce_boundary(
    boundary: BoundaryCondition2D | BoundaryKind,
) -> BoundaryCondition2D:
    if isinstance(boundary, BoundaryCondition2D):
        return boundary
    return BoundaryCondition2D(x=boundary, y=boundary)


def _pad_along_last_axis(
    values: np.ndarray, count: int, boundary: BoundaryKind
) -> np.ndarray:
    mode = "wrap" if boundary == "periodic" else "edge"
    return np.pad(values, ((0, 0), (count, count)), mode=mode)


def _weno5_backward_derivative(
    values: np.ndarray,
    spacing: float,
    *,
    axis: int,
    boundary: BoundaryKind,
    relative_epsilon: float,
    absolute_epsilon_m2: float,
) -> np.ndarray:
    """Return the WENO5-JS left-biased derivative along one array axis."""

    moved = np.moveaxis(values, axis, -1)
    count = moved.shape[-1]
    padded = _pad_along_last_axis(moved, 3, boundary)

    # Left reconstructions at faces i+1/2 for i=-1,...,N-1.  Taking their
    # difference gives the upwind derivative at every original cell centre.
    f_im2 = padded[..., 0 : count + 1]
    f_im1 = padded[..., 1 : count + 2]
    f_i = padded[..., 2 : count + 3]
    f_ip1 = padded[..., 3 : count + 4]
    f_ip2 = padded[..., 4 : count + 5]

    candidate_0 = (1.0 / 3.0) * f_im2 - (7.0 / 6.0) * f_im1 + (11.0 / 6.0) * f_i
    candidate_1 = -(1.0 / 6.0) * f_im1 + (5.0 / 6.0) * f_i + (1.0 / 3.0) * f_ip1
    candidate_2 = (1.0 / 3.0) * f_i + (5.0 / 6.0) * f_ip1 - (1.0 / 6.0) * f_ip2

    beta_0 = (13.0 / 12.0) * (f_im2 - 2.0 * f_im1 + f_i) ** 2 + 0.25 * (
        f_im2 - 4.0 * f_im1 + 3.0 * f_i
    ) ** 2
    beta_1 = (13.0 / 12.0) * (f_im1 - 2.0 * f_i + f_ip1) ** 2 + 0.25 * (
        f_im1 - f_ip1
    ) ** 2
    beta_2 = (13.0 / 12.0) * (f_i - 2.0 * f_ip1 + f_ip2) ** 2 + 0.25 * (
        3.0 * f_i - 4.0 * f_ip1 + f_ip2
    ) ** 2

    local_maximum_beta = np.maximum.reduce((beta_0, beta_1, beta_2))
    epsilon = absolute_epsilon_m2 + relative_epsilon * local_maximum_beta
    alpha_0 = 0.1 / (epsilon + beta_0) ** 2
    alpha_1 = 0.6 / (epsilon + beta_1) ** 2
    alpha_2 = 0.3 / (epsilon + beta_2) ** 2
    alpha_sum = alpha_0 + alpha_1 + alpha_2
    faces = (
        alpha_0 * candidate_0 + alpha_1 * candidate_1 + alpha_2 * candidate_2
    ) / alpha_sum

    derivative = (faces[..., 1:] - faces[..., :-1]) / spacing
    if not np.all(np.isfinite(derivative)):
        raise ReactiveNumericalError(
            "nonfinite_weno_derivative", "WENO level-set derivative became non-finite"
        )
    return np.moveaxis(derivative, -1, axis)


def weno5_one_sided_derivatives(
    values: np.ndarray,
    spacing: float,
    *,
    axis: int,
    boundary: BoundaryKind,
    relative_epsilon: float = DEFAULT_WENO_RELATIVE_EPSILON,
    absolute_epsilon_m2: float = DEFAULT_WENO_ABSOLUTE_EPSILON_M2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(D_minus, D_plus)`` WENO5-JS derivatives.

    The WENO variant is a documented numerical assumption because the paper
    does not provide enough detail to identify its exact nonlinear weights.
    """

    phi = _validated_phi(values, minimum_axis_size=5)
    if axis not in (0, 1):
        raise ReactiveConfigurationError("WENO derivative axis must be 0 or 1")
    if boundary not in {"periodic", "outflow"}:
        raise ReactiveConfigurationError("WENO boundary must be periodic or outflow")
    h = float(spacing)
    if not math.isfinite(h) or h <= 0.0:
        raise ReactiveConfigurationError("WENO spacing must be positive and finite")
    relative = float(relative_epsilon)
    absolute = float(absolute_epsilon_m2)
    if not math.isfinite(relative) or relative < 0.0:
        raise ReactiveConfigurationError(
            "level-set WENO relative epsilon must be finite and non-negative"
        )
    if not math.isfinite(absolute) or absolute <= 0.0:
        raise ReactiveConfigurationError(
            "level-set WENO absolute epsilon must be finite and positive"
        )
    backward = _weno5_backward_derivative(
        phi,
        h,
        axis=axis,
        boundary=boundary,
        relative_epsilon=relative,
        absolute_epsilon_m2=absolute,
    )
    reversed_phi = np.flip(phi, axis=axis)
    reversed_backward = _weno5_backward_derivative(
        reversed_phi,
        h,
        axis=axis,
        boundary=boundary,
        relative_epsilon=relative,
        absolute_epsilon_m2=absolute,
    )
    forward = -np.flip(reversed_backward, axis=axis)
    return backward, forward


def _broadcast_velocity(
    component: float | np.ndarray, shape: tuple[int, int], name: str
) -> np.ndarray:
    raw = np.asarray(component, dtype=np.float64)
    try:
        result = np.broadcast_to(raw, shape)
    except ValueError as exc:
        raise ReactiveConfigurationError(
            f"{name} cannot be broadcast to material_level_set shape {shape}"
        ) from exc
    if not np.all(np.isfinite(result)):
        raise ReactiveNumericalError(
            "nonfinite_level_set_velocity", f"{name} contains NaN or infinity"
        )
    return result


def level_set_advection_rhs(
    material_level_set: np.ndarray,
    velocity_x_m_s: float | np.ndarray,
    velocity_y_m_s: float | np.ndarray,
    *,
    dx_m: float,
    dy_m: float | None = None,
    boundary: BoundaryCondition2D | BoundaryKind = BoundaryCondition2D(),
    weno_relative_epsilon: float = DEFAULT_WENO_RELATIVE_EPSILON,
    weno_absolute_epsilon_m2: float = DEFAULT_WENO_ABSOLUTE_EPSILON_M2,
) -> np.ndarray:
    """Evaluate ``-(u phi_x + v phi_y)`` with sign-selected WENO derivatives."""

    phi = _validated_phi(material_level_set, minimum_axis_size=5)
    dx, dy = _validated_spacing(dx_m, dy_m)
    bc = _coerce_boundary(boundary)
    velocity_x = _broadcast_velocity(velocity_x_m_s, phi.shape, "velocity_x_m_s")
    velocity_y = _broadcast_velocity(velocity_y_m_s, phi.shape, "velocity_y_m_s")
    dphi_dx_minus, dphi_dx_plus = weno5_one_sided_derivatives(
        phi,
        dx,
        axis=1,
        boundary=bc.x,
        relative_epsilon=weno_relative_epsilon,
        absolute_epsilon_m2=weno_absolute_epsilon_m2,
    )
    dphi_dy_minus, dphi_dy_plus = weno5_one_sided_derivatives(
        phi,
        dy,
        axis=0,
        boundary=bc.y,
        relative_epsilon=weno_relative_epsilon,
        absolute_epsilon_m2=weno_absolute_epsilon_m2,
    )
    dphi_dx = np.where(velocity_x >= 0.0, dphi_dx_minus, dphi_dx_plus)
    dphi_dy = np.where(velocity_y >= 0.0, dphi_dy_minus, dphi_dy_plus)
    rhs = -(velocity_x * dphi_dx + velocity_y * dphi_dy)
    if not np.all(np.isfinite(rhs)):
        raise ReactiveNumericalError(
            "nonfinite_level_set_rhs",
            "material level-set advection RHS became non-finite",
        )
    return rhs


def recommended_level_set_timestep_s(
    velocity_x_m_s: float | np.ndarray,
    velocity_y_m_s: float | np.ndarray,
    *,
    shape: tuple[int, int],
    dx_m: float,
    dy_m: float | None = None,
    target_cfl: float = 0.4,
) -> float:
    """Return the advective step from ``CFL/(|u|/dx+|v|/dy)``.

    A zero velocity field has no finite advective restriction and returns
    ``math.inf``; a coupled solver must then apply its other restrictions.
    """

    dx, dy = _validated_spacing(dx_m, dy_m)
    if not math.isfinite(target_cfl) or target_cfl <= 0.0 or target_cfl > 1.0:
        raise ReactiveConfigurationError("target_cfl must lie in (0, 1]")
    velocity_x = _broadcast_velocity(velocity_x_m_s, shape, "velocity_x_m_s")
    velocity_y = _broadcast_velocity(velocity_y_m_s, shape, "velocity_y_m_s")
    spectral_rate = float(np.max(np.abs(velocity_x) / dx + np.abs(velocity_y) / dy))
    if spectral_rate == 0.0:
        return math.inf
    return target_cfl / spectral_rate


def _ssprk3(
    state: np.ndarray,
    dt: float,
    rhs,
) -> np.ndarray:
    stage_1 = state + dt * rhs(state)
    stage_2 = 0.75 * state + 0.25 * (stage_1 + dt * rhs(stage_1))
    output = (1.0 / 3.0) * state + (2.0 / 3.0) * (stage_2 + dt * rhs(stage_2))
    return np.asarray(output, dtype=np.float64)


def _first_differences(
    values: np.ndarray,
    spacing: float,
    *,
    axis: int,
    boundary: BoundaryKind,
) -> tuple[np.ndarray, np.ndarray]:
    moved = np.moveaxis(values, axis, -1)
    padded = _pad_along_last_axis(moved, 1, boundary)
    centre = padded[..., 1:-1]
    backward = (centre - padded[..., :-2]) / spacing
    forward = (padded[..., 2:] - centre) / spacing
    return np.moveaxis(backward, -1, axis), np.moveaxis(forward, -1, axis)


def _centred_gradient_components(
    values: np.ndarray,
    dx_m: float,
    dy_m: float,
    boundary: BoundaryCondition2D,
) -> tuple[np.ndarray, np.ndarray]:
    x_backward, x_forward = _first_differences(
        values, dx_m, axis=1, boundary=boundary.x
    )
    y_backward, y_forward = _first_differences(
        values, dy_m, axis=0, boundary=boundary.y
    )
    return 0.5 * (x_backward + x_forward), 0.5 * (y_backward + y_forward)


def signed_distance_rms_error(
    material_level_set: np.ndarray,
    *,
    dx_m: float,
    dy_m: float | None = None,
    boundary: BoundaryCondition2D | BoundaryKind = BoundaryCondition2D(),
    half_band_width_m: float | None = None,
) -> float:
    """RMS of ``abs(grad(phi))-1`` in a contour-centred narrow band."""

    phi = _validated_phi(material_level_set, minimum_axis_size=2)
    dx, dy = _validated_spacing(dx_m, dy_m)
    bc = _coerce_boundary(boundary)
    gradient_x, gradient_y = _centred_gradient_components(phi, dx, dy, bc)
    gradient_norm = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    width = 3.0 * max(dx, dy) if half_band_width_m is None else float(half_band_width_m)
    if not math.isfinite(width) or width <= 0.0:
        raise ReactiveConfigurationError(
            "signed-distance error band width must be positive"
        )
    band = np.abs(phi) <= width
    if not np.any(band):
        raise ReactiveNumericalError(
            "level_set_missing_narrow_band",
            "No material level-set cells lie in the signed-distance error band",
        )
    error = float(np.sqrt(np.mean((gradient_norm[band] - 1.0) ** 2)))
    if not math.isfinite(error):
        raise ReactiveNumericalError(
            "nonfinite_signed_distance_error", "signed-distance error became non-finite"
        )
    return error


def _negative_triangle_fraction(values: tuple[float, float, float]) -> float:
    """Area fraction below zero for a scalar linear over a unit triangle."""

    vertices = [
        (np.array((0.0, 0.0)), values[0]),
        (np.array((1.0, 0.0)), values[1]),
        (np.array((1.0, 1.0)), values[2]),
    ]
    polygon: list[np.ndarray] = []
    for index, (point_a, value_a) in enumerate(vertices):
        point_b, value_b = vertices[(index + 1) % len(vertices)]
        inside_a = value_a <= 0.0
        inside_b = value_b <= 0.0
        if inside_a:
            polygon.append(point_a)
        if inside_a != inside_b:
            denominator = value_a - value_b
            fraction = 0.5 if denominator == 0.0 else value_a / denominator
            polygon.append(point_a + fraction * (point_b - point_a))
    if len(polygon) < 3:
        return 0.0
    points = np.asarray(polygon, dtype=np.float64)
    twice_area = abs(
        float(
            np.dot(points[:, 0], np.roll(points[:, 1], -1))
            - np.dot(points[:, 1], np.roll(points[:, 0], -1))
        )
    )
    # The reference triangle has area 1/2.
    return min(1.0, max(0.0, twice_area))


def zero_contour_interior_area_m2(
    material_level_set: np.ndarray,
    *,
    dx_m: float,
    dy_m: float | None = None,
    boundary: BoundaryCondition2D | BoundaryKind = BoundaryCondition2D(),
) -> float:
    """Piecewise-linear area on the cell-centre mesh where ``phi <= 0``.

    Each rectangle between four cell centres is split along the lower-left to
    upper-right diagonal.  Linear clipping of both triangles gives a subcell
    zero-contour metric that responds to interface movement; it is not the
    legacy consumed-cell-area/front-length proxy.  Periodic directions include
    the wrapped last-to-first rectangle.  An outflow direction measures the
    cell-centre hull and does not invent a subcell physical boundary location.
    """

    phi = _validated_phi(material_level_set, minimum_axis_size=2)
    dx, dy = _validated_spacing(dx_m, dy_m)
    bc = _coerce_boundary(boundary)
    total_fraction = 0.0
    row_count = phi.shape[0] if bc.y == "periodic" else phi.shape[0] - 1
    column_count = phi.shape[1] if bc.x == "periodic" else phi.shape[1] - 1
    for row in range(row_count):
        next_row = (row + 1) % phi.shape[0]
        for column in range(column_count):
            next_column = (column + 1) % phi.shape[1]
            lower_left = float(phi[row, column])
            lower_right = float(phi[row, next_column])
            upper_left = float(phi[next_row, column])
            upper_right = float(phi[next_row, next_column])
            total_fraction += 0.5 * _negative_triangle_fraction(
                (lower_left, lower_right, upper_right)
            )
            # Map the second physical triangle to the helper's reference
            # triangle; only scalar vertex values matter for its area fraction.
            total_fraction += 0.5 * _negative_triangle_fraction(
                (lower_left, upper_right, upper_left)
            )
    area = total_fraction * dx * dy
    if not math.isfinite(area) or area < 0.0:
        raise ReactiveNumericalError(
            "nonfinite_zero_contour_area", "zero-contour interior area became invalid"
        )
    return float(area)


def reinitialize_transported_level_set(
    transported_level_set: np.ndarray,
    *,
    dx_m: float,
    dy_m: float | None = None,
    boundary: BoundaryCondition2D | BoundaryKind = BoundaryCondition2D(),
    config: ReinitializationConfig,
) -> np.ndarray:
    """Reinitialise the transported zero contour by a Godunov HJ pseudo-solve."""

    config.validate()
    if not config.enabled:
        return np.array(transported_level_set, dtype=np.float64, copy=True)
    phi_zero = _validated_phi(transported_level_set, minimum_axis_size=2)
    dx, dy = _validated_spacing(dx_m, dy_m)
    bc = _coerce_boundary(boundary)
    if not (float(np.min(phi_zero)) <= 0.0 <= float(np.max(phi_zero))):
        raise ReactiveNumericalError(
            "level_set_missing_interface",
            "Cannot reinitialise a material level set with no zero contour",
        )
    gradient_x, gradient_y = _centred_gradient_components(phi_zero, dx, dy, bc)
    gradient_squared = gradient_x * gradient_x + gradient_y * gradient_y
    h = min(dx, dy)
    frozen_sign = phi_zero / np.sqrt(
        phi_zero * phi_zero + h * h * gradient_squared + 1.0e-300
    )
    if config.narrow_band_half_width_m is None:
        update_mask = np.ones(phi_zero.shape, dtype=bool)
    else:
        update_mask = np.abs(phi_zero) <= config.narrow_band_half_width_m
    pseudo_dt = config.pseudo_cfl * h

    def reinitialisation_rhs(state: np.ndarray) -> np.ndarray:
        dx_minus, dx_plus = _first_differences(state, dx, axis=1, boundary=bc.x)
        dy_minus, dy_plus = _first_differences(state, dy, axis=0, boundary=bc.y)
        positive_gradient_squared = (
            np.maximum(dx_minus, 0.0) ** 2
            + np.minimum(dx_plus, 0.0) ** 2
            + np.maximum(dy_minus, 0.0) ** 2
            + np.minimum(dy_plus, 0.0) ** 2
        )
        negative_gradient_squared = (
            np.minimum(dx_minus, 0.0) ** 2
            + np.maximum(dx_plus, 0.0) ** 2
            + np.minimum(dy_minus, 0.0) ** 2
            + np.maximum(dy_plus, 0.0) ** 2
        )
        gradient = np.sqrt(
            np.where(
                frozen_sign >= 0.0, positive_gradient_squared, negative_gradient_squared
            )
        )
        result = -frozen_sign * (gradient - 1.0)
        return np.where(update_mask, result, 0.0)

    output = np.array(phi_zero, dtype=np.float64, copy=True)
    for _ in range(config.pseudo_steps):
        output = _ssprk3(output, pseudo_dt, reinitialisation_rhs)
        if not np.all(np.isfinite(output)):
            raise ReactiveNumericalError(
                "nonfinite_level_set_reinitialisation",
                "material level-set reinitialisation became non-finite",
            )
    return output


def advect_material_level_set(
    material_level_set: np.ndarray | MaterialLevelSet,
    velocity_x_m_s: float | np.ndarray,
    velocity_y_m_s: float | np.ndarray,
    *,
    dt_s: float,
    dx_m: float | None = None,
    dy_m: float | None = None,
    boundary: BoundaryCondition2D | BoundaryKind = BoundaryCondition2D(),
    cfl_limit: float = 0.6,
    reinitialization: ReinitializationConfig | None = None,
    weno_relative_epsilon: float = DEFAULT_WENO_RELATIVE_EPSILON,
    weno_absolute_epsilon_m2: float = DEFAULT_WENO_ABSOLUTE_EPSILON_M2,
) -> LevelSetStepResult:
    """Advance a material level set by one WENO5/SSP-RK3 physical time step."""

    if isinstance(material_level_set, MaterialLevelSet):
        if dx_m is not None or dy_m is not None:
            raise ReactiveConfigurationError(
                "Do not repeat grid spacing when passing a MaterialLevelSet object"
            )
        phi = np.asarray(material_level_set.values_m)
        dx = material_level_set.dx_m
        dy = material_level_set.dy_m
    else:
        if dx_m is None:
            raise ReactiveConfigurationError(
                "dx_m is required for a raw level-set array"
            )
        phi = _validated_phi(material_level_set, minimum_axis_size=5)
        dx, dy = _validated_spacing(dx_m, dy_m)
    phi = _validated_phi(phi, minimum_axis_size=5)
    dt = float(dt_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ReactiveConfigurationError(
            "material level-set dt_s must be positive and finite"
        )
    if not math.isfinite(cfl_limit) or cfl_limit <= 0.0 or cfl_limit > 1.0:
        raise ReactiveConfigurationError(
            "material level-set cfl_limit must lie in (0, 1]"
        )
    bc = _coerce_boundary(boundary)
    velocity_x = _broadcast_velocity(velocity_x_m_s, phi.shape, "velocity_x_m_s")
    velocity_y = _broadcast_velocity(velocity_y_m_s, phi.shape, "velocity_y_m_s")
    cfl_number = dt * float(np.max(np.abs(velocity_x) / dx + np.abs(velocity_y) / dy))
    if cfl_number > cfl_limit * (1.0 + 16.0 * np.finfo(np.float64).eps):
        raise ReactiveNumericalError(
            "level_set_cfl_violation",
            f"material level-set CFL {cfl_number:.9g} exceeds limit {cfl_limit:.9g}",
            {"cfl_number": cfl_number, "cfl_limit": cfl_limit},
        )

    def physical_rhs(state: np.ndarray) -> np.ndarray:
        return level_set_advection_rhs(
            state,
            velocity_x,
            velocity_y,
            dx_m=dx,
            dy_m=dy,
            boundary=bc,
            weno_relative_epsilon=weno_relative_epsilon,
            weno_absolute_epsilon_m2=weno_absolute_epsilon_m2,
        )

    transported = _ssprk3(phi, dt, physical_rhs)
    if not np.all(np.isfinite(transported)):
        raise ReactiveNumericalError(
            "nonfinite_material_level_set",
            "material level set became non-finite after physical transport",
        )
    transported_area = zero_contour_interior_area_m2(
        transported, dx_m=dx, dy_m=dy, boundary=bc
    )
    distance_error_before = signed_distance_rms_error(
        transported, dx_m=dx, dy_m=dy, boundary=bc
    )

    config = (
        ReinitializationConfig(enabled=False)
        if reinitialization is None
        else reinitialization
    )
    config.validate()
    if config.enabled:
        output = reinitialize_transported_level_set(
            transported,
            dx_m=dx,
            dy_m=dy,
            boundary=bc,
            config=config,
        )
        provenance: Provenance | None = Provenance(config.provenance)
    else:
        output = np.array(transported, copy=True)
        provenance = None
    output_area = zero_contour_interior_area_m2(output, dx_m=dx, dy_m=dy, boundary=bc)
    area_drift = output_area - transported_area
    relative_drift = area_drift / max(abs(transported_area), dx * dy)
    sign_changed = int(np.count_nonzero(np.signbit(output) != np.signbit(transported)))
    distance_error_after = signed_distance_rms_error(
        output, dx_m=dx, dy_m=dy, boundary=bc
    )
    result = MaterialLevelSet(output, dx, dy)
    diagnostics = LevelSetStepDiagnostics(
        spatial_scheme=WENO_VARIANT,
        spatial_scheme_provenance=WENO_VARIANT_PROVENANCE,
        time_integrator=SSP_RK_VARIANT,
        time_integrator_provenance=SSP_RK_VARIANT_PROVENANCE,
        boundary_x=bc.x,
        boundary_y=bc.y,
        boundary_closure_provenance=BOUNDARY_CLOSURE_PROVENANCE,
        weno_relative_epsilon=float(weno_relative_epsilon),
        weno_absolute_epsilon_m2=float(weno_absolute_epsilon_m2),
        cfl_number=cfl_number,
        reinitialized=config.enabled,
        reinitialization_provenance=provenance,
        transported_zero_contour_area_m2=transported_area,
        output_zero_contour_area_m2=output_area,
        reinitialization_zero_contour_area_drift_m2=area_drift,
        reinitialization_zero_contour_relative_drift=relative_drift,
        sign_changed_cell_count_during_reinitialization=sign_changed,
        signed_distance_rms_error_before=distance_error_before,
        signed_distance_rms_error_after=distance_error_after,
        zero_contour_area_metric=ZERO_CONTOUR_AREA_METRIC,
        zero_contour_area_metric_provenance=ZERO_CONTOUR_AREA_METRIC_PROVENANCE,
    )
    return LevelSetStepResult(material_level_set=result, diagnostics=diagnostics)


# A concise alias for callers that already use unambiguous variable names.  It
# still returns a ``material_level_set`` field and accepts no reaction progress.
advect_level_set = advect_material_level_set


__all__ = [
    "BoundaryCondition2D",
    "BOUNDARY_CLOSURE_PROVENANCE",
    "DEFAULT_WENO_ABSOLUTE_EPSILON_M2",
    "DEFAULT_WENO_RELATIVE_EPSILON",
    "LevelSetStepDiagnostics",
    "LevelSetStepResult",
    "MATERIAL_LEVEL_SET_SIGN_CONVENTION",
    "MaterialLevelSet",
    "ReinitializationConfig",
    "SSP_RK_VARIANT",
    "SSP_RK_VARIANT_PROVENANCE",
    "WENO_VARIANT",
    "WENO_VARIANT_PROVENANCE",
    "ZERO_CONTOUR_AREA_METRIC",
    "ZERO_CONTOUR_AREA_METRIC_PROVENANCE",
    "advect_level_set",
    "advect_material_level_set",
    "level_set_advection_rhs",
    "recommended_level_set_timestep_s",
    "reinitialize_transported_level_set",
    "signed_distance_rms_error",
    "weno5_one_sided_derivatives",
    "zero_contour_interior_area_m2",
]
