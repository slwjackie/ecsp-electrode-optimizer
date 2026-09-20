"""Reference FP64 finite-volume numerics for the reactive Euler equations.

The paper states WENO spatial differencing and third-order Runge--Kutta time
integration, but does not identify the WENO variant, reconstruction variables,
Riemann solver, epsilon, or domain boundary closures.  This reference chooses
primitive-variable, component-wise WENO5-JS-like reconstruction, HLL, SSP-RK3
and explicit boundary types.  The input primitive values are derived from cell
averages and are not exact primitive-variable cell averages; all such choices
are labelled ``ASSUMED_NOT_FROM_PAPER`` below.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np

from .core import (
    MOMENTUM_X,
    MOMENTUM_Y,
    NCONS,
    REACTION_PROGRESS,
    RHO,
    TOTAL_ENERGY,
    conservative_to_primitive,
    flux_x,
    flux_y,
    physical_state_mask,
)
from .eos import EquationOfState
from .provenance import ReactiveConfigurationError, ReactiveNumericalError


NUMERICAL_PROVENANCE: dict[str, dict[str, str]] = {
    "reconstruction": {
        "choice": "primitive_from_conservative_cell_averages_componentwise_face_local_WENO5_JS_assumption",
        "provenance": "ASSUMED_NOT_FROM_PAPER",
        "paper_statement": "WENO order and variant are not specified",
    },
    "riemann_solver": {
        "choice": "HLL",
        "provenance": "ASSUMED_NOT_FROM_PAPER",
        "paper_statement": "Riemann solver / flux splitting is not specified",
    },
    "time_integrator": {
        "choice": "SSP_RK3_Shu_Osher",
        "provenance": "ASSUMED_NOT_FROM_PAPER",
        "paper_statement": "only third-order Runge-Kutta is specified",
    },
    "boundary_conditions": {
        "choice": "explicit_periodic_transmissive_or_reflective",
        "provenance": "ASSUMED_NOT_FROM_PAPER",
        "paper_statement": "external flow boundary conditions are not specified",
    },
    "positivity_fallback": {
        "choice": "face_local_first_order_HLL_on_invalid_WENO_state",
        "provenance": "ASSUMED_NOT_FROM_PAPER",
        "paper_statement": "positivity treatment is not specified",
    },
}


class BoundaryType(str, Enum):
    PERIODIC = "periodic"
    TRANSMISSIVE = "transmissive"
    REFLECTIVE = "reflective"


@dataclass(frozen=True)
class FluxDiagnostics:
    interface_count: int
    positivity_face_fallback_count: int
    positivity_face_fallback_max_abs_state_correction: float
    boundary_type: str
    reconstruction: str = (
        "primitive_from_conservative_cell_averages_componentwise_face_local_WENO5_JS_assumption"
    )
    riemann_solver: str = "HLL"

    @property
    def positivity_face_fallback_fraction(self) -> float:
        if self.interface_count == 0:
            return 0.0
        return self.positivity_face_fallback_count / self.interface_count

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "interface_count": self.interface_count,
            "positivity_face_fallback_count": self.positivity_face_fallback_count,
            "positivity_face_fallback_fraction": self.positivity_face_fallback_fraction,
            "positivity_face_fallback_max_abs_state_correction": (
                self.positivity_face_fallback_max_abs_state_correction
            ),
            "boundary_type": self.boundary_type,
            "reconstruction": self.reconstruction,
            "riemann_solver": self.riemann_solver,
        }


def _boundary_value(boundary: BoundaryType | str) -> BoundaryType:
    try:
        return (
            boundary
            if isinstance(boundary, BoundaryType)
            else BoundaryType(str(boundary))
        )
    except ValueError as exc:
        raise ReactiveConfigurationError(
            f"boundary must be one of {[member.value for member in BoundaryType]}"
        ) from exc


def _normalise_spatial_axis(array: np.ndarray, spatial_axis: int) -> int:
    if array.ndim < 2:
        raise ReactiveConfigurationError(
            "state/reconstruction array needs a spatial and component axis"
        )
    axis = int(spatial_axis)
    if axis < 0:
        axis += array.ndim
    if axis < 0 or axis >= array.ndim - 1:
        raise ReactiveConfigurationError(
            "spatial_axis must select an axis before the final component axis"
        )
    return axis


def _pad_spatial(
    values: np.ndarray,
    ghost_cells: int,
    spatial_axis: int,
    boundary: BoundaryType | str,
) -> np.ndarray:
    boundary_kind = _boundary_value(boundary)
    axis = _normalise_spatial_axis(values, spatial_axis)
    moved = np.moveaxis(values, axis, 0)
    count = moved.shape[0]
    if ghost_cells < 1 or count < ghost_cells:
        raise ReactiveConfigurationError(
            f"need at least {ghost_cells} cells along reconstruction axis, received {count}"
        )
    if boundary_kind is BoundaryType.PERIODIC:
        indices = np.arange(-ghost_cells, count + ghost_cells, dtype=np.int64) % count
        padded = moved[indices]
    elif boundary_kind is BoundaryType.TRANSMISSIVE:
        padded = np.concatenate(
            (
                np.repeat(moved[:1], ghost_cells, axis=0),
                moved,
                np.repeat(moved[-1:], ghost_cells, axis=0),
            ),
            axis=0,
        )
    else:
        if moved.shape[-1] != NCONS:
            raise ReactiveConfigurationError(
                "reflective boundary requires a conservative state with five components"
            )
        left = moved[:ghost_cells][::-1].copy()
        right = moved[-ghost_cells:][::-1].copy()
        normal_momentum = MOMENTUM_X if axis == values.ndim - 2 else MOMENTUM_Y
        # In conventional (ny,nx,nvar), axis 1 is x and axis 0 is y.  For a
        # one-dimensional (nx,nvar) state, axis 0 is x.
        if values.ndim == 2:
            normal_momentum = MOMENTUM_X
        left[..., normal_momentum] *= -1.0
        right[..., normal_momentum] *= -1.0
        padded = np.concatenate((left, moved, right), axis=0)
    return np.moveaxis(padded, 0, axis)


def _weno_js_weights(
    beta_0: np.ndarray,
    beta_1: np.ndarray,
    beta_2: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return WENO-JS weights without squaring a tiny denominator inversely.

    Directly evaluating ``d_k / (epsilon + beta_k)**2`` overflows for valid but
    very small positive epsilon.  Scaling every denominator by their pointwise
    minimum is algebraically equivalent after normalization and keeps all
    intermediate weights bounded by their linear coefficient.
    """

    denominators = np.stack(
        (epsilon + beta_0, epsilon + beta_1, epsilon + beta_2), axis=0
    )
    if not np.isfinite(denominators).all() or np.any(denominators <= 0.0):
        raise ReactiveNumericalError(
            "invalid_weno_smoothness_denominator",
            "WENO-JS smoothness denominator became non-finite or non-positive",
        )
    scale = np.min(denominators, axis=0)
    ratios = scale[None, ...] / denominators
    scaled = np.asarray((0.1, 0.6, 0.3), dtype=np.float64).reshape(
        (3,) + (1,) * beta_0.ndim
    ) * ratios**2
    weight_sum = np.sum(scaled, axis=0)
    if not np.isfinite(weight_sum).all() or np.any(weight_sum <= 0.0):
        raise ReactiveNumericalError(
            "invalid_weno_weight_sum", "WENO-JS nonlinear weights are invalid"
        )
    weights = scaled / weight_sum[None, ...]
    return weights[0], weights[1], weights[2]


def weno5_js_reconstruct(
    cell_average: Any,
    *,
    spatial_axis: int = 0,
    boundary: BoundaryType | str = BoundaryType.PERIODIC,
    epsilon: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return left/right states at all ``n+1`` finite-volume faces.

    Reconstruction is independently component-wise.  Inputs must have a final
    component axis, e.g. ``(nx,nvar)`` or ``(ny,nx,nvar)``.
    """
    values = np.asarray(cell_average, dtype=np.float64)
    axis = _normalise_spatial_axis(values, spatial_axis)
    if values.shape[axis] < 5:
        raise ReactiveConfigurationError("WENO5 requires at least five cells")
    if not np.isfinite(values).all():
        raise ReactiveNumericalError(
            "nonfinite_weno_input", "WENO input contains NaN or infinity"
        )
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ReactiveConfigurationError("WENO-JS epsilon must be finite and positive")

    # Each component has different physical units and can differ
    # by many orders of magnitude.  Applying one absolute epsilon directly to
    # them changes the nonlinear weights under a mere unit rescaling.  Each
    # left/right trace gets its own affine normalization over exactly its five
    # cells.  Sharing a six-cell scale would contaminate the left trace with
    # fp3 (and the right trace with fm2), outside the mathematical WENO5
    # stencil. ``epsilon`` is dimensionless after this normalization.
    padded = np.moveaxis(_pad_spatial(values, 3, axis, boundary), axis, 0)
    count = values.shape[axis]
    face = np.arange(count + 1, dtype=np.int64)
    center = 3 + face - 1
    fm2 = padded[center - 2]
    fm1 = padded[center - 1]
    f0 = padded[center]
    fp1 = padded[center + 1]
    fp2 = padded[center + 2]
    fp3 = padded[center + 3]

    left_center = (fm2 + fm1 + f0 + fp1 + fp2) / 5.0
    left_scale = np.maximum.reduce(
        (
            np.abs(fm2 - left_center),
            np.abs(fm1 - left_center),
            np.abs(f0 - left_center),
            np.abs(fp1 - left_center),
            np.abs(fp2 - left_center),
        )
    )
    left_scale = np.where(left_scale > 0.0, left_scale, 1.0)
    lfm2 = (fm2 - left_center) / left_scale
    lfm1 = (fm1 - left_center) / left_scale
    lf0 = (f0 - left_center) / left_scale
    lfp1 = (fp1 - left_center) / left_scale
    lfp2 = (fp2 - left_center) / left_scale

    candidate_0 = (2.0 * lfm2 - 7.0 * lfm1 + 11.0 * lf0) / 6.0
    candidate_1 = (-lfm1 + 5.0 * lf0 + 2.0 * lfp1) / 6.0
    candidate_2 = (2.0 * lf0 + 5.0 * lfp1 - lfp2) / 6.0
    beta_0 = (13.0 / 12.0) * (lfm2 - 2.0 * lfm1 + lf0) ** 2 + 0.25 * (
        lfm2 - 4.0 * lfm1 + 3.0 * lf0
    ) ** 2
    beta_1 = (13.0 / 12.0) * (lfm1 - 2.0 * lf0 + lfp1) ** 2 + 0.25 * (
        lfm1 - lfp1
    ) ** 2
    beta_2 = (13.0 / 12.0) * (lf0 - 2.0 * lfp1 + lfp2) ** 2 + 0.25 * (
        3.0 * lf0 - 4.0 * lfp1 + lfp2
    ) ** 2
    alpha_0, alpha_1, alpha_2 = _weno_js_weights(
        beta_0, beta_1, beta_2, epsilon
    )
    left_normalized = (
        alpha_0 * candidate_0 + alpha_1 * candidate_1 + alpha_2 * candidate_2
    )

    right_center = (fm1 + f0 + fp1 + fp2 + fp3) / 5.0
    right_scale = np.maximum.reduce(
        (
            np.abs(fm1 - right_center),
            np.abs(f0 - right_center),
            np.abs(fp1 - right_center),
            np.abs(fp2 - right_center),
            np.abs(fp3 - right_center),
        )
    )
    right_scale = np.where(right_scale > 0.0, right_scale, 1.0)
    rfm1 = (fm1 - right_center) / right_scale
    rf0 = (f0 - right_center) / right_scale
    rfp1 = (fp1 - right_center) / right_scale
    rfp2 = (fp2 - right_center) / right_scale
    rfp3 = (fp3 - right_center) / right_scale

    candidate_0 = (2.0 * rfp3 - 7.0 * rfp2 + 11.0 * rfp1) / 6.0
    candidate_1 = (-rfp2 + 5.0 * rfp1 + 2.0 * rf0) / 6.0
    candidate_2 = (2.0 * rfp1 + 5.0 * rf0 - rfm1) / 6.0
    beta_0 = (13.0 / 12.0) * (rfp3 - 2.0 * rfp2 + rfp1) ** 2 + 0.25 * (
        rfp3 - 4.0 * rfp2 + 3.0 * rfp1
    ) ** 2
    beta_1 = (13.0 / 12.0) * (rfp2 - 2.0 * rfp1 + rf0) ** 2 + 0.25 * (
        rfp2 - rf0
    ) ** 2
    beta_2 = (13.0 / 12.0) * (rfp1 - 2.0 * rf0 + rfm1) ** 2 + 0.25 * (
        3.0 * rfp1 - 4.0 * rf0 + rfm1
    ) ** 2
    alpha_0, alpha_1, alpha_2 = _weno_js_weights(
        beta_0, beta_1, beta_2, epsilon
    )
    right_normalized = (
        alpha_0 * candidate_0 + alpha_1 * candidate_1 + alpha_2 * candidate_2
    )

    if not np.isfinite(left_normalized).all() or not np.isfinite(
        right_normalized
    ).all():
        raise ReactiveNumericalError(
            "nonfinite_weno_state", "WENO reconstruction became non-finite"
        )
    left = left_normalized * left_scale + left_center
    right = right_normalized * right_scale + right_center
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ReactiveNumericalError(
            "nonfinite_weno_denormalization",
            "WENO reconstruction became non-finite while restoring physical units",
        )
    return np.moveaxis(left, 0, axis), np.moveaxis(right, 0, axis)


def _first_order_face_states(
    conservative_state: np.ndarray,
    spatial_axis: int,
    boundary: BoundaryType | str,
) -> tuple[np.ndarray, np.ndarray]:
    axis = _normalise_spatial_axis(conservative_state, spatial_axis)
    padded = np.moveaxis(_pad_spatial(conservative_state, 1, axis, boundary), axis, 0)
    count = conservative_state.shape[axis]
    left = padded[: count + 1]
    right = padded[1 : count + 2]
    return np.moveaxis(left, 0, axis), np.moveaxis(right, 0, axis)


def _primitive_reconstruction_fields(
    conservative_state: np.ndarray, eos: EquationOfState
) -> np.ndarray:
    primitive = conservative_to_primitive(conservative_state, eos)
    return np.stack(
        (
            primitive.density_kg_per_m3,
            primitive.velocity_x_m_per_s,
            primitive.velocity_y_m_per_s,
            primitive.temperature_K,
            primitive.reaction_progress,
        ),
        axis=-1,
    )


def _primitive_faces_to_conservative(
    primitive_faces: np.ndarray, eos: EquationOfState
) -> np.ndarray:
    """Map possibly nonphysical reconstructed primitives without clipping.

    Invalid values are intentionally carried to ``physical_state_mask`` so
    the caller can replace only those faces with first-order cell averages.
    """

    rho = primitive_faces[..., RHO]
    velocity_x = primitive_faces[..., MOMENTUM_X]
    velocity_y = primitive_faces[..., MOMENTUM_Y]
    temperature = primitive_faces[..., TOTAL_ENERGY]
    progress = primitive_faces[..., REACTION_PROGRESS]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        internal = np.asarray(
            eos.specific_internal_energy_from_temperature(
                rho, temperature, validate=False
            ),
            dtype=np.float64,
        )
        total_specific = internal + 0.5 * (velocity_x**2 + velocity_y**2)
        return np.stack(
            (
                rho,
                rho * velocity_x,
                rho * velocity_y,
                rho * total_specific,
                rho * progress,
            ),
            axis=-1,
        )


def hll_flux(
    left_state: Any,
    right_state: Any,
    eos: EquationOfState,
    *,
    direction: str,
) -> np.ndarray:
    """Compute a two-wave HLL interface flux from physical states."""
    left = np.asarray(left_state, dtype=np.float64)
    right = np.asarray(right_state, dtype=np.float64)
    if left.shape != right.shape or left.ndim < 1 or left.shape[-1] != NCONS:
        raise ReactiveConfigurationError(
            "HLL left/right states must have one matching (...,5) shape"
        )
    if direction not in {"x", "y"}:
        raise ReactiveConfigurationError("HLL direction must be 'x' or 'y'")
    primitive_left = conservative_to_primitive(left, eos)
    primitive_right = conservative_to_primitive(right, eos)
    if direction == "x":
        velocity_left = primitive_left.velocity_x_m_per_s
        velocity_right = primitive_right.velocity_x_m_per_s
        physical_flux_left = flux_x(left, eos)
        physical_flux_right = flux_x(right, eos)
    else:
        velocity_left = primitive_left.velocity_y_m_per_s
        velocity_right = primitive_right.velocity_y_m_per_s
        physical_flux_left = flux_y(left, eos)
        physical_flux_right = flux_y(right, eos)
    sound_left = np.asarray(
        eos.sound_speed(primitive_left.density_kg_per_m3, primitive_left.pressure_Pa),
        dtype=np.float64,
    )
    sound_right = np.asarray(
        eos.sound_speed(primitive_right.density_kg_per_m3, primitive_right.pressure_Pa),
        dtype=np.float64,
    )
    speed_left = np.minimum(velocity_left - sound_left, velocity_right - sound_right)
    speed_right = np.maximum(velocity_left + sound_left, velocity_right + sound_right)
    denominator = speed_right - speed_left
    middle = (speed_left < 0.0) & (speed_right > 0.0)
    if np.any(middle & (denominator <= 0.0)):
        raise ReactiveNumericalError(
            "invalid_hll_waves", "HLL wave-speed interval is not positive"
        )
    denominator_safe = np.where(middle, denominator, 1.0)
    hll_middle = (
        speed_right[..., None] * physical_flux_left
        - speed_left[..., None] * physical_flux_right
        + (speed_left * speed_right)[..., None] * (right - left)
    ) / denominator_safe[..., None]
    result = np.where(
        (speed_left >= 0.0)[..., None],
        physical_flux_left,
        np.where((speed_right <= 0.0)[..., None], physical_flux_right, hll_middle),
    )
    if not np.isfinite(result).all():
        raise ReactiveNumericalError("nonfinite_hll_flux", "HLL flux became non-finite")
    return result


def interface_fluxes(
    conservative_state: Any,
    eos: EquationOfState,
    *,
    spatial_axis: int = 0,
    direction: str = "x",
    boundary: BoundaryType | str = BoundaryType.TRANSMISSIVE,
    weno_epsilon: float = 1.0e-6,
) -> tuple[np.ndarray, FluxDiagnostics]:
    """Reconstruct face states, applying a measured local positivity fallback."""
    state = np.asarray(conservative_state, dtype=np.float64)
    axis = _normalise_spatial_axis(state, spatial_axis)
    if state.shape[-1] != NCONS:
        raise ReactiveConfigurationError(
            f"reactive Euler state last dimension must be {NCONS}"
        )
    if not np.all(physical_state_mask(state, eos)):
        raise ReactiveNumericalError(
            "nonphysical_cell_average",
            "face reconstruction received nonphysical cell averages",
        )
    # Reconstruct primitive variables and then rebuild conservative face
    # states.  Component-wise reconstruction of conservative rho*E with
    # independent nonlinear weights is not covariant to the arbitrary
    # reference-energy gauge E -> E+C, whereas this primitive representation
    # is.  Characteristic reconstruction remains an unimplemented paper detail.
    primitive_cells = _primitive_reconstruction_fields(state, eos)
    primitive_left, primitive_right = weno5_js_reconstruct(
        primitive_cells,
        spatial_axis=axis,
        boundary=boundary,
        epsilon=weno_epsilon,
    )
    reconstructed_left = _primitive_faces_to_conservative(primitive_left, eos)
    reconstructed_right = _primitive_faces_to_conservative(primitive_right, eos)
    valid_left = physical_state_mask(reconstructed_left, eos)
    valid_right = physical_state_mask(reconstructed_right, eos)
    fallback = ~(valid_left & valid_right)
    first_left, first_right = _first_order_face_states(state, axis, boundary)
    maximum_correction = 0.0
    if np.any(fallback):
        first_valid = physical_state_mask(first_left, eos) & physical_state_mask(
            first_right, eos
        )
        if np.any(fallback & ~first_valid):
            raise ReactiveNumericalError(
                "nonphysical_first_order_face",
                "positivity fallback face states are nonphysical; refusing to continue",
            )
        maximum_correction = float(
            max(
                np.max(np.abs(first_left[fallback] - reconstructed_left[fallback])),
                np.max(np.abs(first_right[fallback] - reconstructed_right[fallback])),
            )
        )
        reconstructed_left = np.where(
            fallback[..., None], first_left, reconstructed_left
        )
        reconstructed_right = np.where(
            fallback[..., None], first_right, reconstructed_right
        )
    flux = hll_flux(reconstructed_left, reconstructed_right, eos, direction=direction)
    diagnostics = FluxDiagnostics(
        interface_count=int(fallback.size),
        positivity_face_fallback_count=int(np.count_nonzero(fallback)),
        positivity_face_fallback_max_abs_state_correction=maximum_correction,
        boundary_type=_boundary_value(boundary).value,
    )
    return flux, diagnostics


def finite_volume_rhs(
    conservative_state: Any,
    cell_width_m: float,
    eos: EquationOfState,
    *,
    spatial_axis: int = 0,
    direction: str = "x",
    boundary: BoundaryType | str = BoundaryType.TRANSMISSIVE,
    weno_epsilon: float = 1.0e-6,
) -> tuple[np.ndarray, FluxDiagnostics]:
    state = np.asarray(conservative_state, dtype=np.float64)
    axis = _normalise_spatial_axis(state, spatial_axis)
    if not np.isfinite(cell_width_m) or cell_width_m <= 0.0:
        raise ReactiveConfigurationError("cell_width_m must be finite and positive")
    flux, diagnostics = interface_fluxes(
        state,
        eos,
        spatial_axis=axis,
        direction=direction,
        boundary=boundary,
        weno_epsilon=weno_epsilon,
    )
    result = -np.diff(flux, axis=axis) / cell_width_m
    if result.shape != state.shape or not np.isfinite(result).all():
        raise ReactiveNumericalError(
            "invalid_flux_divergence", "finite-volume flux divergence is invalid"
        )
    return result, diagnostics


def max_signal_speed(
    conservative_state: Any, eos: EquationOfState, *, direction: str = "x"
) -> float:
    if direction not in {"x", "y"}:
        raise ReactiveConfigurationError("signal-speed direction must be 'x' or 'y'")
    primitive = conservative_to_primitive(conservative_state, eos)
    velocity = (
        primitive.velocity_x_m_per_s
        if direction == "x"
        else primitive.velocity_y_m_per_s
    )
    sound = np.asarray(
        eos.sound_speed(primitive.density_kg_per_m3, primitive.pressure_Pa),
        dtype=np.float64,
    )
    result = float(np.max(np.abs(velocity) + sound))
    if not np.isfinite(result) or result <= 0.0:
        raise ReactiveNumericalError(
            "invalid_signal_speed", "maximum Euler signal speed is invalid"
        )
    return result


def cfl_time_step(
    conservative_state: Any,
    eos: EquationOfState,
    dx_m: float,
    *,
    cfl: float,
    dy_m: float | None = None,
) -> float:
    if not np.isfinite(dx_m) or dx_m <= 0.0:
        raise ReactiveConfigurationError("dx_m must be finite and positive")
    if not np.isfinite(cfl) or cfl <= 0.0 or cfl > 1.0:
        raise ReactiveConfigurationError("CFL number must be in (0, 1]")
    inverse_dt = max_signal_speed(conservative_state, eos, direction="x") / dx_m
    if dy_m is not None:
        if not np.isfinite(dy_m) or dy_m <= 0.0:
            raise ReactiveConfigurationError("dy_m must be finite and positive")
        inverse_dt += max_signal_speed(conservative_state, eos, direction="y") / dy_m
    dt = cfl / inverse_dt
    if not np.isfinite(dt) or dt <= 0.0:
        raise ReactiveNumericalError("invalid_cfl_timestep", "CFL time step is invalid")
    return float(dt)


RhsFunction = Callable[[float, np.ndarray], np.ndarray]
StateValidator = Callable[[np.ndarray], None]


def ssprk3_step(
    state: Any,
    time_s: float,
    dt_s: float,
    rhs: RhsFunction,
    *,
    validator: StateValidator | None = None,
) -> np.ndarray:
    """Advance one non-autonomous Shu--Osher SSP-RK3 step."""
    initial = np.asarray(state, dtype=np.float64)
    if not np.isfinite(initial).all():
        raise ReactiveNumericalError(
            "nonfinite_rk_state", "RK3 initial state is non-finite"
        )
    if not np.isfinite(time_s) or not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ReactiveConfigurationError(
            "RK3 time must be finite and dt_s must be positive"
        )
    if validator is not None:
        validator(initial)

    def evaluate(
        stage_time: float, stage_state: np.ndarray, stage_name: str
    ) -> np.ndarray:
        derivative = np.asarray(rhs(stage_time, stage_state), dtype=np.float64)
        if derivative.shape != initial.shape:
            raise ReactiveConfigurationError(
                f"RK3 RHS shape changed at {stage_name}: {derivative.shape} != {initial.shape}"
            )
        if not np.isfinite(derivative).all():
            raise ReactiveNumericalError(
                "nonfinite_rk_rhs", f"RK3 RHS is non-finite at {stage_name}"
            )
        return derivative

    stage_1 = initial + dt_s * evaluate(time_s, initial, "stage_1")
    if validator is not None:
        validator(stage_1)
    stage_2 = 0.75 * initial + 0.25 * (
        stage_1 + dt_s * evaluate(time_s + dt_s, stage_1, "stage_2")
    )
    if validator is not None:
        validator(stage_2)
    result = (1.0 / 3.0) * initial + (2.0 / 3.0) * (
        stage_2 + dt_s * evaluate(time_s + 0.5 * dt_s, stage_2, "stage_3")
    )
    if not np.isfinite(result).all():
        raise ReactiveNumericalError(
            "nonfinite_rk_state", "RK3 final state is non-finite"
        )
    if validator is not None:
        validator(result)
    return result
