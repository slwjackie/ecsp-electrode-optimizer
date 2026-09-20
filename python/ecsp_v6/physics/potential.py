from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F

from .geometry import GeometryBatch
from .errors import BCCandidateBatchError
from .numerics import SolverDiagnostics, apply_neumann_boundary, harmonic_mean, matlab_gradient


def _batch_voltage_field(voltage: object, reference: torch.Tensor) -> torch.Tensor:
    """Broadcast scalar/per-lane voltage over a batched potential field."""
    value = torch.as_tensor(
        voltage, device=reference.device, dtype=reference.dtype
    )
    if value.ndim == 0:
        return value.expand_as(reference)
    if value.ndim == 1 and value.numel() == reference.shape[0]:
        return value[:, None, None].expand_as(reference)
    if value.shape == (reference.shape[0], 1, 1):
        return value.expand_as(reference)
    raise ValueError("Voltage must be scalar or contain one value per batch lane")


def diffusion_current_divergence(
    cation: torch.Tensor,
    anion: torch.Tensor,
    dcation: torch.Tensor,
    danion: torch.Tensor,
    spacing: float,
    config: dict,
    geometry: GeometryBatch | None = None,
) -> torch.Tensor:
    """Return div(J_diff) for the current-conservation potential equation.

    v7.3.2 uses a finite-volume, propellant-only face flux when the nonlinear
    electrochemical Robin boundary is enabled.  This avoids creating an
    artificial concentration gradient from the electrolyte into zero-valued
    metal cells.  The legacy MATLAB-parity gradient remains available through
    ``interface.boundaryCouplingModel: legacy_posthoc``.
    """
    if not bool(config["electrical"].get("diffusionPotentialEnabled", True)):
        return torch.zeros_like(cation)
    faraday = float(config["transport"]["faradayConstant_C_per_mol"])
    z_plus = float(config["transport"]["chargeNumberCation"])
    z_minus = float(config["transport"]["chargeNumberAnion"])
    boundary_model = str(
        config.get("interface", {}).get("boundaryCouplingModel", "legacy_posthoc")
    ).lower()
    if geometry is None or boundary_model in {"legacy_posthoc", "legacy", "posthoc"}:
        dcplus_dx, dcplus_dy = matlab_gradient(cation, spacing)
        dcminus_dx, dcminus_dy = matlab_gradient(anion, spacing)
        jdiff_x = -faraday * (
            z_plus * dcation * dcplus_dx + z_minus * danion * dcminus_dx
        )
        jdiff_y = -faraday * (
            z_plus * dcation * dcplus_dy + z_minus * danion * dcminus_dy
        )
        djx_dx, _ = matlab_gradient(jdiff_x, spacing)
        _, djy_dy = matlab_gradient(jdiff_y, spacing)
        return djx_dx + djy_dy

    propellant = geometry.propellant
    batch, rows, columns = cation.shape
    flux_x = torch.zeros(
        (batch, rows, columns + 1), device=cation.device, dtype=cation.dtype
    )
    flux_y = torch.zeros(
        (batch, rows + 1, columns), device=cation.device, dtype=cation.dtype
    )
    valid_x = propellant[..., :, :-1] & propellant[..., :, 1:]
    dp = harmonic_mean(dcation[..., :, :-1], dcation[..., :, 1:])
    dm = harmonic_mean(danion[..., :, :-1], danion[..., :, 1:])
    jx = -faraday * (
        z_plus * dp * (cation[..., :, 1:] - cation[..., :, :-1]) / spacing
        + z_minus * dm * (anion[..., :, 1:] - anion[..., :, :-1]) / spacing
    )
    flux_x[..., :, 1:columns] = torch.where(valid_x, jx, torch.zeros_like(jx))

    valid_y = propellant[..., :-1, :] & propellant[..., 1:, :]
    dp = harmonic_mean(dcation[..., :-1, :], dcation[..., 1:, :])
    dm = harmonic_mean(danion[..., :-1, :], danion[..., 1:, :])
    jy = -faraday * (
        z_plus * dp * (cation[..., 1:, :] - cation[..., :-1, :]) / spacing
        + z_minus * dm * (anion[..., 1:, :] - anion[..., :-1, :]) / spacing
    )
    flux_y[..., 1:rows, :] = torch.where(valid_y, jy, torch.zeros_like(jy))

    divergence = (
        (flux_x[..., :, 1:] - flux_x[..., :, :-1]) / spacing
        + (flux_y[..., 1:, :] - flux_y[..., :-1, :]) / spacing
    )
    return torch.where(propellant, divergence, torch.zeros_like(divergence))


def _shift_east(field: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    result = torch.full_like(field, fill)
    result[..., :, :-1] = field[..., :, 1:]
    return result


def _shift_west(field: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    result = torch.full_like(field, fill)
    result[..., :, 1:] = field[..., :, :-1]
    return result


def _shift_north(field: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    result = torch.full_like(field, fill)
    result[..., :-1, :] = field[..., 1:, :]
    return result


def _shift_south(field: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    result = torch.full_like(field, fill)
    result[..., 1:, :] = field[..., :-1, :]
    return result


def _neighbor_coefficients(sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    east = torch.zeros_like(sigma)
    west = torch.zeros_like(sigma)
    north = torch.zeros_like(sigma)
    south = torch.zeros_like(sigma)
    east[..., :, :-1] = harmonic_mean(sigma[..., :, :-1], sigma[..., :, 1:])
    west[..., :, 1:] = harmonic_mean(sigma[..., :, 1:], sigma[..., :, :-1])
    north[..., :-1, :] = harmonic_mean(sigma[..., :-1, :], sigma[..., 1:, :])
    south[..., 1:, :] = harmonic_mean(sigma[..., 1:, :], sigma[..., :-1, :])
    return east, west, north, south


def _active_mask(geometry: GeometryBatch) -> torch.Tensor:
    active = geometry.propellant.clone()
    active[..., 0, :] = False
    active[..., -1, :] = False
    active[..., :, 0] = False
    active[..., :, -1] = False
    return active


def _fixed_potential(geometry: GeometryBatch, voltage: float, cathode_voltage: float, dtype: torch.dtype) -> torch.Tensor:
    result = torch.zeros(geometry.fixed.shape, device=geometry.fixed.device, dtype=dtype)
    result = torch.where(geometry.anode, _batch_voltage_field(voltage, result), result)
    result = torch.where(geometry.cathode, torch.as_tensor(cathode_voltage, device=result.device, dtype=dtype), result)
    return result


def _build_linear_system(
    sigma: torch.Tensor,
    rhs: torch.Tensor,
    geometry: GeometryBatch,
    voltage: float,
    cathode_voltage: float,
    spacing: float,
    face_fixed_values: dict[str, torch.Tensor] | None = None,
    robin_linearization: dict[str, torch.Tensor] | None = None,
    neumann_boundary: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    surface_overlay = bool(
        robin_linearization is not None
        and str(robin_linearization.get("model", "")).lower()
        == "surface_overlay_bv"
    )
    # Surface-contact masks live on the physical top face of every grid cell,
    # including cells on the lateral edge of the sheet.  They are therefore
    # unknown control volumes, not electrode/ghost cells.  A missing neighbour
    # at the outside of the sheet is the natural zero-normal-current boundary.
    base_active = (
        geometry.propellant.clone() if surface_overlay else _active_mask(geometry)
    )
    fixed = geometry.fixed
    fixed_value = _fixed_potential(geometry, voltage, cathode_voltage, sigma.dtype)
    gauge_mask = torch.zeros_like(base_active)
    gauge_value = torch.zeros_like(sigma)
    if neumann_boundary is not None:
        gauge_mask = neumann_boundary["gaugeMask"].to(device=sigma.device, dtype=torch.bool) & base_active
        value = neumann_boundary.get("gaugeValue_V", torch.zeros(sigma.shape[0], device=sigma.device, dtype=sigma.dtype))
        if value.ndim == 1:
            gauge_value = value[:, None, None].expand_as(sigma)
        else:
            gauge_value = value.to(device=sigma.device, dtype=sigma.dtype)
    active = base_active & ~gauge_mask
    s_e, s_w, s_n, s_s = _neighbor_coefficients(sigma)

    active_e, active_w = _shift_east(active), _shift_west(active)
    active_n, active_s = _shift_north(active), _shift_south(active)
    gauge_e, gauge_w = _shift_east(gauge_mask), _shift_west(gauge_mask)
    gauge_n, gauge_s = _shift_north(gauge_mask), _shift_south(gauge_mask)
    gauge_value_e, gauge_value_w = _shift_east(gauge_value), _shift_west(gauge_value)
    gauge_value_n, gauge_value_s = _shift_north(gauge_value), _shift_south(gauge_value)
    fixed_e, fixed_w = _shift_east(fixed), _shift_west(fixed)
    fixed_n, fixed_s = _shift_north(fixed), _shift_south(fixed)
    if face_fixed_values is None:
        value_e, value_w = _shift_east(fixed_value), _shift_west(fixed_value)
        value_n, value_s = _shift_north(fixed_value), _shift_south(fixed_value)
    else:
        required = {"east", "west", "north", "south"}
        missing = required.difference(face_fixed_values)
        if missing:
            raise ValueError(f"face_fixed_values missing directions: {sorted(missing)}")
        value_e = face_fixed_values["east"].to(device=sigma.device, dtype=sigma.dtype)
        value_w = face_fixed_values["west"].to(device=sigma.device, dtype=sigma.dtype)
        value_n = face_fixed_values["north"].to(device=sigma.device, dtype=sigma.dtype)
        value_s = face_fixed_values["south"].to(device=sigma.device, dtype=sigma.dtype)

    if neumann_boundary is not None:
        # Electrode faces carry prescribed Faradaic current.  A single
        # propellant cell per batch is fixed only as a numerical gauge; its
        # value is eliminated symmetrically from neighbouring equations.
        connection_e = active & (active_e | gauge_e)
        connection_w = active & (active_w | gauge_w)
        connection_n = active & (active_n | gauge_n)
        connection_s = active & (active_s | gauge_s)
    elif robin_linearization is None:
        connection_e = active & (active_e | fixed_e)
        connection_w = active & (active_w | fixed_w)
        connection_n = active & (active_n | fixed_n)
        connection_s = active & (active_s | fixed_s)
    else:
        connection_e = active & active_e
        connection_w = active & active_w
        connection_n = active & active_n
        connection_s = active & active_s

    ce = s_e * connection_e.to(sigma.dtype)
    cw = s_w * connection_w.to(sigma.dtype)
    cn = s_n * connection_n.to(sigma.dtype)
    cs = s_s * connection_s.to(sigma.dtype)
    diagonal = ce + cw + cn + cs
    b = -rhs * (spacing * spacing)
    if neumann_boundary is not None:
        b = b + (
            ce * gauge_e.to(sigma.dtype) * gauge_value_e
            + cw * gauge_w.to(sigma.dtype) * gauge_value_w
            + cn * gauge_n.to(sigma.dtype) * gauge_value_n
            + cs * gauge_s.to(sigma.dtype) * gauge_value_s
        )
        outward_sum = neumann_boundary["outwardCurrentFaceSum_A_per_m2"].to(
            device=sigma.device, dtype=sigma.dtype
        )
        b = b - spacing * outward_sum
    elif robin_linearization is None:
        b = b + (
            ce * fixed_e.to(sigma.dtype) * value_e
            + cw * fixed_w.to(sigma.dtype) * value_w
            + cn * fixed_n.to(sigma.dtype) * value_n
            + cs * fixed_s.to(sigma.dtype) * value_s
        )
    elif surface_overlay:
        anode_current = robin_linearization["anodeCurrentDensity_A_per_m2"].to(
            device=sigma.device, dtype=sigma.dtype
        )
        cathode_current = robin_linearization["cathodeCurrentDensity_A_per_m2"].to(
            device=sigma.device, dtype=sigma.dtype
        )
        anode_slope = torch.clamp(
            robin_linearization["anodeSlope_S_per_m2"].to(
                device=sigma.device, dtype=sigma.dtype
            ), min=0.0
        )
        cathode_slope = torch.clamp(
            robin_linearization["cathodeSlope_S_per_m2"].to(
                device=sigma.device, dtype=sigma.dtype
            ), min=0.0
        )
        linearization_potential = robin_linearization["potential_V"].to(
            device=sigma.device, dtype=sigma.dtype
        )
        source_layer = float(robin_linearization["surfaceLayerThickness_m"])
        if not math.isfinite(source_layer) or source_layer <= 0.0:
            raise ValueError("surfaceLayerThickness_m must be finite and positive")
        source_scale = spacing * spacing / source_layer
        slope = anode_slope + cathode_slope
        diagonal = diagonal + source_scale * slope
        b = b + source_scale * (
            anode_current - cathode_current + slope * linearization_potential
        )
    else:
        conductance_sum = robin_linearization[
            "dOutwardCurrentFaceSum_dPotential_S_per_m2"
        ].to(device=sigma.device, dtype=sigma.dtype)
        current_sum = robin_linearization["outwardCurrentFaceSum_A_per_m2"].to(
            device=sigma.device, dtype=sigma.dtype
        )
        linearization_potential = robin_linearization["potential_V"].to(
            device=sigma.device, dtype=sigma.dtype
        )
        diagonal = diagonal + spacing * conductance_sum
        b = b - spacing * (current_sum - conductance_sum * linearization_potential)
    b = torch.where(active, b, torch.zeros_like(b))
    invalid_diagonal = active & ((diagonal <= 0.0) | ~torch.isfinite(diagonal))
    if bool(torch.any(invalid_diagonal)):
        bad = (
            torch.nonzero(invalid_diagonal.flatten(1).any(dim=1))
            .flatten()
            .detach()
            .cpu()
            .tolist()
        )
        examples = (
            torch.nonzero(invalid_diagonal, as_tuple=False)[:8]
            .detach()
            .cpu()
            .tolist()
        )
        raise BCCandidateBatchError(
            "Potential system is singular or non-finite on active cells; "
            f"examples={examples}",
            bad,
            "potential_singular_diagonal",
        )
    diagonal = torch.where(active, diagonal, torch.ones_like(diagonal))
    return {
        "active": active,
        "fixed_value": fixed_value,
        "ce": ce,
        "cw": cw,
        "cn": cn,
        "cs": cs,
        "diagonal": diagonal,
        "b": b,
        "active_e": active_e,
        "active_w": active_w,
        "active_n": active_n,
        "active_s": active_s,
        "gauge_mask": gauge_mask,
        "gauge_value": gauge_value,
        "surface_overlay": surface_overlay,
    }


def _apply_operator(x: torch.Tensor, system: dict[str, torch.Tensor]) -> torch.Tensor:
    active = system["active"]
    value = system["diagonal"] * x
    value = value - system["ce"] * system["active_e"].to(x.dtype) * _shift_east(x)
    value = value - system["cw"] * system["active_w"].to(x.dtype) * _shift_west(x)
    value = value - system["cn"] * system["active_n"].to(x.dtype) * _shift_north(x)
    value = value - system["cs"] * system["active_s"].to(x.dtype) * _shift_south(x)
    return torch.where(active, value, torch.zeros_like(value))


def _dot_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Two-stage batch dot product with a shallower FP32 reduction tree."""
    return (a * b).sum(dim=-1).sum(dim=-1)


def _norm_batch(value: torch.Tensor) -> torch.Tensor:
    """Overflow-safe Euclidean norm for each batch row.

    Squaring a finite FP64 value larger than roughly ``1e154`` overflows even
    when the Euclidean norm itself is representable.  Besides losing a useful
    diagnostic, the former ``sqrt(sum(x*x))`` implementation could make both
    the residual norm and its relative tolerance infinite; IEEE ``inf <= inf``
    is true and could therefore report a completely unsolved row as converged.
    Scale by the largest component before squaring (the LAPACK ``xLASSQ``
    principle).  A genuinely non-finite input, or a mathematically
    unrepresentable final norm, remains non-finite and is rejected by the
    convergence predicate below.
    """
    flattened = value.reshape(value.shape[0], -1)
    scale = torch.amax(torch.abs(flattened), dim=1)
    finite_positive = torch.isfinite(scale) & (scale > 0.0)
    safe_scale = torch.where(finite_positive, scale, torch.ones_like(scale))
    scaled = flattened / safe_scale[:, None]
    scaled_norm = safe_scale * torch.sqrt(
        torch.clamp(torch.sum(scaled * scaled, dim=1), min=0.0)
    )
    zero = torch.zeros_like(scale)
    return torch.where(
        scale == 0.0,
        zero,
        torch.where(torch.isfinite(scale), scaled_norm, scale),
    )


def _finite_norm_converged(
    residual_norm: torch.Tensor, threshold: torch.Tensor
) -> torch.Tensor:
    """A non-finite norm or tolerance can never prove convergence."""
    return (
        torch.isfinite(residual_norm)
        & torch.isfinite(threshold)
        & (residual_norm <= threshold)
    )


def _solver_threshold(
    reference_norm: torch.Tensor, relative_tolerance: float, absolute_tolerance: float
) -> torch.Tensor:
    return torch.maximum(
        float(relative_tolerance) * torch.clamp(reference_norm, min=1.0),
        torch.full_like(reference_norm, float(absolute_tolerance)),
    )


def _equilibrate_linear_system(
    system: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Return the symmetric Jacobi-equilibrated system ``S A S``.

    With ``S=D^{-1/2}`` and ``x=S y``, the transformed matrix has unit
    diagonal while retaining exact symmetry and the same physical solution.
    This removes the conductivity/Robin diagonal scale from all Krylov scalar
    products and is particularly important for MPS FP32.
    """
    active = system["active"]
    diagonal = system["diagonal"]
    invalid = active & ((diagonal <= 0.0) | ~torch.isfinite(diagonal))
    if bool(torch.any(invalid)):
        bad = (
            torch.nonzero(invalid.flatten(1).any(dim=1))
            .flatten()
            .detach()
            .cpu()
            .tolist()
        )
        indices = torch.nonzero(invalid, as_tuple=False)[:8].detach().cpu().tolist()
        raise BCCandidateBatchError(
            "Potential system has non-positive or non-finite active diagonal entries; "
            f"examples={indices}",
            bad,
            "potential_invalid_equilibration_diagonal",
        )
    scale = torch.where(active, torch.rsqrt(diagonal), torch.ones_like(diagonal))
    equilibrated = dict(system)
    equilibrated["ce"] = system["ce"] * scale * _shift_east(scale)
    equilibrated["cw"] = system["cw"] * scale * _shift_west(scale)
    equilibrated["cn"] = system["cn"] * scale * _shift_north(scale)
    equilibrated["cs"] = system["cs"] * scale * _shift_south(scale)
    equilibrated["diagonal"] = torch.ones_like(diagonal)
    equilibrated["b"] = system["b"] * scale
    equilibrated["_equilibration_scale"] = scale
    return equilibrated, scale



def _mg_restrict(field: torch.Tensor, target: int) -> torch.Tensor:
    """Restrict a fine-grid field to the next multigrid level.

    PyTorch implements ``interpolate(..., mode="area")`` through adaptive
    average pooling.  Apple MPS currently rejects non-divisible input/output
    sizes (for example 193 -> 97), which are exactly the odd grid sizes used
    by this solver.  On MPS use the standard 2-D full-weighting restriction
    stencil instead.  It is local, GPU-resident, and produces ceil(n/2) for
    odd n without a CPU synchronization.  CPU/CUDA retain the historical
    area-restriction path so their FP64 reference results are unchanged.
    """
    if field.device.type == "mps":
        expected = (int(field.shape[-2]) + 1) // 2
        if int(target) != expected or int(field.shape[-2]) != int(field.shape[-1]):
            raise RuntimeError(
                "The MPS multigrid hierarchy requires square odd grids with "
                f"target=ceil(n/2); got n={tuple(field.shape[-2:])}, target={target}. "
                "CPU transfer fallback is intentionally disabled."
            )

        kernel = field.new_tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]
        ).div_(16.0)
        restricted = F.conv2d(
            field[:, None, ...],
            kernel[None, None, ...],
            stride=2,
            padding=1,
        )[:, 0]
        if restricted.shape[-2:] != (target, target):
            raise RuntimeError(
                "MPS full-weighting restriction produced unexpected shape "
                f"{tuple(restricted.shape[-2:])}; expected {(target, target)}"
            )
        return restricted

    return F.interpolate(field[:, None, ...], size=(target, target), mode="area")[:, 0]


def _mg_prolong(field: torch.Tensor, target: int) -> torch.Tensor:
    return F.interpolate(
        field[:, None, ...], size=(target, target), mode="bilinear", align_corners=True
    )[:, 0]


def _coarsen_linear_system(system: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    n = int(system["active"].shape[-1])
    target = (n + 1) // 2
    active = F.max_pool2d(
        system["active"].to(torch.float32)[:, None, ...], kernel_size=3, stride=2, padding=1
    )[:, 0] > 0.5
    # Never treat the outer numerical boundary as an unknown.
    active[..., 0, :] = False
    active[..., -1, :] = False
    active[..., :, 0] = False
    active[..., :, -1] = False
    result: dict[str, torch.Tensor] = {"active": active}
    for name in ("ce", "cw", "cn", "cs", "diagonal"):
        result[name] = _mg_restrict(system[name], target)
    eps = torch.finfo(system["diagonal"].dtype).eps
    result["diagonal"] = torch.where(
        active, torch.clamp(result["diagonal"], min=eps), torch.ones_like(result["diagonal"])
    )
    result["active_e"] = _shift_east(active)
    result["active_w"] = _shift_west(active)
    result["active_n"] = _shift_north(active)
    result["active_s"] = _shift_south(active)
    # These fields are not needed by the error equation but keep the structure
    # compatible with _apply_operator.
    result["fixed_value"] = torch.zeros_like(result["diagonal"])
    result["b"] = torch.zeros_like(result["diagonal"])
    return result


def _mg_jacobi(
    x: torch.Tensor,
    rhs: torch.Tensor,
    system: dict[str, torch.Tensor],
    iterations: int,
    omega: float,
) -> torch.Tensor:
    active = system["active"]
    current = torch.where(active, x, torch.zeros_like(x))
    for _ in range(max(iterations, 0)):
        residual = rhs - _apply_operator(current, system)
        current = current + omega * residual / system["diagonal"]
        current = torch.where(active, current, torch.zeros_like(current))
    return current


def _mg_v_cycle(
    x: torch.Tensor,
    rhs: torch.Tensor,
    system: dict[str, torch.Tensor],
    cfg: dict,
    level: int = 0,
) -> torch.Tensor:
    mg = cfg.get("multigrid", {})
    maximum_levels = int(mg.get("maximumLevels", 6))
    minimum_grid = int(mg.get("minimumGridSize", 7))
    pre = int(mg.get("preSmooth", 2))
    post = int(mg.get("postSmooth", 2))
    coarse = int(mg.get("coarseSmooth", 24))
    omega = float(mg.get("jacobiOmega", 0.7))
    n = int(x.shape[-1])
    if level >= maximum_levels - 1 or n <= minimum_grid:
        return _mg_jacobi(x, rhs, system, coarse, omega)
    current = _mg_jacobi(x, rhs, system, pre, omega)
    residual = rhs - _apply_operator(current, system)
    coarse_system = _coarsen_linear_system(system)
    target = int(coarse_system["active"].shape[-1])
    residual_c = _mg_restrict(residual, target)
    if bool(mg.get("scaleRestrictedResidualByGridRatioSquared", True)):
        # The matrix-free stencil is stored after multiplying the physical
        # equation by h^2.  On the 2h grid, the matching rediscretized error
        # equation therefore needs a (H/h)^2 RHS scale.  Omitting this factor
        # makes each coarse correction roughly four times too weak.
        grid_ratio = float(max(n - 1, 1)) / float(max(target - 1, 1))
        residual_c = residual_c * (grid_ratio * grid_ratio)
    residual_c = torch.where(coarse_system["active"], residual_c, torch.zeros_like(residual_c))
    error_c = _mg_v_cycle(
        torch.zeros_like(residual_c), residual_c, coarse_system, cfg, level + 1
    )
    correction = _mg_prolong(error_c, n)
    current = torch.where(system["active"], current + correction, torch.zeros_like(current))
    return _mg_jacobi(current, rhs, system, post, omega)


def _apply_pcg_preconditioner(
    residual: torch.Tensor,
    system: dict[str, torch.Tensor],
    solver_cfg: dict,
) -> torch.Tensor:
    method = str(solver_cfg.get("preconditioner", "multigrid")).lower()
    if method in {"jacobi", "diagonal"}:
        return residual / system["diagonal"]
    if method in {"multigrid", "mg", "geometric_multigrid"}:
        return _mg_v_cycle(torch.zeros_like(residual), residual, system, solver_cfg)
    raise ValueError(f"Unknown potential PCG preconditioner: {method}")

def _pcg_correction_core(
    system: dict[str, torch.Tensor],
    rhs: torch.Tensor,
    solver_cfg: dict,
    *,
    maximum_iterations: int,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Solve an equilibrated SPD correction equation from a zero initial guess.

    The routine uses the true residual as convergence authority, stores the
    best iterate per batch row, and restarts from that iterate if finite-
    precision drift, residual growth, or loss of positive curvature is
    detected.  Consequently a single bad candidate cannot launch the search
    direction to the 1e11--1e12 residuals observed in the previous MPS path.
    """
    active = system["active"]
    rhs = torch.where(active, rhs, torch.zeros_like(rhs))
    batch = rhs.shape[0]
    dtype = rhs.dtype
    eps = torch.finfo(dtype).eps
    tiny = torch.finfo(dtype).tiny
    guard_factor = float(solver_cfg.get("breakdownGuardFactor", 16.0))
    growth_factor = float(solver_cfg.get("residualGrowthRestartFactor", 4.0))
    drift_factor = float(solver_cfg.get("residualReplacementRelativeDrift", 0.25))
    maximum_restarts = max(0, int(solver_cfg.get("maximumKrylovRestarts", 8)))
    check_interval = max(1, int(solver_cfg.get("convergenceCheckInterval", 5)))

    correction = torch.zeros_like(rhs)
    residual = rhs.clone()
    z = residual.clone()  # unit diagonal after symmetric equilibration
    direction = z.clone()
    rz = _dot_batch(residual, z)
    rhs_norm = _norm_batch(rhs)
    threshold = _solver_threshold(rhs_norm, relative_tolerance, absolute_tolerance)
    current_norm = _norm_batch(residual)
    converged = _finite_norm_converged(current_norm, threshold)
    failed = torch.zeros(batch, device=rhs.device, dtype=torch.bool)
    iterations = torch.zeros(batch, device=rhs.device, dtype=torch.int64)
    restarts = torch.zeros_like(iterations)
    best_correction = correction.clone()
    best_norm = current_norm.clone()

    def restart_rows(mask: torch.Tensor) -> None:
        nonlocal correction, residual, z, direction, rz, failed, restarts
        if not bool(torch.any(mask)):
            return
        restarts = torch.where(mask, restarts + 1, restarts)
        exhausted = mask & (restarts > maximum_restarts)
        failed = failed | exhausted
        retry = mask & ~exhausted
        if bool(torch.any(retry)):
            mask3 = retry[:, None, None]
            correction = torch.where(mask3, best_correction, correction)
            true_residual = torch.where(
                active, rhs - _apply_operator(correction, system), torch.zeros_like(rhs)
            )
            residual = torch.where(mask3, true_residual, residual)
            z = torch.where(mask3, true_residual, z)
            direction = torch.where(mask3, true_residual, direction)
            replacement_rz = _dot_batch(true_residual, true_residual)
            rz = torch.where(retry, replacement_rz, rz)

    for iteration in range(1, max(0, int(maximum_iterations)) + 1):
        unresolved = ~(converged | failed)
        if not bool(torch.any(unresolved)):
            break

        ap = _apply_operator(direction, system)
        p_ap = _dot_batch(direction, ap)
        p_norm = _norm_batch(direction)
        ap_norm = _norm_batch(ap)
        curvature_floor = guard_factor * eps * p_norm * ap_norm
        bad_curvature = unresolved & (
            ~torch.isfinite(p_ap)
            | ~torch.isfinite(p_norm)
            | ~torch.isfinite(ap_norm)
            | (p_ap <= torch.maximum(curvature_floor, torch.full_like(p_ap, tiny)))
        )
        restart_rows(bad_curvature)
        step = ~(converged | failed | bad_curvature)
        if not bool(torch.any(step)):
            continue

        safe_p_ap = torch.where(step, p_ap, torch.ones_like(p_ap))
        alpha = torch.where(step, rz / safe_p_ap, torch.zeros_like(rz))
        nonfinite_alpha = step & ~torch.isfinite(alpha)
        restart_rows(nonfinite_alpha)
        step = step & ~nonfinite_alpha & ~failed
        if not bool(torch.any(step)):
            continue

        mask3 = step[:, None, None] & active
        correction = torch.where(
            mask3, correction + alpha[:, None, None] * direction, correction
        )
        residual = torch.where(
            mask3, residual - alpha[:, None, None] * ap, residual
        )
        z_new = residual.clone()
        rz_new = _dot_batch(residual, z_new)
        bad_rz = step & (
            ~torch.isfinite(rz_new)
            | (rz_new <= 0.0)
        )

        perform_check = iteration % check_interval == 0 or iteration == maximum_iterations
        restart = bad_rz.clone()
        if perform_check:
            recurrence_norm = _norm_batch(residual)
            true_residual = torch.where(
                active, rhs - _apply_operator(correction, system), torch.zeros_like(rhs)
            )
            true_norm = _norm_batch(true_residual)
            previous_best = best_norm.clone()
            finite_true = torch.isfinite(true_norm)
            improved = step & finite_true & (true_norm < best_norm)
            best_correction = torch.where(
                improved[:, None, None], correction, best_correction
            )
            best_norm = torch.where(improved, true_norm, best_norm)
            newly = step & _finite_norm_converged(true_norm, threshold)
            iterations = torch.where(
                newly, torch.full_like(iterations, iteration), iterations
            )
            converged = converged | newly

            scale = torch.maximum(true_norm, threshold)
            drifted = step & ~newly & (
                (_finite_norm_converged(recurrence_norm, threshold) & (true_norm > threshold))
                | (torch.abs(recurrence_norm - true_norm) > drift_factor * scale)
            )
            grew = step & ~newly & finite_true & (
                true_norm > growth_factor * torch.clamp(previous_best, min=tiny)
            )
            nonfinite = step & ~finite_true
            restart = restart | drifted | grew | nonfinite

        restart = restart & ~(converged | failed)
        restart_rows(restart)
        continue_mask = step & ~(converged | failed | restart)
        if bool(torch.any(continue_mask)):
            # For SPD PCG, r^T z must be strictly positive.  Do not compare the
            # previous scalar against norms of the *new* residual: that can
            # falsely classify a legitimate late-iteration value as breakdown.
            safe_old_rz = torch.isfinite(rz) & (rz > 0.0)
            bad_old_rz = continue_mask & ~safe_old_rz
            restart_rows(bad_old_rz)
            continue_mask = continue_mask & ~bad_old_rz & ~failed
            beta = torch.where(
                continue_mask,
                rz_new / torch.where(safe_old_rz, rz, torch.ones_like(rz)),
                torch.zeros_like(rz),
            )
            bad_beta = continue_mask & ~torch.isfinite(beta)
            restart_rows(bad_beta)
            continue_mask = continue_mask & ~bad_beta & ~failed
            direction = torch.where(
                continue_mask[:, None, None] & active,
                z_new + beta[:, None, None] * direction,
                direction,
            )
            z = torch.where(continue_mask[:, None, None], z_new, z)
            rz = torch.where(continue_mask, rz_new, rz)

    true_current = torch.where(
        active, rhs - _apply_operator(correction, system), torch.zeros_like(rhs)
    )
    current_norm = _norm_batch(true_current)
    choose_best = best_norm < current_norm
    correction = torch.where(choose_best[:, None, None], best_correction, correction)
    final_residual = torch.where(
        active, rhs - _apply_operator(correction, system), torch.zeros_like(rhs)
    )
    final_norm = _norm_batch(final_residual)
    converged = _finite_norm_converged(final_norm, threshold)
    iterations = torch.where(
        converged & (iterations == 0), torch.ones_like(iterations), iterations
    )
    iterations = torch.where(
        ~converged, torch.full_like(iterations, int(maximum_iterations)), iterations
    )
    return correction, {
        "iterations": iterations,
        "restarts": restarts,
        "converged": converged,
        "failed": failed,
        "residual_norm": final_norm,
        "rhs_norm": rhs_norm,
    }


def _steepest_descent_correction_core(
    system: dict[str, torch.Tensor],
    rhs: torch.Tensor,
    solver_cfg: dict,
    *,
    maximum_iterations: int,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """MPS-resident residual-minimising fallback for an equilibrated SPD system."""
    active = system["active"]
    rhs = torch.where(active, rhs, torch.zeros_like(rhs))
    correction = torch.zeros_like(rhs)
    residual = rhs.clone()
    rhs_norm = _norm_batch(rhs)
    threshold = _solver_threshold(rhs_norm, relative_tolerance, absolute_tolerance)
    best = correction.clone()
    best_norm = _norm_batch(residual)
    converged = _finite_norm_converged(best_norm, threshold)
    iterations = torch.zeros(rhs.shape[0], device=rhs.device, dtype=torch.int64)
    eps = torch.finfo(rhs.dtype).eps
    tiny = torch.finfo(rhs.dtype).tiny
    guard = float(solver_cfg.get("breakdownGuardFactor", 16.0))
    check_interval = max(1, int(solver_cfg.get("fallbackCheckInterval", 10)))

    for iteration in range(1, max(0, int(maximum_iterations)) + 1):
        unresolved = ~converged
        if not bool(torch.any(unresolved)):
            break
        direction = torch.where(unresolved[:, None, None] & active, residual, torch.zeros_like(residual))
        ad = _apply_operator(direction, system)
        # Choose alpha to minimise ||r - alpha A r||_2 along the residual
        # direction.  Unlike the energy-minimising steepest-descent step
        # (r,r)/(r,Ar), this makes the fallback residual non-increasing in
        # exact arithmetic and is therefore the safer MPS last resort.
        r_ar = _dot_batch(residual, ad)
        ar_ar = _dot_batch(ad, ad)
        ad_norm = _norm_batch(ad)
        denominator_floor = guard * eps * ad_norm * ad_norm
        safe = unresolved & torch.isfinite(r_ar) & torch.isfinite(ar_ar) & (
            ar_ar > torch.maximum(denominator_floor, torch.full_like(ar_ar, tiny))
        )
        alpha = torch.where(
            safe,
            r_ar / torch.where(safe, ar_ar, torch.ones_like(ar_ar)),
            torch.zeros_like(r_ar),
        )
        candidate = correction + alpha[:, None, None] * direction
        candidate_residual = torch.where(
            active, rhs - _apply_operator(candidate, system), torch.zeros_like(rhs)
        )
        candidate_norm = _norm_batch(candidate_residual)
        improved = safe & torch.isfinite(candidate_norm) & (candidate_norm < best_norm)
        best = torch.where(improved[:, None, None], candidate, best)
        best_norm = torch.where(improved, candidate_norm, best_norm)
        correction = torch.where(improved[:, None, None], candidate, correction)
        residual = torch.where(improved[:, None, None], candidate_residual, residual)
        if iteration % check_interval == 0 or iteration == maximum_iterations:
            newly = unresolved & _finite_norm_converged(best_norm, threshold)
            iterations = torch.where(newly, torch.full_like(iterations, iteration), iterations)
            converged = converged | newly
        # A row that cannot produce a positive-curvature improving step has
        # exhausted this conservative fallback; leave its best iterate intact.
        stalled = unresolved & ~improved
        converged = converged | (
            stalled & _finite_norm_converged(best_norm, threshold)
        )

    iterations = torch.where(
        converged & (iterations == 0), torch.ones_like(iterations), iterations
    )
    iterations = torch.where(
        ~converged, torch.full_like(iterations, int(maximum_iterations)), iterations
    )
    return best, {
        "iterations": iterations,
        "converged": converged,
        "residual_norm": best_norm,
        "rhs_norm": rhs_norm,
    }


def _solve_pcg_system(
    potential: torch.Tensor,
    system: dict[str, torch.Tensor],
    solver_cfg: dict,
    *,
    maximum_iterations: int,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> tuple[torch.Tensor, SolverDiagnostics]:
    """Solve ``A x=b`` by equilibrated correction-form PCG and refinement."""
    active = system["active"]
    x = torch.where(active, potential, torch.zeros_like(potential))
    b = system["b"]
    b_norm = _norm_batch(b)
    threshold = _solver_threshold(b_norm, relative_tolerance, absolute_tolerance)
    equilibrated, scale = _equilibrate_linear_system(system)
    refinement_rounds_max = max(1, int(solver_cfg.get("iterativeRefinementRounds", 3)))
    fallback_method = str(solver_cfg.get("fallbackMethod", "equilibrated_steepest_descent")).lower()
    fallback_iterations = max(0, int(solver_cfg.get("fallbackMaximumIterations", 4000)))

    total_iterations = torch.zeros_like(b_norm, dtype=torch.int64)
    total_restarts = torch.zeros_like(total_iterations)
    rounds = torch.zeros_like(total_iterations)
    fallback_used = torch.zeros_like(b_norm, dtype=torch.bool)
    best_x = x.clone()
    residual = torch.where(active, b - _apply_operator(x, system), torch.zeros_like(x))
    best_norm = _norm_batch(residual)
    converged = _finite_norm_converged(best_norm, threshold)

    for refinement_round in range(1, refinement_rounds_max + 1):
        unresolved = ~converged
        if not bool(torch.any(unresolved)):
            break
        residual = torch.where(active, b - _apply_operator(x, system), torch.zeros_like(x))
        rhs_hat = torch.where(
            unresolved[:, None, None] & active, scale * residual, torch.zeros_like(residual)
        )
        correction_y, core = _pcg_correction_core(
            equilibrated,
            rhs_hat,
            solver_cfg,
            maximum_iterations=maximum_iterations,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        total_iterations = total_iterations + torch.where(unresolved, core["iterations"], torch.zeros_like(total_iterations))
        total_restarts = total_restarts + torch.where(unresolved, core["restarts"], torch.zeros_like(total_restarts))
        rounds = torch.where(unresolved, torch.full_like(rounds, refinement_round), rounds)
        candidate = torch.where(
            unresolved[:, None, None] & active,
            x + scale * correction_y,
            x,
        )
        candidate_residual = torch.where(
            active, b - _apply_operator(candidate, system), torch.zeros_like(candidate)
        )
        candidate_norm = _norm_batch(candidate_residual)
        improved = unresolved & torch.isfinite(candidate_norm) & (candidate_norm < best_norm)
        x = torch.where(improved[:, None, None], candidate, x)
        best_x = torch.where(improved[:, None, None], candidate, best_x)
        best_norm = torch.where(improved, candidate_norm, best_norm)
        converged = _finite_norm_converged(best_norm, threshold)
        # If a full correction round cannot improve a row, additional PCG
        # refinement from the same arithmetic state will not help.
        if not bool(torch.any(improved & ~converged)):
            break

    remaining = ~converged
    if bool(torch.any(remaining)) and fallback_method not in {"none", "off", "disabled"}:
        if fallback_method not in {
            "equilibrated_minimal_residual",
            "minimal_residual",
            "mr",
            "equilibrated_steepest_descent",
            "steepest_descent",
            "sd",
        }:
            raise ValueError(f"Unknown MPS-safe potential fallback method: {fallback_method}")
        residual = torch.where(active, b - _apply_operator(best_x, system), torch.zeros_like(best_x))
        rhs_hat = torch.where(
            remaining[:, None, None] & active, scale * residual, torch.zeros_like(residual)
        )
        correction_y, fallback = _steepest_descent_correction_core(
            equilibrated,
            rhs_hat,
            solver_cfg,
            maximum_iterations=fallback_iterations,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        candidate = torch.where(
            remaining[:, None, None] & active,
            best_x + scale * correction_y,
            best_x,
        )
        candidate_residual = torch.where(
            active, b - _apply_operator(candidate, system), torch.zeros_like(candidate)
        )
        candidate_norm = _norm_batch(candidate_residual)
        improved = remaining & torch.isfinite(candidate_norm) & (candidate_norm < best_norm)
        best_x = torch.where(improved[:, None, None], candidate, best_x)
        best_norm = torch.where(improved, candidate_norm, best_norm)
        total_iterations = total_iterations + torch.where(
            remaining, fallback["iterations"], torch.zeros_like(total_iterations)
        )
        fallback_used = fallback_used | remaining
        converged = _finite_norm_converged(best_norm, threshold)

    relative = best_norm / torch.clamp(b_norm, min=1.0)
    converged = converged & torch.isfinite(relative)
    diagnostics = SolverDiagnostics(
        iterations=total_iterations,
        relative_residual=relative,
        converged=converged,
        method=(
            "pcg_symmetric_equilibrated_correction"
            + ("_fp64" if potential.dtype == torch.float64 else "_fp32")
            + "_true_residual"
        ),
        restarts=total_restarts,
        refinement_rounds=rounds,
        fallback_used=fallback_used,
    )
    return best_x, diagnostics


def solve_pcg(
    potential: torch.Tensor,
    sigma: torch.Tensor,
    rhs: torch.Tensor,
    geometry: GeometryBatch,
    config: dict,
    voltage: float,
    spacing: float,
    *,
    static: bool = False,
    face_fixed_values: dict[str, torch.Tensor] | None = None,
    robin_linearization: dict[str, torch.Tensor] | None = None,
    neumann_boundary: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, SolverDiagnostics]:
    """Batched matrix-free SPD solve for Jacobi/equilibrated profiles."""
    solver_cfg = config["numerics"]["potentialSolver"]
    preconditioner = str(solver_cfg.get("preconditioner", "jacobi")).lower()
    if not bool(solver_cfg.get("symmetricEquilibration", True)):
        raise ValueError("This PCG implementation requires symmetricEquilibration=true")
    if not bool(solver_cfg.get("correctionForm", True)):
        raise ValueError("This PCG implementation requires correctionForm=true")
    if preconditioner not in {"jacobi", "diagonal", "identity", "none"}:
        raise ValueError(
            "PCG requires a symmetric positive preconditioner.  This release "
            "uses symmetric Jacobi equilibration; choose preconditioner=jacobi "
            "or select BiCGStab explicitly for the legacy multigrid V-cycle."
        )
    cathode_voltage = float(config["electrical"]["cathodeVoltage_V"])
    system = _build_linear_system(
        sigma, rhs, geometry, voltage, cathode_voltage, spacing,
        face_fixed_values, robin_linearization, neumann_boundary
    )
    maximum_iterations = int(
        solver_cfg["maximumIterationsStatic"] if static else solver_cfg["maximumIterationsCoupled"]
    )
    relative_tolerance = float(
        solver_cfg["relativeToleranceStatic"] if static else solver_cfg["relativeToleranceCoupled"]
    )
    absolute_tolerance = float(solver_cfg.get("absoluteTolerance", 1e-12))
    x, diagnostics = _solve_pcg_system(
        potential,
        system,
        solver_cfg,
        maximum_iterations=maximum_iterations,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )

    full = system["fixed_value"].clone()
    full = torch.where(system["active"], x, full)
    full = torch.where(system["gauge_mask"], system["gauge_value"], full)
    if not system["surface_overlay"]:
        full = apply_neumann_boundary(full)
    full = torch.where(
        geometry.anode & geometry.fixed,
        _batch_voltage_field(voltage, full),
        full,
    )
    full = torch.where(
        geometry.cathode & geometry.fixed,
        torch.as_tensor(cathode_voltage, device=full.device, dtype=full.dtype),
        full,
    )
    if bool(solver_cfg.get("failOnNonConvergence", True)) and not bool(
        torch.all(diagnostics.converged)
    ):
        bad = torch.nonzero(~diagnostics.converged).flatten().detach().cpu().tolist()
        values = diagnostics.relative_residual[~diagnostics.converged].detach().cpu().tolist()
        raise BCCandidateBatchError(
            f"PCG potential solve did not converge for batch indices {bad}; "
            f"residuals={values}; method={diagnostics.method}; "
            f"restarts={diagnostics.restarts[~diagnostics.converged].detach().cpu().tolist()}",
            bad,
            "potential_pcg_nonconvergence",
        )
    return full, diagnostics


def solve_bicgstab(
    potential: torch.Tensor,
    sigma: torch.Tensor,
    rhs: torch.Tensor,
    geometry: GeometryBatch,
    config: dict,
    voltage: float,
    spacing: float,
    *,
    static: bool = False,
    face_fixed_values: dict[str, torch.Tensor] | None = None,
    robin_linearization: dict[str, torch.Tensor] | None = None,
    neumann_boundary: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, SolverDiagnostics]:
    """Batched left-preconditioned BiCGStab with relative breakdown guards.

    This path is retained for the legacy non-symmetric geometric multigrid
    preconditioner.  All five sensitive scalars use norm-relative tests, and a
    candidate is restored to its best true-residual iterate before restart.
    """
    solver_cfg = config["numerics"]["potentialSolver"]
    cathode_voltage = float(config["electrical"]["cathodeVoltage_V"])
    system = _build_linear_system(
        sigma, rhs, geometry, voltage, cathode_voltage, spacing,
        face_fixed_values, robin_linearization, neumann_boundary
    )
    active = system["active"]
    x = torch.where(active, potential, torch.zeros_like(potential))
    b = system["b"]
    max_iterations = int(
        solver_cfg["maximumIterationsStatic"] if static else solver_cfg["maximumIterationsCoupled"]
    )
    tolerance = float(
        solver_cfg["relativeToleranceStatic"] if static else solver_cfg["relativeToleranceCoupled"]
    )
    absolute_tolerance = float(solver_cfg.get("absoluteTolerance", 1e-12))
    check_interval = max(1, int(solver_cfg.get("convergenceCheckInterval", 5)))
    guard_factor = float(solver_cfg.get("breakdownGuardFactor", 16.0))
    growth_factor = float(solver_cfg.get("residualGrowthRestartFactor", 4.0))
    maximum_restarts = max(0, int(solver_cfg.get("maximumKrylovRestarts", 8)))
    eps = torch.finfo(potential.dtype).eps
    tiny = torch.finfo(potential.dtype).tiny

    residual = torch.where(active, b - _apply_operator(x, system), torch.zeros_like(x))
    shadow = residual.clone()
    p_direction = torch.zeros_like(residual)
    v_direction = torch.zeros_like(residual)
    rho_old = torch.ones(residual.shape[0], device=residual.device, dtype=residual.dtype)
    alpha = torch.ones_like(rho_old)
    omega = torch.ones_like(rho_old)
    b_norm = _norm_batch(b)
    threshold = _solver_threshold(b_norm, tolerance, absolute_tolerance)
    r_norm = _norm_batch(residual)
    previous_r_norm = r_norm.clone()
    converged = _finite_norm_converged(r_norm, threshold)
    failed = torch.zeros_like(converged)
    iterations = torch.zeros_like(b_norm, dtype=torch.int64)
    restarts = torch.zeros_like(iterations)
    best_x = x.clone()
    best_norm = r_norm.clone()

    def restart_rows(mask: torch.Tensor) -> None:
        nonlocal x, residual, shadow, p_direction, v_direction
        nonlocal rho_old, alpha, omega, previous_r_norm, failed, restarts
        if not bool(torch.any(mask)):
            return
        restarts = torch.where(mask, restarts + 1, restarts)
        exhausted = mask & (restarts > maximum_restarts)
        failed = failed | exhausted
        retry = mask & ~exhausted
        if bool(torch.any(retry)):
            mask3 = retry[:, None, None]
            x = torch.where(mask3, best_x, x)
            true_residual = torch.where(
                active, b - _apply_operator(x, system), torch.zeros_like(x)
            )
            residual = torch.where(mask3, true_residual, residual)
            shadow = torch.where(mask3, true_residual, shadow)
            p_direction = torch.where(mask3, torch.zeros_like(p_direction), p_direction)
            v_direction = torch.where(mask3, torch.zeros_like(v_direction), v_direction)
            rho_old = torch.where(retry, torch.ones_like(rho_old), rho_old)
            alpha = torch.where(retry, torch.ones_like(alpha), alpha)
            omega = torch.where(retry, torch.ones_like(omega), omega)
            previous_r_norm = torch.where(retry, _norm_batch(true_residual), previous_r_norm)

    for iteration in range(1, max_iterations + 1):
        unresolved = ~(converged | failed)
        if not bool(torch.any(unresolved)):
            break
        current_r_norm = _norm_batch(residual)
        shadow_norm = _norm_batch(shadow)
        rho = _dot_batch(shadow, residual)
        rho_floor = guard_factor * eps * shadow_norm * current_r_norm
        rho_old_floor = guard_factor * eps * shadow_norm * previous_r_norm
        safe_rho = torch.isfinite(rho) & (
            rho.abs() > torch.maximum(rho_floor, torch.full_like(rho, tiny))
        )
        safe_rho_old = torch.isfinite(rho_old) & (
            rho_old.abs() > torch.maximum(rho_old_floor, torch.full_like(rho_old, tiny))
        )
        safe_omega_old = torch.isfinite(omega) & (omega.abs() > guard_factor * eps)
        breakdown = unresolved & ~(safe_rho & safe_rho_old & safe_omega_old)
        restart_rows(breakdown)
        step = ~(converged | failed | breakdown)
        if not bool(torch.any(step)):
            continue

        beta = torch.where(
            step,
            (rho / torch.where(safe_rho_old, rho_old, torch.ones_like(rho_old)))
            * (alpha / torch.where(safe_omega_old, omega, torch.ones_like(omega))),
            torch.zeros_like(rho),
        )
        bad_beta = step & ~torch.isfinite(beta)
        restart_rows(bad_beta)
        step = step & ~bad_beta & ~failed
        if not bool(torch.any(step)):
            continue
        p_direction = torch.where(
            step[:, None, None] & active,
            residual + beta[:, None, None] * (
                p_direction - omega[:, None, None] * v_direction
            ),
            p_direction,
        )
        p_hat = _apply_pcg_preconditioner(p_direction, system, solver_cfg)
        v_direction = _apply_operator(p_hat, system)
        denominator = _dot_batch(shadow, v_direction)
        v_norm = _norm_batch(v_direction)
        denominator_floor = guard_factor * eps * shadow_norm * v_norm
        safe_denominator = torch.isfinite(denominator) & (
            denominator.abs() > torch.maximum(
                denominator_floor, torch.full_like(denominator, tiny)
            )
        )
        breakdown = step & ~safe_denominator
        restart_rows(breakdown)
        step = step & ~breakdown & ~failed
        if not bool(torch.any(step)):
            continue
        alpha_new = torch.where(
            step,
            rho / torch.where(safe_denominator, denominator, torch.ones_like(denominator)),
            torch.zeros_like(rho),
        )
        bad_alpha = step & ~torch.isfinite(alpha_new)
        restart_rows(bad_alpha)
        step = step & ~bad_alpha & ~failed
        if not bool(torch.any(step)):
            continue
        alpha = torch.where(step, alpha_new, alpha)
        s_residual = residual - alpha[:, None, None] * v_direction
        x_after_alpha = x + torch.where(
            step[:, None, None] & active,
            alpha[:, None, None] * p_hat,
            torch.zeros_like(x),
        )
        s_norm = _norm_batch(s_residual)
        half_candidate = step & _finite_norm_converged(s_norm, threshold)
        if bool(torch.any(half_candidate)):
            true_half = torch.where(
                active, b - _apply_operator(x_after_alpha, system), torch.zeros_like(x)
            )
            true_half_norm = _norm_batch(true_half)
            half_converged = half_candidate & _finite_norm_converged(
                true_half_norm, threshold
            )
        else:
            true_half = torch.zeros_like(x)
            true_half_norm = torch.full_like(b_norm, torch.inf)
            half_converged = torch.zeros_like(converged)
        x = torch.where(step[:, None, None], x_after_alpha, x)
        residual = torch.where(
            half_converged[:, None, None] & active, true_half, residual
        )
        improved_half = half_converged & (true_half_norm < best_norm)
        best_x = torch.where(improved_half[:, None, None], x, best_x)
        best_norm = torch.where(improved_half, true_half_norm, best_norm)
        iterations = torch.where(
            half_converged, torch.full_like(iterations, iteration), iterations
        )
        converged = converged | half_converged
        step = step & ~half_converged
        if not bool(torch.any(step)):
            rho_old = torch.where(~converged, rho, rho_old)
            continue

        s_hat = _apply_pcg_preconditioner(s_residual, system, solver_cfg)
        t_direction = _apply_operator(s_hat, system)
        tt = _dot_batch(t_direction, t_direction)
        ts = _dot_batch(t_direction, s_residual)
        t_norm = _norm_batch(t_direction)
        tt_floor = guard_factor * eps * t_norm * t_norm
        safe_tt = torch.isfinite(tt) & (
            tt > torch.maximum(tt_floor, torch.full_like(tt, tiny))
        )
        omega_new = torch.where(
            step,
            ts / torch.where(safe_tt, tt, torch.ones_like(tt)),
            torch.ones_like(tt),
        )
        safe_omega = torch.isfinite(omega_new) & (omega_new.abs() > guard_factor * eps)
        breakdown = step & ~(safe_tt & safe_omega)
        restart_rows(breakdown)
        step = step & ~breakdown & ~failed
        if not bool(torch.any(step)):
            continue
        x = torch.where(
            step[:, None, None] & active,
            x + omega_new[:, None, None] * s_hat,
            x,
        )
        residual = torch.where(
            step[:, None, None] & active,
            s_residual - omega_new[:, None, None] * t_direction,
            residual,
        )
        omega = torch.where(step, omega_new, omega)
        rho_old = torch.where(step, rho, rho_old)
        previous_r_norm = torch.where(step, current_r_norm, previous_r_norm)

        if iteration % check_interval == 0 or iteration == max_iterations:
            recurrence_norm = _norm_batch(residual)
            true_now = torch.where(
                active, b - _apply_operator(x, system), torch.zeros_like(x)
            )
            true_norm = _norm_batch(true_now)
            previous_best = best_norm.clone()
            finite_true = torch.isfinite(true_norm)
            improved = step & finite_true & (true_norm < best_norm)
            best_x = torch.where(improved[:, None, None], x, best_x)
            best_norm = torch.where(improved, true_norm, best_norm)
            newly = step & _finite_norm_converged(true_norm, threshold)
            iterations = torch.where(
                newly, torch.full_like(iterations, iteration), iterations
            )
            converged = converged | newly
            drifted = step & ~newly & (
                _finite_norm_converged(recurrence_norm, threshold)
                & (true_norm > threshold)
            )
            grew = step & ~newly & finite_true & (
                true_norm > growth_factor * torch.clamp(previous_best, min=tiny)
            )
            restart_rows((drifted | grew | ~finite_true) & ~converged)

    true_current = torch.where(active, b - _apply_operator(x, system), torch.zeros_like(x))
    current_norm = _norm_batch(true_current)
    use_best = best_norm < current_norm
    x = torch.where(use_best[:, None, None], best_x, x)
    true_residual = torch.where(active, b - _apply_operator(x, system), torch.zeros_like(x))
    r_norm = _norm_batch(true_residual)
    converged = _finite_norm_converged(r_norm, threshold) & ~failed
    iterations = torch.where(
        converged & (iterations == 0), torch.ones_like(iterations), iterations
    )
    iterations = torch.where(
        ~converged, torch.full_like(iterations, max_iterations), iterations
    )

    full = system["fixed_value"].clone()
    full = torch.where(active, x, full)
    full = torch.where(system["gauge_mask"], system["gauge_value"], full)
    if not system["surface_overlay"]:
        full = apply_neumann_boundary(full)
    full = torch.where(
        geometry.anode & geometry.fixed,
        _batch_voltage_field(voltage, full),
        full,
    )
    full = torch.where(
        geometry.cathode & geometry.fixed,
        torch.as_tensor(cathode_voltage, device=full.device, dtype=full.dtype),
        full,
    )
    relative = r_norm / torch.clamp(b_norm, min=1.0)
    converged = converged & torch.isfinite(relative)
    diagnostics = SolverDiagnostics(
        iterations=iterations,
        relative_residual=relative,
        converged=converged,
        method=(
            "bicgstab_"
            + str(solver_cfg.get("preconditioner", "multigrid")).lower()
            + ("_fp64" if potential.dtype == torch.float64 else "_fp32")
            + "_relative_guards_best_restart"
        ),
        restarts=restarts,
    )
    if bool(solver_cfg.get("failOnNonConvergence", True)) and not bool(torch.all(converged)):
        bad = torch.nonzero(~converged).flatten().detach().cpu().tolist()
        values = relative[~converged].detach().cpu().tolist()
        raise RuntimeError(
            f"BiCGStab potential solve did not converge for batch indices {bad}; "
            f"residuals={values}; restarts={restarts[~converged].detach().cpu().tolist()}"
        )
    return full, diagnostics


def solve_direct_cpu(
    potential: torch.Tensor,
    sigma: torch.Tensor,
    rhs: torch.Tensor,
    geometry: GeometryBatch,
    config: dict,
    voltage: float,
    spacing: float,
    *,
    static: bool = False,
    face_fixed_values: dict[str, torch.Tensor] | None = None,
    robin_linearization: dict[str, torch.Tensor] | None = None,
    neumann_boundary: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, SolverDiagnostics]:
    """Solve the same discrete potential system with SciPy sparse direct LU.

    This path is intended as a CPU reference for port verification and small
    publication cross-checks.  It uses the *same eliminated-boundary linear
    system* as the matrix-free PCG path, then returns the result on the input
    tensor device.  It is not the recommended production solver on an A100.
    """
    from scipy.sparse import coo_matrix  # imported lazily for CUDA production
    from scipy.sparse.linalg import MatrixRankWarning, spsolve
    import warnings

    if potential.dtype != torch.float64 or sigma.dtype != torch.float64 or rhs.dtype != torch.float64:
        raise ValueError(
            "SciPy direct reference solver requires FP64 tensors; "
            "set numerics.physicsDtype: float64"
        )
    solver_cfg = config["numerics"]["potentialSolver"]
    cathode_voltage = float(config["electrical"]["cathodeVoltage_V"])
    system = _build_linear_system(
        sigma, rhs, geometry, voltage, cathode_voltage, spacing,
        face_fixed_values, robin_linearization, neumann_boundary
    )
    batch_size = int(potential.shape[0])
    maximum_unknowns = int(solver_cfg.get("directMaximumUnknowns", 500_000))
    active_solution = torch.zeros_like(potential)

    # Convert coefficient fields once.  Direct solves are deliberately done
    # in float64 regardless of the caller dtype to make this a reference path.
    active_all = system["active"].detach().cpu().numpy().astype(bool, copy=False)
    diagonal_all = system["diagonal"].detach().cpu().numpy().astype(np.float64, copy=False)
    ce_all = system["ce"].detach().cpu().numpy().astype(np.float64, copy=False)
    cw_all = system["cw"].detach().cpu().numpy().astype(np.float64, copy=False)
    cn_all = system["cn"].detach().cpu().numpy().astype(np.float64, copy=False)
    cs_all = system["cs"].detach().cpu().numpy().astype(np.float64, copy=False)
    b_all = system["b"].detach().cpu().numpy().astype(np.float64, copy=False)

    for batch_index in range(batch_size):
        active = active_all[batch_index]
        coordinates = np.argwhere(active)
        unknown_count = int(len(coordinates))
        if unknown_count > maximum_unknowns:
            raise RuntimeError(
                "SciPy direct potential solve exceeds directMaximumUnknowns: "
                f"{unknown_count} > {maximum_unknowns}. Use PCG for production."
            )
        if unknown_count == 0:
            continue

        index_map = np.full(active.shape, -1, dtype=np.int64)
        index_map[active] = np.arange(unknown_count, dtype=np.int64)
        matrix_rows: list[int] = []
        matrix_columns: list[int] = []
        matrix_values: list[float] = []

        for local_index, (row, column) in enumerate(coordinates):
            row_i = int(row)
            column_i = int(column)
            matrix_rows.append(local_index)
            matrix_columns.append(local_index)
            matrix_values.append(float(diagonal_all[batch_index, row_i, column_i]))

            neighbors = (
                (row_i, column_i + 1, ce_all[batch_index, row_i, column_i]),
                (row_i, column_i - 1, cw_all[batch_index, row_i, column_i]),
                (row_i + 1, column_i, cn_all[batch_index, row_i, column_i]),
                (row_i - 1, column_i, cs_all[batch_index, row_i, column_i]),
            )
            for neighbor_row, neighbor_column, coefficient in neighbors:
                # A surface-overlay control volume may lie on the outer ring.
                # Its geometrically outward face has a zero coefficient (the
                # natural no-flux condition), but indexing that neighbour
                # before checking the coefficient can either wrap at -1 or
                # raise at shape[row/column].  Reject such faces explicitly so
                # the direct reference path assembles exactly the same stencil
                # as the matrix-free solvers.
                if coefficient == 0.0 or not (
                    0 <= neighbor_row < active.shape[0]
                    and 0 <= neighbor_column < active.shape[1]
                ):
                    continue
                neighbor_index = index_map[neighbor_row, neighbor_column]
                if neighbor_index >= 0:
                    matrix_rows.append(local_index)
                    matrix_columns.append(int(neighbor_index))
                    matrix_values.append(-float(coefficient))

        matrix = coo_matrix(
            (matrix_values, (matrix_rows, matrix_columns)),
            shape=(unknown_count, unknown_count),
            dtype=np.float64,
        ).tocsr()
        right_hand_side = b_all[batch_index][active]
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            solution_np = np.asarray(spsolve(matrix, right_hand_side), dtype=np.float64)
        if not np.all(np.isfinite(solution_np)):
            raise RuntimeError(
                f"Non-finite SciPy direct potential solution for batch {batch_index}"
            )
        solution_tensor = torch.as_tensor(
            solution_np,
            device=potential.device,
            dtype=potential.dtype,
        )
        active_solution[batch_index][system["active"][batch_index]] = solution_tensor

    full = system["fixed_value"].clone()
    full = torch.where(system["active"], active_solution, full)
    full = torch.where(system["gauge_mask"], system["gauge_value"], full)
    if not system["surface_overlay"]:
        full = apply_neumann_boundary(full)
    full = torch.where(
        geometry.anode & geometry.fixed,
        _batch_voltage_field(voltage, full),
        full,
    )
    full = torch.where(
        geometry.cathode & geometry.fixed,
        torch.as_tensor(cathode_voltage, device=full.device, dtype=full.dtype),
        full,
    )

    x_active = torch.where(system["active"], full, torch.zeros_like(full))
    residual = system["b"] - _apply_operator(x_active, system)
    b_norm = _norm_batch(system["b"])
    r_norm = _norm_batch(residual)
    relative = r_norm / torch.clamp(b_norm, min=1.0)
    relative_tolerance = float(
        solver_cfg.get(
            "relativeToleranceStatic" if static else "relativeToleranceCoupled",
            1e-10 if static else 1e-9,
        )
    )
    absolute_tolerance = float(solver_cfg.get("absoluteTolerance", 1e-12))
    threshold = torch.maximum(
        relative_tolerance * torch.clamp(b_norm, min=1.0),
        torch.full_like(b_norm, absolute_tolerance),
    )
    converged = _finite_norm_converged(r_norm, threshold) & torch.isfinite(relative)
    diagnostics = SolverDiagnostics(
        iterations=torch.ones_like(b_norm, dtype=torch.int64),
        relative_residual=relative,
        converged=converged,
        method="scipy_sparse_direct_cpu_fp64",
    )
    if bool(solver_cfg.get("failOnNonConvergence", True)) and not bool(
        torch.all(converged)
    ):
        raise RuntimeError(
            "SciPy direct potential solve residual exceeded configured tolerance: "
            f"{relative.detach().cpu().tolist()}"
        )
    return full, diagnostics


def _weighted_candidate(potential: torch.Tensor, sigma: torch.Tensor, rhs: torch.Tensor, spacing: float) -> torch.Tensor:
    candidate = potential.clone()
    sc = sigma[..., 1:-1, 1:-1]
    se = harmonic_mean(sc, sigma[..., 1:-1, 2:])
    sw = harmonic_mean(sc, sigma[..., 1:-1, :-2])
    sn = harmonic_mean(sc, sigma[..., 2:, 1:-1])
    ss = harmonic_mean(sc, sigma[..., :-2, 1:-1])
    denominator = se + sw + sn + ss
    numerator = (
        se * potential[..., 1:-1, 2:]
        + sw * potential[..., 1:-1, :-2]
        + sn * potential[..., 2:, 1:-1]
        + ss * potential[..., :-2, 1:-1]
        - rhs[..., 1:-1, 1:-1] * (spacing * spacing)
    )
    positive = denominator > 0.0
    candidate[..., 1:-1, 1:-1] = torch.where(
        positive,
        numerator / torch.where(positive, denominator, torch.ones_like(denominator)),
        potential[..., 1:-1, 1:-1],
    )
    return candidate


def solve_rb_sor(
    potential: torch.Tensor,
    sigma: torch.Tensor,
    rhs: torch.Tensor,
    geometry: GeometryBatch,
    config: dict,
    voltage: float,
    spacing: float,
    *,
    static: bool = False,
    face_fixed_values: dict[str, torch.Tensor] | None = None,
    robin_linearization: dict[str, torch.Tensor] | None = None,
    neumann_boundary: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, SolverDiagnostics]:
    """Port of the MATLAB red-black SOR path for parity checks."""
    cathode_voltage = float(config["electrical"]["cathodeVoltage_V"])
    if static:
        omega = float(config["electrical"]["omega"])
        max_iterations = int(config["electrical"]["maximumIterations"])
        tolerance = float(config["electrical"]["tolerance_V"])
    else:
        omega = float(config["coupled"]["electricalOmega"])
        max_iterations = int(config["coupled"]["electricalMaximumIterations"])
        tolerance = float(config["coupled"]["electricalTolerance_V"])
    rows = torch.arange(potential.shape[-2], device=potential.device)[:, None]
    cols = torch.arange(potential.shape[-1], device=potential.device)[None, :]
    interior = torch.ones_like(geometry.fixed)
    interior[..., 0, :] = False
    interior[..., -1, :] = False
    interior[..., :, 0] = False
    interior[..., :, -1] = False
    red = ((rows + cols) % 2 == 0)[None, ...] & interior & ~geometry.fixed
    black = ((rows + cols) % 2 == 1)[None, ...] & interior & ~geometry.fixed
    unresolved = torch.ones(potential.shape[0], device=potential.device, dtype=torch.bool)
    iterations = torch.zeros(potential.shape[0], device=potential.device, dtype=torch.int64)
    maximum_change = torch.full((potential.shape[0],), torch.inf, device=potential.device, dtype=potential.dtype)
    fixed_value = _fixed_potential(geometry, voltage, cathode_voltage, potential.dtype)
    system_override = None
    if face_fixed_values is not None or robin_linearization is not None or neumann_boundary is not None:
        system_override = _build_linear_system(
            sigma, rhs, geometry, voltage, cathode_voltage, spacing,
            face_fixed_values, robin_linearization, neumann_boundary
        )
        if system_override["surface_overlay"]:
            red = ((rows + cols) % 2 == 0)[None, ...] & system_override["active"]
            black = ((rows + cols) % 2 == 1)[None, ...] & system_override["active"]
    current = potential.clone()
    for iteration in range(1, max_iterations + 1):
        maximum_change = torch.zeros_like(maximum_change)
        for color in (red, black):
            if system_override is None or not system_override["surface_overlay"]:
                current = apply_neumann_boundary(current)
            current = torch.where(geometry.fixed, fixed_value, current)
            if system_override is not None:
                current = torch.where(system_override["gauge_mask"], system_override["gauge_value"], current)
            if system_override is None:
                candidate = _weighted_candidate(current, sigma, rhs, spacing)
            else:
                candidate = current.clone()
                active = system_override["active"]
                numerator = (
                    system_override["b"]
                    + system_override["ce"] * system_override["active_e"].to(current.dtype) * _shift_east(current)
                    + system_override["cw"] * system_override["active_w"].to(current.dtype) * _shift_west(current)
                    + system_override["cn"] * system_override["active_n"].to(current.dtype) * _shift_north(current)
                    + system_override["cs"] * system_override["active_s"].to(current.dtype) * _shift_south(current)
                )
                candidate = torch.where(
                    active, numerator / torch.clamp(system_override["diagonal"], min=torch.finfo(current.dtype).eps), candidate
                )
            update_mask = color & unresolved[:, None, None]
            proposed = (1.0 - omega) * current + omega * candidate
            difference = torch.where(update_mask, torch.abs(proposed - current), torch.zeros_like(current))
            maximum_change = torch.maximum(maximum_change, difference.amax(dim=(-2, -1)))
            current = torch.where(update_mask, proposed, current)
        current = torch.where(geometry.fixed, fixed_value, current)
        if system_override is not None:
            current = torch.where(system_override["gauge_mask"], system_override["gauge_value"], current)
        newly = unresolved & (maximum_change < tolerance)
        iterations = torch.where(newly, torch.full_like(iterations, iteration), iterations)
        unresolved = unresolved & ~newly
        if not bool(torch.any(unresolved)):
            break
    iterations = torch.where(unresolved, torch.full_like(iterations, max_iterations), iterations)
    system = _build_linear_system(
        sigma, rhs, geometry, voltage, cathode_voltage, spacing,
        face_fixed_values, robin_linearization, neumann_boundary
    )
    x = torch.where(system["active"], current, torch.zeros_like(current))
    residual = system["b"] - _apply_operator(x, system)
    b_norm = _norm_batch(system["b"])
    r_norm = _norm_batch(residual)
    relative = r_norm / torch.clamp(b_norm, min=1.0)
    converged = ~unresolved & torch.isfinite(relative)
    return current, SolverDiagnostics(iterations=iterations, relative_residual=relative, converged=converged, method="rb_sor_matlab_parity")


def solve_potential(
    potential: torch.Tensor,
    cation: torch.Tensor,
    anion: torch.Tensor,
    dcation: torch.Tensor,
    danion: torch.Tensor,
    sigma_total: torch.Tensor,
    geometry: GeometryBatch,
    config: dict,
    voltage: float,
    spacing: float,
    *,
    static: bool = False,
    face_fixed_values: dict[str, torch.Tensor] | None = None,
    robin_linearization: dict[str, torch.Tensor] | None = None,
    neumann_boundary: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, SolverDiagnostics, torch.Tensor]:
    rhs = diffusion_current_divergence(
        cation, anion, dcation, danion, spacing, config, geometry
    )
    rhs = torch.where(geometry.fixed, torch.zeros_like(rhs), rhs)
    solver_cfg = config.get("numerics", {}).get("potentialSolver", {})
    stage_key = "methodStatic" if static else "methodCoupled"
    method = str(solver_cfg.get(stage_key, solver_cfg.get("method", "auto"))).lower()
    preconditioner = str(solver_cfg.get("preconditioner", "multigrid")).lower()
    if method == "auto":
        method = (
            "bicgstab"
            if preconditioner in {"multigrid", "mg", "geometric_multigrid"}
            else "pcg"
        )
    # Never silently change an explicitly requested algorithm.  The legacy
    # geometric V-cycle is not a symmetric Galerkin preconditioner, so using it
    # with PCG would invalidate the method's assumptions and make diagnostics
    # misleading.
    if (
        method == "pcg"
        and preconditioner in {"multigrid", "mg", "geometric_multigrid"}
        and not bool(solver_cfg.get("allowNonSPDPreconditionedPCG", False))
    ):
        raise ValueError(
            "Invalid potential solver combination: method=pcg with the legacy "
            "non-symmetric multigrid preconditioner. Use preconditioner=jacobi "
            "for equilibrated PCG, or request method=bicgstab explicitly."
        )
    if method == "pcg":
        result, diagnostics = solve_pcg(
            potential, sigma_total, rhs, geometry, config, voltage, spacing,
            static=static, face_fixed_values=face_fixed_values,
            robin_linearization=robin_linearization, neumann_boundary=neumann_boundary
        )
    elif method in {"bicgstab", "bicg", "nonsymmetric_krylov"}:
        result, diagnostics = solve_bicgstab(
            potential, sigma_total, rhs, geometry, config, voltage, spacing,
            static=static, face_fixed_values=face_fixed_values,
            robin_linearization=robin_linearization, neumann_boundary=neumann_boundary
        )
    elif method in {"rb_sor", "sor", "matlab_parity"}:
        result, diagnostics = solve_rb_sor(
            potential, sigma_total, rhs, geometry, config, voltage, spacing,
            static=static, face_fixed_values=face_fixed_values,
            robin_linearization=robin_linearization, neumann_boundary=neumann_boundary
        )
    elif method in {"direct", "direct_cpu", "scipy_direct"}:
        result, diagnostics = solve_direct_cpu(
            potential, sigma_total, rhs, geometry, config, voltage, spacing,
            static=static, face_fixed_values=face_fixed_values,
            robin_linearization=robin_linearization, neumann_boundary=neumann_boundary
        )
    else:
        raise ValueError(f"Unknown potential solver method: {method}")
    return result, diagnostics, rhs
