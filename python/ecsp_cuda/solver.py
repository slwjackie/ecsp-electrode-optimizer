from __future__ import annotations

"""Batched CUDA/FP64 implementation of the v7.9.x surface-contact model.

The implementation mirrors ``cpp/ecsp_cpp_solver.cpp`` at the equation level:
full propellant domain, surface-contact Butler--Volmer terms, matrix-free PCG,
Nernst--Planck species transport, thermal/decomposition update and the same
scalar metrics.  It is intentionally a *second backend*; the existing C++ CPU
solver remains available as the reference/fallback path.

The CUDA solver advances many independent electrode candidates in one tensor
batch.  Candidate-specific voltages and early-stop flags are supported, which
allows the Python V_min controller to batch different bisection voltages on a
single A100 launch stream.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import math
import time

import numpy as np


class CudaPhysicsError(RuntimeError):
    pass


@dataclass(frozen=True)
class CudaBatchCase:
    anode: np.ndarray
    cathode: np.ndarray
    config: Mapping[str, Any]
    output_dir: Path
    geometry_id: str


def _canonical_numpy_bool_mask(mask: Any) -> np.ndarray:
    """Return an owned 0/1 bool array, including for Pillow mode-1 input."""
    return np.asarray(mask, dtype=np.uint8) != 0


def _strict_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _strict_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _same_float(a: Any, b: Any, *, atol: float = 0.0, rtol: float = 1.0e-13) -> bool:
    af = float(a)
    bf = float(b)
    return abs(af - bf) <= atol + rtol * max(abs(af), abs(bf), 1.0)


class TorchCudaSurfaceContactBatchRunner:
    """Execute a batch of complete condensed-phase trials on one CUDA device.

    The runner deliberately keeps every state tensor resident on the GPU from
    initialization to the final scalar reduction.  No timestep-level host
    transfer occurs.  A single runner instance is reusable across reference
    evaluations and V_min bisection trials.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        *,
        dtype_name: str = "float64",
        deterministic: bool = True,
        allow_tf32: bool = False,
        allow_cpu_for_test: bool = False,
    ) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda" and not allow_cpu_for_test:
            raise CudaPhysicsError(f"CUDA runner requires a CUDA device, got {device!r}")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise CudaPhysicsError("torch.cuda.is_available() is false")
        if dtype_name.lower() not in {"float64", "double", "fp64"}:
            raise CudaPhysicsError("A100 production backend is intentionally FP64")
        self.dtype = torch.float64
        self.deterministic = bool(deterministic)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
            torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
        if self.deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
        # Host-side convergence checks synchronize the CUDA stream.  The
        # per-candidate masks are still updated every iteration on-device; only
        # the expensive host question "are all rows done?" is throttled.
        self.pcg_host_check_interval = 8
        self.local_face_host_check_interval = 8
        self.robin_host_check_interval = 4
        self.time_host_check_interval = 64

    def verify(self) -> dict[str, Any]:
        torch = self.torch
        props = torch.cuda.get_device_properties(self.device) if self.device.type == "cuda" else None
        probe = torch.tensor([1.0, 2.0], device=self.device, dtype=self.dtype)
        value = float((probe * probe).sum().item())
        if value != 5.0:
            raise CudaPhysicsError("CUDA FP64 arithmetic probe failed")
        return {
            "engine": "torch_cuda_surface_contact_batch",
            "torch_version": torch.__version__,
            "device": str(self.device),
            "device_name": props.name if props is not None else "cpu_test_mode",
            "total_memory_bytes": int(props.total_memory) if props is not None else 0,
            "compute_capability": f"{props.major}.{props.minor}" if props is not None else None,
            "dtype": "float64",
        }

    @staticmethod
    def _validate_common_config(cases: Sequence[CudaBatchCase]) -> Mapping[str, Any]:
        if not cases:
            raise CudaPhysicsError("empty CUDA batch")
        base = cases[0].config
        invariant_keys = (
            "n", "domain", "layer", "cathode_voltage", "dt", "end_time",
            "eval_time", "electrical_interval", "density", "R", "F",
            "pcg_max_iter", "pcg_rtol_static", "pcg_rtol_coupled", "pcg_atol",
        )
        for case in cases[1:]:
            for key in invariant_keys:
                if key not in base or key not in case.config:
                    raise CudaPhysicsError(f"missing common CUDA config key {key!r}")
                a, b = base[key], case.config[key]
                if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
                    if int(a) != int(b):
                        raise CudaPhysicsError(f"CUDA batch mixes {key}: {a} vs {b}")
                elif not _same_float(a, b):
                    raise CudaPhysicsError(f"CUDA batch mixes {key}: {a} vs {b}")
        return base

    @staticmethod
    def _channel(config: Mapping[str, Any], prefix: str) -> dict[str, float]:
        return {
            "Eeq": float(config[f"{prefix}.Eeq"]),
            "j0": float(config[f"{prefix}.j0"]),
            "alpha": float(config[f"{prefix}.alpha"]),
            "reverse": float(config[f"{prefix}.reverse"]),
            "n": float(config[f"{prefix}.n"]),
            "km": float(config[f"{prefix}.km"]),
            "dH": float(config[f"{prefix}.dH"]),
            "water_stoich": float(config[f"{prefix}.water_stoich"]),
            "salt_stoich": float(config[f"{prefix}.salt_stoich"]),
            "gas_yield": float(config[f"{prefix}.gas_yield"]),
            "ref_activity": float(config[f"{prefix}.ref_activity"]),
            "nernst_exp": float(config[f"{prefix}.nernst_exp"]),
        }

    def _to_scalar_tensor(self, values: Sequence[float]) -> Any:
        torch = self.torch
        return torch.as_tensor(values, device=self.device, dtype=self.dtype).view(-1, 1, 1)

    def _harmonic(self, a: Any, b: Any) -> Any:
        torch = self.torch
        denom = a + b
        return torch.where((a > 0.0) & (b > 0.0) & (denom > 0.0), 2.0 * a * b / denom, torch.zeros_like(denom))

    def _sigmoid(self, x: Any) -> Any:
        return self.torch.sigmoid(x)

    def _neumann_edges(self, field: Any) -> Any:
        # Clone avoids autograd/version concerns and guarantees source values
        # are read before the four edge writes.  Physics runs in inference mode.
        out = field.clone()
        out[:, 0, :] = out[:, 1, :]
        out[:, -1, :] = out[:, -2, :]
        out[:, :, 0] = out[:, :, 1]
        out[:, :, -1] = out[:, :, -2]
        return out

    def _init_state(self, anode: Any, cathode: Any, cfg: Mapping[str, Any], voltage: Any) -> dict[str, Any]:
        torch = self.torch
        batch, n, _ = anode.shape
        rr = torch.arange(n, device=self.device, dtype=self.dtype).view(1, n, 1)
        cc = torch.arange(n, device=self.device, dtype=self.dtype).view(1, 1, n)
        af = anode.to(self.dtype)
        cf = cathode.to(self.dtype)
        acount = af.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        ccount = cf.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        ar = (af * rr).sum(dim=(-2, -1), keepdim=True) / acount
        acol = (af * cc).sum(dim=(-2, -1), keepdim=True) / acount
        cr = (cf * rr).sum(dim=(-2, -1), keepdim=True) / ccount
        ccol = (cf * cc).sum(dim=(-2, -1), keepdim=True) / ccount
        vr = ar - cr
        vc = acol - ccol
        norm2 = (vr * vr + vc * vc).clamp_min(1.0e-20)
        projection = ((rr - cr) * vr + (cc - ccol) * vc) / norm2
        xi = projection.clamp(0.0, 1.0)
        cathode_v = float(cfg["cathode_voltage"])
        drop = float(cfg["initial_contact_drop"])
        available = (voltage - cathode_v - 2.0 * drop).clamp_min(0.1)
        shape = (batch, n, n)
        full = lambda value: torch.full(shape, float(value), device=self.device, dtype=self.dtype)
        return {
            "T": full(cfg["T0"]),
            "cp": full(cfg["cation0"]),
            "cm": full(cfg["anion0"]),
            "water": full(cfg["water0"]),
            "liquid": torch.zeros(shape, device=self.device, dtype=self.dtype),
            "alpha": torch.zeros(shape, device=self.device, dtype=self.dtype),
            "gas": torch.zeros(shape, device=self.device, dtype=self.dtype),
            "pass": torch.zeros(shape, device=self.device, dtype=self.dtype),
            "coverage": torch.zeros(shape, device=self.device, dtype=self.dtype),
            "phi": cathode_v + drop + xi * available,
        }

    def _transport_fields(self, s: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
        torch = self.torch
        T = s["T"].clamp_min(1.0)
        wa = (s["water"] / max(float(cfg["water0"]), 1.0e-12)).clamp(0.0, 1.0)
        wf = wa.pow(float(cfg["water_activity_exp"]))
        Dp = float(cfg["Dp_dry"]) + (float(cfg["Dp_wet"]) - float(cfg["Dp_dry"])) * wf
        Dm = float(cfg["Dm_dry"]) + (float(cfg["Dm_wet"]) - float(cfg["Dm_dry"])) * wf
        R = float(cfg["R"])
        Tref = float(cfg["Tref"])
        pa = torch.exp(-float(cfg["Ea_Dp"]) / R * (1.0 / T - 1.0 / Tref))
        ma = torch.exp(-float(cfg["Ea_Dm"]) / R * (1.0 / T - 1.0 / Tref))
        threshold = float(cfg["fast_ion_liquid_threshold"])
        fast = ((s["liquid"] - threshold) / max(1.0 - threshold, 1.0e-30)).clamp(0.0, 1.0)
        phase_gain = 1.0 + float(cfg["liquid_D_gain"]) * fast
        plasticizer = 1.0 + float(cfg["glycerol_gain"]) * float(cfg["glycerol_pva_ratio"])
        crosslink = 1.0 / (1.0 + float(cfg["crosslink_penalty"]) * float(cfg["boric_pva_ratio"]))
        Dp = Dp * pa * phase_gain * plasticizer * crosslink
        Dm = Dm * ma * phase_gain * plasticizer * crosslink
        Dw = float(cfg["Dw0"]) * torch.exp(-12000.0 / R * (1.0 / T - 1.0 / Tref)) * (1.0 + 4.0 * fast)
        F = float(cfg["F"])
        zplus = float(cfg["zplus"])
        zminus = float(cfg["zminus"])
        sigi = F * F / (R * T) * (
            zplus * zplus * Dp * s["cp"].clamp_min(0.0)
            + zminus * zminus * Dm * s["cm"].clamp_min(0.0)
        )
        solid = (1.0 - float(cfg["sigma_e_liquid_suppression"]) * s["liquid"]).clamp_min(0.01)
        sige = float(cfg["sigma_e0"]) * torch.exp(
            float(cfg["sigma_e_temp_coeff"]) * (T - float(cfg["T0"]))
        ) * solid
        sig = (sige + sigi).clamp(float(cfg["sigma_min"]), float(cfg["sigma_max"]))
        return {
            "Dp": Dp, "Dm": Dm, "Dw": Dw, "sigi": sigi, "sige": sige,
            "sig": sig, "waterAct": wa, "fast": fast,
        }

    def _diffusion_rhs(self, s: Mapping[str, Any], t: Mapping[str, Any], cfg: Mapping[str, Any], h: float) -> Any:
        torch = self.torch
        if not bool(cfg["diffusion_potential"]):
            return torch.zeros_like(s["phi"])
        batch, n, _ = s["phi"].shape
        fx = torch.zeros((batch, n, n + 1), device=self.device, dtype=self.dtype)
        fy = torch.zeros((batch, n + 1, n), device=self.device, dtype=self.dtype)
        dp = self._harmonic(t["Dp"][:, :, :-1], t["Dp"][:, :, 1:])
        dm = self._harmonic(t["Dm"][:, :, :-1], t["Dm"][:, :, 1:])
        fx[:, :, 1:n] = -float(cfg["F"]) * (
            float(cfg["zplus"]) * dp * (s["cp"][:, :, 1:] - s["cp"][:, :, :-1]) / h
            + float(cfg["zminus"]) * dm * (s["cm"][:, :, 1:] - s["cm"][:, :, :-1]) / h
        )
        dp = self._harmonic(t["Dp"][:, :-1, :], t["Dp"][:, 1:, :])
        dm = self._harmonic(t["Dm"][:, :-1, :], t["Dm"][:, 1:, :])
        fy[:, 1:n, :] = -float(cfg["F"]) * (
            float(cfg["zplus"]) * dp * (s["cp"][:, 1:, :] - s["cp"][:, :-1, :]) / h
            + float(cfg["zminus"]) * dm * (s["cm"][:, 1:, :] - s["cm"][:, :-1, :]) / h
        )
        return (fx[:, :, 1:] - fx[:, :, :-1]) / h + (fy[:, 1:, :] - fy[:, :-1, :]) / h

    def _build_system(
        self,
        s: Mapping[str, Any],
        t: Mapping[str, Any],
        cfg: Mapping[str, Any],
        h: float,
        contact: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        torch = self.torch
        sig = t["sig"]
        batch, n, _ = sig.shape
        zeros = lambda: torch.zeros_like(sig)
        ce, cw, cn, cs = zeros(), zeros(), zeros(), zeros()
        if n > 3:
            ce[:, 1:-1, 1:-2] = self._harmonic(sig[:, 1:-1, 1:-2], sig[:, 1:-1, 2:-1])
            cw[:, 1:-1, 2:-1] = self._harmonic(sig[:, 1:-1, 2:-1], sig[:, 1:-1, 1:-2])
            cn[:, 1:-2, 1:-1] = self._harmonic(sig[:, 1:-2, 1:-1], sig[:, 2:-1, 1:-1])
            cs[:, 2:-1, 1:-1] = self._harmonic(sig[:, 2:-1, 1:-1], sig[:, 1:-2, 1:-1])
        diag = ce + cw + cn + cs
        rhs = self._diffusion_rhs(s, t, cfg, h)
        b = -rhs * h * h
        if contact is not None:
            ja = contact["anodeCurrent"]
            jc = contact["cathodeCurrent"]
            ka = contact["anodeSlope"].clamp_min(0.0)
            kc = contact["cathodeSlope"].clamp_min(0.0)
            slope = ka + kc
            scale = h * h / max(float(cfg["layer"]), 1.0e-30)
            diag = diag + scale * slope
            b = b + scale * (ja - jc + slope * s["phi"])
        active = torch.zeros((1, n, n), device=self.device, dtype=torch.bool)
        active[:, 1:-1, 1:-1] = True
        diagonal_valid = (
            (diag[:, 1:-1, 1:-1] > 0.0)
            & torch.isfinite(diag[:, 1:-1, 1:-1])
        ).all(dim=(-2, -1))
        return {
            "ce": ce, "cw": cw, "cn": cn, "cs": cs,
            "diag": diag, "b": b, "active": active,
            "diagonal_valid": diagonal_valid,
        }

    def _apply_A(self, A: Mapping[str, Any], x: Any) -> Any:
        torch = self.torch
        y = torch.zeros_like(x)
        center = A["diag"][:, 1:-1, 1:-1] * x[:, 1:-1, 1:-1]
        center = center - A["ce"][:, 1:-1, 1:-1] * x[:, 1:-1, 2:]
        center = center - A["cw"][:, 1:-1, 1:-1] * x[:, 1:-1, :-2]
        center = center - A["cn"][:, 1:-1, 1:-1] * x[:, 2:, 1:-1]
        center = center - A["cs"][:, 1:-1, 1:-1] * x[:, :-2, 1:-1]
        y[:, 1:-1, 1:-1] = center
        return y

    @staticmethod
    def _dot(a: Any, b: Any) -> Any:
        # Two-stage reduction is deterministic on CUDA under the selected
        # PyTorch deterministic policy and reduces the summation depth.
        return (a[:, 1:-1, 1:-1] * b[:, 1:-1, 1:-1]).sum(dim=-1).sum(dim=-1)

    def _solve_pcg(
        self,
        A: Mapping[str, Any],
        x: Any,
        cfg: Mapping[str, Any],
        relative_tolerance: float,
        case_mask: Any,
    ) -> tuple[Any, dict[str, Any]]:
        torch = self.torch
        batch = x.shape[0]
        requested_case = case_mask.clone()
        diagonal_valid = A.get(
            "diagonal_valid",
            torch.ones(batch, device=self.device, dtype=torch.bool),
        )
        active_case = requested_case & diagonal_valid
        Ax = self._apply_A(A, x)
        r = A["b"] - Ax
        r = torch.where(A["active"], r, torch.zeros_like(r))
        z = torch.where(A["active"], r / A["diag"].clamp_min(1.0e-300), torch.zeros_like(r))
        p = z.clone()
        bnorm = self._dot(A["b"], A["b"]).clamp_min(0.0).sqrt()
        scale = torch.maximum(bnorm, torch.ones_like(bnorm))
        threshold = torch.maximum(
            float(relative_tolerance) * scale,
            torch.full_like(scale, float(cfg["pcg_atol"])),
        )
        rz = self._dot(r, z)
        iterations = torch.zeros(batch, device=self.device, dtype=torch.int64)
        failed = requested_case & ~diagonal_valid
        eps = torch.finfo(self.dtype).eps
        guard = 32.0 * eps
        max_iter = int(cfg["pcg_max_iter"])

        for it in range(max_iter):
            rn = self._dot(r, r).clamp_min(0.0).sqrt()
            newly_done = active_case & (rn <= threshold)
            active_case = active_case & ~newly_done
            if (
                it % self.pcg_host_check_interval == 0
                or it + 1 == max_iter
            ) and not bool(active_case.any().item()):
                break
            Ap = self._apply_A(A, p)
            pap = self._dot(p, Ap)
            pnorm = self._dot(p, p).clamp_min(0.0).sqrt()
            apnorm = self._dot(Ap, Ap).clamp_min(0.0).sqrt()
            bad = active_case & (
                ~torch.isfinite(pap)
                | (pap <= guard * pnorm * apnorm)
                | ~torch.isfinite(rz)
                | (rz <= 0.0)
            )
            failed |= bad
            active_case &= ~bad
            safe_pap = torch.where(active_case, pap, torch.ones_like(pap))
            alpha = torch.where(active_case, rz / safe_pap, torch.zeros_like(rz))
            mask3 = active_case.view(-1, 1, 1)
            x = torch.where(mask3 & A["active"], x + alpha.view(-1, 1, 1) * p, x)
            r = torch.where(mask3 & A["active"], r - alpha.view(-1, 1, 1) * Ap, r)
            z_new = torch.where(A["active"], r / A["diag"].clamp_min(1.0e-300), torch.zeros_like(r))
            rz2 = self._dot(r, z_new)
            bad2 = active_case & (~torch.isfinite(rz2) | (rz2 <= 0.0))
            failed |= bad2
            active_case &= ~bad2
            beta = torch.where(active_case, rz2 / rz.clamp_min(1.0e-300), torch.zeros_like(rz2))
            p = torch.where(
                active_case.view(-1, 1, 1) & A["active"],
                z_new + beta.view(-1, 1, 1) * p,
                p,
            )
            z = z_new
            rz = torch.where(active_case, rz2, rz)
            iterations += active_case.to(torch.int64)

        true_r = A["b"] - self._apply_A(A, x)
        true_r = torch.where(A["active"], true_r, torch.zeros_like(true_r))
        true_norm = self._dot(true_r, true_r).clamp_min(0.0).sqrt()
        converged = requested_case & ~failed & (true_norm <= threshold)
        relres = true_norm / scale
        return x, {
            "iterations": iterations,
            "relres": relres,
            "converged": converged,
            "failed": failed | (requested_case & ~converged),
            "method": "torch_cuda_fp64_batched_pcg_jacobi_surface_contact",
        }

    def _activity_gamma(self, salt: Any, cfg: Mapping[str, Any]) -> Any:
        torch = self.torch
        ionic = (salt / 1000.0).clamp_min(0.0)
        sq = torch.sqrt(ionic)
        lg = -float(cfg["activity_A"]) * sq / (1.0 + float(cfg["activity_B"]) * sq).clamp_min(1.0e-12)
        lg = lg + float(cfg["activity_linear"]) * ionic
        lim = abs(float(cfg["max_log10_gamma"]))
        return torch.pow(torch.tensor(10.0, device=self.device, dtype=self.dtype), lg.clamp(-lim, lim))

    def _eeq(self, channel: Mapping[str, float], activity: Any, T: Any, cfg: Mapping[str, Any]) -> tuple[Any, Any]:
        torch = self.torch
        if bool(cfg["nernst_enabled"]):
            ratio = (activity / max(channel["ref_activity"], float(cfg["activity_floor"]))).clamp_min(float(cfg["activity_floor"]))
            shift = float(cfg["R"]) * T / (channel["n"] * float(cfg["F"])) * channel["nernst_exp"] * torch.log(ratio)
            max_shift = abs(float(cfg["max_nernst_shift"]))
            shift = shift.clamp(-max_shift, max_shift)
        else:
            shift = torch.zeros_like(T)
        return channel["Eeq"] + shift, shift

    def _channel_eval(
        self,
        channel: Mapping[str, float],
        concentration_factor: Any,
        concentration: Any,
        diffusivity: Any,
        eta: Any,
        T: Any,
        kinetic: Any,
        cfg: Mapping[str, Any],
    ) -> tuple[Any, Any]:
        torch = self.torch
        eta = eta.clamp_min(0.0)
        kf = channel["alpha"] * channel["n"] * float(cfg["F"]) / (float(cfg["R"]) * T)
        kr = (1.0 - channel["alpha"]) * channel["n"] * float(cfg["F"]) / (float(cfg["R"]) * T)
        forward = (kf * eta).clamp(-float(cfg["exp_limit"]), float(cfg["exp_limit"]))
        reverse = (-kr * eta).clamp(-float(cfg["exp_limit"]), float(cfg["exp_limit"]))
        pref = channel["j0"] * kinetic
        ef = torch.exp(forward)
        er = torch.exp(reverse)
        if bool(cfg["full_bv"]):
            raw = pref * (concentration_factor * ef - channel["reverse"] * er)
            draw = pref * (concentration_factor * ef * kf + channel["reverse"] * er * kr)
        else:
            raw = pref * concentration_factor * ef
            draw = pref * concentration_factor * ef * kf
        draw = torch.where((raw > 0.0) & (eta > 0.0), draw, torch.zeros_like(draw))
        jk = raw.clamp_min(0.0)
        if bool(cfg["mass_transfer_saturation"]):
            if bool(cfg["derive_km"]):
                km = diffusivity / max(float(cfg["diffusion_layer"]), 1.0e-30)
            else:
                km = torch.full_like(diffusivity, channel["km"])
            jl = channel["n"] * float(cfg["F"]) * km * concentration.clamp_min(0.0)
            usable = jl > float(cfg["current_density_floor"])
            inv = 1.0 / (1.0 + jk / jl.clamp_min(float(cfg["current_density_floor"])))
            j = torch.where(usable, jk * inv, torch.zeros_like(jk))
            dj = torch.where(usable, draw * inv * inv, torch.zeros_like(draw))
        else:
            j, dj = jk, draw
        finite = torch.isfinite(j) & torch.isfinite(dj)
        return torch.where(finite, j, torch.zeros_like(j)), torch.where(finite, dj, torch.zeros_like(dj))

    def _cell_context(self, s: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict[str, Any]:
        torch = self.torch
        T = s["T"].clamp_min(1.0)
        water_factor = (s["water"] / max(float(cfg["water0"]), 1.0e-30)).clamp(0.0, 1.0)
        salt_conc = torch.minimum(s["cp"], s["cm"])
        salt_factor = (salt_conc / max(float(cfg["lp0"]), 1.0e-30)).clamp(0.0, 1.0)
        threshold = float(cfg["fast_ion_liquid_threshold"])
        kinetic = 1.0 + float(cfg["liquid_kinetics_gain"]) * (
            (s["liquid"] - threshold) / max(1.0 - threshold, 1.0e-30)
        ).clamp(0.0, 1.0)
        active = ((1.0 - s["pass"]) * (1.0 - s["coverage"])).clamp(float(cfg["min_active_area"]), 1.0)
        wa = water_factor.clamp_min(float(cfg["activity_floor"]))
        salt_activity = (salt_factor * self._activity_gamma(salt_conc, cfg)).clamp_min(float(cfg["activity_floor"]))
        channels = {name: self._channel(cfg, name) for name in ("water_a", "water_c", "lp_a", "lp_c")}
        eeq_wa, shift_wa = self._eeq(channels["water_a"], wa, T, cfg)
        eeq_wc, shift_wc = self._eeq(channels["water_c"], wa, T, cfg)
        eeq_la, shift_la = self._eeq(channels["lp_a"], salt_activity, T, cfg)
        eeq_lc, shift_lc = self._eeq(channels["lp_c"], salt_activity, T, cfg)
        shiftmax = torch.maximum(torch.maximum(shift_wa.abs(), shift_wc.abs()), torch.maximum(shift_la.abs(), shift_lc.abs()))
        return {
            "T": T, "water_factor": water_factor, "salt_factor": salt_factor,
            "salt_conc": salt_conc, "kinetic": kinetic, "active": active,
            "eeq_wa": eeq_wa, "eeq_wc": eeq_wc, "eeq_la": eeq_la,
            "eeq_lc": eeq_lc, "shiftmax": shiftmax, "channels": channels,
        }

    def _solve_faces(
        self,
        s: Mapping[str, Any],
        t: Mapping[str, Any],
        cfg: Mapping[str, Any],
        voltage: Any,
        contact_mask: Any,
        anode_side: bool,
        ctx: Mapping[str, Any],
    ) -> dict[str, Any]:
        torch = self.torch
        cathode_v = float(cfg["cathode_voltage"])
        if anode_side:
            maxdrop = (voltage - s["phi"]).clamp_min(0.0)
            eq_w, eq_l = ctx["eeq_wa"], ctx["eeq_la"]
            ch_w, ch_l = ctx["channels"]["water_a"], ctx["channels"]["lp_a"]
        else:
            maxdrop = (s["phi"] - cathode_v).clamp_min(0.0)
            eq_w, eq_l = ctx["eeq_wc"], ctx["eeq_lc"]
            ch_w, ch_l = ctx["channels"]["water_c"], ctx["channels"]["lp_c"]
        G = torch.maximum(
            t["sig"] / max(float(cfg["contact_normal_length"]), 1.0e-30),
            torch.full_like(t["sig"], float(cfg["sigma_min"]) / max(float(cfg["contact_normal_length"]), 1.0e-30)),
        )
        lo = torch.zeros_like(maxdrop)
        hi = maxdrop.clone()
        gap = 0.5 * maxdrop
        solved = ~contact_mask | (maxdrop <= 0.0)

        def evaluate(g: Any) -> tuple[Any, ...]:
            eta_w = (g - eq_w).clamp_min(0.0)
            eta_l = (g - eq_l).clamp_min(0.0)
            jw, djw = self._channel_eval(ch_w, ctx["water_factor"], s["water"], t["Dw"], eta_w, ctx["T"], torch.ones_like(g), cfg)
            jl, djl = self._channel_eval(ch_l, ctx["salt_factor"], ctx["salt_conc"], torch.minimum(t["Dp"], t["Dm"]), eta_l, ctx["T"], ctx["kinetic"], cfg)
            j = ctx["active"] * (jw + jl)
            dj = ctx["active"] * (djw + djl)
            jb = G * (maxdrop - g).clamp_min(0.0)
            residual = j - jb
            scale = torch.maximum(torch.maximum(j.abs(), jb.abs()), torch.ones_like(j))
            return j, dj, residual, scale, eta_w, eta_l, jw, jl, djw, djl

        min_iter = int(cfg["local_min_iter"])
        max_iter = int(cfg["local_max_iter"])
        abs_tol = float(cfg["local_abs_tol"])
        rel_tol = float(cfg["local_rel_tol"])
        for it in range(1, max_iter + 1):
            _, _, residual, scale, *_ = evaluate(gap)
            tolerance = torch.minimum(torch.full_like(scale, abs_tol), rel_tol * scale)
            if it >= min_iter:
                solved = solved | (contact_mask & (residual.abs() <= tolerance))
            active = contact_mask & ~solved
            if (
                it % self.local_face_host_check_interval == 0
                or it == max_iter
            ) and not bool(active.any().item()):
                break
            lo = torch.where(active & (residual < 0.0), gap, lo)
            hi = torch.where(active & (residual >= 0.0), gap, hi)
            gap = torch.where(active, 0.5 * (lo + hi), gap)

        for _ in range(int(cfg["local_newton"])):
            _, dj, residual, scale, *_ = evaluate(gap)
            active = contact_mask & ~solved
            derivative = dj + G
            valid = active & torch.isfinite(derivative) & (derivative > 1.0e-12)
            candidate = (gap - residual / derivative.clamp_min(1.0e-300)).clamp_min(0.0)
            candidate = torch.minimum(torch.maximum(candidate, lo), hi)
            gap = torch.where(valid, candidate, gap)
            _, _, residual2, scale2, *_ = evaluate(gap)
            lo = torch.where(valid & (residual2 < 0.0), gap, lo)
            hi = torch.where(valid & (residual2 >= 0.0), gap, hi)
            tolerance = torch.minimum(torch.full_like(scale2, abs_tol), rel_tol * scale2)
            solved = solved | (contact_mask & (residual2.abs() <= tolerance))

        _, _, _, _, eta_w, eta_l, jw0, jl0, djw0, djl0 = evaluate(gap)
        jw = ctx["active"] * jw0
        jl = ctx["active"] * jl0
        ks = ctx["active"] * (djw0 + djl0)
        big = torch.maximum(ks, G)
        small = torch.minimum(ks, G)
        sensitivity = torch.where(big > 1.0e-12, small / (1.0 + small / big.clamp_min(1.0e-300)), torch.zeros_like(big))
        phi_face = voltage - gap if anode_side else cathode_v + gap
        zero = torch.zeros_like(jw)
        return {
            "jw": torch.where(contact_mask, jw, zero),
            "jl": torch.where(contact_mask, jl, zero),
            "eta_w": torch.where(contact_mask, eta_w, zero),
            "eta_l": torch.where(contact_mask, eta_l, zero),
            "phi_face": torch.where(contact_mask, phi_face, zero),
            "sensitivity": torch.where(contact_mask, sensitivity, zero),
            "ok": solved,
        }

    def _eval_boundary(
        self,
        s: Mapping[str, Any],
        t: Mapping[str, Any],
        anode: Any,
        cathode: Any,
        cfg: Mapping[str, Any],
        voltage: Any,
        h: float,
    ) -> dict[str, Any]:
        torch = self.torch
        ctx = self._cell_context(s, cfg)
        a = self._solve_faces(s, t, cfg, voltage, anode, True, ctx)
        c = self._solve_faces(s, t, cfg, voltage, cathode, False, ctx)
        contact_area = h * h
        ja = a["jw"] + a["jl"]
        jc = c["jw"] + c["jl"]
        rawA = ja.sum(dim=(-2, -1)) * contact_area
        rawC = jc.sum(dim=(-2, -1)) * contact_area
        sensitivity = (a["sensitivity"] + c["sensitivity"]).sum(dim=(-2, -1)) * contact_area
        return {
            "anodeCurrent": ja,
            "cathodeCurrent": jc,
            "anodeSlope": a["sensitivity"].clamp_min(0.0),
            "cathodeSlope": c["sensitivity"].clamp_min(0.0),
            "jwa": a["jw"], "jla": a["jl"], "jwc": c["jw"], "jlc": c["jl"],
            "eta_wa": a["eta_w"], "eta_la": a["eta_l"],
            "eta_wc": c["eta_w"], "eta_lc": c["eta_l"],
            "rawA": rawA, "rawC": rawC, "sensitivity": sensitivity,
            "anodeZone": anode, "cathodeZone": cathode,
            "maxNernstShift": ctx["shiftmax"].amax(dim=(-2, -1)),
            "local_ok": (a["ok"] & c["ok"]),
            "channels": ctx["channels"],
        }

    def _finish_reaction(self, b: Mapping[str, Any], cfg: Mapping[str, Any], reaction_layer: float) -> dict[str, Any]:
        torch = self.torch
        channels = b["channels"]
        def heat(j: Any, eta: Any, ch: Mapping[str, float]) -> Any:
            mol = j / (ch["n"] * float(cfg["F"]))
            q = mol * ch["dH"] / reaction_layer
            if bool(cfg["activation_heat"]):
                q = q + float(cfg["activation_heat_fraction"]) * j * eta / reaction_layer
            return q.clamp_min(0.0)
        q = (
            heat(b["jwa"], b["eta_wa"], channels["water_a"])
            + heat(b["jwc"], b["eta_wc"], channels["water_c"])
            + heat(b["jla"], b["eta_la"], channels["lp_a"])
            + heat(b["jlc"], b["eta_lc"], channels["lp_c"])
        )
        water_sink = (
            channels["water_a"]["water_stoich"] * b["jwa"] / (channels["water_a"]["n"] * float(cfg["F"]) * reaction_layer)
            + channels["water_c"]["water_stoich"] * b["jwc"] / (channels["water_c"]["n"] * float(cfg["F"]) * reaction_layer)
        )
        salt_sink = (
            channels["lp_a"]["salt_stoich"] * b["jla"] / (channels["lp_a"]["n"] * float(cfg["F"]) * reaction_layer)
            + channels["lp_c"]["salt_stoich"] * b["jlc"] / (channels["lp_c"]["n"] * float(cfg["F"]) * reaction_layer)
        )
        gas_source = (
            channels["water_a"]["gas_yield"] * b["jwa"] / (channels["water_a"]["n"] * float(cfg["F"]) * reaction_layer)
            + channels["water_c"]["gas_yield"] * b["jwc"] / (channels["water_c"]["n"] * float(cfg["F"]) * reaction_layer)
            + channels["lp_a"]["gas_yield"] * b["jla"] / (channels["lp_a"]["n"] * float(cfg["F"]) * reaction_layer)
            + channels["lp_c"]["gas_yield"] * b["jlc"] / (channels["lp_c"]["n"] * float(cfg["F"]) * reaction_layer)
        )
        totalW = (b["jwa"] + b["jwc"]).sum(dim=(-2, -1))
        totalL = (b["jla"] + b["jlc"]).sum(dim=(-2, -1))
        total = totalW + totalL
        totalI = 0.5 * (b["rawA"] + b["rawC"])
        mismatch = (b["rawA"] - b["rawC"]).abs() / torch.maximum(
            torch.maximum(b["rawA"].abs(), b["rawC"].abs()),
            torch.full_like(totalI, float(cfg["current_floor"])),
        )
        out = dict(b)
        out.update({
            "q": q, "waterSink": water_sink, "saltSink": salt_sink,
            "gasSource": gas_source, "totalI": totalI, "mismatch": mismatch,
            "waterFrac": totalW / total.clamp_min(float(cfg["current_density_floor"])),
            "lpFrac": totalL / total.clamp_min(float(cfg["current_density_floor"])),
        })
        return out

    def _solve_electrochem(
        self,
        s: dict[str, Any],
        t: Mapping[str, Any],
        anode: Any,
        cathode: Any,
        cfg: Mapping[str, Any],
        voltage: Any,
        h: float,
        case_mask: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], Any]:
        torch = self.torch
        batch = s["phi"].shape[0]
        boundary = self._eval_boundary(s, t, anode, cathode, cfg, voltage, h)
        previous_current = 0.5 * (boundary["rawA"] + boundary["rawC"])
        converged = ~case_mask
        failed = torch.zeros(batch, device=self.device, dtype=torch.bool)
        dphi = torch.full((batch,), float("inf"), device=self.device, dtype=self.dtype)
        dI = dphi.clone()
        used = torch.zeros(batch, device=self.device, dtype=torch.int64)
        total_linear_iterations = torch.zeros(batch, device=self.device, dtype=torch.int64)
        total_linear_solves = torch.zeros(batch, device=self.device, dtype=torch.int64)
        last_diag: dict[str, Any] | None = None

        for iteration in range(1, int(cfg["robin_max_iter"]) + 1):
            active = case_mask & ~converged & ~failed
            if (
                iteration % self.robin_host_check_interval == 1
                or iteration == int(cfg["robin_max_iter"])
            ) and not bool(active.any().item()):
                break
            A = self._build_system(s, t, cfg, h, boundary)
            proposed, diag = self._solve_pcg(
                A, s["phi"].clone(), cfg, float(cfg["pcg_rtol_coupled"]), active
            )
            if last_diag is None:
                last_diag = {
                    key: (value.clone() if hasattr(value, "clone") else value)
                    for key, value in diag.items()
                }
            else:
                for key, value in diag.items():
                    if hasattr(value, "shape") and value.shape[:1] == (batch,):
                        last_diag[key] = torch.where(active, value, last_diag[key])
                    elif key == "method":
                        last_diag[key] = value
            failed_now = active & ~diag["converged"]
            failed |= failed_now
            successful = active & diag["converged"]
            total_linear_iterations += torch.where(successful, diag["iterations"], torch.zeros_like(diag["iterations"]))
            total_linear_solves += successful.to(torch.int64)
            previous_phi = s["phi"].clone()
            s["phi"] = torch.where(
                successful.view(-1, 1, 1),
                s["phi"] + float(cfg["robin_relax"]) * (proposed - s["phi"]),
                s["phi"],
            )
            s["phi"] = self._neumann_edges(s["phi"])
            next_boundary = self._eval_boundary(s, t, anode, cathode, cfg, voltage, h)
            # Preserve the previous boundary for failed/already-completed cases.
            for key, value in list(next_boundary.items()):
                if hasattr(value, "shape") and value.shape[:1] == (batch,):
                    mask = successful
                    if value.ndim > 1:
                        mask = mask.view(-1, *([1] * (value.ndim - 1)))
                    next_boundary[key] = torch.where(mask, value, boundary[key])
            boundary = next_boundary
            dphi_now = (s["phi"] - previous_phi).abs().amax(dim=(-2, -1))
            current = 0.5 * (boundary["rawA"] + boundary["rawC"])
            dI_now = (current - previous_current).abs() / torch.maximum(
                torch.maximum(current.abs(), previous_current.abs()),
                torch.full_like(current, 1.0e-12),
            )
            mismatch_abs = (boundary["rawA"] - boundary["rawC"]).abs()
            threshold = torch.maximum(
                float(cfg["robin_balance_tol"]) * torch.maximum(boundary["rawA"].abs(), boundary["rawC"].abs()),
                torch.full_like(current, float(cfg["robin_balance_abs"])),
            )
            conv_now = successful & (iteration >= int(cfg["robin_min_iter"])) & (
                dphi_now <= float(cfg["robin_phi_tol"])
            ) & (dI_now <= float(cfg["robin_current_tol"])) & (mismatch_abs <= threshold)
            converged |= conv_now
            dphi = torch.where(successful, dphi_now, dphi)
            dI = torch.where(successful, dI_now, dI)
            previous_current = torch.where(successful, current, previous_current)
            used = torch.where(successful, torch.full_like(used, iteration), used)

        if last_diag is None:
            last_diag = {
                "iterations": torch.zeros(batch, device=self.device, dtype=torch.int64),
                "relres": torch.full((batch,), float("inf"), device=self.device, dtype=self.dtype),
                "converged": ~case_mask,
                "method": "torch_cuda_fp64_batched_pcg_jacobi_surface_contact",
            }
        valid = case_mask & converged & ~failed
        scale = torch.maximum(boundary["rawA"].abs(), boundary["rawC"].abs())
        threshold = torch.maximum(
            float(cfg["robin_balance_tol"]) * scale,
            torch.full_like(scale, float(cfg["robin_balance_abs"])),
        )
        reaction = self._finish_reaction(
            boundary, cfg, max(float(cfg["diffusion_layer"]), float(cfg["reaction_layer_min"]), h)
        )
        reaction.update({
            "outerIters": used,
            "gaugeIters": torch.zeros_like(used),
            "linearIterations": total_linear_iterations,
            "linearSolves": total_linear_solves,
            "gaugeOffset": torch.zeros(batch, device=self.device, dtype=self.dtype),
            "converged": valid,
            "balanceCombined": (boundary["rawA"] - boundary["rawC"]).abs() / threshold.clamp_min(1.0e-300),
            "dphi": dphi, "dI": dI,
        })
        return reaction, last_diag, valid

    def _gradient(self, field: Any, h: float) -> tuple[Any, Any]:
        torch = self.torch
        gx = torch.zeros_like(field)
        gy = torch.zeros_like(field)
        gx[:, :, 1:-1] = (field[:, :, 2:] - field[:, :, :-2]) / (2.0 * h)
        gx[:, :, 0] = (field[:, :, 1] - field[:, :, 0]) / h
        gx[:, :, -1] = (field[:, :, -1] - field[:, :, -2]) / h
        gy[:, 1:-1, :] = (field[:, 2:, :] - field[:, :-2, :]) / (2.0 * h)
        gy[:, 0, :] = (field[:, 1, :] - field[:, 0, :]) / h
        gy[:, -1, :] = (field[:, -1, :] - field[:, -2, :]) / h
        return gx, gy

    def _compute_current(self, s: Mapping[str, Any], t: Mapping[str, Any], cfg: Mapping[str, Any], h: float) -> dict[str, Any]:
        torch = self.torch
        px, py = self._gradient(s["phi"], h)
        cpx, cpy = self._gradient(s["cp"], h)
        cmx, cmy = self._gradient(s["cm"], h)
        Ex, Ey = -px, -py
        if bool(cfg["diffusion_potential"]):
            Jdx = -float(cfg["F"]) * (
                float(cfg["zplus"]) * t["Dp"] * cpx + float(cfg["zminus"]) * t["Dm"] * cmx
            )
            Jdy = -float(cfg["F"]) * (
                float(cfg["zplus"]) * t["Dp"] * cpy + float(cfg["zminus"]) * t["Dm"] * cmy
            )
        else:
            Jdx = torch.zeros_like(Ex)
            Jdy = torch.zeros_like(Ey)
        Jox, Joy = t["sig"] * Ex, t["sig"] * Ey
        Jx, Jy = Jox + Jdx, Joy + Jdy
        Jmag = torch.hypot(Jx, Jy)
        qj = (Jox * Ex + Joy * Ey).clamp_min(0.0)
        Em = torch.hypot(Ex, Ey)
        ii, ie = t["sigi"] * Em, t["sige"] * Em
        ionic = ii / (ii + ie).clamp_min(float(cfg["current_density_floor"]))
        return {"Ex": Ex, "Ey": Ey, "Jx": Jx, "Jy": Jy, "Jmag": Jmag, "qj": qj, "ionicFrac": ionic}

    def _laplacian(self, field: Any, h: float) -> Any:
        padded = self.torch.nn.functional.pad(field.unsqueeze(1), (1, 1, 1, 1), mode="replicate").squeeze(1)
        return (
            padded[:, 1:-1, 2:] + padded[:, 1:-1, :-2]
            + padded[:, 2:, 1:-1] + padded[:, :-2, 1:-1]
            - 4.0 * field
        ) / (h * h)

    def _flux_div_charged(self, conc: Any, D: Any, s: Mapping[str, Any], cfg: Mapping[str, Any], h: float, z: float) -> Any:
        torch = self.torch
        batch, n, _ = conc.shape
        fx = torch.zeros((batch, n, n + 1), device=self.device, dtype=self.dtype)
        fy = torch.zeros((batch, n + 1, n), device=self.device, dtype=self.dtype)
        df = self._harmonic(D[:, :, :-1], D[:, :, 1:])
        cf = 0.5 * (conc[:, :, :-1] + conc[:, :, 1:])
        Tf = (0.5 * (s["T"][:, :, :-1] + s["T"][:, :, 1:])).clamp_min(1.0)
        fx[:, :, 1:n] = -df * (conc[:, :, 1:] - conc[:, :, :-1]) / h - z * df * float(cfg["F"]) / (float(cfg["R"]) * Tf) * cf * (s["phi"][:, :, 1:] - s["phi"][:, :, :-1]) / h
        df = self._harmonic(D[:, :-1, :], D[:, 1:, :])
        cf = 0.5 * (conc[:, :-1, :] + conc[:, 1:, :])
        Tf = (0.5 * (s["T"][:, :-1, :] + s["T"][:, 1:, :])).clamp_min(1.0)
        fy[:, 1:n, :] = -df * (conc[:, 1:, :] - conc[:, :-1, :]) / h - z * df * float(cfg["F"]) / (float(cfg["R"]) * Tf) * cf * (s["phi"][:, 1:, :] - s["phi"][:, :-1, :]) / h
        return (fx[:, :, 1:] - fx[:, :, :-1]) / h + (fy[:, 1:, :] - fy[:, :-1, :]) / h

    def _flux_div_neutral(self, conc: Any, D: Any, h: float) -> Any:
        torch = self.torch
        batch, n, _ = conc.shape
        fx = torch.zeros((batch, n, n + 1), device=self.device, dtype=self.dtype)
        fy = torch.zeros((batch, n + 1, n), device=self.device, dtype=self.dtype)
        df = self._harmonic(D[:, :, :-1], D[:, :, 1:])
        fx[:, :, 1:n] = -df * (conc[:, :, 1:] - conc[:, :, :-1]) / h
        df = self._harmonic(D[:, :-1, :], D[:, 1:, :])
        fy[:, 1:n, :] = -df * (conc[:, 1:, :] - conc[:, :-1, :]) / h
        return (fx[:, :, 1:] - fx[:, :, :-1]) / h + (fy[:, 1:, :] - fy[:, :-1, :]) / h

    def _limited_update(self, field: Any, rate: Any, dt: float, ref: float, cfg: Mapping[str, Any]) -> tuple[Any, Any]:
        torch = self.torch
        raw = dt * rate
        lower = ref * float(cfg["concentration_min_fraction"])
        upper = ref * float(cfg["concentration_max_multiple"])
        limit = float(cfg["max_relative_concentration_change"]) * torch.maximum(field, torch.full_like(field, lower))
        limited_delta = torch.minimum(torch.maximum(raw, -limit), limit)
        raw_updated = field + limited_delta
        updated = raw_updated.clamp(lower, upper)
        tolerance = 1.0e-12 + 1.0e-7 * torch.maximum(
            torch.maximum(raw.abs(), limited_delta.abs()),
            torch.maximum(field.abs(), torch.full_like(field, lower)),
        )
        limited = (limited_delta - raw).abs() > tolerance
        limited |= (updated - raw_updated).abs() > (1.0e-12 + 1.0e-7 * torch.maximum(updated.abs(), torch.maximum(raw_updated.abs(), torch.full_like(updated, lower))))
        return updated, limited

    def _update_species(
        self,
        s: dict[str, Any],
        t: dict[str, Any],
        reaction: Mapping[str, Any],
        chem: Any,
        cfg: Mapping[str, Any],
        h: float,
        dt: float,
        alive: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], Any]:
        torch = self.torch
        substeps = max(1, int(cfg["species_substeps"]))
        limiter = torch.zeros(s["T"].shape[0], device=self.device, dtype=self.dtype)
        for sub in range(substeps):
            if sub > 0:
                t = self._transport_fields(s, cfg)
            if bool(cfg["full_np"]):
                dc = self._flux_div_charged(s["cp"], t["Dp"], s, cfg, h, float(cfg["zplus"]))
                dm = self._flux_div_charged(s["cm"], t["Dm"], s, cfg, h, float(cfg["zminus"]))
                dw = self._flux_div_neutral(s["water"], t["Dw"], h)
            else:
                dc = torch.zeros_like(s["cp"])
                dm = torch.zeros_like(s["cm"])
                dw = torch.zeros_like(s["water"])
            cs = chem * float(cfg["lp0"]) * float(cfg["oxidizer_consume"])
            cp_new, a = self._limited_update(s["cp"], -dc - reaction["saltSink"] - cs, dt / substeps, float(cfg["cation0"]), cfg)
            cm_new, b = self._limited_update(s["cm"], -dm - reaction["saltSink"] - cs, dt / substeps, float(cfg["anion0"]), cfg)
            water_new, d = self._limited_update(s["water"], -dw - reaction["waterSink"], dt / substeps, float(cfg["water0"]), cfg)
            mask = alive.view(-1, 1, 1)
            cp_new = torch.where(mask, cp_new, s["cp"])
            cm_new = torch.where(mask, cm_new, s["cm"])
            water_new = torch.where(mask, water_new, s["water"])
            neutral = 0.5 * (cp_new + cm_new)
            relax = float(cfg["electroneutral_relax"])
            s["cp"] = (1.0 - relax) * cp_new + relax * neutral
            s["cm"] = (1.0 - relax) * cm_new + relax * neutral
            s["water"] = water_new
            limiter += ((a | b | d).to(self.dtype).mean(dim=(-2, -1)))
        return s, t, limiter / substeps

    def _update_blocking(self, s: dict[str, Any], reaction: Mapping[str, Any], cfg: Mapping[str, Any], dt: float, alive: Any) -> None:
        torch = self.torch
        zone = reaction["anodeZone"] | reaction["cathodeZone"]
        jt = reaction["jwa"] + reaction["jla"] + reaction["jwc"] + reaction["jlc"]
        mask = alive.view(-1, 1, 1) & zone
        if bool(cfg["passivation_enabled"]):
            gate = self._sigmoid((s["T"] - float(cfg["pass_T0"])) / max(float(cfg["pass_Tw"]), 1.0e-12))
            form = float(cfg["pass_form"]) * (jt / max(float(cfg["pass_jref"]), 1.0e-30)).clamp_min(0.0).pow(float(cfg["pass_jexp"])) * gate * (1.0 - s["pass"])
            candidate = (s["pass"] + dt * (form - float(cfg["pass_remove"]) * s["pass"])).clamp(0.0, 1.0)
            s["pass"] = torch.where(mask, candidate, torch.where(zone, s["pass"], torch.zeros_like(s["pass"])))
        else:
            s["pass"] = torch.where(zone, s["pass"], torch.zeros_like(s["pass"]))
        if bool(cfg["gas_coverage_enabled"]):
            source = (reaction["gasSource"] / max(float(cfg["gas_source_ref"]), 1.0e-30)).clamp_min(0.0)
            form = float(cfg["gas_cov_form"]) * source * (1.0 - s["coverage"])
            candidate = (s["coverage"] + dt * (form - float(cfg["gas_cov_detach"]) * s["coverage"])).clamp(0.0, 1.0)
            s["coverage"] = torch.where(mask, candidate, torch.where(zone, s["coverage"], torch.zeros_like(s["coverage"])))
        else:
            s["coverage"] = torch.where(zone, s["coverage"], torch.zeros_like(s["coverage"]))

    def _percentile99(self, values: Any) -> Any:
        # torch.quantile stays on CUDA and is only called when electrical fields
        # are refreshed, not at every explicit timestep.
        return self.torch.quantile(values.flatten(1), 0.99, dim=1)

    def _simulate_batch(self, cases: Sequence[CudaBatchCase]) -> tuple[list[dict[str, Any]], list[str | None]]:
        torch = self.torch
        cfg = self._validate_common_config(cases)
        batch = len(cases)
        n = int(cfg["n"])
        anode_np = np.stack([_canonical_numpy_bool_mask(c.anode) for c in cases], axis=0)
        cathode_np = np.stack([_canonical_numpy_bool_mask(c.cathode) for c in cases], axis=0)
        if anode_np.shape != (batch, n, n) or cathode_np.shape != anode_np.shape:
            raise CudaPhysicsError(f"CUDA mask shape mismatch: {anode_np.shape}, expected {(batch, n, n)}")
        if np.any(anode_np & cathode_np):
            raise CudaPhysicsError("CUDA batch contains polarity overlap")
        anode = torch.as_tensor(anode_np, device=self.device, dtype=torch.bool)
        cathode = torch.as_tensor(cathode_np, device=self.device, dtype=torch.bool)
        voltage = self._to_scalar_tensor([float(c.config["voltage"]) for c in cases])
        stop_on_ignition = torch.as_tensor(
            [bool(c.config.get("stop_on_ignition", False)) for c in cases],
            device=self.device,
            dtype=torch.bool,
        )
        s = self._init_state(anode, cathode, cfg, voltage)
        t = self._transport_fields(s, cfg)
        h = float(cfg["domain"]) / max(n - 1, 1)
        alphaT = float(cfg["k"]) / (float(cfg["density"]) * float(cfg["cp"]))
        steps = int(round(float(cfg["end_time"]) / float(cfg["dt"])))
        solve_every = max(1, int(round(float(cfg["electrical_interval"]) / float(cfg["dt"]))))
        case_mask = torch.ones(batch, device=self.device, dtype=torch.bool)
        errors: list[str | None] = [None] * batch

        reaction, last_solver, valid = self._solve_electrochem(s, t, anode, cathode, cfg, voltage, h, case_mask)
        failed_initial = case_mask & ~valid
        if bool(failed_initial.any().item()):
            idxs = torch.nonzero(failed_initial, as_tuple=False).flatten().tolist()
            for i in idxs:
                errors[i] = (
                    "CUDA FP64 initial nonlinear electrochemical solve failed; "
                    f"linear_relres={float(last_solver['relres'][i].item()):.8g}, "
                    f"dphi={float(reaction['dphi'][i].item()):.8g}, "
                    f"dI={float(reaction['dI'][i].item()):.8g}"
                )
        alive = valid.clone()
        electrical = self._compute_current(s, t, cfg, h)
        meanJ = electrical["Jmag"].mean(dim=(-2, -1))
        congestion_now = self._percentile99(electrical["Jmag"]) / meanJ.clamp_min(float(cfg["current_density_floor"]))

        ignition = torch.full((batch,), float("nan"), device=self.device, dtype=self.dtype)
        ignited = torch.zeros(batch, device=self.device, dtype=torch.bool)
        energy = torch.zeros(batch, device=self.device, dtype=self.dtype)
        prev_power = torch.zeros_like(energy)
        have_prev = torch.zeros(batch, device=self.device, dtype=torch.bool)
        eval_undecomp = torch.ones(batch, device=self.device, dtype=self.dtype)
        eval_energy = torch.zeros_like(energy)
        ignition_energy = torch.full_like(energy, float("nan"))
        peakT = torch.zeros_like(energy)
        peakI = torch.zeros_like(energy)
        peak_congestion = congestion_now.clone()
        max_species = torch.zeros_like(energy)
        max_temp = torch.zeros_like(energy)
        max_gas = torch.zeros_like(energy)
        max_chem = torch.zeros_like(energy)
        simulated_time = torch.zeros_like(energy)
        terminated = torch.zeros(batch, device=self.device, dtype=torch.bool)
        electrical_solves = torch.ones(batch, device=self.device, dtype=torch.int64)
        total_linear_iterations = reaction["linearIterations"].clone()
        total_linear_solves = reaction["linearSolves"].clone()
        total_robin = reaction["outerIters"].clone()
        total_gauge = reaction["gaugeIters"].clone()

        dt = float(cfg["dt"])
        for step in range(steps):
            if (
                step % self.time_host_check_interval == 0
                or step + 1 == steps
            ) and not bool(alive.any().item()):
                break
            time_s = (step + 1) * dt
            t = self._transport_fields(s, cfg)
            if step > 0 and step % solve_every == 0:
                attempted_step = alive.clone()
                new_reaction, new_solver, valid_step = self._solve_electrochem(
                    s, t, anode, cathode, cfg, voltage, h, attempted_step
                )
                failed_step = alive & ~valid_step
                if bool(failed_step.any().item()):
                    idxs = torch.nonzero(failed_step, as_tuple=False).flatten().tolist()
                    for i in idxs:
                        errors[i] = (
                            f"CUDA FP64 nonlinear electrochemical solve failed at t={time_s:.8g}s; "
                            f"linear_relres={float(new_solver['relres'][i].item()):.8g}"
                        )
                    alive &= ~failed_step
                # Merge tensor-valued reaction fields for still-valid cases.
                mask3 = valid_step.view(-1, 1, 1)
                for key, value in list(new_reaction.items()):
                    if hasattr(value, "shape") and value.shape[:1] == (batch,):
                        mask = valid_step if value.ndim == 1 else mask3
                        new_reaction[key] = torch.where(mask, value, reaction[key])
                reaction = new_reaction
                merged_solver: dict[str, Any] = {}
                for key, value in new_solver.items():
                    if hasattr(value, "shape") and value.shape[:1] == (batch,):
                        merged_solver[key] = torch.where(
                            attempted_step, value, last_solver[key]
                        )
                    else:
                        merged_solver[key] = value
                last_solver = merged_solver
                electrical_solves += valid_step.to(torch.int64)
                total_linear_iterations += torch.where(valid_step, reaction["linearIterations"], torch.zeros_like(reaction["linearIterations"]))
                total_linear_solves += torch.where(valid_step, reaction["linearSolves"], torch.zeros_like(reaction["linearSolves"]))
                total_robin += torch.where(valid_step, reaction["outerIters"], torch.zeros_like(reaction["outerIters"]))
                total_gauge += torch.where(valid_step, reaction["gaugeIters"], torch.zeros_like(reaction["gaugeIters"]))
                electrical = self._compute_current(s, t, cfg, h)
                meanJ = electrical["Jmag"].mean(dim=(-2, -1))
                congestion_now = self._percentile99(electrical["Jmag"]) / meanJ.clamp_min(float(cfg["current_density_floor"]))

            ox = torch.minimum(s["cp"], s["cm"]) / max(float(cfg["lp0"]), 1.0e-12)
            gate = self._sigmoid((s["T"] - float(cfg["chem_activation_T"])) / 20.0)
            if bool(cfg["chemical_enabled"]):
                chem_raw = (
                    float(cfg["chem_A"])
                    * torch.exp(-float(cfg["chem_Ea"]) / (float(cfg["R"]) * s["T"].clamp_min(1.0)))
                    * (1.0 - s["alpha"]).clamp_min(0.0).pow(float(cfg["chem_order"]))
                    * ox.clamp_min(0.0).pow(float(cfg["oxidizer_order"]))
                    * gate
                )
                chem = chem_raw.clamp(0.0, float(cfg["chem_max_rate"]))
            else:
                chem = torch.zeros_like(s["T"])
            s, t, limiter = self._update_species(s, t, reaction, chem, cfg, h, dt, alive)
            max_species = torch.maximum(max_species, torch.where(alive, limiter, torch.zeros_like(limiter)))
            self._update_blocking(s, reaction, cfg, dt, alive)

            leq = 1.0 / (1.0 + torch.exp(-(s["T"] - float(cfg["effective_softening_T"])) / float(cfg["transition_width"])))
            liquid_rate = (leq - s["liquid"]) / max(float(cfg["phase_relax_time"]), 1.0e-30)
            chemgas = float(cfg["density"]) * float(cfg["gas_yield_mass"]) * chem / max(float(cfg["gas_molar_mass"]), 1.0e-30)
            gasrate = float(cfg["gas_D"]) * self._laplacian(s["gas"], h) + reaction["gasSource"] + chemgas - float(cfg["gas_loss"]) * s["gas"]
            chemical_heat = float(cfg["density"]) * float(cfg["chem_heat"]) * chem
            convfac = (1.0 - float(cfg["gas_thermal_insulation"]) * s["coverage"]).clamp(0.05, 1.0)
            loss = (
                convfac * float(cfg["hconv"]) / float(cfg["layer"]) * (s["T"] - float(cfg["Tamb"]))
                + float(cfg["emissivity"]) * float(cfg["sigma_sb"]) / float(cfg["layer"]) * (s["T"].pow(4) - float(cfg["Tamb"]) ** 4)
            )
            latent = float(cfg["density"]) * float(cfg["latent_heat"]) * liquid_rate
            # Buffered explicit update: all stencil values come from the same
            # timestep state.  This is the mathematically intended explicit
            # discretisation and the only parallel-order-independent form.
            Trate = alphaT * self._laplacian(s["T"], h) + (
                electrical["qj"] + reaction["q"] + chemical_heat - loss - latent
            ) / (float(cfg["density"]) * float(cfg["cp"]))
            rawT = s["T"] + dt * Trate
            rawG = s["gas"] + dt * gasrate
            mask3 = alive.view(-1, 1, 1)
            tempcap = ((rawT < float(cfg["Tmin"])) | (rawT > float(cfg["Tmax"]))).to(self.dtype).mean(dim=(-2, -1))
            gascap = ((rawG < 0.0) | (rawG > float(cfg["gas_max"]))).to(self.dtype).mean(dim=(-2, -1))
            chemcap = (chem >= float(cfg["chem_max_rate"]) * (1.0 - 1.0e-12)).to(self.dtype).mean(dim=(-2, -1))
            s["T"] = torch.where(mask3, rawT.clamp(float(cfg["Tmin"]), float(cfg["Tmax"])), s["T"])
            s["liquid"] = torch.where(mask3, (s["liquid"] + dt * liquid_rate).clamp(0.0, 1.0), s["liquid"])
            s["alpha"] = torch.where(mask3, (s["alpha"] + dt * chem).clamp(0.0, 1.0), s["alpha"])
            s["gas"] = torch.where(mask3, rawG.clamp(0.0, float(cfg["gas_max"])), s["gas"])
            maxT_now = s["T"].amax(dim=(-2, -1))
            sumAlpha = s["alpha"].mean(dim=(-2, -1))
            ignition_cells = (
                (s["T"] >= float(cfg["ignition_T"]))
                & (s["alpha"] >= float(cfg["ignition_progress_guard"]))
                & (chem >= float(cfg["ignition_rate_guard"]))
            )
            ignited_this = alive & ~ignited & ignition_cells.any(dim=(-2, -1))
            ignition = torch.where(ignited_this, torch.full_like(ignition, time_s), ignition)
            ignited |= ignited_this

            power = reaction["totalI"] * (voltage.flatten() - float(cfg["cathode_voltage"]))
            first_energy = power * dt
            trapezoid = energy + 0.5 * dt * (power + prev_power)
            energy = torch.where(alive, torch.where(have_prev, trapezoid, first_energy), energy)
            have_prev |= alive
            prev_power = torch.where(alive, power, prev_power)
            ignition_energy = torch.where(ignited_this & ~torch.isfinite(ignition_energy), energy, ignition_energy)
            undecomp = 1.0 - sumAlpha
            at_eval = alive & (time_s >= float(cfg["eval_time"]) - 0.5 * dt)
            eval_undecomp = torch.where(at_eval, undecomp, eval_undecomp)
            eval_energy = torch.where(at_eval, energy, eval_energy)
            peakT = torch.maximum(peakT, torch.where(alive, maxT_now, torch.zeros_like(maxT_now)))
            peakI = torch.maximum(peakI, torch.where(alive, reaction["totalI"], torch.zeros_like(peakI)))
            peak_congestion = torch.maximum(peak_congestion, torch.where(alive, congestion_now, torch.zeros_like(congestion_now)))
            max_temp = torch.maximum(max_temp, torch.where(alive, tempcap, torch.zeros_like(tempcap)))
            max_gas = torch.maximum(max_gas, torch.where(alive, gascap, torch.zeros_like(gascap)))
            max_chem = torch.maximum(max_chem, torch.where(alive, chemcap, torch.zeros_like(chemcap)))
            simulated_time = torch.where(alive, torch.full_like(simulated_time, time_s), simulated_time)
            stop_now = alive & stop_on_ignition & ignited
            terminated |= stop_now
            alive &= ~stop_now

        # CUDA work completes before host scalar extraction.
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        finalR = (voltage.flatten() - float(cfg["cathode_voltage"])) / reaction["totalI"].clamp_min(float(cfg["current_floor"]))
        outputs: list[dict[str, Any]] = []
        anode_fraction = anode.to(self.dtype).mean(dim=(-2, -1))
        cathode_fraction = cathode.to(self.dtype).mean(dim=(-2, -1))
        for i, case in enumerate(cases):
            if errors[i] is not None:
                outputs.append({})
                continue
            outputs.append({
                "appliedVoltage_V": float(case.config["voltage"]),
                "cathodeVoltage_V": float(case.config["cathode_voltage"]),
                "ignitionDelay_s": float(ignition[i].item()) if bool(ignited[i].item()) else None,
                "condensedPhaseIgnitionDelay_s": float(ignition[i].item()) if bool(ignited[i].item()) else None,
                "ignitionSucceeded": bool(ignited[i].item()),
                "areaAveragedUndecomposedFractionAt2s": float(eval_undecomp[i].item()),
                "areaWeightedUndecomposedFractionAt2s": float(eval_undecomp[i].item()),
                "inputElectricalEnergyToIgnition_J": float(ignition_energy[i].item()) if math.isfinite(float(ignition_energy[i].item())) else None,
                "inputElectricalEnergyAt2s_J": float(eval_energy[i].item()),
                "inputElectricalEnergyAtEvaluationTime_J": float(eval_energy[i].item()),
                "inputElectricalEnergy_J": float(eval_energy[i].item()),
                "peakCurrentCongestion": float(peak_congestion[i].item()),
                "peakCurrentCongestionTo2s": float(peak_congestion[i].item()),
                "peakMaximumTemperature_K": float(peakT[i].item()),
                "peakCurrent_A": float(peakI[i].item()),
                "finalEffectiveResistance_ohm": float(finalR[i].item()),
                "finalElectricalRelativeResidual": float(last_solver["relres"][i].item()),
                "finalElectricalConverged": bool(reaction["converged"][i].item()),
                "converged": bool(reaction["converged"][i].item()),
                "finalElectricalSolverMethod": str(last_solver["method"]),
                "finalElectricalIterations": int(last_solver["iterations"][i].item()),
                "electricalSolveCount": int(electrical_solves[i].item()),
                "totalLinearSolveCount": int(total_linear_solves[i].item()),
                "totalLinearIterations": int(total_linear_iterations[i].item()),
                "totalNonlinearRobinOuterIterations": int(total_robin[i].item()),
                "totalGaugeIterations": int(total_gauge[i].item()),
                "finalAnodeCathodeCurrentMismatch": float(reaction["mismatch"][i].item()),
                "finalNonlinearRobinAnodeCurrent_A": float(reaction["rawA"][i].item()),
                "finalNonlinearRobinCathodeCurrent_A": float(reaction["rawC"][i].item()),
                "finalNonlinearRobinGaugeOffset_V": 0.0,
                "finalNonlinearRobinCurrentBalanceCombinedResidual": float(reaction["balanceCombined"][i].item()),
                "maximumSpeciesLimiterFraction": float(max_species[i].item()),
                "maximumTemperatureCapFraction": float(max_temp[i].item()),
                "maximumGasCapFraction": float(max_gas[i].item()),
                "maximumChemicalRateCapFraction": float(max_chem[i].item()),
                "simulatedTime_s": float(simulated_time[i].item()),
                "stopOnIgnitionRequested": bool(stop_on_ignition[i].item()),
                "terminatedAtIgnition": bool(terminated[i].item()),
                "propellantDomainAreaFraction": 1.0,
                "anodeContactAreaFraction": float(anode_fraction[i].item()),
                "cathodeContactAreaFraction": float(cathode_fraction[i].item()),
                "totalContactAreaFraction": float((anode_fraction[i] + cathode_fraction[i]).item()),
                "contactNormalConductionLength_m": float(cfg["contact_normal_length"]),
                "propellantCellCount": int(n * n),
                "surfaceContactModel": True,
                "electrodeMasksRemovePropellant": False,
                "hiddenBusConnectionAssumed": bool(cfg["hidden_bus_assumed"]),
                "physicsDevice": str(self.device),
                "physicsDtype": "float64",
                "backend": "torch_cuda_fp64_a100",
                "solverRevision": "v7_9_5_torch_cuda_fp64_surface_contact_hybrid",
                "empiricalSurfaceReactionProgressUsed": False,
            })
        return outputs, errors

    def run_batch(self, cases: Sequence[CudaBatchCase]) -> list[dict[str, Any]]:
        if not cases:
            return []
        started = time.perf_counter()
        torch = self.torch
        with torch.inference_mode():
            rows, errors = self._simulate_batch(cases)
        wall = time.perf_counter() - started
        per_case_wall = wall / max(len(cases), 1)
        final: list[dict[str, Any]] = []
        for case, row, error in zip(cases, rows, errors):
            case.output_dir.mkdir(parents=True, exist_ok=True)
            if error is not None:
                (case.output_dir / "cuda_physics_rejection.txt").write_text(
                    error + "\n", encoding="utf-8"
                )
                final.append({
                    "geometry_id": case.geometry_id,
                    "physicsRejected": True,
                    "physicsRejectionReason": error,
                    "backend": "torch_cuda_fp64_a100",
                })
                continue
            row = dict(row)
            row["geometry_id"] = case.geometry_id
            row["cudaBatchWallClock_s"] = wall
            row["cudaAmortisedWallClockPerCase_s"] = per_case_wall
            (case.output_dir / "condensed_metrics.json").write_text(
                json.dumps(_strict_json_value(row), indent=2, allow_nan=False),
                encoding="utf-8",
            )
            final.append(row)
        return final
