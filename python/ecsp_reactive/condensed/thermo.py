"""Barotropic Tait pressure with the *resolved BC* heat-capacity integral.

The paper does not prescribe this caloric completion. We explicitly choose
  e(rho,T) = integral_{rho0}^{rho} p(r)/r**2 dr + integral_{Tref}^{T} cp_BC(t) dt.
For p independent of T, cp=cv (zero thermal expansion). Thus a reversible
barotropic compression changes the cold energy, not the thermal energy.
Neither a new constant cv nor the independent legacy solid temperature is used.
"""
from __future__ import annotations
from typing import Mapping, Any
import math
import numpy as np
from ecsp_nsga2.propagation import PropagationConfigurationError, PropagationCandidateNumericalError


class BCHeatCapacity:
    """Analytic integral and inverse of positive constant/piecewise-linear cp."""
    def __init__(self, raw: Any, reference_temperature_K: float):
        if isinstance(raw, (float, int)):
            grid, vals = [1.0, 2.0], [float(raw)] * 2
        elif isinstance(raw, Mapping) and str(raw.get("mode", "table")) == "constant":
            grid, vals = [1.0, 2.0], [float(raw["value"])] * 2
        elif isinstance(raw, Mapping) and str(raw.get("mode", "table")) == "table":
            grid, vals = raw["temperature_K"], raw["values"]
        else:
            raise PropagationConfigurationError(
                "BC Reactive caloric closure requires constant or table heat_capacity; "
                "other BC property laws need their own enthalpy integral, not a substitute cv")
        self.grid = np.asarray(grid, dtype=np.float64)
        self.values = np.asarray(vals, dtype=np.float64)
        if (self.grid.ndim != 1 or self.grid.size < 2 or self.values.shape != self.grid.shape
                or not np.isfinite(self.grid).all() or not np.isfinite(self.values).all()
                or np.any(np.diff(self.grid) <= 0) or np.any(self.values <= 0)):
            raise PropagationConfigurationError("heat_capacity must have positive finite values on an increasing grid")
        self.slopes = np.diff(self.values) / np.diff(self.grid)
        self.integrals = np.r_[0.0, np.cumsum(0.5 * (self.values[1:] + self.values[:-1]) * np.diff(self.grid))]
        self.reference_temperature = float(reference_temperature_K)
        if not math.isfinite(self.reference_temperature) or self.reference_temperature <= 0:
            raise PropagationConfigurationError("caloric reference temperature must be positive")
        self.offset = float(self._absolute_integral(np.asarray(self.reference_temperature)))

    def cp(self, temperature):
        return np.interp(np.asarray(temperature, dtype=np.float64), self.grid, self.values)

    def _absolute_integral(self, temperature):
        t = np.asarray(temperature, dtype=np.float64)
        i = np.clip(np.searchsorted(self.grid, t, side="right") - 1, 0, self.grid.size - 2)
        d = t - self.grid[i]
        y = self.integrals[i] + self.values[i] * d + 0.5 * self.slopes[i] * d * d
        y = np.where(t <= self.grid[0], self.values[0] * (t - self.grid[0]), y)
        return np.where(t >= self.grid[-1], self.integrals[-1] + self.values[-1] * (t - self.grid[-1]), y)

    def sensible_energy(self, temperature):
        return self._absolute_integral(temperature) - self.offset

    def temperature(self, sensible_energy):
        h = np.asarray(sensible_energy, dtype=np.float64) + self.offset
        i = np.clip(np.searchsorted(self.integrals, h, side="right") - 1, 0, self.grid.size - 2)
        # Solve cp_i*d + slope_i*d^2/2 = h-H_i without cancellation.
        y = np.clip(h, self.integrals[0], self.integrals[-1]) - self.integrals[i]
        root = np.sqrt(np.maximum(self.values[i] ** 2 + 2 * self.slopes[i] * y, 0.0))
        d = 2 * y / (self.values[i] + root)
        t = self.grid[i] + d
        t = np.where(h <= 0, self.grid[0] + h / self.values[0], t)
        return np.where(h >= self.integrals[-1], self.grid[-1] + (h-self.integrals[-1])/self.values[-1], t)


class BCTaitThermodynamics:
    def __init__(self, eos_config: Mapping[str, Any], thermal: Mapping[str, Any], rho_bc: float):
        required = ("A_Pa", "B_Pa", "N", "provenance")
        if any(k not in eos_config for k in required):
            raise PropagationConfigurationError("reactive_euler.eos requires A_Pa, B_Pa, N and explicit provenance")
        if not str(eos_config["provenance"]).strip():
            raise PropagationConfigurationError("Tait coefficient provenance cannot be empty")
        self.A = float(eos_config["A_Pa"])
        self.B = float(eos_config["B_Pa"])
        self.N = float(eos_config["N"])
        rho0 = eos_config.get("rho0_kg_per_m3")
        self.rho0 = float(rho_bc if rho0 is None else rho0)
        if not all(math.isfinite(x) and x > 0 for x in (self.A,self.B,self.N,self.rho0)):
            raise PropagationConfigurationError("Tait parameters must be finite and positive")
        self.heat = BCHeatCapacity(thermal["heat_capacity"], thermal.get("initialTemperature_K", 298.15))
        self.tmin = float(thermal.get("minimumTemperature_K", 1.0))
        self.tmax = float(thermal.get("maximumTemperature_K", 10000.0))
        if not 0 < self.tmin < self.tmax or not math.isfinite(self.tmax):
            raise PropagationConfigurationError("invalid BC temperature bounds")

    def pressure(self, rho):
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            return self.A + self.B * np.expm1(self.N * np.log(np.asarray(rho) / self.rho0))

    def sound_speed(self, rho):
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            return np.sqrt(self.B * self.N / self.rho0 * (np.asarray(rho)/self.rho0) ** (self.N-1))

    def cold_energy(self, rho):
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            r = np.asarray(rho, dtype=np.float64) / self.rho0
            lr = np.log(r)
            first = lr if self.N == 1.0 else np.expm1((self.N-1)*lr)/(self.N-1)
            return self.B/self.rho0*first + (self.A-self.B)/self.rho0*(-np.expm1(-lr))

    def internal_energy(self, rho, temperature):
        return self.cold_energy(rho) + self.heat.sensible_energy(temperature)

    def temperature(self, rho, internal_energy):
        return self.heat.temperature(np.asarray(internal_energy) - self.cold_energy(rho))

    def primitive(self, state):
        """[rho,u,v,T,alpha1,alpha2, inventory/rho ...]."""
        U = np.asarray(state, dtype=np.float64)
        rho = U[..., 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            P = U / rho[..., None]
            P[..., 0] = rho
            internal = P[..., 3] - .5*(P[...,1]**2+P[...,2]**2)
            P[..., 3] = self.temperature(rho, internal)
        return P

    def conservative(self, primitive):
        P = np.asarray(primitive, dtype=np.float64)
        U = P * P[..., 0, None]
        U[..., 0] = P[..., 0]
        U[..., 3] = P[..., 0] * (self.internal_energy(P[...,0], P[...,3]) + .5*(P[...,1]**2+P[...,2]**2))
        return U

    def valid_primitive(self, P):
        pressure = self.pressure(P[...,0])
        return (np.isfinite(P).all(axis=-1) & (P[...,0]>0) & np.isfinite(pressure) & (pressure>0)
                & (P[...,3]>=self.tmin-1e-10) & (P[...,3]<=self.tmax+1e-10)
                & (P[...,4:6]>=-1e-12).all(axis=-1) & (P[...,4:6]<=1+1e-12).all(axis=-1)
                & (P[...,6:]>=-1e-12).all(axis=-1))

    def validate(self, U):
        if np.asarray(U).ndim != 3 or U.shape[-1] < 6:
            raise PropagationCandidateNumericalError("Reactive state must be [ny,nx,ncons>=6]")
        if not np.all(self.valid_primitive(self.primitive(U))):
            raise PropagationCandidateNumericalError("Nonphysical condensed Euler stage; no temperature/density clipping applied")
