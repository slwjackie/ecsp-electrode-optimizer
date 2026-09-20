"""Exactly the BC two conversion-dependent channels and inventory convention.

U[:6] = rho, rho*u, rho*v, rho*E, rho*alpha1, rho*alpha2.
Other components are *conservative* molar inventories (mol/m3), not two extra
Euler densities and not gas sources. Product water is not mobile BV water.
"""
from __future__ import annotations
import math
import numpy as np
from ecsp_nsga2.propagation import (_kinetic_rate, PropagationConfigurationError,
                                   PropagationCandidateNumericalError)

RHO, MX, MY, ENERGY, A1, A2 = range(6)
CATION, ANION, WATER, PVA, PRODUCT_WATER, EC_LP = range(6, 12)
NCONS = 12
STATE_NAMES = ("rho_kg_per_m3", "rho_u", "rho_v", "rho_E_J_per_m3", "rho_alpha1", "rho_alpha2",
               "cation_mol_per_m3", "anion_mol_per_m3", "mobile_water_mol_per_m3",
               "pva_repeat_mol_per_m3", "generated_water_product_mol_per_m3", "ec_lp_consumed_mol_per_m3")


class BCTwoChannelChemistry:
    def __init__(self, config, handoff, rho_bc, gas_constant):
        kinetics = config["kinetics"]
        self.channels = list(kinetics["channels"])
        self.weights = np.asarray(kinetics["mass_conversion_weights"], dtype=np.float64)
        self.maximum_rate = float(kinetics.get("maximum_rate_per_s", 1e6))
        self.R = float(gas_constant)
        self.rho_bc = float(rho_bc)
        self.xi_per_kg = float(handoff["xiMax_mol_per_m3"]) / self.rho_bc
        self.initial_lp_per_kg = float(handoff["initialMobileLP_mol_per_m3"]) / self.rho_bc
        self.initial_pva_per_kg = float(handoff["initialPVARepeat_mol_per_m3"]) / self.rho_bc
        self.initial_water_per_kg = float(handoff["initialMobileWater_mol_per_m3"]) / self.rho_bc
        self.m_lp = float(handoff["molarMassLP_kg_per_mol"])
        self.m_pva = float(handoff["molarMassPVARepeat_kg_per_mol"])
        if (len(self.channels)!=2 or self.weights.shape!=(2,) or not np.isfinite(self.weights).all()
                or np.any(self.weights<0) or abs(self.weights.sum()-1)>1e-12
                or not math.isfinite(self.maximum_rate) or self.maximum_rate<=0 or self.R<=0):
            raise PropagationConfigurationError("Exactly two BC channels and normalized mass_conversion_weights are required")
        basis = kinetics.get("heat_release_basis", "per_initial_bulk_propellant_channel_contribution")
        if basis != "per_initial_bulk_propellant_channel_contribution":
            raise PropagationConfigurationError("Q1/Q2 must be per-initial-bulk channel contributions, not weighted twice")
        for ch in self.channels:
            a = np.asarray(ch["alpha_grid"], dtype=np.float64)
            ea = np.asarray(ch["activation_energy_J_per_mol"], dtype=np.float64)
            l = np.asarray(ch["ln_Af_per_s"], dtype=np.float64)
            q = float(ch["heat_release_J_per_kg"])
            if (a.ndim!=1 or a.size<2 or ea.shape!=a.shape or l.shape!=a.shape
                    or not all(np.isfinite(v).all() for v in (a,ea,l))
                    or np.any(np.diff(a)<=0) or a[0]!=0 or a[-1]!=1 or np.any(ea<0)
                    or not math.isfinite(q) or q<0):
                raise PropagationConfigurationError("Invalid BC channel table")
        self.Q = np.asarray([ch["heat_release_J_per_kg"] for ch in self.channels], dtype=np.float64)

    def progress(self, U):
        return (self.weights[0]*U[...,A1]+self.weights[1]*U[...,A2])/U[...,RHO]

    def raw_rates(self, U, temperature):
        # Deliberately call the baseline implementation. No new A/E/n/Q fit.
        return np.stack([_kinetic_rate(U[...,A1+i]/U[...,RHO], temperature, self.channels[i], self.R)
                         for i in range(2)], axis=-1)

    def source(self, U, temperature, dt, *, available=None,
               concentration_floor=0.0, allowed_rate_cap_fraction=0.0,
               integration_mode="legacy_cap", maximum_channel_increment=0.02,
               maximum_subcycles=4096):
        """Return a conservative chemistry source over ``dt``.

        ``legacy_cap`` preserves the old post-onset contract.
        ``subcycle_raw`` integrates the uncapped Arrhenius kinetics locally in
        chemistry substeps while the PDE/RK stage keeps its own CFL timestep.
        In that mode ``maximum_rate_per_s`` is diagnostic only and never clips
        the accepted chemistry or rejects a candidate.
        """
        mode = str(integration_mode).lower()
        if mode not in {"legacy_cap", "subcycle_raw"}:
            raise PropagationConfigurationError(
                "chemistry_integration_mode must be legacy_cap or subcycle_raw"
            )
        if (not math.isfinite(float(dt))) or float(dt) <= 0.0:
            raise PropagationCandidateNumericalError("Invalid chemistry timestep")
        if (not math.isfinite(float(maximum_channel_increment))
                or not 0.0 < float(maximum_channel_increment) <= 1.0):
            raise PropagationConfigurationError("Invalid maximum_channel_increment")
        maximum_subcycles = int(maximum_subcycles)
        if maximum_subcycles < 1:
            raise PropagationConfigurationError("maximum_subcycles must be positive")

        rho = U[...,RHO]
        alpha0 = U[...,A1:A2+1] / rho[...,None]
        rates0 = self.raw_rates(U, temperature)
        if not np.isfinite(rates0).all() or np.any(rates0 < 0.0):
            raise PropagationCandidateNumericalError("Nonfinite/negative BC raw chemistry rate")

        if mode == "legacy_cap":
            cap_mask = np.any(rates0 > self.maximum_rate, axis=-1)
            cap_count = int(np.count_nonzero(cap_mask))
            cap_cells = int(cap_mask.size)
            allowed_cap_cells = int(math.floor(
                float(allowed_rate_cap_fraction) * cap_cells + 0.5
            ))
            cap_fraction = cap_count / max(cap_cells, 1)
            if cap_count > allowed_cap_cells:
                raise PropagationCandidateNumericalError(
                    "BC chemical rate cap would change the model; candidate rejected"
                )
            delta = np.minimum(
                np.maximum(1.0-alpha0, 0.0),
                float(dt) * np.minimum(rates0, self.maximum_rate),
            )
            resources = U if available is None else available
            if available is not None:
                capacity = np.maximum(
                    resources[...,RHO,None]-resources[...,A1:A2+1], 0.0
                ) / rho[...,None]
                delta = np.minimum(delta, capacity)
            proposed_xi = rho*self.xi_per_kg*(delta @ self.weights)
            salt = .5*(resources[...,CATION]+resources[...,ANION])
            maximum_xi = np.minimum(
                np.maximum(salt-concentration_floor,0.0)/1.45,
                np.maximum(resources[...,PVA],0.0),
            )
            scale = np.ones_like(rho)
            positive = proposed_xi > 0.0
            scale[positive] = np.minimum(
                1.0, maximum_xi[positive]/proposed_xi[positive]
            )
            delta *= scale[...,None]
            d_rho_alpha = rho[...,None]*delta
            d_xi = self.xi_per_kg*(d_rho_alpha @ self.weights)
            S = np.zeros_like(U)
            S[...,A1:A2+1] = d_rho_alpha/float(dt)
            S[...,ENERGY] = (d_rho_alpha @ self.Q)/float(dt)
            S[...,CATION] = S[...,ANION] = -1.45*d_xi/float(dt)
            S[...,PVA] = -d_xi/float(dt)
            S[...,PRODUCT_WATER] = 2.0*d_xi/float(dt)
            return S, {
                "chemical_rate_cap_fraction": float(cap_fraction),
                "inventory_limiter_fraction": float(np.mean(scale < 1.0-1e-14)),
                "maximum_raw_rate_per_s": float(np.max(rates0)),
                "chemistry_subcycles": 1,
                "chemical_rate_threshold_diagnostic_only": False,
            }

        # Raw-kinetics local chemistry subcycling. Temperature is frozen only
        # within this RK-stage source evaluation; the next RK stage sees the
        # energy/temperature change, matching the existing stage coupling.
        resources = U if available is None else available
        alpha = alpha0.copy()
        base_cation = np.asarray(resources[...,CATION], dtype=float)
        base_anion = np.asarray(resources[...,ANION], dtype=float)
        base_pva = np.asarray(resources[...,PVA], dtype=float)
        base_product = np.asarray(resources[...,PRODUCT_WATER], dtype=float)
        cation = base_cation.copy(); anion = base_anion.copy()
        pva = base_pva.copy(); product = base_product.copy()
        energy_increment = np.zeros_like(rho)
        remaining = float(dt)
        cap_max = 0.0
        limiter_max = 0.0
        maximum_raw_rate = 0.0
        subcycles = 0
        tol = 64.0*np.finfo(float).eps*max(float(dt), 1.0)

        while remaining > tol:
            rates = np.stack([
                _kinetic_rate(alpha[...,i], temperature, self.channels[i], self.R)
                for i in range(2)
            ], axis=-1)
            if not np.isfinite(rates).all() or np.any(rates < 0.0):
                raise PropagationCandidateNumericalError(
                    "Nonfinite/negative BC raw chemistry rate during subcycling"
                )
            maximum_raw_rate = max(maximum_raw_rate, float(np.max(rates)))
            cap_max = max(
                cap_max,
                float(np.mean(np.any(rates > self.maximum_rate, axis=-1))),
            )
            has_stock = (
                (.5*(cation+anion) > concentration_floor)
                & (pva > 0.0)
                & np.any(alpha < 1.0-1e-15, axis=-1)
            )
            active_rates = np.where(has_stock[...,None], rates, 0.0)
            maxrate = float(np.max(active_rates))
            if maxrate <= 0.0:
                remaining = 0.0
                break
            h = min(remaining, float(maximum_channel_increment)/maxrate)
            if not math.isfinite(h) or h <= 0.0:
                raise PropagationCandidateNumericalError(
                    "Invalid local chemistry substep"
                )
            delta = np.minimum(np.maximum(1.0-alpha,0.0), h*rates)
            proposed_xi = rho*self.xi_per_kg*(delta @ self.weights)
            salt = .5*(cation+anion)
            maximum_xi = np.minimum(
                np.maximum(salt-concentration_floor,0.0)/1.45,
                np.maximum(pva,0.0),
            )
            scale = np.ones_like(rho)
            positive = proposed_xi > 0.0
            scale[positive] = np.minimum(
                1.0, maximum_xi[positive]/proposed_xi[positive]
            )
            limiter_max = max(
                limiter_max, float(np.mean(scale < 1.0-1e-14))
            )
            delta *= scale[...,None]
            if not np.any(delta > 0.0):
                remaining = 0.0
                break
            d_rho_alpha = rho[...,None]*delta
            d_xi = self.xi_per_kg*(d_rho_alpha @ self.weights)
            alpha += delta
            cation -= 1.45*d_xi
            anion -= 1.45*d_xi
            pva -= d_xi
            product += 2.0*d_xi
            energy_increment += d_rho_alpha @ self.Q
            remaining = max(0.0, remaining-h)
            subcycles += 1
            if subcycles >= maximum_subcycles and remaining > tol:
                raise PropagationCandidateNumericalError(
                    "BC chemistry subcycle budget exceeded; use an implicit local "
                    "chemistry solver or raise the explicit subcycle budget"
                )

        S = np.zeros_like(U)
        invdt = 1.0/float(dt)
        S[...,A1:A2+1] = rho[...,None]*(alpha-alpha0)*invdt
        S[...,ENERGY] = energy_increment*invdt
        S[...,CATION] = (cation-base_cation)*invdt
        S[...,ANION] = (anion-base_anion)*invdt
        S[...,PVA] = (pva-base_pva)*invdt
        S[...,PRODUCT_WATER] = (product-base_product)*invdt
        return S, {
            "chemical_rate_cap_fraction": float(cap_max),
            "inventory_limiter_fraction": float(limiter_max),
            "maximum_raw_rate_per_s": float(maximum_raw_rate),
            "chemistry_subcycles": int(subcycles),
            "chemical_rate_threshold_diagnostic_only": True,
        }

    def conserved_inventory_residuals(self, U):
        rho = U[...,RHO]
        extent = self.xi_per_kg*(self.weights[0]*U[...,A1]+self.weights[1]*U[...,A2])
        return {
            "pva_plus_extent":U[...,PVA]+extent-rho*self.initial_pva_per_kg,
            "product_water_minus_extent":U[...,PRODUCT_WATER]-2*extent,
            # Salt redistributes by NP, so this is an integrated invariant.
            "salt_plus_ec_plus_extent":.5*(U[...,CATION]+U[...,ANION])+U[...,EC_LP]+1.45*extent-rho*self.initial_lp_per_kg,
        }

    def chemical_reservoir(self, U):
        return self.Q[0]*(U[...,RHO]-U[...,A1])+self.Q[1]*(U[...,RHO]-U[...,A2])
