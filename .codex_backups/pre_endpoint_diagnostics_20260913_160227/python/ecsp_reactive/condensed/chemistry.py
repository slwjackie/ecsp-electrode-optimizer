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

# Fixed Gauss--Legendre pairs for the one-dimensional thermochemical clock.
# Keeping the coefficients here makes the NumPy and Torch implementations use
# the same deterministic rule and avoids a production dependency on SciPy.
_GL8_NODES = np.asarray([
    -0.9602898564975363, -0.7966664774136267,
    -0.5255324099163290, -0.1834346424956498,
     0.1834346424956498,  0.5255324099163290,
     0.7966664774136267,  0.9602898564975363,
], dtype=np.float64)
_GL8_WEIGHTS = np.asarray([
    0.1012285362903763, 0.2223810344533745,
    0.3137066458778873, 0.3626837833783620,
    0.3626837833783620, 0.3137066458778873,
    0.2223810344533745, 0.1012285362903763,
], dtype=np.float64)
_GL16_NODES = np.asarray([
    -0.9894009349916499, -0.9445750230732326,
    -0.8656312023878318, -0.7554044083550030,
    -0.6178762444026438, -0.4580167776572274,
    -0.2816035507792589, -0.09501250983763744,
     0.09501250983763744, 0.2816035507792589,
     0.4580167776572274,  0.6178762444026438,
     0.7554044083550030,  0.8656312023878318,
     0.9445750230732326,  0.9894009349916499,
], dtype=np.float64)
_GL16_WEIGHTS = np.asarray([
    0.02715245941175409, 0.06225352393864789,
    0.09515851168249278, 0.1246289712555339,
    0.1495959888165767,  0.1691565193950025,
    0.1826034150449236,  0.1894506104550685,
    0.1894506104550685,  0.1826034150449236,
    0.1691565193950025,  0.1495959888165767,
    0.1246289712555339,  0.09515851168249278,
    0.06225352393864789, 0.02715245941175409,
], dtype=np.float64)

# Dormand--Prince 5(4) pair for the two-active-channel reaction path.  The
# independent variable is the conversion of the currently dominant channel;
# physical time is one of the integrated dependent variables.  These literal
# coefficients are also the scalar/tensor parity contract.
_DP54_C = (0.0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0)
_DP54_A = (
    (),
    (1.0/5.0,),
    (3.0/40.0, 9.0/40.0),
    (44.0/45.0, -56.0/15.0, 32.0/9.0),
    (19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0,
     -212.0/729.0),
    (9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0,
     -5103.0/18656.0),
    (35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0,
     -2187.0/6784.0, 11.0/84.0),
)
_DP54_B5 = np.asarray(
    (35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0,
     -2187.0/6784.0, 11.0/84.0, 0.0), dtype=np.float64,
)
_DP54_B4 = np.asarray(
    (5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0,
     -92097.0/339200.0, 187.0/2100.0, 1.0/40.0),
    dtype=np.float64,
)


class _CoordinateCertificationFailure(RuntimeError):
    """Internal signal to retain the conservative midpoint fallback."""


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
        self._kinetic_tables = []
        for ch in self.channels:
            grid = np.asarray(ch["alpha_grid"], dtype=np.float64)
            activation = np.asarray(ch["activation_energy_J_per_mol"], dtype=np.float64)
            log_prefactor = np.asarray(ch["ln_Af_per_s"], dtype=np.float64)
            width = np.diff(grid)
            self._kinetic_tables.append({
                "grid": grid,
                "activation": activation,
                "log_prefactor": log_prefactor,
                "activation_slope": np.diff(activation) / width,
                "log_prefactor_slope": np.diff(log_prefactor) / width,
            })
        self._coordinate_channels_identical = (
            self.Q[0] == self.Q[1]
            and all(np.array_equal(
                self._kinetic_tables[0][name],
                self._kinetic_tables[1][name],
            ) for name in ("grid", "activation", "log_prefactor"))
        )

    def _progress_tolerance(self, reference, *, relative_tolerance,
                            absolute_tolerance, accuracy_fraction=1.0):
        """Return an error allowance in weighted-progress units.

        The configured absolute tolerance is expressed in conversion.  Its
        progress-space image therefore includes a channel weight; using the
        raw alpha tolerance here could erase a real low-weight inventory.
        """
        reference = np.asarray(reference, dtype=np.float64)
        fraction = np.asarray(accuracy_fraction, dtype=np.float64)
        allowance = fraction*(
            float(absolute_tolerance)*float(np.min(self.weights))
            +float(relative_tolerance)*np.abs(reference)
        )
        magnitude = np.abs(reference)
        ulp = np.nextafter(magnitude, np.inf)-magnitude
        roundoff = 256.0*ulp
        combined = np.maximum(allowance, roundoff)
        return np.where(magnitude > 0.0,
                        np.minimum(combined, 0.5*magnitude), 0.0)

    def progress(self, U):
        return (self.weights[0]*U[...,A1]+self.weights[1]*U[...,A2])/U[...,RHO]

    def raw_rates(self, U, temperature):
        # Deliberately call the baseline implementation. No new A/E/n/Q fit.
        return np.stack([_kinetic_rate(U[...,A1+i]/U[...,RHO], temperature, self.channels[i], self.R)
                         for i in range(2)], axis=-1)

    def _rates_from_alpha(self, alpha, temperature):
        """The unmodified physical rates evaluated from primitive conversion."""
        return np.stack([
            _kinetic_rate(alpha[..., i], temperature, self.channels[i], self.R)
            for i in range(2)
        ], axis=-1)

    def _traversed_rate_upper_bound(self, alpha0, alpha1, temperature1):
        """Conservative raw-rate maximum over each accepted alpha interval.

        Heat releases are nonnegative, so temperature is nondecreasing during
        the local map and every Arrhenius rate is bounded above by its value at
        the final temperature.  At fixed temperature the tabulated log rate is
        affine between knots, hence its extrema occur at the accepted interval
        endpoints or at a traversed knot.
        """
        maxima = np.zeros(alpha0.shape[:-1], dtype=np.float64)
        for channel_index, channel in enumerate(self.channels):
            lo = alpha0[..., channel_index]
            hi = alpha1[..., channel_index]
            candidates = [
                _kinetic_rate(lo, temperature1, channel, self.R),
                _kinetic_rate(hi, temperature1, channel, self.R),
            ]
            for knot in np.asarray(channel["alpha_grid"], dtype=np.float64)[1:]:
                traversed = (lo <= knot) & (hi >= knot)
                evaluation_alpha = (
                    np.nextafter(1.0, 0.0) if knot == 1.0 else knot
                )
                knot_alpha = np.full_like(lo, evaluation_alpha)
                knot_rate = _kinetic_rate(
                    knot_alpha, temperature1, channel, self.R
                )
                candidates.append(np.where(traversed, knot_rate, 0.0))
            maxima = np.maximum(maxima, np.maximum.reduce(candidates))
        return maxima

    @staticmethod
    def _clock_piece(log_rate, slope, width):
        """Integral of exp(-(log_rate+slope*x)) over a positive width."""
        result = np.zeros_like(width)
        positive = width > 0.0
        if not np.any(positive):
            return result
        z = -slope[positive] * width[positive]
        exprel = np.empty_like(z)
        small = np.abs(z) <= 1.0e-7
        zs = z[small]
        exprel[small] = 1.0 + zs * (
            0.5 + zs * (1.0/6.0 + zs * (1.0/24.0 + zs/120.0))
        )
        exprel[~small] = np.expm1(z[~small]) / z[~small]
        result[positive] = (
            np.exp(-log_rate[positive]) * width[positive] * exprel
        )
        return result

    @staticmethod
    def _invert_clock_piece(log_rate, slope, elapsed, width):
        """Invert a partial affine-log-rate clock without cancellation."""
        distance = np.empty_like(elapsed)
        linear_limit = np.abs(slope * width) <= 1.0e-7
        base = elapsed[linear_limit] * np.exp(log_rate[linear_limit])
        scaled = slope[linear_limit] * base
        distance[linear_limit] = base * (
            1.0 + scaled*(0.5+scaled*(1.0/3.0+scaled*0.25))
        )
        nonlinear = ~linear_limit
        if np.any(nonlinear):
            argument = (
                -slope[nonlinear] * elapsed[nonlinear]
                * np.exp(log_rate[nonlinear])
            )
            roundoff = 128.0 * np.finfo(np.float64).eps
            if np.any(argument < -1.0-roundoff):
                raise PropagationCandidateNumericalError(
                    "Local chemistry kinetic-clock inverse left its real domain"
                )
            argument = np.maximum(argument, np.nextafter(-1.0, 0.0))
            distance[nonlinear] = (
                -np.log1p(argument) / slope[nonlinear]
            )
        tolerance = 256.0 * np.finfo(np.float64).eps
        if (np.any(distance < -tolerance)
                or np.any(distance > width+tolerance)
                or not np.isfinite(distance).all()):
            raise PropagationCandidateNumericalError(
                "Invalid local chemistry kinetic-clock inverse"
            )
        return np.minimum(np.maximum(distance, 0.0), width)

    def _fixed_beta_flow_channel_one(self, alpha, duration, beta,
                                     channel_index):
        """Scalar-math fixed-beta kinetic clock for one channel and cell."""
        machine = np.finfo(np.float64).eps
        alpha_tolerance = 256.0*machine
        table = self._kinetic_tables[channel_index]
        grid = table["grid"]
        activation = table["activation"]
        log_prefactor = table["log_prefactor"]
        activation_slope = table["activation_slope"]
        log_prefactor_slope = table["log_prefactor_slope"]
        alpha = float(alpha)
        remaining = float(duration)
        inverse_temperature = float(beta)
        for j in range(grid.size-1):
            left = float(grid[j]); right = float(grid[j+1])
            for _ in range(5):
                if not (remaining > 0.0
                        and alpha < right-alpha_tolerance
                        and alpha >= left-alpha_tolerance
                        and alpha < 1.0):
                    break
                x = max(alpha, left)
                slope = (float(log_prefactor_slope[j])
                         - float(activation_slope[j])
                           * inverse_temperature/self.R)
                raw = (float(log_prefactor[j])
                       - float(activation[j])*inverse_temperature/self.R
                       + slope*(x-left))
                piece_end = right
                effective_slope = slope
                effective_log_rate = raw

                lower = (raw < -100.0
                         or (raw <= -100.0 and slope <= 0.0))
                if lower:
                    effective_slope = 0.0
                    effective_log_rate = -100.0
                    if slope > 0.0:
                        piece_end = min(
                            right, x+(-100.0-raw)/slope
                        )
                upper = (raw > 60.0
                         or (raw >= 60.0 and slope >= 0.0))
                if upper:
                    effective_slope = 0.0
                    effective_log_rate = 60.0
                    if slope < 0.0:
                        piece_end = min(
                            right, x+(60.0-raw)/slope
                        )
                if not (lower or upper):
                    if slope > 0.0:
                        piece_end = min(right, x+(60.0-raw)/slope)
                    elif slope < 0.0:
                        piece_end = min(right, x+(-100.0-raw)/slope)
                piece_end = min(max(piece_end, x), right)
                width = piece_end-x
                if width <= alpha_tolerance:
                    alpha = math.nextafter(piece_end, right)
                    continue

                z = -effective_slope*width
                if abs(z) <= 1.0e-7:
                    exprel = 1.0 + z*(
                        0.5+z*(1.0/6.0+z*(1.0/24.0+z/120.0))
                    )
                else:
                    exprel = math.expm1(z)/z
                piece_clock = math.exp(-effective_log_rate)*width*exprel
                if not math.isfinite(piece_clock) or piece_clock <= 0.0:
                    raise PropagationCandidateNumericalError(
                        "Nonfinite scalar local chemistry kinetic clock"
                    )
                if remaining >= piece_clock:
                    alpha = piece_end
                    remaining = max(0.0, remaining-piece_clock)
                    continue

                if abs(effective_slope*width) <= 1.0e-7:
                    base = remaining*math.exp(effective_log_rate)
                    scaled = effective_slope*base
                    distance = base*(
                        1.0+scaled*(0.5+scaled*(1.0/3.0+scaled*0.25))
                    )
                else:
                    argument = (-effective_slope*remaining
                                * math.exp(effective_log_rate))
                    roundoff = 128.0*machine
                    if argument < -1.0-roundoff:
                        raise PropagationCandidateNumericalError(
                            "Scalar chemistry clock inverse left its real domain"
                        )
                    argument = max(argument, math.nextafter(-1.0, 0.0))
                    distance = -math.log1p(argument)/effective_slope
                tolerance = 256.0*machine
                if (not math.isfinite(distance) or distance < -tolerance
                        or distance > width+tolerance):
                    raise PropagationCandidateNumericalError(
                        "Invalid scalar local chemistry kinetic-clock inverse"
                    )
                alpha += min(max(distance, 0.0), width)
                remaining = 0.0

            if (remaining > 0.0 and alpha < right-alpha_tolerance
                    and alpha >= left-alpha_tolerance):
                raise PropagationCandidateNumericalError(
                    "Scalar clipped affine chemistry traversal did not terminate"
                )
        return 1.0 if alpha >= 1.0-alpha_tolerance else alpha

    def _fixed_beta_flow_channel_scalar(self, alpha0, duration, beta,
                                        channel_index):
        """Scalar-math twin of the fixed-beta clock for small gathered sets."""
        answer = np.empty_like(alpha0)
        for cell in range(alpha0.size):
            answer[cell] = self._fixed_beta_flow_channel_one(
                alpha0[cell], duration[cell], beta[cell], channel_index
            )
        return answer

    def _fixed_beta_flow_channel(self, alpha0, duration, beta, channel_index):
        """Exact endpoint at fixed inverse temperature for one table channel.

        The physical log rate is affine on every alpha-table interval.  The
        exponent-protection intersections at -100 and 60 are treated as real
        sub-interval boundaries, so neither a rate cap nor an alpha step is
        introduced here.
        """
        alpha, duration, beta = np.broadcast_arrays(
            np.asarray(alpha0, dtype=np.float64),
            np.asarray(duration, dtype=np.float64),
            np.asarray(beta, dtype=np.float64),
        )
        shape = alpha.shape
        alpha = alpha.reshape(-1).copy()
        remaining = duration.reshape(-1).copy()
        beta = beta.reshape(-1)
        if (not np.isfinite(alpha).all() or not np.isfinite(remaining).all()
                or not np.isfinite(beta).all() or np.any(remaining < 0.0)
                or np.any(beta <= 0.0)):
            raise PropagationCandidateNumericalError(
                "Invalid fixed-temperature chemistry endpoint input"
            )

        machine = np.finfo(np.float64).eps
        alpha_tolerance = 256.0 * machine
        if np.any(alpha < -alpha_tolerance) or np.any(alpha > 1.0+alpha_tolerance):
            raise PropagationCandidateNumericalError(
                "Conversion is outside the local chemistry domain"
            )
        alpha = np.minimum(np.maximum(alpha, 0.0), 1.0)
        if alpha.size <= 64:
            return self._fixed_beta_flow_channel_scalar(
                alpha, remaining, beta, channel_index
            ).reshape(shape)
        table = self._kinetic_tables[channel_index]
        grid = table["grid"]
        activation = table["activation"]
        log_prefactor = table["log_prefactor"]
        activation_slope = table["activation_slope"]
        log_prefactor_slope = table["log_prefactor_slope"]

        for j in range(grid.size-1):
            left = float(grid[j]); right = float(grid[j+1])
            # A clipped affine function has at most three forward pieces:
            # lower plateau, affine interior, upper plateau (or reverse).
            # Three mathematical pieces suffice.  Two extra passes permit a
            # zero-width roundoff snap exactly at either clipping intersection.
            for _ in range(5):
                active = ((remaining > 0.0) & (alpha < right-alpha_tolerance)
                          & (alpha >= left-alpha_tolerance) & (alpha < 1.0))
                if not np.any(active):
                    continue
                indices = np.flatnonzero(active)
                x = np.maximum(alpha[indices], left)
                b = beta[indices]
                slope = log_prefactor_slope[j] - activation_slope[j]*b/self.R
                raw = (log_prefactor[j] - activation[j]*b/self.R
                       + slope*(x-left))

                piece_end = np.full_like(x, right)
                effective_slope = slope.copy()
                effective_log_rate = raw.copy()

                lower = ((raw < -100.0)
                         | ((raw <= -100.0) & (slope <= 0.0)))
                if np.any(lower):
                    effective_slope[lower] = 0.0
                    effective_log_rate[lower] = -100.0
                    crossing = lower & (slope > 0.0)
                    piece_end[crossing] = np.minimum(
                        right, x[crossing]+(-100.0-raw[crossing])/slope[crossing]
                    )

                upper = ((raw > 60.0)
                         | ((raw >= 60.0) & (slope >= 0.0)))
                if np.any(upper):
                    effective_slope[upper] = 0.0
                    effective_log_rate[upper] = 60.0
                    crossing = upper & (slope < 0.0)
                    piece_end[crossing] = np.minimum(
                        right, x[crossing]+(60.0-raw[crossing])/slope[crossing]
                    )

                interior = ~(lower | upper)
                rising = interior & (slope > 0.0)
                falling = interior & (slope < 0.0)
                piece_end[rising] = np.minimum(
                    right, x[rising]+(60.0-raw[rising])/slope[rising]
                )
                piece_end[falling] = np.minimum(
                    right, x[falling]+(-100.0-raw[falling])/slope[falling]
                )
                piece_end = np.minimum(np.maximum(piece_end, x), right)
                width = piece_end-x

                zero_width = width <= alpha_tolerance
                if np.any(zero_width):
                    alpha[indices[zero_width]] = np.nextafter(
                        piece_end[zero_width], right
                    )
                usable = ~zero_width
                if not np.any(usable):
                    continue

                use_indices = indices[usable]
                use_width = width[usable]
                use_log_rate = effective_log_rate[usable]
                use_slope = effective_slope[usable]
                piece_clock = self._clock_piece(
                    use_log_rate, use_slope, use_width
                )
                if not np.isfinite(piece_clock).all() or np.any(piece_clock <= 0.0):
                    raise PropagationCandidateNumericalError(
                        "Nonfinite local chemistry kinetic clock"
                    )
                consume = remaining[use_indices] >= piece_clock
                if np.any(consume):
                    consumed_indices = use_indices[consume]
                    alpha[consumed_indices] = piece_end[usable][consume]
                    remaining[consumed_indices] = np.maximum(
                        0.0, remaining[consumed_indices]-piece_clock[consume]
                    )
                partial = ~consume
                if np.any(partial):
                    partial_indices = use_indices[partial]
                    distance = self._invert_clock_piece(
                        use_log_rate[partial], use_slope[partial],
                        remaining[partial_indices], use_width[partial],
                    )
                    alpha[partial_indices] += distance
                    remaining[partial_indices] = 0.0

            stranded = ((remaining > 0.0) & (alpha < right-alpha_tolerance)
                        & (alpha >= left-alpha_tolerance))
            if np.any(stranded):
                raise PropagationCandidateNumericalError(
                    "Clipped affine chemistry segment traversal did not terminate"
                )

        alpha = np.where(alpha >= 1.0-alpha_tolerance, 1.0, alpha)
        return alpha.reshape(shape)

    def _fixed_beta_flow(self, alpha0, duration, beta):
        return np.stack([
            self._fixed_beta_flow_channel(
                alpha0[..., i], duration, beta, i
            ) for i in range(2)
        ], axis=-1)

    def _endpoint_at_beta(self, alpha0, duration, beta, progress_capacity,
                          *, relative_tolerance, absolute_tolerance,
                          maximum_depletion_iterations,
                          accuracy_fraction=None):
        """Fixed-beta endpoint with one shared inventory-depletion event."""
        if accuracy_fraction is None:
            accuracy_fraction = np.ones_like(duration, dtype=np.float64)
        accuracy_fraction = np.broadcast_to(
            np.asarray(accuracy_fraction, dtype=np.float64), duration.shape
        )
        endpoint = self._fixed_beta_flow(alpha0, duration, beta)
        progress = (endpoint-alpha0) @ self.weights
        capacity_tolerance = self._progress_tolerance(
            progress_capacity,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            accuracy_fraction=accuracy_fraction,
        )
        no_capacity = progress_capacity <= 0.0
        # A positive represented inventory is physical state, even when it is
        # smaller than the alpha error tolerance.  Tolerances control only the
        # event-root residual; they never suppress that inventory or decide
        # whether an unconstrained endpoint binds.
        binds = (progress > progress_capacity) & ~no_capacity
        would_react = progress > 0.0
        depleted = binds | (no_capacity & would_react)
        depletion_time = np.full_like(duration, np.nan)
        depletion_iterations = np.zeros(duration.shape, dtype=np.int64)

        if np.any(no_capacity):
            # Exactly empty stock cannot accept any reaction increment.
            stopped = no_capacity
            endpoint[stopped] = alpha0[stopped]
            progress[stopped] = 0.0
            depletion_time[stopped & would_react] = 0.0

        depletion_ok = np.ones(duration.shape, dtype=bool)
        if np.any(binds):
            indices = np.flatnonzero(binds)
            a0 = alpha0[indices]
            q = duration[indices]
            b = beta[indices]
            target = progress_capacity[indices]
            tolerance = np.minimum(
                capacity_tolerance[indices], 0.5*target
            )
            lower = np.zeros_like(q)
            upper = q.copy()
            converged = np.zeros_like(q, dtype=bool)
            iterations = np.zeros(q.shape, dtype=np.int64)
            lower_progress = np.zeros_like(q)

            for iteration in range(1, maximum_depletion_iterations+1):
                trial_time = 0.5*(lower+upper)
                trial_alpha = self._fixed_beta_flow(a0, trial_time, b)
                trial_progress = (trial_alpha-a0) @ self.weights
                residual = trial_progress-target
                # Update the sign bracket first.  Only the non-overshooting
                # lower-time side may be accepted; otherwise an event within
                # tolerance could still make C/A/PVA negative.
                go_right = residual <= 0.0
                lower = np.where(~converged & go_right, trial_time, lower)
                lower_progress = np.where(
                    ~converged & go_right, trial_progress, lower_progress
                )
                upper = np.where(~converged & ~go_right, trial_time, upper)
                newly = (~converged
                         & (target-lower_progress >= 0.0)
                         & (target-lower_progress <= tolerance))
                iterations[newly] = iteration
                converged |= newly
                if np.all(converged):
                    break

            trial_time = lower
            trial_alpha = self._fixed_beta_flow(a0, trial_time, b)
            trial_progress = (trial_alpha-a0) @ self.weights
            target_magnitude = np.abs(target)
            root_roundoff = 256.0*(
                np.nextafter(target_magnitude, np.inf)-target_magnitude
            )
            nonovershooting = trial_progress <= target + root_roundoff
            converged &= (nonovershooting
                          & (target-trial_progress <= tolerance))
            iterations[iterations == 0] = maximum_depletion_iterations

            endpoint[indices] = trial_alpha
            progress[indices] = (trial_alpha-a0) @ self.weights
            depletion_time[indices] = trial_time
            depletion_iterations[indices] = iterations
            depletion_ok[indices] = converged

        return {
            "alpha": endpoint,
            "progress": progress,
            "depleted": depleted,
            "depletion_time": depletion_time,
            "depletion_iterations": depletion_iterations,
            "depletion_ok": depletion_ok,
        }

    def _implicit_midpoint_panel(self, alpha0, sensible0, temperature0,
                                 duration, progress_capacity, thermo, *,
                                 relative_tolerance, absolute_tolerance,
                                 temperature_tolerance_K,
                                 maximum_corrector_iterations,
                                 maximum_depletion_iterations,
                                 accuracy_fraction=None):
        """One reciprocal-temperature midpoint panel for a batch of cells.

        Corrector evaluations are gathered over unresolved cells only.  One
        thermally difficult cell therefore cannot keep reevaluating every
        already-converged cell in the domain.
        """
        count = duration.size
        if accuracy_fraction is None:
            accuracy_fraction = np.ones_like(duration, dtype=np.float64)
        accuracy_fraction = np.broadcast_to(
            np.asarray(accuracy_fraction, dtype=np.float64), duration.shape
        )
        beta0 = 1.0/np.maximum(temperature0, 1.0)
        adiabatic = sensible0 + ((1.0-alpha0) @ self.Q)
        temperature_adiabatic = thermo.heat.temperature(adiabatic)
        if (not np.isfinite(temperature_adiabatic).all()
                or np.any(temperature_adiabatic <= 0.0)):
            raise PropagationCandidateNumericalError(
                "Invalid adiabatic bound in local chemistry"
            )
        beta_lower = 0.5*(
            beta0 + 1.0/np.maximum(temperature_adiabatic, 1.0)
        )

        def evaluate(indices, beta):
            value = self._endpoint_at_beta(
                alpha0[indices], duration[indices], beta,
                progress_capacity[indices],
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                maximum_depletion_iterations=maximum_depletion_iterations,
                accuracy_fraction=accuracy_fraction[indices],
            )
            sensible1 = (sensible0[indices]
                         + ((value["alpha"]-alpha0[indices]) @ self.Q))
            temperature1 = thermo.heat.temperature(sensible1)
            if (not np.isfinite(temperature1).all()
                    or np.any(temperature1 <= 0.0)):
                raise PropagationCandidateNumericalError(
                    "Nonfinite temperature in local chemistry corrector"
                )
            value["sensible"] = sensible1
            value["temperature"] = temperature1
            value["beta_target"] = 0.5*(
                beta0[indices] + 1.0/np.maximum(temperature1, 1.0)
            )
            roundoff = (512.0*np.finfo(np.float64).eps
                        * np.maximum(beta0[indices], 1.0))
            if (np.any(value["beta_target"] < beta_lower[indices]-roundoff)
                    or np.any(value["beta_target"] > beta0[indices]+roundoff)):
                raise PropagationCandidateNumericalError(
                    "Local chemistry thermal corrector left its physical bracket"
                )
            value["beta_target"] = np.minimum(
                np.maximum(value["beta_target"], beta_lower[indices]),
                beta0[indices],
            )
            return value

        all_indices = np.arange(count)
        predictor = evaluate(all_indices, beta0)
        beta = predictor["beta_target"].copy()
        previous_alpha = predictor["alpha"].copy()
        previous_temperature = predictor["temperature"].copy()
        active = np.ones(duration.shape, dtype=bool)
        corrector_iterations = np.zeros(duration.shape, dtype=np.int64)
        endpoint_evaluations = np.ones(duration.shape, dtype=np.int64)
        depletion_iteration_sum = predictor["depletion_iterations"].copy()
        maximum_depletion_iterations_used = predictor["depletion_iterations"].copy()
        normalized_corrector_residual = np.full(duration.shape, np.inf)
        feedback_alpha_correction = np.zeros(duration.shape)
        feedback_temperature_correction = np.zeros(duration.shape)
        result = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in predictor.items()
        }

        for iteration in range(1, maximum_corrector_iterations+1):
            indices = np.flatnonzero(active)
            if indices.size == 0:
                break
            candidate = evaluate(indices, beta[indices])
            endpoint_evaluations[indices] += 1
            depletion_iteration_sum[indices] += candidate["depletion_iterations"]
            maximum_depletion_iterations_used[indices] = np.maximum(
                maximum_depletion_iterations_used[indices],
                candidate["depletion_iterations"],
            )
            alpha_scale = np.maximum(
                np.maximum(
                    np.abs(candidate["alpha"]),
                    np.abs(previous_alpha[indices]),
                ),
                1.0,
            )
            alpha_allowance = np.maximum(
                accuracy_fraction[indices, None]
                *(absolute_tolerance+relative_tolerance*alpha_scale),
                256.0*np.finfo(np.float64).eps*alpha_scale,
            )
            alpha_residual = np.max(
                np.abs(candidate["alpha"]-previous_alpha[indices])
                /alpha_allowance, axis=-1
            )
            temperature_scale = np.maximum.reduce([
                np.abs(candidate["temperature"]),
                np.abs(previous_temperature[indices]),
                np.ones(indices.size),
            ])
            temperature_allowance = np.maximum(
                accuracy_fraction[indices]*temperature_tolerance_K,
                256.0*np.finfo(np.float64).eps*temperature_scale,
            )
            endpoint_temperature_residual = (
                np.abs(candidate["temperature"]-previous_temperature[indices])
                /temperature_allowance
            )
            effective_temperature = 1.0/np.maximum(
                beta[indices], np.finfo(float).tiny
            )
            target_temperature = 1.0/np.maximum(
                candidate["beta_target"], np.finfo(float).tiny
            )
            midpoint_temperature_residual = (
                np.abs(effective_temperature-target_temperature)
                /temperature_allowance
            )
            residual = np.maximum.reduce([
                alpha_residual,
                endpoint_temperature_residual,
                midpoint_temperature_residual,
            ])
            normalized_corrector_residual[indices] = residual
            feedback_alpha_correction[indices] = np.maximum(
                feedback_alpha_correction[indices],
                np.max(np.abs(candidate["alpha"]-predictor["alpha"][indices]), axis=-1),
            )
            feedback_temperature_correction[indices] = np.maximum(
                feedback_temperature_correction[indices],
                np.abs(candidate["temperature"]-predictor["temperature"][indices]),
            )
            for key in ("alpha", "progress", "depleted", "depletion_time",
                        "depletion_iterations", "depletion_ok", "sensible",
                        "temperature", "beta_target"):
                result[key][indices] = candidate[key]
            corrector_iterations[indices] = iteration
            converged_now = (residual <= 1.0) & candidate["depletion_ok"]
            active[indices[converged_now]] = False
            unresolved = indices[~converged_now]
            beta[unresolved] = candidate["beta_target"][~converged_now]
            previous_alpha[unresolved] = candidate["alpha"][~converged_now]
            previous_temperature[unresolved] = candidate["temperature"][~converged_now]

        result["converged"] = ~active
        result["corrector_iterations"] = corrector_iterations
        result["endpoint_evaluations"] = endpoint_evaluations
        result["depletion_iteration_sum"] = depletion_iteration_sum
        result["maximum_depletion_iterations_used"] = maximum_depletion_iterations_used
        result["normalized_corrector_residual"] = normalized_corrector_residual
        result["feedback_alpha_correction"] = feedback_alpha_correction
        result["feedback_temperature_correction"] = feedback_temperature_correction
        return result

    def _integrate_uniform_panels(self, alpha0, sensible0, temperature0,
                                  duration, progress_capacity, panel_count,
                                  thermo, *, relative_tolerance,
                                  absolute_tolerance, temperature_tolerance_K,
                                  maximum_corrector_iterations,
                                  maximum_depletion_iterations,
                                  accuracy_fraction=None):
        """Integrate a cell batch on one uniform dyadic panel grid."""
        count = duration.size
        if accuracy_fraction is None:
            accuracy_fraction = np.ones_like(duration, dtype=np.float64)
        accuracy_fraction = np.broadcast_to(
            np.asarray(accuracy_fraction, dtype=np.float64), duration.shape
        )
        result = {
            "alpha": alpha0.copy(),
            "sensible": sensible0.copy(),
            "temperature": temperature0.copy(),
            "capacity_left": progress_capacity.copy(),
            "depleted": np.zeros(count, dtype=bool),
            "depletion_time": np.full(count, np.nan),
            "valid": np.ones(count, dtype=bool),
            "maximum_corrector_iterations_used": np.zeros(count, dtype=np.int64),
            "maximum_depletion_iterations_used": np.zeros(count, dtype=np.int64),
            "corrector_iteration_sum": np.zeros(count, dtype=np.int64),
            "depletion_iteration_sum": np.zeros(count, dtype=np.int64),
            "endpoint_evaluations": np.zeros(count, dtype=np.int64),
            "evaluated_cell_count": np.zeros(count, dtype=np.int64),
            "maximum_normalized_corrector_residual": np.zeros(count),
            "feedback_alpha_correction": np.zeros(count),
            "feedback_temperature_correction": np.zeros(count),
            "invalid_panel_attempt_count": np.zeros(count, dtype=np.int64),
        }
        panel_duration = duration/float(panel_count)
        for panel_number in range(panel_count):
            indices = np.flatnonzero(result["valid"])
            if indices.size == 0:
                break
            panel = self._implicit_midpoint_panel(
                result["alpha"][indices], result["sensible"][indices],
                result["temperature"][indices], panel_duration[indices],
                result["capacity_left"][indices], thermo,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                temperature_tolerance_K=temperature_tolerance_K,
                maximum_corrector_iterations=maximum_corrector_iterations,
                maximum_depletion_iterations=maximum_depletion_iterations,
                accuracy_fraction=(
                    0.125*accuracy_fraction[indices]/float(panel_count)
                ),
            )
            result["evaluated_cell_count"][indices] += 1
            result["corrector_iteration_sum"][indices] += panel["corrector_iterations"]
            result["depletion_iteration_sum"][indices] += panel["depletion_iteration_sum"]
            result["endpoint_evaluations"][indices] += panel["endpoint_evaluations"]
            result["maximum_corrector_iterations_used"][indices] = np.maximum(
                result["maximum_corrector_iterations_used"][indices],
                panel["corrector_iterations"],
            )
            result["maximum_depletion_iterations_used"][indices] = np.maximum(
                result["maximum_depletion_iterations_used"][indices],
                panel["maximum_depletion_iterations_used"],
            )
            finite_residual = np.where(
                np.isfinite(panel["normalized_corrector_residual"]),
                panel["normalized_corrector_residual"], 0.0,
            )
            result["maximum_normalized_corrector_residual"][indices] = np.maximum(
                result["maximum_normalized_corrector_residual"][indices],
                finite_residual,
            )
            result["feedback_alpha_correction"][indices] = np.maximum(
                result["feedback_alpha_correction"][indices],
                panel["feedback_alpha_correction"],
            )
            result["feedback_temperature_correction"][indices] = np.maximum(
                result["feedback_temperature_correction"][indices],
                panel["feedback_temperature_correction"],
            )

            good_local = panel["converged"]
            good = indices[good_local]
            bad = indices[~good_local]
            if good.size:
                old_alpha = result["alpha"][good].copy()
                consumed = (panel["alpha"][good_local]-old_alpha) @ self.weights
                result["alpha"][good] = panel["alpha"][good_local]
                result["sensible"][good] = panel["sensible"][good_local]
                result["temperature"][good] = panel["temperature"][good_local]
                result["capacity_left"][good] = np.maximum(
                    result["capacity_left"][good]-consumed, 0.0
                )
                newly_depleted = (panel["depleted"][good_local]
                                  & ~result["depleted"][good])
                if np.any(newly_depleted):
                    event_cells = good[newly_depleted]
                    local_event_time = panel["depletion_time"][good_local][newly_depleted]
                    result["depletion_time"][event_cells] = (
                        panel_number*panel_duration[event_cells]+local_event_time
                    )
                result["depleted"][good] |= panel["depleted"][good_local]
            result["valid"][bad] = False
            result["invalid_panel_attempt_count"][bad] += 1
        return result

    @staticmethod
    def _accumulate_work(total, attempt, indices):
        for key in ("corrector_iteration_sum", "depletion_iteration_sum",
                    "endpoint_evaluations", "evaluated_cell_count",
                    "invalid_panel_attempt_count"):
            total[key][indices] += attempt[key]
        for key in ("maximum_corrector_iterations_used",
                    "maximum_depletion_iterations_used",
                    "maximum_normalized_corrector_residual",
                    "feedback_alpha_correction",
                    "feedback_temperature_correction"):
            total[key][indices] = np.maximum(total[key][indices], attempt[key])

    def _unclipped_log_rates(self, alpha, temperature):
        """Arrhenius exponents before the physical [-100, 60] guard."""
        alpha = np.asarray(alpha, dtype=np.float64)
        answer = np.empty(2, dtype=np.float64)
        for channel_index, channel in enumerate(self.channels):
            activation = float(np.interp(
                float(alpha[channel_index]), channel["alpha_grid"],
                channel["activation_energy_J_per_mol"],
            ))
            log_prefactor = float(np.interp(
                float(alpha[channel_index]), channel["alpha_grid"],
                channel["ln_Af_per_s"],
            ))
            answer[channel_index] = (
                log_prefactor-activation/(self.R*max(float(temperature), 1.0))
            )
        return answer

    @staticmethod
    def _clamp_region(exponent):
        exponent = np.asarray(exponent, dtype=np.float64)
        return np.where(exponent < -100.0, -1,
                        np.where(exponent > 60.0, 1, 0))

    def _coupled_coordinate_dp54_step(
            self, alpha_reference, sensible_reference, driver, x, y, width,
            thermo):
        """One DP5(4) step in a dominant channel's reaction coordinate.

        ``y[0]`` is the other channel conversion and ``y[1]`` is physical
        elapsed time.  Sensible energy is reconstructed from conversion on
        every stage, so the chemical-energy transfer remains exact.
        """
        driver = int(driver)
        other = 1-driver
        stages = []
        stage_alpha = []
        stage_sensible = []
        stage_rates = []
        stage_exponents = []
        state_tolerance = 1024.0*np.finfo(np.float64).eps

        for coefficient, row in zip(_DP54_C, _DP54_A):
            value = np.asarray(y, dtype=np.float64).copy()
            for scale, derivative in zip(row, stages):
                value += float(width)*float(scale)*derivative
            alpha = np.asarray(alpha_reference, dtype=np.float64).copy()
            alpha[driver] = float(x)+float(coefficient)*float(width)
            alpha[other] = value[0]
            if (not np.isfinite(alpha).all()
                    or alpha[driver] < -state_tolerance
                    or alpha[driver] >= 1.0
                    or alpha[other] < -state_tolerance
                    or alpha[other] >= 1.0):
                raise _CoordinateCertificationFailure(
                    "coordinate stage left the open conversion domain"
                )
            sensible = (float(sensible_reference)
                        +float(np.dot(self.Q, alpha-alpha_reference)))
            temperature = float(thermo.heat.temperature(sensible))
            rates = self._rates_from_alpha(
                alpha[None, :], np.asarray([temperature], dtype=np.float64)
            )[0]
            if (not math.isfinite(temperature) or temperature <= 0.0
                    or not np.isfinite(rates).all()
                    or np.any(rates <= 0.0)):
                raise _CoordinateCertificationFailure(
                    "invalid dominant-coordinate stage rate or temperature"
                )
            derivative = np.asarray([
                float(rates[other]/rates[driver]),
                float(1.0/rates[driver]),
            ], dtype=np.float64)
            if not np.isfinite(derivative).all() or np.any(derivative <= 0.0):
                raise _CoordinateCertificationFailure(
                    "invalid dominant-coordinate derivative"
                )
            stages.append(derivative)
            stage_alpha.append(alpha)
            stage_sensible.append(sensible)
            stage_rates.append(rates)
            stage_exponents.append(
                self._unclipped_log_rates(alpha, temperature)
            )

        derivatives = np.asarray(stages, dtype=np.float64)
        high = np.asarray(y, dtype=np.float64)+float(width)*(
            _DP54_B5 @ derivatives
        )
        low = np.asarray(y, dtype=np.float64)+float(width)*(
            _DP54_B4 @ derivatives
        )
        if (not np.isfinite(high).all() or not np.isfinite(low).all()
                or high[0] < y[0]-state_tolerance
                or high[0] >= 1.0
                or high[1] <= y[1]):
            raise _CoordinateCertificationFailure(
                "nonmonotone dominant-coordinate endpoint"
            )
        return {
            "high": high,
            "low": low,
            "stage_alpha": np.asarray(stage_alpha),
            "stage_sensible": np.asarray(stage_sensible),
            "stage_rates": np.asarray(stage_rates),
            "stage_exponents": np.asarray(stage_exponents),
            "rate_evaluations": len(stages),
        }

    def _coupled_coordinate_normalized_error(
            self, step, alpha_reference, sensible_reference, driver,
            endpoint_x, thermo, *, relative_tolerance, absolute_tolerance,
            temperature_tolerance_K):
        """Map the embedded composition and clock error to state tolerances."""
        other = 1-int(driver)
        high = step["high"]
        low = step["low"]
        high_alpha = np.asarray(alpha_reference, dtype=np.float64).copy()
        low_alpha = high_alpha.copy()
        high_alpha[driver] = low_alpha[driver] = float(endpoint_x)
        high_alpha[other] = high[0]
        low_alpha[other] = low[0]
        state_tolerance = 1024.0*np.finfo(np.float64).eps
        if (np.any(high_alpha < -state_tolerance)
                or np.any(high_alpha >= 1.0)
                or np.any(low_alpha < -state_tolerance)
                or np.any(low_alpha >= 1.0)):
            raise _CoordinateCertificationFailure(
                "embedded endpoint left the open conversion domain"
            )
        high_sensible = (float(sensible_reference)
                         +float(np.dot(self.Q, high_alpha-alpha_reference)))
        low_sensible = (float(sensible_reference)
                        +float(np.dot(self.Q, low_alpha-alpha_reference)))
        high_temperature = float(thermo.heat.temperature(high_sensible))
        low_temperature = float(thermo.heat.temperature(low_sensible))
        rates = self._rates_from_alpha(
            high_alpha[None, :],
            np.asarray([high_temperature], dtype=np.float64),
        )[0]
        if (not np.isfinite(rates).all() or np.any(rates <= 0.0)
                or not math.isfinite(high_temperature)
                or not math.isfinite(low_temperature)):
            raise _CoordinateCertificationFailure(
                "invalid embedded endpoint state"
            )
        alpha_scale = np.maximum(
            np.maximum(np.abs(high_alpha), np.abs(low_alpha)), 1.0
        )
        alpha_tolerance = (
            float(absolute_tolerance)+float(relative_tolerance)*alpha_scale
        )
        direct_alpha = abs(float(high[0]-low[0]))/alpha_tolerance[other]
        direct_temperature = (
            abs(high_temperature-low_temperature)
            /float(temperature_tolerance_K)
        )
        clock_error = abs(float(high[1]-low[1]))
        mapped_alpha = float(np.max(clock_error*rates/alpha_tolerance))
        heat_rate = float(np.dot(self.Q, rates))
        mapped_temperature = (
            clock_error*heat_rate
            /float(thermo.heat.cp(high_temperature))
            /float(temperature_tolerance_K)
        )
        # The 5(4) difference is an estimator, not a proof.  The factor two is
        # the scalar/tensor safety contract and kept the independent hard-cell
        # error below one fifth of both requested endpoint tolerances.
        return 2.0*max(
            direct_alpha, direct_temperature,
            mapped_alpha, mapped_temperature,
        )

    def _solve_coupled_reaction_coordinate_cell(
            self, alpha0, sensible0, temperature0, duration,
            progress_capacity, thermo, *, relative_tolerance,
            absolute_tolerance, temperature_tolerance_K,
            maximum_depletion_iterations, maximum_local_refinements,
            maximum_reaction_coordinate_steps):
        """Error-controlled two-active path using a dominant coordinate.

        The fast path integrates only while a panel stays inside one smooth
        kinetic/caloric/exponent region. Driver and non-driver kinetic knots
        are explicit restart points; target-time, shared-inventory, and
        dominance events are located on the same reaction-coordinate path.
        Caloric or exponent-clamp kinks that cannot use that monotone event
        construction retain the conservative midpoint fallback.  The embedded
        DP5(4) pair plus a factor-two safety margin is an error estimator, not a
        formal proof, so the implementation deliberately avoids describing the
        accepted path as mathematically certified.
        """
        alpha_reference = np.asarray(alpha0, dtype=np.float64).copy()
        alpha = alpha_reference.copy()
        sensible_reference = float(sensible0)
        sensible = sensible_reference
        temperature = float(temperature0)
        duration = float(duration)
        capacity_reference = float(progress_capacity)
        capacity_left = capacity_reference
        elapsed = 0.0
        accepted_steps = 0
        rejected_steps = 0
        rate_evaluations = 0
        driver_switches = 0
        maximum_depth = 0
        maximum_normalized = 0.0
        maximum_rejected = 0.0
        forced_driver = None
        step_hint = [None, None]
        machine = np.finfo(np.float64).eps
        state_tolerance = 256.0*machine
        capacity_tolerance = float(self._progress_tolerance(
            progress_capacity,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        ))
        # ``duration`` may legitimately be as small as the solver's 1e-14 s
        # floor.  Scaling this guard to one second would then classify the
        # entire interval as roundoff and return the unchanged state.  Keep
        # the guard at the ULP scale of the time coordinates being compared.
        time_roundoff = 256.0*machine*max(
            abs(duration), abs(elapsed), np.finfo(np.float64).tiny
        )

        caloric_knots = np.asarray(
            thermo.heat.sensible_energy(thermo.heat.grid), dtype=np.float64
        )

        while elapsed < duration-time_roundoff:
            if accepted_steps >= int(maximum_reaction_coordinate_steps):
                raise _CoordinateCertificationFailure(
                    "dominant-coordinate step guard requested midpoint fallback"
                )
            active = alpha < 1.0-state_tolerance
            active_indices = np.flatnonzero(active)
            if active_indices.size == 0:
                break
            if capacity_left <= 0.0:
                return {
                    "alpha": alpha, "sensible": sensible,
                    "temperature": temperature,
                    "capacity_left": max(capacity_left, 0.0),
                    "depleted": True, "depletion_time": elapsed,
                    "rate_evaluations": rate_evaluations,
                    "coupled_rate_evaluations": rate_evaluations,
                    "accepted_steps": accepted_steps,
                    "rejected_steps": rejected_steps,
                    "driver_switches": driver_switches,
                    "maximum_depth": maximum_depth,
                    "maximum_normalized_residual": maximum_normalized,
                    "maximum_rejected_normalized_residual": maximum_rejected,
                    "one_active": None,
                }
            if active_indices.size == 1:
                channel_index = int(active_indices[0])
                coupled_rate_evaluations = rate_evaluations
                remaining_duration = duration-elapsed
                one_active = self._solve_one_active_reaction_coordinate_cell(
                    alpha, sensible, temperature, remaining_duration,
                    capacity_left, channel_index, thermo,
                    relative_tolerance=relative_tolerance,
                    absolute_tolerance=absolute_tolerance,
                    temperature_tolerance_K=temperature_tolerance_K,
                    maximum_depletion_iterations=maximum_depletion_iterations,
                    maximum_local_refinements=maximum_local_refinements,
                )
                rate_evaluations += int(one_active["rate_evaluations"])
                depletion_time = (
                    elapsed+float(one_active["depletion_time"])
                    if one_active["depleted"] else math.nan
                )
                tail_capacity_left = (
                    capacity_reference-float(np.dot(
                        self.weights,
                        one_active["alpha"]-alpha_reference,
                    ))
                )
                if (not math.isfinite(tail_capacity_left)
                        or tail_capacity_left < 0.0):
                    raise _CoordinateCertificationFailure(
                        "one-active tail would overshoot shared inventory"
                    )
                return {
                    "alpha": one_active["alpha"],
                    "sensible": one_active["sensible"],
                    "temperature": one_active["temperature"],
                    "capacity_left": tail_capacity_left,
                    "depleted": one_active["depleted"],
                    "depletion_time": depletion_time,
                    "rate_evaluations": rate_evaluations,
                    "coupled_rate_evaluations": coupled_rate_evaluations,
                    "accepted_steps": accepted_steps,
                    "rejected_steps": rejected_steps,
                    "driver_switches": driver_switches,
                    "maximum_depth": max(
                        maximum_depth, int(one_active["maximum_depth"])
                    ),
                    "maximum_normalized_residual": max(
                        maximum_normalized,
                        float(one_active["maximum_normalized_residual"]),
                    ),
                    "maximum_rejected_normalized_residual": max(
                        maximum_rejected,
                        float(one_active[
                            "maximum_rejected_normalized_residual"
                        ]),
                    ),
                    "one_active": one_active,
                }

            rates = self._rates_from_alpha(
                alpha[None, :], np.asarray([temperature], dtype=np.float64)
            )[0]
            rate_evaluations += 1
            if not np.isfinite(rates).all() or np.any(rates <= 0.0):
                raise _CoordinateCertificationFailure(
                    "invalid rate selecting a dominant coordinate"
                )
            if forced_driver is None:
                driver = int(np.argmax(rates))
                other = 1-driver
                relative_gap = abs(float(rates[driver]-rates[other]))/max(
                    float(rates[driver]), float(rates[other]),
                    np.finfo(np.float64).tiny,
                )
                if (relative_gap <= math.sqrt(machine)
                        and not self._coordinate_channels_identical):
                    # The symmetric midpoint path avoids an arbitrary index
                    # tie-break and preserves channel-permutation symmetry.
                    raise _CoordinateCertificationFailure(
                        "ambiguous equal-rate dominant coordinate"
                    )
            else:
                driver = int(forced_driver)
                other = 1-driver
                forced_driver = None
                if rates[driver] < rates[other]*(1.0-math.sqrt(machine)):
                    raise _CoordinateCertificationFailure(
                        "dominance switch did not enter the new region"
                    )

            grid = np.asarray(
                self.channels[driver]["alpha_grid"], dtype=np.float64
            )
            candidates = grid[grid > alpha[driver]+256.0*machine]
            if candidates.size == 0:
                driver_target = math.nextafter(1.0, 0.0)
                completes_driver = True
            else:
                next_value = float(candidates[0])
                completes_driver = next_value >= 1.0
                driver_target = (
                    math.nextafter(1.0, 0.0)
                    if completes_driver else next_value
                )
            span = driver_target-float(alpha[driver])
            if not span > 0.0:
                raise _CoordinateCertificationFailure(
                    "dominant-coordinate interval made no progress"
                )
            width = span if step_hint[driver] is None else min(
                span, float(step_hint[driver])
            )
            depth = 0
            start_alpha = alpha.copy()
            y = np.asarray([alpha[other], elapsed], dtype=np.float64)
            start_exponents = self._unclipped_log_rates(alpha, temperature)
            start_regions = self._clamp_region(start_exponents)

            while True:
                if (not math.isfinite(width) or width <= 0.0
                        or float(alpha[driver])+width
                            == float(alpha[driver])):
                    raise _CoordinateCertificationFailure(
                        "dominant-coordinate width underflow"
                    )
                try:
                    step = self._coupled_coordinate_dp54_step(
                        alpha_reference, sensible_reference, driver,
                        float(alpha[driver]), y, width, thermo,
                    )
                    rate_evaluations += int(step["rate_evaluations"])
                    normalized = self._coupled_coordinate_normalized_error(
                        step, alpha_reference, sensible_reference, driver,
                        float(alpha[driver])+width, thermo,
                        relative_tolerance=relative_tolerance,
                        absolute_tolerance=absolute_tolerance,
                        temperature_tolerance_K=temperature_tolerance_K,
                    )
                    rate_evaluations += 1
                except _CoordinateCertificationFailure:
                    normalized = math.inf
                    step = None

                if not math.isfinite(normalized) or normalized > 1.0:
                    rejected_steps += 1
                    if math.isfinite(normalized):
                        maximum_rejected = max(maximum_rejected, normalized)
                    depth += 1
                    maximum_depth = max(maximum_depth, depth)
                    if depth > int(maximum_local_refinements):
                        raise _CoordinateCertificationFailure(
                            "dominant-coordinate embedded pair did not certify"
                        )
                    factor = (
                        0.5 if not math.isfinite(normalized) or normalized <= 0.0
                        else min(0.5, max(0.1, 0.8*normalized**-0.2))
                    )
                    width *= factor
                    continue

                high = step["high"]
                endpoint_alpha = alpha.copy()
                endpoint_alpha[driver] = float(alpha[driver])+width
                endpoint_alpha[other] = high[0]
                endpoint_sensible = (
                    sensible_reference
                    +float(np.dot(self.Q, endpoint_alpha-alpha_reference))
                )
                endpoint_temperature = float(
                    thermo.heat.temperature(endpoint_sensible)
                )

                # Stop and restart at a non-driver kinetic knot just as at a
                # driver knot.  This is needed before locating a later common
                # time/inventory event, and avoids sending an otherwise smooth
                # two-channel path through thousands of midpoint panels.
                other_grid = np.asarray(
                    self.channels[other]["alpha_grid"], dtype=np.float64
                )
                other_candidates = other_grid[
                    other_grid > alpha[other]+256.0*machine
                ]
                other_boundary = (
                    float(other_candidates[0])
                    if other_candidates.size else 1.0
                )
                other_root_threshold = (
                    other_boundary if other_boundary < 1.0
                    else other_boundary-state_tolerance
                )
                if (np.max(step["stage_alpha"][:, other])
                        >= other_root_threshold):
                    if high[0] < other_root_threshold:
                        rejected_steps += 1
                        depth += 1
                        maximum_depth = max(maximum_depth, depth)
                        if depth > int(maximum_local_refinements):
                            raise _CoordinateCertificationFailure(
                                "interior non-driver knot did not isolate"
                            )
                        width *= 0.5
                        continue
                    lower_width = 0.0
                    upper_width = width
                    upper_step = step
                    for _ in range(int(maximum_depletion_iterations)):
                        trial_width = 0.5*(lower_width+upper_width)
                        if trial_width in {lower_width, upper_width}:
                            break
                        trial = self._coupled_coordinate_dp54_step(
                            alpha_reference, sensible_reference, driver,
                            float(alpha[driver]), y, trial_width, thermo,
                        )
                        rate_evaluations += int(trial["rate_evaluations"])
                        if trial["high"][0] >= other_root_threshold:
                            upper_width = trial_width
                            upper_step = trial
                        else:
                            lower_width = trial_width
                        relative_bracket = (
                            (upper_width-lower_width)
                            /max(abs(float(alpha[driver])), width, 1.0)
                        )
                        if relative_bracket <= 256.0*machine:
                            break
                    step = upper_step
                    width = upper_width
                    normalized = self._coupled_coordinate_normalized_error(
                        step, alpha_reference, sensible_reference, driver,
                        float(alpha[driver])+width, thermo,
                        relative_tolerance=relative_tolerance,
                        absolute_tolerance=absolute_tolerance,
                        temperature_tolerance_K=temperature_tolerance_K,
                    )
                    rate_evaluations += 1
                    if not math.isfinite(normalized) or normalized > 1.0:
                        raise _CoordinateCertificationFailure(
                            "non-driver knot endpoint did not certify"
                        )
                    high = step["high"]
                    endpoint_alpha = alpha.copy()
                    endpoint_alpha[driver] = float(alpha[driver])+width
                    endpoint_alpha[other] = high[0]
                    endpoint_sensible = (
                        sensible_reference+float(np.dot(
                            self.Q, endpoint_alpha-alpha_reference
                        ))
                    )
                    endpoint_temperature = float(
                        thermo.heat.temperature(endpoint_sensible)
                    )
                    # Restart the next panel on the exact table boundary.  The
                    # bisection above limits this signed snap to the accepted
                    # endpoint tolerance; all heat and capacity ledgers are
                    # reconstructed from this single endpoint below.
                    unsnapped_other = float(endpoint_alpha[other])
                    unsnapped_temperature = endpoint_temperature
                    endpoint_alpha[other] = other_boundary
                    endpoint_sensible = (
                        sensible_reference+float(np.dot(
                            self.Q, endpoint_alpha-alpha_reference
                        ))
                    )
                    endpoint_temperature = float(
                        thermo.heat.temperature(endpoint_sensible)
                    )
                    snap_alpha_tolerance = (
                        float(absolute_tolerance)
                        +float(relative_tolerance)*max(
                            abs(float(endpoint_alpha[other])),
                            abs(unsnapped_other), 1.0,
                        )
                    )
                    snap_normalized = 2.0*max(
                        abs(float(endpoint_alpha[other])-unsnapped_other)
                            /snap_alpha_tolerance,
                        abs(endpoint_temperature-unsnapped_temperature)
                            /float(temperature_tolerance_K),
                    )
                    normalized = max(normalized, snap_normalized)
                    if not math.isfinite(normalized) or normalized > 1.0:
                        raise _CoordinateCertificationFailure(
                            "non-driver knot snap did not certify"
                        )

                # Caloric and exponent-clamp crossings still decline to the
                # general midpoint path; their roots are not monotone in the
                # selected reaction coordinate without a wider proof.
                caloric_candidates = caloric_knots[
                    caloric_knots > sensible+256.0*machine*max(abs(sensible), 1.0)
                ]
                if (caloric_candidates.size
                        and np.max(step["stage_sensible"])
                            >= float(caloric_candidates[0])):
                    raise _CoordinateCertificationFailure(
                        "caloric knot requires midpoint fallback"
                    )
                if np.any(
                    self._clamp_region(step["stage_exponents"])
                    != start_regions[None, :]
                ):
                    raise _CoordinateCertificationFailure(
                        "exponent clamp kink requires midpoint fallback"
                    )

                stage_rates = step["stage_rates"]
                dominance_crossed = (
                    not self._coordinate_channels_identical
                    and np.any(
                        stage_rates[:, other] >= stage_rates[:, driver]
                    )
                )
                endpoint_crossed = (
                    not self._coordinate_channels_identical
                    and stage_rates[-1, other] >= stage_rates[-1, driver]
                )
                if dominance_crossed:
                    if not endpoint_crossed:
                        rejected_steps += 1
                        depth += 1
                        maximum_depth = max(maximum_depth, depth)
                        if depth > int(maximum_local_refinements):
                            raise _CoordinateCertificationFailure(
                                "interior dominance crossing did not isolate"
                            )
                        width *= 0.5
                        continue
                    # Bracket the crossing by old- and new-dominant sides.
                    # Accepting the certified upper side below records which
                    # physical channel crossed and avoids an index-based tie
                    # decision.
                    lower_width = 0.0
                    upper_width = width
                    lower_step = None
                    upper_step = step
                    for _ in range(int(maximum_depletion_iterations)):
                        trial_width = 0.5*(lower_width+upper_width)
                        trial = self._coupled_coordinate_dp54_step(
                            alpha_reference, sensible_reference, driver,
                            float(alpha[driver]), y, trial_width, thermo,
                        )
                        rate_evaluations += int(trial["rate_evaluations"])
                        trial_rates = trial["stage_rates"][-1]
                        if trial_rates[other] >= trial_rates[driver]:
                            upper_width = trial_width
                            upper_step = trial
                        else:
                            lower_width = trial_width
                            lower_step = trial
                        relative_bracket = (
                            (upper_width-lower_width)
                            /max(abs(float(alpha[driver])), width, 1.0)
                        )
                        if relative_bracket <= 256.0*machine:
                            break
                    if upper_step is None or upper_width <= 0.0:
                        raise _CoordinateCertificationFailure(
                            "dominance crossing has no certified bracket"
                        )
                    # Accept the upper/new-dominant side of the roundoff-scale
                    # bracket.  The old coordinate only traverses an
                    # infinitesimal post-crossing sliver, while the next panel
                    # starts with a genuinely dominant forced driver.
                    step = upper_step
                    width = upper_width
                    normalized = self._coupled_coordinate_normalized_error(
                        step, alpha_reference, sensible_reference, driver,
                        float(alpha[driver])+width, thermo,
                        relative_tolerance=relative_tolerance,
                        absolute_tolerance=absolute_tolerance,
                        temperature_tolerance_K=temperature_tolerance_K,
                    )
                    rate_evaluations += 1
                    if not math.isfinite(normalized) or normalized > 1.0:
                        rejected_steps += 1
                        depth += 1
                        maximum_depth = max(maximum_depth, depth)
                        if depth > int(maximum_local_refinements):
                            raise _CoordinateCertificationFailure(
                                "dominance crossing endpoint did not certify"
                            )
                        width *= 0.5
                        continue
                    high = step["high"]
                    endpoint_alpha = alpha.copy()
                    endpoint_alpha[driver] = float(alpha[driver])+width
                    endpoint_alpha[other] = high[0]
                    endpoint_sensible = (
                        sensible_reference+float(np.dot(
                            self.Q, endpoint_alpha-alpha_reference
                        ))
                    )
                    endpoint_temperature = float(
                        thermo.heat.temperature(endpoint_sensible)
                    )
                    forced_driver = other
                    driver_switches += 1

                consumed = float(np.dot(
                    self.weights, endpoint_alpha-start_alpha
                ))
                if consumed < -capacity_tolerance:
                    raise _CoordinateCertificationFailure(
                        "negative weighted progress in coordinate panel"
                    )
                # Locate the earliest common target-time/shared-inventory event
                # in coordinate width.  The union predicate is monotone because
                # both elapsed time and weighted progress increase along the
                # certified two-channel path.  The noncrossing lower endpoint
                # preserves inventory exactly; the lower/upper state bracket
                # consumes the half-budget left by the factor-two DP estimator.
                crosses_time = float(high[1]) > duration
                crosses_inventory = consumed > capacity_left
                if crosses_time or crosses_inventory:
                    lower_width = 0.0
                    lower_alpha = start_alpha.copy()
                    lower_sensible = sensible
                    lower_temperature = temperature
                    lower_time = elapsed
                    lower_consumed = 0.0
                    lower_normalized = 0.0
                    upper_width = width
                    upper_alpha = endpoint_alpha.copy()
                    upper_sensible = endpoint_sensible
                    upper_temperature = endpoint_temperature
                    upper_time = float(high[1])
                    upper_consumed = consumed
                    upper_normalized = float(normalized)
                    upper_crosses_time = crosses_time
                    upper_crosses_inventory = crosses_inventory
                    event_is_inventory = False
                    event_certified = False
                    bracket_normalized = math.inf

                    for _ in range(int(maximum_depletion_iterations)):
                        trial_width = 0.5*(lower_width+upper_width)
                        if trial_width in {lower_width, upper_width}:
                            break
                        trial = self._coupled_coordinate_dp54_step(
                            alpha_reference, sensible_reference, driver,
                            float(alpha[driver]), y, trial_width, thermo,
                        )
                        rate_evaluations += int(trial["rate_evaluations"])
                        trial_normalized = (
                            self._coupled_coordinate_normalized_error(
                                trial, alpha_reference, sensible_reference,
                                driver, float(alpha[driver])+trial_width,
                                thermo,
                                relative_tolerance=relative_tolerance,
                                absolute_tolerance=absolute_tolerance,
                                temperature_tolerance_K=(
                                    temperature_tolerance_K
                                ),
                            )
                        )
                        rate_evaluations += 1
                        if (not math.isfinite(trial_normalized)
                                or trial_normalized > 1.0):
                            raise _CoordinateCertificationFailure(
                                "common-event coordinate trial did not certify"
                            )
                        trial_high = trial["high"]
                        trial_alpha = alpha.copy()
                        trial_alpha[driver] = (
                            float(alpha[driver])+trial_width
                        )
                        trial_alpha[other] = float(trial_high[0])
                        trial_sensible = (
                            sensible_reference+float(np.dot(
                                self.Q, trial_alpha-alpha_reference
                            ))
                        )
                        trial_temperature = float(
                            thermo.heat.temperature(trial_sensible)
                        )
                        trial_consumed = float(np.dot(
                            self.weights, trial_alpha-start_alpha
                        ))
                        if (not math.isfinite(trial_temperature)
                                or not math.isfinite(trial_consumed)
                                or trial_consumed < -capacity_tolerance):
                            raise _CoordinateCertificationFailure(
                                "invalid common-event coordinate trial"
                            )
                        trial_crosses_time = (
                            float(trial_high[1]) >= duration
                        )
                        trial_crosses_inventory = (
                            trial_consumed >= capacity_left
                        )
                        if trial_crosses_time or trial_crosses_inventory:
                            upper_width = trial_width
                            upper_alpha = trial_alpha
                            upper_sensible = trial_sensible
                            upper_temperature = trial_temperature
                            upper_time = float(trial_high[1])
                            upper_consumed = trial_consumed
                            upper_normalized = float(trial_normalized)
                            upper_crosses_time = trial_crosses_time
                            upper_crosses_inventory = (
                                trial_crosses_inventory
                            )
                        else:
                            lower_width = trial_width
                            lower_alpha = trial_alpha
                            lower_sensible = trial_sensible
                            lower_temperature = trial_temperature
                            lower_time = float(trial_high[1])
                            lower_consumed = trial_consumed
                            lower_normalized = float(trial_normalized)

                        alpha_scale = np.maximum(
                            np.maximum(
                                np.abs(lower_alpha), np.abs(upper_alpha)
                            ),
                            1.0,
                        )
                        alpha_allowance = (
                            float(absolute_tolerance)
                            +float(relative_tolerance)*alpha_scale
                        )
                        bracket_normalized = 2.0*max(
                            float(np.max(
                                np.abs(upper_alpha-lower_alpha)
                                /alpha_allowance
                            )),
                            abs(upper_temperature-lower_temperature)
                            /float(temperature_tolerance_K),
                        )
                        remaining_capacity = (
                            capacity_left-lower_consumed
                        )
                        inventory_certified = (
                            upper_crosses_inventory
                            and remaining_capacity >= 0.0
                            and remaining_capacity
                                <= 0.5*capacity_tolerance
                        )
                        if bracket_normalized <= 1.0:
                            if inventory_certified:
                                event_is_inventory = True
                                event_certified = True
                                break
                            if upper_crosses_time:
                                event_is_inventory = False
                                event_certified = True
                                break

                    if not event_certified:
                        raise _CoordinateCertificationFailure(
                            "common time/inventory event root did not certify"
                        )
                    root_capacity_left = (
                        capacity_reference-float(np.dot(
                            self.weights, lower_alpha-alpha_reference
                        ))
                    )
                    if (not math.isfinite(root_capacity_left)
                            or root_capacity_left < 0.0):
                        raise _CoordinateCertificationFailure(
                            "common-event root overshot shared inventory"
                        )
                    return {
                        "alpha": lower_alpha,
                        "sensible": lower_sensible,
                        "temperature": lower_temperature,
                        "capacity_left": root_capacity_left,
                        "depleted": event_is_inventory,
                        "depletion_time": (
                            lower_time if event_is_inventory else math.nan
                        ),
                        "rate_evaluations": rate_evaluations,
                        "coupled_rate_evaluations": rate_evaluations,
                        "accepted_steps": accepted_steps+1,
                        "rejected_steps": rejected_steps,
                        "driver_switches": driver_switches,
                        "maximum_depth": maximum_depth,
                        "maximum_normalized_residual": max(
                            maximum_normalized, lower_normalized,
                            upper_normalized, bracket_normalized,
                        ),
                        "maximum_rejected_normalized_residual": (
                            maximum_rejected
                        ),
                        "one_active": None,
                    }

                endpoint_capacity_left = (
                    capacity_reference-float(np.dot(
                        self.weights, endpoint_alpha-alpha_reference
                    ))
                )
                if (not math.isfinite(endpoint_capacity_left)
                        or endpoint_capacity_left < 0.0):
                    raise _CoordinateCertificationFailure(
                        "shared inventory root requires midpoint fallback"
                    )

                alpha = endpoint_alpha
                sensible = endpoint_sensible
                temperature = endpoint_temperature
                elapsed = min(float(high[1]), duration)
                # Reconstruct from the initial capacity and cumulative progress
                # instead of accumulating rounded panel decrements.
                capacity_left = endpoint_capacity_left
                accepted_steps += 1
                maximum_normalized = max(maximum_normalized, normalized)
                next_factor = (
                    5.0 if normalized == 0.0 else
                    min(5.0, max(0.2, 0.9*normalized**-0.2))
                )
                step_hint[driver] = width*next_factor

                at_driver_boundary = (
                    driver_target-float(alpha[driver])
                    <= 256.0*machine*max(
                        abs(driver_target), abs(float(alpha[driver])), 1.0
                    )
                )
                if at_driver_boundary:
                    exact_target = 1.0 if completes_driver else driver_target
                    snapped_alpha = alpha.copy()
                    snapped_alpha[driver] = float(exact_target)
                    snapped_capacity_left = (
                        capacity_reference-float(np.dot(
                            self.weights, snapped_alpha-alpha_reference
                        ))
                    )
                    if (not math.isfinite(snapped_capacity_left)
                            or snapped_capacity_left < 0.0):
                        raise _CoordinateCertificationFailure(
                            "coordinate boundary snap would overshoot inventory"
                        )
                    # The signed cumulative reconstruction also refunds the
                    # one-ULP case where alpha+width rounded above the knot.
                    alpha = snapped_alpha
                    capacity_left = snapped_capacity_left
                    sensible = (
                        sensible_reference+float(np.dot(
                            self.Q, alpha-alpha_reference
                        ))
                    )
                    temperature = float(thermo.heat.temperature(sensible))
                    step_hint[driver] = None
                break

        return {
            "alpha": alpha, "sensible": sensible,
            "temperature": temperature,
            "capacity_left": max(capacity_left, 0.0),
            "depleted": False, "depletion_time": math.nan,
            "rate_evaluations": rate_evaluations,
            "coupled_rate_evaluations": rate_evaluations,
            "accepted_steps": accepted_steps,
            "rejected_steps": rejected_steps,
            "driver_switches": driver_switches,
            "maximum_depth": maximum_depth,
            "maximum_normalized_residual": maximum_normalized,
            "maximum_rejected_normalized_residual": maximum_rejected,
            "one_active": None,
        }

    def _reaction_coordinate_breaks(self, alpha0, alpha1, sensible0,
                                    channel_index, heat):
        """Kinetic, caloric, and exponent-clamp knots on the alpha path."""
        points = [float(alpha0), float(alpha1)]
        grid = self._kinetic_tables[channel_index]["grid"]
        points.extend(float(value) for value in grid
                      if alpha0 < value < alpha1)
        heat_release = float(self.Q[channel_index])
        if heat_release > 0.0:
            knot_energy = heat.sensible_energy(heat.grid)
            caloric_alpha = (float(alpha0)
                             +(knot_energy-float(sensible0))/heat_release)
            points.extend(float(value) for value in caloric_alpha
                          if alpha0 < value < alpha1)
        base = np.asarray(sorted(set(points)), dtype=np.float64)

        def unclipped_log_rate(alpha):
            alpha = np.asarray(alpha, dtype=np.float64)
            sensible = (float(sensible0)+heat_release
                        *(alpha-float(alpha0)))
            temperature = heat.temperature(sensible)
            channel = self.channels[channel_index]
            activation = np.interp(
                alpha, channel["alpha_grid"],
                channel["activation_energy_J_per_mol"],
            )
            log_prefactor = np.interp(
                alpha, channel["alpha_grid"], channel["ln_Af_per_s"]
            )
            return log_prefactor-activation/(self.R*np.maximum(temperature, 1.0))

        # On each kinetic/caloric-smooth interval locate the real boundaries
        # introduced by exponent protection.  A short scan finds sign brackets;
        # bisection then makes each detected crossing an explicit quadrature
        # endpoint rather than asking an embedded rule to integrate a kink.
        clamp_points = []
        log_evaluations = 0
        for left, right in zip(base[:-1], base[1:]):
            scan = np.linspace(left, right, 9)
            values = unclipped_log_rate(scan)
            log_evaluations += scan.size
            for threshold in (-100.0, 60.0):
                residual = values-threshold
                for j in range(scan.size-1):
                    lo=float(scan[j]); hi=float(scan[j+1])
                    flo=float(residual[j]); fhi=float(residual[j+1])
                    if flo == 0.0 and alpha0 < lo < alpha1:
                        clamp_points.append(lo)
                    if flo*fhi >= 0.0:
                        continue
                    for _ in range(60):
                        midpoint=0.5*(lo+hi)
                        if midpoint in {lo,hi}:
                            break
                        fm=float(unclipped_log_rate(midpoint)-threshold)
                        log_evaluations += 1
                        if flo*fm <= 0.0:
                            hi=midpoint; fhi=fm
                        else:
                            lo=midpoint; flo=fm
                        if hi-lo <= 64.0*np.finfo(np.float64).eps*max(
                                abs(lo),abs(hi),1.0):
                            break
                    root=0.5*(lo+hi)
                    if alpha0 < root < alpha1:
                        clamp_points.append(root)
        points.extend(clamp_points)
        return (np.asarray(sorted(set(points)), dtype=np.float64),
                int(log_evaluations))

    def _reaction_coordinate_clock_pair(self, alpha_left, alpha_right,
                                        alpha0, sensible0, channel_index,
                                        heat):
        """Return GL16 clock and its GL8 difference on one smooth interval."""
        half = 0.5*(float(alpha_right)-float(alpha_left))
        midpoint = 0.5*(float(alpha_right)+float(alpha_left))
        if half <= 0.0:
            return {
                "clock": 0.0, "clock_error": 0.0,
                "alpha_error": 0.0, "temperature_error": 0.0,
                "rate_evaluations": 0,
            }

        def evaluate(nodes):
            alpha = midpoint+half*nodes
            sensible = (float(sensible0)
                        +float(self.Q[channel_index])*(alpha-float(alpha0)))
            temperature = heat.temperature(sensible)
            rate = _kinetic_rate(
                alpha, temperature, self.channels[channel_index], self.R
            )
            if (not np.isfinite(temperature).all()
                    or np.any(temperature <= 0.0)
                    or not np.isfinite(rate).all() or np.any(rate <= 0.0)):
                raise PropagationCandidateNumericalError(
                    "Invalid one-active-channel thermochemical clock integrand"
                )
            return temperature, rate

        temperature8, rate8 = evaluate(_GL8_NODES)
        temperature16, rate16 = evaluate(_GL16_NODES)
        clock8 = half*float(np.dot(_GL8_WEIGHTS, 1.0/rate8))
        clock16 = half*float(np.dot(_GL16_WEIGHTS, 1.0/rate16))
        if (not math.isfinite(clock8) or not math.isfinite(clock16)
                or clock8 <= 0.0 or clock16 <= 0.0):
            raise PropagationCandidateNumericalError(
                "Nonfinite one-active-channel thermochemical clock"
            )

        # Endpoint samples make the time-to-state error conversion stable for
        # rapidly increasing Arrhenius rates.  Use the left limit at alpha=1,
        # because alpha=1 itself correctly has zero physical rate.
        endpoint_alpha = np.asarray([
            float(alpha_left),
            (math.nextafter(1.0, 0.0)
             if alpha_right >= 1.0 else float(alpha_right)),
        ])
        endpoint_sensible = (float(sensible0)
                             +float(self.Q[channel_index])
                              *(endpoint_alpha-float(alpha0)))
        endpoint_temperature = heat.temperature(endpoint_sensible)
        endpoint_rate = _kinetic_rate(
            endpoint_alpha, endpoint_temperature,
            self.channels[channel_index], self.R,
        )
        if (not np.isfinite(endpoint_rate).all()
                or np.any(endpoint_rate <= 0.0)):
            raise PropagationCandidateNumericalError(
                "Invalid endpoint rate in one-active-channel clock"
            )
        maximum_rate = float(max(np.max(rate8), np.max(rate16),
                                 np.max(endpoint_rate)))
        all_temperature = np.concatenate((temperature8, temperature16,
                                          endpoint_temperature))
        cp = heat.cp(all_temperature)
        maximum_temperature_derivative = (
            float(self.Q[channel_index])/float(np.min(cp))
            if self.Q[channel_index] > 0.0 else 0.0
        )
        # The pair difference is an estimator, not an exact bound.  A safety
        # multiplier plus a floating-point floor prevents false zero error.
        clock_error = (8.0*abs(clock16-clock8)
                       +64.0*np.finfo(np.float64).eps*abs(clock16))
        alpha_error = clock_error*maximum_rate
        return {
            "clock": clock16,
            "clock_error": clock_error,
            "alpha_error": alpha_error,
            "temperature_error": alpha_error*maximum_temperature_derivative,
            "rate_evaluations": 26,
        }

    def _reaction_coordinate_clock_interval(
            self, alpha_left, alpha_right, alpha0, sensible0, channel_index,
            heat, *, alpha_tolerance, temperature_tolerance_K, total_span,
            depth, maximum_local_refinements):
        pair = self._reaction_coordinate_clock_pair(
            alpha_left, alpha_right, alpha0, sensible0, channel_index, heat
        )
        fraction = ((float(alpha_right)-float(alpha_left))/float(total_span))
        local_alpha_tolerance = max(
            float(alpha_tolerance)*fraction,
            256.0*np.finfo(np.float64).eps,
        )
        local_temperature_tolerance = max(
            float(temperature_tolerance_K)*fraction,
            256.0*np.finfo(np.float64).eps,
        )
        normalized = max(
            pair["alpha_error"]/local_alpha_tolerance,
            pair["temperature_error"]/local_temperature_tolerance,
        )
        if math.isfinite(normalized) and normalized <= 0.125:
            pair.update({
                "maximum_depth": int(depth),
                "quadrature_panel_evaluations": 1,
                "maximum_normalized_residual": float(normalized),
                "maximum_rejected_normalized_residual": 0.0,
            })
            return pair
        if depth >= maximum_local_refinements:
            raise _CoordinateCertificationFailure(
                "One-active-channel GL8/GL16 thermochemical clock failed "
                "at its local refinement budget"
            )
        midpoint = 0.5*(float(alpha_left)+float(alpha_right))
        left = self._reaction_coordinate_clock_interval(
            alpha_left, midpoint, alpha0, sensible0, channel_index, heat,
            alpha_tolerance=alpha_tolerance,
            temperature_tolerance_K=temperature_tolerance_K,
            total_span=total_span, depth=depth+1,
            maximum_local_refinements=maximum_local_refinements,
        )
        right = self._reaction_coordinate_clock_interval(
            midpoint, alpha_right, alpha0, sensible0, channel_index, heat,
            alpha_tolerance=alpha_tolerance,
            temperature_tolerance_K=temperature_tolerance_K,
            total_span=total_span, depth=depth+1,
            maximum_local_refinements=maximum_local_refinements,
        )
        return {
            "clock": left["clock"]+right["clock"],
            "clock_error": left["clock_error"]+right["clock_error"],
            "alpha_error": left["alpha_error"]+right["alpha_error"],
            "temperature_error": (left["temperature_error"]
                                    +right["temperature_error"]),
            "rate_evaluations": (pair["rate_evaluations"]
                                 +left["rate_evaluations"]
                                 +right["rate_evaluations"]),
            "maximum_depth": max(left["maximum_depth"],
                                 right["maximum_depth"]),
            "quadrature_panel_evaluations": (
                1+left["quadrature_panel_evaluations"]
                +right["quadrature_panel_evaluations"]
            ),
            "maximum_normalized_residual": max(
                left["maximum_normalized_residual"],
                right["maximum_normalized_residual"],
            ),
            "maximum_rejected_normalized_residual": max(
                float(normalized),
                left["maximum_rejected_normalized_residual"],
                right["maximum_rejected_normalized_residual"],
            ),
        }

    def _reaction_coordinate_clock_path(
            self, alpha0, alpha1, sensible0, channel_index, heat, *,
            alpha_tolerance, temperature_tolerance_K,
            maximum_local_refinements):
        """Integrate dt/dalpha, split at both kinetic and caloric knots."""
        span = float(alpha1)-float(alpha0)
        empty = {
            "clock": 0.0, "clock_error": 0.0,
            "alpha_error": 0.0, "temperature_error": 0.0,
            "rate_evaluations": 0, "maximum_depth": 0,
            "quadrature_panel_evaluations": 0,
            "maximum_normalized_residual": 0.0,
            "normalized_alpha_residual": 0.0,
            "normalized_temperature_residual": 0.0,
            "maximum_rejected_normalized_residual": 0.0,
        }
        if span <= 0.0:
            return empty
        breaks, breakpoint_evaluations = self._reaction_coordinate_breaks(
            alpha0, alpha1, sensible0, channel_index, heat
        )
        answer = dict(empty)
        answer["rate_evaluations"] += breakpoint_evaluations
        for left, right in zip(breaks[:-1], breaks[1:]):
            interval = self._reaction_coordinate_clock_interval(
                left, right, alpha0, sensible0, channel_index, heat,
                alpha_tolerance=alpha_tolerance,
                temperature_tolerance_K=temperature_tolerance_K,
                total_span=span, depth=0,
                maximum_local_refinements=maximum_local_refinements,
            )
            for key in ("clock", "clock_error", "alpha_error",
                        "temperature_error", "rate_evaluations",
                        "quadrature_panel_evaluations"):
                answer[key] += interval[key]
            for key in ("maximum_depth", "maximum_normalized_residual",
                        "maximum_rejected_normalized_residual"):
                answer[key] = max(answer[key], interval[key])
        answer["normalized_alpha_residual"] = (
            answer["alpha_error"]/float(alpha_tolerance)
        )
        answer["normalized_temperature_residual"] = (
            answer["temperature_error"]/float(temperature_tolerance_K)
        )
        answer["maximum_normalized_residual"] = max(
            answer["maximum_normalized_residual"],
            answer["normalized_alpha_residual"],
            answer["normalized_temperature_residual"],
        )
        return answer

    def _solve_one_active_reaction_coordinate_cell(
            self, alpha0, sensible0, temperature0, duration,
            progress_capacity, channel_index, thermo, *,
            relative_tolerance, absolute_tolerance,
            temperature_tolerance_K, maximum_depletion_iterations,
            maximum_local_refinements):
        """Monotone one-channel thermochemical endpoint in alpha-space."""
        channel_index = int(channel_index)
        start = float(alpha0[channel_index])
        weight = float(self.weights[channel_index])
        inventory_bound = (start+float(progress_capacity)/weight
                           if weight > 0.0 else math.inf)
        bound = min(1.0, inventory_bound)
        inventory_binds = inventory_bound <= 1.0
        if inventory_binds:
            # Choose the greatest representable endpoint that cannot consume
            # more than the available shared inventory.  The analytic bound
            # can otherwise overshoot by one rounding unit when reconstructed
            # as weight*(bound-start).
            while weight*(bound-start) > float(progress_capacity):
                bound = math.nextafter(bound, start)
        alpha_tolerance = (float(absolute_tolerance)
                           +float(relative_tolerance)
                            *max(abs(start), abs(bound), 1.0))
        if not (bound > start):
            raise PropagationCandidateNumericalError(
                "Invalid one-active-channel reaction-coordinate bound"
            )

        total = self._reaction_coordinate_clock_path(
            start, bound, sensible0, channel_index, thermo.heat,
            alpha_tolerance=alpha_tolerance,
            temperature_tolerance_K=temperature_tolerance_K,
            maximum_local_refinements=maximum_local_refinements,
        )
        total_rate_evaluations = int(total["rate_evaluations"])
        total_quadrature_evaluations = int(
            total["quadrature_panel_evaluations"]
        )
        maximum_depth = int(total["maximum_depth"])
        maximum_normalized = float(total["maximum_normalized_residual"])
        maximum_normalized_alpha = float(total["normalized_alpha_residual"])
        maximum_normalized_temperature = float(
            total["normalized_temperature_residual"]
        )
        maximum_rejected = float(
            total["maximum_rejected_normalized_residual"]
        )
        root_iterations = 0
        roundoff_time = (128.0*np.finfo(np.float64).eps
                         *max(abs(float(duration)), abs(total["clock"]),
                              np.finfo(np.float64).tiny))
        reaches_bound = (float(duration)
                         >= total["clock"]+total["clock_error"]-roundoff_time)

        if reaches_bound:
            endpoint = bound
        else:
            lower = start
            upper = bound
            endpoint = start+(bound-start)*min(
                max(float(duration)/total["clock"], 0.0),
                math.nextafter(1.0, 0.0),
            )
            for root_iterations in range(1, maximum_depletion_iterations+1):
                clock = self._reaction_coordinate_clock_path(
                    start, endpoint, sensible0, channel_index, thermo.heat,
                    alpha_tolerance=alpha_tolerance,
                    temperature_tolerance_K=temperature_tolerance_K,
                    maximum_local_refinements=maximum_local_refinements,
                )
                total_rate_evaluations += int(clock["rate_evaluations"])
                total_quadrature_evaluations += int(
                    clock["quadrature_panel_evaluations"]
                )
                maximum_depth = max(maximum_depth, int(clock["maximum_depth"]))
                maximum_normalized = max(
                    maximum_normalized,
                    float(clock["maximum_normalized_residual"]),
                )
                maximum_normalized_alpha = max(
                    maximum_normalized_alpha,
                    float(clock["normalized_alpha_residual"]),
                )
                maximum_normalized_temperature = max(
                    maximum_normalized_temperature,
                    float(clock["normalized_temperature_residual"]),
                )
                maximum_rejected = max(
                    maximum_rejected,
                    float(clock["maximum_rejected_normalized_residual"]),
                )
                residual = clock["clock"]-float(duration)
                mapped_alpha_error = (abs(residual)+clock["clock_error"])
                endpoint_temperature = float(thermo.heat.temperature(
                    float(sensible0)+float(self.Q[channel_index])
                    *(endpoint-start)
                ))
                endpoint_rate = float(_kinetic_rate(
                    np.asarray(endpoint), np.asarray(endpoint_temperature),
                    self.channels[channel_index], self.R,
                ))
                total_rate_evaluations += 1
                mapped_alpha_error *= endpoint_rate
                mapped_temperature_error = (
                    mapped_alpha_error*float(self.Q[channel_index])
                    /float(thermo.heat.cp(endpoint_temperature))
                    if self.Q[channel_index] > 0.0 else 0.0
                )
                if (mapped_alpha_error <= 0.125*alpha_tolerance
                        and mapped_temperature_error
                            <= 0.125*temperature_tolerance_K):
                    break
                if clock["clock"]+clock["clock_error"] < duration:
                    lower = endpoint
                elif clock["clock"]-clock["clock_error"] > duration:
                    upper = endpoint
                else:
                    raise _CoordinateCertificationFailure(
                        "One-active-channel clock uncertainty straddled the "
                        "target outside the endpoint state tolerances"
                    )
                bracket_temperature = np.asarray(thermo.heat.temperature(
                    float(sensible0)+float(self.Q[channel_index])
                    *(np.asarray([lower, upper])-start)
                ))
                if (upper-lower <= 0.125*alpha_tolerance
                        and abs(float(bracket_temperature[1]
                                      -bracket_temperature[0]))
                            <= 0.125*temperature_tolerance_K):
                    endpoint = lower
                    break
                newton = endpoint-residual*endpoint_rate
                margin = 0.1*(upper-lower)
                if not (lower+margin < newton < upper-margin):
                    newton = 0.5*(lower+upper)
                endpoint = newton
            else:
                raise _CoordinateCertificationFailure(
                    "One-active-channel thermochemical clock root failed at "
                    "its iteration budget"
                )

        alpha = np.asarray(alpha0, dtype=np.float64).copy()
        alpha[channel_index] = endpoint
        sensible = (float(sensible0)+float(self.Q[channel_index])
                    *(endpoint-start))
        temperature = float(thermo.heat.temperature(sensible))
        consumed = weight*(endpoint-start)
        capacity_left = max(float(progress_capacity)-consumed, 0.0)
        depleted = bool(reaches_bound and inventory_binds)
        depletion_time = total["clock"] if depleted else math.nan

        frozen = self._fixed_beta_flow_channel_one(
            start, float(duration), 1.0/max(float(temperature0), 1.0),
            channel_index,
        )
        frozen = min(frozen, bound)
        frozen_temperature = float(thermo.heat.temperature(
            float(sensible0)+float(self.Q[channel_index])*(frozen-start)
        ))
        return {
            "alpha": alpha,
            "sensible": sensible,
            "temperature": temperature,
            "capacity_left": capacity_left,
            "depleted": depleted,
            "depletion_time": depletion_time,
            "root_iterations": int(root_iterations),
            "rate_evaluations": total_rate_evaluations,
            "quadrature_panel_evaluations": total_quadrature_evaluations,
            "maximum_depth": maximum_depth,
            "maximum_normalized_residual": maximum_normalized,
            "maximum_normalized_alpha_residual": maximum_normalized_alpha,
            "maximum_normalized_temperature_residual": (
                maximum_normalized_temperature
            ),
            "maximum_rejected_normalized_residual": maximum_rejected,
            "feedback_alpha_correction": abs(endpoint-frozen),
            "feedback_temperature_correction": abs(
                temperature-frozen_temperature
            ),
            "reaches_bound": bool(reaches_bound),
        }

    def _solve_local_cells(self, alpha0, sensible0, temperature0, duration,
                           progress_capacity, thermo, *,
                           relative_tolerance, absolute_tolerance,
                           temperature_tolerance_K,
                           maximum_corrector_iterations,
                           maximum_depletion_iterations,
                           maximum_local_refinements,
                           maximum_reaction_coordinate_steps):
        """Chronological cell-local scheduler with an embedded endpoint test.

        Each attempted panel compares one full midpoint map with two half
        maps.  The fine endpoint is accepted.  Only failed cells retry that
        same chronological panel at half its size; accepted cells advance and
        may grow their panel again.  This avoids reintegrating a complete
        interval on 2**depth uniform panels because one short interval is
        thermally sharp.
        """
        count = duration.size
        work = {
            "corrector_iteration_sum": np.zeros(count, dtype=np.int64),
            "depletion_iteration_sum": np.zeros(count, dtype=np.int64),
            "endpoint_evaluations": np.zeros(count, dtype=np.int64),
            "evaluated_cell_count": np.zeros(count, dtype=np.int64),
            "maximum_corrector_iterations_used": np.zeros(count, dtype=np.int64),
            "maximum_depletion_iterations_used": np.zeros(count, dtype=np.int64),
            "maximum_normalized_corrector_residual": np.zeros(count),
            "feedback_alpha_correction": np.zeros(count),
            "feedback_temperature_correction": np.zeros(count),
            "invalid_panel_attempt_count": np.zeros(count, dtype=np.int64),
            "reaction_coordinate_rate_evaluations": np.zeros(
                count, dtype=np.int64
            ),
            "reaction_coordinate_quadrature_panel_evaluations": np.zeros(
                count, dtype=np.int64
            ),
            "reaction_coordinate_root_iterations": np.zeros(
                count, dtype=np.int64
            ),
            "reaction_coordinate_cell_count": np.zeros(count, dtype=np.int64),
            "reaction_coordinate_bound_reached_count": np.zeros(
                count, dtype=np.int64
            ),
            "maximum_reaction_coordinate_quadrature_depth": np.zeros(
                count, dtype=np.int64
            ),
            "maximum_reaction_coordinate_normalized_residual": np.zeros(count),
            "coupled_reaction_coordinate_attempt_count": np.zeros(
                count, dtype=np.int64
            ),
            "coupled_reaction_coordinate_cell_count": np.zeros(
                count, dtype=np.int64
            ),
            "coupled_reaction_coordinate_fallback_count": np.zeros(
                count, dtype=np.int64
            ),
            "coupled_reaction_coordinate_rate_evaluations": np.zeros(
                count, dtype=np.int64
            ),
            "coupled_reaction_coordinate_accepted_steps": np.zeros(
                count, dtype=np.int64
            ),
            "coupled_reaction_coordinate_rejected_steps": np.zeros(
                count, dtype=np.int64
            ),
            "coupled_reaction_coordinate_driver_switches": np.zeros(
                count, dtype=np.int64
            ),
            "maximum_coupled_reaction_coordinate_depth": np.zeros(
                count, dtype=np.int64
            ),
            "maximum_coupled_reaction_coordinate_normalized_residual": (
                np.zeros(count)
            ),
            "maximum_rejected_coupled_reaction_coordinate_residual": (
                np.zeros(count)
            ),
        }
        result = {
            "alpha": alpha0.copy(),
            "sensible": sensible0.copy(),
            "temperature": temperature0.copy(),
            "capacity_left": progress_capacity.copy(),
            "depleted": np.zeros(count, dtype=bool),
            "depletion_time": np.full(count, np.nan),
            "refinement_depth": np.zeros(count, dtype=np.int64),
            "normalized_embedded_residual": np.zeros(count),
            "normalized_embedded_alpha_residual": np.zeros(count),
            "normalized_embedded_temperature_residual": np.zeros(count),
            "maximum_rejected_embedded_residual": np.zeros(count),
            "refined": np.zeros(count, dtype=bool),
            "accepted_panel_count": np.zeros(count, dtype=np.int64),
        }
        elapsed = np.zeros(count)
        panel_size = duration.copy()
        level = np.zeros(count, dtype=np.int64)
        attempt_count = np.zeros(count, dtype=np.int64)
        # A level-r dyadic traversal has at most 2**r accepted minimum-size
        # panels.  Initial descent plus any grow/fail oscillation stays below
        # twice that accepted count plus r; 4*2**r is therefore a complete
        # scheduler bound, not a user-tunable trajectory rejection threshold.
        maximum_panel_attempts_per_cell = 4*(2**maximum_local_refinements)
        # Use the represented duration/elapsed scale, not a one-second floor:
        # a valid sub-picosecond chemistry panel must not be mistaken for a
        # zero-length remainder.
        time_tolerance = (
            256.0*np.finfo(np.float64).eps
            * np.maximum.reduce([
                np.abs(duration), np.abs(elapsed),
                np.full(duration.shape, np.finfo(np.float64).tiny),
            ])
        )
        uncompleted = np.any(
            alpha0 < 1.0-256.0*np.finfo(float).eps, axis=-1
        )
        timed = duration > time_tolerance
        unfinished_channels = (
            alpha0 < 1.0-256.0*np.finfo(float).eps
        )
        next_alpha = np.nextafter(alpha0, np.ones_like(alpha0))
        progress_quantum = np.min(np.where(
            unfinished_channels,
            self.weights[None, :]*(next_alpha-alpha0),
            np.inf,
        ), axis=-1)
        representation_exhausted = (
            (progress_capacity > 0.0)
            & (progress_capacity < progress_quantum)
        )
        initially_depleted = (
            (progress_capacity <= 0.0) & uncompleted & timed
        )
        result["depleted"][initially_depleted] = True
        result["depletion_time"][initially_depleted] = 0.0
        has_capacity = ((progress_capacity > 0.0)
                        & ~representation_exhausted)
        one_active = ((np.count_nonzero(unfinished_channels, axis=-1) == 1)
                      & has_capacity & timed)
        one_active_succeeded = np.zeros(count, dtype=bool)
        for cell in np.flatnonzero(one_active):
            channel_index = int(np.flatnonzero(unfinished_channels[cell])[0])
            try:
                fast = self._solve_one_active_reaction_coordinate_cell(
                    alpha0[cell], sensible0[cell], temperature0[cell],
                    duration[cell], progress_capacity[cell], channel_index,
                    thermo,
                    relative_tolerance=relative_tolerance,
                    absolute_tolerance=absolute_tolerance,
                    temperature_tolerance_K=temperature_tolerance_K,
                    maximum_depletion_iterations=maximum_depletion_iterations,
                    maximum_local_refinements=maximum_local_refinements,
                )
            except _CoordinateCertificationFailure:
                # The coordinate clock is only an accelerator.  Failure to
                # certify its quadrature/root must retain the conservative
                # chronological midpoint path, not reject a physical cell.
                continue
            one_active_succeeded[cell] = True
            result["alpha"][cell] = fast["alpha"]
            result["sensible"][cell] = fast["sensible"]
            result["temperature"][cell] = fast["temperature"]
            result["capacity_left"][cell] = fast["capacity_left"]
            result["depleted"][cell] = fast["depleted"]
            result["depletion_time"][cell] = fast["depletion_time"]
            result["refinement_depth"][cell] = fast["maximum_depth"]
            result["normalized_embedded_residual"][cell] = (
                fast["maximum_normalized_residual"]
            )
            result["normalized_embedded_alpha_residual"][cell] = (
                fast["maximum_normalized_alpha_residual"]
            )
            result["normalized_embedded_temperature_residual"][cell] = (
                fast["maximum_normalized_temperature_residual"]
            )
            result["maximum_rejected_embedded_residual"][cell] = (
                fast["maximum_rejected_normalized_residual"]
            )
            result["refined"][cell] = fast["maximum_depth"] > 0
            result["accepted_panel_count"][cell] = 1
            attempt_count[cell] = 1
            work["endpoint_evaluations"][cell] = (
                fast["quadrature_panel_evaluations"]
            )
            work["evaluated_cell_count"][cell] = 1
            work["feedback_alpha_correction"][cell] = (
                fast["feedback_alpha_correction"]
            )
            work["feedback_temperature_correction"][cell] = (
                fast["feedback_temperature_correction"]
            )
            work["reaction_coordinate_rate_evaluations"][cell] = (
                fast["rate_evaluations"]
            )
            work["reaction_coordinate_quadrature_panel_evaluations"][cell] = (
                fast["quadrature_panel_evaluations"]
            )
            work["reaction_coordinate_root_iterations"][cell] = (
                fast["root_iterations"]
            )
            work["reaction_coordinate_cell_count"][cell] = 1
            work["reaction_coordinate_bound_reached_count"][cell] = int(
                fast["reaches_bound"]
            )
            work["maximum_reaction_coordinate_quadrature_depth"][cell] = (
                fast["maximum_depth"]
            )
            work["maximum_reaction_coordinate_normalized_residual"][cell] = (
                fast["maximum_normalized_residual"]
            )

        active = (has_capacity & uncompleted & ~one_active_succeeded & timed)

        while np.any(active):
            indices = np.flatnonzero(active)
            remaining = duration[indices]-elapsed[indices]
            q = np.minimum(panel_size[indices], remaining)
            if np.any(q <= 0.0):
                raise PropagationCandidateNumericalError(
                    "Invalid chronological local chemistry panel"
                )
            attempt_count[indices] += 1
            if np.any(
                    attempt_count[indices] > maximum_panel_attempts_per_cell):
                raise PropagationCandidateNumericalError(
                    "Local chemistry chronological scheduler exceeded the "
                    "configured panel-attempt work bound"
                )

            panel_time_fraction = np.minimum(np.maximum(
                q/np.maximum(duration[indices], np.finfo(np.float64).tiny),
                256.0*np.finfo(np.float64).eps,
            ), 1.0)

            coarse = self._integrate_uniform_panels(
                result["alpha"][indices], result["sensible"][indices],
                result["temperature"][indices], q,
                result["capacity_left"][indices], 1, thermo,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                temperature_tolerance_K=temperature_tolerance_K,
                maximum_corrector_iterations=maximum_corrector_iterations,
                maximum_depletion_iterations=maximum_depletion_iterations,
                accuracy_fraction=panel_time_fraction,
            )
            fine = self._integrate_uniform_panels(
                result["alpha"][indices], result["sensible"][indices],
                result["temperature"][indices], q,
                result["capacity_left"][indices], 2, thermo,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                temperature_tolerance_K=temperature_tolerance_K,
                maximum_corrector_iterations=maximum_corrector_iterations,
                maximum_depletion_iterations=maximum_depletion_iterations,
                accuracy_fraction=panel_time_fraction,
            )
            self._accumulate_work(work, coarse, indices)
            self._accumulate_work(work, fine, indices)

            comparable = coarse["valid"] & fine["valid"]
            alpha_scale = np.maximum(
                np.maximum(np.abs(fine["alpha"]), np.abs(coarse["alpha"])),
                1.0,
            )
            # The coarse/fine difference is an estimator rather than a proof;
            # reserve half the global budget as the scalar/tensor safety margin.
            embedded_budget_fraction = np.maximum(
                0.5*panel_time_fraction,
                256.0*np.finfo(np.float64).eps,
            )
            alpha_residual = np.max(
                np.abs(fine["alpha"]-coarse["alpha"])
                / ((absolute_tolerance+relative_tolerance*alpha_scale)
                   *embedded_budget_fraction[:, None]), axis=-1
            )
            temperature_residual = (
                np.abs(fine["temperature"]-coarse["temperature"])
                / (temperature_tolerance_K*embedded_budget_fraction)
            )
            embedded = np.maximum(alpha_residual, temperature_residual)
            event_time_roundoff = (
                256.0*np.finfo(np.float64).eps
                *np.maximum(np.abs(q), np.finfo(np.float64).tiny)
            )
            fine_event_in_first_half = (
                fine["depleted"]
                & np.isfinite(fine["depletion_time"])
                & (fine["depletion_time"]
                   <= 0.5*q+event_time_roundoff)
            )
            # If both embedded maps deplete inside their first panel they can
            # agree with each other while sharing the same wrong one-midpoint
            # channel allocation.  Require one complete nondepleted fine
            # half-panel before an event-bearing attempt can be accepted.
            pass_local = (
                comparable & np.isfinite(embedded) & (embedded <= 1.0)
                & ~fine_event_in_first_half
            )
            passed = indices[pass_local]
            if passed.size:
                result["alpha"][passed] = fine["alpha"][pass_local]
                result["sensible"][passed] = fine["sensible"][pass_local]
                result["temperature"][passed] = fine["temperature"][pass_local]
                result["capacity_left"][passed] = fine["capacity_left"][pass_local]
                result["depleted"][passed] = fine["depleted"][pass_local]
                local_depletion = fine["depletion_time"][pass_local]
                first_event = (np.isfinite(local_depletion)
                               & ~np.isfinite(result["depletion_time"][passed]))
                if np.any(first_event):
                    event_cells = passed[first_event]
                    result["depletion_time"][event_cells] = (
                        elapsed[event_cells]+local_depletion[first_event]
                    )
                result["refinement_depth"][passed] = np.maximum(
                    result["refinement_depth"][passed], level[passed]
                )
                result["normalized_embedded_residual"][passed] = np.maximum(
                    result["normalized_embedded_residual"][passed], embedded[pass_local]
                )
                result["normalized_embedded_alpha_residual"][passed] = np.maximum(
                    result["normalized_embedded_alpha_residual"][passed],
                    alpha_residual[pass_local],
                )
                result["normalized_embedded_temperature_residual"][passed] = np.maximum(
                    result["normalized_embedded_temperature_residual"][passed],
                    temperature_residual[pass_local],
                )
                elapsed[passed] += q[pass_local]
                result["accepted_panel_count"][passed] += 1

                completed = np.all(
                    result["alpha"][passed] >= 1.0-256.0*np.finfo(float).eps,
                    axis=-1,
                )
                exhausted = (result["capacity_left"][passed] <= 0.0)
                newly_exhausted = exhausted & ~result["depleted"][passed]
                if np.any(newly_exhausted):
                    exhausted_cells = passed[newly_exhausted]
                    result["depleted"][exhausted_cells] = True
                    result["depletion_time"][exhausted_cells] = elapsed[exhausted_cells]
                finished_time = duration[passed]-elapsed[passed] <= time_tolerance[passed]
                done = completed | exhausted | finished_time
                active[passed[done]] = False

                continuing = passed[~done]
                if continuing.size:
                    local_error = embedded[pass_local][~done]
                    grow = (local_error <= 0.125) & (level[continuing] > 0)
                    if np.any(grow):
                        grow_cells = continuing[grow]
                        panel_size[grow_cells] *= 2.0
                        level[grow_cells] -= 1
                    left = duration[continuing]-elapsed[continuing]
                    panel_size[continuing] = np.minimum(
                        panel_size[continuing], left
                    )

            failed_local = ~pass_local
            failed_positions = np.flatnonzero(failed_local)
            failed = indices[failed_positions]
            if failed.size:
                finite_failed = np.where(
                    np.isfinite(embedded[failed_positions]),
                    embedded[failed_positions], 0.0,
                )
                result["maximum_rejected_embedded_residual"][failed] = np.maximum(
                    result["maximum_rejected_embedded_residual"][failed],
                    finite_failed,
                )
                coordinate_succeeded = np.zeros(failed.size, dtype=bool)
                for local_position, cell in enumerate(failed):
                    cell_panel_fraction = float(
                        panel_time_fraction[failed_positions[local_position]]
                    )
                    work["coupled_reaction_coordinate_attempt_count"][cell] += 1
                    try:
                        coordinate = self._solve_coupled_reaction_coordinate_cell(
                            result["alpha"][cell],
                            result["sensible"][cell],
                            result["temperature"][cell],
                            q[failed_positions[local_position]],
                            result["capacity_left"][cell], thermo,
                            relative_tolerance=(
                                relative_tolerance*cell_panel_fraction
                            ),
                            absolute_tolerance=(
                                absolute_tolerance*cell_panel_fraction
                            ),
                            temperature_tolerance_K=(
                                temperature_tolerance_K*cell_panel_fraction
                            ),
                            maximum_depletion_iterations=(
                                maximum_depletion_iterations
                            ),
                            maximum_local_refinements=(
                                maximum_local_refinements
                            ),
                            maximum_reaction_coordinate_steps=(
                                maximum_reaction_coordinate_steps
                            ),
                        )
                    except _CoordinateCertificationFailure:
                        work[
                            "coupled_reaction_coordinate_fallback_count"
                        ][cell] += 1
                        continue

                    coordinate_succeeded[local_position] = True
                    work["coupled_reaction_coordinate_cell_count"][cell] = 1
                    work[
                        "coupled_reaction_coordinate_rate_evaluations"
                    ][cell] += int(coordinate["coupled_rate_evaluations"])
                    work[
                        "coupled_reaction_coordinate_accepted_steps"
                    ][cell] += int(coordinate["accepted_steps"])
                    work[
                        "coupled_reaction_coordinate_rejected_steps"
                    ][cell] += int(coordinate["rejected_steps"])
                    work[
                        "coupled_reaction_coordinate_driver_switches"
                    ][cell] += int(coordinate["driver_switches"])
                    work[
                        "maximum_coupled_reaction_coordinate_depth"
                    ][cell] = max(
                        work[
                            "maximum_coupled_reaction_coordinate_depth"
                        ][cell],
                        int(coordinate["maximum_depth"]),
                    )
                    work[
                        "maximum_coupled_reaction_coordinate_normalized_residual"
                    ][cell] = max(
                        work[
                            "maximum_coupled_reaction_coordinate_normalized_residual"
                        ][cell],
                        float(coordinate["maximum_normalized_residual"]),
                    )
                    work[
                        "maximum_rejected_coupled_reaction_coordinate_residual"
                    ][cell] = max(
                        work[
                            "maximum_rejected_coupled_reaction_coordinate_residual"
                        ][cell],
                        float(coordinate[
                            "maximum_rejected_normalized_residual"
                        ]),
                    )

                    old_elapsed = elapsed[cell]
                    result["alpha"][cell] = coordinate["alpha"]
                    result["sensible"][cell] = coordinate["sensible"]
                    result["temperature"][cell] = coordinate["temperature"]
                    result["capacity_left"][cell] = coordinate["capacity_left"]
                    result["depleted"][cell] = coordinate["depleted"]
                    if coordinate["depleted"]:
                        result["depletion_time"][cell] = (
                            old_elapsed+float(coordinate["depletion_time"])
                        )
                    result["refinement_depth"][cell] = max(
                        result["refinement_depth"][cell],
                        int(coordinate["maximum_depth"]),
                    )
                    result["normalized_embedded_residual"][cell] = max(
                        result["normalized_embedded_residual"][cell],
                        float(coordinate["maximum_normalized_residual"]),
                    )
                    # The coordinate residual is a joint alpha/temperature
                    # certificate.  Reporting it in both component maxima is
                    # conservative and avoids understating either tolerance.
                    result[
                        "normalized_embedded_alpha_residual"
                    ][cell] = max(
                        result[
                            "normalized_embedded_alpha_residual"
                        ][cell],
                        float(coordinate["maximum_normalized_residual"]),
                    )
                    result[
                        "normalized_embedded_temperature_residual"
                    ][cell] = max(
                        result[
                            "normalized_embedded_temperature_residual"
                        ][cell],
                        float(coordinate["maximum_normalized_residual"]),
                    )
                    result["maximum_rejected_embedded_residual"][cell] = max(
                        result[
                            "maximum_rejected_embedded_residual"
                        ][cell],
                        float(coordinate[
                            "maximum_rejected_normalized_residual"
                        ]),
                    )
                    result["refined"][cell] = True
                    result["accepted_panel_count"][cell] += 1
                    elapsed[cell] += q[failed_positions[local_position]]

                    one_active_tail = coordinate["one_active"]
                    if one_active_tail is not None:
                        work["reaction_coordinate_cell_count"][cell] = 1
                        work[
                            "reaction_coordinate_bound_reached_count"
                        ][cell] += int(one_active_tail["reaches_bound"])
                        work[
                            "reaction_coordinate_rate_evaluations"
                        ][cell] += int(one_active_tail["rate_evaluations"])
                        work[
                            "reaction_coordinate_quadrature_panel_evaluations"
                        ][cell] += int(one_active_tail[
                            "quadrature_panel_evaluations"
                        ])
                        work[
                            "reaction_coordinate_root_iterations"
                        ][cell] += int(one_active_tail["root_iterations"])
                        work[
                            "maximum_reaction_coordinate_quadrature_depth"
                        ][cell] = max(
                            work[
                                "maximum_reaction_coordinate_quadrature_depth"
                            ][cell],
                            int(one_active_tail["maximum_depth"]),
                        )
                        work[
                            "maximum_reaction_coordinate_normalized_residual"
                        ][cell] = max(
                            work[
                                "maximum_reaction_coordinate_normalized_residual"
                            ][cell],
                            float(one_active_tail[
                                "maximum_normalized_residual"
                            ]),
                        )

                    completed = np.all(
                        result["alpha"][cell]
                        >= 1.0-256.0*np.finfo(float).eps
                    )
                    exhausted = result["capacity_left"][cell] <= 0.0
                    if exhausted and not result["depleted"][cell] and not completed:
                        result["depleted"][cell] = True
                        result["depletion_time"][cell] = elapsed[cell]
                    finished_time = (
                        duration[cell]-elapsed[cell] <= time_tolerance[cell]
                    )
                    if completed or exhausted or finished_time:
                        active[cell] = False
                    else:
                        left = duration[cell]-elapsed[cell]
                        panel_size[cell] = min(panel_size[cell], left)

                fallback_failed = failed[~coordinate_succeeded]
                if fallback_failed.size:
                    fallback_positions = failed_positions[
                        ~coordinate_succeeded
                    ]
                    result["refined"][fallback_failed] = True
                    level[fallback_failed] += 1
                    result["refinement_depth"][fallback_failed] = np.maximum(
                        result["refinement_depth"][fallback_failed],
                        level[fallback_failed],
                    )
                    if np.any(
                            level[fallback_failed]
                            > maximum_local_refinements):
                        raise PropagationCandidateNumericalError(
                            "Local thermochemical embedded endpoint failed "
                            "at its refinement budget"
                        )
                    next_panel_size = 0.5*panel_size[fallback_failed]
                    event_limited = fine_event_in_first_half[
                        fallback_positions
                    ]
                    if np.any(event_limited):
                        event_cells = fallback_failed[event_limited]
                        event_times = fine["depletion_time"][
                            fallback_positions[event_limited]
                        ]
                        minimum_increment = (
                            np.nextafter(
                                elapsed[event_cells], np.inf
                            )-elapsed[event_cells]
                        )
                        pre_event_size = np.maximum(
                            0.5*event_times, minimum_increment
                        )
                        next_panel_size[event_limited] = np.minimum(
                            next_panel_size[event_limited], pre_event_size
                        )
                    if np.any(next_panel_size <= 0.0):
                        raise PropagationCandidateNumericalError(
                            "Local chemistry event-aware refinement made no "
                            "representable time progress"
                        )
                    panel_size[fallback_failed] = next_panel_size

        result.update(work)
        result["panel_attempt_count"] = attempt_count
        return result

    def advance_local(self, U, thermo, dt, *, concentration_floor,
                      relative_tolerance, absolute_tolerance,
                      temperature_tolerance_K, maximum_corrector_iterations,
                      maximum_depletion_iterations,
                      maximum_local_refinements,
                      maximum_reaction_coordinate_steps=64):
        """Advance uncapped local chemistry with thermochemical feedback.

        This is a finite-time conservative state map, not an RHS.  It uses the
        exact fixed-inverse-temperature flow of the supplied kinetic tables,
        a reciprocal-temperature midpoint corrector, cell-local refinement,
        and a shared inventory-depletion event.  ``maximum_rate_per_s`` is
        diagnostic only and cannot affect the returned state.
        """
        U = np.asarray(U, dtype=np.float64)
        if U.ndim < 2 or U.shape[-1] != NCONS or not np.isfinite(U).all():
            raise PropagationCandidateNumericalError(
                "Invalid conservative state for local chemistry"
            )
        if not math.isfinite(float(dt)) or float(dt) < 0.0:
            raise PropagationCandidateNumericalError("Invalid local chemistry timestep")
        numeric = (relative_tolerance, absolute_tolerance,
                   temperature_tolerance_K, concentration_floor)
        if (not all(math.isfinite(float(v)) for v in numeric)
                or relative_tolerance < 0.0 or absolute_tolerance <= 0.0
                or temperature_tolerance_K <= 0.0 or concentration_floor < 0.0):
            raise PropagationConfigurationError(
                "Invalid local chemistry tolerance or concentration floor"
            )
        maximum_corrector_iterations = int(maximum_corrector_iterations)
        maximum_depletion_iterations = int(maximum_depletion_iterations)
        maximum_local_refinements = int(maximum_local_refinements)
        maximum_reaction_coordinate_steps = int(
            maximum_reaction_coordinate_steps
        )
        if (maximum_corrector_iterations < 1
                or maximum_depletion_iterations < 1
                or maximum_local_refinements < 1
                or maximum_local_refinements > 20
                or maximum_reaction_coordinate_steps < 1
                or maximum_reaction_coordinate_steps > 1_000_000):
            raise PropagationConfigurationError(
                "Invalid local chemistry iteration/refinement budget"
            )
        if not math.isfinite(self.xi_per_kg) or self.xi_per_kg <= 0.0:
            raise PropagationConfigurationError(
                "Local chemistry requires positive reaction extent per mass"
            )
        if np.any(self.weights <= 0.0):
            raise PropagationConfigurationError(
                "local_adaptive_thermochemical requires strictly positive "
                "mass_conversion_weights; legacy chemistry remains available "
                "for zero-weight channel models"
            )

        spatial_shape = U.shape[:-1]
        flat = U.reshape(-1, NCONS)
        rho = flat[:, RHO]
        machine = np.finfo(np.float64).eps
        if np.any(rho <= 0.0):
            raise PropagationCandidateNumericalError(
                "Local chemistry requires positive density"
            )
        alpha0 = flat[:, A1:A2+1]/rho[:, None]
        # Match the primitive-state admissibility contract used by every
        # nonchemical SSPRK stage.  Conservative transport can leave a trace
        # scalar a few ulps below zero; accepting that trace without clipping
        # preserves both the transported state and its linear inventories.
        alpha_input_roundoff = np.maximum(
            256.0*machine*np.maximum(np.abs(alpha0), 1.0), 1.0e-12
        )
        inventory0 = flat[:, [CATION, ANION, PVA]]
        inventory_input_roundoff = np.maximum(
            256.0*machine*np.maximum(np.abs(inventory0), 1.0),
            1.0e-12*rho[:, None],
        )
        if (np.any(alpha0 < -alpha_input_roundoff)
                or np.any(alpha0 > 1.0+alpha_input_roundoff)
                or np.any(inventory0 < -inventory_input_roundoff)):
            raise PropagationCandidateNumericalError(
                "Invalid conversion or inventory entering local chemistry"
            )
        alpha0 = np.minimum(np.maximum(alpha0, 0.0), 1.0)

        velocity_x = flat[:, MX]/rho
        velocity_y = flat[:, MY]/rho
        internal0 = (flat[:, ENERGY]/rho
                     - 0.5*(velocity_x*velocity_x+velocity_y*velocity_y))
        sensible0 = internal0-thermo.cold_energy(rho)
        temperature0 = thermo.heat.temperature(sensible0)
        temperature_input_roundoff = 256.0*machine*np.maximum(
            np.abs(temperature0), 1.0
        )
        if (not np.isfinite(temperature0).all()
                or np.any(temperature0
                          < thermo.tmin-temperature_input_roundoff)
                or np.any(temperature0
                          > thermo.tmax+temperature_input_roundoff)):
            raise PropagationCandidateNumericalError(
                "Invalid initial temperature for local chemistry"
            )

        salt = 0.5*(flat[:, CATION]+flat[:, ANION])
        maximum_extent = np.minimum.reduce([
            np.maximum(salt-float(concentration_floor), 0.0)/1.45,
            np.maximum(flat[:, PVA], 0.0),
            np.maximum(flat[:, CATION], 0.0)/1.45,
            np.maximum(flat[:, ANION], 0.0)/1.45,
        ])
        progress_capacity = maximum_extent/(rho*self.xi_per_kg)
        duration = np.full(rho.shape, float(dt), dtype=np.float64)
        solved = self._solve_local_cells(
            alpha0, sensible0, temperature0, duration, progress_capacity,
            thermo,
            relative_tolerance=float(relative_tolerance),
            absolute_tolerance=float(absolute_tolerance),
            temperature_tolerance_K=float(temperature_tolerance_K),
            maximum_corrector_iterations=maximum_corrector_iterations,
            maximum_depletion_iterations=maximum_depletion_iterations,
            maximum_local_refinements=maximum_local_refinements,
            maximum_reaction_coordinate_steps=(
                maximum_reaction_coordinate_steps
            ),
        )

        delta = solved["alpha"]-alpha0
        # Only representation-scale negative increments may be cleaned up.
        # Configured accuracy tolerances are not physical limiters and must not
        # silently turn a materially decreasing conversion into a zero source.
        alpha_roundoff = 256.0*np.finfo(np.float64).eps*np.maximum(
            np.maximum(np.abs(solved["alpha"]), np.abs(alpha0)), 1.0
        )
        if (not np.isfinite(delta).all()
                or np.any(delta < -alpha_roundoff)
                or np.any(solved["alpha"] > 1.0+alpha_roundoff)):
            raise PropagationCandidateNumericalError(
                "Local chemistry produced invalid conversion"
            )
        delta = np.maximum(delta, 0.0)
        d_rho_alpha = rho[:, None]*delta
        energy_increment = d_rho_alpha @ self.Q
        extent_increment = self.xi_per_kg*(d_rho_alpha @ self.weights)

        after = flat.copy()
        after[:, A1:A2+1] += d_rho_alpha
        after[:, ENERGY] += energy_increment
        after[:, CATION] -= 1.45*extent_increment
        after[:, ANION] -= 1.45*extent_increment
        after[:, PVA] -= extent_increment
        after[:, PRODUCT_WATER] += 2.0*extent_increment

        alpha_after = after[:, A1:A2+1]/rho[:, None]
        final_sensible = sensible0+(delta @ self.Q)
        final_temperature = thermo.heat.temperature(final_sensible)
        inventory_after = after[:, [CATION, ANION, PVA]]
        inventory_output_roundoff = np.maximum(
            256.0*machine*np.maximum(
                np.maximum(np.abs(inventory_after), np.abs(inventory0)), 1.0
            ),
            1.0e-12*rho[:, None],
        )
        temperature_output_roundoff = 256.0*machine*np.maximum(
            np.maximum(np.abs(final_temperature), np.abs(temperature0)), 1.0
        )
        if (not np.isfinite(after).all() or not np.isfinite(final_temperature).all()
                or np.any(alpha_after < -np.maximum(alpha_roundoff, 1.0e-12))
                or np.any(alpha_after > 1.0+np.maximum(
                    alpha_roundoff, 1.0e-12
                ))
                or np.any(inventory_after < -inventory_output_roundoff)
                or np.any(final_temperature
                          < thermo.tmin-temperature_output_roundoff)
                or np.any(final_temperature
                          > thermo.tmax+temperature_output_roundoff)):
            raise PropagationCandidateNumericalError(
                "Nonphysical local thermochemical endpoint"
            )

        heat_residual = (
            after[:, ENERGY]-flat[:, ENERGY]-energy_increment
        )
        event_residual = np.zeros_like(rho)
        depleted = solved["depleted"]
        if np.any(depleted):
            event_residual[depleted] = (
                extent_increment[depleted]-maximum_extent[depleted]
            )
        allowed_event_error = rho*self.xi_per_kg*self._progress_tolerance(
            progress_capacity,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        if (np.any(np.abs(heat_residual)
                   > 64.0*np.finfo(float).eps*np.maximum(
                       np.abs(after[:, ENERGY]), 1.0))
                or np.any(np.abs(event_residual) > allowed_event_error)):
            raise PropagationCandidateNumericalError(
                "Local chemistry conservative/event closure failed"
            )

        initial_rates = self._rates_from_alpha(alpha0, temperature0)
        endpoint_rates = self._rates_from_alpha(alpha_after, final_temperature)
        if (not np.isfinite(initial_rates).all()
                or not np.isfinite(endpoint_rates).all()):
            raise PropagationCandidateNumericalError(
                "Nonfinite local chemistry diagnostic rate"
            )
        initial_exceedance = float(np.mean(
            np.any(initial_rates > self.maximum_rate, axis=-1)
        ))
        endpoint_exceedance = float(np.mean(
            np.any(endpoint_rates > self.maximum_rate, axis=-1)
        ))
        traversed_rate_upper_bound = self._traversed_rate_upper_bound(
            alpha0, alpha_after, final_temperature
        )
        traversed_exceedance = float(np.mean(
            traversed_rate_upper_bound > self.maximum_rate
        ))
        finite_depletion = np.isfinite(solved["depletion_time"])
        depletion_fractions = (
            solved["depletion_time"][finite_depletion]/float(dt)
        )
        caloric_residual = (
            thermo.heat.sensible_energy(final_temperature)-final_sensible
        )
        diagnostics = {
            "local_chemistry_method": (
                "bulk_midpoint_with_dominant_driver_reaction_coordinate_and_"
                "error_controlled_midpoint_fallback"
            ),
            "coupled_reaction_coordinate_rule": "dormand_prince_5_4",
            "coupled_reaction_coordinate_safety_factor": 2.0,
            "coupled_reaction_coordinate_work_counter_scope": (
                "rate_step_depth_residual_counters_are_successful_endpoint_"
                "only; attempt_and_fallback_counts_cover_all_attempts; "
                "fallback_work_is_included_in_chemistry_wall_clock"
            ),
            "coupled_reaction_coordinate_attempt_count": int(np.sum(
                solved["coupled_reaction_coordinate_attempt_count"]
            )),
            "coupled_reaction_coordinate_cell_count": int(np.sum(
                solved["coupled_reaction_coordinate_cell_count"]
            )),
            "coupled_reaction_coordinate_cell_fraction": float(np.mean(
                solved["coupled_reaction_coordinate_cell_count"]
            )),
            "coupled_reaction_coordinate_fallback_count": int(np.sum(
                solved["coupled_reaction_coordinate_fallback_count"]
            )),
            "coupled_reaction_coordinate_rate_evaluation_count": int(np.sum(
                solved["coupled_reaction_coordinate_rate_evaluations"]
            )),
            "coupled_reaction_coordinate_accepted_step_count": int(np.sum(
                solved["coupled_reaction_coordinate_accepted_steps"]
            )),
            "coupled_reaction_coordinate_rejected_step_count": int(np.sum(
                solved["coupled_reaction_coordinate_rejected_steps"]
            )),
            "coupled_reaction_coordinate_driver_switch_count": int(np.sum(
                solved["coupled_reaction_coordinate_driver_switches"]
            )),
            "maximum_coupled_reaction_coordinate_refinement_depth": int(
                np.max(solved["maximum_coupled_reaction_coordinate_depth"])
            ),
            "maximum_coupled_reaction_coordinate_normalized_residual": float(
                np.max(solved[
                    "maximum_coupled_reaction_coordinate_normalized_residual"
                ])
            ),
            "maximum_rejected_coupled_reaction_coordinate_residual": float(
                np.max(solved[
                    "maximum_rejected_coupled_reaction_coordinate_residual"
                ])
            ),
            "one_active_reaction_coordinate_quadrature_rule": (
                "gauss_legendre_8_16"
            ),
            "one_active_reaction_coordinate_work_counter_scope": (
                "successful_coordinate_endpoints_only; fallback_work_is_"
                "included_in_chemistry_wall_clock"
            ),
            "one_active_reaction_coordinate_cell_count": int(np.sum(
                solved["reaction_coordinate_cell_count"]
            )),
            "one_active_reaction_coordinate_cell_fraction": float(np.mean(
                solved["reaction_coordinate_cell_count"]
            )),
            "one_active_reaction_coordinate_bound_reached_count": int(np.sum(
                solved["reaction_coordinate_bound_reached_count"]
            )),
            "one_active_reaction_coordinate_rate_evaluation_count": int(np.sum(
                solved["reaction_coordinate_rate_evaluations"]
            )),
            "one_active_reaction_coordinate_quadrature_panel_evaluation_count": int(np.sum(
                solved["reaction_coordinate_quadrature_panel_evaluations"]
            )),
            "one_active_reaction_coordinate_root_iteration_count": int(np.sum(
                solved["reaction_coordinate_root_iterations"]
            )),
            "maximum_one_active_reaction_coordinate_root_iterations": int(np.max(
                solved["reaction_coordinate_root_iterations"]
            )),
            "maximum_one_active_reaction_coordinate_quadrature_refinement_depth": int(np.max(
                solved["maximum_reaction_coordinate_quadrature_depth"]
            )),
            "maximum_one_active_reaction_coordinate_normalized_quadrature_residual": float(np.max(
                solved["maximum_reaction_coordinate_normalized_residual"]
            )),
            "chemical_rate_threshold_diagnostic_only": True,
            "chemical_rate_threshold_exceedance_fraction": max(
                initial_exceedance, endpoint_exceedance, traversed_exceedance
            ),
            "chemical_rate_threshold_exceedance_fraction_initial": initial_exceedance,
            "chemical_rate_threshold_exceedance_fraction_endpoint": endpoint_exceedance,
            "chemical_rate_threshold_exceedance_fraction_traversed_upper_bound": traversed_exceedance,
            "maximum_raw_rate_per_s_initial": float(np.max(initial_rates)),
            "maximum_raw_rate_per_s_endpoint": float(np.max(endpoint_rates)),
            "maximum_raw_rate_per_s_traversed_upper_bound": float(np.max(
                traversed_rate_upper_bound
            )),
            "maximum_raw_rate_per_s": float(max(
                np.max(initial_rates), np.max(endpoint_rates),
                np.max(traversed_rate_upper_bound),
            )),
            "raw_rate_observation_scope": (
                "chemistry_half_step_initial_endpoint_and_monotone_temperature_"
                "traversed_knot_upper_bound"
            ),
            "inventory_depleted_fraction": float(np.mean(depleted)),
            "inventory_limiter_fraction": float(np.mean(depleted)),
            "minimum_inventory_depletion_time_fraction": (
                float(np.min(depletion_fractions))
                if depletion_fractions.size else None
            ),
            "maximum_inventory_depletion_time_fraction": (
                float(np.max(depletion_fractions))
                if depletion_fractions.size else None
            ),
            "maximum_corrector_iterations_used": int(np.max(
                solved["maximum_corrector_iterations_used"]
            )),
            "maximum_depletion_iterations_used": int(np.max(
                solved["maximum_depletion_iterations_used"]
            )),
            "corrector_iteration_sum": int(np.sum(
                solved["corrector_iteration_sum"]
            )),
            "depletion_iteration_sum": int(np.sum(
                solved["depletion_iteration_sum"]
            )),
            "endpoint_evaluation_count": int(np.sum(
                solved["endpoint_evaluations"]
            )),
            "invalid_panel_attempt_count": int(np.sum(
                solved["invalid_panel_attempt_count"]
            )),
            "panel_attempt_count": int(np.sum(solved["panel_attempt_count"])),
            "accepted_panel_count": int(np.sum(solved["accepted_panel_count"])),
            "maximum_panel_attempts_per_cell": int(np.max(
                solved["panel_attempt_count"]
            )),
            "maximum_accepted_panels_per_cell": int(np.max(
                solved["accepted_panel_count"]
            )),
            "evaluated_cell_count": int(np.sum(
                solved["evaluated_cell_count"]
            )),
            "mean_corrector_iterations_used": float(
                np.sum(solved["corrector_iteration_sum"])
                / max(int(np.sum(solved["evaluated_cell_count"])), 1)
            ),
            "mean_endpoint_evaluations_per_evaluated_cell": float(
                np.sum(solved["endpoint_evaluations"])
                / max(int(np.sum(solved["evaluated_cell_count"])), 1)
            ),
            "maximum_local_refinement_depth": int(np.max(
                solved["refinement_depth"]
            )),
            "local_refined_cell_fraction": float(np.mean(solved["refined"])),
            "maximum_normalized_embedded_residual": float(np.max(
                solved["normalized_embedded_residual"]
            )),
            "maximum_normalized_embedded_alpha_residual": float(np.max(
                solved["normalized_embedded_alpha_residual"]
            )),
            "maximum_normalized_embedded_temperature_residual": float(np.max(
                solved["normalized_embedded_temperature_residual"]
            )),
            "maximum_rejected_normalized_embedded_residual": float(np.max(
                solved["maximum_rejected_embedded_residual"]
            )),
            "maximum_normalized_corrector_residual": float(np.max(
                solved["maximum_normalized_corrector_residual"]
            )),
            "maximum_temperature_feedback_alpha_correction": float(np.max(
                solved["feedback_alpha_correction"]
            )),
            "maximum_temperature_feedback_endpoint_temperature_correction_K": float(np.max(
                solved["feedback_temperature_correction"]
            )),
            "maximum_temperature_rise_K": float(np.max(
                final_temperature-temperature0
            )),
            "maximum_caloric_inverse_residual_J_per_kg": float(np.max(
                np.abs(caloric_residual)
            )),
            "maximum_heat_closure_residual_J_per_m3": float(np.max(
                np.abs(heat_residual)
            )),
            "maximum_inventory_event_residual_mol_per_m3": float(np.max(
                np.abs(event_residual)
            )),
            "nonconverged_cell_count": 0,
        }
        return after.reshape(U.shape), diagnostics

    def source(self, U, temperature, dt, *, available=None,
               concentration_floor=0.0, allowed_rate_cap_fraction=0.0,
               integration_mode="legacy_cap", maximum_channel_increment=0.02,
               maximum_subcycles=4096):
        """Return a conservative chemistry source over ``dt``.

        ``legacy_cap`` preserves the old post-onset contract.  The former
        explicit ``subcycle_raw`` path is retired: uncapped kinetics are now
        available only through :meth:`advance_local`, where chemistry has its
        own accuracy/work controls and is split from the PDE timestep.
        """
        mode = str(integration_mode).lower()
        if mode == "subcycle_raw":
            raise PropagationConfigurationError(
                "subcycle_raw is retired; use the local adaptive "
                "thermochemical state map"
            )
        if mode != "legacy_cap":
            raise PropagationConfigurationError(
                "source chemistry_integration_mode must be legacy_cap"
            )
        if (not math.isfinite(float(dt))) or float(dt) <= 0.0:
            raise PropagationCandidateNumericalError("Invalid chemistry timestep")
        if (not math.isfinite(float(maximum_channel_increment))
                or not 0.0 < float(maximum_channel_increment) <= 1.0):
            raise PropagationConfigurationError("Invalid maximum_channel_increment")
        rho = U[...,RHO]
        alpha0 = U[...,A1:A2+1] / rho[...,None]
        rates0 = self.raw_rates(U, temperature)
        if not np.isfinite(rates0).all() or np.any(rates0 < 0.0):
            raise PropagationCandidateNumericalError("Nonfinite/negative BC raw chemistry rate")

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
