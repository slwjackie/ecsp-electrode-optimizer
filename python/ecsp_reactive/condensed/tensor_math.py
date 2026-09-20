"""Device-resident FP64 operators for batched, two-channel condensed Euler.

All arrays have shape [case, y, x, conserved_component]. This is an arithmetic
port of the NumPy reference, not a learned surrogate or a reduced chemistry.
No .item() or .numpy() extraction occurs in the numerical kernels. Dynamic
cell compaction uses Torch index tensors; the caller transfers only a small
per-case acceptance/report table after each complete trial.
"""
from __future__ import annotations
import math
import time
import numpy as np
import torch
from .chemistry import (RHO, MX, MY, ENERGY, A1, A2, CATION, ANION, WATER,
                        PVA, PRODUCT_WATER, EC_LP, NCONS, _GL8_NODES,
                        _GL8_WEIGHTS, _GL16_NODES, _GL16_WEIGHTS,
                        _DP54_C, _DP54_A, _DP54_B5, _DP54_B4)
from ecsp_nsga2.propagation import PropagationConfigurationError, _table_property

SPATIAL = (1, 2)
RECORD_NAMES = (
    "time_after_onset_s", "unreacted_area_fraction", "mean_global_progress", "maximum_temperature_K",
    "maximum_speed_m_per_s", "minimum_pressure_Pa", "maximum_pressure_Pa", "mass_kg", "total_energy_J",
    "chemical_reservoir_J", "mass_budget_residual_kg", "mass_budget_relative_residual", "energy_budget_residual_J",
    "energy_budget_relative_residual", "energy_plus_chemical_budget_residual_J", "joule_energy_J",
    "electrochemical_energy_J", "chemical_energy_J", "heat_loss_J", "net_conduction_energy_J",
    "boundary_mass_in_kg", "boundary_energy_in_J", "maximum_local_inventory_residual_mol_per_m3",
    "integrated_lp_inventory_residual_mol",
)

# Per-half-step device diagnostics.  The tensor runner transfers these small
# [case, half-step, field] tables only after a complete trial; evolving fields
# and all local corrector/refinement decisions remain in Torch FP64.
LOCAL_CHEMISTRY_DIAGNOSTIC_NAMES = (
    "attempted", "converged", "chemical_rate_threshold_exceedance_fraction",
    "inventory_limiter_fraction", "maximum_raw_rate_per_s",
    "maximum_corrector_iterations_used", "maximum_depletion_iterations_used",
    "maximum_local_refinement_depth", "maximum_normalized_corrector_residual",
    "maximum_normalized_embedded_residual",
    "maximum_normalized_embedded_alpha_residual",
    "maximum_normalized_embedded_temperature_residual",
    "maximum_rejected_normalized_embedded_residual",
    "maximum_temperature_feedback_alpha_correction",
    "maximum_temperature_feedback_endpoint_temperature_correction_K",
    "maximum_temperature_rise_K", "corrector_iteration_sum",
    "depletion_iteration_sum", "endpoint_evaluation_count",
    "evaluated_cell_count", "local_refined_cell_fraction",
    "maximum_raw_rate_per_s_initial", "maximum_raw_rate_per_s_endpoint",
    "minimum_inventory_depletion_time_fraction",
    "maximum_inventory_depletion_time_fraction",
    "maximum_caloric_inverse_residual_J_per_kg",
    "maximum_heat_closure_residual_J_per_m3",
    "maximum_inventory_event_residual_mol_per_m3",
    "nonconverged_cell_count", "invalid_panel_attempt_count",
    "panel_attempt_count", "accepted_panel_count",
    "maximum_panel_attempts_per_cell", "maximum_accepted_panels_per_cell",
    "chemical_rate_threshold_exceedance_fraction_initial",
    "chemical_rate_threshold_exceedance_fraction_endpoint",
    "chemical_rate_threshold_exceedance_fraction_traversed_upper_bound",
    "maximum_raw_rate_per_s_traversed_upper_bound",
    "one_active_reaction_coordinate_cell_count",
    "one_active_reaction_coordinate_cell_fraction",
    "one_active_reaction_coordinate_rate_evaluation_count",
    "one_active_reaction_coordinate_quadrature_panel_evaluation_count",
    "one_active_reaction_coordinate_root_iteration_count",
    "maximum_one_active_reaction_coordinate_root_iterations",
    "maximum_one_active_reaction_coordinate_quadrature_refinement_depth",
    "maximum_one_active_reaction_coordinate_normalized_quadrature_residual",
    "coupled_reaction_coordinate_attempt_count",
    "coupled_reaction_coordinate_cell_count",
    "coupled_reaction_coordinate_cell_fraction",
    "coupled_reaction_coordinate_fallback_count",
    "coupled_reaction_coordinate_rate_evaluation_count",
    "coupled_reaction_coordinate_accepted_step_count",
    "coupled_reaction_coordinate_rejected_step_count",
    "coupled_reaction_coordinate_driver_switch_count",
    "maximum_coupled_reaction_coordinate_refinement_depth",
    "maximum_coupled_reaction_coordinate_normalized_residual",
    "maximum_rejected_coupled_reaction_coordinate_residual",
    "configured_model_temperature_range_exceeded",
    "configured_maximum_temperature_K", "observed_maximum_temperature_K",
    "offending_temperature_cell_index", "offending_temperature_roundoff_K",
)


def interp(x: torch.Tensor, grid: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Piecewise linear, with constant extrapolation, matching np.interp."""
    xc = torch.clamp(x, min=grid[0], max=grid[-1])
    i = torch.clamp(torch.searchsorted(grid, xc.contiguous(), right=True)-1, 0, grid.numel()-2)
    return values[i] + (xc-grid[i])*(values[i+1]-values[i])/(grid[i+1]-grid[i])


def harmonic(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    lo, hi = torch.minimum(a,b), torch.maximum(a,b)
    return torch.where(hi>0, 2*lo/(1+lo/torch.clamp(hi,min=1e-300)), torch.zeros_like(lo))


def conduction_and_diagonal(T: torch.Tensor, k: torch.Tensor, dx: float):
    """No-flux FV Fourier operator AND its actual positive row sum [W/m3/K]."""
    kx,ky = harmonic(k[:,:,:-1],k[:,:,1:]),harmonic(k[:,:-1,:],k[:,1:,:])
    fx=kx*(T[:,:,1:]-T[:,:,:-1])/dx**2
    fy=ky*(T[:,1:,:]-T[:,:-1,:])/dx**2
    zc=torch.zeros_like(T[:,:,:1]);zr=torch.zeros_like(T[:,:1,:])
    q=(torch.cat((fx,zc),2)-torch.cat((zc,fx),2)
       +torch.cat((fy,zr),1)-torch.cat((zr,fy),1))
    diagonal=(torch.cat((kx,zc),2)+torch.cat((zc,kx),2)
              +torch.cat((ky,zr),1)+torch.cat((zr,ky),1))/dx**2
    return q,diagonal


class TensorCondensedKernel:
    """Static model data + pure tensor RHS/trial/diagnostic functions.

    Only the runner chooses CPU versus CUDA. All physical coefficients come
    from the already-validated BC handoff and resolved configuration.
    """
    def __init__(self, solvers, device):
        self.device=torch.device(device)
        self.B=len(solvers); s=solvers[0]; self.s=s
        self.ny,self.nx=s.U.shape[:2]
        self.dx=s.a.dx;self.vol=s.vol;self.depth=s.a.thickness
        self.boundary=s.boundary;self.kind=s.riemann;self.epsilon=s.epsilon
        self.fast=s.fast and self.kind=="hllc"
        self.stationary_all=bool(self.fast and all(np.all(v.U[...,1:3]==0) and np.all(v.U[...,0]==v.U[0,0,0]) for v in solvers))
        self.electrical=None
        # Local-mode phase timing is read only after the runner's existing
        # packed diagnostic transfer has synchronized the trial.  CUDA events
        # therefore add no chemistry-loop host synchronization.
        self.trial_chemistry_wall_clock_time_s=0.0
        self.trial_nonchemical_wall_clock_time_s=0.0
        self.trial_cuda_phase_events=None
        self._local_phase_events=(
            tuple(torch.cuda.Event(enable_timing=True) for _ in range(6))
            if (self.device.type=='cuda'
                and s.chemistry_integration_mode
                    =="local_adaptive_thermochemical") else None
        )
        self.A,self.B_tait,self.N,self.rho0=s.thermo.A,s.thermo.B,s.thermo.N,s.thermo.rho0
        self.tmin,self.tmax=s.thermo.tmin,s.thermo.tmax
        self.R=s.chem.R;self.weights=self.tensor(s.chem.weights);self.Q=self.tensor(s.chem.Q)
        self.coordinate_channels_identical=bool(
            s.chem._coordinate_channels_identical)
        if (s.chemistry_integration_mode=="local_adaptive_thermochemical"
                and np.any(np.asarray(s.chem.weights,dtype=float)<=0.0)):
            raise PropagationConfigurationError(
                "local_adaptive_thermochemical requires strictly positive "
                "mass_conversion_weights; legacy chemistry remains available "
                "for zero-weight channel models"
            )
        self.maxrate=s.chem.maximum_rate;self.allowed_cap=s.allowed_cap;self.floor=s.concentration_floor
        self.heat_grid=self.tensor(s.thermo.heat.grid);self.heat_values=self.tensor(s.thermo.heat.values)
        self.heat_slopes=self.tensor(s.thermo.heat.slopes);self.heat_integrals=self.tensor(s.thermo.heat.integrals)
        self.heat_offset=s.thermo.heat.offset
        self.channels=[(self.tensor(ch['alpha_grid']), self.tensor(ch['activation_energy_J_per_mol']),
                        self.tensor(ch['ln_Af_per_s'])) for ch in s.chem.channels]
        self.kinetic_tables=[]
        for grid,activation,log_prefactor in self.channels:
            width=grid[1:]-grid[:-1]
            self.kinetic_tables.append((
                grid,activation,log_prefactor,
                (activation[1:]-activation[:-1])/width,
                (log_prefactor[1:]-log_prefactor[:-1])/width,
            ))
        self.gl8_nodes=self.tensor(_GL8_NODES)
        self.gl8_weights=self.tensor(_GL8_WEIGHTS)
        self.gl16_nodes=self.tensor(_GL16_NODES)
        self.gl16_weights=self.tensor(_GL16_WEIGHTS)
        self.dp54_b5=self.tensor(_DP54_B5)
        self.dp54_b4=self.tensor(_DP54_B4)
        self.heat_knot_sensible=self.heat_integrals-self.heat_offset
        raw=s.thermal['thermal_conductivity']
        # Validate table/law on the CPU once, not inside every GPU stage.
        _table_property(np.asarray([s.ambient]),raw,self.R)
        if isinstance(raw,(float,int)):
            self.kmode='constant';self.kvalue=float(raw)
        else:
            self.kmode=str(raw.get('mode','table')).lower()
            if self.kmode=='constant': self.kvalue=float(raw['value'])
            elif self.kmode=='table':
                self.kgrid=self.tensor(raw['temperature_K']);self.kvalues=self.tensor(raw['values'])
            else:
                self.kvalue=float(raw['reference_value']);self.ktref=float(raw['reference_temperature_K'])
                self.kea=float(raw.get('activation_energy_J_per_mol',0.))
        self.xi=self.tensor([v.chem.xi_per_kg for v in solvers])[:,None,None]
        self.lp0=self.tensor([v.chem.initial_lp_per_kg for v in solvers])[:,None,None]
        self.pva0=self.tensor([v.chem.initial_pva_per_kg for v in solvers])[:,None,None]
        self.m_lp=s.chem.m_lp;self.m_pva=s.chem.m_pva
        self.contacts=self.tensor(np.stack([v.a.anode_mask|v.a.cathode_mask for v in solvers]),dtype=torch.bool)
        self.initial=self.tensor(np.stack([v.U for v in solvers]))
        self.initial_sums=self.initial.sum(SPATIAL)*self.vol
        self.reservoir0=self.reservoir(self.initial).sum(SPATIAL)*self.vol

    def tensor(self, v, dtype=torch.float64):
        return torch.as_tensor(np.asarray(v),dtype=dtype,device=self.device).clone()

    def progress_tolerance(self,reference,accuracy_fraction=None):
        """Numerical event allowance in weighted-conversion units."""
        if accuracy_fraction is None:
            accuracy_fraction=torch.ones_like(reference)
        magnitude=abs(reference)
        allowance=accuracy_fraction*(
            self.s.chemistry_absolute_tolerance*torch.amin(self.weights)
            +self.s.chemistry_relative_tolerance*magnitude)
        ulp=torch.nextafter(
            magnitude,torch.full_like(magnitude,math.inf))-magnitude
        combined=torch.maximum(allowance,256*ulp)
        return torch.where(magnitude>0,
                           torch.minimum(combined,.5*magnitude),
                           torch.zeros_like(combined))

    def pressure(self,rho):
        return self.A+self.B_tait*torch.expm1(self.N*torch.log(rho/self.rho0))

    def sound(self,rho):
        return torch.sqrt(self.B_tait*self.N/self.rho0*(rho/self.rho0)**(self.N-1))

    def cold(self,rho):
        lr=torch.log(rho/self.rho0)
        term=lr if self.N==1. else torch.expm1((self.N-1)*lr)/(self.N-1)
        return self.B_tait/self.rho0*term+(self.A-self.B_tait)/self.rho0*(-torch.expm1(-lr))

    def sensible(self,T):
        g,v,H,m=self.heat_grid,self.heat_values,self.heat_integrals,self.heat_slopes
        i=torch.clamp(torch.searchsorted(g,T.contiguous(),right=True)-1,0,g.numel()-2)
        d=T-g[i]
        h=H[i]+v[i]*d+.5*m[i]*d*d
        h=torch.where(T<=g[0],v[0]*(T-g[0]),h)
        h=torch.where(T>=g[-1],H[-1]+v[-1]*(T-g[-1]),h)
        return h-self.heat_offset

    def heat_capacity(self,T):
        return interp(T,self.heat_grid,self.heat_values)

    def temperature(self,rho,e):
        return self.temperature_from_sensible(e-self.cold(rho))

    def temperature_from_sensible(self,sensible):
        g,v,H,m=self.heat_grid,self.heat_values,self.heat_integrals,self.heat_slopes
        h=sensible+self.heat_offset
        i=torch.clamp(torch.searchsorted(H,h.contiguous(),right=True)-1,0,g.numel()-2)
        y=torch.clamp(h,min=H[0],max=H[-1])-H[i]
        root=torch.sqrt(torch.clamp(v[i]**2+2*m[i]*y,min=0.))
        T=g[i]+2*y/(v[i]+root)
        T=torch.where(h<=0,g[0]+h/v[0],T)
        return torch.where(h>=H[-1],g[-1]+(h-H[-1])/v[-1],T)

    def primitive(self,U):
        rho=U[...,0];P=U/rho[...,None]
        e=P[...,3]-.5*(P[...,1]**2+P[...,2]**2)
        return torch.cat((rho[...,None],P[...,1:3],self.temperature(rho,e)[...,None],P[...,4:]),-1)

    def conservative(self,P):
        rho=P[...,0];U=P*rho[...,None]
        energy=rho*(self.cold(rho)+self.sensible(P[...,3])+.5*(P[...,1]**2+P[...,2]**2))
        return torch.cat((rho[...,None],U[...,1:3],energy[...,None],U[...,4:]),-1)

    def valid(self,P):
        p=self.pressure(P[...,0])
        return (torch.isfinite(P).all(-1)&(P[...,0]>0)&torch.isfinite(p)&(p>0)
                &(P[...,3]>=self.tmin-1e-10)&(P[...,3]<=self.tmax+1e-10)
                &(P[...,4:6]>=-1e-12).all(-1)&(P[...,4:6]<=1+1e-12).all(-1)
                &(P[...,6:]>=-1e-12).all(-1))

    def progress(self,U):
        return (self.weights[0]*U[...,4]+self.weights[1]*U[...,5])/U[...,0]

    def channel_rate(self,channel_alpha,T,channel_index):
        a,e,l=self.channels[channel_index]
        rate=torch.exp(torch.clamp(
            interp(channel_alpha,a,l)-interp(channel_alpha,a,e)
            /(self.R*torch.clamp(T,min=1.)),min=-100.,max=60.))
        return torch.where(channel_alpha<1,rate,torch.zeros_like(rate))

    def rates_from_alpha(self,alpha,T):
        return torch.stack([
            self.channel_rate(alpha[...,i],T,i) for i in range(2)
        ],-1)

    def rates(self,U,T):
        return self.rates_from_alpha(U[...,4:6]/U[...,0,None],T)

    def traversed_rate_upper_bound(self,alpha0,alpha1,temperature1):
        """Raw-rate upper bound matching the NumPy endpoint/knot diagnostic."""
        maxima=torch.zeros_like(temperature1)
        for channel_index,(grid,_,_) in enumerate(self.channels):
            lo=alpha0[...,channel_index];hi=alpha1[...,channel_index]
            candidates=[
                self.channel_rate(lo,temperature1,channel_index),
                self.channel_rate(hi,temperature1,channel_index),
            ]
            for knot in grid[1:]:
                traversed=(lo<=knot)&(hi>=knot)
                evaluation=torch.where(
                    knot==1,
                    torch.nextafter(knot,torch.zeros_like(knot)),knot)
                knot_alpha=torch.ones_like(lo)*evaluation
                knot_rate=self.channel_rate(
                    knot_alpha,temperature1,channel_index)
                candidates.append(torch.where(
                    traversed,knot_rate,torch.zeros_like(knot_rate)))
            maxima=torch.maximum(maxima,torch.stack(candidates,-1).amax(-1))
        return maxima

    @staticmethod
    def _clock_piece(log_rate,slope,width):
        z=-slope*width
        small=abs(z)<=1.e-7
        polynomial=1+z*(.5+z*(1/6+z*(1/24+z/120)))
        safe_z=torch.where(small,torch.ones_like(z),z)
        exprel=torch.where(small,polynomial,torch.expm1(z)/safe_z)
        return torch.exp(-log_rate)*width*exprel

    def _fixed_beta_flow_channel(self,alpha0,duration,beta,channel_index):
        """Vectorized exact clipped-affine kinetic clock for one channel."""
        grid,activation,log_prefactor,activation_slope,log_prefactor_slope=(
            self.kinetic_tables[channel_index]
        )
        eps=torch.finfo(torch.float64).eps
        alpha_tolerance=256*eps
        input_ok=(torch.isfinite(alpha0)&torch.isfinite(duration)&torch.isfinite(beta)
                  &(duration>=0)&(beta>0)&(alpha0>=-alpha_tolerance)
                  &(alpha0<=1+alpha_tolerance))
        alpha=torch.clamp(alpha0,min=0,max=1)
        safe_duration=torch.clamp(duration,min=0)

        left=grid[:-1][None,:]
        right=grid[1:][None,:]
        b=beta[:,None]
        slope=log_prefactor_slope[None,:]-activation_slope[None,:]*b/self.R
        raw_left=log_prefactor[:-1][None,:]-activation[:-1][None,:]*b/self.R
        safe_slope=torch.where(abs(slope)>torch.finfo(torch.float64).tiny,
                               slope,torch.ones_like(slope))
        crossing_lower=torch.clamp(left+(-100-raw_left)/safe_slope,min=left,max=right)
        crossing_upper=torch.clamp(left+(60-raw_left)/safe_slope,min=left,max=right)
        crossing0=torch.minimum(crossing_lower,crossing_upper)
        crossing1=torch.maximum(crossing_lower,crossing_upper)
        piece_columns=3*slope.shape[1]
        piece_left=torch.stack((left.expand_as(slope),crossing0,crossing1),-1).reshape(alpha.shape[0],piece_columns)
        piece_right=torch.stack((crossing0,crossing1,right.expand_as(slope)),-1).reshape(alpha.shape[0],piece_columns)
        base_left=left.expand_as(slope)[...,None].expand(-1,-1,3).reshape(alpha.shape[0],piece_columns)
        piece_raw_left=raw_left[...,None].expand(-1,-1,3).reshape(alpha.shape[0],piece_columns)
        piece_slope=slope[...,None].expand(-1,-1,3).reshape(alpha.shape[0],piece_columns)

        start=torch.maximum(alpha[:,None],piece_left)
        width=torch.clamp(piece_right-start,min=0)
        midpoint=.5*(start+piece_right)
        midpoint_raw=piece_raw_left+piece_slope*(midpoint-base_left)
        affine=(midpoint_raw>-100)&(midpoint_raw<60)
        effective_slope=torch.where(affine,piece_slope,torch.zeros_like(piece_slope))
        effective_log=torch.clamp(
            piece_raw_left+piece_slope*(start-base_left),min=-100,max=60
        )
        clock=self._clock_piece(effective_log,effective_slope,width)
        usable=width>alpha_tolerance
        clock_ok=(torch.isfinite(clock)&((clock>0)|~usable)).all(-1)
        clock=torch.where(usable,clock,torch.zeros_like(clock))
        cumulative=torch.cumsum(clock,-1)
        total=cumulative[:,-1]
        complete=safe_duration>=total
        partial_piece=cumulative>safe_duration[:,None]
        selected=partial_piece.to(torch.int64).argmax(-1)
        gather=selected[:,None]
        selected_clock=torch.gather(clock,1,gather)[:,0]
        selected_cumulative=torch.gather(cumulative,1,gather)[:,0]
        selected_start=torch.gather(start,1,gather)[:,0]
        selected_width=torch.gather(width,1,gather)[:,0]
        selected_log=torch.gather(effective_log,1,gather)[:,0]
        selected_slope=torch.gather(effective_slope,1,gather)[:,0]
        elapsed=torch.clamp(safe_duration-(selected_cumulative-selected_clock),min=0)
        linear=abs(selected_slope*selected_width)<=1.e-7
        argument=-selected_slope*elapsed*torch.exp(selected_log)
        roundoff=128*eps
        domain_ok=argument>=-1-roundoff
        argument=torch.clamp(argument,min=float(np.nextafter(-1.,0.)))
        safe_selected_slope=torch.where(
            abs(selected_slope)>torch.finfo(torch.float64).tiny,
            selected_slope,torch.ones_like(selected_slope),
        )
        base=elapsed*torch.exp(selected_log)
        scaled=selected_slope*base
        linear_distance=base*(1+scaled*(.5+scaled*(1/3+scaled*.25)))
        nonlinear_distance=-torch.log1p(argument)/safe_selected_slope
        distance=torch.where(linear,linear_distance,nonlinear_distance)
        inverse_ok=(torch.isfinite(distance)&(distance>=-alpha_tolerance)
                    &(distance<=selected_width+alpha_tolerance))
        endpoint=torch.clamp(selected_start+torch.clamp(distance,min=0),min=0,max=1)
        endpoint=torch.where(complete,torch.ones_like(endpoint),endpoint)
        endpoint=torch.where(safe_duration>0,endpoint,alpha)
        endpoint=torch.where(endpoint>=1-alpha_tolerance,torch.ones_like(endpoint),endpoint)
        valid=input_ok&clock_ok&(complete|(domain_ok&inverse_ok))&torch.isfinite(endpoint)
        return endpoint,valid

    def _fixed_beta_flow(self,alpha0,duration,beta):
        values=[];valid=torch.ones_like(duration,dtype=torch.bool)
        for channel_index in range(2):
            endpoint,ok=self._fixed_beta_flow_channel(
                alpha0[:,channel_index],duration,beta,channel_index
            )
            values.append(endpoint);valid&=ok
        return torch.stack(values,-1),valid

    def _endpoint_at_beta(self,alpha0,duration,beta,progress_capacity,
                          accuracy_fraction=None):
        """Fixed-beta endpoint with the same safe shared depletion event."""
        if accuracy_fraction is None:
            accuracy_fraction=torch.ones_like(duration)
        endpoint,flow_ok=self._fixed_beta_flow(alpha0,duration,beta)
        progress=(endpoint-alpha0)@self.weights
        tolerance=self.progress_tolerance(
            progress_capacity,accuracy_fraction)
        no_capacity=progress_capacity<=0
        # Any represented overshoot must be located in physical time.  The
        # accuracy tolerance controls the root residual, never whether a
        # positive shared inventory exists or whether that inventory binds.
        binds=(progress>progress_capacity)&~no_capacity
        would_react=progress>0
        stopped=no_capacity&would_react
        depleted=binds|stopped
        endpoint=torch.where(stopped[:,None],alpha0,endpoint)
        progress=torch.where(stopped,torch.zeros_like(progress),progress)
        depletion_time=torch.where(stopped,torch.zeros_like(duration),
                                    torch.full_like(duration,math.nan))
        depletion_iterations=torch.zeros_like(duration,dtype=torch.int64)
        depletion_ok=flow_ok.clone()

        # Inventory roots are evaluated only for the rows that bind.  This
        # device-side dynamic compaction is required: otherwise every cold,
        # nonbinding cell performs the forty root iterations of the hottest
        # cell.  Local chemistry already disables CUDA graph capture.
        bind_indices=torch.nonzero(binds).flatten()
        if bind_indices.numel()==0:
            return {"alpha":endpoint,"progress":progress,
                    "depleted":depleted,"depletion_time":depletion_time,
                    "depletion_iterations":depletion_iterations,
                    "depletion_ok":depletion_ok}
        a0=alpha0.index_select(0,bind_indices)
        q=duration.index_select(0,bind_indices)
        b=beta.index_select(0,bind_indices)
        target=progress_capacity.index_select(0,bind_indices)
        event_tolerance=tolerance.index_select(0,bind_indices)
        event_tolerance=torch.minimum(event_tolerance,.5*target)
        lower=torch.zeros_like(q);upper=q.clone()
        lower_progress=torch.zeros_like(q)
        converged=torch.zeros_like(q,dtype=torch.bool)
        iterations=torch.zeros_like(q,dtype=torch.int64)
        event_ok=torch.ones_like(q,dtype=torch.bool)
        for iteration in range(1,self.s.max_chemistry_depletion_iterations+1):
            trial_time=.5*(lower+upper)
            trial_alpha,trial_ok=self._fixed_beta_flow(a0,trial_time,b)
            event_ok&=trial_ok
            trial_progress=(trial_alpha-a0)@self.weights
            residual=trial_progress-target
            unresolved=~converged
            go_right=residual<=0
            lower=torch.where(unresolved&go_right,trial_time,lower)
            lower_progress=torch.where(
                unresolved&go_right,trial_progress,lower_progress)
            upper=torch.where(unresolved&~go_right,trial_time,upper)
            newly=(unresolved&(target-lower_progress>=0)
                   &(target-lower_progress<=event_tolerance))
            iterations=torch.where(
                newly,torch.full_like(iterations,iteration),iterations)
            converged|=newly
        trial_alpha,trial_ok=self._fixed_beta_flow(a0,lower,b)
        event_ok&=trial_ok
        trial_progress=(trial_alpha-a0)@self.weights
        root_roundoff=256*(torch.nextafter(
            abs(target),torch.full_like(target,math.inf))-abs(target))
        nonovershooting=trial_progress<=target+root_roundoff
        event_converged=(converged&event_ok&nonovershooting
                         &(target-trial_progress<=event_tolerance))
        iterations=torch.where(
            iterations==0,
            torch.full_like(iterations,self.s.max_chemistry_depletion_iterations),
            iterations)
        endpoint.index_copy_(0,bind_indices,trial_alpha)
        progress.index_copy_(0,bind_indices,trial_progress)
        depletion_time.index_copy_(0,bind_indices,lower)
        depletion_iterations.index_copy_(0,bind_indices,iterations)
        depletion_ok.index_copy_(0,bind_indices,event_converged)
        return {"alpha":endpoint,"progress":progress,"depleted":depleted,
                "depletion_time":depletion_time,
                "depletion_iterations":depletion_iterations,
                "depletion_ok":depletion_ok}

    def _implicit_midpoint_panel(self,alpha0,sensible0,temperature0,duration,
                                 progress_capacity,accuracy_fraction=None):
        """Reciprocal-temperature midpoint panel over a packed batch.

        Predictor work spans the supplied cohort.  Later correctors gather
        only unresolved rows, so a hot cell cannot keep reevaluating every
        already-converged cell in the field.
        """
        if accuracy_fraction is None:
            accuracy_fraction=torch.ones_like(duration)
        beta0=1/torch.clamp(temperature0,min=1)
        adiabatic=sensible0+(1-alpha0)@self.Q
        temperature_adiabatic=self.temperature_from_sensible(adiabatic)
        panel_ok=(torch.isfinite(temperature_adiabatic)&(temperature_adiabatic>0))
        beta_lower=.5*(beta0+1/torch.clamp(temperature_adiabatic,min=1))

        def evaluate(indices,beta):
            local_alpha0=alpha0.index_select(0,indices)
            local_sensible0=sensible0.index_select(0,indices)
            value=self._endpoint_at_beta(
                local_alpha0,duration.index_select(0,indices),beta,
                progress_capacity.index_select(0,indices),
                accuracy_fraction.index_select(0,indices))
            sensible1=(local_sensible0
                       +(value["alpha"]-local_alpha0)@self.Q)
            temperature1=self.temperature_from_sensible(sensible1)
            local_beta0=beta0.index_select(0,indices)
            beta_target=.5*(local_beta0
                             +1/torch.clamp(temperature1,min=1))
            roundoff=(512*torch.finfo(torch.float64).eps
                      *torch.maximum(local_beta0,
                                     torch.ones_like(beta_target)))
            lower=beta_lower.index_select(0,indices);upper=local_beta0
            valid=(value["depletion_ok"]&torch.isfinite(temperature1)
                   &(temperature1>0)&(beta_target>=lower-roundoff)
                   &(beta_target<=upper+roundoff))
            value.update({"sensible":sensible1,"temperature":temperature1,
                          "beta_target":torch.minimum(
                              torch.maximum(beta_target,lower),upper),
                          "valid":valid})
            return value

        all_indices=torch.arange(duration.shape[0],device=duration.device)
        predictor=evaluate(all_indices,beta0)
        beta=predictor["beta_target"].clone()
        previous_alpha=predictor["alpha"].clone()
        previous_temperature=predictor["temperature"].clone()
        active=panel_ok&predictor["valid"]
        corrector_iterations=torch.zeros_like(duration,dtype=torch.int64)
        endpoint_evaluations=torch.ones_like(corrector_iterations)
        depletion_sum=predictor["depletion_iterations"].clone()
        depletion_max=predictor["depletion_iterations"].clone()
        normalized_residual=torch.full_like(duration,math.inf)
        alpha_correction=torch.zeros_like(duration)
        temperature_correction=torch.zeros_like(duration)
        result={key:(value.clone() if torch.is_tensor(value) else value)
                for key,value in predictor.items()}
        for iteration in range(1,self.s.max_chemistry_corrector_iterations+1):
            indices=torch.nonzero(active).flatten()
            if indices.numel()==0:
                break
            candidate=evaluate(indices,beta.index_select(0,indices))
            endpoint_evaluations.index_add_(
                0,indices,torch.ones_like(indices,dtype=torch.int64))
            depletion_sum.index_add_(
                0,indices,candidate["depletion_iterations"])
            depletion_max.index_copy_(0,indices,torch.maximum(
                depletion_max.index_select(0,indices),
                candidate["depletion_iterations"]))
            prior_alpha=previous_alpha.index_select(0,indices)
            alpha_scale=torch.maximum(
                torch.maximum(abs(candidate["alpha"]),abs(prior_alpha)),
                torch.ones_like(candidate["alpha"]))
            local_fraction=accuracy_fraction.index_select(0,indices)
            alpha_allowance=torch.maximum(
                local_fraction[:,None]
                *(self.s.chemistry_absolute_tolerance
                  +self.s.chemistry_relative_tolerance*alpha_scale),
                256*torch.finfo(torch.float64).eps*alpha_scale)
            alpha_residual=(abs(candidate["alpha"]-prior_alpha)
                            /alpha_allowance).amax(-1)
            prior_temperature=previous_temperature.index_select(0,indices)
            temperature_scale=torch.maximum(
                torch.maximum(abs(candidate["temperature"]),
                              abs(prior_temperature)),
                torch.ones_like(prior_temperature))
            temperature_allowance=torch.maximum(
                local_fraction*self.s.chemistry_temperature_tolerance,
                256*torch.finfo(torch.float64).eps*temperature_scale)
            endpoint_temperature_residual=(
                abs(candidate["temperature"]-prior_temperature)
                /temperature_allowance)
            local_beta=beta.index_select(0,indices)
            effective_temperature=1/torch.clamp(
                local_beta,min=torch.finfo(torch.float64).tiny)
            target_temperature=1/torch.clamp(
                candidate["beta_target"],
                min=torch.finfo(torch.float64).tiny)
            midpoint_temperature_residual=(
                abs(effective_temperature-target_temperature)
                /temperature_allowance)
            residual=torch.maximum(alpha_residual,torch.maximum(
                endpoint_temperature_residual,midpoint_temperature_residual))
            normalized_residual.index_copy_(0,indices,residual)
            alpha_correction.index_copy_(0,indices,torch.maximum(
                alpha_correction.index_select(0,indices),
                abs(candidate["alpha"]
                    -predictor["alpha"].index_select(0,indices)).amax(-1)))
            temperature_correction.index_copy_(0,indices,torch.maximum(
                temperature_correction.index_select(0,indices),
                abs(candidate["temperature"]
                    -predictor["temperature"].index_select(0,indices))))
            for key in ("alpha","progress","depleted","depletion_time",
                        "depletion_iterations","depletion_ok","sensible",
                        "temperature","beta_target","valid"):
                result[key].index_copy_(0,indices,candidate[key])
            corrector_iterations.index_copy_(
                0,indices,torch.full_like(indices,iteration,
                                           dtype=torch.int64))
            converged=(residual<=1)&candidate["valid"]
            active.index_copy_(0,indices,~converged)
            beta.index_copy_(0,indices,candidate["beta_target"])
            previous_alpha.index_copy_(0,indices,candidate["alpha"])
            previous_temperature.index_copy_(
                0,indices,candidate["temperature"])
        result.update({"converged":~active&panel_ok&predictor["valid"],
            "corrector_iterations":corrector_iterations,
            "endpoint_evaluations":endpoint_evaluations,
            "depletion_iteration_sum":depletion_sum,
            "maximum_depletion_iterations_used":depletion_max,
            "normalized_corrector_residual":normalized_residual,
            "feedback_alpha_correction":alpha_correction,
            "feedback_temperature_correction":temperature_correction})
        return result

    @staticmethod
    def _local_work(count,device):
        integer=lambda:torch.zeros(count,dtype=torch.int64,device=device)
        real=lambda:torch.zeros(count,dtype=torch.float64,device=device)
        return {"corrector_iteration_sum":integer(),
                "depletion_iteration_sum":integer(),
                "endpoint_evaluations":integer(),"evaluated_cell_count":integer(),
                "invalid_panel_attempt_count":integer(),
                "reaction_coordinate_rate_evaluations":integer(),
                "reaction_coordinate_quadrature_panel_evaluations":integer(),
                "reaction_coordinate_root_iterations":integer(),
                "reaction_coordinate_cell_count":integer(),
                "maximum_reaction_coordinate_quadrature_depth":integer(),
                "coupled_coordinate_attempt_count":integer(),
                "coupled_coordinate_cell_count":integer(),
                "coupled_coordinate_fallback_count":integer(),
                "coupled_coordinate_rate_evaluations":integer(),
                "coupled_coordinate_accepted_steps":integer(),
                "coupled_coordinate_rejected_steps":integer(),
                "coupled_coordinate_driver_switches":integer(),
                "maximum_coupled_coordinate_depth":integer(),
                "maximum_corrector_iterations_used":integer(),
                "maximum_depletion_iterations_used":integer(),
                "maximum_normalized_corrector_residual":real(),
                "maximum_reaction_coordinate_normalized_residual":real(),
                "maximum_coupled_coordinate_normalized_residual":real(),
                "maximum_rejected_coupled_coordinate_residual":real(),
                "feedback_alpha_correction":real(),
                "feedback_temperature_correction":real()}

    @staticmethod
    def _accumulate_local_work(total,attempt,indices):
        for key in ("corrector_iteration_sum","depletion_iteration_sum",
                    "endpoint_evaluations","evaluated_cell_count",
                    "invalid_panel_attempt_count",
                    "reaction_coordinate_rate_evaluations",
                    "reaction_coordinate_quadrature_panel_evaluations",
                    "reaction_coordinate_root_iterations",
                    "reaction_coordinate_cell_count",
                    "coupled_coordinate_attempt_count",
                    "coupled_coordinate_cell_count",
                    "coupled_coordinate_fallback_count",
                    "coupled_coordinate_rate_evaluations",
                    "coupled_coordinate_accepted_steps",
                    "coupled_coordinate_rejected_steps",
                    "coupled_coordinate_driver_switches"):
            total[key].index_add_(0,indices,attempt[key])
        for key in ("maximum_corrector_iterations_used",
                    "maximum_depletion_iterations_used",
                    "maximum_normalized_corrector_residual",
                    "maximum_reaction_coordinate_quadrature_depth",
                    "maximum_reaction_coordinate_normalized_residual",
                    "maximum_coupled_coordinate_depth",
                    "maximum_coupled_coordinate_normalized_residual",
                    "maximum_rejected_coupled_coordinate_residual",
                    "feedback_alpha_correction","feedback_temperature_correction"):
            total[key].index_copy_(0,indices,torch.maximum(
                total[key].index_select(0,indices),attempt[key]))

    def _reaction_coordinate_clock_path_depth0(
            self,start,end,sensible0,channel_index,alpha_tolerance,enabled):
        """Vectorized GL8/GL16 thermochemical clock on pre-split smooth spans.

        Kinetic-alpha and caloric-temperature knots are inserted before the
        quadrature pair, exactly as in the NumPy reaction-coordinate path.
        This bulk pass intentionally reports intervals that require recursive
        refinement; only those cells enter the packed hard-cell path.
        """
        q=self.Q[channel_index]
        candidates=[start,end]
        for table_channel,(grid,_,_) in enumerate(self.channels):
            for knot in grid:
                inside=(enabled&(channel_index==table_channel)
                        &(start<knot)&(knot<end))
                candidates.append(torch.where(
                    inside,torch.ones_like(start)*knot,end))
        safe_q=torch.where(q>0,q,torch.ones_like(q))
        for knot_energy in self.heat_knot_sensible:
            caloric_alpha=start+(knot_energy-sensible0)/safe_q
            inside=(enabled&(q>0)&(start<caloric_alpha)&(caloric_alpha<end))
            candidates.append(torch.where(inside,caloric_alpha,end))
        points=torch.sort(torch.stack(candidates,-1),-1).values
        left=points[:,:-1];right=points[:,1:]
        interval_enabled=enabled[:,None]&(right>left)
        half=.5*(right-left);midpoint=.5*(right+left)

        def evaluate(nodes):
            alpha=midpoint[...,None]+half[...,None]*nodes
            sensible=(sensible0[:,None,None]
                      +q[:,None,None]*(alpha-start[:,None,None]))
            temperature=self.temperature_from_sensible(sensible)
            rate0=self.channel_rate(alpha,temperature,0)
            rate1=self.channel_rate(alpha,temperature,1)
            rate=torch.where(channel_index[:,None,None]==0,rate0,rate1)
            raw0=(interp(alpha,self.channels[0][0],self.channels[0][2])
                  -interp(alpha,self.channels[0][0],self.channels[0][1])
                   /(self.R*torch.clamp(temperature,min=1.)))
            raw1=(interp(alpha,self.channels[1][0],self.channels[1][2])
                  -interp(alpha,self.channels[1][0],self.channels[1][1])
                   /(self.R*torch.clamp(temperature,min=1.)))
            raw=torch.where(channel_index[:,None,None]==0,raw0,raw1)
            valid=(torch.isfinite(temperature)&(temperature>0)
                   &torch.isfinite(rate)&(rate>0)&torch.isfinite(raw))
            return (temperature,torch.where(valid,rate,torch.ones_like(rate)),
                    raw,valid)

        temperature8,rate8,raw8,valid8=evaluate(self.gl8_nodes)
        temperature16,rate16,raw16,valid16=evaluate(self.gl16_nodes)
        clock8=half*(self.gl8_weights/rate8).sum(-1)
        clock16=half*(self.gl16_weights/rate16).sum(-1)

        endpoint_right=torch.where(
            right>=1,torch.nextafter(torch.ones_like(right),torch.zeros_like(right)),right)
        endpoint_alpha=torch.stack((left,endpoint_right),-1)
        endpoint_sensible=(sensible0[:,None,None]
            +q[:,None,None]*(endpoint_alpha-start[:,None,None]))
        endpoint_temperature=self.temperature_from_sensible(endpoint_sensible)
        endpoint_rate0=self.channel_rate(endpoint_alpha,endpoint_temperature,0)
        endpoint_rate1=self.channel_rate(endpoint_alpha,endpoint_temperature,1)
        endpoint_rate=torch.where(
            channel_index[:,None,None]==0,endpoint_rate0,endpoint_rate1)
        endpoint_raw0=(interp(endpoint_alpha,self.channels[0][0],self.channels[0][2])
            -interp(endpoint_alpha,self.channels[0][0],self.channels[0][1])
             /(self.R*torch.clamp(endpoint_temperature,min=1.)))
        endpoint_raw1=(interp(endpoint_alpha,self.channels[1][0],self.channels[1][2])
            -interp(endpoint_alpha,self.channels[1][0],self.channels[1][1])
             /(self.R*torch.clamp(endpoint_temperature,min=1.)))
        endpoint_raw=torch.where(
            channel_index[:,None,None]==0,endpoint_raw0,endpoint_raw1)
        endpoint_valid=(torch.isfinite(endpoint_temperature)&(endpoint_temperature>0)
                        &torch.isfinite(endpoint_rate)&(endpoint_rate>0)
                        &torch.isfinite(endpoint_raw))
        pair_valid=(valid8.all(-1)&valid16.all(-1)&endpoint_valid.all(-1)
                    &torch.isfinite(clock8)&torch.isfinite(clock16)
                    &(clock8>0)&(clock16>0))
        all_raw=torch.cat((raw8,raw16,endpoint_raw),-1)
        raw_min=all_raw.amin(-1);raw_max=all_raw.amax(-1)
        raw_span=raw_max-raw_min
        clamp_cross=((raw_span>256*torch.finfo(torch.float64).eps)
            &(((raw_min<=-100.)&(raw_max>=-100.))
              |((raw_min<=60.)&(raw_max>=60.))))
        maximum_rate=torch.maximum(
            torch.maximum(rate8.amax(-1),rate16.amax(-1)),endpoint_rate.amax(-1))
        all_temperature=torch.cat((temperature8,temperature16,endpoint_temperature),-1)
        minimum_cp=self.heat_capacity(all_temperature).amin(-1)
        maximum_temperature_derivative=torch.where(
            q[:,None]>0,q[:,None]/torch.clamp(minimum_cp,min=torch.finfo(torch.float64).tiny),
            torch.zeros_like(minimum_cp))
        clock_error=(8*abs(clock16-clock8)
                     +64*torch.finfo(torch.float64).eps*abs(clock16))
        alpha_error=clock_error*maximum_rate
        temperature_error=alpha_error*maximum_temperature_derivative
        span=end-start
        fraction=(right-left)/torch.clamp(span[:,None],min=torch.finfo(torch.float64).tiny)
        local_alpha_tolerance=torch.maximum(
            alpha_tolerance[:,None]*fraction,
            torch.full_like(fraction,256*torch.finfo(torch.float64).eps))
        local_temperature_tolerance=torch.maximum(
            self.s.chemistry_temperature_tolerance*fraction,
            torch.full_like(fraction,256*torch.finfo(torch.float64).eps))
        normalized_alpha=alpha_error/local_alpha_tolerance
        normalized_temperature=temperature_error/local_temperature_tolerance
        normalized=torch.maximum(normalized_alpha,normalized_temperature)
        accepted=(interval_enabled&pair_valid&~clamp_cross
                  &torch.isfinite(normalized)
                  &(normalized<=.125))
        rejected=interval_enabled&~accepted
        certification_rejected=(interval_enabled&pair_valid
            &torch.isfinite(normalized)&(clamp_cross|(normalized>.125)))
        physical_rejected=(interval_enabled
            &(~pair_valid|~torch.isfinite(normalized)))
        include=accepted.to(torch.float64)
        def total(value):
            return (value*include).sum(-1)
        def maximum(value):
            return torch.where(interval_enabled,value,torch.zeros_like(value)).amax(-1)
        total_alpha_error=total(alpha_error)
        total_temperature_error=total(temperature_error)
        path_valid=(~rejected).all(-1)
        path_valid=torch.where(enabled,path_valid,torch.ones_like(path_valid))
        return {
            "clock":total(clock16),"clock_error":total(clock_error),
            "alpha_error":total_alpha_error,
            "temperature_error":total_temperature_error,
            "rate_evaluations":26*interval_enabled.to(torch.int64).sum(-1),
            "quadrature_panel_evaluations":interval_enabled.to(torch.int64).sum(-1),
            "maximum_normalized_residual":maximum(normalized),
            "normalized_alpha_residual":total_alpha_error/alpha_tolerance,
            "normalized_temperature_residual":(
                total_temperature_error/self.s.chemistry_temperature_tolerance),
            "maximum_rejected_normalized_residual":torch.where(
                rejected,torch.where(torch.isfinite(normalized),normalized,
                                     torch.zeros_like(normalized)),
                torch.zeros_like(normalized)).amax(-1),
            "valid":path_valid,
            "needs_refinement":rejected.any(-1),
            "certification_failure":certification_rejected.any(-1),
            "physical_failure":physical_rejected.any(-1),
        }

    def _solve_one_active_reaction_coordinate(
            self,alpha0,sensible0,temperature0,duration,progress_capacity,
            channel_index,enabled):
        """Packed FP64 one-active-channel thermochemical endpoint map.

        The production BC path is smooth on the explicit kinetic/caloric
        subintervals and accepts the GL8/GL16 pair at depth zero.  The caller
        routes a pair needing recursive quadrature refinement through the same
        configured chronological midpoint/refinement solver, preserving a
        convergent fail-closed fallback without changing a physical rate.
        """
        gather=channel_index[:,None]
        start=torch.gather(alpha0,1,gather)[:,0]
        q=self.Q[channel_index];weight=self.weights[channel_index]
        safe_weight=torch.where(weight>0,weight,torch.ones_like(weight))
        inventory_bound=torch.where(
            weight>0,start+progress_capacity/safe_weight,
            torch.full_like(start,math.inf))
        bound=torch.minimum(torch.ones_like(start),inventory_bound)
        inventory_binds=inventory_bound<=1
        # Match the scalar nonovershooting endpoint exactly.  The analytic
        # division can round upward, so move only binding bounds toward the
        # start until their reconstructed weighted progress fits.
        for _ in range(8):
            overshoots=(inventory_binds
                        &(weight*(bound-start)>progress_capacity))
            bound=torch.where(
                overshoots,torch.nextafter(bound,start),bound)
        inventory_safe=(~inventory_binds
                        |(weight*(bound-start)<=progress_capacity))
        alpha_tolerance=(self.s.chemistry_absolute_tolerance
            +self.s.chemistry_relative_tolerance*torch.maximum(
                torch.maximum(abs(start),abs(bound)),torch.ones_like(start)))
        input_ok=enabled&torch.isfinite(bound)&(bound>start)&inventory_safe
        total=self._reaction_coordinate_clock_path_depth0(
            start,bound,sensible0,channel_index,alpha_tolerance,input_ok)
        maximum_normalized=total["maximum_normalized_residual"].clone()
        maximum_normalized_alpha=total["normalized_alpha_residual"].clone()
        maximum_normalized_temperature=total["normalized_temperature_residual"].clone()
        maximum_rejected=total["maximum_rejected_normalized_residual"].clone()
        rate_evaluations=total["rate_evaluations"].clone()
        panel_evaluations=total["quadrature_panel_evaluations"].clone()
        roundoff_time=(128*torch.finfo(torch.float64).eps*torch.maximum(
            torch.maximum(abs(duration),abs(total["clock"])),
            torch.full_like(duration,torch.finfo(torch.float64).tiny)))
        total_ok=input_ok&total["valid"]
        reaches_bound=(total_ok
            &(duration>=total["clock"]+total["clock_error"]-roundoff_time))
        partial=total_ok&~reaches_bound
        safe_total=torch.clamp(total["clock"],min=torch.finfo(torch.float64).tiny)
        fraction=torch.clamp(duration/safe_total,min=0,
                             max=float(np.nextafter(1.,0.)))
        endpoint=start+(bound-start)*fraction
        endpoint=torch.where(reaches_bound,bound,endpoint)
        lower=start.clone();upper=bound.clone()
        root_active=partial.clone()
        root_converged=torch.zeros_like(enabled)
        terminal_failure=(enabled&~input_ok)|total["physical_failure"]
        certification_failure=total["certification_failure"]
        root_failed=terminal_failure.clone()
        root_iterations=torch.zeros_like(channel_index,dtype=torch.int64)

        for iteration in range(1,self.s.max_chemistry_depletion_iterations+1):
            evaluated=root_active
            clock=self._reaction_coordinate_clock_path_depth0(
                start,endpoint,sensible0,channel_index,alpha_tolerance,evaluated)
            path_ok=evaluated&clock["valid"]
            rate_evaluations+=torch.where(
                evaluated,clock["rate_evaluations"],
                torch.zeros_like(clock["rate_evaluations"]))
            panel_evaluations+=torch.where(
                evaluated,clock["quadrature_panel_evaluations"],
                torch.zeros_like(clock["quadrature_panel_evaluations"]))
            maximum_normalized=torch.where(evaluated,torch.maximum(
                maximum_normalized,clock["maximum_normalized_residual"]),
                maximum_normalized)
            maximum_normalized_alpha=torch.where(evaluated,torch.maximum(
                maximum_normalized_alpha,clock["normalized_alpha_residual"]),
                maximum_normalized_alpha)
            maximum_normalized_temperature=torch.where(evaluated,torch.maximum(
                maximum_normalized_temperature,
                clock["normalized_temperature_residual"]),
                maximum_normalized_temperature)
            maximum_rejected=torch.where(evaluated,torch.maximum(
                maximum_rejected,clock["maximum_rejected_normalized_residual"]),
                maximum_rejected)
            residual=clock["clock"]-duration
            endpoint_sensible=sensible0+q*(endpoint-start)
            endpoint_temperature=self.temperature_from_sensible(endpoint_sensible)
            endpoint_rate0=self.channel_rate(endpoint,endpoint_temperature,0)
            endpoint_rate1=self.channel_rate(endpoint,endpoint_temperature,1)
            endpoint_rate=torch.where(channel_index==0,endpoint_rate0,endpoint_rate1)
            endpoint_ok=(torch.isfinite(endpoint_temperature)&(endpoint_temperature>0)
                         &torch.isfinite(endpoint_rate)&(endpoint_rate>0))
            path_ok&=endpoint_ok
            rate_evaluations+=evaluated.to(torch.int64)
            root_iterations=torch.where(
                evaluated,torch.full_like(root_iterations,iteration),root_iterations)
            mapped_alpha_error=(abs(residual)+clock["clock_error"])*endpoint_rate
            mapped_temperature_error=torch.where(
                q>0,mapped_alpha_error*q/torch.clamp(
                    self.heat_capacity(endpoint_temperature),
                    min=torch.finfo(torch.float64).tiny),
                torch.zeros_like(mapped_alpha_error))
            state_converged=(path_ok
                &(mapped_alpha_error<=.125*alpha_tolerance)
                &(mapped_temperature_error
                  <=.125*self.s.chemistry_temperature_tolerance))

            unresolved=evaluated&~state_converged
            below=(clock["clock"]+clock["clock_error"]<duration)
            above=(clock["clock"]-clock["clock_error"]>duration)
            straddled=unresolved&~below&~above
            terminal_failure|=evaluated&~path_ok
            certification_failure|=straddled
            root_failed|=(evaluated&~path_ok)|straddled
            bracketable=unresolved&path_ok&~straddled
            lower=torch.where(bracketable&below,endpoint,lower)
            upper=torch.where(bracketable&above,endpoint,upper)
            bracket_temperature=self.temperature_from_sensible(torch.stack((
                sensible0+q*(lower-start),sensible0+q*(upper-start)), -1))
            bracket_small=(bracketable
                &(upper-lower<=.125*alpha_tolerance)
                &(abs(bracket_temperature[:,1]-bracket_temperature[:,0])
                  <=.125*self.s.chemistry_temperature_tolerance))
            endpoint=torch.where(bracket_small,lower,endpoint)
            root_converged|=state_converged|bracket_small
            continuing=bracketable&~bracket_small
            newton=endpoint-residual*endpoint_rate
            margin=.1*(upper-lower)
            safeguarded=(newton>lower+margin)&(newton<upper-margin)
            proposal=torch.where(safeguarded,newton,.5*(lower+upper))
            endpoint=torch.where(continuing,proposal,endpoint)
            root_active=continuing

        certification_failure|=root_active
        root_failed|=root_active
        valid=(~enabled)|(reaches_bound|root_converged)
        valid&=~root_failed
        endpoint=torch.where(reaches_bound,bound,endpoint)
        alpha=alpha0.clone()
        channel0=enabled&(channel_index==0);channel1=enabled&(channel_index==1)
        alpha[:,0]=torch.where(channel0,endpoint,alpha[:,0])
        alpha[:,1]=torch.where(channel1,endpoint,alpha[:,1])
        sensible=sensible0+q*(endpoint-start)
        temperature=self.temperature_from_sensible(sensible)
        consumed=weight*(endpoint-start)
        capacity_left=torch.clamp(progress_capacity-consumed,min=0)
        depleted=reaches_bound&inventory_binds
        depletion_time=torch.where(
            depleted,total["clock"],torch.full_like(duration,math.nan))

        beta0=1/torch.clamp(temperature0,min=1)
        frozen0,frozen0_ok=self._fixed_beta_flow_channel(
            start,duration,beta0,0)
        frozen1,frozen1_ok=self._fixed_beta_flow_channel(
            start,duration,beta0,1)
        frozen=torch.where(channel_index==0,frozen0,frozen1)
        frozen_ok=torch.where(channel_index==0,frozen0_ok,frozen1_ok)
        frozen=torch.minimum(frozen,bound)
        frozen_temperature=self.temperature_from_sensible(
            sensible0+q*(frozen-start))
        endpoint_invalid=(enabled
            &(~frozen_ok|~torch.isfinite(temperature)|(temperature<=0)))
        terminal_failure|=endpoint_invalid
        valid&=~endpoint_invalid
        return {
            "alpha":alpha,"sensible":sensible,"temperature":temperature,
            "capacity_left":capacity_left,"depleted":depleted,
            "depletion_time":depletion_time,"valid":valid,
            "root_iterations":root_iterations,
            "rate_evaluations":rate_evaluations,
            "quadrature_panel_evaluations":panel_evaluations,
            "maximum_depth":torch.zeros_like(root_iterations),
            "maximum_normalized_residual":maximum_normalized,
            "maximum_normalized_alpha_residual":maximum_normalized_alpha,
            "maximum_normalized_temperature_residual":maximum_normalized_temperature,
            "maximum_rejected_normalized_residual":maximum_rejected,
            "feedback_alpha_correction":abs(endpoint-frozen),
            "feedback_temperature_correction":abs(temperature-frozen_temperature),
            "certification_failure":enabled&certification_failure,
            "terminal_failure":enabled&terminal_failure,
        }

    @staticmethod
    def _coordinate_gather(values,channel):
        return torch.gather(values,1,channel[:,None])[:,0]

    @staticmethod
    def _coordinate_alpha(driver,driver_value,other_value):
        driver0=driver==0
        return torch.stack((
            torch.where(driver0,driver_value,other_value),
            torch.where(driver0,other_value,driver_value),
        ),-1)

    def _unclipped_log_rates(self,alpha,temperature):
        return torch.stack([
            interp(alpha[...,index],grid,log_prefactor)
            -interp(alpha[...,index],grid,activation)
             /(self.R*torch.clamp(temperature,min=1.))
            for index,(grid,activation,log_prefactor) in enumerate(self.channels)
        ],-1)

    @staticmethod
    def _clamp_region(exponent):
        return torch.where(
            exponent < -100.,-torch.ones_like(exponent,dtype=torch.int64),
            torch.where(exponent > 60.,torch.ones_like(exponent,dtype=torch.int64),
                        torch.zeros_like(exponent,dtype=torch.int64)))

    def _next_coordinate_knot(self,alpha,channel):
        """First table point strictly above alpha+roundoff for mixed channels."""
        machine=torch.finfo(torch.float64).eps
        candidates=[];completions=[]
        for index,(grid,_,_) in enumerate(self.channels):
            threshold=alpha[:,index]+256*machine
            slot=torch.searchsorted(grid,threshold.contiguous(),right=True)
            has=slot<grid.numel()
            safe_slot=torch.clamp(slot,max=grid.numel()-1)
            value=grid[safe_slot]
            completes=(~has)|(value>=1.)
            target=torch.where(
                completes,
                torch.nextafter(torch.ones_like(value),torch.zeros_like(value)),
                value)
            candidates.append(target);completions.append(completes)
        candidate=torch.stack(candidates,-1)
        completion=torch.stack(completions,-1)
        return (self._coordinate_gather(candidate,channel),
                self._coordinate_gather(completion,channel).bool())

    def _coupled_coordinate_dp54_step(
            self,origin_alpha,origin_sensible,alpha,elapsed,driver,width,enabled):
        """Fixed-flow FP64 DP5(4) map; all evolving decisions remain masks."""
        other=1-driver
        driver0=self._coordinate_gather(alpha,driver)
        other0=self._coordinate_gather(alpha,other)
        y0=torch.stack((other0,elapsed),-1)
        stages=[];stage_alpha=[];stage_sensible=[];stage_rates=[]
        stage_exponents=[]
        valid=torch.ones_like(enabled)
        tolerance=1024*torch.finfo(torch.float64).eps
        one=torch.ones_like(width)
        for coefficient,row in zip(_DP54_C,_DP54_A):
            y=y0
            for scale,derivative in zip(row,stages):
                y=y+width[:,None]*float(scale)*derivative
            driver_value=driver0+float(coefficient)*width
            stage=self._coordinate_alpha(driver,driver_value,y[:,0])
            domain=(torch.isfinite(stage).all(-1)
                    &(stage>=-tolerance).all(-1)&(stage<1.).all(-1))
            sensible=(origin_sensible
                      +(self.Q[None,:]*(stage-origin_alpha)).sum(-1))
            temperature=self.temperature_from_sensible(sensible)
            rates=self.rates_from_alpha(stage,temperature)
            rate_valid=(torch.isfinite(temperature)&(temperature>0)
                        &torch.isfinite(rates).all(-1)&(rates>0).all(-1))
            stage_ok=domain&rate_valid
            valid&=torch.where(enabled,stage_ok,torch.ones_like(stage_ok))
            driver_rate=self._coordinate_gather(rates,driver)
            other_rate=self._coordinate_gather(rates,other)
            safe_driver=torch.where(enabled&stage_ok,driver_rate,one)
            derivative=torch.stack((other_rate/safe_driver,one/safe_driver),-1)
            derivative_ok=(torch.isfinite(derivative).all(-1)
                           &(derivative>0).all(-1))
            valid&=torch.where(enabled,derivative_ok,torch.ones_like(derivative_ok))
            derivative=torch.where(enabled[:,None],derivative,
                                   torch.zeros_like(derivative))
            stages.append(derivative);stage_alpha.append(stage)
            stage_sensible.append(sensible);stage_rates.append(rates)
            stage_exponents.append(self._unclipped_log_rates(stage,temperature))
        derivatives=torch.stack(stages,1)
        high=y0+width[:,None]*(self.dp54_b5[None,:,None]*derivatives).sum(1)
        low=y0+width[:,None]*(self.dp54_b4[None,:,None]*derivatives).sum(1)
        endpoint_ok=(torch.isfinite(high).all(-1)&torch.isfinite(low).all(-1)
                     &(high[:,0]>=y0[:,0]-tolerance)&(high[:,0]<1.)
                     &(high[:,1]>y0[:,1]))
        valid&=torch.where(enabled,endpoint_ok,torch.ones_like(endpoint_ok))
        return {"high":high,"low":low,"valid":valid,
            "stage_alpha":torch.stack(stage_alpha,1),
            "stage_sensible":torch.stack(stage_sensible,1),
            "stage_rates":torch.stack(stage_rates,1),
            "stage_exponents":torch.stack(stage_exponents,1)}

    def _coupled_coordinate_error(
            self,step,origin_alpha,origin_sensible,alpha,driver,width,enabled):
        other=1-driver
        endpoint_driver=self._coordinate_gather(alpha,driver)+width
        high_alpha=self._coordinate_alpha(driver,endpoint_driver,step["high"][:,0])
        low_alpha=self._coordinate_alpha(driver,endpoint_driver,step["low"][:,0])
        tolerance=1024*torch.finfo(torch.float64).eps
        domain=((high_alpha>=-tolerance).all(-1)&(high_alpha<1.).all(-1)
                &(low_alpha>=-tolerance).all(-1)&(low_alpha<1.).all(-1))
        high_sensible=(origin_sensible
            +(self.Q[None,:]*(high_alpha-origin_alpha)).sum(-1))
        low_sensible=(origin_sensible
            +(self.Q[None,:]*(low_alpha-origin_alpha)).sum(-1))
        high_temperature=self.temperature_from_sensible(high_sensible)
        low_temperature=self.temperature_from_sensible(low_sensible)
        rates=self.rates_from_alpha(high_alpha,high_temperature)
        endpoint_ok=(torch.isfinite(rates).all(-1)&(rates>0).all(-1)
                     &torch.isfinite(high_temperature)
                     &torch.isfinite(low_temperature))
        scale=(self.s.chemistry_absolute_tolerance
            +self.s.chemistry_relative_tolerance*torch.maximum(
                torch.maximum(abs(high_alpha),abs(low_alpha)),
                torch.ones_like(high_alpha)))
        direct_alpha=(abs(step["high"][:,0]-step["low"][:,0])
                      /self._coordinate_gather(scale,other))
        direct_temperature=(abs(high_temperature-low_temperature)
                            /self.s.chemistry_temperature_tolerance)
        clock_error=abs(step["high"][:,1]-step["low"][:,1])
        mapped_alpha=(clock_error[:,None]*rates/scale).amax(-1)
        mapped_temperature=(clock_error*(rates*self.Q[None,:]).sum(-1)
            /torch.clamp(self.heat_capacity(high_temperature),
                         min=torch.finfo(torch.float64).tiny)
            /self.s.chemistry_temperature_tolerance)
        normalized=2*torch.maximum(
            torch.maximum(direct_alpha,direct_temperature),
            torch.maximum(mapped_alpha,mapped_temperature))
        valid=step["valid"]&domain&endpoint_ok&torch.isfinite(normalized)
        valid=torch.where(enabled,valid,torch.ones_like(valid))
        return normalized,valid,high_alpha,high_sensible,high_temperature

    def _commit_coupled_coordinate_step(
            self,state,endpoint_alpha,endpoint_time,normalized,width,
            provisional,switch_driver):
        """Commit one certified coordinate endpoint with an exact ledger."""
        alpha=state["alpha"];device=alpha.device;count=alpha.shape[0]
        one=torch.ones(count,dtype=torch.float64,device=device)
        machine=torch.finfo(torch.float64).eps
        capacity_tolerance=self.progress_tolerance(state["capacity0"])
        time_tolerance=256*machine*abs(state["duration"])
        driver_endpoint=self._coordinate_gather(endpoint_alpha,state["driver"])
        boundary_tolerance=256*machine*torch.maximum(
            torch.maximum(abs(state["driver_target"]),abs(driver_endpoint)),one)
        at_boundary=provisional&(
            state["driver_target"]-driver_endpoint<=boundary_tolerance)
        exact_driver=torch.where(
            state["completes_driver"],torch.ones_like(driver_endpoint),
            state["driver_target"])
        snapped=self._coordinate_alpha(
            state["driver"],exact_driver,
            self._coordinate_gather(endpoint_alpha,1-state["driver"]))
        endpoint_alpha=torch.where(at_boundary[:,None],snapped,endpoint_alpha)
        endpoint_sensible=(state["origin_sensible"]
            +(self.Q[None,:]
              *(endpoint_alpha-state["origin_alpha"])).sum(-1))
        endpoint_temperature=self.temperature_from_sensible(endpoint_sensible)
        incremental=(self.weights[None,:]*(endpoint_alpha-alpha)).sum(-1)
        time_cross=endpoint_time>=state["duration"]
        capacity_cross=incremental>=state["capacity"]
        event_pending=provisional&(time_cross|capacity_cross)
        state["event_pending"]|=event_pending
        state["event_upper_width"]=torch.where(
            event_pending,width,state["event_upper_width"])
        provisional&=~event_pending
        event_failure=provisional&(incremental<-capacity_tolerance)
        state["fallback"]|=event_failure
        accepted=provisional&~event_failure

        alpha=torch.where(accepted[:,None],endpoint_alpha,alpha)
        state["alpha"]=alpha
        state["sensible"]=torch.where(
            accepted,endpoint_sensible,state["sensible"])
        state["temperature"]=torch.where(
            accepted,endpoint_temperature,state["temperature"])
        state["elapsed"]=torch.where(
            accepted,torch.minimum(endpoint_time,state["duration"]),
            state["elapsed"])
        cumulative=(self.weights[None,:]
            *(endpoint_alpha-state["origin_alpha"])).sum(-1)
        remaining_capacity=torch.clamp(state["capacity0"]-cumulative,min=0)
        state["capacity"]=torch.where(
            accepted,remaining_capacity,state["capacity"])
        state["accepted_steps"]+=accepted.to(torch.int64)
        state["maximum_residual"]=torch.where(
            accepted,torch.maximum(state["maximum_residual"],normalized),
            state["maximum_residual"])
        switch=accepted&switch_driver
        other=1-state["driver"]
        state["forced_driver"]=torch.where(
            switch,other,state["forced_driver"])
        state["driver_switches"]+=switch.to(torch.int64)
        next_factor=torch.where(
            normalized==0,torch.full_like(normalized,5.),
            torch.clamp(.9*torch.pow(
                torch.clamp(normalized,min=1e-300),-.2),min=.2,max=5.))
        hint_value=width*next_factor
        for channel in range(2):
            update=accepted&(state["driver"]==channel)
            state["step_hint"][:,channel]=torch.where(
                update,hint_value,state["step_hint"][:,channel])
        at_boundary&=accepted
        for channel in range(2):
            clear=at_boundary&(state["driver"]==channel)
            state["step_hint"][:,channel]=torch.where(
                clear,torch.full_like(hint_value,math.nan),
                state["step_hint"][:,channel])
        state["retrying"]&=~accepted
        state["retrying"]&=~state["fallback"]
        state["active"]=(state["active"]&~state["fallback"]
                         &~state["success"]&~state["needs_tail"])
        return state

    def _resolve_coupled_coordinate_bracket(self,state):
        """Resolve compact dominance crossings at an outer block boundary."""
        alpha=state["alpha"];device=alpha.device;count=alpha.shape[0]
        enabled=state["bracket_pending"].clone()
        zero=torch.zeros(count,dtype=torch.float64,device=device)
        one=torch.ones_like(zero);machine=torch.finfo(torch.float64).eps
        lower=zero.clone();upper=state["width"].clone()
        original=self._coupled_coordinate_dp54_step(
            state["origin_alpha"],state["origin_sensible"],alpha,
            state["elapsed"],state["driver"],upper,enabled)
        state["rate_evaluations"]+=7*enabled.to(torch.int64)
        upper_high=original["high"].clone();upper_low=original["low"].clone()
        bracket_done=~enabled
        bracket_valid=torch.where(
            enabled,original["valid"],torch.ones_like(enabled))
        other=1-state["driver"]
        for _root in range(self.s.max_chemistry_depletion_iterations):
            root_active=enabled&~bracket_done
            trial_width=.5*(lower+upper)
            trial=self._coupled_coordinate_dp54_step(
                state["origin_alpha"],state["origin_sensible"],alpha,
                state["elapsed"],state["driver"],trial_width,root_active)
            state["rate_evaluations"]+=7*root_active.to(torch.int64)
            bracket_valid&=torch.where(
                root_active,trial["valid"],torch.ones_like(root_active))
            trial_driver=self._coordinate_gather(
                trial["stage_rates"][:,-1,:],state["driver"])
            trial_other=self._coordinate_gather(
                trial["stage_rates"][:,-1,:],other)
            crossed=trial_other>=trial_driver
            advance=root_active&~crossed&trial["valid"]
            retreat=root_active&(crossed|~trial["valid"])
            lower=torch.where(advance,trial_width,lower)
            upper=torch.where(retreat,trial_width,upper)
            certified_upper=root_active&crossed&trial["valid"]
            upper_high=torch.where(
                certified_upper[:,None],trial["high"],upper_high)
            upper_low=torch.where(
                certified_upper[:,None],trial["low"],upper_low)
            relative=(upper-lower)/torch.maximum(
                torch.maximum(abs(self._coordinate_gather(
                    alpha,state["driver"])),state["width"]),one)
            bracket_done|=root_active&(relative<=256*machine)

        bad=enabled&(~bracket_valid|(upper<=0))
        state["fallback"]|=bad
        candidate=enabled&~bad
        bracket_step={"high":upper_high,"low":upper_low,
                      "valid":bracket_valid}
        normalized,ok,endpoint_alpha,_,_=self._coupled_coordinate_error(
            bracket_step,state["origin_alpha"],state["origin_sensible"],
            alpha,state["driver"],upper,candidate)
        state["rate_evaluations"]+=candidate.to(torch.int64)
        rejected=candidate&(~ok|(normalized>1))
        finite_reject=rejected&torch.isfinite(normalized)
        state["maximum_rejected_residual"]=torch.where(
            finite_reject,torch.maximum(
                state["maximum_rejected_residual"],normalized),
            state["maximum_rejected_residual"])
        depth=state["depth"]+rejected.to(torch.int64)
        state["rejected_steps"]+=rejected.to(torch.int64)
        state["maximum_depth"]=torch.maximum(state["maximum_depth"],depth)
        over=rejected&(depth>self.s.max_chemistry_local_refinements)
        state["fallback"]|=over
        state["depth"]=torch.where(rejected,depth,state["depth"])
        state["width"]=torch.where(
            rejected&~over,.5*upper,state["width"])
        accepted=candidate&ok&(normalized<=1)&~rejected
        state["bracket_pending"]&=~enabled
        state=self._commit_coupled_coordinate_step(
            state,endpoint_alpha,upper_high[:,1],normalized,upper,
            accepted,accepted)
        return state

    def _resolve_coupled_coordinate_knot(self,state):
        """Stop compact pending rows on an exact non-driver table knot.

        The bisection is intentionally executed only after hard-cell
        compaction at an outer block boundary.  Its loop count is fixed, and
        every evolving decision remains a device mask.
        """
        alpha=state["alpha"];device=alpha.device;count=alpha.shape[0]
        enabled=state["knot_pending"].clone()
        zero=torch.zeros(count,dtype=torch.float64,device=device)
        one=torch.ones_like(zero);machine=torch.finfo(torch.float64).eps
        lower=zero.clone();upper=state["knot_upper_width"].clone()
        other=1-state["driver"]
        target=state["knot_target"]
        state_tolerance=torch.maximum(
            torch.full_like(zero,self.s.chemistry_absolute_tolerance),
            torch.full_like(zero,256*machine))
        root_threshold=torch.where(
            target<1.,target,target-state_tolerance)

        original=self._coupled_coordinate_dp54_step(
            state["origin_alpha"],state["origin_sensible"],alpha,
            state["elapsed"],state["driver"],upper,enabled)
        state["rate_evaluations"]+=7*enabled.to(torch.int64)
        upper_high=original["high"].clone();upper_low=original["low"].clone()
        original_other=upper_high[:,0]
        upper_cross=original_other>=root_threshold
        root_valid=torch.where(
            enabled,original["valid"]&upper_cross,torch.ones_like(enabled))
        done=~enabled
        for _root in range(self.s.max_chemistry_depletion_iterations):
            root_active=enabled&~done&root_valid
            trial_width=.5*(lower+upper)
            trial=self._coupled_coordinate_dp54_step(
                state["origin_alpha"],state["origin_sensible"],alpha,
                state["elapsed"],state["driver"],trial_width,root_active)
            state["rate_evaluations"]+=7*root_active.to(torch.int64)
            valid=root_active&trial["valid"]
            crossed=trial["high"][:,0]>=root_threshold
            advance=valid&~crossed
            retreat=valid&crossed
            lower=torch.where(advance,trial_width,lower)
            upper=torch.where(retreat,trial_width,upper)
            upper_high=torch.where(
                retreat[:,None],trial["high"],upper_high)
            upper_low=torch.where(
                retreat[:,None],trial["low"],upper_low)
            root_valid&=torch.where(
                root_active,trial["valid"],torch.ones_like(root_active))
            relative=(upper-lower)/torch.maximum(
                torch.maximum(abs(self._coordinate_gather(
                    alpha,state["driver"])),state["knot_upper_width"]),one)
            done|=root_active&(relative<=256*machine)

        candidate=enabled&root_valid&(upper>0)
        root_step={"high":upper_high,"low":upper_low,"valid":root_valid}
        normalized,ok,endpoint_alpha,_,endpoint_temperature=(
            self._coupled_coordinate_error(
                root_step,state["origin_alpha"],state["origin_sensible"],
                alpha,state["driver"],upper,candidate))
        state["rate_evaluations"]+=candidate.to(torch.int64)

        unsnapped_alpha=endpoint_alpha.clone()
        unsnapped_temperature=endpoint_temperature.clone()
        endpoint_alpha=self._coordinate_alpha(
            state["driver"],
            self._coordinate_gather(endpoint_alpha,state["driver"]),target)
        endpoint_sensible=(state["origin_sensible"]
            +(self.Q[None,:]
              *(endpoint_alpha-state["origin_alpha"])).sum(-1))
        endpoint_temperature=self.temperature_from_sensible(endpoint_sensible)
        alpha_tolerance=(self.s.chemistry_absolute_tolerance
            +self.s.chemistry_relative_tolerance*torch.maximum(
                torch.maximum(abs(endpoint_alpha),abs(unsnapped_alpha)),
                torch.ones_like(endpoint_alpha)))
        snap_normalized=2*torch.maximum(
            (abs(endpoint_alpha-unsnapped_alpha)/alpha_tolerance).amax(-1),
            abs(endpoint_temperature-unsnapped_temperature)
                /self.s.chemistry_temperature_tolerance)
        normalized=torch.maximum(normalized,snap_normalized)

        sensible_threshold=(state["sensible"]
            +256*machine*torch.maximum(abs(state["sensible"]),one))
        caloric_candidates=torch.where(
            self.heat_knot_sensible[None,:]>sensible_threshold[:,None],
            self.heat_knot_sensible[None,:],
            torch.full((count,self.heat_knot_sensible.numel()),math.inf,
                       dtype=torch.float64,device=device))
        next_caloric=caloric_candidates.amin(-1)
        # ``root_step`` carries only the embedded endpoints; the saved upper
        # stage data below is selected explicitly during the bisection.
        # Re-evaluate once at the final upper width so kink guards apply to
        # exactly the panel that will be committed.
        final_step=self._coupled_coordinate_dp54_step(
            state["origin_alpha"],state["origin_sensible"],alpha,
            state["elapsed"],state["driver"],upper,candidate)
        state["rate_evaluations"]+=7*candidate.to(torch.int64)
        caloric_cross=final_step["stage_sensible"].amax(-1)>=next_caloric
        start_regions=self._clamp_region(self._unclipped_log_rates(
            alpha,state["temperature"]))
        clamp_cross=(self._clamp_region(final_step["stage_exponents"])
                     !=start_regions[:,None,:]).any((1,2))
        accepted=(candidate&ok&final_step["valid"]
                  &torch.isfinite(normalized)&(normalized<=1)
                  &~caloric_cross&~clamp_cross)
        failed=enabled&~accepted
        finite_failed=failed&torch.isfinite(normalized)
        state["maximum_rejected_residual"]=torch.where(
            finite_failed,torch.maximum(
                state["maximum_rejected_residual"],normalized),
            state["maximum_rejected_residual"])
        state["fallback"]|=failed
        endpoint_rates=self.rates_from_alpha(
            endpoint_alpha,endpoint_temperature)
        driver_rate=self._coordinate_gather(
            endpoint_rates,state["driver"])
        other_rate=self._coordinate_gather(endpoint_rates,other)
        switch=(accepted
            &(other_rate>driver_rate*(1+math.sqrt(machine))))
        state["knot_pending"]&=~enabled
        state=self._commit_coupled_coordinate_step(
            state,endpoint_alpha,upper_high[:,1],normalized,upper,
            accepted,switch)
        return state

    def _resolve_coupled_coordinate_event(self,state):
        """Locate the earliest time/inventory event on compact pending rows."""
        alpha=state["alpha"];device=alpha.device;count=alpha.shape[0]
        enabled=state["event_pending"].clone()
        zero=torch.zeros(count,dtype=torch.float64,device=device)
        lower=zero.clone();upper=state["event_upper_width"].clone()
        lower_high=torch.stack((
            self._coordinate_gather(alpha,1-state["driver"]),
            state["elapsed"]),-1)
        lower_low=lower_high.clone()
        lower_alpha=alpha.clone()
        lower_temperature=state["temperature"].clone()
        lower_normalized=zero.clone()

        original=self._coupled_coordinate_dp54_step(
            state["origin_alpha"],state["origin_sensible"],alpha,
            state["elapsed"],state["driver"],upper,enabled)
        (upper_normalized,upper_valid,upper_alpha,_,upper_temperature)=(
            self._coupled_coordinate_error(
                original,state["origin_alpha"],state["origin_sensible"],
                alpha,state["driver"],upper,enabled))
        state["rate_evaluations"]+=8*enabled.to(torch.int64)
        upper_time=original["high"][:,1]
        upper_progress=(self.weights[None,:]*(upper_alpha-alpha)).sum(-1)
        upper_cross=((upper_time>=state["duration"])
                     |(upper_progress>=state["capacity"]))
        root_valid=torch.where(
            enabled,upper_valid&(upper_normalized<=1)&upper_cross,
            torch.ones_like(enabled))
        done=~enabled
        for _root in range(self.s.max_chemistry_depletion_iterations):
            root_active=enabled&~done&root_valid
            trial_width=.5*(lower+upper)
            trial=self._coupled_coordinate_dp54_step(
                state["origin_alpha"],state["origin_sensible"],alpha,
                state["elapsed"],state["driver"],trial_width,root_active)
            normalized,valid,trial_alpha,_,trial_temperature=(
                self._coupled_coordinate_error(
                    trial,state["origin_alpha"],state["origin_sensible"],
                    alpha,state["driver"],trial_width,root_active))
            state["rate_evaluations"]+=8*root_active.to(torch.int64)
            certified=root_active&valid&(normalized<=1)
            root_valid&=torch.where(
                root_active,certified,torch.ones_like(root_active))
            progress=(self.weights[None,:]*(trial_alpha-alpha)).sum(-1)
            time_cross=trial["high"][:,1]>=state["duration"]
            capacity_cross=progress>=state["capacity"]
            crossed=time_cross|capacity_cross
            advance=certified&~crossed
            retreat=certified&crossed
            lower=torch.where(advance,trial_width,lower)
            upper=torch.where(retreat,trial_width,upper)
            lower_high=torch.where(advance[:,None],trial["high"],lower_high)
            lower_low=torch.where(advance[:,None],trial["low"],lower_low)
            lower_alpha=torch.where(advance[:,None],trial_alpha,lower_alpha)
            lower_temperature=torch.where(
                advance,trial_temperature,lower_temperature)
            lower_normalized=torch.where(
                advance,normalized,lower_normalized)
            upper_alpha=torch.where(retreat[:,None],trial_alpha,upper_alpha)
            upper_temperature=torch.where(
                retreat,trial_temperature,upper_temperature)
            upper_time=torch.where(retreat,trial["high"][:,1],upper_time)
            upper_normalized=torch.where(
                retreat,normalized,upper_normalized)
            alpha_scale=torch.maximum(
                torch.maximum(abs(lower_alpha),abs(upper_alpha)),
                torch.ones_like(lower_alpha))
            alpha_tolerance=(self.s.chemistry_absolute_tolerance
                +self.s.chemistry_relative_tolerance*alpha_scale)
            bracket_normalized=2*torch.maximum(
                (abs(upper_alpha-lower_alpha)/alpha_tolerance).amax(-1),
                abs(upper_temperature-lower_temperature)
                    /self.s.chemistry_temperature_tolerance)
            iteration_lower_progress=(
                self.weights[None,:]*(lower_alpha-alpha)).sum(-1)
            iteration_upper_progress=(
                self.weights[None,:]*(upper_alpha-alpha)).sum(-1)
            iteration_capacity_tolerance=self.progress_tolerance(
                state["capacity0"])
            iteration_inventory_cert=(
                (iteration_upper_progress>=state["capacity"])
                &(state["capacity"]-iteration_lower_progress>=0)
                &(state["capacity"]-iteration_lower_progress
                  <=.5*iteration_capacity_tolerance))
            iteration_time_only=(
                (upper_time>=state["duration"])
                &(iteration_upper_progress<state["capacity"]))
            done|=(root_active&(bracket_normalized<=1)
                   &(iteration_inventory_cert|iteration_time_only))

        lower_progress=(self.weights[None,:]*(lower_alpha-alpha)).sum(-1)
        upper_progress=(self.weights[None,:]*(upper_alpha-alpha)).sum(-1)
        capacity_tolerance=self.progress_tolerance(state["capacity0"])
        lower_capacity_residual=state["capacity"]-lower_progress
        upper_capacity_residual=upper_progress-state["capacity"]
        upper_capacity_cross=upper_progress>=state["capacity"]
        upper_time_cross=upper_time>=state["duration"]
        inventory_cert=(upper_capacity_cross
            &(lower_capacity_residual>=0)
            &(lower_capacity_residual<=.5*capacity_tolerance)
            &(upper_capacity_residual>=0))
        alpha_scale=torch.maximum(
            torch.maximum(abs(lower_alpha),abs(upper_alpha)),
            torch.ones_like(lower_alpha))
        bracket_normalized=2*torch.maximum(
            (abs(upper_alpha-lower_alpha)
             /(self.s.chemistry_absolute_tolerance
               +self.s.chemistry_relative_tolerance*alpha_scale)).amax(-1),
            abs(upper_temperature-lower_temperature)
                /self.s.chemistry_temperature_tolerance)
        # If both predicates fall in the final tolerance-sized bracket,
        # capacity wins only when its non-overshooting lower residual itself
        # satisfies the inventory tolerance.  Otherwise this is the requested
        # target-time endpoint, not a silently manufactured depletion event.
        time_cert=upper_time_cross&~inventory_cert
        accepted=(enabled&root_valid
                  &(lower_normalized<=1)&(upper_normalized<=1)
                  &(bracket_normalized<=1)
                  &(inventory_cert|time_cert))
        state["event_pending"]&=~enabled
        state["fallback"]|=enabled&~accepted
        cumulative=(self.weights[None,:]
            *(lower_alpha-state["origin_alpha"])).sum(-1)
        remaining_capacity=state["capacity0"]-cumulative
        accepted&=torch.isfinite(remaining_capacity)&(remaining_capacity>=0)
        state["fallback"]|=enabled&~accepted
        state["alpha"]=torch.where(
            accepted[:,None],lower_alpha,state["alpha"])
        lower_sensible=(state["origin_sensible"]
            +(self.Q[None,:]
              *(lower_alpha-state["origin_alpha"])).sum(-1))
        state["sensible"]=torch.where(
            accepted,lower_sensible,state["sensible"])
        state["temperature"]=torch.where(
            accepted,lower_temperature,state["temperature"])
        state["capacity"]=torch.where(
            accepted,remaining_capacity,state["capacity"])
        depleted=accepted&inventory_cert
        timed=accepted&time_cert
        state["depleted"]|=depleted
        state["depletion_time"]=torch.where(
            depleted,lower_high[:,1],state["depletion_time"])
        state["elapsed"]=torch.where(
            depleted,lower_high[:,1],
            torch.where(timed,state["duration"],state["elapsed"]))
        state["accepted_steps"]+=accepted.to(torch.int64)
        state["maximum_residual"]=torch.where(
            accepted,torch.maximum(
                state["maximum_residual"],torch.maximum(
                    torch.maximum(lower_normalized,upper_normalized),
                    bracket_normalized)),
            state["maximum_residual"])
        state["retrying"]&=~accepted
        state["success"]|=accepted
        state["active"]&=~(state["success"]|state["fallback"])
        return state

    def _coupled_coordinate_attempt_block(self,state):
        """Eight masked attempts with no tensor-to-host control decisions.

        A caller may compact at block boundaries.  Inside this function there
        is deliberately no ``item/cpu/numpy``, tensor-valued Python condition,
        dynamic indexing, or data-dependent ``break``.
        """
        alpha=state["alpha"];device=alpha.device;count=alpha.shape[0]
        zero=torch.zeros(count,dtype=torch.float64,device=device)
        one=torch.ones_like(zero);machine=torch.finfo(torch.float64).eps
        state_tolerance=torch.full_like(zero,256*machine)
        # Scale roundoff to the requested interval itself.  An absolute
        # O(eps) floor would silently turn valid sub-eps positive half-steps
        # into no-ops under sufficiently stiff raw kinetics.
        time_tolerance=256*machine*abs(state["duration"])

        for _attempt in range(8):
            base_active=(state["active"]&~state["fallback"]
                         &~state["success"])
            active=(base_active&~state["bracket_pending"]
                    &~state["knot_pending"]&~state["event_pending"])
            unfinished=alpha<1-state_tolerance[:,None]
            unfinished_count=unfinished.to(torch.int64).sum(-1)
            time_done=active&(state["elapsed"]>=state["duration"]-time_tolerance)
            depleted=active&(state["capacity"]<=0)
            complete=active&(unfinished_count==0)
            tail=active&(unfinished_count==1)&~time_done&~depleted
            guard=active&(state["accepted_steps"]
                          >=self.s.max_chemistry_reaction_coordinate_steps)
            state["success"]|=time_done|depleted|complete
            state["depleted"]|=depleted
            state["depletion_time"]=torch.where(
                depleted,state["elapsed"],state["depletion_time"])
            state["needs_tail"]|=tail
            state["fallback"]|=guard
            active&=~(time_done|depleted|complete|tail|guard)
            # Keep the stored mask in lockstep before constructing this
            # attempt.  Otherwise a row classified as a tail/end state above
            # could still enter one final DP stage in the same fixed block.
            state["active"]=base_active&~(
                time_done|depleted|complete|tail|guard)

            new_panel=active&~state["retrying"]
            rates=self.rates_from_alpha(alpha,state["temperature"])
            state["rate_evaluations"]+=new_panel.to(torch.int64)
            rates_valid=torch.isfinite(rates).all(-1)&(rates>0).all(-1)
            dominant=torch.argmax(rates,-1)
            high_rate=self._coordinate_gather(rates,dominant)
            low_rate=self._coordinate_gather(rates,1-dominant)
            gap=abs(high_rate-low_rate)/torch.clamp(
                torch.maximum(high_rate,low_rate),
                min=torch.finfo(torch.float64).tiny)
            forced=state["forced_driver"]>=0
            chosen=torch.where(forced,state["forced_driver"],dominant)
            chosen_rate=self._coordinate_gather(rates,chosen)
            other_rate=self._coordinate_gather(rates,1-chosen)
            forced_bad=forced&(chosen_rate<other_rate*(1-math.sqrt(machine)))
            tie=((~forced)&(gap<=math.sqrt(machine))
                 &~torch.full_like(forced,self.coordinate_channels_identical))
            initialization_bad=new_panel&(~rates_valid|tie|forced_bad)
            state["fallback"]|=initialization_bad
            initialize=new_panel&~initialization_bad
            state["driver"]=torch.where(initialize,chosen,state["driver"])
            state["forced_driver"]=torch.where(
                initialize,torch.full_like(state["forced_driver"],-1),
                state["forced_driver"])
            target,completes=self._next_coordinate_knot(alpha,state["driver"])
            driver_value=self._coordinate_gather(alpha,state["driver"])
            span=target-driver_value
            hint=self._coordinate_gather(state["step_hint"],state["driver"])
            width=torch.where(torch.isfinite(hint),torch.minimum(span,hint),span)
            span_bad=initialize&(~torch.isfinite(width)|(width<=0)
                |(driver_value+width==driver_value))
            state["fallback"]|=span_bad
            initialize&=~span_bad
            state["driver_target"]=torch.where(
                initialize,target,state["driver_target"])
            state["completes_driver"]=torch.where(
                initialize,completes,state["completes_driver"])
            state["width"]=torch.where(initialize,width,state["width"])
            state["depth"]=torch.where(
                initialize,torch.zeros_like(state["depth"]),state["depth"])
            state["retrying"]|=initialize

            attempt=active&state["retrying"]&~state["fallback"]
            step=self._coupled_coordinate_dp54_step(
                state["origin_alpha"],state["origin_sensible"],alpha,
                state["elapsed"],state["driver"],state["width"],attempt)
            normalized,step_valid,endpoint_alpha,endpoint_sensible,endpoint_temperature=(
                self._coupled_coordinate_error(
                    step,state["origin_alpha"],state["origin_sensible"],alpha,
                    state["driver"],state["width"],attempt))
            state["rate_evaluations"]+=8*attempt.to(torch.int64)
            reject=attempt&(~step_valid|(normalized>1))
            finite_reject=reject&torch.isfinite(normalized)
            state["maximum_rejected_residual"]=torch.where(
                finite_reject,torch.maximum(
                    state["maximum_rejected_residual"],normalized),
                state["maximum_rejected_residual"])
            state["rejected_steps"]+=reject.to(torch.int64)
            next_depth=state["depth"]+reject.to(torch.int64)
            state["maximum_depth"]=torch.maximum(
                state["maximum_depth"],next_depth)
            over=reject&(next_depth>self.s.max_chemistry_local_refinements)
            state["fallback"]|=over
            state["depth"]=torch.where(reject,next_depth,state["depth"])
            reject_factor=torch.where(
                torch.isfinite(normalized)&(normalized>0),
                torch.clamp(.8*torch.pow(torch.clamp(normalized,min=1e-300),-.2),
                            min=.1,max=.5),
                torch.full_like(normalized,.5))
            retry=reject&~over
            state["width"]=torch.where(
                retry,state["width"]*reject_factor,state["width"])

            candidate=attempt&step_valid&(normalized<=1)&~reject
            other=1-state["driver"]
            other_target,other_completes=self._next_coordinate_knot(
                alpha,other)
            exact_other_target=torch.where(
                other_completes,torch.ones_like(other_target),other_target)
            other_stage=self._coordinate_gather(
                step["stage_alpha"].reshape(-1,2),
                other[:,None].expand(-1,step["stage_alpha"].shape[1]).reshape(-1)
            ).reshape(count,-1)
            non_driver_cross=(other_stage.amax(-1)
                              >=exact_other_target-state_tolerance)
            endpoint_other=self._coordinate_gather(endpoint_alpha,other)
            non_driver_endpoint_cross=(endpoint_other
                >=exact_other_target-state_tolerance)
            isolate_knot=candidate&non_driver_cross&~non_driver_endpoint_cross
            knot_depth=state["depth"]+isolate_knot.to(torch.int64)
            state["rejected_steps"]+=isolate_knot.to(torch.int64)
            state["maximum_depth"]=torch.maximum(
                state["maximum_depth"],knot_depth)
            knot_over=(isolate_knot
                &(knot_depth>self.s.max_chemistry_local_refinements))
            state["fallback"]|=knot_over
            state["depth"]=torch.where(
                isolate_knot,knot_depth,state["depth"])
            state["width"]=torch.where(
                isolate_knot&~knot_over,.5*state["width"],state["width"])
            candidate&=~isolate_knot
            knot=candidate&non_driver_cross&non_driver_endpoint_cross
            state["knot_pending"]|=knot
            state["knot_upper_width"]=torch.where(
                knot,state["width"],state["knot_upper_width"])
            state["knot_target"]=torch.where(
                knot,exact_other_target,state["knot_target"])
            candidate&=~knot
            sensible_threshold=(state["sensible"]
                +256*machine*torch.maximum(abs(state["sensible"]),one))
            caloric_candidates=torch.where(
                self.heat_knot_sensible[None,:]>sensible_threshold[:,None],
                self.heat_knot_sensible[None,:],
                torch.full((count,self.heat_knot_sensible.numel()),math.inf,
                           dtype=torch.float64,device=device))
            next_caloric=caloric_candidates.amin(-1)
            caloric_cross=step["stage_sensible"].amax(-1)>=next_caloric
            start_regions=self._clamp_region(self._unclipped_log_rates(
                alpha,state["temperature"]))
            clamp_cross=(self._clamp_region(step["stage_exponents"])
                         !=start_regions[:,None,:]).any((1,2))
            safety_failure=candidate&(caloric_cross|clamp_cross)
            state["fallback"]|=safety_failure
            candidate&=~safety_failure

            driver_stage=self._coordinate_gather(
                step["stage_rates"].reshape(-1,2),
                state["driver"][:,None].expand(
                    -1,step["stage_rates"].shape[1]).reshape(-1)
            ).reshape(count,-1)
            other_stage_rate=self._coordinate_gather(
                step["stage_rates"].reshape(-1,2),
                other[:,None].expand(-1,step["stage_rates"].shape[1]).reshape(-1)
            ).reshape(count,-1)
            dominance_cross=((other_stage_rate>=driver_stage).any(-1)
                &~torch.full_like(candidate,
                                  self.coordinate_channels_identical))
            endpoint_cross=other_stage_rate[:,-1]>=driver_stage[:,-1]
            isolate=candidate&dominance_cross&~endpoint_cross
            isolate_depth=state["depth"]+isolate.to(torch.int64)
            state["rejected_steps"]+=isolate.to(torch.int64)
            state["maximum_depth"]=torch.maximum(
                state["maximum_depth"],isolate_depth)
            isolate_over=(isolate
                &(isolate_depth>self.s.max_chemistry_local_refinements))
            state["fallback"]|=isolate_over
            state["depth"]=torch.where(isolate,isolate_depth,state["depth"])
            state["width"]=torch.where(
                isolate&~isolate_over,.5*state["width"],state["width"])
            candidate&=~isolate

            bracket=candidate&dominance_cross&endpoint_cross
            state["bracket_pending"]|=bracket
            plain_accept=candidate&~dominance_cross
            state=self._commit_coupled_coordinate_step(
                state,endpoint_alpha,step["high"][:,1],normalized,
                state["width"],plain_accept,
                torch.zeros_like(plain_accept))
            alpha=state["alpha"]
        return state

    def _solve_coupled_reaction_coordinate(
            self,alpha0,sensible0,temperature0,duration,progress_capacity,
            enabled):
        """Compact eight-attempt blocks; block boundaries are the only sync."""
        count=duration.shape[0];device=duration.device
        integer=lambda:torch.zeros(count,dtype=torch.int64,device=device)
        real=lambda:torch.zeros(count,dtype=torch.float64,device=device)
        boolean=lambda:torch.zeros(count,dtype=torch.bool,device=device)
        state={"origin_alpha":alpha0.clone(),"origin_sensible":sensible0.clone(),
            "alpha":alpha0.clone(),"sensible":sensible0.clone(),
            "temperature":temperature0.clone(),"duration":duration.clone(),
            "capacity0":progress_capacity.clone(),"capacity":progress_capacity.clone(),
            "elapsed":real(),"active":enabled.clone(),"success":boolean(),
            "fallback":boolean(),"tail_failure":boolean(),
            "needs_tail":boolean(),"bracket_pending":boolean(),
            "knot_pending":boolean(),"event_pending":boolean(),
            "knot_upper_width":real(),"knot_target":real(),
            "event_upper_width":real(),
            "depleted":boolean(),
            "depletion_time":torch.full_like(duration,math.nan),
            "retrying":boolean(),"driver":integer(),
            "forced_driver":torch.full((count,),-1,dtype=torch.int64,device=device),
            "driver_target":real(),"completes_driver":boolean(),"width":real(),
            "depth":integer(),"step_hint":torch.full((count,2),math.nan,
                dtype=torch.float64,device=device),"rate_evaluations":integer(),
            "accepted_steps":integer(),"rejected_steps":integer(),
            "driver_switches":integer(),"maximum_depth":integer(),
            "maximum_residual":real(),"maximum_rejected_residual":real()}
        # At most max_steps accepted panels, each preceded by max_refinements
        # rejected attempts, plus one boundary classification attempt.
        attempt_bound=(self.s.max_chemistry_reaction_coordinate_steps
            *(self.s.max_chemistry_local_refinements+1)+1)
        # A crossing pauses its row until the block-boundary root pass, so in
        # the worst case one accepted panel consumes one outer block.
        block_bound=attempt_bound

        def resolve_pending(compact,pending_name,resolver):
            """Compact one root cohort at this documented sync boundary."""
            pending_indices=torch.nonzero(
                compact[pending_name]).flatten()
            if pending_indices.numel()==0:
                return compact
            compact_count=compact["alpha"].shape[0]
            pending={key:(value.index_select(0,pending_indices)
                          if value.ndim and value.shape[0]==compact_count
                          else value)
                     for key,value in compact.items()}
            pending=resolver(pending)
            for key,value in pending.items():
                if value.ndim and value.shape[0]==pending_indices.numel():
                    compact[key].index_copy_(0,pending_indices,value)
            return compact

        for _block in range(block_bound):
            unresolved=state["active"]&~state["fallback"]&~state["success"]
            indices=torch.nonzero(unresolved).flatten()
            # Dynamic-shape synchronization is deliberately confined to this
            # block boundary; local CUDA graph execution is disabled.
            if indices.numel()==0:
                break
            compact={key:(value.index_select(0,indices)
                          if value.ndim and value.shape[0]==count else value)
                     for key,value in state.items()}
            compact=self._coupled_coordinate_attempt_block(compact)
            compact=resolve_pending(
                compact,"knot_pending",
                self._resolve_coupled_coordinate_knot)
            compact=resolve_pending(
                compact,"bracket_pending",
                self._resolve_coupled_coordinate_bracket)
            compact=resolve_pending(
                compact,"event_pending",
                self._resolve_coupled_coordinate_event)
            for key,value in compact.items():
                if value.ndim and value.shape[0]==indices.numel():
                    state[key].index_copy_(0,indices,value)
        state["fallback"]|=(state["active"]&~state["success"]
                            &~state["needs_tail"])

        tail=state["needs_tail"]&~state["fallback"]
        state["coupled_rate_evaluations"]=state["rate_evaluations"].clone()
        unfinished=(state["alpha"]<1-256*torch.finfo(torch.float64).eps)
        channel=unfinished.to(torch.int64).argmax(-1)
        tail_duration=torch.clamp(state["duration"]-state["elapsed"],min=0)
        fast=self._solve_one_active_reaction_coordinate(
            state["alpha"],state["sensible"],state["temperature"],
            tail_duration,state["capacity"],channel,tail)
        tail_good=tail&fast["valid"]
        tail_certification=tail&fast["certification_failure"]
        tail_terminal=tail&(
            fast["terminal_failure"]
            |(~fast["valid"]&~fast["certification_failure"]))
        state["fallback"]|=tail_certification
        state["tail_failure"]|=tail_terminal
        state["alpha"]=torch.where(tail_good[:,None],fast["alpha"],state["alpha"])
        for key in ("sensible","temperature","capacity_left","depleted",
                    "depletion_time"):
            target="capacity" if key=="capacity_left" else key
            value=fast[key]
            if key=="depletion_time":
                value=state["elapsed"]+value
            state[target]=torch.where(tail_good,value,state[target])
        state["rate_evaluations"]+=torch.where(
            tail,fast["rate_evaluations"],torch.zeros_like(fast["rate_evaluations"]))
        state["maximum_depth"]=torch.where(
            tail,torch.maximum(state["maximum_depth"],fast["maximum_depth"]),
            state["maximum_depth"])
        state["maximum_residual"]=torch.where(
            tail,torch.maximum(state["maximum_residual"],
                               fast["maximum_normalized_residual"]),
            state["maximum_residual"])
        state["maximum_rejected_residual"]=torch.where(
            tail,torch.maximum(state["maximum_rejected_residual"],
                               fast["maximum_rejected_normalized_residual"]),
            state["maximum_rejected_residual"])
        state["success"]|=tail_good
        state["active"]&=~(
            state["success"]|state["fallback"]|state["tail_failure"])
        state["one_active"]=fast
        state["one_active_enabled"]=tail
        return state

    def _integrate_local_panels(self,alpha0,sensible0,temperature0,duration,
                                progress_capacity,panel_count,enabled=None,
                                accuracy_fraction=None):
        """One- or two-panel map used by the embedded chronological trial."""
        count=duration.shape[0];device=duration.device
        if enabled is None:
            enabled=torch.ones(count,dtype=torch.bool,device=device)
        if accuracy_fraction is None:
            accuracy_fraction=torch.ones_like(duration)
        work=self._local_work(count,device)
        result={"alpha":alpha0.clone(),"sensible":sensible0.clone(),
                "temperature":temperature0.clone(),
                "capacity_left":progress_capacity.clone(),
                "depleted":torch.zeros(count,dtype=torch.bool,device=device),
                "depletion_time":torch.full_like(duration,math.nan),
                "valid":enabled.clone()}
        panel_duration=duration/float(panel_count)
        for panel_number in range(panel_count):
            panel_active=result["valid"]
            panel=self._implicit_midpoint_panel(
                result["alpha"],result["sensible"],result["temperature"],
                torch.where(panel_active,panel_duration,torch.zeros_like(panel_duration)),
                result["capacity_left"],
                0.125*accuracy_fraction/float(panel_count),
            )
            integer_mask=panel_active.to(torch.int64)
            work["evaluated_cell_count"]+=integer_mask
            work["corrector_iteration_sum"]+=integer_mask*panel["corrector_iterations"]
            work["depletion_iteration_sum"]+=integer_mask*panel["depletion_iteration_sum"]
            work["endpoint_evaluations"]+=integer_mask*panel["endpoint_evaluations"]
            for key,panel_key in (
                    ("maximum_corrector_iterations_used","corrector_iterations"),
                    ("maximum_depletion_iterations_used","maximum_depletion_iterations_used")):
                work[key]=torch.where(panel_active,torch.maximum(
                    work[key],panel[panel_key]),work[key])
            finite_residual=torch.where(torch.isfinite(panel["normalized_corrector_residual"]),
                                        panel["normalized_corrector_residual"],
                                        torch.zeros_like(panel["normalized_corrector_residual"]))
            for key,value in (
                    ("maximum_normalized_corrector_residual",finite_residual),
                    ("feedback_alpha_correction",panel["feedback_alpha_correction"]),
                    ("feedback_temperature_correction",panel["feedback_temperature_correction"])):
                work[key]=torch.where(panel_active,torch.maximum(
                    work[key],value),work[key])

            good=panel_active&panel["converged"]
            bad=panel_active&~panel["converged"]
            old_alpha=result["alpha"]
            consumed=(panel["alpha"]-old_alpha)@self.weights
            result["alpha"]=torch.where(good[:,None],panel["alpha"],result["alpha"])
            result["sensible"]=torch.where(good,panel["sensible"],result["sensible"])
            result["temperature"]=torch.where(good,panel["temperature"],result["temperature"])
            capacity=torch.clamp(result["capacity_left"]-consumed,min=0)
            result["capacity_left"]=torch.where(good,capacity,result["capacity_left"])
            newly_depleted=good&panel["depleted"]&~result["depleted"]
            event_times=panel_number*panel_duration+panel["depletion_time"]
            result["depletion_time"]=torch.where(
                newly_depleted,event_times,result["depletion_time"])
            result["depleted"]|=good&panel["depleted"]
            result["valid"]&=~bad
            work["invalid_panel_attempt_count"]+=bad.to(torch.int64)
        result.update(work)
        return result

    def _midpoint_fallback_attempt_block(
            self,result,work,elapsed,panel_size,level,attempt_count,failed,
            active,duration,time_tolerance,
            maximum_attempts):
        """Eight fixed masked midpoint attempts over one compact cohort."""
        count=duration.shape[0];device=duration.device
        indices=torch.arange(count,device=device)
        for _attempt in range(8):
            scheduled=active&(attempt_count<maximum_attempts)
            failed|=active&~scheduled
            active=scheduled
            remaining=duration-elapsed
            q=torch.where(active,torch.minimum(panel_size,remaining),
                          torch.zeros_like(remaining))
            q_ok=torch.isfinite(q)&(q>0)
            attempt_count+=active.to(torch.int64)
            panel_fraction=torch.where(
                duration>0,torch.clamp(q/duration,min=0,max=1),
                torch.zeros_like(duration))
            coarse=self._integrate_local_panels(
                result["alpha"],result["sensible"],result["temperature"],q,
                result["capacity_left"],1,enabled=active,
                accuracy_fraction=panel_fraction)
            fine=self._integrate_local_panels(
                result["alpha"],result["sensible"],result["temperature"],q,
                result["capacity_left"],2,enabled=active,
                accuracy_fraction=panel_fraction)
            self._accumulate_local_work(work,coarse,indices)
            self._accumulate_local_work(work,fine,indices)
            comparable=active&q_ok&coarse["valid"]&fine["valid"]
            alpha_scale=torch.maximum(torch.maximum(
                abs(fine["alpha"]),abs(coarse["alpha"])),
                torch.ones_like(fine["alpha"]))
            embedded_fraction=torch.where(
                duration>0,torch.clamp(.5*q/duration,min=0,max=.5),
                torch.zeros_like(duration))
            alpha_budget=torch.maximum(
                (self.s.chemistry_absolute_tolerance
                 +self.s.chemistry_relative_tolerance*alpha_scale)
                *embedded_fraction[:,None],
                torch.full_like(alpha_scale,
                                256*torch.finfo(torch.float64).eps))
            alpha_residual=(abs(fine["alpha"]-coarse["alpha"])
                            /alpha_budget).amax(-1)
            temperature_budget=torch.maximum(
                self.s.chemistry_temperature_tolerance*embedded_fraction,
                torch.full_like(embedded_fraction,
                                256*torch.finfo(torch.float64).eps))
            temperature_residual=(abs(fine["temperature"]-coarse["temperature"])
                                  /temperature_budget)
            embedded=torch.maximum(alpha_residual,temperature_residual)
            event_time_roundoff=(
                256*torch.finfo(torch.float64).eps
                *torch.maximum(abs(q),torch.full_like(q,torch.finfo(
                    torch.float64).tiny)))
            early_event=(active&fine["valid"]&fine["depleted"]
                &torch.isfinite(fine["depletion_time"])
                &(fine["depletion_time"]<=.5*q+event_time_roundoff))
            passed=(comparable&torch.isfinite(embedded)&(embedded<=1)
                    &~early_event)
            result["alpha"]=torch.where(
                passed[:,None],fine["alpha"],result["alpha"])
            for key in ("sensible","temperature","capacity_left"):
                result[key]=torch.where(passed,fine[key],result[key])
            first_event=(passed&torch.isfinite(fine["depletion_time"])
                         &~torch.isfinite(result["depletion_time"]))
            result["depletion_time"]=torch.where(
                first_event,elapsed+fine["depletion_time"],
                result["depletion_time"])
            result["depleted"]=torch.where(
                passed,fine["depleted"],result["depleted"])
            result["refinement_depth"]=torch.where(
                passed,torch.maximum(result["refinement_depth"],level),
                result["refinement_depth"])
            for key,value in (
                    ("normalized_embedded_residual",embedded),
                    ("normalized_embedded_alpha_residual",alpha_residual),
                    ("normalized_embedded_temperature_residual",
                     temperature_residual)):
                result[key]=torch.where(
                    passed,torch.maximum(result[key],value),result[key])
            elapsed+=torch.where(passed,q,torch.zeros_like(q))
            result["accepted_panel_count"]+=passed.to(torch.int64)
            completed=(result["alpha"]
                       >=1-256*torch.finfo(torch.float64).eps).all(-1)
            exhausted=result["capacity_left"]<=0
            newly_exhausted=passed&exhausted&~result["depleted"]
            result["depleted"]|=newly_exhausted
            result["depletion_time"]=torch.where(
                newly_exhausted,elapsed,result["depletion_time"])
            finished=duration-elapsed<=time_tolerance
            done=passed&(completed|exhausted|finished)
            continuing=passed&~done
            grow=continuing&(embedded<=.125)&(level>0)
            panel_size=torch.where(grow,2*panel_size,panel_size)
            level-=grow.to(torch.int64)
            panel_size=torch.where(
                continuing,torch.minimum(panel_size,duration-elapsed),panel_size)

            rejected=active&~passed
            finite_failed=torch.where(
                torch.isfinite(embedded),embedded,torch.zeros_like(embedded))
            result["maximum_rejected_embedded_residual"]=torch.where(
                rejected,torch.maximum(
                    result["maximum_rejected_embedded_residual"],finite_failed),
                result["maximum_rejected_embedded_residual"])
            result["refined"]|=rejected
            next_level=level+rejected.to(torch.int64)
            result["refinement_depth"]=torch.where(
                rejected,torch.maximum(result["refinement_depth"],next_level),
                result["refinement_depth"])
            over=(rejected
                  &(next_level>self.s.max_chemistry_local_refinements))
            failed|=over
            level=torch.where(rejected,next_level,level)
            retry=rejected&~over
            ordinary_size=.5*panel_size
            event_size=torch.minimum(.5*q,.5*fine["depletion_time"])
            retry_size=torch.where(early_event,event_size,ordinary_size)
            panel_size=torch.where(retry,retry_size,panel_size)
            active=continuing|retry
        return (result,work,elapsed,panel_size,level,attempt_count,failed,
                active)

    def _solve_local_cells(self,alpha0,sensible0,temperature0,duration,
                           progress_capacity):
        """Bulk first attempt, then one packed hard-cell scheduler.

        Torch index tensors compact hard cells, binding inventory roots, and
        unresolved correctors.  Once a hard-cell cohort is packed, its
        chronological one-channel/coupled fallback uses device masks.
        """
        count=duration.shape[0];device=duration.device
        work=self._local_work(count,device)
        result={"alpha":alpha0.clone(),"sensible":sensible0.clone(),
                "temperature":temperature0.clone(),
                "capacity_left":progress_capacity.clone(),
                "depleted":torch.zeros(count,dtype=torch.bool,device=device),
                "depletion_time":torch.full_like(duration,math.nan),
                "refinement_depth":torch.zeros(count,dtype=torch.int64,device=device),
                "normalized_embedded_residual":torch.zeros_like(duration),
                "normalized_embedded_alpha_residual":torch.zeros_like(duration),
                "normalized_embedded_temperature_residual":torch.zeros_like(duration),
                "maximum_rejected_embedded_residual":torch.zeros_like(duration),
                "refined":torch.zeros(count,dtype=torch.bool,device=device),
                "accepted_panel_count":torch.zeros(count,dtype=torch.int64,device=device)}
        elapsed=torch.zeros_like(duration);panel_size=duration.clone()
        level=torch.zeros(count,dtype=torch.int64,device=device)
        attempt_count=torch.zeros_like(level)
        failed_budget=torch.zeros(count,dtype=torch.bool,device=device)
        time_tolerance=256*torch.finfo(torch.float64).eps*abs(duration)
        uncompleted=(alpha0<1-256*torch.finfo(torch.float64).eps).any(-1)
        timed=duration>time_tolerance
        unfinished_channels=alpha0<1-256*torch.finfo(torch.float64).eps
        next_alpha=torch.nextafter(alpha0,torch.ones_like(alpha0))
        progress_quantum=torch.where(
            unfinished_channels,
            self.weights[None,:]*(next_alpha-alpha0),
            torch.full_like(alpha0,math.inf)).amin(-1)
        representation_exhausted=((progress_capacity>0)
                                  &(progress_capacity<progress_quantum))
        initially_depleted=(progress_capacity<=0)&uncompleted&timed
        result["depleted"]|=initially_depleted
        result["depletion_time"]=torch.where(
            initially_depleted,torch.zeros_like(duration),result["depletion_time"])
        has_capacity=(progress_capacity>0)&~representation_exhausted
        one_active=(unfinished_channels.to(torch.int64).sum(-1)==1)&has_capacity&timed
        coupled_active=has_capacity&uncompleted&timed&~one_active

        # One domain-wide embedded attempt.  Passing cells necessarily consume
        # their whole requested duration; only failures need chronological work.
        coarse=self._integrate_local_panels(
            alpha0,sensible0,temperature0,duration,progress_capacity,1,
            enabled=coupled_active)
        fine=self._integrate_local_panels(
            alpha0,sensible0,temperature0,duration,progress_capacity,2,
            enabled=coupled_active)
        all_indices=torch.arange(count,device=device)
        self._accumulate_local_work(work,coarse,all_indices)
        self._accumulate_local_work(work,fine,all_indices)
        attempt_count+=coupled_active.to(torch.int64)
        alpha_scale=torch.maximum(torch.maximum(abs(fine["alpha"]),abs(coarse["alpha"])),
                                  torch.ones_like(fine["alpha"]))
        alpha_budget=torch.maximum(
            .5*(self.s.chemistry_absolute_tolerance
                +self.s.chemistry_relative_tolerance*alpha_scale),
            torch.full_like(alpha_scale,256*torch.finfo(torch.float64).eps))
        alpha_residual=(abs(fine["alpha"]-coarse["alpha"])
                        /alpha_budget).amax(-1)
        temperature_budget=max(
            .5*self.s.chemistry_temperature_tolerance,
            256*torch.finfo(torch.float64).eps)
        temperature_residual=(abs(fine["temperature"]-coarse["temperature"])
                              /temperature_budget)
        embedded=torch.maximum(alpha_residual,temperature_residual)
        event_time_roundoff=(
            256*torch.finfo(torch.float64).eps
            *torch.maximum(abs(duration),torch.full_like(
                duration,torch.finfo(torch.float64).tiny)))
        early_event=(coupled_active&fine["valid"]&fine["depleted"]
            &torch.isfinite(fine["depletion_time"])
            &(fine["depletion_time"]
              <=.5*duration+event_time_roundoff))
        passed=(coupled_active&coarse["valid"]&fine["valid"]
                &torch.isfinite(embedded)&(embedded<=1)&~early_event)
        result["alpha"]=torch.where(passed[:,None],fine["alpha"],result["alpha"])
        result["sensible"]=torch.where(passed,fine["sensible"],result["sensible"])
        result["temperature"]=torch.where(passed,fine["temperature"],result["temperature"])
        result["capacity_left"]=torch.where(
            passed,fine["capacity_left"],result["capacity_left"])
        result["depleted"]=torch.where(passed,fine["depleted"],result["depleted"])
        first_event=passed&torch.isfinite(fine["depletion_time"])
        result["depletion_time"]=torch.where(
            first_event,fine["depletion_time"],result["depletion_time"])
        for key,value in (
                ("normalized_embedded_residual",embedded),
                ("normalized_embedded_alpha_residual",alpha_residual),
                ("normalized_embedded_temperature_residual",temperature_residual)):
            result[key]=torch.where(passed,torch.maximum(result[key],value),result[key])
        elapsed=torch.where(passed,duration,elapsed)
        result["accepted_panel_count"]+=passed.to(torch.int64)
        newly_exhausted=(passed
            &(result["capacity_left"]<=0)&~result["depleted"])
        result["depleted"]|=newly_exhausted
        result["depletion_time"]=torch.where(
            newly_exhausted,elapsed,result["depletion_time"])

        coupled_failed=coupled_active&~passed
        finite_failed=torch.where(
            torch.isfinite(embedded),embedded,torch.zeros_like(embedded))
        result["maximum_rejected_embedded_residual"]=torch.where(
            coupled_failed,torch.maximum(
                result["maximum_rejected_embedded_residual"],finite_failed),
            result["maximum_rejected_embedded_residual"])

        # Pack only one-active cells and coupled bulk failures rather than the
        # complete field.  The hard cohort changes with state and half-step.
        hard_mask=one_active|coupled_failed
        hard_indices=torch.nonzero(hard_mask).flatten()
        p_result={key:value.index_select(0,hard_indices)
                  for key,value in result.items()}
        p_work={key:value.index_select(0,hard_indices)
                for key,value in work.items()}
        p_alpha0=alpha0.index_select(0,hard_indices)
        p_sensible0=sensible0.index_select(0,hard_indices)
        p_temperature0=temperature0.index_select(0,hard_indices)
        p_duration=duration.index_select(0,hard_indices)
        p_progress_capacity=progress_capacity.index_select(0,hard_indices)
        p_time_tolerance=time_tolerance.index_select(0,hard_indices)
        p_one=one_active.index_select(0,hard_indices)
        p_coupled=coupled_failed.index_select(0,hard_indices)
        p_elapsed=elapsed.index_select(0,hard_indices)
        p_panel_size=panel_size.index_select(0,hard_indices)
        p_level=level.index_select(0,hard_indices)
        p_attempt_count=attempt_count.index_select(0,hard_indices)
        p_failed=failed_budget.index_select(0,hard_indices)
        p_early_event=early_event.index_select(0,hard_indices)
        p_fine_depletion_time=fine["depletion_time"].index_select(
            0,hard_indices)

        channel_index=unfinished_channels.index_select(
            0,hard_indices).to(torch.int64).argmax(-1)
        fast=self._solve_one_active_reaction_coordinate(
            p_alpha0,p_sensible0,p_temperature0,p_duration,
            p_progress_capacity,channel_index,p_one)
        fast_good=p_one&fast["valid"]
        fast_fallback=p_one&fast["certification_failure"]
        fast_failure=(p_one
            &(fast["terminal_failure"]
              |(~fast["valid"]&~fast["certification_failure"])))
        p_failed|=fast_failure
        p_result["alpha"]=torch.where(
            fast_good[:,None],fast["alpha"],p_result["alpha"])
        for key in ("sensible","temperature","capacity_left","depleted",
                    "depletion_time"):
            p_result[key]=torch.where(fast_good,fast[key],p_result[key])
        p_result["refinement_depth"]=torch.where(
            fast_good,fast["maximum_depth"],p_result["refinement_depth"])
        p_result["normalized_embedded_residual"]=torch.where(
            fast_good,fast["maximum_normalized_residual"],
            p_result["normalized_embedded_residual"])
        p_result["normalized_embedded_alpha_residual"]=torch.where(
            fast_good,fast["maximum_normalized_alpha_residual"],
            p_result["normalized_embedded_alpha_residual"])
        p_result["normalized_embedded_temperature_residual"]=torch.where(
            fast_good,fast["maximum_normalized_temperature_residual"],
            p_result["normalized_embedded_temperature_residual"])
        p_result["maximum_rejected_embedded_residual"]=torch.where(
            fast_good,fast["maximum_rejected_normalized_residual"],
            p_result["maximum_rejected_embedded_residual"])
        p_result["refined"]|=p_one&(fast["maximum_depth"]>0)
        p_result["accepted_panel_count"]+=fast_good.to(torch.int64)
        p_attempt_count+=fast_good.to(torch.int64)
        p_work["endpoint_evaluations"]+=torch.where(
            fast_good,fast["quadrature_panel_evaluations"],
            torch.zeros_like(fast["quadrature_panel_evaluations"]))
        p_work["evaluated_cell_count"]+=fast_good.to(torch.int64)
        p_work["feedback_alpha_correction"]=torch.where(
            fast_good,fast["feedback_alpha_correction"],
            p_work["feedback_alpha_correction"])
        p_work["feedback_temperature_correction"]=torch.where(
            fast_good,fast["feedback_temperature_correction"],
            p_work["feedback_temperature_correction"])
        p_work["reaction_coordinate_rate_evaluations"]+=torch.where(
            fast_good,fast["rate_evaluations"],
            torch.zeros_like(fast["rate_evaluations"]))
        p_work["reaction_coordinate_quadrature_panel_evaluations"]+=torch.where(
            fast_good,fast["quadrature_panel_evaluations"],
            torch.zeros_like(fast["quadrature_panel_evaluations"]))
        p_work["reaction_coordinate_root_iterations"]+=torch.where(
            fast_good,fast["root_iterations"],
            torch.zeros_like(fast["root_iterations"]))
        p_work["reaction_coordinate_cell_count"]+=fast_good.to(torch.int64)
        p_work["maximum_reaction_coordinate_quadrature_depth"]=torch.where(
            fast_good,fast["maximum_depth"],
            p_work["maximum_reaction_coordinate_quadrature_depth"])
        p_work["maximum_reaction_coordinate_normalized_residual"]=torch.where(
            fast_good,fast["maximum_normalized_residual"],
            p_work["maximum_reaction_coordinate_normalized_residual"])

        # A failed bulk embedded panel gets one error-controlled dominant-
        # coordinate attempt before entering the conservative midpoint fallback.
        # Any coordinate guard, knot/event tolerance failure, or exhausted
        # retry budget leaves p_result untouched and follows that same fallback.
        coordinate=self._solve_coupled_reaction_coordinate(
            p_alpha0,p_sensible0,p_temperature0,p_duration,
            p_progress_capacity,p_coupled)
        coordinate_good=(p_coupled&coordinate["success"]
                         &~coordinate["fallback"]
                         &~coordinate["tail_failure"])
        coordinate_failure=p_coupled&coordinate["tail_failure"]
        coordinate_fallback=p_coupled&coordinate["fallback"]
        p_failed|=coordinate_failure
        p_work["coupled_coordinate_attempt_count"]+=p_coupled.to(torch.int64)
        p_work["coupled_coordinate_fallback_count"]+=(
            coordinate_fallback.to(torch.int64))
        p_work["coupled_coordinate_cell_count"]+=coordinate_good.to(torch.int64)
        # The scalar fallback signal does not carry partially completed DP
        # work.  Keep these detailed counters backend-consistent and explicitly
        # success-scoped; complete attempted cost remains in the phase wall
        # clock and attempt/fallback counts.
        p_work["coupled_coordinate_rate_evaluations"]+=torch.where(
            coordinate_good,coordinate["coupled_rate_evaluations"],
            torch.zeros_like(coordinate["coupled_rate_evaluations"]))
        p_work["coupled_coordinate_accepted_steps"]+=torch.where(
            coordinate_good,coordinate["accepted_steps"],
            torch.zeros_like(coordinate["accepted_steps"]))
        p_work["coupled_coordinate_rejected_steps"]+=torch.where(
            coordinate_good,coordinate["rejected_steps"],
            torch.zeros_like(coordinate["rejected_steps"]))
        p_work["coupled_coordinate_driver_switches"]+=torch.where(
            coordinate_good,coordinate["driver_switches"],
            torch.zeros_like(coordinate["driver_switches"]))
        p_work["maximum_coupled_coordinate_depth"]=torch.where(
            coordinate_good,torch.maximum(
                p_work["maximum_coupled_coordinate_depth"],
                coordinate["maximum_depth"]),
            p_work["maximum_coupled_coordinate_depth"])
        p_work["maximum_coupled_coordinate_normalized_residual"]=torch.where(
            coordinate_good,torch.maximum(
                p_work["maximum_coupled_coordinate_normalized_residual"],
                coordinate["maximum_residual"]),
            p_work["maximum_coupled_coordinate_normalized_residual"])
        p_work["maximum_rejected_coupled_coordinate_residual"]=torch.where(
            coordinate_good,torch.maximum(
                p_work["maximum_rejected_coupled_coordinate_residual"],
                coordinate["maximum_rejected_residual"]),
            p_work["maximum_rejected_coupled_coordinate_residual"])

        # Match the scalar telemetry contract: a rejected bulk pair is first
        # offered to the dominant-coordinate solver.  It becomes a local
        # midpoint refinement only when that rescue explicitly falls back.
        p_result["refined"]|=coordinate_fallback
        p_level+=coordinate_fallback.to(torch.int64)
        p_result["refinement_depth"]=torch.where(
            coordinate_fallback,torch.maximum(
                p_result["refinement_depth"],p_level),
            p_result["refinement_depth"])
        initial_retry_size=torch.where(
            p_early_event,
            torch.minimum(.5*p_duration,.5*p_fine_depletion_time),
            .5*p_panel_size)
        p_panel_size=torch.where(
            coordinate_fallback,initial_retry_size,p_panel_size)

        p_result["alpha"]=torch.where(
            coordinate_good[:,None],coordinate["alpha"],p_result["alpha"])
        p_result["sensible"]=torch.where(
            coordinate_good,coordinate["sensible"],p_result["sensible"])
        p_result["temperature"]=torch.where(
            coordinate_good,coordinate["temperature"],p_result["temperature"])
        p_result["capacity_left"]=torch.where(
            coordinate_good,coordinate["capacity"],p_result["capacity_left"])
        p_result["depleted"]=torch.where(
            coordinate_good,coordinate["depleted"],p_result["depleted"])
        coordinate_event=coordinate_good&torch.isfinite(
            coordinate["depletion_time"])
        p_result["depletion_time"]=torch.where(
            coordinate_event,coordinate["depletion_time"],
            p_result["depletion_time"])
        p_result["refinement_depth"]=torch.where(
            coordinate_good,torch.maximum(
                p_result["refinement_depth"],coordinate["maximum_depth"]),
            p_result["refinement_depth"])
        for key in ("normalized_embedded_residual",
                    "normalized_embedded_alpha_residual",
                    "normalized_embedded_temperature_residual"):
            p_result[key]=torch.where(
                coordinate_good,torch.maximum(
                    p_result[key],coordinate["maximum_residual"]),p_result[key])
        p_result["maximum_rejected_embedded_residual"]=torch.where(
            coordinate_good,torch.maximum(
                p_result["maximum_rejected_embedded_residual"],
                coordinate["maximum_rejected_residual"]),
            p_result["maximum_rejected_embedded_residual"])
        p_result["accepted_panel_count"]+=coordinate_good.to(torch.int64)
        p_elapsed=torch.where(coordinate_good,p_duration,p_elapsed)

        tail_enabled=(coordinate_good&coordinate["one_active_enabled"])
        tail_work=coordinate["one_active"]
        p_work["endpoint_evaluations"]+=torch.where(
            tail_enabled,tail_work["quadrature_panel_evaluations"],
            torch.zeros_like(tail_work["quadrature_panel_evaluations"]))
        p_work["evaluated_cell_count"]+=tail_enabled.to(torch.int64)
        p_work["reaction_coordinate_rate_evaluations"]+=torch.where(
            tail_enabled,tail_work["rate_evaluations"],
            torch.zeros_like(tail_work["rate_evaluations"]))
        p_work["reaction_coordinate_quadrature_panel_evaluations"]+=torch.where(
            tail_enabled,tail_work["quadrature_panel_evaluations"],
            torch.zeros_like(tail_work["quadrature_panel_evaluations"]))
        p_work["reaction_coordinate_root_iterations"]+=torch.where(
            tail_enabled,tail_work["root_iterations"],
            torch.zeros_like(tail_work["root_iterations"]))
        p_work["reaction_coordinate_cell_count"]+=tail_enabled.to(torch.int64)
        p_work["maximum_reaction_coordinate_quadrature_depth"]=torch.where(
            tail_enabled,torch.maximum(
                p_work["maximum_reaction_coordinate_quadrature_depth"],
                tail_work["maximum_depth"]),
            p_work["maximum_reaction_coordinate_quadrature_depth"])
        p_work["maximum_reaction_coordinate_normalized_residual"]=torch.where(
            tail_enabled,torch.maximum(
                p_work["maximum_reaction_coordinate_normalized_residual"],
                tail_work["maximum_normalized_residual"]),
            p_work["maximum_reaction_coordinate_normalized_residual"])

        p_active=(coordinate_fallback|fast_fallback)&~p_failed
        p_elapsed=torch.where(fast_fallback,torch.zeros_like(p_elapsed),p_elapsed)
        p_panel_size=torch.where(fast_fallback,p_duration,p_panel_size)
        p_level=torch.where(fast_fallback,torch.zeros_like(p_level),p_level)
        # A binary chronological tree of depth R contains at most 2**R
        # accepted leaves.  Four attempts per leaf covers each accepted leaf,
        # its rejected parent, and conservative bookkeeping margin without an
        # arbitrary user-facing panel-attempt control.
        maximum_attempts=4*(2**self.s.max_chemistry_local_refinements)
        # Recompact at each eight-attempt boundary.  This is the only dynamic
        # shape/synchronization point in the midpoint fallback; each device
        # block itself executes a fixed masked schedule.
        block_count=(maximum_attempts+7)//8
        for _block in range(block_count):
            block_indices=torch.nonzero(p_active).flatten()
            if block_indices.numel()==0:
                break
            c_result={key:value.index_select(0,block_indices)
                      for key,value in p_result.items()}
            c_work={key:value.index_select(0,block_indices)
                    for key,value in p_work.items()}
            c_elapsed=p_elapsed.index_select(0,block_indices)
            c_panel_size=p_panel_size.index_select(0,block_indices)
            c_level=p_level.index_select(0,block_indices)
            c_attempt_count=p_attempt_count.index_select(0,block_indices)
            c_failed=p_failed.index_select(0,block_indices)
            c_active=p_active.index_select(0,block_indices)
            (c_result,c_work,c_elapsed,c_panel_size,c_level,c_attempt_count,
             c_failed,c_active)=self._midpoint_fallback_attempt_block(
                c_result,c_work,c_elapsed,c_panel_size,c_level,
                c_attempt_count,c_failed,c_active,
                p_duration.index_select(0,block_indices),
                p_time_tolerance.index_select(0,block_indices),
                maximum_attempts)
            for key,value in c_result.items():
                p_result[key].index_copy_(0,block_indices,value)
            for key,value in c_work.items():
                p_work[key].index_copy_(0,block_indices,value)
            p_elapsed.index_copy_(0,block_indices,c_elapsed)
            p_panel_size.index_copy_(0,block_indices,c_panel_size)
            p_level.index_copy_(0,block_indices,c_level)
            p_attempt_count.index_copy_(0,block_indices,c_attempt_count)
            p_failed.index_copy_(0,block_indices,c_failed)
            p_active.index_copy_(0,block_indices,c_active)

        p_failed|=p_active
        for key,value in p_result.items():
            result[key].index_copy_(0,hard_indices,value)
        for key,value in p_work.items():
            work[key].index_copy_(0,hard_indices,value)
        attempt_count.index_copy_(0,hard_indices,p_attempt_count)
        failed_budget.index_copy_(0,hard_indices,p_failed)
        result.update(work);result["valid"]=~failed_budget
        result["panel_attempt_count"]=attempt_count
        return result

    def advance_local(self,U,dt):
        """Finite-time conservative local chemistry map for independent lanes."""
        B=U.shape[0];cells=U.shape[1]*U.shape[2]
        flat=U.reshape(-1,NCONS);rho=flat[:,RHO]
        dt_cell=dt[:,None].expand(B,cells).reshape(-1)
        machine=torch.finfo(torch.float64).eps
        safe_rho=torch.where(rho>0,rho,torch.ones_like(rho))
        alpha0=flat[:,A1:A2+1]/safe_rho[:,None]
        alpha_input_roundoff=torch.maximum(
            256*machine*torch.maximum(abs(alpha0),torch.ones_like(alpha0)),
            torch.full_like(alpha0,1.0e-12))
        inventory0=flat[:,[CATION,ANION,PVA]]
        inventory_input_roundoff=torch.maximum(
            256*machine*torch.maximum(abs(inventory0),torch.ones_like(inventory0)),
            1.0e-12*safe_rho[:,None])
        input_valid=(torch.isfinite(flat).all(-1)&torch.isfinite(dt_cell)&(dt_cell>=0)
                     &(rho>0)&(alpha0>=-alpha_input_roundoff).all(-1)
                     &(alpha0<=1+alpha_input_roundoff).all(-1)
                     &(inventory0>=-inventory_input_roundoff).all(-1))
        alpha0=torch.clamp(alpha0,min=0,max=1)
        vx=flat[:,MX]/safe_rho;vy=flat[:,MY]/safe_rho
        internal0=flat[:,ENERGY]/safe_rho-.5*(vx*vx+vy*vy)
        sensible0=internal0-self.cold(safe_rho)
        temperature0=self.temperature_from_sensible(sensible0)
        temperature_roundoff=256*machine*torch.maximum(
            abs(temperature0),torch.ones_like(temperature0))
        input_valid&=(torch.isfinite(temperature0)
                      &(temperature0>=self.tmin-temperature_roundoff)
                      &(temperature0<=self.tmax+temperature_roundoff))
        salt=.5*(flat[:,CATION]+flat[:,ANION])
        maximum_extent=torch.minimum(torch.minimum(
            torch.clamp(salt-self.floor,min=0)/1.45,torch.clamp(flat[:,PVA],min=0)),
            torch.minimum(torch.clamp(flat[:,CATION],min=0)/1.45,
                          torch.clamp(flat[:,ANION],min=0)/1.45))
        xi_cell=self.xi.expand(B,self.ny,self.nx).reshape(-1)
        progress_capacity=maximum_extent/(safe_rho*torch.clamp(xi_cell,min=1e-300))
        progress_capacity=torch.where(input_valid,progress_capacity,torch.zeros_like(progress_capacity))
        solved=self._solve_local_cells(alpha0,sensible0,temperature0,dt_cell,
                                       progress_capacity)
        delta=solved["alpha"]-alpha0
        alpha_roundoff=256*machine*torch.maximum(
            torch.maximum(abs(solved["alpha"]),abs(alpha0)),
            torch.ones_like(alpha0))
        local_valid=(solved["valid"]&torch.isfinite(delta).all(-1)
                     &(delta>=-alpha_roundoff).all(-1)
                     &(solved["alpha"]<=1+alpha_roundoff).all(-1))
        delta=torch.clamp(delta,min=0)
        d_rho_alpha=safe_rho[:,None]*delta
        energy_increment=d_rho_alpha@self.Q
        extent_increment=xi_cell*(d_rho_alpha@self.weights)
        after=flat.clone()
        after[:,A1:A2+1]+=d_rho_alpha
        after[:,ENERGY]+=energy_increment
        after[:,CATION]-=1.45*extent_increment
        after[:,ANION]-=1.45*extent_increment
        after[:,PVA]-=extent_increment
        after[:,PRODUCT_WATER]+=2*extent_increment
        alpha_after=after[:,A1:A2+1]/safe_rho[:,None]
        final_sensible=sensible0+delta@self.Q
        final_temperature=self.temperature_from_sensible(final_sensible)
        temperature_output_roundoff=256*machine*torch.maximum(
            torch.maximum(abs(final_temperature),abs(temperature0)),
            torch.ones_like(final_temperature))
        heat_residual=after[:,ENERGY]-flat[:,ENERGY]-energy_increment
        event_residual=torch.where(solved["depleted"],
                                   extent_increment-maximum_extent,torch.zeros_like(rho))
        allowed_event=(safe_rho*xi_cell
                       *self.progress_tolerance(progress_capacity))
        closure=(abs(heat_residual)<=64*torch.finfo(torch.float64).eps
                 *torch.maximum(abs(after[:,ENERGY]),torch.ones_like(rho)))&(abs(event_residual)<=allowed_event)
        inventory_before=flat[:,[CATION,ANION,PVA]]
        inventory_after=after[:,[CATION,ANION,PVA]]
        inventory_roundoff=torch.maximum(
            256*machine*torch.maximum(
                torch.maximum(abs(inventory_before),abs(inventory_after)),
                torch.ones_like(inventory_after)),
            1.0e-12*safe_rho[:,None])
        alpha_bound_roundoff=torch.maximum(
            alpha_roundoff,torch.full_like(alpha_roundoff,1.0e-12))
        endpoint_numerical_valid=(local_valid
                      &torch.isfinite(after).all(-1)&torch.isfinite(final_temperature)
                      &(alpha_after>=-alpha_bound_roundoff).all(-1)
                      &(alpha_after<=1+alpha_bound_roundoff).all(-1)
                      &(inventory_after>=-inventory_roundoff).all(-1)
                      &(final_temperature>=self.tmin-temperature_output_roundoff)
                      &input_valid)
        temperature_above=(final_temperature>self.tmax+temperature_output_roundoff)
        # The same endpoint boundary/precedence as NumPy, before later budget
        # certification. Structured lane diagnostics survive state rollback;
        # no evolving field or compacted-cell index crosses to the host.
        range_exceeded=(endpoint_numerical_valid.reshape(B,cells).all(-1)
                        &temperature_above.reshape(B,cells).any(-1)
                        &(dt>=0)&torch.isfinite(dt))
        local_valid=(endpoint_numerical_valid
                     &(final_temperature<=self.tmax+temperature_output_roundoff)
                     &closure)
        cell_valid=local_valid.reshape(B,cells)
        lane_valid=cell_valid.all(-1)&(dt>=0)&torch.isfinite(dt)
        after=after.reshape_as(U)
        after=torch.where(lane_valid[:,None,None,None],after,U)

        initial_rates=self.rates(U,self.primitive(U)[...,3]).reshape(B,cells,2)
        endpoint_rates=self.rates(after,self.primitive(after)[...,3]).reshape(B,cells,2)
        traversed_rates=self.traversed_rate_upper_bound(
            alpha0,solved["alpha"],final_temperature).reshape(B,cells)
        initial_exceed=(initial_rates>self.maxrate).any(-1).double().mean(-1)
        endpoint_exceed=(endpoint_rates>self.maxrate).any(-1).double().mean(-1)
        traversed_exceed=(traversed_rates>self.maxrate).double().mean(-1)
        def view(value):return value.reshape(B,cells)
        def maximum(value):return view(value).amax(-1)
        def total(value):return view(value).sum(-1)
        depletion_fraction=solved["depletion_time"]/torch.where(
            dt_cell>0,dt_cell,torch.ones_like(dt_cell))
        finite_depletion=torch.isfinite(depletion_fraction)
        minimum_depletion=view(torch.where(
            finite_depletion,depletion_fraction,torch.full_like(depletion_fraction,math.inf))).amin(-1)
        maximum_depletion=view(torch.where(
            finite_depletion,depletion_fraction,torch.zeros_like(depletion_fraction))).amax(-1)
        caloric_residual=abs(self.sensible(final_temperature)-final_sensible)
        cell_indices=torch.arange(cells,device=U.device).expand(B,cells)
        offending_index=torch.where(
            temperature_above.reshape(B,cells),cell_indices,cells).amin(-1)
        offending_roundoff=view(temperature_output_roundoff).gather(
            1,torch.clamp(offending_index,max=cells-1)[:,None]).squeeze(-1)
        diagnostic=torch.stack((
            (dt>0).double(),lane_valid.double(),torch.maximum(
                torch.maximum(initial_exceed,endpoint_exceed),traversed_exceed),
            view(solved["depleted"].double()).mean(-1),
            torch.maximum(torch.maximum(initial_rates.amax((1,2)),
                                        endpoint_rates.amax((1,2))),
                          traversed_rates.amax(-1)),
            maximum(solved["maximum_corrector_iterations_used"]).double(),
            maximum(solved["maximum_depletion_iterations_used"]).double(),
            maximum(solved["refinement_depth"]).double(),
            maximum(solved["maximum_normalized_corrector_residual"]),
            maximum(solved["normalized_embedded_residual"]),
            maximum(solved["normalized_embedded_alpha_residual"]),
            maximum(solved["normalized_embedded_temperature_residual"]),
            maximum(solved["maximum_rejected_embedded_residual"]),
            maximum(solved["feedback_alpha_correction"]),
            maximum(solved["feedback_temperature_correction"]),
            view(final_temperature-temperature0).amax(-1),
            total(solved["corrector_iteration_sum"]).double(),
            total(solved["depletion_iteration_sum"]).double(),
            total(solved["endpoint_evaluations"]).double(),
            total(solved["evaluated_cell_count"]).double(),
            view(solved["refined"].double()).mean(-1),
            initial_rates.amax((1,2)),endpoint_rates.amax((1,2)),
            minimum_depletion,maximum_depletion,maximum(caloric_residual),
            maximum(abs(heat_residual)),maximum(abs(event_residual)),
            (~cell_valid).sum(-1).double(),
            total(solved["invalid_panel_attempt_count"]).double(),
            total(solved["panel_attempt_count"]).double(),
            total(solved["accepted_panel_count"]).double(),
            maximum(solved["panel_attempt_count"]).double(),
            maximum(solved["accepted_panel_count"]).double(),
            initial_exceed,endpoint_exceed,traversed_exceed,
            traversed_rates.amax(-1),
            total(solved["reaction_coordinate_cell_count"]).double(),
            view(solved["reaction_coordinate_cell_count"].double()).mean(-1),
            total(solved["reaction_coordinate_rate_evaluations"]).double(),
            total(solved[
                "reaction_coordinate_quadrature_panel_evaluations"]).double(),
            total(solved["reaction_coordinate_root_iterations"]).double(),
            maximum(solved["reaction_coordinate_root_iterations"]).double(),
            maximum(solved[
                "maximum_reaction_coordinate_quadrature_depth"]).double(),
            maximum(solved[
                "maximum_reaction_coordinate_normalized_residual"]),
            total(solved["coupled_coordinate_attempt_count"]).double(),
            total(solved["coupled_coordinate_cell_count"]).double(),
            view(solved["coupled_coordinate_cell_count"].double()).mean(-1),
            total(solved["coupled_coordinate_fallback_count"]).double(),
            total(solved["coupled_coordinate_rate_evaluations"]).double(),
            total(solved["coupled_coordinate_accepted_steps"]).double(),
            total(solved["coupled_coordinate_rejected_steps"]).double(),
            total(solved["coupled_coordinate_driver_switches"]).double(),
            maximum(solved["maximum_coupled_coordinate_depth"]).double(),
            maximum(solved[
                "maximum_coupled_coordinate_normalized_residual"]),
            maximum(solved[
                "maximum_rejected_coupled_coordinate_residual"]),
            range_exceeded.double(),torch.full_like(dt,self.tmax),
            maximum(final_temperature),
            torch.where(range_exceeded,offending_index,-1).double(),
            torch.where(range_exceeded,offending_roundoff,0.),
        ),-1)
        return after,lane_valid,diagnostic

    def chemistry(self,U,T,dt,available):
        """Legacy clipped chemistry RHS; local chemistry bypasses this path."""
        if self.s.chemistry_integration_mode != "legacy_cap":
            raise PropagationConfigurationError(
                "Tensor RHS chemistry is available only for legacy_cap"
            )
        rho=U[...,0]
        alpha0=U[...,4:6]/rho[...,None]
        rates0=self.rates(U,T)
        cap0=(rates0>self.maxrate).any(-1).double().mean(SPATIAL)
        resources=U if available is None else available
        d=dt[:,None,None,None]
        delta=torch.minimum(
            torch.clamp(1-alpha0,min=0),
            d*torch.clamp(rates0,max=self.maxrate),
        )
        capacity=torch.clamp(resources[...,0,None]-resources[...,4:6],min=0)/rho[...,None]
        delta=torch.minimum(delta,capacity)
        proposed_xi=rho*self.xi*(delta@self.weights)
        salt=.5*(resources[...,6]+resources[...,7])
        maximum_xi=torch.minimum(
            torch.clamp(salt-self.floor,min=0)/1.45,
            torch.clamp(resources[...,9],min=0),
        )
        scale=torch.where(
            proposed_xi>0,
            torch.minimum(torch.ones_like(rho),maximum_xi/torch.clamp(proposed_xi,min=1e-300)),
            torch.ones_like(rho),
        )
        delta=delta*scale[...,None]
        d_rho_alpha=rho[...,None]*delta
        d_xi=self.xi*(d_rho_alpha@self.weights)
        safe_dt=torch.where(dt>0,dt,torch.ones_like(dt))
        q=safe_dt[:,None,None]
        S=torch.zeros_like(U)
        S[...,4:6]=d_rho_alpha/q[...,None]
        S[...,3]=(d_rho_alpha@self.Q)/q
        S[...,6]=S[...,7]=-1.45*d_xi/q
        S[...,9]=-d_xi/q
        S[...,10]=2*d_xi/q
        S=torch.where((dt>0)[:,None,None,None],S,torch.zeros_like(S))
        limiter=(scale<1-1e-14).double().mean(SPATIAL)
        return S,cap0,limiter,torch.ones_like(cap0,dtype=torch.bool)

    def reservoir(self,U):
        return self.Q[0]*(U[...,0]-U[...,4])+self.Q[1]*(U[...,0]-U[...,5])

    def thermal(self,U):
        P=self.primitive(U);T=P[...,3];cp=interp(T,self.heat_grid,self.heat_values)
        if self.kmode=='constant': k=torch.full_like(T,self.kvalue)
        elif self.kmode=='table': k=interp(T,self.kgrid,self.kvalues)
        else: k=self.kvalue*torch.exp(-self.kea/self.R*(1/torch.clamp(T,min=1.)-1/self.ktref))
        conduction,diagonal=conduction_and_diagonal(T,k,self.dx)
        s=self.s
        loss=s.h/self.depth*(T-s.ambient)+s.emissivity*s.sigma_sb/self.depth*(T**4-s.ambient**4)
        return P,cp,k,conduction,loss,diagonal

    def nonchemical_step_limit(self,U):
        P,cp,k,q,loss,diag=self.thermal(U);T=P[...,3];s=self.s
        loss_jac=s.h/self.depth+4*s.emissivity*s.sigma_sb/self.depth*torch.maximum(T,torch.full_like(T,s.ambient))**3
        inv=((diag+loss_jac)/(P[...,0]*cp)).amax(SPATIAL)
        thermal=s.thermal_cfl/torch.clamp(inv,min=1e-300)
        if self.stationary_all:
            acoustic=torch.full_like(thermal,math.inf)
        else:
            speed=(abs(P[...,1])+abs(P[...,2])+2*self.sound(P[...,0]))/self.dx
            acoustic=s.cfl/torch.clamp(speed.amax(SPATIAL),min=1e-300)
            if self.fast:
                stationary=((U[...,1:3]==0).all((1,2,3))&(U[...,0]==U[:,0:1,0:1,0]).all(SPATIAL))
                acoustic=torch.where(stationary,torch.full_like(acoustic,math.inf),acoustic)
        return torch.minimum(thermal,acoustic)

    def step_size(self,U,remaining):
        s=self.s
        nonchemical=self.nonchemical_step_limit(U)
        if s.chemistry_integration_mode == "legacy_cap":
            P=self.primitive(U);T=P[...,3]
            rates=self.rates(U,T)
            has_stock=(.5*(U[...,6]+U[...,7])>self.floor)&(U[...,9]>0)
            effective_rates=torch.clamp(rates,max=self.maxrate)
            active_rates=torch.where(has_stock[...,None],effective_rates,torch.zeros_like(rates))
            reaction=s.alpha_step/torch.clamp(active_rates.amax((1,2,3)),min=1e-300)
        else:
            reaction=torch.full_like(remaining,math.inf)
        return torch.minimum(torch.minimum(nonchemical,reaction),torch.clamp(remaining,max=s.dtmax))

    def _pad(self,P,axis):
        n=P.shape[axis]
        if self.boundary=='periodic':
            ids=torch.arange(-3,n+3,device=P.device)%n
            return P.index_select(axis,ids)
        if self.boundary=='transmissive':
            ids=torch.clamp(torch.arange(-3,n+3,device=P.device),min=0,max=n-1)
            return P.index_select(axis,ids)
        left=P.narrow(axis,0,3).flip((axis,)).clone()
        right=P.narrow(axis,n-3,3).flip((axis,)).clone()
        normal=1 if axis==2 else 2
        left[...,normal]=-left[...,normal];right[...,normal]=-right[...,normal]
        return torch.cat((left,P,right),axis)

    def _trace(self,a,b,c,d,e):
        center=(a+b+c+d+e)/5
        scale=torch.stack([abs(v-center) for v in (a,b,c,d,e)]).amax(0)
        scale=torch.where(scale>0,scale,torch.ones_like(scale))
        aa,bb,cc,dd,ee=[(v-center)/scale for v in (a,b,c,d,e)]
        betas=[13/12*(aa-2*bb+cc)**2+.25*(aa-4*bb+3*cc)**2,
               13/12*(bb-2*cc+dd)**2+.25*(bb-dd)**2,
               13/12*(cc-2*dd+ee)**2+.25*(3*cc-4*dd+ee)**2]
        betas=[torch.cat((b[...,:4],b[...,4:].amax(-1,keepdim=True).expand_as(b[...,4:])), -1) for b in betas]
        den=torch.stack(betas)+self.epsilon
        scale=den.amin(0)
        weights=torch.stack((.1*(scale/den[0])**2,.6*(scale/den[1])**2,.3*(scale/den[2])**2))
        weights=weights/weights.sum(0)
        return weights[0]*(2*a-7*b+11*c)/6+weights[1]*(-b+5*c+2*d)/6+weights[2]*(2*c+5*d-e)/6

    def reconstruct(self,P,axis,first_order):
        pad=self._pad(P,axis);n=P.shape[axis]
        vals=[pad.narrow(axis,i,n+1) for i in range(6)]
        left=self._trace(*vals[:5]);right=self._trace(*list(reversed(vals[1:6])))
        invalid=~(self.valid(left)&self.valid(right))
        fallback=invalid|first_order[:,None,None]
        left=torch.where(fallback[...,None],vals[2],left)
        right=torch.where(fallback[...,None],vals[3],right)
        count=(invalid&~first_order[:,None,None]).sum(SPATIAL)
        return left,right,count

    def physical_flux(self,U,P,p,normal):
        F=U*P[...,normal,None]
        F[...,normal]=F[...,normal]+p
        F[...,3]=F[...,3]+p*P[...,normal]
        return F

    def riemann(self,L,R,normal):
        UL,UR=self.conservative(L),self.conservative(R)
        rl,rr=L[...,0],R[...,0];ul,ur=L[...,normal],R[...,normal]
        pl,pr=self.pressure(rl),self.pressure(rr)
        cl,cr=self.sound(rl),self.sound(rr)
        sl=torch.minimum(ul-cl,ur-cr);sr=torch.maximum(ul+cl,ur+cr)
        FL,FR=self.physical_flux(UL,L,pl,normal),self.physical_flux(UR,R,pr,normal)
        H=(sr[...,None]*FL-sl[...,None]*FR+(sl*sr)[...,None]*(UR-UL))/(sr-sl)[...,None]
        H=torch.where((sl>=0)[...,None],FL,torch.where((sr<=0)[...,None],FR,H))
        if self.kind=='hll': return H,torch.zeros(self.B,dtype=torch.int64,device=UL.device)
        dl,dr=rl*(sl-ul),rr*(sr-ur)
        sm=(pr-pl+dl*ul-dr*ur)/(dl-dr);ps=pl+dl*(sm-ul)
        def star(U,P,speed,p):
            rho=P[...,0];vel=P[...,normal];den=speed-sm
            safe=torch.where(abs(den)>1e-100,den,torch.full_like(den,1e-100))
            factor=(speed-vel)/safe;V=U*factor[...,None]
            V[...,normal]=rho*factor*sm
            spden=rho*(speed-vel)
            spden=torch.where(abs(spden)>1e-100,spden,torch.full_like(spden,1e-100))
            energy=U[...,3]/rho+(sm-vel)*(sm+p/spden)
            V[...,3]=rho*factor*energy
            return V
        USL,USR=star(UL,L,sl,pl),star(UR,R,sr,pr)
        bad=~torch.isfinite(sm)|~torch.isfinite(ps)|(ps<=0)|~(self.valid(self.primitive(USL))&self.valid(self.primitive(USR)))
        fsl=FL+sl[...,None]*(USL-UL);fsr=FR+sr[...,None]*(USR-UR)
        result=torch.where((sl>=0)[...,None],FL,torch.where((sr<=0)[...,None],FR,torch.where((sm>=0)[...,None],fsl,fsr)))
        result=torch.where(bad[...,None],H,result)
        contact=(ul==0)&(ur==0)&(pl==pr)
        exact=torch.zeros_like(result);exact[...,normal]=pl
        return torch.where(contact[...,None],exact,result),bad.sum(SPATIAL)

    def hydro(self,U,first_order):
        if self.stationary_all:
            z=torch.zeros(self.B,device=U.device,dtype=torch.float64)
            return torch.zeros_like(U),torch.zeros((self.B,NCONS),device=U.device,dtype=torch.float64),torch.stack((z,z,z,z+1),-1)
        P=self.primitive(U);fluxes=[]
        fallback=torch.zeros(self.B,device=U.device,dtype=torch.float64);star=fallback.clone();faces=0
        for axis,normal in ((2,1),(1,2)):
            l,r,fb=self.reconstruct(P,axis,first_order)
            F,sb=self.riemann(l,r,normal)
            if self.boundary=='periodic':
                if axis==2:F[:,:,-1,:]=F[:,:,0,:]
                else:F[:,-1,:,:]=F[:,0,:,:]
            elif self.boundary=='reflective':
                for end in (0,-1):
                    f=F[:,:,end,:] if axis==2 else F[:,end,:,:]
                    keep=f[...,normal].clone();f.zero_();f[...,normal]=keep
            fluxes.append(F);fallback+=fb;star+=sb;faces+=l.shape[1]*l.shape[2]
        fx,fy=fluxes
        rhs=-(torch.diff(fx,dim=2)+torch.diff(fy,dim=1))/self.dx
        bound=-((fx[:,:,-1,:]-fx[:,:,0,:]).sum(1)+(fy[:,-1,:,:]-fy[:,0,:,:]).sum(1))*self.dx*self.depth
        skip=torch.zeros_like(fallback,dtype=torch.bool)
        if self.fast:
            skip=(U[...,1:3]==0).all((1,2,3))&(U[...,0]==U[:,0:1,0:1,0]).all(SPATIAL)
            rhs=torch.where(skip[:,None,None,None],torch.zeros_like(rhs),rhs)
            bound=torch.where(skip[:,None],torch.zeros_like(bound),bound)
        return rhs,bound,torch.stack((torch.where(skip,0.,fallback),torch.where(skip,0.,fallback*0+faces),
                                      torch.where(skip,0.,star),skip.double()),-1)

    def rhs(self,U,dt,first_order):
        hydro,bound,fd=self.hydro(U,first_order)
        P,cp,k,qcond,loss,diagonal=self.thermal(U);T=P[...,3]
        if self.electrical is None:
            electric=torch.zeros_like(U);qj=torch.zeros_like(T);qe=qj
            electrical_ok=torch.ones(self.B,device=U.device,dtype=torch.bool)
        else:
            electric,qj,qe,electrical_ok=self.electrical.evaluate(U,T,dt)
        available=U+dt[:,None,None,None]*(hydro+electric)
        availability_ok=(available[...,6:10]>=-1e-9).all((1,2,3))
        chem,cap,limiter,chemistry_ok=self.chemistry(U,T,dt,available)
        derivative=hydro+electric+chem
        derivative[...,3]=derivative[...,3]+qcond-loss
        # Validate actual row sum at EACH RK stage, not only the start of a step.
        jac=self.s.h/self.depth+4*self.s.emissivity*self.s.sigma_sb/self.depth*torch.maximum(T,torch.full_like(T,self.s.ambient))**3
        cfl=((diagonal+jac)/(P[...,0]*cp)*dt[:,None,None]).amax(SPATIAL)
        thermal_ok=torch.isfinite(k).all(SPATIAL)&(k>=0).all(SPATIAL)&(cfl<=self.s.thermal_cfl*(1+1e-12))
        fields=torch.stack((qj,qe,chem[...,3],loss,qcond),-1)
        ledger=torch.cat((bound,fields.sum(SPATIAL)*self.vol),-1)
        ok=availability_ok&electrical_ok&thermal_ok&chemistry_ok&torch.isfinite(derivative).all((1,2,3))
        if self.s.chemistry_integration_mode == "legacy_cap":
            cells=U.shape[1]*U.shape[2]
            allowed_cells=math.floor(self.allowed_cap*cells+0.5)
            allowed_discrete=allowed_cells/cells
            ok=ok&(cap<=allowed_discrete+1e-15)
        return derivative,ledger,torch.cat((fd,cap[:,None],limiter[:,None]),-1),fields,ok

    def nonchemical_rhs(self,U,dt,first_order):
        """Explicit Euler/Fourier/electrical RHS with chemistry omitted."""
        hydro,bound,fd=self.hydro(U,first_order)
        P,cp,k,qcond,loss,diagonal=self.thermal(U);T=P[...,3]
        if self.electrical is None:
            electric=torch.zeros_like(U);qj=torch.zeros_like(T);qe=qj
            electrical_ok=torch.ones(self.B,device=U.device,dtype=torch.bool)
        else:
            electric,qj,qe,electrical_ok=self.electrical.evaluate(U,T,dt)
        available=U+dt[:,None,None,None]*(hydro+electric)
        availability_ok=(available[...,6:10]>=-1e-9).all((1,2,3))
        derivative=hydro+electric
        derivative[...,3]=derivative[...,3]+qcond-loss
        jac=self.s.h/self.depth+4*self.s.emissivity*self.s.sigma_sb/self.depth*torch.maximum(
            T,torch.full_like(T,self.s.ambient))**3
        cfl=((diagonal+jac)/(P[...,0]*cp)*dt[:,None,None]).amax(SPATIAL)
        thermal_ok=(torch.isfinite(k).all(SPATIAL)&(k>=0).all(SPATIAL)
                    &(cfl<=self.s.thermal_cfl*(1+1e-12)))
        zero=torch.zeros_like(qj)
        fields=torch.stack((qj,qe,zero,loss,qcond),-1)
        ledger=torch.cat((bound,fields.sum(SPATIAL)*self.vol),-1)
        ok=(availability_ok&electrical_ok&thermal_ok
            &torch.isfinite(derivative).all((1,2,3)))
        # Keep the six-column transport diagnostic shape used by legacy packs.
        diagnostic=torch.cat((fd,torch.zeros((self.B,2),dtype=torch.float64,
                                              device=U.device)),-1)
        return derivative,ledger,diagnostic,fields,ok

    def _nonchemical_trial(self,U,dt,first_order):
        d=dt[:,None,None,None]
        k0,l0,g0,f0,v0=self.nonchemical_rhs(U,dt,first_order)
        u1=U+d*k0;p1=self.primitive(u1);ok1=self.valid(p1).all(SPATIAL)
        u1safe=torch.where(ok1[:,None,None,None],u1,U)
        k1,l1,g1,f1,v1=self.nonchemical_rhs(u1safe,dt,first_order)
        u2=.75*U+.25*(u1safe+d*k1);ok2=self.valid(self.primitive(u2)).all(SPATIAL)
        u2safe=torch.where(ok2[:,None,None,None],u2,U)
        k2,l2,g2,f2,v2=self.nonchemical_rhs(u2safe,dt,first_order)
        result=U/3+(2/3)*(u2safe+d*k2)
        skip=(g0[:,3]>0)&(g1[:,3]>0)&(g2[:,3]>0)
        result[...,:3]=torch.where(skip[:,None,None,None],U[...,:3],result[...,:3])
        valid=v0&v1&v2&ok1&ok2&self.valid(self.primitive(result)).all(SPATIAL)
        integrated=dt[:,None]*(l0/6+l1/6+2*l2/3)
        diagnostics=torch.stack((g0,g1,g2),1)
        fields=f0/6+f1/6+2*f2/3
        return result,integrated,diagnostics,fields,valid

    def _local_split_trial(self,U,dt,first_order):
        """C(dt/2) -> nonchemical SSPRK33(dt) -> C(dt/2), transactionally."""
        half=.5*dt
        cuda_timing=U.device.type=='cuda'
        if cuda_timing:
            phase_events=self._local_phase_events
            phase_stream=torch.cuda.current_stream(U.device)
            phase_events[0].record(phase_stream)
        else:
            chemistry0_start=time.perf_counter()
        chemistry0,chemistry0_ok,diag0=self.advance_local(U,half)
        if cuda_timing:
            phase_events[1].record(phase_stream)
        else:
            chemistry0_elapsed=time.perf_counter()-chemistry0_start
        stability_limit=self.nonchemical_step_limit(chemistry0)
        rechecked=(diag0[:,0]>0)&chemistry0_ok
        stability_ok=dt<=stability_limit*(1+1e-12)
        transport_eligible=chemistry0_ok&stability_ok
        chemistry0_safe=torch.where(transport_eligible[:,None,None,None],chemistry0,U)
        transport_dt=torch.where(transport_eligible,dt,torch.zeros_like(dt))
        if cuda_timing:
            phase_events[2].record(phase_stream)
        else:
            nonchemical_start=time.perf_counter()
        nonchemical,integrated,transport_diagnostics,fields,transport_ok=(
            self._nonchemical_trial(chemistry0_safe,transport_dt,first_order)
        )
        if cuda_timing:
            phase_events[3].record(phase_stream)
        else:
            nonchemical_elapsed=time.perf_counter()-nonchemical_start
        second_eligible=transport_eligible&transport_ok
        nonchemical_safe=torch.where(second_eligible[:,None,None,None],nonchemical,U)
        if cuda_timing:
            phase_events[4].record(phase_stream)
        else:
            chemistry1_start=time.perf_counter()
        chemistry1,chemistry1_ok,diag1=self.advance_local(
            nonchemical_safe,torch.where(second_eligible,half,torch.zeros_like(half))
        )
        if cuda_timing:
            phase_events[5].record(phase_stream)
            self.trial_cuda_phase_events=phase_events
        else:
            chemistry1_elapsed=time.perf_counter()-chemistry1_start
            self.trial_cuda_phase_events=None
            self.trial_chemistry_wall_clock_time_s=(
                chemistry0_elapsed+chemistry1_elapsed
            )
            self.trial_nonchemical_wall_clock_time_s=nonchemical_elapsed
        valid=((dt>0)&chemistry0_ok&stability_ok&transport_ok&chemistry1_ok
               &self.valid(self.primitive(chemistry1)).all(SPATIAL))
        result=torch.where(valid[:,None,None,None],chemistry1,U)
        heat0=chemistry0[...,ENERGY]-U[...,ENERGY]
        heat1=chemistry1[...,ENERGY]-nonchemical_safe[...,ENERGY]
        total_heat=torch.where(valid[:,None,None],heat0+heat1,torch.zeros_like(heat0))
        integrated=integrated.clone()
        integrated[:,NCONS+2]=integrated[:,NCONS+2]+total_heat.sum(SPATIAL)*self.vol
        safe_dt=torch.where(dt>0,dt,torch.ones_like(dt))
        fields=fields.clone();fields[...,2]=total_heat/safe_dt[:,None,None]

        failure_reason=torch.zeros(self.B,dtype=torch.int64,device=U.device)
        failure_reason=torch.where(~chemistry0_ok,torch.ones_like(failure_reason),failure_reason)
        failure_reason=torch.where(chemistry0_ok&~stability_ok,
                                   torch.full_like(failure_reason,2),failure_reason)
        failure_reason=torch.where(transport_eligible&~transport_ok,
                                   torch.full_like(failure_reason,3),failure_reason)
        failure_reason=torch.where(second_eligible&~chemistry1_ok,
                                   torch.ones_like(failure_reason),failure_reason)
        range_field=LOCAL_CHEMISTRY_DIAGNOSTIC_NAMES.index(
            "configured_model_temperature_range_exceeded")
        range_failure=((~chemistry0_ok&(diag0[:,range_field]>0))
                       |(second_eligible&~chemistry1_ok&(diag1[:,range_field]>0)))
        failure_reason=torch.where(range_failure,
                                   torch.full_like(failure_reason,4),failure_reason)
        self.trial_local_diagnostics=torch.stack((diag0,diag1),1)
        self.trial_post_stability_limit=stability_limit
        self.trial_post_rechecked=rechecked
        self.trial_post_rejected=rechecked&~stability_ok
        self.trial_failure_reason=failure_reason
        return result,integrated,transport_diagnostics,fields,valid

    def local_trial_phase_wall_times(self):
        """Return completed local-trial phase times without adding a sync.

        The tensor runner calls this only after its packed ``cpu()`` transfer,
        which already waits for all six recorded CUDA events.  Returned times
        are whole-batch latency and are attributed to each lane that was active
        for the attempted trial, including rejected attempts.
        """
        events=self.trial_cuda_phase_events
        if events is not None:
            chemistry=(events[0].elapsed_time(events[1])
                       +events[4].elapsed_time(events[5]))/1000.0
            nonchemical=events[2].elapsed_time(events[3])/1000.0
            return float(chemistry),float(nonchemical)
        return (float(self.trial_chemistry_wall_clock_time_s),
                float(self.trial_nonchemical_wall_clock_time_s))

    def trial(self,U,dt,first_order):
        if self.s.chemistry_integration_mode == "local_adaptive_thermochemical":
            return self._local_split_trial(U,dt,first_order)
        d=dt[:,None,None,None]
        k0,l0,g0,f0,v0=self.rhs(U,dt,first_order)
        u1=U+d*k0;p1=self.primitive(u1);ok1=self.valid(p1).all(SPATIAL)
        # Invalid attempts must not feed NaN/negative values to later table/EOS
        # stages. The whole attempt remains invalid and will be retried.
        u1safe=torch.where(ok1[:,None,None,None],u1,U)
        k1,l1,g1,f1,v1=self.rhs(u1safe,dt,first_order)
        u2=.75*U+.25*(u1safe+d*k1);ok2=self.valid(self.primitive(u2)).all(SPATIAL)
        u2safe=torch.where(ok2[:,None,None,None],u2,U)
        k2,l2,g2,f2,v2=self.rhs(u2safe,dt,first_order)
        result=U/3+(2/3)*(u2safe+d*k2)
        skip=(g0[:,3]>0)&(g1[:,3]>0)&(g2[:,3]>0)
        result[...,:3]=torch.where(skip[:,None,None,None],U[...,:3],result[...,:3])
        valid=v0&v1&v2&ok1&ok2&self.valid(self.primitive(result)).all(SPATIAL)
        return result,dt[:,None]*(l0/6+l1/6+2*l2/3),torch.stack((g0,g1,g2),1),f0/6+f1/6+2*f2/3,valid

    def record(self,t,U,budget):
        P=self.primitive(U);X=self.progress(U)
        sums=U.sum(SPATIAL)*self.vol;b=budget[:,:NCONS]
        qj,qe,qc,loss,cond=budget[:,NCONS:].unbind(-1)
        mass=sums[:,0]-self.initial_sums[:,0]-b[:,0]
        energy=sums[:,3]-self.initial_sums[:,3]-b[:,3]-qj-qe-qc+loss-cond
        scale=torch.stack((abs(self.initial_sums[:,3]),abs(sums[:,3]),abs(qj)+abs(qe)+abs(qc)+abs(loss),torch.ones_like(qj))).amax(0)
        massrel=abs(mass)/torch.clamp(abs(self.initial_sums[:,0]),min=1e-300);energyrel=abs(energy)/scale
        reservoir=self.reservoir(U).sum(SPATIAL)*self.vol
        rb=self.Q.sum()*b[:,0]-self.Q[0]*b[:,4]-self.Q[1]*b[:,5]
        total=(sums[:,3]+reservoir)-(self.initial_sums[:,3]+self.reservoir0)-b[:,3]-rb-qj-qe+loss-cond
        extent=self.xi*(self.weights[0]*U[...,4]+self.weights[1]*U[...,5])
        rp=U[...,9]+extent-U[...,0]*self.pva0
        rw=U[...,10]-2*extent
        rs=.5*(U[...,6]+U[...,7])+U[...,11]+1.45*extent-U[...,0]*self.lp0
        local=torch.maximum(abs(rp).amax(SPATIAL),abs(rw).amax(SPATIAL))
        sb=(.5*(b[:,6]+b[:,7])+b[:,11]+1.45*self.xi[:,0,0]*(self.weights[0]*b[:,4]+self.weights[1]*b[:,5])-self.lp0[:,0,0]*b[:,0])
        salt=rs.sum(SPATIAL)*self.vol-sb
        stock=torch.clamp(U[...,0].amax(SPATIAL)*self.lp0[:,0,0],min=1.)
        pressure=self.pressure(P[...,0])
        record=torch.stack((t,(X<self.s.front_threshold).double().mean(SPATIAL),X.mean(SPATIAL),P[...,3].amax(SPATIAL),
                 torch.hypot(P[...,1],P[...,2]).amax(SPATIAL),pressure.amin(SPATIAL),pressure.amax(SPATIAL),sums[:,0],sums[:,3],
                 reservoir,mass,massrel,energy,energyrel,total,qj,qe,qc,loss,cond,b[:,0],b[:,3],local,salt),-1)
        valid=(massrel<=self.s.mass_tolerance)&(energyrel<=self.s.energy_tolerance)&(local<=1e-8*stock)&(abs(salt)<=1e-8*stock*self.ny*self.nx*self.vol)&torch.isfinite(record).all(-1)
        return record,valid

    def front_edges(self,X):
        m=X<self.s.front_threshold
        return (m[:,:,1:]!=m[:,:,:-1]).sum(SPATIAL)+(m[:,1:,:]!=m[:,:-1,:]).sum(SPATIAL)
