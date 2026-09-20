from pathlib import Path
import re, shutil, datetime

ROOT = Path('.')
STAMP = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
FILES = [
    Path('python/ecsp_reactive/condensed/chemistry.py'),
    Path('python/ecsp_reactive/condensed/solver.py'),
    Path('python/ecsp_reactive/condensed/tensor_math.py'),
    Path('python/ecsp_reactive/condensed/tensor_solver.py'),
]
for p in FILES:
    if not p.is_file():
        raise SystemExit(f'missing required file: {p}')
    shutil.copy2(p, p.with_name(p.name + f'.bak_subcycling_{STAMP}'))


def replace_method(text: str, name: str, new_method: str) -> str:
    m = re.search(rf'(?m)^    def {re.escape(name)}\(', text)
    if not m:
        raise SystemExit(f'method not found: {name}')
    tail = text[m.end():]
    n = re.search(r'(?m)^    def [A-Za-z_][A-Za-z0-9_]*\(', tail)
    end = m.end() + n.start() if n else len(text)
    return text[:m.start()] + new_method.rstrip() + '\n\n' + text[end:]

# ---------------------------------------------------------------------------
# chemistry.py: legacy behaviour remains the default; subcycle_raw is opt-in.
# ---------------------------------------------------------------------------
p = FILES[0]
s = p.read_text()
new_source = r'''    def source(self, U, temperature, dt, *, available=None,
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
        }'''
s = replace_method(s, 'source', new_source)
p.write_text(s)

# ---------------------------------------------------------------------------
# solver.py: post-onset solver opts into chemistry subcycling via config.
# ---------------------------------------------------------------------------
p = FILES[1]
s = p.read_text()
marker = '# ECSP_POST_ONSET_SUBCYCLING_V1'
if marker not in s:
    pat = r'(?m)^(\s*self\.alpha_step=float\([^\n]+\)\s*)$'
    m = re.search(pat, s)
    if not m:
        raise SystemExit('solver.py alpha_step assignment not found')
    insert = m.group(1) + '''\n        # ECSP_POST_ONSET_SUBCYCLING_V1\n        self.chemistry_integration_mode=str(\n            self.cfg.get("chemistry_integration_mode", "legacy_cap")\n        ).lower()\n        if self.chemistry_integration_mode not in {"legacy_cap", "subcycle_raw"}:\n            raise PropagationConfigurationError(\n                "chemistry_integration_mode must be legacy_cap or subcycle_raw"\n            )\n        self.max_chemistry_subcycles=_configuration_integer(\n            self.cfg.get("maximum_chemistry_subcycles_per_pde_step", 4096),\n            name="maximum_chemistry_subcycles_per_pde_step",\n            minimum=1, maximum=1000000,\n        )\n        self.max_chemistry_subcycles_used=0'''
    s = s[:m.start()] + insert + s[m.end():]

new_time_step = r'''    def time_step(self,U,remaining):
        P,cp,k,_,_=self._thermal_terms(U)
        rho=P[...,RHO]; T=P[...,3]
        loss_jac=self.h/self.a.thickness+4*self.emissivity*self.sigma_sb/self.a.thickness*np.maximum(T,self.ambient)**3
        inverse_thermal=np.max(_thermal_diffusive_cfl(k,rho*cp,self.a.propellant_mask,self.a.dx,1.)+loss_jac/(rho*cp))
        thermal_dt=self.thermal_cfl/inverse_thermal if inverse_thermal>0 else math.inf
        stationary=self.fast and self.riemann=="hllc" and mechanically_stationary(U,self.boundary)
        if stationary:
            acoustic_dt=math.inf
        else:
            sound=self.thermo.sound_speed(rho)
            acoustic_dt=self.cfl/np.max((np.abs(P[...,MX])+np.abs(P[...,MY])+2*sound)/self.a.dx)
        if self.chemistry_integration_mode == "legacy_cap":
            rates=self.chem.raw_rates(U,T)
            has_stock=(.5*(U[...,CATION]+U[...,ANION])>self.concentration_floor)&(U[...,PVA]>0)
            effective_rates=np.minimum(rates,self.chem.maximum_rate)
            maxrate=float(np.max(np.where(has_stock[...,None],effective_rates,0.)))
            reaction_dt=self.alpha_step/maxrate if maxrate>0 else math.inf
        else:
            # Stiff chemistry is integrated locally inside source(); it must
            # not force the entire Euler/Fourier PDE onto the chemistry scale.
            reaction_dt=math.inf
        dt=min(self.dtmax,remaining,thermal_dt,acoustic_dt,reaction_dt)
        if not math.isfinite(dt) or dt<=0 or dt<self.min_dt:
            raise PropagationCandidateNumericalError("Reactive timestep fell below minimum_time_step_s; no artificial sound-speed reduction used")
        return dt'''
s = replace_method(s, 'time_step', new_time_step)

# Add the new source options without depending on line wrapping.
pat = re.compile(
    r'self\.chem\.source\(U,T,dt,available=available,concentration_floor=self\.concentration_floor,\s*'
    r'allowed_rate_cap_fraction=self\.allowed_cap\)'
)
repl = '''self.chem.source(\n            U,T,dt,available=available,\n            concentration_floor=self.concentration_floor,\n            allowed_rate_cap_fraction=self.allowed_cap,\n            integration_mode=self.chemistry_integration_mode,\n            maximum_channel_increment=self.alpha_step,\n            maximum_subcycles=self.max_chemistry_subcycles,\n        )'''
s, n = pat.subn(repl, s, count=1)
if n != 1:
    raise SystemExit('solver.py chemistry source call not found')

# Accumulate subcycle telemetry on the scalar backend.
old = 'self.max_inventory_limiter=max(self.max_inventory_limiter,d["inventory_limiter_fraction"])'
if old in s and 'max_chemistry_subcycles_used=max' not in s[s.find(old):s.find(old)+300]:
    s = s.replace(
        old,
        old + '\n            self.max_chemistry_subcycles_used=max(self.max_chemistry_subcycles_used,int(d.get("chemistry_subcycles",0)))',
        1,
    )

# Make the output semantics explicit.
old = '"maximumChemicalRateCapFraction":self.max_rate_cap,"maximumChemicalInventoryLimiterFraction":self.max_inventory_limiter,'
if old in s:
    new = ('"maximumChemicalRateCapFraction":self.max_rate_cap,'
           '"maximumChemicalInventoryLimiterFraction":self.max_inventory_limiter,'
           '"maximumChemistrySubcyclesPerPDEStage":self.max_chemistry_subcycles_used,'
           '"chemicalRateThresholdDiagnosticOnly":self.chemistry_integration_mode=="subcycle_raw",')
    s = s.replace(old, new, 1)
p.write_text(s)

# ---------------------------------------------------------------------------
# tensor_math.py: same numerical contract on torch CPU/CUDA.
# ---------------------------------------------------------------------------
p = FILES[2]
s = p.read_text()
new_step = r'''    def step_size(self,U,remaining):
        P,cp,k,q,loss,diag=self.thermal(U);T=P[...,3];s=self.s
        loss_jac=s.h/self.depth+4*s.emissivity*s.sigma_sb/self.depth*torch.maximum(T,torch.full_like(T,s.ambient))**3
        inv=((diag+loss_jac)/(P[...,0]*cp)).amax(SPATIAL)
        thermal=s.thermal_cfl/torch.clamp(inv,min=1e-300)
        if self.stationary_all:
            acoustic=torch.full_like(remaining,math.inf)
        else:
            speed=(abs(P[...,1])+abs(P[...,2])+2*self.sound(P[...,0]))/self.dx
            acoustic=s.cfl/torch.clamp(speed.amax(SPATIAL),min=1e-300)
            if self.fast:
                stationary=((U[...,1:3]==0).all((1,2,3))&(U[...,0]==U[:,0:1,0:1,0]).all(SPATIAL))
                acoustic=torch.where(stationary,torch.full_like(acoustic,math.inf),acoustic)
        if s.chemistry_integration_mode == "legacy_cap":
            rates=self.rates(U,T)
            has_stock=(.5*(U[...,6]+U[...,7])>self.floor)&(U[...,9]>0)
            effective_rates=torch.clamp(rates,max=self.maxrate)
            active_rates=torch.where(has_stock[...,None],effective_rates,torch.zeros_like(rates))
            reaction=s.alpha_step/torch.clamp(active_rates.amax((1,2,3)),min=1e-300)
        else:
            reaction=torch.full_like(remaining,math.inf)
        return torch.minimum(torch.minimum(torch.minimum(thermal,acoustic),reaction),torch.clamp(remaining,max=s.dtmax))'''
s = replace_method(s, 'step_size', new_step)

new_chem = r'''    def chemistry(self,U,T,dt,available):
        rho=U[...,0]
        alpha0=U[...,4:6]/rho[...,None]
        rates0=self.rates(U,T)
        cap0=(rates0>self.maxrate).any(-1).double().mean(SPATIAL)
        mode=self.s.chemistry_integration_mode
        if mode == "legacy_cap":
            d=dt[:,None,None,None]
            delta=torch.minimum(
                torch.clamp(1-alpha0,min=0),
                d*torch.clamp(rates0,max=self.maxrate),
            )
            resources=U if available is None else available
            capacity=torch.clamp(resources[...,0,None]-resources[...,4:6],min=0)/rho[...,None]
            delta=torch.minimum(delta,capacity)
            proposed_xi=rho*self.xi*(delta@self.weights)
            salt=.5*(resources[...,6]+resources[...,7])
            maximum_xi=torch.minimum(
                torch.clamp(salt-self.floor,min=0)/1.45,
                torch.clamp(resources[...,9],min=0),
            )
            scale=torch.ones_like(rho)
            positive=proposed_xi>0
            scale=torch.where(
                positive,
                torch.minimum(torch.ones_like(scale),maximum_xi/torch.clamp(proposed_xi,min=1e-300)),
                scale,
            )
            delta=delta*scale[...,None]
            d_rho_alpha=rho[...,None]*delta
            d_xi=self.xi*(d_rho_alpha@self.weights)
            S=torch.zeros_like(U)
            safe_dt=torch.where(dt>0,dt,torch.ones_like(dt))
            q=safe_dt[:,None,None]
            S[...,4:6]=d_rho_alpha/q[...,None]
            S[...,3]=(d_rho_alpha@self.Q)/q
            S[...,6]=S[...,7]=-1.45*d_xi/q
            S[...,9]=-d_xi/q
            S[...,10]=2*d_xi/q
            S=torch.where((dt>0)[:,None,None,None],S,torch.zeros_like(S))
            limiter=(scale<1-1e-14).double().mean(SPATIAL)
            return S,cap0,limiter,torch.ones_like(cap0,dtype=torch.bool),torch.ones_like(cap0,dtype=torch.int64)

        resources=U if available is None else available
        alpha=alpha0.clone()
        base_cation=resources[...,6].clone(); cation=base_cation.clone()
        base_anion=resources[...,7].clone(); anion=base_anion.clone()
        base_pva=resources[...,9].clone(); pva=base_pva.clone()
        base_product=resources[...,10].clone(); product=base_product.clone()
        energy_increment=torch.zeros_like(rho)
        remaining=torch.clamp(dt,min=0).clone()
        counts=torch.zeros_like(dt,dtype=torch.int64)
        capmax=cap0.clone()
        limitermax=torch.zeros_like(dt)
        rate_state=U.clone()
        eps=64*torch.finfo(torch.float64).eps*torch.maximum(torch.ones_like(dt),torch.abs(dt))

        for _ in range(self.s.max_chemistry_subcycles):
            active_time=remaining>eps
            if not bool(active_time.any().item()):
                break
            rate_state[...,4:6]=rho[...,None]*alpha
            rates=self.rates(rate_state,T)
            cap=(rates>self.maxrate).any(-1).double().mean(SPATIAL)
            capmax=torch.maximum(capmax,cap)
            has_stock=(.5*(cation+anion)>self.floor)&(pva>0)&(alpha<1-1e-15).any(-1)
            active_rates=torch.where(has_stock[...,None],rates,torch.zeros_like(rates))
            maxrate=active_rates.amax((1,2,3))
            reacting=active_time&(maxrate>0)
            # Cases with no remaining reactive stock are complete for this stage.
            remaining=torch.where(active_time&~reacting,torch.zeros_like(remaining),remaining)
            if not bool(reacting.any().item()):
                break
            h=torch.minimum(
                remaining,
                self.s.alpha_step/torch.clamp(maxrate,min=1e-300),
            )
            h=torch.where(reacting,h,torch.zeros_like(h))
            delta=torch.minimum(
                torch.clamp(1-alpha,min=0),
                h[:,None,None,None]*rates,
            )
            proposed_xi=rho*self.xi*(delta@self.weights)
            salt=.5*(cation+anion)
            maximum_xi=torch.minimum(
                torch.clamp(salt-self.floor,min=0)/1.45,
                torch.clamp(pva,min=0),
            )
            scale=torch.where(
                proposed_xi>0,
                torch.minimum(torch.ones_like(rho),maximum_xi/torch.clamp(proposed_xi,min=1e-300)),
                torch.ones_like(rho),
            )
            limiter=(scale<1-1e-14).double().mean(SPATIAL)
            limitermax=torch.maximum(limitermax,limiter)
            delta=delta*scale[...,None]
            accepted_case=(delta>0).any((1,2,3))&reacting
            d_rho_alpha=rho[...,None]*delta
            d_xi=self.xi*(d_rho_alpha@self.weights)
            alpha=alpha+delta
            cation=cation-1.45*d_xi
            anion=anion-1.45*d_xi
            pva=pva-d_xi
            product=product+2*d_xi
            energy_increment=energy_increment+(d_rho_alpha@self.Q)
            counts=counts+accepted_case.to(torch.int64)
            remaining=torch.where(
                accepted_case,
                torch.clamp(remaining-h,min=0),
                torch.zeros_like(remaining),
            )

        chemistry_ok=remaining<=eps
        safe_dt=torch.where(dt>0,dt,torch.ones_like(dt))
        q=safe_dt[:,None,None]
        S=torch.zeros_like(U)
        S[...,4:6]=rho[...,None]*(alpha-alpha0)/q[...,None]
        S[...,3]=energy_increment/q
        S[...,6]=(cation-base_cation)/q
        S[...,7]=(anion-base_anion)/q
        S[...,9]=(pva-base_pva)/q
        S[...,10]=(product-base_product)/q
        S=torch.where((dt>0)[:,None,None,None],S,torch.zeros_like(S))
        if hasattr(self,"trial_max_subcycles"):
            self.trial_max_subcycles=torch.maximum(self.trial_max_subcycles,counts)
        return S,capmax,limitermax,chemistry_ok,counts'''
s = replace_method(s, 'chemistry', new_chem)

new_rhs = r'''    def rhs(self,U,dt,first_order):
        hydro,bound,fd=self.hydro(U,first_order)
        P,cp,k,qcond,loss,diagonal=self.thermal(U);T=P[...,3]
        if self.electrical is None:
            electric=torch.zeros_like(U);qj=torch.zeros_like(T);qe=qj
            electrical_ok=torch.ones(self.B,device=U.device,dtype=torch.bool)
        else:
            electric,qj,qe,electrical_ok=self.electrical.evaluate(U,T,dt)
        available=U+dt[:,None,None,None]*(hydro+electric)
        availability_ok=(available[...,6:10]>=-1e-9).all((1,2,3))
        chem,cap,limiter,chemistry_ok,subcycles=self.chemistry(U,T,dt,available)
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
        return derivative,ledger,torch.cat((fd,cap[:,None],limiter[:,None]),-1),fields,ok'''
s = replace_method(s, 'rhs', new_rhs)

# Reset per-trial telemetry before the three SSPRK RHS calls.
needle = '    def trial(self,U,dt,first_order):\n'
if needle not in s:
    raise SystemExit('tensor_math.py trial method not found')
if 'self.trial_max_subcycles=torch.zeros' not in s:
    s = s.replace(
        needle,
        needle + '        self.trial_max_subcycles=torch.zeros(self.B,device=U.device,dtype=torch.int64)\n',
        1,
    )
p.write_text(s)

# ---------------------------------------------------------------------------
# tensor_solver.py: preserve per-candidate retry, add honest telemetry wording.
# ---------------------------------------------------------------------------
p = FILES[3]
s = p.read_text()
trial_line = 'candidate,integrated,diag,fields,valid=(captured or k.trial)(U,dt,first)'
if trial_line in s and 'subcycles_cpu=' not in s:
    s = s.replace(
        trial_line,
        trial_line + '\n        subcycles_cpu=k.trial_max_subcycles.detach().cpu().numpy()',
        1,
    )

accepted_anchor = 'v.step_number+=1;v.first_order_steps+=int(retry[i]>=2)'
if accepted_anchor in s and 'v.max_chemistry_subcycles_used=max' not in s:
    s = s.replace(
        accepted_anchor,
        accepted_anchor + '\n            v.max_chemistry_subcycles_used=max(v.max_chemistry_subcycles_used,int(subcycles_cpu[i]))',
        1,
    )

# Existing diagnostic patch: relabel cap as diagnostic-only and report subcycles.
s = s.replace('chemical_cap_fraction={cap_max:.12g}',
              'chemical_rate_threshold_fraction_diagnostic={cap_max:.12g}')
s = s.replace('allowed_cap_fraction={s.allowed_cap:.12g}',
              'configured_threshold_fraction_reference={s.allowed_cap:.12g}')
if 'f"chemistry_subcycles={int(subcycles_cpu[i])} "' not in s:
    target = 'f"retry={retry[i]} "'
    if target in s:
        s = s.replace(target, target + '\n                    f"chemistry_subcycles={int(subcycles_cpu[i])} "', 1)

# Generic fallback message, if the custom diagnostic patch is absent.
s = s.replace(
    "PropagationCandidateNumericalError('Reactive tensor stage/budget/CFL failed after retry limit; no clipping or backend fallback')",
    "PropagationCandidateNumericalError(f'Reactive tensor stage/budget/CFL/positivity failed after retry limit; chemistry_subcycles={int(subcycles_cpu[i])}; chemical rate threshold is diagnostic-only in subcycle_raw mode')",
)
p.write_text(s)

# ---------------------------------------------------------------------------
# Activate only in the post-onset configs; pre-flame config/physics are intact.
# ---------------------------------------------------------------------------
configs = [
    Path('config/nsga2_bc_reactive_m2cpu_200x3.yaml'),
    Path('config/nsga2_bc_reactive_a100_gpuonly_200x3.yaml'),
]
for c in configs:
    if not c.is_file():
        continue
    shutil.copy2(c, c.with_name(c.name + f'.bak_subcycling_{STAMP}'))
    t = c.read_text()
    # Remove stale copies if this patch is re-run manually from a backed-up tree.
    t = re.sub(r'(?m)^    chemistry_integration_mode:.*\n', '', t)
    t = re.sub(r'(?m)^    maximum_chemistry_subcycles_per_pde_step:.*\n', '', t)
    m = re.search(r'(?m)^    maximum_channel_increment:\s*[^\n]+\n', t)
    if not m:
        raise SystemExit(f'{c}: maximum_channel_increment not found')
    addition = (
        m.group(0)
        + '    chemistry_integration_mode: subcycle_raw\n'
        + '    maximum_chemistry_subcycles_per_pde_step: 4096\n'
    )
    t = t[:m.start()] + addition + t[m.end():]
    c.write_text(t)

print('post-onset raw-kinetics chemistry subcycling patch installed')
print('pre-flame physics/config values were not changed')
