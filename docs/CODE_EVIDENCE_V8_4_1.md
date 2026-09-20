# v8.4.1 핵심 소스 근거

전체 파일은 ZIP 안의 같은 상대 경로입니다. 아래 줄 번호는 이 릴리스 기준입니다.

## C01 — python/ecsp_reactive/condensed/solver.py:115–134

```python
 115:     def time_step(self,U,remaining):
 116:         P,cp,k,_,_=self._thermal_terms(U)
 117:         rho=P[...,RHO]; T=P[...,3]
 118:         loss_jac=self.h/self.a.thickness+4*self.emissivity*self.sigma_sb/self.a.thickness*np.maximum(T,self.ambient)**3
 119:         inverse_thermal=np.max(_thermal_diffusive_cfl(k,rho*cp,self.a.propellant_mask,self.a.dx,1.)+loss_jac/(rho*cp))
 120:         thermal_dt=self.thermal_cfl/inverse_thermal if inverse_thermal>0 else math.inf
 121:         stationary=self.fast and self.riemann=="hllc" and mechanically_stationary(U,self.boundary)
 122:         if stationary:
 123:             acoustic_dt=math.inf
 124:         else:
 125:             sound=self.thermo.sound_speed(rho)
 126:             acoustic_dt=self.cfl/np.max((np.abs(P[...,MX])+np.abs(P[...,MY])+2*sound)/self.a.dx)
 127:         rates=self.chem.raw_rates(U,T)
 128:         has_stock=(.5*(U[...,CATION]+U[...,ANION])>self.concentration_floor)&(U[...,PVA]>0)
 129:         maxrate=float(np.max(np.where(has_stock[...,None],rates,0.)))
 130:         reaction_dt=self.alpha_step/maxrate if maxrate>0 else math.inf
 131:         dt=min(self.dtmax,remaining,thermal_dt,acoustic_dt,reaction_dt)
 132:         if not math.isfinite(dt) or dt<=0 or dt<self.min_dt:
 133:             raise PropagationCandidateNumericalError("Reactive timestep fell below minimum_time_step_s; no artificial sound-speed reduction used")
 134:         return dt
```

## C02 — python/ecsp_reactive/condensed/solver.py:136–164

```python
 136:     def rhs(self,t,U,dt,first_order=False):
 137:         hydro,boundary,fd=flux_divergence(U,self.thermo,self.a.dx,self.a.thickness,boundary=self.boundary,
 138:                 epsilon=self.epsilon,kind=self.riemann,first_order=first_order,stationary_fast_path=self.fast)
 139:         P,cp,k,conduction,loss=self._thermal_terms(U)
 140:         T=P[...,3]
 141:         jac=self.h/self.a.thickness+4*self.emissivity*self.sigma_sb/self.a.thickness*np.maximum(T,self.ambient)**3
 142:         local_cfl=_thermal_diffusive_cfl(k,P[...,0]*cp,self.a.propellant_mask,self.a.dx,dt)+dt*jac/(P[...,0]*cp)
 143:         if np.max(local_cfl)>self.thermal_cfl*(1+1e-12):
 144:             raise PropagationCandidateNumericalError("Reactive explicit thermal CFL exceeded at an RK stage")
 145:         if self.electrical is not None:
 146:             electric,qj,qe,mismatch=self.electrical.evaluate(U,T,dt)
 147:         else:
 148:             electric=np.zeros_like(U); qj=np.zeros(U.shape[:2]); qe=qj; mismatch=0.
 149:         available=U+dt*(hydro+electric)
 150:         if np.any(available[...,CATION:WATER+1]<-1e-9) or np.any(available[...,PVA]<-1e-9):
 151:             raise PropagationCandidateNumericalError("Flux/NP stage would make an inventory negative")
 152:         chemistry,cd=self.chem.source(U,T,dt,available=available,concentration_floor=self.concentration_floor,
 153:                                       allowed_rate_cap_fraction=self.allowed_cap)
 154:         derivative=hydro+electric+chemistry
 155:         derivative[...,ENERGY]+=conduction-loss
 156:         if not np.isfinite(derivative).all():
 157:             raise PropagationCandidateNumericalError("Nonfinite condensed Euler source")
 158:         # Source terms use exactly the accepted rates. No second solid update.
 159:         ledger=np.r_[boundary, [np.sum(v)*self.vol for v in
 160:                     (qj,qe,chemistry[...,ENERGY],loss,conduction)]]
 161:         source_fields={"qJ_W_per_m3":qj,"qEchem_W_per_m3":qe,
 162:                        "qChem_W_per_m3":chemistry[...,ENERGY],"conduction_W_per_m3":conduction,
 163:                        "loss_W_per_m3":loss}
 164:         return derivative,ledger,{**fd,**cd,"current_mismatch":mismatch},source_fields
```

## C03 — python/ecsp_reactive/condensed/solver.py:166–181

```python
 166:     def advance(self,t,U,dt,first_order=False):
 167:         """SSPRK(3,3), with source/boundary quadrature weights 1/6,1/6,2/3."""
 168:         k0,l0,d0,f0=self.rhs(t,U,dt,first_order)
 169:         u1=U+dt*k0; self.thermo.validate(u1)
 170:         k1,l1,d1,f1=self.rhs(t+dt,u1,dt,first_order)
 171:         u2=.75*U+.25*(u1+dt*k1); self.thermo.validate(u2)
 172:         k2,l2,d2,f2=self.rhs(t+.5*dt,u2,dt,first_order)
 173:         result=U/3+(2/3)*(u2+dt*k2); self.thermo.validate(result)
 174:         # Restore exactly constant mechanically inactive fields only if they
 175:         # were analytically unchanged: avoid RK convex-combination roundoff
 176:         # breaking the uniform-rho invariant and triggering acoustic work.
 177:         if all(d["mechanics_skipped_exact"] for d in (d0,d1,d2)):
 178:             result[...,:3]=U[...,:3]
 179:         integrated=dt*(l0/6+l1/6+2*l2/3)
 180:         sources={k:f0[k]/6+f1[k]/6+2*f2[k]/3 for k in f0}
 181:         return result,integrated,(d0,d1,d2),sources
```

## C04 — python/ecsp_reactive/condensed/thermo.py:103–104

```python
 103:     def internal_energy(self, rho, temperature):
 104:         return self.cold_energy(rho) + self.heat.sensible_energy(temperature)
```

## C05 — python/ecsp_reactive/condensed/chemistry.py:62–99

```python
  62:     def source(self, U, temperature, dt, *, available=None, concentration_floor=0.0, allowed_rate_cap_fraction=0.0):
  63:         """Return accepted conservative source, with one common resource limiter.
  64: 
  65:         available is the forward-Euler inventory after NP/Faradaic transport;
  66:         it permits source splitting within one SSP stage without double use.
  67:         Heat equals exactly Q1*delta(rho alpha1)+Q2*delta(rho alpha2).
  68:         """
  69:         rho = U[...,RHO]
  70:         alpha = U[...,A1:A2+1]/rho[...,None]
  71:         rates = self.raw_rates(U, temperature)
  72:         cap_fraction = float(np.mean(np.any(rates>self.maximum_rate, axis=-1)))
  73:         if cap_fraction > allowed_rate_cap_fraction+1e-15:
  74:             raise PropagationCandidateNumericalError("BC chemical rate cap would change the model; candidate rejected")
  75:         delta = np.minimum(np.maximum(1-alpha,0), dt*np.minimum(rates,self.maximum_rate))
  76:         proposed_xi = rho*self.xi_per_kg*(delta @ self.weights)
  77:         resources = U if available is None else available
  78:         if available is not None:
  79:             capacity = np.maximum(resources[...,RHO,None]-resources[...,A1:A2+1],0) / rho[...,None]
  80:             delta = np.minimum(delta,capacity)
  81:             proposed_xi = rho*self.xi_per_kg*(delta @ self.weights)
  82:         salt = .5*(resources[...,CATION]+resources[...,ANION])
  83:         maximum_xi = np.minimum(np.maximum(salt-concentration_floor,0)/1.45,
  84:                                 np.maximum(resources[...,PVA],0))
  85:         scale = np.ones_like(rho)
  86:         positive = proposed_xi>0
  87:         scale[positive] = np.minimum(1,maximum_xi[positive]/proposed_xi[positive])
  88:         delta *= scale[...,None]
  89:         d_rho_alpha = rho[...,None]*delta
  90:         d_xi = self.xi_per_kg*(d_rho_alpha @ self.weights)
  91:         S = np.zeros_like(U)
  92:         S[...,A1:A2+1] = d_rho_alpha/dt
  93:         S[...,ENERGY] = (d_rho_alpha @ self.Q)/dt
  94:         S[...,CATION] = S[...,ANION] = -1.45*d_xi/dt
  95:         S[...,PVA] = -d_xi/dt
  96:         S[...,PRODUCT_WATER] = 2*d_xi/dt
  97:         return S, {"chemical_rate_cap_fraction":cap_fraction,
  98:                    "inventory_limiter_fraction":float(np.mean(scale<1-1e-14)),
  99:                    "maximum_raw_rate_per_s":float(np.max(rates))}
```

## C06 — python/ecsp_nsga2/nsga2.py:107–134

```python
 107: def assign_crowding_distance(front: Sequence[Individual]) -> None:
 108:     if not front:
 109:         return
 110:     for ind in front:
 111:         ind.crowding_distance = 0.0
 112:     if len(front) <= 2:
 113:         for ind in front:
 114:             ind.crowding_distance = inf
 115:         return
 116: 
 117:     if front[0].objectives is None:
 118:         raise ValueError("Crowding distance requires evaluated individuals")
 119:     m = len(front[0].objectives)
 120:     for k in range(m):
 121:         ordered = sorted(front, key=lambda x: float(x.objectives[k]))
 122:         lo = float(ordered[0].objectives[k])
 123:         hi = float(ordered[-1].objectives[k])
 124:         span = hi - lo
 125:         if span <= 1e-30:
 126:             continue
 127:         ordered[0].crowding_distance = inf
 128:         ordered[-1].crowding_distance = inf
 129:         for i in range(1, len(ordered) - 1):
 130:             if np.isinf(ordered[i].crowding_distance):
 131:                 continue
 132:             prev_v = float(ordered[i - 1].objectives[k])
 133:             next_v = float(ordered[i + 1].objectives[k])
 134:             ordered[i].crowding_distance += (next_v - prev_v) / span
```

## C07 — python/ecsp_nsga2/post_onset_batch.py:52–154

```python
  52: def run_post_onset_batch(handoffs,propagation_config,bc_config,output_dirs,*,post_onset_config,full_bc_config=None):
  53:     from ecsp_reactive.condensed.tensor_solver import validate_execution_config,run_tensor_propagation_batch
  54:     import torch
  55:     cfg=validate_post_onset_config(post_onset_config)
  56:     ex=validate_execution_config(cfg.get('execution',{}))
  57:     if len(handoffs)!=len(output_dirs):raise PropagationConfigurationError('Post-onset batch input/output length mismatch')
  58:     if not handoffs:return []
  59:     if cfg['compare_backends'] and propagation_config.get('continued_electrical_heating',False):
  60:         raise PropagationConfigurationError('Paired comparison requires power-off on both models')
  61:     n=len(handoffs);out=[Path(o) for o in output_dirs]
  62:     if len({str(p.resolve()) for p in out})!=n:
  63:         raise PropagationConfigurationError('Post-onset output directories must be unique')
  64:     hashes=[handoff_digest(h) for h in handoffs];results=[None]*n
  65:     budget=available_cpu_budget(ex['cpu_budget']);reserve=min(ex['host_reserve'],max(0,budget-1))
  66:     workers=max(1,min(ex['cpu_workers'],max(1,budget-reserve),n))
  67:     cpu_post=copy.deepcopy(cfg);cpu_post.pop('execution',None)
  68:     if ex['backend']=='torch_batch' and str(ex['device']).startswith('cuda') and not torch.cuda.is_available():
  69:         raise PropagationConfigurationError('CUDA requested for Reactive batch but unavailable; no implicit CPU fallback')
  70:     torch.set_num_threads(min(ex['torch_threads'],max(1,reserve or budget)))
  71:     metadata={'schema':'ecsp.post-onset-batch/v1','requested_cpu_budget':ex['cpu_budget'],'detected_cpu_budget':budget,
  72:               'cpu_workers':workers,'host_reserve':reserve,'process_start_method':'spawn',
  73:               'execution':ex,'case_count':n,'candidate_order_preserved':True,'cuda_oom_splits':0,
  74:               'note':'execution evidence, not material/experimental validation'}
  75:     start=time.perf_counter()
  76:     with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=_cpu_init) as pool:
  77:         if ex['backend']=='numpy_cpu':
  78:             # A bounded queue avoids pickling every large handoff at once.
  79:             pending={};next_index=0
  80:             while next_index<n or pending:
  81:                 while next_index<n and len(pending)<2*workers:
  82:                     i=next_index;next_index+=1
  83:                     pending[i]=pool.submit(_cpu_case,handoffs[i],propagation_config,bc_config,str(out[i]),cpu_post,full_bc_config)
  84:                 i=min(pending);results[i]=pending.pop(i).result()
  85:         else:
  86:             if cfg['backend']!='reactive_euler' and not cfg['compare_backends']:
  87:                 raise PropagationConfigurationError('torch_batch requested without an active Reactive solver')
  88:             baseline_futures={};baseline_results={};baseline_cursor=0
  89:             successful=[i for i,h in enumerate(handoffs) if bool(h.get('onsetSucceeded',False))]
  90:             if cfg['compare_backends']:
  91:                 def fill_baselines():
  92:                     nonlocal baseline_cursor
  93:                     ready=[i for i,f in baseline_futures.items() if f.done()]
  94:                     for i in ready:baseline_results[i]=baseline_futures.pop(i).result()
  95:                     while baseline_cursor<len(successful) and len(baseline_futures)<2*workers:
  96:                         i=successful[baseline_cursor];baseline_cursor+=1
  97:                         baseline_futures[i]=pool.submit(_cpu_baseline,handoffs[i],propagation_config,bc_config,str(out[i]))
  98:                 fill_baselines()
  99:             for i,h in enumerate(handoffs):
 100:                 if not bool(h.get('onsetSucceeded',False)):
 101:                     results[i]=_cpu_case(h,propagation_config,bc_config,str(out[i]),cpu_post,full_bc_config)
 102:             reactive={}
 103:             def group(indices):
 104:                 if not indices:return
 105:                 try:
 106:                     rr=run_tensor_propagation_batch([handoffs[i] for i in indices],propagation_config,bc_config,
 107:                              [out[i]/'reactive_euler' for i in indices],reactive_config=cfg['reactive_euler'],
 108:                              execution_config=ex,full_bc_config=full_bc_config)
 109:                     reactive.update(zip(indices,rr))
 110:                 except torch.cuda.OutOfMemoryError:
 111:                     if not ex['oom_split_retry'] or len(indices)==1:raise
 112:                     metadata['cuda_oom_splits']+=1
 113:                     torch.cuda.empty_cache();mid=len(indices)//2;group(indices[:mid]);group(indices[mid:])
 114:                 except PropagationConfigurationError as exc:
 115:                     if 'exceeds estimated memory budget' not in str(exc) or len(indices)==1:raise
 116:                     metadata['estimated_memory_splits']=metadata.get('estimated_memory_splits',0)+1
 117:                     mid=len(indices)//2;group(indices[:mid]);group(indices[mid:])
 118:                 except PropagationCandidateError as exc:
 119:                     # Handoff/nonlinear callback failures can originate in just
 120:                     # one lane. Isolate by rerunning same physics on the SAME device.
 121:                     if len(indices)==1:reactive[indices[0]]=exc
 122:                     else:
 123:                         metadata['candidate_isolation_splits']=metadata.get('candidate_isolation_splits',0)+1
 124:                         mid=len(indices)//2;group(indices[:mid]);group(indices[mid:])
 125:             for start_i in range(0,len(successful),ex['batch_size']):
 126:                 if cfg['compare_backends']:fill_baselines()
 127:                 group(successful[start_i:start_i+ex['batch_size']])
 128:             if cfg['compare_backends']:
 129:                 while baseline_cursor<len(successful) or baseline_futures:
 130:                     fill_baselines()
 131:                     if baseline_futures:
 132:                         i=min(baseline_futures);baseline_results[i]=baseline_futures.pop(i).result()
 133:             for i in successful:
 134:                 out[i].mkdir(parents=True,exist_ok=True)
 135:                 paired={};errors={}
 136:                 for name,value in [('reactive_euler',reactive[i])]+([('condensed_propagation',baseline_results[i])] if cfg['compare_backends'] else []):
 137:                     if isinstance(value,PropagationCandidateError):
 138:                         errors[name]=value;paired[name]={'status':'failed','errorType':type(value).__name__,'message':str(value)}
 139:                         (out[i]/name).mkdir(exist_ok=True);_json(out[i]/name/'PROPAGATION_FAILED.json',paired[name])
 140:                     else:paired[name]=dict(value)
 141:                 comparison=_comparison(paired,hashes[i],out[i]) if cfg['compare_backends'] else None
 142:                 selected=cfg['backend']
 143:                 if selected in errors:results[i]=errors[selected]
 144:                 else:
 145:                     result=dict(paired[selected]);result.update(postOnsetBackend=selected,
 146:                         propagationDirectory=str(out[i]/selected),postOnsetSameHandoffSHA256=hashes[i],
 147:                         postOnsetBackendComparison=comparison,experimentalReactiveRankingRequested=selected=='reactive_euler')
 148:                     results[i]=result
 149:     metadata['wall_clock_s']=time.perf_counter()-start
 150:     for i,h in enumerate(handoffs):
 151:         if handoff_digest(h)!=hashes[i]:raise RuntimeError('Post-onset batch mutated input handoff')
 152:         out[i].mkdir(parents=True,exist_ok=True)
 153:         _json(out[i]/'post_onset_execution.json',metadata)
 154:     return results
```

## C08 — python/ecsp_reactive/condensed/tensor_solver.py:23–53

```python
  23: def validate_execution_config(raw=None):
  24:     cfg=copy.deepcopy(dict(raw or {}))
  25:     allowed={'backend','device','batch_size','cuda_graph','cpu_budget','host_reserve','cpu_workers','torch_threads',
  26:              'maximum_batch_working_bytes','maximum_batch_history_bytes','oom_split_retry'}
  27:     if set(cfg)-allowed:
  28:         raise PropagationConfigurationError('Unknown post_onset.execution keys: '+', '.join(sorted(set(cfg)-allowed)))
  29:     cfg.setdefault('backend','numpy_cpu');cfg.setdefault('device','cpu')
  30:     if cfg['backend'] not in {'numpy_cpu','torch_batch'}:
  31:         raise PropagationConfigurationError('post_onset.execution.backend must be numpy_cpu or torch_batch')
  32:     device=str(cfg['device'])
  33:     if device!='cpu' and device!='cuda' and not (device.startswith('cuda:') and device[5:].isdigit()):
  34:         raise PropagationConfigurationError('Reactive device must be cpu, cuda, or cuda:<index>; FP32 MPS is not used')
  35:     if cfg['backend']=='numpy_cpu' and device!='cpu':
  36:         raise PropagationConfigurationError('numpy_cpu cannot request a CUDA device')
  37:     for name,default,minimum,maximum in (
  38:         ('batch_size',8,1,256),('cpu_budget',8,1,65536),('host_reserve',2,0,65535),
  39:         ('cpu_workers',6,1,65536),('torch_threads',1,1,256),
  40:         ('maximum_batch_working_bytes',8*1024**3,1024,2**63-1),
  41:         ('maximum_batch_history_bytes',2*1024**3,1024,2**63-1)):
  42:         cfg[name]=_configuration_integer(cfg.get(name,default),name='post_onset.execution.'+name,minimum=minimum,maximum=maximum)
  43:     for name,default in (('cuda_graph',False),('oom_split_retry',True)):
  44:         v=cfg.get(name,default)
  45:         if not isinstance(v,bool):raise PropagationConfigurationError(name+' must be boolean')
  46:         cfg[name]=v
  47:     if cfg['cuda_graph'] and (cfg['backend']!='torch_batch' or not device.startswith('cuda')):
  48:         raise PropagationConfigurationError('cuda_graph requires torch_batch on CUDA')
  49:     if cfg['host_reserve']>=cfg['cpu_budget']:
  50:         raise PropagationConfigurationError('host_reserve must be below cpu_budget')
  51:     if cfg['cpu_workers']+cfg['host_reserve']>cfg['cpu_budget']:
  52:         raise PropagationConfigurationError('CPU workers + host reserve exceeds requested CPU budget')
  53:     return cfg
```
