#include "bc_engine.h"
#include <ATen/TensorIndexing.h>
#include <c10/core/InferenceMode.h>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <limits>
using namespace at::indexing;
namespace bc_native {
static double get(const Settings&c,const std::string&k,double d){auto i=c.find(k);return i==c.end()?d:i->second;}
static int get_int(const Settings&c,const std::string&k,int d){
 double v=get(c,k,d);
 TORCH_CHECK(v>=std::numeric_limits<int>::min()&&v<=std::numeric_limits<int>::max()&&v==std::floor(v),
             "native integer setting is invalid: ",k);
 return static_cast<int>(v);
}
static Params params(const Settings&c,int b,int n){
 Params p;p.n=n;p.batch=b;
#define RI(x) p.x=get_int(c,#x,p.x)
 RI(full_bv);RI(full_np);RI(diffusion);RI(augmented);RI(joule_mode);RI(local_min);RI(local_max);RI(newton);
#undef RI
#define RD(x) p.x=get(c,#x,p.x)
 RD(R);RD(F);RD(zp);RD(zm);RD(dx);RD(dt);RD(sigma_min);RD(sigma_max);RD(initial_T);RD(ambient);
 RD(cp0);RD(cm0);RD(water0);RD(lp0);RD(pva0);RD(m_lp);RD(m_pva);RD(density);RD(w1);RD(w2);RD(rate_max);RD(layer);
 RD(min_T);RD(max_T);RD(convection);RD(emissivity);RD(sb);RD(min_fraction);RD(max_multiple);RD(max_delta);RD(lim_rtol);RD(conc_floor);
 RD(cathode);RD(exp_limit);RD(local_abs);RD(local_rel);RD(roundoff);RD(conductivity_floor);RD(slope_floor);RD(contact_normal);
 RD(thermal_cfl_safety);
#undef RD
 for(int i=0;i<6;i++){auto key=std::string("p")+std::to_string(i)+".";auto&q=p.prop[i];q.mode=get_int(c,key+"mode",0);q.offset=get_int(c,key+"offset",0);q.size=get_int(c,key+"size",0);q.value=get(c,key+"value",0);q.tref=get(c,key+"tref",298.15);q.ea=get(c,key+"ea",0);}
 for(int i=0;i<2;i++){auto k=std::string("k")+std::to_string(i)+".";p.kin[i].offset=get_int(c,k+"offset",0);p.kin[i].size=get_int(c,k+"size",0);p.kin[i].heat=get(c,k+"heat",0);}
 for(int i=0;i<4;i++){auto k=std::string("ch")+std::to_string(i)+".";auto&q=p.ch[i];q.eq=get(c,k+"eq",0);q.j0=get(c,k+"j0",0);q.alpha=get(c,k+"alpha",.5);q.n=get(c,k+"n",1);q.reverse=get(c,k+"reverse",1);q.heat=get(c,k+"heat",0);q.stoich=get(c,k+"stoich",0);}
 return p;
}
static Tensor b3(const Tensor&t){return t.reshape({-1,1,1});}
static Tensor sum2(const Tensor&t){return t.sum({-2,-1});}
static Tensor max2(const Tensor&t){return t.amax({-2,-1});}
static Tensor scaled_l2_norm(const Tensor&t){
 // sqrt(sum(t*t)) overflows for finite components above roughly 1e154.  Scale
 // before squaring (LAPACK xLASSQ principle), then reject a genuinely
 // non-representable result through finite_norm_converged below.
 auto scale=max2(t.abs());auto finite=at::isfinite(scale),positive=finite&(scale>0);
 auto safe=at::where(positive,scale,at::ones_like(scale));auto normalized=t/b3(safe);
 auto norm=safe*sum2(normalized*normalized).clamp_min(0).sqrt();
 return at::where(scale==0,at::zeros_like(scale),at::where(finite,norm,scale));
}
static Tensor finite_norm_converged(const Tensor&norm,const Tensor&tolerance){
 return at::isfinite(norm)&at::isfinite(tolerance)&(norm<=tolerance);
}
static Tensor shift(const Tensor&x,int d){
 auto y=at::zeros_like(x);int n=x.size(-1);
 if(d==0)y.index_put_({Ellipsis,Slice(),Slice(0,n-1)},x.index({Ellipsis,Slice(),Slice(1,n)}));
 if(d==1)y.index_put_({Ellipsis,Slice(),Slice(1,n)},x.index({Ellipsis,Slice(),Slice(0,n-1)}));
 if(d==2)y.index_put_({Ellipsis,Slice(0,n-1),Slice()},x.index({Ellipsis,Slice(1,n),Slice()}));
 if(d==3)y.index_put_({Ellipsis,Slice(1,n),Slice()},x.index({Ellipsis,Slice(0,n-1),Slice()}));
 return y;
}
struct Geometry {
 Tensor an,ca,prop,active,count,contact_cells,contact_anode,contact_dense_index;int b,n;int64_t total;
 Geometry(Tensor a,Tensor c):an(a),ca(c),prop(at::ones_like(a)),b(a.size(0)),n(a.size(1)),total(a.numel()){
  // Cell-centred finite volumes cover the complete surface, including the
  // outer ring. Missing exterior neighbours are the natural no-flux boundary.
  active=prop.clone();
  count=sum2(prop.to(at::kDouble)).clamp_min(1);
  auto contact=(an|ca).reshape({-1});
  contact_cells=at::nonzero(contact).reshape({-1});
  contact_anode=an.reshape({-1}).index_select(0,contact_cells);
  contact_dense_index=contact_cells;
 }
 Tensor dense(const Tensor&x) const {auto o=at::zeros({b,n,n},x.options());o.reshape({-1}).index_copy_(0,contact_dense_index,x);return o;}
};
struct Boundary {Tensor ia,ic,sensitivity,local_res,jwa,jwc,jla,jlc,ja,jc,ka,kc,normal_power;};
static Boundary boundary(Tensor s,const Tensor&tr,const Tensor&v,const Tensor&running,const Params&p,const Geometry&g){
 auto f=contacts(g.contact_cells,g.contact_anode,s,tr,v,running,p);
 auto w=g.dense(f[0]),l=g.dense(f[1]),sens=g.dense(f[3]),err=g.dense(f[4]),normal_power=g.dense(f[8]);
 auto aw=at::where(g.an,w,0),cw=at::where(g.ca,w,0),al=at::where(g.an,l,0),cl=at::where(g.ca,l,0);
 auto ka=at::where(g.an,sens,0),kc=at::where(g.ca,sens,0);
 // A top-surface contact cell represents dx^2 of physical footprint area.
 double contact_area=p.dx*p.dx;
 Boundary r;r.ia=sum2(aw+al)*contact_area;r.ic=sum2(cw+cl)*contact_area;
 r.sensitivity=sum2(sens)*contact_area;r.local_res=max2(err);
 r.jwa=aw;r.jla=al;r.jwc=cw;r.jlc=cl;r.ja=aw+al;r.jc=cw+cl;r.ka=ka;r.kc=kc;r.normal_power=normal_power;return r;
}
struct Linear {Tensor coeff,b,active;};
static Tensor harmonic_tensor(const Tensor&a,const Tensor&b){
 auto valid=at::isfinite(a)&at::isfinite(b)&(a>=0)&(b>=0);
 auto lo=at::minimum(a,b),hi=at::maximum(a,b);
 auto safe_hi=at::where(hi>0,hi,at::ones_like(hi));
 // Algebraically 2ab/(a+b), but neither the product nor the sum can overflow.
 auto value=lo*(2.0/(1.0+lo/safe_hi));
 value=at::where(lo==0,at::zeros_like(value),value);
 return at::where(valid,value,at::full_like(value,NAN));
}
static Linear linear_system(const Tensor&tr,const Tensor&rhs,const Boundary&bd,const Tensor&phi,const Geometry&g,const Params&p){
 auto sig=tr[5];std::vector<Tensor> all;auto diag=at::zeros_like(sig),b=-rhs*(p.dx*p.dx);
 for(int d=0;d<4;d++){
  auto sn=shift(sig,d);auto hm=harmonic_tensor(sig,sn);auto other=shift(g.active,d);
  auto c=at::where(g.active&other,hm,0);diag+=c;all.push_back(c);
 }
 // -div(sigma grad(phi))=(j_anode-j_cathode)/delta_s.  Multiplying
 // the cell equation by dx^2 gives dx^2/delta_s.  Linearising both
 // polarities adds the non-negative slope to the SPD diagonal.
 double source_scale=p.dx*p.dx/p.layer;auto slope=bd.ka+bd.kc;
 diag+=source_scale*slope;
 b+=source_scale*(bd.ja-bd.jc+slope*phi);
 TORCH_CHECK(!(g.active&((diag<=0)|~at::isfinite(diag))).any().item<bool>(),"BC native: singular/nonfinite potential diagonal");
 diag=at::where(g.active,diag,1);all.insert(all.begin(),diag);return {at::stack(all),at::where(g.active,b,0),g.active};
}
struct LinearResult {Tensor x,ok,rel,iterations;};
static LinearResult pcg(const Linear&A,Tensor initial,const Settings&cfg,const Tensor&running){
 // Symmetric Jacobi-equilibrated correction form. Check true original-system
 // residuals; never classify a failed solve as a non-igniting physics trial.
 auto active=A.active&b3(running);auto x=at::where(active,initial,0);auto b=at::where(active,A.b,0);
 auto bnorm=scaled_l2_norm(b).clamp_min(1.);auto tol=at::maximum(get(cfg,"linear_rtol",1e-9)*bnorm,at::full_like(bnorm,get(cfg,"linear_atol",1e-12)));
 auto d=A.coeff[0],scale=d.rsqrt();std::vector<Tensor> co{at::ones_like(d)};
 for(int dir=0;dir<4;dir++){co.push_back(A.coeff[dir+1]*scale*shift(scale,dir));}
 auto eq=at::stack(co);
 auto original_r=b-stencil(A.coeff,x);auto r=at::where(active,original_r*scale,0);auto pv=r.clone();auto rr=sum2(r*r);
 auto done=finite_norm_converged(scaled_l2_norm(original_r),tol)|~running;
 auto iters=at::zeros_like(running,at::TensorOptions().dtype(at::kLong));
 int maxit=get_int(cfg,"linear_max",1500),interval=std::max(1,get_int(cfg,"linear_check",8));
 for(int it=1;it<=maxit;it++){
  auto working=~done;pv=at::where(b3(working),pv,0);
  if((it==1||it%interval==0)&&!working.any().item<bool>())break;
  auto ap=stencil(eq,pv);auto pap=sum2(pv*ap);auto good=working&at::isfinite(pap)&(pap>DBL_MIN)&at::isfinite(rr);
  auto step=at::where(good,rr/at::where(good,pap,1),0);x+=b3(step)*scale*pv;r-=b3(step)*ap;
  iters=at::where(working,at::full_like(iters,it),iters);
  auto rr2=sum2(r*r);
  // True residual replaces the recurrence periodically and at convergence.
  if(it%interval==0||it==maxit){
   original_r=b-stencil(A.coeff,x);auto norm=scaled_l2_norm(original_r);
   done=finite_norm_converged(norm,tol)|~running;
   auto drift=scaled_l2_norm(original_r-r/scale);auto replace=(drift>.25*norm.clamp_min(1e-30))&~done;
   auto beta=at::where(good,rr2/rr.clamp_min(DBL_MIN),0);
   pv=r+b3(beta)*pv;
   r=at::where(b3(replace),original_r*scale,r);pv=at::where(b3(replace),r,pv);rr=sum2(r*r);
  }else{auto beta=at::where(good,rr2/rr.clamp_min(DBL_MIN),0);pv=r+b3(beta)*pv;rr=rr2;}
 }
 original_r=b-stencil(A.coeff,x);auto norm=scaled_l2_norm(original_r),relative=norm/bnorm;
 auto ok=(finite_norm_converged(norm,tol)&at::isfinite(relative))|~running;
 return {x,ok,relative,iters};
}
struct Electrical {Tensor heat,current,mismatch,linear_rel,linear_iterations,converged,outer_iterations;};
static Electrical electrical(Tensor&s,const Tensor&tr,const Tensor&v,const Tensor&running,const Params&p,const Geometry&g,const Settings&cfg){
 auto raw=currents(s,tr,g.prop,p),rhs=raw[2];
 // The distributed Robin terms on the physical contact footprint anchor the
 // absolute potential and close global charge balance.  There is no separate
 // Dirichlet electrode region or post-solve scalar gauge adjustment.
 auto bd=boundary(s,tr,v,running,p,g);auto previous=.5*(bd.ia+bd.ic);
 auto converged=~running;auto rel=at::zeros_like(previous);auto iters=at::zeros_like(running,at::TensorOptions().dtype(at::kLong)),outer=at::zeros_like(iters);
 int maxit=get_int(cfg,"outer_max",120),minit=get_int(cfg,"outer_min",2);double relax=get(cfg,"outer_relax",.35);
 Tensor dphi=at::zeros_like(previous),di=at::zeros_like(previous),combined=at::zeros_like(previous);
 for(int it=1;it<=maxit;it++){
  auto working=running&~converged;
  if(!working.any().item<bool>())break;
  auto old=s[7].clone();auto A=linear_system(tr,rhs,bd,s[7],g,p);auto linear=pcg(A,s[7],cfg,working);
  rel=at::where(working,linear.rel,rel);iters=at::where(working,linear.iterations,iters);
  TORCH_CHECK(linear.ok.all().item<bool>(),"BC native PCG failed, max true relative residual=",rel.max().item<double>());
  // Every cell, including the outer ring and cells under either contact, is a
  // finite-volume unknown.  Missing exterior neighbours simply contribute no
  // conductance, so boundary values must never be overwritten as ghost cells.
  auto proposed=at::where(g.active,linear.x,s[7]);
  auto relaxed=s[7]+relax*(proposed-s[7]);
  s[7].copy_(at::where(b3(working)&g.prop,relaxed,s[7]));
  bd=boundary(s,tr,v,running,p,g);
  auto total=.5*(bd.ia+bd.ic);
  dphi=max2(at::where(g.prop,(s[7]-old).abs(),0));di=(total-previous).abs()/at::maximum(total.abs(),previous.abs()).clamp_min(1e-12);
  auto current_scale=at::maximum(bd.ia.abs(),bd.ic.abs());
  auto balance_threshold=at::maximum(current_scale*get(cfg,"balance_rel",.005),at::full_like(current_scale,get(cfg,"balance_abs",1e-12)));
  combined=(bd.ia-bd.ic).abs()/balance_threshold.clamp_min(DBL_MIN);
  auto reached=(dphi<=get(cfg,"outer_phi_tol",.01))&(di<=get(cfg,"outer_current_tol",.005))&
               (combined<=1)&(bd.local_res<=1)&linear.ok&(it>=minit);
  converged=converged|(working&reached);outer=at::where(working,at::full_like(outer,it),outer);previous=at::where(working,total,previous);
  if(converged.all().item<bool>())break;
 }
 TORCH_CHECK(converged.all().item<bool>(),"BC native surface-overlay BV did not converge: max dphi=",dphi.max().item<double>()," dI=",di.max().item<double>()," balance=",combined.max().item<double>()," local=",bd.local_res.max().item<double>());
 raw=currents(s,tr,g.prop,p);
 // Internal heat/source bundle: qJ, qEchem-total, LP sink, mobile-water
 // sink, |J|, qEchem-water, qEchem-LP.  The split heat fields allow the
 // integrator to scale heat by the actually accepted Faradaic inventory.
 auto heat=at::zeros({7,g.b,g.n,g.n},s.options());
 // raw[1] is conservative in-plane face-network power density.  The compact
 // contact solve returns W/m^2 for the unresolved normal resistor.
 heat[0].copy_(raw[1]+bd.normal_power/p.layer);heat[4].copy_(raw[0]);
 std::vector<Tensor> js{bd.jwa,bd.jwc,bd.jla,bd.jlc};
 for(int i=0;i<4;i++){
  // A/m^2 on the top contact becomes A/m^3 in the condensed surface layer.
  auto mol=js[i]/(p.ch[i].n*p.F*p.layer);
  auto q=(mol*p.ch[i].heat).clamp_min(0);heat[1].add_(q);heat[i<2?5:6].add_(q);
  heat[i<2?3:2].add_(mol*p.ch[i].stoich);
 }
 auto mismatch=(bd.ia-bd.ic).abs()/at::maximum(bd.ia.abs(),bd.ic.abs()).clamp_min(get(cfg,"current_floor",1e-15));
 return {heat,.5*(bd.ia+bd.ic),mismatch,rel,iters,converged,outer};
}
static Tensor quantile_masked(const Tensor&f,const Tensor&m,double q){
 // No variable-size GPU gathers inside the time loop: sort padded rows,
 // then interpolate at the exact masked rank (the reference quantile rule).
 auto flat=at::where(m,f,INFINITY).reshape({f.size(0),-1});auto sorted=std::get<0>(flat.sort(1));
 auto count=m.reshape({m.size(0),-1}).sum(1);auto pos=(count.to(at::kDouble)-1).clamp_min(0)*q;
 auto lo=pos.floor().to(at::kLong),hi=pos.ceil().to(at::kLong),w=pos-lo;
 auto low=sorted.gather(1,lo.unsqueeze(1)).squeeze(1),high=sorted.gather(1,hi.unsqueeze(1)).squeeze(1);
 return at::where(count>0,low+w*(high-low),0);
}
struct TimeLoopHostStatus {
 bool any_running;
 int64_t maximum_steps_completed;
 double latest_onset_time;
};
static Tensor invalid_constitutive_cells(
    const Tensor&tr,const Tensor&active,const Params&p){
 auto invalid=~at::isfinite(tr[0])|(tr[0]<0)|
              ~at::isfinite(tr[1])|(tr[1]<0)|
              ~at::isfinite(tr[2])|(tr[2]<0)|
              ~at::isfinite(tr[3])|(tr[3]<0)|
              ~at::isfinite(tr[4])|(tr[4]<0)|
              ~at::isfinite(tr[5])|(tr[5]<p.sigma_min)|(tr[5]>p.sigma_max)|
              ~at::isfinite(tr[6])|(tr[6]<=0)|
              ~at::isfinite(p.density*tr[6])|(p.density*tr[6]<=0)|
              ~at::isfinite(tr[7])|(tr[7]<0);
 return active&invalid;
}
static TimeLoopHostStatus check_time_loop_failures(
    const Tensor&invalid_thermal_property,const Tensor&rejected_echem,
    const Tensor&invalid_thermal_cfl,const Tensor&invalid_integrated_state,
    const Tensor&invalid_updated_thermal_property,
    const Tensor&running,
    const Tensor&steps_completed,const Tensor&ignition_delay,
    const Tensor&state,const Tensor&worst_thermal_cfl,
    double thermal_cfl_safety){
 // One compact device-to-host transfer services all fail-closed checks and
 // the early-stop decision.  The time loop calls this helper only at the
 // configured host-check cadence and at its final step.
 auto device_status=at::stack(std::vector<Tensor>{
   invalid_thermal_property.any().to(at::kDouble),
   rejected_echem.any().to(at::kDouble),
   invalid_thermal_cfl.any().to(at::kDouble),
   (invalid_integrated_state.any()|
    at::logical_not(at::isfinite(state)).any()).to(at::kDouble),
   invalid_updated_thermal_property.any().to(at::kDouble),
   running.any().to(at::kDouble),
   steps_completed.max().to(at::kDouble),
   worst_thermal_cfl.max(),
   at::where(at::isfinite(ignition_delay),ignition_delay,0).max(),
 });
 auto host_status=device_status.cpu().contiguous();
 const double*status=host_status.data_ptr<double>();
  TORCH_CHECK(status[0]==0,
             "BC native thermal properties became nonfinite/nonphysical or transport constitutive properties became invalid; heat capacity must be positive and conductivity/diffusivity fields must be finite and nonnegative within configured conductivity bounds");
 TORCH_CHECK(status[1]==0,
             "BC native electrochemical inventory limiter triggered; reduce timeStep_s or revise the interfacial model/current bracket");
 TORCH_CHECK(status[2]==0,
             "BC native total explicit thermal stability CFL limit exceeded: max=",status[7],
             " safety=",thermal_cfl_safety,"; reduce timeStep_s or coarsen the grid");
 TORCH_CHECK(status[3]==0,"BC native nonfinite integrated state");
  TORCH_CHECK(status[4]==0,
             "BC native thermal properties became nonfinite/nonphysical or transport constitutive properties became invalid at the accepted updated temperature; heat capacity must be positive and conductivity/diffusivity fields must be finite and nonnegative within configured conductivity bounds");
 return {status[5]!=0,static_cast<int64_t>(status[6]),status[8]};
}
std::map<std::string,Tensor> run(const Tensor&anode,const Tensor&cathode,const Tensor&voltage,const Tensor&table,const Settings&cfg){
 c10::InferenceMode guard;
 for(const auto&item:cfg)TORCH_CHECK(std::isfinite(item.second),"nonfinite native setting: ",item.first);
 TORCH_CHECK(anode.dim()==3&&anode.size(0)>0&&anode.size(1)==anode.size(2)&&anode.size(1)>=5,"expected square [B,N,N], N>=5");
 TORCH_CHECK(anode.sizes()==cathode.sizes()&&anode.device()==cathode.device(),"geometry mismatch");
 TORCH_CHECK(anode.scalar_type()==at::kBool&&cathode.scalar_type()==at::kBool,"masks must be bool");
 TORCH_CHECK(voltage.scalar_type()==at::kDouble&&table.scalar_type()==at::kDouble,"FP64 required");
 TORCH_CHECK(voltage.device()==anode.device()&&table.device()==anode.device(),"all inputs must share a device");
 TORCH_CHECK(voltage.numel()==anode.size(0)&&anode.is_contiguous()&&cathode.is_contiguous()&&voltage.is_contiguous()&&table.is_contiguous(),"invalid tensor layout");
 TORCH_CHECK(anode.device().is_cpu()||anode.device().is_cuda(),"native B/C supports CPU/CUDA only, not MPS");
#ifndef WITH_CUDA
 TORCH_CHECK(!anode.is_cuda(),"CPU-only extension; rebuild with CUDA toolkit");
#endif
 // A zero-copy Torch bool tensor can inherit noncanonical 0xff true bytes
 // from a Pillow mode-1 NumPy array.  Reinterpret the storage as bytes before
 // testing truth, then materialise valid 0/1 bool objects.  This protects the
 // direct pybind entry point even when callers bypass the Python adapter.
 auto canonical_anode=anode.view(at::kByte).ne(0).contiguous();
 auto canonical_cathode=cathode.view(at::kByte).ne(0).contiguous();
 TORCH_CHECK(!(canonical_anode&canonical_cathode).any().item<bool>(),"contact-mask overlap");
 TORCH_CHECK((sum2(canonical_anode.to(at::kLong))>0).all().item<bool>(),"each candidate must contain an anode contact cell");
 TORCH_CHECK((sum2(canonical_cathode.to(at::kLong))>0).all().item<bool>(),"each candidate must contain a cathode contact cell");
 TORCH_CHECK(table.dim()==1&&voltage.dim()==1,"table and voltages must be vectors");
 TORCH_CHECK(at::isfinite(voltage).all().item<bool>()&&at::isfinite(table).all().item<bool>(),"nonfinite native input");
 auto opts=voltage.options();Geometry g(canonical_anode,canonical_cathode);Params p=params(cfg,g.b,g.n);
 TORCH_CHECK((p.full_bv==0||p.full_bv==1)&&(p.full_np==0||p.full_np==1)&&
             (p.diffusion==0||p.diffusion==1)&&(p.augmented==0||p.augmented==1)&&
             p.joule_mode>=0&&p.joule_mode<=2,"invalid native model selector");
 TORCH_CHECK(p.local_min>=1&&p.local_max>=p.local_min&&p.newton>=0,"invalid local contact iteration budget");
 TORCH_CHECK(p.local_abs>=0&&p.local_rel>=0&&(p.local_abs>0||p.local_rel>0)&&p.roundoff>=1&&p.exp_limit>0,
             "invalid local contact tolerance");
 for(int property_index=0;property_index<6;property_index++){
  const auto& item=p.prop[property_index];
  TORCH_CHECK(item.mode>=0&&item.mode<=2,"invalid property mode");
  if(item.mode==1){
   TORCH_CHECK(item.offset>=0&&item.size>=2&&int64_t(item.offset)+2LL*item.size<=table.numel(),"property table bounds");
   auto xs=table.slice(0,item.offset,item.offset+item.size);
   auto ys=table.slice(0,item.offset+item.size,item.offset+2*item.size);
   TORCH_CHECK((xs.slice(0,1,item.size)>xs.slice(0,0,item.size-1)).all().item<bool>(),"property temperatures must be strictly increasing");
   TORCH_CHECK((ys>=0).all().item<bool>(),"property values must be non-negative");
   if(property_index==4)TORCH_CHECK((ys>0).all().item<bool>(),"heat capacity must be strictly positive");
  }else{
   TORCH_CHECK(item.value>=0,"property reference value must be non-negative");
   if(property_index==4)TORCH_CHECK(item.value>0,"heat capacity must be strictly positive");
   if(item.mode==2)TORCH_CHECK(item.tref>0&&item.ea>=0,"invalid Arrhenius property parameters");
  }
 }
 for(const auto& item:p.kin){
  TORCH_CHECK(item.offset>=0&&item.size>=2&&int64_t(item.offset)+3LL*item.size<=table.numel(),"kinetic table bounds");
  auto xs=table.slice(0,item.offset,item.offset+item.size);
  auto ea=table.slice(0,item.offset+item.size,item.offset+2*item.size);
  TORCH_CHECK((xs.slice(0,1,item.size)>xs.slice(0,0,item.size-1)).all().item<bool>(),"kinetic progress grid must be strictly increasing");
  TORCH_CHECK(std::abs(xs[0].item<double>())<=1e-12&&std::abs(xs[item.size-1].item<double>()-1)<=1e-12,
              "kinetic progress grid must span zero to one");
  TORCH_CHECK((ea>=0).all().item<bool>()&&item.heat>=0,"invalid kinetic activation energy/heat release");
 }
 for(const auto& item:p.ch)TORCH_CHECK(item.j0>=0&&item.alpha>=0&&item.alpha<=1&&item.n>0&&item.reverse>=0&&item.heat>=0&&item.stoich>=0,
                                      "invalid Butler-Volmer channel parameters");
 TORCH_CHECK((voltage>p.cathode).all().item<bool>(),"voltage must exceed cathode potential");
 TORCH_CHECK(p.dt>0&&p.dx>0&&p.layer>0&&p.contact_normal>0&&p.density>0,"nonpositive physical scale");
 double cell_area=p.dx*p.dx,cell_volume=cell_area*p.layer;
 double source_scale=cell_area/p.layer,max_contact_conductance=p.sigma_max/p.contact_normal;
 TORCH_CHECK(std::isfinite(cell_area)&&cell_area>0&&std::isfinite(cell_volume)&&cell_volume>0&&
             std::isfinite(source_scale)&&source_scale>0&&
             std::isfinite(max_contact_conductance)&&max_contact_conductance>0,
             "native geometric/source scales are not representable as positive finite FP64 values");
 TORCH_CHECK(p.R>0&&p.F>0&&p.sigma_min>0&&p.sigma_max>=p.sigma_min&&p.conductivity_floor>0,"invalid physical constant/conductivity bounds");
 TORCH_CHECK(get(cfg,"current_floor",1e-15)>0&&get(cfg,"j_floor",1e-12)>0,
             "native current/current-density floors must be strictly positive");
 TORCH_CHECK(p.cp0>0&&p.cm0>0&&p.lp0>0&&p.pva0>0&&p.water0>=0&&p.m_lp>0&&p.m_pva>0,"invalid reactive inventory");
 TORCH_CHECK(p.w1>=0&&p.w2>=0&&std::abs(p.w1+p.w2-1)<=1e-12,"invalid kinetic progress weights");
 TORCH_CHECK(p.min_fraction>=0&&p.min_fraction<1&&p.max_multiple>=1&&p.max_delta>0&&p.conc_floor>=0&&p.lim_rtol>=0,"invalid concentration bounds");
 TORCH_CHECK(p.min_T<p.max_T&&p.initial_T>=p.min_T&&p.initial_T<=p.max_T&&
             p.ambient>=p.min_T&&p.ambient<=p.max_T,"invalid temperature bounds");
 TORCH_CHECK(p.convection>=0&&p.emissivity>=0&&p.emissivity<=1&&p.sb>0,"invalid surface heat-loss coefficients");
 TORCH_CHECK(p.rate_max>0&&p.slope_floor>=0,"invalid kinetic rate/slope floor");
 TORCH_CHECK(p.thermal_cfl_safety>0&&p.thermal_cfl_safety<=1,"thermal CFL safety factor must lie in (0,1]");
 int linear_max=get_int(cfg,"linear_max",1500),linear_check=get_int(cfg,"linear_check",8),time_check=get_int(cfg,"time_check",16);
 int outer_max=get_int(cfg,"outer_max",120),outer_min=get_int(cfg,"outer_min",2);
 double linear_rtol=get(cfg,"linear_rtol",1e-9),linear_atol=get(cfg,"linear_atol",1e-12);
 double outer_relax=get(cfg,"outer_relax",.35),outer_phi_tol=get(cfg,"outer_phi_tol",.01),outer_current_tol=get(cfg,"outer_current_tol",.005);
 double balance_rel=get(cfg,"balance_rel",.005),balance_abs=get(cfg,"balance_abs",1e-12);
 TORCH_CHECK(linear_max>=1&&linear_check>=1&&time_check>=1&&outer_min>=1&&outer_max>=2&&outer_max>=outer_min,
             "invalid native solver iteration budget");
 TORCH_CHECK(linear_rtol>0&&linear_atol>=0&&outer_relax>0&&outer_relax<=1&&outer_phi_tol>=0&&outer_current_tol>=0,
             "invalid native solver tolerance/relaxation");
 TORCH_CHECK(balance_rel>=0&&balance_abs>=0&&(balance_rel>0||balance_abs>0),"invalid native current-balance tolerance");
 double end=get(cfg,"end",2),eval=get(cfg,"evaluation",end),snapshot=get(cfg,"snapshot",.02),electrical_interval=get(cfg,"electrical_interval",.0025);
 double evaluation_scale=std::max(std::abs(end),std::abs(eval));
 double evaluation_end_tol=64*std::max(std::numeric_limits<double>::denorm_min(),
                                       std::numeric_limits<double>::epsilon()*evaluation_scale);
 TORCH_CHECK(std::isfinite(end)&&std::isfinite(eval)&&std::isfinite(snapshot)&&std::isfinite(electrical_interval)&&
             end>0&&eval>0&&eval<=end&&snapshot>0&&electrical_interval>0,"invalid native time horizon/interval");
 double electrical_step_ratio=electrical_interval/p.dt;
 double electrical_step_nearest=std::round(electrical_step_ratio);
 double electrical_step_tol=64*std::numeric_limits<double>::epsilon()*std::max(1.,std::abs(electrical_step_ratio));
 TORCH_CHECK(std::isfinite(electrical_step_ratio)&&electrical_step_ratio>=1&&
             electrical_step_ratio<=std::numeric_limits<int>::max()&&
             std::abs(electrical_step_ratio-electrical_step_nearest)<=electrical_step_tol,
             "native electricalUpdateInterval_s must be an integer multiple of timeStep_s and at least one step");
 double evaluation_ratio=eval/p.dt,evaluation_nearest=std::round(evaluation_ratio);
 double evaluation_tol=64*std::numeric_limits<double>::epsilon()*std::max(1.,std::abs(evaluation_ratio));
 bool evaluation_is_end=std::abs(eval-end)<=evaluation_end_tol;
 TORCH_CHECK(std::isfinite(evaluation_ratio)&&(evaluation_is_end||std::abs(evaluation_ratio-evaluation_nearest)<=evaluation_tol),
             "native evaluationTime_s must align with the fixed timeStep_s grid");
 double onset_T=get(cfg,"onset_T",523.15),onset_X=get(cfg,"onset_X",.01),onset_A=get(cfg,"onset_area",.01);
 TORCH_CHECK(onset_T>0&&onset_X>0&&onset_X<=1&&onset_A>0&&onset_A<=1,"invalid native onset criterion");
 constexpr int NH=24;
 double max_steps_raw=get(cfg,"max_steps",100000);
 double max_history_bytes=get(cfg,"max_history_bytes",16.0*1024.0*1024.0*1024.0);
 TORCH_CHECK(std::isfinite(max_steps_raw)&&max_steps_raw>=1&&
             max_steps_raw<=std::numeric_limits<int>::max()&&std::floor(max_steps_raw)==max_steps_raw,
             "native maximum time-step budget is invalid");
 TORCH_CHECK(std::isfinite(max_history_bytes)&&max_history_bytes>=NH*sizeof(double),
             "native maximum scalar-history allocation budget is invalid");
 double ratio=end/p.dt,nearest=std::round(ratio);
 double ratio_tol=64*std::numeric_limits<double>::epsilon()*std::max(1.,std::abs(ratio));
 TORCH_CHECK(std::isfinite(ratio)&&ratio>0&&ratio<=std::numeric_limits<int>::max(),"native time-step count is out of range");
 int64_t step_count=std::abs(ratio-nearest)<=ratio_tol
     ?std::max<int64_t>(1,static_cast<int64_t>(nearest))
     :static_cast<int64_t>(std::floor(ratio))+1;
 TORCH_CHECK(step_count>=1&&step_count<=std::numeric_limits<int>::max()&&step_count<=static_cast<int64_t>(max_steps_raw),
             "native time-step count exceeds the configured resource budget");
 int steps=static_cast<int>(step_count);
 long double history_bytes=static_cast<long double>(NH)*static_cast<long double>(steps)*
                           static_cast<long double>(g.b)*sizeof(double);
 double stop_value=get(cfg,"stop_on_onset",0),save_value=get(cfg,"save_handoff",0);
 TORCH_CHECK((stop_value==0||stop_value==1)&&(save_value==0||save_value==1),"native stop/save flags must be boolean");
 bool stop=stop_value!=0,save=save_value!=0;
 TORCH_CHECK(!(stop&&save),"onset-only trial cannot supply full-horizon handoff");
 long double retained_history_bytes=history_bytes;
 if(save){
  double snapshot_period=std::max(snapshot,p.dt);
  int64_t scheduled_upper=static_cast<int64_t>(std::ceil(end/snapshot_period));
  int64_t snapshot_count_upper=std::min<int64_t>(step_count,scheduled_upper+2);
  long double snapshot_values=static_cast<long double>(4)*snapshot_count_upper*
      static_cast<long double>(g.b)*static_cast<long double>(g.n)*
      static_cast<long double>(g.n);
  // Four full-grid fields are retained and then stacked for the return map.
  retained_history_bytes+=2*snapshot_values*sizeof(double);
 }
 TORCH_CHECK(retained_history_bytes<=static_cast<long double>(max_history_bytes),
             "native retained-history allocation exceeds the configured byte budget");
 auto s=at::zeros({NS,g.b,g.n,g.n},opts);s[0].fill_(p.initial_T);
 auto prop64=g.prop.to(at::kDouble);
 s[1].copy_(prop64*p.cp0);s[2].copy_(prop64*p.cm0);s[3].copy_(prop64*p.water0);
 // Contacts overlay propellant; they are not fixed-potential material cells.
 s[7].copy_(at::ones_like(s[0])*b3(.5*(voltage+p.cathode)));
 auto running=at::ones({g.b},anode.options()),ign=at::full({g.b},NAN,opts);
 auto stepsdone=at::zeros({g.b},anode.options().dtype(at::kLong));
 auto tr=properties(s,g.prop,table,p);
 auto initial_invalid_thermal_property=invalid_constitutive_cells(tr,g.prop,p);
 TORCH_CHECK(!initial_invalid_thermal_property.any().item<bool>(),
             "BC native thermal properties became nonfinite/nonphysical or transport constitutive properties became invalid; heat capacity must be positive and conductivity/diffusivity fields must be finite and nonnegative within configured conductivity bounds");
 auto e=electrical(s,tr,voltage,running,p,g,cfg);
 auto onset_state=s.clone(),onset_heat=e.heat.clone();
 auto congestion=quantile_masked(e.heat[4],g.prop,.99)/(sum2(at::where(g.prop,e.heat[4],0))/g.count).clamp_min(get(cfg,"j_floor",1e-12));
 auto h=at::zeros({NH,steps,g.b},opts);auto energy=at::zeros({g.b},opts),eqenergy=at::zeros_like(energy),ignenergy=at::full_like(energy,NAN);
 auto maxcaps=at::zeros({5,g.b},opts);auto qtime=std::vector<double>{};std::vector<Tensor> qjs,qes,Ts,Xs;
 auto lastrel=e.linear_rel.clone(),lastiters=e.linear_iterations.clone(),lastok=e.converged.clone(),lastmismatch=e.mismatch.clone();
 double nextsnapshot=0;int solveevery=static_cast<int>(electrical_step_nearest),executed=0;
 auto onsetT=onset_T,onsetX=onset_X,onsetA=onset_A;
 double initial_lp_mass=p.m_lp*p.lp0,initial_pva_mass=p.m_pva*p.pva0;
 double initialmass=initial_lp_mass+initial_pva_mass,xi=mn(p.lp0/1.45,p.pva0);
 TORCH_CHECK(std::isfinite(initial_lp_mass)&&initial_lp_mass>0&&
             std::isfinite(initial_pva_mass)&&initial_pva_mass>0&&
             std::isfinite(initialmass)&&initialmass>0&&std::isfinite(xi)&&xi>0,
             "invalid global reaction inventory: initial mass and extent must be finite positive FP64 values");
 auto pending_invalid_thermal_property=at::zeros({g.b},anode.options());
 auto pending_rejected_echem=at::zeros_like(pending_invalid_thermal_property);
 auto pending_invalid_thermal_cfl=at::zeros_like(pending_invalid_thermal_property);
 auto pending_invalid_integrated_state=at::zeros_like(pending_invalid_thermal_property);
 auto pending_invalid_updated_thermal_property=at::zeros_like(pending_invalid_thermal_property);
 auto worst_thermal_cfl=at::full({g.b},-INFINITY,opts);
 double thermal_cfl_tolerance=64*std::numeric_limits<double>::epsilon()*std::max(1.,p.thermal_cfl_safety);
 auto time_tolerance=[](double a,double b){
  double scale=std::max(std::abs(a),std::abs(b));
  return 64*std::max(std::numeric_limits<double>::denorm_min(),
                     std::numeric_limits<double>::epsilon()*scale);
 };
 for(int step=0;step<steps;step++){
  double step_start=step*p.dt;
  double step_dt=step==steps-1?end-step_start:p.dt;
  double t=step==steps-1?end:step_start+step_dt;
  TORCH_CHECK(std::isfinite(step_dt)&&step_dt>0,"native timestep invariant violated: nonpositive/nonfinite step_dt");
  Params step_params=p;step_params.dt=step_dt;auto proposed_tr=properties(s,g.prop,table,p);
  auto active_prop=g.prop&b3(running);
  auto invalid_thermal_property=invalid_constitutive_cells(
      proposed_tr,active_prop,p);
  auto invalid_thermal_property_lane=
      invalid_thermal_property.reshape({g.b,-1}).any(1);
  pending_invalid_thermal_property=pending_invalid_thermal_property|
      invalid_thermal_property_lane;
  auto step_running=running&~pending_invalid_thermal_property&
      ~pending_rejected_echem&~pending_invalid_thermal_cfl&
      ~pending_invalid_integrated_state&
      ~pending_invalid_updated_thermal_property;
  // Keep a known-finite property bundle for quarantined lanes.  The linear
  // system is assembled as a dense batch even though PCG masks those lanes.
  tr=at::where(pending_invalid_thermal_property.reshape({1,-1,1,1}),tr,proposed_tr);
  if(step>0&&step%solveevery==0){
   auto next=electrical(s,tr,voltage,step_running,p,g,cfg);
   auto next_congestion=quantile_masked(next.heat[4],g.prop,.99)/(sum2(at::where(g.prop,next.heat[4],0))/g.count).clamp_min(get(cfg,"j_floor",1e-12));
   auto field_mask=step_running.reshape({1,-1,1,1});
   e.heat=at::where(field_mask,next.heat,e.heat);e.current=at::where(step_running,next.current,e.current);
   e.mismatch=at::where(step_running,next.mismatch,e.mismatch);e.linear_rel=at::where(step_running,next.linear_rel,e.linear_rel);
   e.linear_iterations=at::where(step_running,next.linear_iterations,e.linear_iterations);
   e.converged=at::where(step_running,next.converged,e.converged);e.outer_iterations=at::where(step_running,next.outer_iterations,e.outer_iterations);
   congestion=at::where(step_running,next_congestion,congestion);
   lastrel=at::where(step_running,e.linear_rel,lastrel);lastiters=at::where(step_running,e.linear_iterations,lastiters);lastok=at::where(step_running,e.converged,lastok);lastmismatch=at::where(step_running,e.mismatch,lastmismatch);
  }
  auto updated=advance(s,g.prop,step_running,tr,e.heat,table,step_params);auto diag=updated[1];
  // A clipped Faradaic sink would make the reported BV current, terminal
  // power and Joule field inconsistent with accepted species/enthalpy.  Until
  // a fully coupled current rebalance is implemented, reject that timestep
  // before committing any candidate state or accumulated energy.
  auto step_active_prop=g.prop&b3(step_running);
  auto rejected_echem=(diag[6]>0)&step_active_prop;
  auto rejected_echem_lane=rejected_echem.reshape({g.b,-1}).any(1);
  pending_rejected_echem=pending_rejected_echem|rejected_echem_lane;
  auto step_diffusive_cfl=max2(at::where(step_active_prop,diag[7],0));
  auto step_thermal_cfl=max2(at::where(step_active_prop,diag[8],0));
  auto invalid_thermal_cfl=~at::isfinite(step_thermal_cfl)|(step_thermal_cfl<0)|
      (step_thermal_cfl>p.thermal_cfl_safety+thermal_cfl_tolerance);
  pending_invalid_thermal_cfl=pending_invalid_thermal_cfl|invalid_thermal_cfl;
  worst_thermal_cfl=at::maximum(
      worst_thermal_cfl,
      at::where(at::isfinite(step_thermal_cfl),step_thermal_cfl,
                at::full_like(step_thermal_cfl,INFINITY)));
  auto invalid_integrated_state_lane=at::logical_not(at::isfinite(updated[0]))
      .permute({1,0,2,3}).reshape({g.b,-1}).any(1)&step_running;
  pending_invalid_integrated_state=pending_invalid_integrated_state|
      invalid_integrated_state_lane;
  // A finite temperature update can still leave the constitutive domain
  // (for example, an overflowing Arrhenius cp/k law).  Validate the candidate
  // state before it is committed, including on the final configured step.
  auto updated_tr=properties(updated[0],g.prop,table,p);
  auto invalid_updated_thermal_property=invalid_constitutive_cells(
      updated_tr,step_active_prop,p);
  auto invalid_updated_thermal_property_lane=
      invalid_updated_thermal_property.reshape({g.b,-1}).any(1);
  pending_invalid_updated_thermal_property=
      pending_invalid_updated_thermal_property|
      invalid_updated_thermal_property_lane;
  auto commit_running=step_running&~rejected_echem_lane&~invalid_thermal_cfl&
      ~invalid_integrated_state_lane&~invalid_updated_thermal_property_lane;
  s=at::where(commit_running.reshape({1,-1,1,1}),updated[0],s);
  e.heat[1].copy_(at::where(b3(commit_running),diag[5],e.heat[1]));
  stepsdone=at::where(commit_running,at::full_like(stepsdone,step+1),stepsdone);
  auto q=(s[0]>=onsetT)&(s[6]>=onsetX)&g.prop;auto area=sum2(q.to(at::kDouble))/g.count;
  auto tfrac=sum2(((s[0]>=onsetT)&g.prop).to(at::kDouble))/g.count;
  auto newly=at::isnan(ign)&(area>=onsetA)&commit_running;ign=at::where(newly,at::full_like(ign,t),ign);
  onset_state=at::where(newly.reshape({1,-1,1,1}),s,onset_state);onset_heat=at::where(newly.reshape({1,-1,1,1}),e.heat,onset_heat);
  auto power=e.current*(voltage-p.cathode);energy=energy+at::where(commit_running,power*step_dt,0);ignenergy=at::where(newly,energy,ignenergy);
  auto eqrate=sum2(at::where(g.prop,e.heat[0]+e.heat[1],0))*(p.dx*p.dx*p.layer);eqenergy+=at::where(commit_running,eqrate*step_dt,0);
  auto progress=sum2(at::where(g.prop,s[6],0))/g.count;
  auto lp=.5*(s[1]+s[2]),pva=(p.pva0-xi*s[6]).clamp_min(0);
  auto normalized_reactive_mass=(initial_lp_mass/initialmass)*(lp/p.lp0)+
      (initial_pva_mass/initialmass)*(pva/p.pva0);
  auto remain=sum2(at::where(g.prop,normalized_reactive_mass,0))/g.count;
  double remaining_mass_tolerance=128*std::numeric_limits<double>::epsilon();
  auto invalid_remaining_mass=~at::isfinite(remain)|(remain < -remaining_mass_tolerance)|
      (remain > 1+remaining_mass_tolerance);
  pending_invalid_integrated_state=pending_invalid_integrated_state|
      (invalid_remaining_mass&commit_running);
  remain=remain.clamp(0,1);
  std::vector<Tensor> hs{max2(at::where(g.prop,s[0],-INFINITY)),sum2(at::where(g.prop,s[0],0))/g.count,e.current,power,energy,progress,
   max2(at::where(g.prop,s[6],-INFINITY)),remain,area,tfrac,max2(diag[0]),max2(diag[1]),congestion,
   sum2(diag[2])/g.count,sum2(diag[3])/g.count,sum2(diag[4])/g.count,eqrate,eqenergy,
   sum2(at::where(g.prop,e.heat[0],0))/g.count,sum2(at::where(g.prop,e.heat[1],0))/g.count,
   p.density*(p.kin[0].heat*sum2(diag[0])+p.kin[1].heat*sum2(diag[1]))/g.count,e.mismatch,
   step_diffusive_cfl,step_thermal_cfl};
  // Hold state/cumulative histories after onset, but zero instantaneous
  // current/rate/source/diagnostic channels: there is no post-onset solve.
  const bool instantaneous[NH]={false,false,true,true,false,false,false,false,false,false,
      true,true,true,true,true,true,true,false,true,true,true,true,true,true};
  for(int j=0;j<NH;j++){
   if(instantaneous[j])hs[j]=at::where(commit_running,hs[j],at::zeros_like(hs[j]));
   else if(step>0)hs[j]=at::where(commit_running,hs[j],h.index({j,step-1,Slice()}));
  }
  for(int j=0;j<NH;j++){h.index_put_({j,step,Slice()},hs[j]);}
  maxcaps[0].copy_(at::maximum(maxcaps[0],hs[14]));maxcaps[1].copy_(at::maximum(maxcaps[1],hs[13]));maxcaps[2].copy_(at::maximum(maxcaps[2],hs[15]));
  maxcaps[3].copy_(at::maximum(maxcaps[3],hs[23]));
  maxcaps[4].copy_(at::maximum(maxcaps[4],hs[22]));
  if(save&&(t>=nextsnapshot-time_tolerance(t,nextsnapshot)||step==steps-1)){
   qtime.push_back(t);auto source_mask=b3(commit_running);
   qjs.push_back(at::where(source_mask,e.heat[0],0));qes.push_back(at::where(source_mask,e.heat[1],0));
   Ts.push_back(s[0].clone());Xs.push_back(s[6].clone());nextsnapshot+=std::max(snapshot,step_dt);
  }
  // Strict pre-flame semantics: every lane freezes at its first onset.  The
  // stop flag only controls whether an all-finished Vmin wave returns early.
  running=commit_running&~newly;
  executed=step+1;
  bool host_check_due=(step+1)%time_check==0||step==steps-1;
  if(host_check_due){
   auto host=check_time_loop_failures(
       pending_invalid_thermal_property,pending_rejected_echem,
       pending_invalid_thermal_cfl,pending_invalid_integrated_state,
       pending_invalid_updated_thermal_property,
       running,stepsdone,ign,s,worst_thermal_cfl,
       p.thermal_cfl_safety);
   if(!host.any_running){
    if(stop){executed=static_cast<int>(host.maximum_steps_completed);break;}
    // Full reference runs retain a fixed-size history, filled with the last
    // valid pre-flame row.  Remove deferred-check-only snapshots after the
    // last onset; onset_state/onset_heat carry the exact handoff fields.
    double last_onset_time=host.latest_onset_time;
    if(save){
     double onset_tol=time_tolerance(last_onset_time,last_onset_time);
     while(!qtime.empty()&&qtime.back()>last_onset_time+onset_tol){
      qtime.pop_back();qjs.pop_back();qes.pop_back();Ts.pop_back();Xs.pop_back();
     }
     auto last_onset_lane=at::isfinite(ign)&((ign-last_onset_time).abs()<=onset_tol);
     auto last_source_mask=b3(last_onset_lane);
     auto last_qj=at::where(last_source_mask,onset_heat[0],0);
     auto last_qe=at::where(last_source_mask,onset_heat[1],0);
     if(qtime.empty()||std::abs(qtime.back()-last_onset_time)>onset_tol){
      qtime.push_back(last_onset_time);qjs.push_back(last_qj);qes.push_back(last_qe);
      Ts.push_back(onset_state[0].clone());Xs.push_back(onset_state[6].clone());
     }else{
      qtime.back()=last_onset_time;qjs.back()=last_qj;qes.back()=last_qe;
      Ts.back()=onset_state[0].clone();Xs.back()=onset_state[6].clone();
     }
    }
    if(step+1<steps){
     auto tail=h.index({Slice(),step,Slice()}).clone();
     for(int j=0;j<NH;j++){
      if(instantaneous[j])tail[j].zero_();
     }
     h.index_put_({Slice(),Slice(step+1,steps),Slice()},tail.unsqueeze(1).expand({NH,steps-step-1,g.b}));
    }
    executed=steps;break;
   }
  }
 }
 // Stop-only trial history is explicitly partial. The Python adapter does not
 // label its partial final values as 2-second objectives.
 std::map<std::string,Tensor> out{{"state",s},{"histories",h.index({Slice(),Slice(0,executed),Slice()}).contiguous()},
 {"ignitionDelay",ign},{"onset_state",onset_state},{"onset_heat",onset_heat},{"steps_completed",stepsdone},
 {"linear_converged",lastok},{"linear_residual",lastrel},{"linear_iterations",lastiters},{"current_mismatch",lastmismatch},
 {"caps",maxcaps},{"ignition_energy",at::where(at::isfinite(ign),ignenergy,energy)}};
 if(save){out["snapshot_times"]=at::tensor(qtime,opts);out["snapshot_qj"]=at::stack(qjs);out["snapshot_qe"]=at::stack(qes);out["snapshot_T"]=at::stack(Ts);out["snapshot_X"]=at::stack(Xs);}
 return out;
}
}
