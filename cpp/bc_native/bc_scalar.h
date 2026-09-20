#pragma once
// Shared IEEE-754 FP64 arithmetic for the CPU reference and CUDA kernels.
// Surface electrodes are contact labels over a full propellant domain.  The
// contact-normal drop and the depth-averaged thin-layer source use independent
// physical lengths.  No new chemistry is introduced here.
#include <cmath>
#include <cstdint>
#include <cfloat>
#ifdef __CUDACC__
#define BC_HD __host__ __device__ inline
#else
#define BC_HD inline
#endif
namespace bc_native {
constexpr int NS=10, NP=8, ND=9, NCONTACT=9;
// state: T, cation/mobile-LP+, anion/mobile-LP-, mobile water, alpha1,
// alpha2, chemical progress, potential, generated product water,
// cumulative electrochemical LP consumption
// properties: D+, D-, Dw, sigma_ion, sigma_e, sigma_total, cp, k
// diagnostics: accepted channel1 rate, accepted channel2 rate, species
// limiter, T cap, rate cap, accepted electrochemical heat,
// rejected electrochemical inventory (fail-closed signal), diffusion-only
// thermal CFL, total explicit thermal stability CFL including surface loss
struct Property { int mode=0,offset=0,size=0; double value=0,tref=298.15,ea=0; };
struct Kinetic { int offset=0,size=0; double heat=0; };
struct Channel {double eq=0,j0=0,alpha=.5,n=1,reverse=1,heat=0,stoich=0;};
struct Params {
    int n=0,batch=0,full_bv=1,full_np=1,diffusion=1,augmented=0,joule_mode=0;
    int local_min=8,local_max=60,newton=2;
    double R=8.314462618,F=96485.33212,zp=1,zm=-1,dx=1,dt=.00025;
    double sigma_min=1e-4,sigma_max=.5,initial_T=298.15,ambient=298.15;
    double cp0=0,cm0=0,water0=0,lp0=0,pva0=0,m_lp=0,m_pva=0,density=1000;
    double w1=2./3,w2=1./3,rate_max=1000,layer=.001;
    double min_T=200,max_T=3000,convection=0,emissivity=0,sb=5.670374419e-8;
    double min_fraction=1e-6,max_multiple=2,max_delta=.05,lim_rtol=1e-7,conc_floor=1e-12;
    double cathode=0,exp_limit=45,local_abs=.01,local_rel=1e-4,roundoff=2;
    double conductivity_floor=1e-12,slope_floor=1e-12,contact_normal=3e-5;
    double thermal_cfl_safety=1;
    Property prop[6]; // D+, D-, Dw, electronic sigma, cp, k
    Kinetic kin[2];
    Channel ch[4]; // water anode, water cathode, LP anode, LP cathode
};
static_assert(sizeof(Params)<4096,"CUDA kernel parameter block exceeds 4 KiB");
BC_HD bool finite_fp64(double value){
#if defined(__CUDA_ARCH__)
    // CUDA device code provides the global intrinsic; std::isfinite is not
    // consistently device-callable across supported nvcc/libstdc++ pairs.
    return ::isfinite(value);
#else
    // Standard C++ guarantees std::isfinite, while a global ::isfinite is not
    // portable across host standard libraries (notably strict libc++ builds).
    return std::isfinite(value);
#endif
}
BC_HD double exp_fp64(double value){
#if defined(__CUDA_ARCH__)
    return ::exp(value);
#else
    return std::exp(value);
#endif
}
BC_HD double abs_fp64(double value){
#if defined(__CUDA_ARCH__)
    return ::fabs(value);
#else
    return std::fabs(value);
#endif
}
BC_HD double pow_fp64(double base,double exponent){
#if defined(__CUDA_ARCH__)
    return ::pow(base,exponent);
#else
    return std::pow(base,exponent);
#endif
}
BC_HD double sqrt_fp64(double value){
#if defined(__CUDA_ARCH__)
    return ::sqrt(value);
#else
    return std::sqrt(value);
#endif
}
BC_HD double mn(double a,double b){return (a!=a)?a:((b!=b)?b:(a<b?a:b));}
BC_HD double mx(double a,double b){return (a!=a)?a:((b!=b)?b:(a>b?a:b));}
BC_HD double clamp(double v,double a,double b){return mn(mx(v,a),b);}
BC_HD double harmonic(double a,double b){
    if(!finite_fp64(a)||!finite_fp64(b)||a<0||b<0)return NAN;
    double lo=mn(a,b),hi=mx(a,b);
    if(lo==0)return 0;
    // Algebraically 2ab/(a+b), arranged to avoid product/sum overflow and
    // to preserve the small operand when lo/hi underflows.
    return lo*(2/(1+lo/hi));
}
BC_HD double lookup(double x,const double* table,int offset,int n,int yrow=1){
    const double* xs=table+offset; const double* ys=xs+yrow*n;
    x=clamp(x,xs[0],xs[n-1]); int lo=0,hi=n-1;
    while(hi-lo>1){int mid=(hi+lo)/2;if(x>xs[mid])lo=mid;else hi=mid;}
    return ys[lo]+(x-xs[lo])/(xs[hi]-xs[lo])*(ys[hi]-ys[lo]);
}
BC_HD double property(double T,Property p,const double* table,double R){
    if(p.mode==1)return lookup(T,table,p.offset,p.size);
    if(p.mode==2)return p.value*exp_fp64(-p.ea/R*(1/mx(T,1.)-1/p.tref));
    return p.value;
}
BC_HD double kinetic(double a,double T,Kinetic k,const double* table,double R){
    if(a>=1)return 0;
    double ea=lookup(a,table,k.offset,k.size,1),g=lookup(a,table,k.offset,k.size,2);
    return exp_fp64(clamp(g-ea/(R*mx(T,1.)),-100,60));
}
BC_HD void props_cell(int64_t i,int64_t total,const double* s,const bool* prop,
                      const double* table,double* out,const Params& p){
    double T=mx(s[i],1.);
    double dp=property(T,p.prop[0],table,p.R),dm=property(T,p.prop[1],table,p.R),dw=property(T,p.prop[2],table,p.R);
    double si=p.F*p.F/(p.R*T)*(p.zp*p.zp*dp*mx(s[total+i],0.)+p.zm*p.zm*dm*mx(s[2*total+i],0.));
    double se=p.augmented?property(T,p.prop[3],table,p.R):0;
    out[i]=prop[i]?dp:0;out[total+i]=prop[i]?dm:0;out[2*total+i]=prop[i]?dw:0;
    out[3*total+i]=prop[i]?si:p.sigma_max;out[4*total+i]=prop[i]?se:p.sigma_max;
    out[5*total+i]=prop[i]?clamp(si+se,p.sigma_min,p.sigma_max):p.sigma_max;
    out[6*total+i]=property(T,p.prop[4],table,p.R);out[7*total+i]=property(T,p.prop[5],table,p.R);
}
BC_HD double limit_update(double old,double rate,double ref,const Params& p,bool& limited){
    // A finite clipped value must never hide overflow in the proposed
    // transport update.  In particular, clamp(+/-Inf) is finite and an
    // Inf-vs-Inf tolerance comparison is false.  Return NaN so the engine's
    // integrated-state guard quarantines the lane before commit.
    limited=true;
    if(!finite_fp64(old)||!finite_fp64(rate)||!finite_fp64(ref)||
       !finite_fp64(p.dt)||!finite_fp64(p.max_delta)||
       !finite_fp64(p.min_fraction)||!finite_fp64(p.max_multiple)||
       !finite_fp64(p.conc_floor)||!finite_fp64(p.lim_rtol))return NAN;
    double lower=ref*p.min_fraction,upper=ref*p.max_multiple;
    if(!finite_fp64(lower)||!finite_fp64(upper))return NAN;
    double raw=p.dt*rate,base=mx(old,lower);
    if(!finite_fp64(raw)||!finite_fp64(base))return NAN;
    double lim=p.max_delta*base;
    if(!finite_fp64(lim))return NAN;
    double d=clamp(raw,-lim,lim);
    if(!finite_fp64(d))return NAN;
    double scale=mx(mx(abs_fp64(raw),abs_fp64(d)),mx(abs_fp64(old),lower));
    if(!finite_fp64(scale))return NAN;
    double delta_error=abs_fp64(d-raw);
    double delta_threshold=p.conc_floor+p.lim_rtol*scale;
    if(!finite_fp64(delta_error)||!finite_fp64(delta_threshold))return NAN;
    bool a=delta_error>delta_threshold;
    double ru=old+d;
    if(!finite_fp64(ru))return NAN;
    double nu=clamp(ru,lower,upper);
    if(!finite_fp64(nu))return NAN;
    double bound_scale=mx(mx(abs_fp64(nu),abs_fp64(ru)),lower);
    if(!finite_fp64(bound_scale))return NAN;
    double bound_error=abs_fp64(nu-ru);
    double bound_threshold=p.conc_floor+p.lim_rtol*bound_scale;
    if(!finite_fp64(bound_error)||!finite_fp64(bound_threshold))return NAN;
    bool b=bound_error>bound_threshold;
    limited=a||b;
    return nu;
}
BC_HD void advance_cell(int64_t i,int64_t total,const double* s,const bool* prop,
    const bool* active,const double* tr,const double* heat,const double* table,
    double* out,double* diag,const Params& p){
    int64_t cells=int64_t(p.n)*p.n,b=i/cells,j=i%cells;int row=int(j/p.n),col=int(j%p.n);
    if(!active[b]){
        for(int f=0;f<NS;f++){out[f*total+i]=s[f*total+i];}
        for(int f=0;f<ND;f++){diag[f*total+i]=0;}
        return;
    }
    if(!prop[i]){
        for(int f=0;f<NS;f++){out[f*total+i]=0;}
        out[i]=p.ambient;
        out[7*total+i]=s[7*total+i];
        for(int f=0;f<ND;f++){diag[f*total+i]=0;}
        return;
    }
    double T=s[i],a1=s[4*total+i],a2=s[5*total+i],oldx=s[6*total+i];
    double r1=kinetic(a1,T,p.kin[0],table,p.R),r2=kinetic(a2,T,p.kin[1],table,p.R);
    bool capped=r1>p.rate_max||r2>p.rate_max;r1=clamp(r1,0,p.rate_max);r2=clamp(r2,0,p.rate_max);
    double proposed_a1=clamp(a1+p.dt*r1,0,1),proposed_a2=clamp(a2+p.dt*r2,0,1);
    double proposed_da1=mx(proposed_a1-a1,0),proposed_da2=mx(proposed_a2-a2,0);
    double dc=0,da=0,dw=0,conduct=0,thermal_face_sum=0;
    int64_t nb[4]={i+1,i-1,i+p.n,i-p.n};bool valid[4]={col<p.n-1,col>0,row<p.n-1,row>0};
    for(int d=0;d<4;d++){
        int64_t q=nb[d];if(!valid[d]||!prop[q])continue;
        double kface=harmonic(tr[7*total+i],tr[7*total+q]);
        thermal_face_sum+=kface;
        conduct+=kface*(s[q]-T)/(p.dx*p.dx);
        if(p.full_np){
            double tf=mx(.5*(T+s[q]),1.),dv=s[7*total+q]-s[7*total+i];
            double hp=harmonic(tr[i],tr[q]),hm=harmonic(tr[total+i],tr[total+q]),hw=harmonic(tr[2*total+i],tr[2*total+q]);
            dc+=hp*((s[total+q]-s[total+i])+p.zp*p.F/(p.R*tf)*.5*(s[total+i]+s[total+q])*dv)/(p.dx*p.dx);
            da+=hm*((s[2*total+q]-s[2*total+i])+p.zm*p.F/(p.R*tf)*.5*(s[2*total+i]+s[2*total+q])*dv)/(p.dx*p.dx);
            dw+=hw*(s[3*total+q]-s[3*total+i])/(p.dx*p.dx);
        }
    }
    // Transport is limited first.  Faradaic and chemical sinks are accepted
    // explicitly against the remaining inventory so species, progress and
    // heat all use the same physically accepted chemical increment.
    bool lc,la,lw;
    double nc=limit_update(s[total+i],dc,p.cp0,p,lc),na=limit_update(s[2*total+i],da,p.cm0,p,la);
    double transported_water=limit_update(s[3*total+i],dw,p.water0,p,lw);
    double neutral=.5*(nc+na);
    double lp_floor=mx(p.conc_floor,p.min_fraction*mx(p.cp0,p.cm0));
    double water_floor=mx(p.conc_floor,p.min_fraction*p.water0);

    double requested_ec_lp=mx(heat[2*total+i]*p.dt,0);
    double accepted_ec_lp=mn(requested_ec_lp,mx(neutral-lp_floor,0));
    // A concentration floor is a state bound, not permission to accept a
    // sub-floor Faradaic extent without matching inventory/enthalpy.
    bool ec_lp_limited=requested_ec_lp>accepted_ec_lp;
    neutral-=accepted_ec_lp;

    double requested_ec_water=mx(heat[3*total+i]*p.dt,0);
    double accepted_ec_water=mn(requested_ec_water,mx(transported_water-water_floor,0));
    bool ec_water_limited=requested_ec_water>accepted_ec_water;
    double mobile_water=transported_water-accepted_ec_water;
    double lp_acceptance=requested_ec_lp>0?accepted_ec_lp/requested_ec_lp:1;
    double water_acceptance=requested_ec_water>0?accepted_ec_water/requested_ec_water:1;
    // heat[5] and heat[6] retain the water/LP channel split.  Rejected
    // Faradaic extent cannot release the enthalpy of an unavailable reactant.
    double accepted_echem_heat=heat[5*total+i]*water_acceptance+heat[6*total+i]*lp_acceptance;

    double xi_max=mn(p.lp0/1.45,p.pva0);
    double proposed_dx=p.w1*proposed_da1+p.w2*proposed_da2;
    double proposed_dxi=xi_max*proposed_dx;
    double pva_remaining=mx(p.pva0-xi_max*oldx,0);
    double maximum_dxi=mn(mx(neutral-lp_floor,0)/1.45,pva_remaining);
    // Scale every positive proposal, including values below conc_floor.  A
    // depleted cell must never create progress/product/heat merely because a
    // proposed extent is small.
    double xi_scale=proposed_dxi>0?clamp(maximum_dxi/proposed_dxi,0,1):1;
    double accepted_da1=proposed_da1*xi_scale,accepted_da2=proposed_da2*xi_scale;
    double na1=a1+accepted_da1,na2=a2+accepted_da2;
    double nx=clamp(p.w1*na1+p.w2*na2,0,1);
    double progress_increment=mx(nx-oldx,0),accepted_dxi=xi_max*progress_increment;
    neutral-=1.45*accepted_dxi;

    out[total+i]=out[2*total+i]=neutral;
    out[3*total+i]=mobile_water;
    out[4*total+i]=na1;out[5*total+i]=na2;out[6*total+i]=nx;
    out[8*total+i]=s[8*total+i]+2*accepted_dxi;
    out[9*total+i]=s[9*total+i]+accepted_ec_lp;
    double accepted_r1=accepted_da1/p.dt,accepted_r2=accepted_da2/p.dt;
    double chem_energy=p.density*(p.kin[0].heat*accepted_da1+p.kin[1].heat*accepted_da2);
    double loss=p.convection/p.layer*(T-p.ambient)+p.emissivity*p.sb/p.layer*(pow_fp64(T,4)-pow_fp64(p.ambient,4));
    double nonchemical_energy=p.dt*(conduct+heat[i]+accepted_echem_heat-loss);
    double thermal_capacity=p.density*tr[6*total+i];
    double rt=T+(nonchemical_energy+chem_energy)/thermal_capacity;
    bool invalid_temperature=!finite_fp64(rt);
    out[i]=invalid_temperature?NAN:clamp(rt,p.min_T,p.max_T);
    out[7*total+i]=s[7*total+i];
    diag[i]=accepted_r1;diag[total+i]=accepted_r2;
    // Chemical-inventory scaling is a physical closure, not a numerical
    // species limiter.  It therefore does not invalidate an otherwise
    // conservative trial.
    diag[2*total+i]=(lc||la||lw||ec_lp_limited||ec_water_limited)?1:0;
    diag[3*total+i]=(invalid_temperature||rt<p.min_T||rt>p.max_T)?1:0;
    diag[4*total+i]=capped?1:0;
    diag[5*total+i]=accepted_echem_heat;
    diag[6*total+i]=(ec_lp_limited||ec_water_limited)?1:0;
    double diffusive_cfl=p.dt*thermal_face_sum/(thermal_capacity*p.dx*p.dx);
    double loss_jacobian=p.convection/p.layer
        +4*p.emissivity*p.sb/p.layer*pow_fp64(mx(T,p.ambient),3);
    diag[7*total+i]=diffusive_cfl;
    diag[8*total+i]=diffusive_cfl+p.dt*loss_jacobian/thermal_capacity;
}
struct BV {double j,d;};
BC_HD BV bv(double eta,double cf,Channel c,double T,const Params& p){
    eta=mx(eta,0);double kf=c.alpha*c.n*p.F/(p.R*T),kr=(1-c.alpha)*c.n*p.F/(p.R*T);
    double rf=kf*eta,rr=-kr*eta,ef=exp_fp64(clamp(rf,-p.exp_limit,p.exp_limit)),er=exp_fp64(clamp(rr,-p.exp_limit,p.exp_limit));
    double j=c.j0*(cf*ef-(p.full_bv?c.reverse*er:0));
    double d=c.j0*(cf*ef*kf*((rf>-p.exp_limit&&rf<p.exp_limit&&eta>0)?1:0)
       +(p.full_bv?c.reverse*er*kr*((rr>-p.exp_limit&&rr<p.exp_limit&&eta>0)?1:0):0));
    if(j<=0){j=0;d=0;}
    return {j,d};
}
struct ContactEval {BV w,l;double residual,threshold,slope,ew,el;};
BC_HD ContactEval eval_contact(double gap,double drop,double G,double T,double wf,double sf,bool an,const Params& p){
    Channel w=p.ch[an?0:1],l=p.ch[an?2:3];double ew=mx(gap-w.eq,0),el=mx(gap-l.eq,0);
    BV bw=bv(ew,wf,w,T,p),bl=bv(el,sf,l,T,p);
    double j=bw.j+bl.j,jb=G*mx(drop-gap,0),slope=bw.d+bl.d+G;
    double physical=mn(p.local_abs>0?p.local_abs:DBL_MAX,p.local_rel>0?p.local_rel*mx(mx(abs_fp64(j),abs_fp64(jb)),1.):DBL_MAX);
    double roundoff=.5*p.roundoff*DBL_EPSILON*(mx(abs_fp64(gap),1.)*abs_fp64(slope)+mx(abs_fp64(drop),1.)*abs_fp64(G));
    return {bw,bl,j-jb,mx(mx(physical,roundoff),DBL_MIN),slope,ew,el};
}
BC_HD void contact_cell(int64_t f,int64_t count,int64_t total,const int64_t* cells,
        const bool* anode,const double* s,const double* tr,const double* volts,const bool* active,
        double* out,const Params& p){
    int64_t i=cells[f],b=i/(int64_t(p.n)*p.n); bool an=anode[f];
    if(!active[b]){
        for(int k=0;k<NCONTACT;k++){out[k*count+f]=0;}
        return;
    }
    double T=mx(s[i],1.),wf=p.water0>0?clamp(s[3*total+i]/p.water0,0,1):0,
           sf=clamp(mn(s[total+i],s[2*total+i])/p.lp0,0,1);
    // The metal is an out-of-plane terminal, not a neighbouring material cell.
    // Its unresolved normal conduction length must not be replaced by grid dx.
    double G=mx(tr[5*total+i]/p.contact_normal,p.conductivity_floor/p.contact_normal);
    double drop=mx(an?volts[b]-s[7*total+i]:s[7*total+i]-p.cathode,0),lo=0,hi=drop,gap=.5*drop;
    ContactEval r=eval_contact(gap,drop,G,T,wf,sf,an,p);int used=0;bool ok=drop<=0;
    for(int it=1;it<=p.local_max&&!ok;it++){
        r=eval_contact(gap,drop,G,T,wf,sf,an,p);used=it;
        if(it>=p.local_min&&abs_fp64(r.residual)<=r.threshold){ok=true;break;}
        if(r.residual<0)lo=gap;else hi=gap;gap=.5*(lo+hi);
    }
    for(int it=0;it<p.newton&&!ok;it++){
        r=eval_contact(gap,drop,G,T,wf,sf,an,p);
        if(r.slope>p.slope_floor){double cand=gap-r.residual/r.slope;if(cand>=lo&&cand<=hi)gap=cand;}
        r=eval_contact(gap,drop,G,T,wf,sf,an,p);ok=abs_fp64(r.residual)<=r.threshold;
        if(r.residual<0)lo=gap;else hi=gap;
    }
    r=eval_contact(gap,drop,G,T,wf,sf,an,p);
    double ks=r.w.d+r.l.d,L=mx(ks,G),sm=mn(ks,G);
    out[f]=drop>0?r.w.j:0;out[count+f]=drop>0?r.l.j:0;
    out[2*count+f]=an?volts[b]-gap:p.cathode+gap;
    out[3*count+f]=drop>0&&L>p.slope_floor?sm/(1+sm/L):0;
    out[4*count+f]=drop>0?abs_fp64(r.residual)/r.threshold:0;
    out[5*count+f]=double(used);out[6*count+f]=r.ew;out[7*count+f]=r.el;
    // Normal electronic/ionic transport between the terminal and condensed
    // surface dissipates j * DeltaV [W/m^2].  The engine converts this compact
    // surface result to the surface-layer volume exactly once.
    out[8*count+f]=drop>0?(r.w.j+r.l.j)*mx(drop-gap,0):0;
}
BC_HD double grad_dir(const double* v,const bool* prop,int64_t i,int64_t pidx,int64_t midx,bool pv,bool mv,double dx){
    pv=pv&&prop[pidx];mv=mv&&prop[midx];
    return pv?(mv?(v[pidx]-v[midx])/(2*dx):(v[pidx]-v[i])/dx):(mv?(v[i]-v[midx])/dx:0);
}
BC_HD void current_cell(int64_t i,int64_t total,const double* s,const double* tr,const bool* prop,
                         double* out,const Params& p){
    if(!prop[i]){
        for(int k=0;k<3;k++){out[k*total+i]=0;}
        return;
    }
    int64_t local=i%(int64_t(p.n)*p.n);int row=int(local/p.n),col=int(local%p.n);
    double ex=-grad_dir(s+7*total,prop,i,i+1,i-1,col<p.n-1,col>0,p.dx);
    double ey=-grad_dir(s+7*total,prop,i,i+p.n,i-p.n,row<p.n-1,row>0,p.dx);
    double jdx=0,jdy=0;
    if(p.diffusion){
        jdx=-p.F*(p.zp*tr[i]*grad_dir(s+total,prop,i,i+1,i-1,col<p.n-1,col>0,p.dx)
                  +p.zm*tr[total+i]*grad_dir(s+2*total,prop,i,i+1,i-1,col<p.n-1,col>0,p.dx));
        jdy=-p.F*(p.zp*tr[i]*grad_dir(s+total,prop,i,i+p.n,i-p.n,row<p.n-1,row>0,p.dx)
                  +p.zm*tr[total+i]*grad_dir(s+2*total,prop,i,i+p.n,i-p.n,row<p.n-1,row>0,p.dx));
    }
    double jx=tr[5*total+i]*ex+jdx,jy=tr[5*total+i]*ey+jdy;
    double jm=sqrt_fp64(jx*jx+jy*jy),heat=0,rhs=0;
    int64_t nb[4]={i+1,i-1,i+p.n,i-p.n};bool valid[4]={col<p.n-1,col>0,row<p.n-1,row>0};
    // Conservative FV dissipation: evaluate each internal face with the same
    // harmonic transport used by the potential operator, then give half of
    // that face's power-density contribution to each adjacent control volume.
    // Summing q*cell_volume therefore counts every physical face exactly once.
    for(int d=0;d<4;d++){
      if(valid[d]&&prop[nb[d]]){
        int64_t j=nb[d];double eface=-(s[7*total+j]-s[7*total+i])/p.dx;
        double johm=harmonic(tr[5*total+i],tr[5*total+j])*eface,jdiff=0;
        if(p.diffusion){
            jdiff=-p.F*(p.zp*harmonic(tr[i],tr[j])*(s[total+j]-s[total+i])
                         +p.zm*harmonic(tr[total+i],tr[total+j])*(s[2*total+j]-s[2*total+i]))/p.dx;
            rhs+=jdiff/p.dx;
        }
        double face_power=(johm+jdiff)*eface;
        if(p.joule_mode==1)face_power=johm*eface;
        if(p.joule_mode==2)face_power=abs_fp64(eface)*abs_fp64(johm+jdiff);
        heat+=.5*face_power;
      }
    }
    // total_j_dot_e is a signed electrical power-transfer identity; diffusion
    // can make its local contribution negative.  The irreversible conductive
    // and legacy-magnitude models remain non-negative by construction/closure.
    out[i]=jm;out[total+i]=p.joule_mode==0?heat:mx(heat,0);
    out[2*total+i]=rhs;
}
BC_HD void stencil_cell(int64_t i,int64_t total,const double* coeff,const double* x,double* y,int n){
    double v=coeff[i]*x[i];int64_t j=i%(int64_t(n)*n);int r=int(j/n),c=int(j%n);
    if(c<n-1){v-=coeff[total+i]*x[i+1];}
    if(c>0){v-=coeff[2*total+i]*x[i-1];}
    if(r<n-1){v-=coeff[3*total+i]*x[i+n];}
    if(r>0){v-=coeff[4*total+i]*x[i-n];}
    y[i]=v;
}
} // namespace bc_native
