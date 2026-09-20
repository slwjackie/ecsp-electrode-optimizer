"""WENO5-JS / HLLC / conservative multi-scalar fluxes for the BC continuum.

The original paper-reference HLL solver remains unchanged. HLLC is used here
because HLL diffuses stationary thermal/composition contacts at acoustic speed.
This numerical choice is explicit, not asserted to have been specified by the
paper. All reactive/inventory traces share nonlinear weights so the linear
stoichiometric invariants survive advection. No reaction-front speed is fitted.
"""
from __future__ import annotations
import numpy as np
from ecsp_reactive.numerics import _weno_js_weights
from ecsp_nsga2.propagation import PropagationConfigurationError, PropagationCandidateNumericalError
from .chemistry import *


def _pad(P,axis,boundary):
    moved=np.moveaxis(P,axis,0)
    n=moved.shape[0]
    if boundary=="periodic":
        return moved[np.arange(-3,n+3)%n]
    if boundary=="transmissive":
        return np.concatenate((np.repeat(moved[:1],3,axis=0),moved,np.repeat(moved[-1:],3,axis=0)))
    if boundary=="reflective":
        left=moved[:3][::-1].copy(); right=moved[-3:][::-1].copy()
        normal=MX if axis==1 else MY
        left[...,normal]*=-1; right[...,normal]*=-1
        return np.concatenate((left,moved,right))
    raise PropagationConfigurationError("Condensed boundary must be reflective, periodic or transmissive")


def _trace(a,b,c,d,e,epsilon):
    center=(a+b+c+d+e)/5
    scale=np.maximum.reduce([abs(v-center) for v in (a,b,c,d,e)])
    scale=np.where(scale>0,scale,1.)
    aa,bb,cc,dd,ee=[(v-center)/scale for v in (a,b,c,d,e)]
    beta0=13/12*(aa-2*bb+cc)**2+.25*(aa-4*bb+3*cc)**2
    beta1=13/12*(bb-2*cc+dd)**2+.25*(bb-dd)**2
    beta2=13/12*(cc-2*dd+ee)**2+.25*(3*cc-4*dd+ee)**2
    # A common sensor/weight for all species and both alpha channels preserves
    # PVA + xi*X = constant and productWater - 2*xi*X = 0 on each trace.
    for beta in (beta0,beta1,beta2):
        beta[...,A1:]=np.max(beta[...,A1:],axis=-1,keepdims=True)
    w0,w1,w2=_weno_js_weights(beta0,beta1,beta2,epsilon)
    # Reconstruct in original units so all scalar linear constraints use the
    # exact same linear stencil (component affine normalisation is sensor-only).
    return w0*(2*a-7*b+11*c)/6 + w1*(-b+5*c+2*d)/6 + w2*(2*c+5*d-e)/6


def reconstruct(P,thermo,axis,boundary,epsilon=1e-6,first_order=False):
    n=P.shape[axis]
    if n<5:
        raise PropagationConfigurationError("WENO5 requires at least five cells per direction")
    padded=_pad(P,axis,boundary)
    face=np.arange(n+1); j=face+2
    donors_l=padded[j]; donors_r=padded[j+1]
    if first_order:
        left,right=donors_l,donors_r; fallback=np.zeros(left.shape[:-1],bool)
    else:
        left=_trace(padded[j-2],padded[j-1],padded[j],padded[j+1],padded[j+2],epsilon)
        right=_trace(padded[j+3],padded[j+2],padded[j+1],padded[j],padded[j-1],epsilon)
        fallback=~(thermo.valid_primitive(left)&thermo.valid_primitive(right))
        left=np.where(fallback[...,None],donors_l,left)
        right=np.where(fallback[...,None],donors_r,right)
    return np.moveaxis(left,0,axis),np.moveaxis(right,0,axis),int(fallback.sum()),int(fallback.size)


def physical_flux(U,P,pressure,normal):
    vel=P[...,normal]
    F=U*vel[...,None]
    F[...,normal]+=pressure
    F[...,ENERGY]+=pressure*vel
    return F


def riemann_flux(left,right,thermo,normal,kind="hllc"):
    UL=thermo.conservative(left); UR=thermo.conservative(right)
    rl,rr=left[...,RHO],right[...,RHO]
    ul,ur=left[...,normal],right[...,normal]
    pl,pr=thermo.pressure(rl),thermo.pressure(rr)
    sl=np.minimum(ul-thermo.sound_speed(rl),ur-thermo.sound_speed(rr))
    sr=np.maximum(ul+thermo.sound_speed(rl),ur+thermo.sound_speed(rr))
    FL=physical_flux(UL,left,pl,normal); FR=physical_flux(UR,right,pr,normal)
    denom=sr-sl
    hll=(sr[...,None]*FL-sl[...,None]*FR+(sl*sr)[...,None]*(UR-UL))/denom[...,None]
    hll=np.where((sl>=0)[...,None],FL,np.where((sr<=0)[...,None],FR,hll))
    star_fallback=0
    if kind=="hll":
        result=hll
    elif kind=="hllc":
        dl=rl*(sl-ul); dr=rr*(sr-ur)
        sm=(pr-pl+dl*ul-dr*ur)/(dl-dr)
        ps=pl+dl*(sm-ul)
        def star(U,P,speed,p):
            rho=P[...,RHO]; vel=P[...,normal]
            den=speed-sm
            safe=np.where(abs(den)>1e-100,den,1e-100)
            factor=(speed-vel)/safe
            V=U*factor[...,None]
            V[...,normal]=rho*factor*sm
            spden=rho*(speed-vel)
            spden=np.where(abs(spden)>1e-100,spden,1e-100)
            e=U[...,ENERGY]/rho+(sm-vel)*(sm+p/spden)
            V[...,ENERGY]=rho*factor*e
            return V
        USL=star(UL,left,sl,pl); USR=star(UR,right,sr,pr)
        bad=(~np.isfinite(sm))|(~np.isfinite(ps))|(ps<=0)
        bad|=~(thermo.valid_primitive(thermo.primitive(USL))&thermo.valid_primitive(thermo.primitive(USR)))
        fsl=FL+sl[...,None]*(USL-UL)
        fsr=FR+sr[...,None]*(USR-UR)
        result=np.where((sl>=0)[...,None],FL,np.where((sr<=0)[...,None],FR,
                        np.where((sm>=0)[...,None],fsl,fsr)))
        result=np.where(bad[...,None],hll,result)
        star_fallback=int(bad.sum())
        # Exact stationary-contact solution, not a heat-source suppression:
        # u_L=u_R=0 and p_L=p_R has flux (0,p,0,0,0,...).
        contact=(ul==0)&(ur==0)&(pl==pr)
        exact=np.zeros_like(result); exact[...,normal]=pl
        result=np.where(contact[...,None],exact,result)
    else:
        raise PropagationConfigurationError("riemann_solver must be hllc or hll")
    if not np.isfinite(result).all():
        raise PropagationCandidateNumericalError("Nonfinite condensed Riemann flux")
    return result,star_fallback


def mechanically_stationary(U,boundary):
    # This is an invariant subspace of THIS barotropic model. No tolerance or
    # artificial pressure damping is used. Thermal sources cannot change p.
    return (boundary in {"reflective","periodic","transmissive"}
            and np.all(U[...,MX:MY+1]==0)
            and np.all(U[...,RHO]==U.flat[0]))


def flux_divergence(U,thermo,dx,depth,*,boundary="reflective",epsilon=1e-6,
                    kind="hllc",first_order=False,stationary_fast_path=False):
    if stationary_fast_path and kind=="hllc" and mechanically_stationary(U,boundary):
        return np.zeros_like(U),np.zeros(U.shape[-1]),{"face_fallbacks":0,"faces":0,"hllc_fallbacks":0,"mechanics_skipped_exact":True}
    P=thermo.primitive(U)
    fluxes=[]; total_fallback=0; faces=0; star_fallback=0
    for axis,normal in ((1,MX),(0,MY)):
        left,right,fb,nfaces=reconstruct(P,thermo,axis,boundary,epsilon,first_order)
        F,sfb=riemann_flux(left,right,thermo,normal,kind)
        if boundary=="periodic":
            first=[slice(None)]*3; last=first.copy(); first[axis]=0; last[axis]=-1
            F[tuple(last)]=F[tuple(first)]
        elif boundary=="reflective":
            # A stationary impermeable wall has exactly zero mass, energy and
            # tangential/scalar transport. Its normal stress is the Riemann flux.
            for end in (0,-1):
                idx=[slice(None)]*3; idx[axis]=end
                face=F[tuple(idx)]
                keep=face[...,normal].copy(); face[...]=0; face[...,normal]=keep
        fluxes.append(F); total_fallback+=fb; faces+=nfaces; star_fallback+=sfb
    fx,fy=fluxes
    rhs=-(np.diff(fx,axis=1)+np.diff(fy,axis=0))/dx
    boundary_in=-((fx[:,-1,:]-fx[:,0,:]).sum(axis=0)+(fy[-1,:,:]-fy[0,:,:]).sum(axis=0))*dx*depth
    return rhs,boundary_in,{"face_fallbacks":total_fallback,"faces":faces,"hllc_fallbacks":star_fallback,
                            "mechanics_skipped_exact":False}
