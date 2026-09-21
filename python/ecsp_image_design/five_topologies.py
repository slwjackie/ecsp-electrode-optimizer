"""Five topology-specific electrode fitters. Geometry only; no ECSP physics."""
from __future__ import annotations
import base64, hashlib, math, zlib
import numpy as np
import shapely
from shapely.geometry import Polygon, box, Point, LineString
from shapely.affinity import scale, translate
from scipy import ndimage as ndi
from scipy.optimize import brentq


def parts(g):
    if g.is_empty:return []
    if isinstance(g,Polygon):return [g]
    return [p for q in getattr(g,'geoms',()) for p in parts(q)]

def signature(g):return sorted(len(p.interiors) for p in parts(g))

def mask_signature(m):
    m=np.asarray(m,bool);n4=ndi.label(m)[1];n8=ndi.label(m,np.ones((3,3)))[1]
    holes=ndi.label(ndi.binary_fill_holes(m)&~m)[1]
    return {'components4':int(n4),'components8':int(n8),'holes':int(holes)}

def polygon(m):
    runs=[]
    for y,row in enumerate(np.asarray(m,bool)):
        d=np.diff(np.r_[False,row,False].astype(np.int8))
        for x0,x1 in zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)):runs.append(box(int(x0),y,int(x1),y+1))
    return shapely.union_all(runs) if runs else Polygon()

def unpack(rec):
    b=zlib.decompress(base64.b64decode(rec['mask_zlib_b64']))
    a=np.unpackbits(np.frombuffer(b,np.uint8))[:int(np.prod(rec['shape']))].reshape(rec['shape']).astype(bool)
    if hashlib.sha256(a.astype(np.uint8).tobytes()).hexdigest()!=rec['mask_sha256']:raise ValueError('Reference mask hash mismatch')
    return a

def render(g,n,L):
    x=(np.arange(n)+.5)*L/n;shapely.prepare(g);return shapely.contains_xy(g,x[None,:],x[:,None])

def rolling_open(g,r):return g.buffer(-r,quad_segs=16).buffer(r,quad_segs=16)

def first_erosion_width(g):
    sig=signature(g)
    if not sig:return 0.
    upper=min(min(p.bounds[2]-p.bounds[0],p.bounds[3]-p.bounds[1]) for p in parts(g));prev=0.
    for d in np.linspace(upper/48,upper*1.01,49):
        if signature(g.buffer(-float(d)/2,quad_segs=8))!=sig:
            lo,hi=prev,float(d)
            for _ in range(15):
                mid=(lo+hi)/2
                if signature(g.buffer(-mid/2,quad_segs=8))==sig:lo=mid
                else:hi=mid
            return lo
        prev=float(d)
    return prev

def gap_guard(a,c,h):
    if not a.any() or not c.any() or np.any(a&c):return 0.
    support=ndi.binary_dilation(c,np.ones((3,3)))
    return float(ndi.distance_transform_edt(~support,sampling=(h,h))[a].min())

def offset_area(g,target):
    reach=max(g.bounds[2]-g.bounds[0],g.bounds[3]-g.bounds[1]);lo,hi=-reach,reach
    t=brentq(lambda x:g.buffer(x,quad_segs=16).area-target,lo,hi,xtol=1e-10,rtol=1e-12,maxiter=100)
    return g.buffer(t,quad_segs=16)

def _diamond(cx,cy,q):return Polygon([(cx,cy-q),(cx+q,cy),(cx,cy+q),(cx-q,cy)])

def _sector(cx,cy,r0,r1,t0,t1,n=64):
    ts=np.linspace(t0,t1,n)
    return Polygon([(cx+r1*np.cos(t),cy+r1*np.sin(t)) for t in ts]+[(cx+r0*np.cos(t),cy+r0*np.sin(t)) for t in ts[::-1]])

def staggered_dimension_witness(L,n,area,margin=.75,min_width=2.,min_gap=3.,minimum_overlap=.45,area_tol=.01,target_overlap=.50):
    h=L/n;gap=math.ceil(min_gap/h);edge=math.ceil(margin/h);wp_min=math.ceil(min_width/h);wp_max=(n-3*gap-2*edge)//4
    count=area/(h*h);best=None
    for w in range(wp_min,wp_max+1):
        ideal=count/(2*w)
        for length in {math.floor(ideal),math.ceil(ideal)}:
            if length<1 or length>n:continue
            overlap=max(0,2*length-n)/n;err=abs(2*w*length-count)/count
            if overlap+1e-12<minimum_overlap or err>area_tol:continue
            score=(err+.05*abs(overlap-target_overlap),err,abs(overlap-target_overlap),w)
            if best is None or score<best[0]:best=(score,{'width_px':w,'length_px':length,'gap_px':gap,'minimum_width_mm':w*h,'minimum_gap_mm':gap*h,'overlap_fraction':overlap,'area_relative_error':err})
    return None if best is None else best[1]

def _finalize(rec,a,c,fit_mode,*,min_width=2.,min_gap=3.,min_domain=25.,max_domain=500.,margin=1.,max_pitch=.25,min_grid=193,max_grid=2049,extra=None):
    if a.is_empty or c.is_empty or not a.is_valid or not c.is_valid or not a.disjoint(c):return {'source_id':rec['source_id'],'status':'fit_search_exhausted','reason':'invalid/overlapping special geometry'},None
    common=min(a.area,c.area)
    if a.area>common:a=offset_area(a,common)
    if c.area>common:c=offset_area(c,common)
    if not a.disjoint(c):return {'source_id':rec['source_id'],'status':'fit_search_exhausted','reason':'equalisation created contact'},None
    widths=[first_erosion_width(a),first_erosion_width(c)];gap=a.distance(c);centre=shapely.union_all([a,c]).centroid.coords[0]
    grow=max(1.,(min_width+.30)/max(1e-9,min(widths)),(min_gap+.65)/max(1e-9,gap))
    a=scale(a,grow,grow,origin=centre);c=scale(c,grow,grow,origin=centre)
    b=shapely.union_all([a,c]).bounds;span=max(b[2]-b[0],b[3]-b[1]);L=float(math.ceil(max(min_domain,span+2*margin)))
    if L>max_domain:return {'source_id':rec['source_id'],'status':'fit_search_exhausted','reason':'domain exceeds cap'},None
    dx=L/2-(b[0]+b[2])/2;dy=L/2-(b[1]+b[3])/2;a=translate(a,dx,dy);c=translate(c,dx,dy)
    n=max(min_grid,int(math.ceil(L/max_pitch)));n+=1-n%2
    if n>max_grid:return {'source_id':rec['source_id'],'status':'fit_search_exhausted','reason':'grid exceeds cap'},None
    am=cm=None;ae=imb=float('inf')
    for nn in (n,2*n-1):
        if nn>max_grid:break
        aa,cc=render(a,nn,L),render(c,nn,L);cell=L*L/(nn*nn)
        if any(mask_signature(m)['components8']!=len(signature(p)) for m,p in ((aa,a),(cc,c))):continue
        er=max(abs(aa.sum()*cell-a.area)/a.area,abs(cc.sum()*cell-c.area)/c.area);bi=abs(aa.sum()-cc.sum())/max(1,(aa.sum()+cc.sum())/2)
        am,cm,n,ae,imb=aa,cc,nn,float(er),float(bi)
        if er<=.01 and bi<=.01:break
    if am is None or ae>.01 or imb>.01:return {'source_id':rec['source_id'],'status':'fit_search_exhausted','reason':'raster area/topology QC failed'},None
    gg=gap_guard(am,cm,L/n)
    if gg<min_gap:return {'source_id':rec['source_id'],'status':'fit_search_exhausted','reason':'raster gap below 3 mm'},None
    wd=[]
    for p in (a,c):
        crit=first_erosion_width(p);unc=p.difference(rolling_open(p,min_width/2)).area/max(p.area,1e-30)
        if crit<min_width-.002 or unc>=.002:return {'source_id':rec['source_id'],'status':'fit_search_exhausted','reason':'width QC failed'},None
        wd.append(crit)
    meta={'source_id':rec['source_id'],'status':'geometry_accepted','domain_mm':L,'grid_size':n,'anode_area_mm2':a.area,'cathode_area_mm2':c.area,'electrode_area_fraction':(a.area+c.area)/(L*L),'minimum_gap_actual_mm':a.distance(c),'physics_grid_gap_guard_mm':gg,'minimum_width_actual_mm':min(wd),'raster_area_relative_error':ae,'raster_balance_relative_error':imb,'anode_components':len(signature(a)),'cathode_components':len(signature(c)),'fit_mode':fit_mode,'source_mask_sha256':rec['mask_sha256'],'postflame_physics_changed':False,'source_to_fitted_transform':extra or {}}
    witness=staggered_dimension_witness(L,n,(a.area+c.area)/2)
    if witness is None:
        return {'source_id':rec['source_id'],'status':'matched_baseline_context_search_exhausted','reason':'no same-domain staggered witness'},None
    meta['matched_staggered_dimension_witness']=witness
    return meta,(a,c,am,cm)

def fit_E058(rec,**kw):
    raw=unpack(rec);fitted=[]
    for m in raw:
        lab,n=ndi.label(m);gs=[]
        for k in range(1,n+1):
            yy,xx=np.where(lab==k);area=float(len(xx));gs.append(Point(float(xx.mean()+.5),float(yy.mean()+.5)).buffer(math.sqrt(area/math.pi),quad_segs=48))
        fitted.append(shapely.union_all(gs))
    a,c=fitted;target=min(a.area,c.area);a=scale(a,math.sqrt(target/a.area),math.sqrt(target/a.area),origin='centroid');c=scale(c,math.sqrt(target/c.area),math.sqrt(target/c.area),origin='centroid')
    s=max((kw.get('min_width',2.)+.30)/min(first_erosion_width(a),first_erosion_width(c)),(kw.get('min_gap',3.)+.65)/a.distance(c));o=shapely.union_all([a,c]).centroid.coords[0]
    return _finalize(rec,scale(a,s,s,origin=o),scale(c,s,s,origin=o),'large_pad_override',extra={'topology_fitter':'fit_E058'},**kw)

def fit_E114(rec,**kw):
    R=16.;ri=14.;hub=2.1;sw=1.4;r0=4.2;r1=13.5;half=math.radians(10.)
    red=[Point(0,0).buffer(R,quad_segs=64).difference(Point(0,0).buffer(ri,quad_segs=64)),Point(0,0).buffer(hub,quad_segs=48)]
    blue=[]
    for k in range(8):
        angb=math.radians(45*k);angr=math.radians(22.5+45*k)
        blue.append(_sector(0,0,r0,r1,angb-half,angb+half))
        red.append(LineString([(hub*math.cos(angr),hub*math.sin(angr)),(ri*math.cos(angr),ri*math.sin(angr))]).buffer(sw/2,cap_style=1,join_style=1,quad_segs=16))
    return _finalize(rec,shapely.union_all(red),shapely.union_all(blue),'radial_spoke_sector_fitter',extra={'topology_fitter':'fit_E114'},**kw)

def fit_R038(rec,**kw):
    outer=[(10,.5),(8,3.5),(6,4.5),(5,7.5),(3,8.5),(2,11.5),(.5,13),(2,14),(4,16),(8,17),(12,17),(16,16),(18,14),(19.5,13),(18,11.5),(17,8.5),(15,7.5),(14,4.5),(12,3.5),(10,.5)]
    inner=[(10,6.5),(9,7.8),(8.3,8.2),(7.8,9.2),(7.2,9.8),(7,10.5),(7.5,11),(8.4,11.6),(9.4,11.9),(10.6,11.9),(11.6,11.6),(12.5,11),(13,10.5),(12.8,9.8),(12.2,9.2),(11.7,8.2),(11,7.8),(10,6.5)]
    return _finalize(rec,LineString(outer).buffer(.55,cap_style=1,join_style=1,quad_segs=16),LineString(inner).buffer(.55,cap_style=1,join_style=1,quad_segs=16),'single_trace_bifilar_fitter',extra={'topology_fitter':'fit_R038'},**kw)

def fit_R050(rec,**kw):
    min_gap=kw.get('min_gap',3.);r=8.;hr=r+min_gap+.2;pitch=2*hr+2.4;centres=[((i-1)*pitch,(j-1)*pitch) for j in range(3) for i in range(3)]
    a=shapely.union_all([Point(x,y).buffer(r,quad_segs=48) for x,y in centres]);holes=shapely.union_all([Point(x,y).buffer(hr,quad_segs=48) for x,y in centres]);span=2*pitch+2*hr
    def rs(s):return box(-s/2,-s/2,s/2,s/2).buffer(-1.2,quad_segs=24).buffer(1.2,quad_segs=24)
    side=brentq(lambda s:rs(s).difference(holes).area-a.area,span+1,span+5);c=rs(side).difference(holes)
    return _finalize(rec,a,c,'perforated_plate_dot_fitter',extra={'topology_fitter':'fit_R050'},**kw)

def fit_R091(rec,**kw):
    p=10.;qb=2.22;qr=1.5;conn=.25
    def rd(x,y,q):return _diamond(x,y,q).buffer(-.2,join_style=1,quad_segs=12).buffer(.2,join_style=1,quad_segs=12)
    blue=[]
    for xs,y in [([0,p,2*p],0),([.5*p,1.5*p,2.5*p],p),([0,p,2*p],2*p)]:
        for x in xs:blue.append(rd(x,y,qb))
    red=[]
    for xs,y in [([-.5*p,.5*p,1.5*p,2.5*p],.5*p),([0,p,2*p],1.5*p),([-.5*p,.5*p,1.5*p,2.5*p],2.5*p)]:
        for x in xs:red.append(rd(x,y,qr))
    centres=[g.centroid.coords[0] for g in red]
    for i,(x1,y1) in enumerate(centres):
        for x2,y2 in centres[i+1:]:
            if ((x1-x2)**2+(y1-y2)**2)**.5<1.02*p:red.append(LineString([(x1,y1),(x2,y2)]).buffer(conn,cap_style=1,join_style=1,quad_segs=12))
    red.append(LineString([(-.5*p,.5*p),(-.5*p,2.5*p)]).buffer(conn,cap_style=1,join_style=1,quad_segs=12));red.append(LineString([(-.5*p,1.5*p),(0,1.5*p)]).buffer(conn,cap_style=1,join_style=1,quad_segs=12))
    return _finalize(rec,shapely.union_all(red).buffer(0),shapely.union_all(blue),'diamond_lattice_pitch_fitter',extra={'topology_fitter':'fit_R091'},**kw)

SPECIAL={'E058':fit_E058,'E114':fit_E114,'R038':fit_R038,'R050':fit_R050,'R091':fit_R091}
def fit_reference(rec,**kw):return SPECIAL[rec['source_id']](rec,**kw)
