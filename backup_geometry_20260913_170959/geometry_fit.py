"""Topology-preserving area fit for round LINE/ARC/BRANCH centreline strokes.

Global maximum_width_mm remains a design ceiling, not a placement pitch. A
common width multiplier s is bounded by r_i+r_j+gap <= d_ij, with r=s*w/2.
Length dilation preserves circular arcs and the tree's branch connectivity.
No mask pixel is added/removed to manufacture an apparently feasible solution.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

import numpy as np
from scipy import ndimage

FIT_SCHEMA = "ecsp.clearance_length_first/v1"


def transformed_components(components, length_scale, width_scale=1.0):
    output = []
    centres = []
    for comp in components:
        xy = np.asarray([p for points, _ in comp for p in points], dtype=float)
        centre = (xy.min(axis=0) + xy.max(axis=0)) * 0.5
        centres.append(centre.tolist())
        output.append([
            ([(float(centre[0] + length_scale * (x - centre[0])),
               float(centre[1] + length_scale * (y - centre[1]))) for x, y in points],
             float(w * width_scale)) for points, w in comp
        ])
    return output, centres


def _segments(comp):
    aa, bb, ww = [], [], []
    for points, width in comp:
        for a, b in zip(points[:-1], points[1:]):
            aa.append(a); bb.append(b); ww.append(width)
    return np.asarray(aa, dtype=float), np.asarray(bb, dtype=float), np.asarray(ww, dtype=float)


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _segment_distances(first, second):
    """Exact distance for the finite LINE/chord segments used by rasterization."""
    a, b, _ = first; c, d, _ = second
    a = a[:, None, :]; b = b[:, None, :]
    c = c[None, :, :]; d = d[None, :, :]
    def point_line(p, u, v):
        delta = v-u
        t = np.sum((p-u)*delta, axis=-1) / np.maximum(np.sum(delta*delta, axis=-1), 1e-30)
        closest = u + np.clip(t, 0.0, 1.0)[..., None]*delta
        return np.linalg.norm(p-closest, axis=-1)
    distance = np.minimum.reduce([point_line(a,c,d), point_line(b,c,d),
                                  point_line(c,a,b), point_line(d,a,b)])
    ab, cd = b-a, d-c
    den = _cross(ab,cd)
    nonparallel = np.abs(den) > 1e-14
    safe = np.where(nonparallel,den,1.0)
    t = _cross(c-a,cd)/safe; u = _cross(c-a,ab)/safe
    hit = nonparallel & (t>=0.0)&(t<=1.0)&(u>=0.0)&(u<=1.0)
    return np.where(hit,0.0,distance)


def _width_interval(components, limits, boxes, obstacles):
    widths = [w for comp in components for _,w in comp]
    if not widths or min(widths) <= 0 or not np.isfinite(widths).all():
        return 1.0, 0.0
    lo = max(limits.minimum_width_mm/w for w in widths)
    hi = min(limits.maximum_width_mm/w for w in widths)
    cell = limits.domain_mm/limits.grid_size
    # Pillow's pixel-centre rounding and round endpoint caps each consume some
    # clearance. The final raster is revalidated too; no sampled-distance proxy.
    guard = cell * 1.05
    for comp, box in zip(components, boxes):
        xmin,ymin,xmax,ymax = box or [limits.margin_mm,limits.margin_mm,
                                    limits.domain_mm-limits.margin_mm,
                                    limits.domain_mm-limits.margin_mm]
        for points, w in comp:
            xy = np.asarray(points)
            clearance = min(float(xy[:,0].min()-xmin),float(xmax-xy[:,0].max()),
                            float(xy[:,1].min()-ymin),float(ymax-xy[:,1].max()))
            hi = min(hi, 2.0*(clearance-guard)/w)
    segments = [_segments(comp) for comp in components]
    gap = max(limits.minimum_gap_mm, cell) + 2.0*guard
    for i, first in enumerate(segments):
        for second in segments[i+1:]:
            dist = _segment_distances(first, second)
            hi = min(hi,float(np.min(2.0*(dist-gap)/(first[2][:,None]+second[2][None,:]))))
    for first in segments:
        for obs in obstacles:
            second = _segments(obs)
            dist = _segment_distances(first,second)
            # Obstacle has its final, already fitted width; it is not scaled.
            hi = min(hi,float(np.min((2.0*(dist-gap)-second[2][None,:])/first[2][:,None])))
    return float(lo),float(hi)


def fit_area_components(components, limits, *, placement_boxes=None, obstacles=None):
    from .geometry import _draw_polylines, _component_count, minimum_clear_gap_pixels
    boxes = placement_boxes or [None]*len(components)
    obstacles = obstacles or []
    if len(boxes)!=len(components):
        raise ValueError("One placement box is required per component")
    target = limits.target_area_fraction_per_polarity * limits.grid_size**2
    empty = np.zeros((limits.grid_size,limits.grid_size),dtype=bool)
    if not components or any(not c for c in components):
        return empty,dict(schema=FIT_SCHEMA,status="no_paths",length_scale=1.0,
                          width_scale=1.0,centres=[],area_error_pixels=target)
    # Longest allowed dilation at unchanged width, respecting each component's
    # reservation and maximum LINE length. A global max of 5 mm is never reduced.
    guard = 1.05*limits.domain_mm/limits.grid_size
    length_hi = 1.6
    length_lo = 0.72
    for comp, box in zip(components,boxes):
        xy=np.asarray([p for ps,_ in comp for p in ps]); centre=(xy.min(0)+xy.max(0))/2
        xmin,ymin,xmax,ymax = box or [limits.margin_mm,limits.margin_mm,
                                    limits.domain_mm-limits.margin_mm,
                                    limits.domain_mm-limits.margin_mm]
        for ps,w in comp:
            for p in ps:
                for axis,lower,upper in ((0,xmin,xmax),(1,ymin,ymax)):
                    offset=p[axis]-centre[axis]
                    room=(upper-centre[axis] if offset>0 else centre[axis]-lower)-0.5*w-guard
                    if abs(offset)>1e-12:
                        length_hi=min(length_hi,room/abs(offset))
            if len(ps)==2:
                length=math.dist(ps[0],ps[-1])
                if length>1e-12:
                    length_hi=min(length_hi,limits.maximum_segment_mm/length)
                    length_lo=max(length_lo,limits.minimum_segment_mm/length)
    length_hi=max(1.0,float(length_hi))
    best=None
    eval_count=0
    cached={}
    component_count=len(components)
    def draw_candidate(length_scale,width_scale,phase,interval=None):
        nonlocal best,eval_count
        key=(round(length_scale,13),round(width_scale,13))
        if key in cached:return cached[key]
        trans,centres=transformed_components(components,length_scale)
        lo,hi=interval if interval is not None else _width_interval(trans,limits,boxes,obstacles)
        if width_scale<lo-1e-12 or width_scale>hi+1e-12:
            cached[key]=None;return None
        mask=_draw_polylines([path for comp in trans for path in comp],limits,width_scale)
        eval_count+=1
        # Hard component preservation, not a cost term.
        if _component_count(mask)!=component_count:
            cached[key]=None;return None
        area=float(mask.sum())
        rank=(abs(area-target),abs(width_scale-1.0),-length_scale)
        if best is None or rank<best[0]:
            # Stronger ownership check: fragmentation must not compensate for a
            # merger and preserve only the total component count by accident.
            singles=[_draw_polylines(c,limits,width_scale) for c in trans]
            valid=all(_component_count(m)==1 for m in singles)
            cell=limits.domain_mm/limits.grid_size
            valid=valid and all(minimum_clear_gap_pixels(singles[i],singles[j])*cell
                         >=limits.minimum_gap_mm-1e-12
                         for i in range(len(singles)) for j in range(i+1,len(singles)))
            if not valid:
                cached[key]=None;return None
            best=(rank,mask,dict(schema=FIT_SCHEMA,status="candidate",length_scale=float(length_scale),
                   width_scale=float(width_scale),centres=centres,phase=phase,
                   width_scale_lower=lo,width_scale_clearance_upper=hi,
                   global_maximum_width_mm=limits.maximum_width_mm,area_error_pixels=abs(area-target)))
        cached[key]=area
        return area
    # 1. Prefer additional centreline/arc/branch length at unchanged width.
    base=draw_candidate(1.0,1.0,"original")
    if base is not None and base < target:
        previous=1.0
        for ls in np.linspace(1.0,length_hi,13)[1:]:
            area=draw_candidate(float(ls),1.0,"length_first")
            if area is not None and area>=target:
                lo,hi=previous,float(ls)
                for _ in range(14):
                    mid=(lo+hi)/2; val=draw_candidate(mid,1.0,"length_first")
                    if val is None or val>=target:hi=mid
                    else:lo=mid
                break
            if area is not None:previous=float(ls)
        if best is not None and best[0][0] <= max(1.0,0.3*limits.area_tolerance_fraction*target):
            best[2].update(status="fitted",evaluated_rasters=eval_count)
            return best[1],best[2]
    # 2. Only if necessary, refine width within analytically safe clearance.
    # Backtracking length leaves room for width without merging components.
    # If an outward branch already fills its slot, modest *uniform* packing
    # contraction can release width clearance. It is tried only after length-
    # first fitting and never violates minimum LINE lengths or the 5 mm ceiling.
    for ls in np.linspace(length_hi,min(1.0,length_lo),13):
        trans,_=transformed_components(components,float(ls))
        lo,hi=_width_interval(trans,limits,boxes,obstacles)
        if lo>hi or hi<=0.0:continue
        interval=(lo,hi)
        draw_candidate(float(ls),lo,"clearance_width",interval)
        draw_candidate(float(ls),hi,"clearance_width",interval)
        low,high=lo,hi
        for _ in range(15):
            ws=(low+high)/2
            area=draw_candidate(float(ls),ws,"clearance_width",interval)
            if area is None or area>=target:high=ws
            else:low=ws
        if best is not None and best[0][0]<=max(1.0,0.15*limits.area_tolerance_fraction*target):
            break
    if best is None:
        return empty,dict(schema=FIT_SCHEMA,status="no_topology_preserving_fit",length_scale=1.0,
                          width_scale=1.0,centres=[],area_error_pixels=target,evaluated_rasters=eval_count)
    plan=best[2]
    plan.update(status="fitted" if best[0][0]<=limits.area_tolerance_fraction*target+1e-12
                else "area_target_unreachable",evaluated_rasters=eval_count)
    return best[1],plan


def geometry_digest(genome):
    """Exclude identities/old fit metadata, but include every geometric input."""
    clean={k:genome[k] for k in ("anode","cathode")}
    return hashlib.sha256(json.dumps(clean,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def limits_digest(limits):
    from dataclasses import asdict
    return hashlib.sha256(json.dumps(asdict(limits),sort_keys=True).encode()).hexdigest()


def apply_fit_plan(genome,polarity,plan):
    # Dilate centres/LINE lengths/ARC radii uniformly, not x/y independently:
    # arcs remain circles and branch connectivity and topology IDs are unchanged.
    from .geometry import _nodes_in_component
    if len(plan.get("centres",[])) != len(genome[polarity]["components"]):return
    for root,centre in zip(genome[polarity]["components"],plan["centres"]):
        ls,ws=plan["length_scale"],plan["width_scale"]
        start=root.get("start",genome[polarity].get("start",[0.0,0.0]))
        root["start"]=[centre[k]+ls*(start[k]-centre[k]) for k in range(2)]
        for node in _nodes_in_component(root):
            for key in ("length_mm","radius_mm"):
                if key in node:node[key]=float(node[key])*ls
            node["width_mm"]=float(node.get("width_mm",0.6))*ws
    genome[polarity]["start"]=list(genome[polarity]["components"][0]["start"])
    genome[polarity]["initial_heading_deg"]=float(genome[polarity]["components"][0]["initial_heading_deg"])
