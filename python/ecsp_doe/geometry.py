"""Direct PHIDL CAD realization and fail-closed topology validation.

Only GeometryLimits and the plain RasterizedGeometry carrier are imported from
legacy geometry. No generator, repair or area fitter participates in this path.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import networkx as nx
import numpy as np
from phidl import Device, Path as PhidlPath
from scipy import ndimage
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union
from shapely import contains_xy, prepare
from skimage.morphology import skeletonize

from ecsp_nsga2.geometry import GeometryLimits, RasterizedGeometry, minimum_clear_gap_pixels


class GeometryReject(ValueError):
    def __init__(self, reason: str, details: dict | None = None):
        self.reason = reason
        self.details = details or {}
        super().__init__(f"{reason}: {self.details}")


@dataclass
class GeometryCandidate:
    anode_mask: np.ndarray
    cathode_mask: np.ndarray
    design_anode_mask: np.ndarray
    design_cathode_mask: np.ndarray
    metadata: dict[str, Any]
    polygons: dict[str, list]

    @property
    def geometry_hash(self):
        return self.metadata['geometry_hash']

    @property
    def parameter_vector(self):
        return self.metadata['parameter_vector']

    def rasterized(self):
        n = self.anode_mask.shape[0]
        a = float(self.anode_mask.mean())
        c = float(self.cathode_mask.mean())
        return RasterizedGeometry(self.anode_mask, self.cathode_mask,
                                  {'anode_area_fraction': a, 'cathode_area_fraction': c,
                                   'anode_contact_area_fraction': a,
                                   'cathode_contact_area_fraction': c,
                                   'grid_size': n}, 0.0, {})


def _components(shape):
    if shape.is_empty:
        return []
    return [shape] if isinstance(shape, Polygon) else list(shape.geoms)


def graph_statistics(spec):
    g = nx.Graph()
    g.add_nodes_from(range(len(spec['nodes'])))
    g.add_edges_from((e['u'], e['v']) for e in spec['edges'])
    return {'components': nx.number_connected_components(g),
            'endpoints': sum(d == 1 for _, d in g.degree()),
            'branch_points': sum(d >= 3 for _, d in g.degree()),
            'loops': g.number_of_edges() - g.number_of_nodes() + nx.number_connected_components(g)}


def raster_topology(mask: np.ndarray, pixel_mm: float, width_mm: float = 0.0):
    """Skeleton diagnostics with degree-3 pixel clusters treated as junctions.

    Four-connected foreground components and eight-connected background holes
    deliberately reject diagonal electrical contacts and closed raster holes.
    """
    mask = np.asarray(mask, bool)
    _, count = ndimage.label(mask)
    bg, nbg = ndimage.label(~np.pad(mask, 1), structure=np.ones((3, 3)))
    holes = nbg - 1
    skel = skeletonize(mask)
    kernel = np.ones((3, 3), int)
    degree = ndimage.convolve(skel.astype(int), kernel, mode='constant') - skel
    endpoints = skel & (degree == 1)
    branches = skel & (degree >= 3)
    # At a diagonal corner, digital skeleton pixels form little triangles.
    # Cluster each junction, without joining distinct lattice nodes.
    _, branch_count = ndimage.label(branches, structure=np.ones((3, 3)))
    return {'components': int(count), 'endpoints': int(endpoints.sum()),
            'branch_points': int(branch_count), 'loops': int(holes),
            'skeleton_pixels': int(skel.sum())}


def validate_topology_mask(mask, expected, pixel_mm, width_mm=0.0, *, stage='raster'):
    observed = raster_topology(mask, pixel_mm, width_mm)
    for key in ('components', 'loops', 'endpoints', 'branch_points'):
        if observed[key] != expected[key]:
            raise GeometryReject(f'{stage}_topology_{key}', {'expected': expected, 'observed': observed})
    return observed


def _curve(a, b, kind, bend, sign):
    a, b = np.asarray(a, float), np.asarray(b, float)
    delta = b - a
    length = float(np.linalg.norm(delta))
    tangent = delta / length
    normal = np.array([-tangent[1], tangent[0]]) * sign
    t = np.linspace(0, 1, 25)
    if kind == 'LINE':
        return np.array([a, b])
    if kind == 'ARC':
        # A genuine circular arc specified by chord and sagitta.
        h = length * bend
        radius = length * length / (8*h) + h / 2
        center = (a+b)/2 - normal * (radius-h)
        start = math.atan2(*(a-center)[::-1])
        end = math.atan2(*(b-center)[::-1])
        sweep = (end-start+math.pi) % (2*math.pi)-math.pi
        angles = start + t*sweep
        return center + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    if kind == 'SPLINE':
        p1 = a + delta/3 + normal*length*bend*1.6
        p2 = a + 2*delta/3 - normal*length*bend*1.6
        tt = t[:,None]
        return (1-tt)**3*a + 3*(1-tt)**2*tt*p1 + 3*(1-tt)*tt**2*p2 + tt**3*b
    raise GeometryReject('unknown_primitive', {'primitive': kind})


def _separated_centerlines(spec, parameters, limits, options):
    count = spec['component_pair'][0]
    n = limits.grid_size
    pixel = limits.domain_mm / min(n, options.get('physics_grid_size', n))
    same_gap = max(limits.minimum_gap_mm, float(options.get('minimum_same_polarity_gap_mm', limits.minimum_gap_mm)))
    # Reserve the full CAD and both raster-grid clearances between region boxes.
    split_gap = max(limits.minimum_gap_mm, same_gap) + 2.2*pixel
    d, m = limits.domain_mm, limits.margin_mm
    region_w = (d - 2*m - split_gap)/2
    region_h = (d - 2*m - (count-1)*split_gap)/count
    target = d*d*limits.target_area_fraction_per_polarity/count
    coords = np.asarray(spec['nodes'], float)
    extent = coords.max(axis=0)
    full_points = coords / extent * [region_w, region_h]
    full_length = sum(np.linalg.norm(full_points[e['u']]-full_points[e['v']]) for e in spec['edges'])
    # Area/length determines the initial skeleton envelope BEFORE polygon fitting.
    provisional_width = target/full_length
    reserve = max(provisional_width*.88, limits.minimum_width_mm*.55) + pixel*.8
    inner_w, inner_h = region_w-2*reserve, region_h-2*reserve
    if min(inner_w, inner_h) <= 0:
        raise GeometryReject('insufficient_skeleton_envelope')
    sx, sy = parameters['stretch_x'], parameters['stretch_y']
    local = (coords / extent - .5) * [inner_w*sx, inner_h*sy]
    local[:,0] += parameters['shear'] * local[:,1]
    local += [parameters['shift_x']*pixel, parameters['shift_y']*pixel]
    result = {}
    for pi, polarity in enumerate(('anode', 'cathode')):
        result[polarity] = []
        for ci in range(count):
            center = [m + region_w/2 + pi*(region_w+split_gap),
                      m + region_h/2 + ci*(region_h+split_gap)]
            points = local.copy()
            # mirror operation, preserving electrical components and graph.
            if pi:
                points[:,0] *= -1
            if ci % 2:
                points[:,1] *= -1
            points += center
            paths = [_curve(points[e['u']], points[e['v']], e['primitive'],
                            parameters['bend'], 1 if i%2 else -1)
                     for i,e in enumerate(spec['edges'])]
            result[polarity].append({'points': points, 'paths': paths})
    return result


def _interdigitated_centerlines(spec, parameters, limits, options):
    """Place mirrored comb graphs in a shared window with alternating rows."""
    count = spec['component_pair'][0]
    finger_count = int(spec.get('interdigitated_finger_count', 0))
    if finger_count < 2:
        raise GeometryReject('invalid_interdigitated_graph')
    n = limits.grid_size
    pixel = limits.domain_mm / min(n, options.get('physics_grid_size', n))
    same_gap = max(limits.minimum_gap_mm,
                   float(options.get('minimum_same_polarity_gap_mm', limits.minimum_gap_mm)))
    d, m = limits.domain_mm, limits.margin_mm
    component_gap = same_gap + 2.2*pixel
    region_h = (d - 2*m - (count-1)*component_gap)/count
    active_w = d - 2*m
    target_component = d*d*limits.target_area_fraction_per_polarity/count

    # Estimate the finished width before placing the immutable centerlines so
    # every boundary and row has enough clearance for width fitting/rasterizing.
    reach_fraction=float(options.get('interdigitated_reach_fraction', .74))
    if not .65<=reach_fraction<=.82:
        raise ValueError('interdigitated_reach_fraction must be between 0.65 and 0.82')
    reach_multipliers=np.asarray(spec.get('interdigitated_reach_multipliers',
                                          np.ones(finger_count)),float)
    if reach_multipliers.shape!=(finger_count,) or np.any(reach_multipliers<.85) or np.any(reach_multipliers>1.12):
        raise GeometryReject('invalid_interdigitated_reach_multipliers')
    nominal_reach=active_w*reach_fraction*float(reach_multipliers.mean())
    nominal_pitch=region_h/finger_count
    nominal_length=(finger_count-1)*nominal_pitch+finger_count*nominal_reach
    provisional_width=target_component/nominal_length
    reserve=max(provisional_width*.58,limits.minimum_width_mm*.55)+pixel*.55
    usable_w=active_w-2*reserve
    usable_h=region_h-2*reserve
    if min(usable_w,usable_h)<=0:
        raise GeometryReject('insufficient_interdigitated_envelope')

    sx,sy=parameters['stretch_x'],parameters['stretch_y']
    reaches=np.minimum(usable_w,usable_w*reach_fraction*sx*reach_multipliers)
    # Build an alternating A/C row sequence. Randomized positive gap weights
    # create distinct embeddings while preserving the alternating order.
    spacing=np.asarray(spec.get('interdigitated_spacing_weights',
                                np.ones(2*finger_count-1)),float)
    if spacing.shape!=(2*finger_count-1,) or np.any(spacing<=0):
        raise GeometryReject('invalid_interdigitated_spacing')
    row_margin=.08*usable_h
    row_span=usable_h-2*row_margin
    increments=spacing/spacing.sum()*row_span
    alternating=np.concatenate([[row_margin],row_margin+np.cumsum(increments)])
    first=str(spec.get('interdigitated_first_polarity','anode'))
    if first not in ('anode','cathode'):
        raise GeometryReject('invalid_interdigitated_first_polarity')
    rows_by_polarity={first:alternating[0::2],
                      'cathode' if first=='anode' else 'anode':alternating[1::2]}
    bend_scale=float(options.get('interdigitated_curve_scale',.2))
    if not 0<=bend_scale<=1:
        raise ValueError('interdigitated_curve_scale must be between 0 and 1')
    result={'anode':[],'cathode':[]}
    orientation=int(spec.get('interdigitated_orientation_degrees',0))
    if orientation not in (0,90,180,270):
        raise GeometryReject('invalid_interdigitated_orientation')
    theta=np.deg2rad(orientation)
    rotation=np.array([[np.cos(theta),-np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
    domain_center=np.array([d/2,d/2])
    for ci in range(count):
        bottom=m+ci*(region_h+component_gap)+reserve
        polarity_rows={p:bottom+np.asarray(rows_by_polarity[p],float)
                       for p in ('anode','cathode')}
        if ci%2:
            polarity_rows={'anode':2*(bottom+usable_h/2)-polarity_rows['anode'][::-1],
                           'cathode':2*(bottom+usable_h/2)-polarity_rows['cathode'][::-1]}
        for pi,polarity in enumerate(('anode','cathode')):
            rows=np.asarray(polarity_rows[polarity],float)
            rows=(rows-rows.mean())*sy+rows.mean()+parameters['shift_y']*pixel
            left=m+reserve+parameters['shift_x']*pixel
            right=d-m-reserve+parameters['shift_x']*pixel
            if pi==0:
                spine_x=left
                tip_x=np.minimum(left+reaches,right)
            else:
                spine_x=right
                tip_x=np.maximum(right-reaches,left)
            # A small shear changes the finger endpoints but never the graph.
            centered=(rows-(bottom+usable_h/2))/max(usable_h,1e-12)
            tips=tip_x+parameters['shear']*usable_w*centered
            coordinates=np.asarray(spec['nodes'],float)
            row_indices=np.rint(coordinates[:,1]).astype(int)
            if np.any(row_indices<0) or np.any(row_indices>=finger_count):
                raise GeometryReject('invalid_interdigitated_node_row')
            x_fraction=coordinates[:,0]
            if np.any(x_fraction<0) or np.any(x_fraction>1):
                raise GeometryReject('invalid_interdigitated_node_x')
            point_x=spine_x+(tips[row_indices]-spine_x)*x_fraction
            points=np.column_stack([point_x,rows[row_indices]])
            if orientation:
                points=(points-domain_center)@rotation.T+domain_center
            paths=[_curve(points[e['u']],points[e['v']],e['primitive'],
                          parameters['bend']*bend_scale,1 if i%2 else -1)
                   for i,e in enumerate(spec['edges'])]
            result[polarity].append({'points':points,'paths':paths})
    return result


def _parametric_centerlines(spec, parameters, limits, options):
    """Realize true sinusoidal, Archimedean-spiral, and circular paths."""
    family=str(spec.get('shape_family','')).lower()
    if family not in ('wave','spiral','circle'):
        raise GeometryReject('unknown_shape_family',{'shape_family':family})
    count=spec['component_pair'][0]
    segment_count=len(spec['edges'])
    if segment_count<4:
        raise GeometryReject('invalid_parametric_graph')
    n=limits.grid_size
    pixel=limits.domain_mm/min(n,options.get('physics_grid_size',n))
    same_gap=max(limits.minimum_gap_mm,
                 float(options.get('minimum_same_polarity_gap_mm',limits.minimum_gap_mm)))
    split_gap=max(limits.minimum_gap_mm,same_gap)+2.2*pixel
    d,m=limits.domain_mm,limits.margin_mm
    region_w=(d-2*m-split_gap)/2
    region_h=(d-2*m-(count-1)*split_gap)/count
    # Reserve enough room for the fitted stroke while still using the tall,
    # narrow polarity window. Circle/spiral therefore become ellipses when the
    # available window is rectangular; they remain exact parametric curves.
    reserve=min(limits.maximum_width_mm*.35,
                min(region_w,region_h)*.16)+pixel*.5
    inner_w,inner_h=region_w-2*reserve,region_h-2*reserve
    if min(inner_w,inner_h)<=0:
        raise GeometryReject('insufficient_parametric_envelope')

    def raw_path(t):
        t=np.asarray(t,float)
        if family=='wave':
            cycles=float(parameters['wave_cycles'])
            phase=float(parameters['wave_phase'])
            amplitude=inner_w*float(parameters['wave_amplitude_fraction'])
            values=np.column_stack([amplitude*np.sin(2*np.pi*cycles*t+phase),
                                    (t-.5)*inner_h*.94])
            tilt=float(parameters['wave_tilt'])
            matrix=np.array([[np.cos(tilt),-np.sin(tilt)],
                             [np.sin(tilt), np.cos(tilt)]])
            values=values@matrix.T
            values[:,0]+=inner_w*float(parameters['wave_center_x_fraction'])
            return values
        if family=='spiral':
            turns=float(parameters['spiral_turns'])
            chirality=int(spec.get('parametric_chirality',1))
            theta=chirality*2*np.pi*turns*t+float(parameters['spiral_rotation'])
            outer_x=.47*inner_w
            outer_y=.47*inner_h
            inner=float(parameters['spiral_inner_fraction'])
            fraction=inner+(1-inner)*t**float(parameters['spiral_radial_power'])
            values=np.column_stack([outer_x*fraction*np.cos(theta),
                                    outer_y*fraction*np.sin(theta)])
            values[:,1]+=inner_h*float(parameters['spiral_center_y_fraction'])
            return values
        theta=2*np.pi*t
        radius_fraction=float(parameters['circle_radius_fraction'])
        aspect=float(parameters['circle_aspect_ratio'])
        values=np.column_stack([inner_w*radius_fraction*np.cos(theta),
                                inner_h*radius_fraction*aspect*np.sin(theta)])
        rotation=float(parameters['circle_rotation'])
        matrix=np.array([[np.cos(rotation),-np.sin(rotation)],
                         [np.sin(rotation), np.cos(rotation)]])
        values=values@matrix.T
        values[:,1]+=inner_h*float(parameters['circle_center_y_fraction'])
        return values

    closed=family=='circle'
    node_t=np.arange(segment_count,dtype=float)/segment_count
    breakpoints=np.linspace(0,1,segment_count+1)
    if not closed:
        if family=='spiral':
            # More intervals near the longer outer sweep keeps every fixed
            # graph edge inside the same physical segment-length limits.
            breakpoints=breakpoints**.70
        node_t=breakpoints
    base_points=raw_path(node_t)
    base_paths=[]
    for i in range(segment_count):
        start=breakpoints[i]
        stop=breakpoints[i+1]
        # One PHIDL segment per exact parametric sample interval. Round node
        # disks join the chords into a robust curve without offset cusps.
        base_paths.append(raw_path([start,stop]))

    sx,sy=float(parameters['stretch_x']),float(parameters['stretch_y'])
    shear=float(parameters['shear'])
    result={'anode':[],'cathode':[]}
    for pi,polarity in enumerate(('anode','cathode')):
        for ci in range(count):
            center=np.array([m+region_w/2+pi*(region_w+split_gap),
                             m+region_h/2+ci*(region_h+split_gap)])
            def transform(values):
                values=np.asarray(values,float).copy()
                values[:,0]*=sx
                values[:,1]*=sy
                values[:,0]+=shear*values[:,1]
                if pi:
                    values[:,0]*=-1
                if ci%2:
                    values[:,1]*=-1
                values += center+[parameters['shift_x']*pixel,
                                  parameters['shift_y']*pixel]
                return values
            result[polarity].append({'points':transform(base_points),
                                     'paths':[transform(path) for path in base_paths]})
    return result


def _centerlines(spec, parameters, limits, options):
    style=spec.get('placement_style','separated')
    if style=='interdigitated':
        return _interdigitated_centerlines(spec,parameters,limits,options)
    if style!='separated':
        raise GeometryReject('unknown_placement_style',{'placement_style':style})
    if spec.get('shape_family','lattice') in ('wave','spiral','circle'):
        return _parametric_centerlines(spec,parameters,limits,options)
    return _separated_centerlines(spec,parameters,limits,options)


def _interdigitation_metrics(lines, spec):
    """Measure active-finger projection overlap and A/C row alternation."""
    finger_indices=[i for i,e in enumerate(spec['edges']) if e.get('role','').startswith('finger')]
    if not finger_indices:
        return {'opposed_active_edge_fraction':0.0,'row_alternation_fraction':0.0,
                'finger_count_per_component':0}
    overlap_scores=[]
    alternation_scores=[]
    orientation=int(spec.get('interdigitated_orientation_degrees',0))
    span_axis=0 if orientation%180==0 else 1
    row_axis=1-span_axis
    for ci in range(len(lines['anode'])):
        rows=[]
        spans={}
        for polarity in ('anode','cathode'):
            grouped={}
            for i in finger_indices:
                key=int(spec['edges'][i]['finger_index'])
                grouped.setdefault(key,[]).append(lines[polarity][ci]['paths'][i])
            merged=[np.concatenate(grouped[key],axis=0) for key in sorted(grouped)]
            spans[polarity]=[(float(p[:,span_axis].min()),float(p[:,span_axis].max())) for p in merged]
            rows.extend((float(p[:,row_axis].mean()),polarity) for p in merged)
        for polarity,other in (('anode','cathode'),('cathode','anode')):
            for lo,hi in spans[polarity]:
                length=max(hi-lo,1e-12)
                best=max((max(0.0,min(hi,o_hi)-max(lo,o_lo))/length
                          for o_lo,o_hi in spans[other]),default=0.0)
                overlap_scores.append(best)
        labels=[label for _,label in sorted(rows)]
        transitions=sum(a!=b for a,b in zip(labels,labels[1:]))
        alternation_scores.append(transitions/max(len(labels)-1,1))
    overlap=float(np.mean(overlap_scores)) if overlap_scores else 0.0
    alternation=float(np.mean(alternation_scores)) if alternation_scores else 0.0
    return {'opposed_active_edge_fraction':min(overlap,alternation),
            'finger_projection_overlap_fraction':overlap,
            'row_alternation_fraction':alternation,
            'finger_count_per_component':len({spec['edges'][i]['finger_index'] for i in finger_indices}),
            'orientation_degrees':orientation}


def _cad_component(component, width, *, with_device=False):
    device = Device('doe_component')
    pieces, strokes = [], []
    for path in component['paths']:
        cell = PhidlPath(path).extrude(width=width)
        device.add_ref(cell)
        stroke = unary_union([Polygon(p) for p in cell.get_polygons()])
        strokes.append(stroke)
        pieces.append(stroke)
    angle = np.linspace(0, 2*np.pi, 65)[:-1]
    ring = np.stack([np.cos(angle), np.sin(angle)], axis=1)*width/2
    for p in component['points']:
        disk = ring+p
        device.add_polygon(disk)
        pieces.append(Polygon(disk))
    return unary_union(pieces), strokes, device if with_device else None


def _raster(shape, n, domain):
    prepare(shape)
    x = (np.arange(n)+.5)*domain/n
    y = domain-(np.arange(n)+.5)*domain/n
    xx, yy = np.meshgrid(x,y)
    return np.ascontiguousarray(contains_xy(shape, xx, yy), dtype=bool)


def realize(spec, parameters, limits: GeometryLimits, physics_grid_size: int, options=None):
    options = dict(options or {}, physics_grid_size=int(physics_grid_size))
    pair = tuple(spec['component_pair'])
    if pair not in ((1,1),(2,2)):
        raise GeometryReject('asymmetric_component_pair', {'component_pair': pair})
    if pair[0]>limits.maximum_components_per_polarity or sum(pair)>limits.maximum_total_components:
        raise GeometryReject('component_limits')
    lines = _centerlines(spec, parameters, limits, options)
    segment_lengths=[float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
                     for polarity in ('anode','cathode')
                     for component in lines[polarity] for path in component['paths']]
    minimum_segment=float(limits.minimum_segment_mm)
    maximum_segment=float(limits.maximum_segment_mm)
    if min(segment_lengths)+1e-10<minimum_segment:
        raise GeometryReject('centerline_segment_too_short',
                             {'minimum_observed_mm':min(segment_lengths),
                              'minimum_allowed_mm':minimum_segment})
    if max(segment_lengths)-1e-10>maximum_segment:
        raise GeometryReject('centerline_segment_too_long',
                             {'maximum_observed_mm':max(segment_lengths),
                              'maximum_allowed_mm':maximum_segment})
    interdigitation=None
    if spec.get('placement_style')=='interdigitated':
        interdigitation=_interdigitation_metrics(lines,spec)
        if bool(options.get('require_opposed_active_edges',False)):
            minimum=float(options.get('minimum_opposed_edge_fraction',.45))
            if not 0<=minimum<=1:
                raise ValueError('minimum_opposed_edge_fraction must be between 0 and 1')
            if interdigitation['opposed_active_edge_fraction']+1e-12<minimum:
                raise GeometryReject('insufficient_opposed_active_edges',
                                     {**interdigitation,'minimum_fraction':minimum})
    target = limits.domain_mm**2 * limits.target_area_fraction_per_polarity
    expected_one = graph_statistics(spec)
    expected = {key: value*pair[0] for key,value in expected_one.items()}
    shapes, widths, adjustments, cad_results, polygon_arrays = {}, {}, {}, {}, {}
    max_adjust = float(options.get('maximum_width_adjustment_fraction', .15))
    if not 0 <= max_adjust <= .20:
        raise ValueError('maximum_width_adjustment_fraction must be between 0 and 0.20')
    same_gap = max(limits.minimum_gap_mm, float(options.get('minimum_same_polarity_gap_mm', limits.minimum_gap_mm)))
    for polarity in ('anode','cathode'):
        length = sum(sum(np.linalg.norm(np.diff(path,axis=0),axis=1)) for c in lines[polarity] for path in c['paths'])
        initial_width = target/length
        lo = max(limits.minimum_width_mm, initial_width*(1-max_adjust))
        hi = min(limits.maximum_width_mm, initial_width*(1+max_adjust))
        if lo>hi:
            raise GeometryReject('required_width_out_of_range', {'initial_width_mm': initial_width})
        # Width alone is adjustable. All node positions/curves are immutable.
        def build(w):
            triples = [_cad_component(c,w) for c in lines[polarity]]
            return unary_union([v[0] for v in triples]), triples
        shape_lo, _ = build(lo)
        shape_hi, _ = build(hi)
        if not shape_lo.area <= target <= shape_hi.area:
            raise GeometryReject('modest_width_adjustment_cannot_match_area', {'initial_width_mm': initial_width, 'area_interval': [shape_lo.area,shape_hi.area], 'target':target})
        for _ in range(10):
            mid = (lo+hi)/2
            shape, triples = build(mid)
            if shape.area < target:
                lo=mid
            else:
                hi=mid
        width = (lo+hi)/2
        shape, triples = build(width)
        if abs(shape.area-target)/target > limits.area_tolerance_fraction:
            raise GeometryReject('cad_area_tolerance', {'area_mm2':shape.area, 'target_mm2':target})
        components = _components(shape)
        if len(components)!=pair[0]:
            raise GeometryReject('cad_component_count')
        if sum(len(p.interiors) for p in components)!=expected['loops']:
            raise GeometryReject('cad_loop_collapse', {'expected':expected['loops'], 'observed':sum(len(p.interiors) for p in components)})
        if not box(limits.margin_mm,limits.margin_mm,limits.domain_mm-limits.margin_mm,limits.domain_mm-limits.margin_mm).covers(shape):
            raise GeometryReject('cad_boundary_margin')
        for ci,(component_shape, strokes,_) in enumerate(triples):
            points=lines[polarity][ci]['points']
            strokes=[unary_union([stroke,Point(points[e['u']]).buffer(width/2),Point(points[e['v']]).buffer(width/2)]) for stroke,e in zip(strokes,spec['edges'])]
            for i,e in enumerate(spec['edges']):
                for j,f in enumerate(spec['edges'][i+1:],start=i+1):
                    if {e['u'],e['v']} & {f['u'],f['v']}:
                        continue
                    if spec.get('shape_family') in ('wave','spiral','circle'):
                        separation=j-i
                        # A closed/turning parametric path is one continuous
                        # conductor. Its CAD and both raster graph checks below
                        # catch real self-contact without treating nearby chord
                        # subdivisions as separate same-polarity branches.
                        local_limit=(len(spec['edges'])
                                     if spec.get('shape_family') in ('spiral','circle') else 4)
                        if separation<=local_limit:
                            continue
                    gap=strokes[i].distance(strokes[j])
                    certified_raster_gap = gap - 2*limits.domain_mm/min(limits.grid_size,physics_grid_size)
                    if certified_raster_gap+1e-8 < same_gap:
                        raise GeometryReject('cad_unwanted_self_contact_or_gap', {'gap_mm':gap,'minimum_mm':same_gap,'certified_raster_gap_mm':certified_raster_gap,'edge_indices':[i,j]})
            for other,_,_ in triples[ci+1:]:
                if component_shape.distance(other)+1e-8 < same_gap:
                    raise GeometryReject('cad_same_polarity_component_gap')
        shapes[polarity]=shape
        widths[polarity]=width
        adjustments[polarity]=width/initial_width-1
        polygon_arrays[polarity]=[{'exterior':np.asarray(p.exterior.coords).tolist(),
                                  'holes':[np.asarray(r.coords).tolist() for r in p.interiors]} for p in components]
        cad_results[polarity]={'area_mm2':shape.area,'components':len(components),
                               'loops':sum(len(p.interiors) for p in components),
                               'width_mm':width, 'initial_width_mm':initial_width,
                               'centerline_length_mm':length, 'width_adjustment_fraction':adjustments[polarity]}
    opposite_gap=shapes['anode'].distance(shapes['cathode'])
    if shapes['anode'].intersects(shapes['cathode']) or opposite_gap+1e-8 < limits.minimum_gap_mm:
        raise GeometryReject('cad_opposite_gap', {'gap_mm':opposite_gap})
    masks, grid_results = {}, {}
    for name,n in (('design',limits.grid_size),('physics',physics_grid_size)):
        masks[name]={p:_raster(s,n,limits.domain_mm) for p,s in shapes.items()}
        a,c=masks[name]['anode'],masks[name]['cathode']
        pixel=limits.domain_mm/n
        if np.any(a & c):
            raise GeometryReject(f'{name}_overlap')
        gap=minimum_clear_gap_pixels(a,c)*pixel
        if gap+1e-8<limits.minimum_gap_mm:
            raise GeometryReject(f'{name}_opposite_gap',{'gap_mm':gap})
        results={}
        for p,mask in masks[name].items():
            area=float(mask.sum())*pixel**2
            if abs(area-target)/target>limits.area_tolerance_fraction:
                raise GeometryReject(f'{name}_area_tolerance',{'polarity':p,'area_mm2':area,'target_mm2':target})
            ys,xs=np.nonzero(mask)
            if min(xs.min()*pixel,(n-1-xs.max())*pixel,ys.min()*pixel,(n-1-ys.max())*pixel)+1e-9<limits.margin_mm:
                raise GeometryReject(f'{name}_boundary_margin')
            stats=validate_topology_mask(mask,expected,pixel,widths[p],stage=f'{name}_{p}')
            labels,nc=ndimage.label(mask)
            for i in range(1,nc+1):
                for j in range(i+1,nc+1):
                    if minimum_clear_gap_pixels(labels==i,labels==j)*pixel+1e-8<same_gap:
                        raise GeometryReject(f'{name}_same_polarity_gap')
            results[p]={**stats,'area_mm2':area,'area_relative_error':abs(area-target)/target}
        grid_results[name]={'grid_size':int(n),'opposite_gap_mm':gap,'polarities':results}
    from ecsp_nsga2.bootstrap import validate_solver_grid
    component_metadata={p:{'components':[{} for _ in range(pair[i])]} for i,p in enumerate(('anode','cathode'))}
    solver_grid_report=validate_solver_grid(masks['physics']['anode'], masks['physics']['cathode'], component_metadata, limits, physics_grid_size)
    if solver_grid_report['violations']:
        raise GeometryReject('existing_solver_grid_contract',solver_grid_report)
    physics_mask_hash=hashlib.sha256(masks['physics']['anode'].tobytes()+masks['physics']['cathode'].tobytes()).hexdigest()
    canonical=json.dumps({'spec':spec, 'parameters':parameters, 'polygons':polygon_arrays},sort_keys=True,separators=(',',':'))
    digest=hashlib.sha256(canonical.encode()).hexdigest()
    for name in ('design','physics'):
        digest=hashlib.sha256(digest.encode()+masks[name]['anode'].tobytes()+masks[name]['cathode'].tobytes()).hexdigest()
    metadata={'topology_id':spec['topology_id'],'topology_graph_signature':spec['topology_graph_signature'],
              'component_pair':list(pair),'parameter_vector':dict(parameters),'generator_seed':int(spec.get('generator_seed',0)),
              'geometry_hash':digest,'physics_mask_hash':physics_mask_hash,'topology_spec':spec,'generator':'phidl_direct_graph_v1',
              'placement_style':spec.get('placement_style','separated'),
              'shape_family':spec.get('shape_family','lattice'),
              'electrode_areas_mm2':{p:s.area for p,s in shapes.items()},
              'validation':{'passed':True,'cad':cad_results,'cad_opposite_gap_mm':opposite_gap,
                            'centerline_segment_length_range_mm':[min(segment_lengths),max(segment_lengths)],
                            'expected_graph':expected,'rasters':grid_results,'existing_solver_grid':solver_grid_report},
              'centerline_geometry_frozen_during_width_adjustment':True}
    if interdigitation is not None:
        metadata['interdigitation']=interdigitation
    return GeometryCandidate(masks['physics']['anode'],masks['physics']['cathode'],
                             masks['design']['anode'],masks['design']['cathode'],metadata,polygon_arrays)
