"""Deterministic, independent scrambled Latin-hypercube sample streams."""
from __future__ import annotations
import hashlib
import json
from scipy.stats import qmc
from .geometry import GeometryReject, realize

PARAMETER_BOUNDS = {
    'stretch_x': (.95,1.0), 'stretch_y': (.95,1.0),
    'bend': (.025,.065), 'shear': (-.018,.018),
    'shift_x': (-.32,.32), 'shift_y': (-.32,.32),
}

SHAPE_PARAMETER_BOUNDS = {
    'wave': {
        'wave_amplitude_fraction': (.16, .34),
        'wave_cycles': (.80, 1.80),
        'wave_phase': (-3.14, 3.14),
        'wave_center_x_fraction': (-.06, .06),
        'wave_tilt': (-.10, .10),
    },
    'spiral': {
        'spiral_turns': (.72, .92),
        'spiral_inner_fraction': (.06, .30),
        'spiral_rotation': (-3.14, 3.14),
        'spiral_radial_power': (.65, 1.40),
        'spiral_center_y_fraction': (-.08, .08),
    },
    'circle': {
        'circle_radius_fraction': (.30, .46),
        'circle_aspect_ratio': (.55, 1.00),
        'circle_rotation': (-.22, .22),
        'circle_center_y_fraction': (-.12, .12),
    },
}


def parameter_bounds(shape_family='lattice'):
    """Return independent LHS dimensions used by one geometry family."""
    family=str(shape_family or 'lattice').strip().lower()
    bounds=dict(PARAMETER_BOUNDS)
    bounds.update(SHAPE_PARAMETER_BOUNDS.get(family, {}))
    return bounds


def derived_seed(project_seed, *labels):
    text=json.dumps([int(project_seed),*labels],sort_keys=True,separators=(',',':'))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4],'little')


def parameter_key(parameters):
    return json.dumps(parameters, sort_keys=True, separators=(',',':'))


def lhs_parameters(count, seed, bounds=None):
    bounds=bounds or PARAMETER_BOUNDS
    sampler=qmc.LatinHypercube(d=len(bounds),seed=int(seed))
    points=sampler.random(n=int(count))
    low=[v[0] for v in bounds.values()]
    high=[v[1] for v in bounds.values()]
    return [dict(zip(bounds, map(float,row))) for row in qmc.scale(points,low,high)]


def generate_variants(spec, limits, count, seed, physics_grid_size, options=None,
                      excluded_parameters=(), rejection_callback=None):
    options=options or {}
    if count <= 0:
        raise ValueError('count must be positive')
    maximum=int(options.get('maximum_attempts_per_topology', max(100, count*40)))
    existing={parameter_key(v) for v in excluded_parameters}
    accepted=[]
    # Excluded Stage 1 points are realized under the same immutable topology;
    # prohibit repeated solver masks even if CAD/parameter hashes differ.
    hashes={realize(spec,p,limits,physics_grid_size,options).metadata['physics_mask_hash'] for p in excluded_parameters}
    stream_seed=derived_seed(seed,'variants',spec['topology_id'],spec['topology_graph_signature'])
    # Stratify each proposal batch over the requested variant count (3 or 10),
    # not over the attempt budget. Rejections are replaced by fresh deterministic
    # LHS batches. Accepted points can lose exact stratification after DRC;
    # every proposed batch remains a true count-point Latin hypercube.
    def proposals():
        for batch in range((maximum + count - 1) // count):
            yield from lhs_parameters(count, derived_seed(stream_seed, 'batch', batch),
                                      spec.get('parameter_space', {}).get('bounds'))

    for attempt,parameters in enumerate(proposals()):
        if attempt >= maximum:
            break
        key=parameter_key(parameters)
        if key in existing:
            if rejection_callback:
                rejection_callback({'topology_id':spec['topology_id'],'attempt':attempt,
                                    'reason':'duplicate_parameter_vector','parameter_vector':parameters})
            continue
        try:
            candidate=realize(spec,parameters,limits,physics_grid_size,options)
            if candidate.metadata['physics_mask_hash'] in hashes:
                raise GeometryReject('duplicate_physics_mask')
        except GeometryReject as exc:
            if rejection_callback:
                rejection_callback({'topology_id':spec['topology_id'],'attempt':attempt,
                                    'reason':exc.reason,'details':exc.details,'parameter_vector':parameters})
            continue
        candidate.metadata['parameter_sampling_seed']=stream_seed
        candidate.metadata['parameter_sample_index']=attempt
        candidate.metadata['generator_seed']=int(seed)
        accepted.append(candidate)
        existing.add(key)
        hashes.add(candidate.metadata['physics_mask_hash'])
        if len(accepted)==count:
            return accepted
    raise GeometryReject('parameter_attempt_limit',{'topology_id':spec['topology_id'],
                          'accepted':len(accepted),'requested':count,'maximum_attempts':maximum})
