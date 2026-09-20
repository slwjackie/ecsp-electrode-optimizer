"""Connected lattice graph grammar, typed primitive hashing and CAD library."""
from __future__ import annotations
import hashlib
import json
import math
import networkx as nx
import numpy as np
from .geometry import GeometryReject, graph_statistics, realize
from .sampling import derived_seed, lhs_parameters, parameter_bounds


def graph_signature(spec):
    # Insert one typed node per primitive edge; coordinates, parameters, names,
    # graph insertion order and component numbering are intentionally absent.
    graph=nx.Graph()
    for polarity in ('anode','cathode'):
        for ci in range(spec['component_pair'][0 if polarity=='anode' else 1]):
            prefix=f'{polarity}:{ci}:'
            for vi in range(len(spec['nodes'])):
                graph.add_node(prefix+f'v{vi}', kind=polarity+':junction')
            for ei,edge in enumerate(spec['edges']):
                mid=prefix+f'e{ei}'
                # ``role`` is part of the graph grammar (for example a comb
                # spine versus a finger), while coordinates remain an embedding
                # parameter and therefore do not define topology identity.
                kind=polarity+':'+edge['primitive']
                if edge.get('role'):
                    kind+=':'+edge['role']
                graph.add_node(mid,kind=kind)
                graph.add_edge(prefix+f"v{edge['u']}",mid)
                graph.add_edge(mid,prefix+f"v{edge['v']}")
    wl=nx.weisfeiler_lehman_graph_hash(graph,node_attr='kind',iterations=max(4,len(spec['nodes'])),digest_size=32)
    payload={'wl':wl,'component_pair':spec['component_pair'],
             'shape_family':spec.get('shape_family','lattice'),
             'statistics':graph_statistics(spec)}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def grammar_spec(seed, index, component_count):
    rng=np.random.default_rng(derived_seed(seed,'graph',index,component_count))
    cols=3
    rows=int(rng.choice([4,5,6])) if component_count==1 else 3
    lattice=nx.grid_2d_graph(cols,rows)
    # extend/turn: deterministic randomized spanning tree. Branch/fork emerge at
    # tree vertices; parallel-edge insertion creates electrically valid loops.
    for u,v in lattice.edges():
        lattice.edges[u,v]['weight']=float(rng.random())
    tree=nx.minimum_spanning_tree(lattice,weight='weight')
    non_tree=[(u,v) for u,v in lattice.edges() if not tree.has_edge(u,v)]
    rng.shuffle(non_tree)
    additions=int(rng.integers(0,min(4,len(non_tree))+1))
    tree.add_edges_from(non_tree[:additions])
    nodes=sorted(tree.nodes())
    indices={node:i for i,node in enumerate(nodes)}
    primitives=['LINE','ARC','SPLINE']
    edges=[{'u':indices[u],'v':indices[v], 'primitive':str(rng.choice(primitives,p=[.5,.25,.25]))}
           for u,v in sorted(tree.edges())]
    spec={'topology_id':f'proposal_{index:06d}','generator_seed':int(seed),
          'component_pair':[component_count,component_count],
          'nodes':[list(n) for n in nodes], 'edges':edges,
          'shape_family':'lattice',
          'placement_style':'separated',
          'grammar_operations':['extend','turn','branch','fork','mirror']+(['loop','parallel_branch'] if additions else []),
          'parameter_space':{'type':'scrambled_latin_hypercube','geometry':'fixed_graph_variable_embedding',
                             'bounds':{key:list(value) for key,value in parameter_bounds('lattice').items()}}}
    spec['topology_graph_signature']=graph_signature(spec)
    return spec


def interdigitated_grammar_spec(seed, index, component_count):
    """Generate one connected comb graph which is mirrored for each polarity.

    The graph itself contains a spine plus automatically generated fingers.
    Geometry realization places the two polarity copies on opposite sides of a
    shared window and offsets their finger rows, producing an A/C/A/C ordering
    without changing connectivity during area matching.
    """
    rng=np.random.default_rng(derived_seed(seed,'interdigitated_graph',index,component_count))
    # Four rows leave room for the 1A1C target area. Two rows per component
    # leave the same certified clearances for 2A2C. Graph diversity comes from
    # the independently typed, two-segment fingers rather than unsafe crowding.
    finger_count=4 if component_count==1 else 2
    # Nodes 0..N-1 form the vertical spine. Every finger is split into four
    # typed primitives so each realized segment stays within GeometryLimits.
    nodes=[[0.0,float(row)] for row in range(finger_count)]
    edges=[]
    for row in range(finger_count-1):
        # Keeping the spine straight makes its branch points robust on both
        # raster grids. Primitive diversity is supplied by the active fingers.
        edges.append({'u':row,'v':row+1,'primitive':'LINE','role':'spine'})
    primitives=['LINE','ARC','SPLINE']
    for row in range(finger_count):
        fractions=[float(rng.uniform(.20,.28)),float(rng.uniform(.45,.54)),
                   float(rng.uniform(.71,.79)),1.0]
        previous=row
        for segment,fraction in enumerate(fractions):
            node=len(nodes)
            nodes.append([fraction,float(row)])
            primitive=str(rng.choice(primitives,p=[.45,.275,.275]))
            edges.append({'u':previous,'v':node,'primitive':primitive,
                          'role':f'finger_{segment}','finger_index':row})
            previous=node
    # These values define the representative embedding of a generated graph;
    # LHS parameters subsequently create variants without changing the graph.
    # Alternating row gaps stay close enough to uniform for DRC while producing
    # visibly different masks for the representative-IoU gate.
    spacing_count=2*finger_count-1
    spacing_weights=rng.uniform(.82,1.18,size=spacing_count)
    reach_multipliers=rng.uniform(.90,1.08,size=finger_count)
    spec={'topology_id':f'proposal_{index:06d}','generator_seed':int(seed),
          'component_pair':[component_count,component_count],
          'nodes':nodes,'edges':edges,'shape_family':'interdigitated',
          'placement_style':'interdigitated',
          'interdigitated_finger_count':finger_count,
          'interdigitated_orientation_degrees':int(rng.choice([0,90,180,270])),
          'interdigitated_first_polarity':str(rng.choice(['anode','cathode'])),
          'interdigitated_spacing_weights':list(map(float,spacing_weights)),
          'interdigitated_reach_multipliers':list(map(float,reach_multipliers)),
          'grammar_operations':['extend','branch','mirror','interdigitate'],
          'parameter_space':{'type':'scrambled_latin_hypercube','geometry':'fixed_graph_variable_embedding',
                             'bounds':{key:list(value) for key,value in parameter_bounds('lattice').items()}}}
    spec['topology_graph_signature']=graph_signature(spec)
    return spec


def parametric_grammar_spec(seed, index, component_count, shape_family):
    """Create a fixed graph for a true wave, spiral, or closed circle."""
    family=str(shape_family).strip().lower()
    if family not in ('wave','spiral','circle'):
        raise ValueError('shape_family must be wave, spiral, or circle')
    rng=np.random.default_rng(derived_seed(seed,family+'_graph',index,component_count))
    if family=='wave':
        segment_count=int(rng.integers(9,13))
        nodes=[[float(i)/segment_count,0.0] for i in range(segment_count+1)]
        edges=[{'u':i,'v':i+1,'primitive':'SPLINE',
                'role':f"wave_{i}_{str(rng.choice(['rise','fall','turn']))}"}
               for i in range(segment_count)]
        operations=['extend','wave','mirror']
    elif family=='spiral':
        segment_count=(int(rng.integers(9,12)) if component_count==1
                       else int(rng.integers(6,9)))
        nodes=[[float(i)/segment_count,0.0] for i in range(segment_count+1)]
        edges=[{'u':i,'v':i+1,'primitive':'ARC',
                'role':f"spiral_{i}_{str(rng.choice(['inner','sweep','outer']))}"}
               for i in range(segment_count)]
        operations=['extend','turn','spiral','mirror']
    else:
        segment_count=int(rng.integers(8,13))
        nodes=[[math.cos(2*math.pi*i/segment_count),
                math.sin(2*math.pi*i/segment_count)] for i in range(segment_count)]
        sector_types=rng.choice(['minor','major'],size=segment_count,p=[.7,.3])
        edges=[{'u':i,'v':(i+1)%segment_count,'primitive':'ARC',
                'role':f"circle_{i}_{sector_types[i]}"}
               for i in range(segment_count)]
        operations=['extend','turn','loop','circle','mirror']
    spec={'topology_id':f'proposal_{index:06d}','generator_seed':int(seed),
          'component_pair':[component_count,component_count],
          'nodes':nodes,'edges':edges,'shape_family':family,
          'placement_style':'separated','grammar_operations':operations,
          'parameter_space':{'type':'scrambled_latin_hypercube',
                             'geometry':'fixed_graph_variable_embedding',
                             'bounds':{key:list(value) for key,value in parameter_bounds(family).items()}}}
    if family=='spiral':
        spec['parametric_chirality']=int(rng.choice([-1,1]))
    spec['topology_graph_signature']=graph_signature(spec)
    return spec


def _quota_counts(total, weights, order):
    """Largest-remainder allocation with stable tie breaking and exact total."""
    if total==0:
        return {key:0 for key in order}
    unknown=set(weights)-set(order)
    if unknown:
        raise ValueError(f"unknown shape families: {sorted(unknown)}")
    values={key:float(weights.get(key,0.0)) for key in order}
    if any(value<0 for value in values.values()) or sum(values.values())<=0:
        raise ValueError('shape_family_fractions must be nonnegative and have a positive sum')
    scale=total/sum(values.values())
    raw={key:values[key]*scale for key in order}
    result={key:int(np.floor(raw[key])) for key in order}
    remainder=total-sum(result.values())
    ranked=sorted(order,key=lambda key:(-(raw[key]-result[key]),order.index(key)))
    for key in ranked[:remainder]:
        result[key]+=1
    return result


def _target_sequence(count, n_one, n_interdigitated, shape_fractions=None):
    """Return deterministic component/family targets with exact quotas."""
    if count <= 0:
        return []
    family_order=['interdigitated','lattice','wave','spiral','circle']
    separated=count-n_interdigitated
    fractions=shape_fractions or {'lattice':1.0}
    quotas=_quota_counts(separated,fractions,family_order[1:])
    quotas['interdigitated']=n_interdigitated
    families=[]
    while len(families)<count:
        progressed=False
        for family in family_order:
            if quotas[family]>0:
                families.append(family)
                quotas[family]-=1
                progressed=True
        if not progressed:
            break
    curved=sum(family in ('wave','spiral','circle') for family in families)
    if curved>n_one:
        raise ValueError('wave/spiral/circle quota exceeds the available 1A1C quota; increase fraction_1a1c or reduce curved-family fractions')
    remaining_one=n_one-curved
    targets=[]
    for family in families:
        if family in ('wave','spiral','circle'):
            components=1
        elif remaining_one:
            components=1
            remaining_one-=1
        else:
            components=2
        targets.append((components,family))
    return targets


def representative_iou(a,c,b,d):
    union=np.count_nonzero(a|b)+np.count_nonzero(c|d)
    return (np.count_nonzero(a&b)+np.count_nonzero(c&d))/union if union else 1.0


def generate_topology_library(limits, count, seed, physics_grid_size, options=None,
                              rejection_callback=None):
    options=options or {}
    if count <= 0:
        raise ValueError('count must be positive')
    ratio=float(options.get('fraction_1a1c', .5))
    if not 0<=ratio<=1:
        raise ValueError('fraction_1a1c must be between 0 and 1')
    n_one=int(round(count*ratio))
    if count-n_one and (limits.maximum_components_per_polarity<2 or limits.maximum_total_components<4):
        raise ValueError('requested 2A2C ratio exceeds GeometryLimits component policy')
    threshold=float(options.get('representative_iou_threshold', .9))
    if not 0<threshold<1:
        raise ValueError('representative_iou_threshold must be in (0,1)')
    maximum=int(options.get('maximum_topology_attempts', count*100))
    style=str(options.get('topology_style','separated')).strip().lower()
    if style not in ('separated','interdigitated','hybrid_interdigitated'):
        raise ValueError('topology_style must be separated, interdigitated, or hybrid_interdigitated')
    fraction=float(options.get('interdigitated_fraction', .5))
    if not 0<=fraction<=1:
        raise ValueError('interdigitated_fraction must be between 0 and 1')
    n_interdigitated=(count if style=='interdigitated' else
                      int(round(count*fraction)) if style=='hybrid_interdigitated' else 0)
    shape_fractions=options.get('shape_family_fractions',{'lattice':1.0})
    if not isinstance(shape_fractions,dict):
        raise ValueError('shape_family_fractions must be a mapping')
    targets=_target_sequence(count,n_one,n_interdigitated,shape_fractions)
    specs=[]
    representatives=[]
    signatures=set()
    all_attempts=[]
    for index in range(maximum):
        desired,family=targets[len(specs)]
        if family=='interdigitated':
            spec=interdigitated_grammar_spec(seed,index,desired)
        elif family=='lattice':
            spec=grammar_spec(seed,index,desired)
        else:
            spec=parametric_grammar_spec(seed,index,desired,family)
        reason=None
        details={}
        if spec['topology_graph_signature'] in signatures:
            reason='duplicate_graph_signature'
        else:
            parameters=lhs_parameters(1,derived_seed(seed,'representative',index),
                                      spec.get('parameter_space',{}).get('bounds'))[0]
            try:
                candidate=realize(spec,parameters,limits,physics_grid_size,options)
                overlaps=[representative_iou(candidate.design_anode_mask,candidate.design_cathode_mask,
                                              a,c) for a,c in representatives]
                if overlaps and max(overlaps)>threshold:
                    raise GeometryReject('representative_iou',{'maximum_iou':max(overlaps),'threshold':threshold})
            except GeometryReject as exc:
                reason=exc.reason
                details=exc.details
        if reason:
            row={'topology_id':spec['topology_id'],'attempt':index,
                 'accepted_count':len(specs),'shape_family':spec.get('shape_family'),
                 'component_pair':spec.get('component_pair'),
                 'reason':reason,'details':details}
            if rejection_callback:
                rejection_callback(row)
            all_attempts.append(row)
            continue
        spec['topology_id']=f'T{len(specs):03d}'
        spec['representative_parameters']=parameters
        spec['representative_validation']=candidate.metadata['validation']
        spec['maximum_representative_iou']=max(overlaps,default=0.0)
        specs.append(spec)
        signatures.add(spec['topology_graph_signature'])
        representatives.append((candidate.design_anode_mask,candidate.design_cathode_mask))
        if len(specs)==count:
            return specs
    from collections import Counter
    raise GeometryReject('topology_attempt_limit',{'accepted':len(specs),'requested':count,
                        'attempts':maximum,'rejection_counts':dict(Counter(x['reason'] for x in all_attempts))})
