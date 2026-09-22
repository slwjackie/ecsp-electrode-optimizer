import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'))
from ecsp_image_design.five_topologies import fit_reference

IDS=('E058','E114','R038','R050','R091')

def refs():
    obj=json.loads((ROOT/'data/image_design/five_topology_references.json').read_text())
    return {r['source_id']:r for r in obj['records']}

def test_all_five_geometries_pass_contract():
    r=refs()
    for sid in IDS:
        m,res=fit_reference(r[sid])
        assert m['status']=='geometry_accepted',m
        assert res is not None
        assert abs(m['anode_area_mm2']-m['cathode_area_mm2'])/m['anode_area_mm2']<1e-6
        assert m['minimum_gap_actual_mm']>=3
        assert m['physics_grid_gap_guard_mm']>=3
        assert m['minimum_width_actual_mm']>=1.998
        assert m['matched_staggered_dimension_witness'] is not None

def test_topology_specific_component_counts():
    r=refs(); expected={'E058':(4,4),'E114':(1,8),'R038':(1,1),'R050':(9,1),'R091':(1,9)}
    for sid,pair in expected.items():
        m,_=fit_reference(r[sid]);assert (m['anode_components'],m['cathode_components'])==pair
