from __future__ import annotations

import numpy as np

from ecsp_nsga2.diversity import quota_preserving_mating_pool
from ecsp_nsga2.nsga2 import Individual, rank_and_crowd


DESCRIPTOR_KEYS = {
    "n_line": 1.0,
    "n_arc": 0.0,
    "n_branch": 0.0,
    "branch_depth": 0.0,
    "component_count": 2.0,
    "mean_segment_length_mm": 2.0,
    "std_segment_length_mm": 0.1,
    "mean_arc_radius_mm": 0.0,
    "orientation_entropy": 0.1,
    "minimum_gap_mm": 0.5,
    "perimeter_px": 50.0,
    "spatial_dispersion": 0.2,
}


def make(topology: str, idx: int, offset: float) -> Individual:
    obj = np.asarray([1.0 + offset, 0.5 + offset, 10.0 + offset, 2.0 + offset])
    candidate = Individual(
        geometry_id=f"{topology}_{idx}",
        genome={},
        objectives=obj,
        topology_id=topology,
    )
    descriptors = dict(DESCRIPTOR_KEYS)
    descriptors["n_arc"] = float(idx % 3)
    descriptors["n_branch"] = float(idx % 2)
    descriptors["spatial_dispersion"] = offset
    candidate.metrics["geometry_descriptors"] = descriptors
    return candidate


def test_quota_pool_has_target_size_and_topology_coverage() -> None:
    population = []
    for t in range(4):
        for i in range(10):
            population.append(make(f"T{t}", i, 0.01 * (t * 10 + i)))
    rank_and_crowd(population)
    pool = quota_preserving_mating_pool(
        population,
        performance_count=28,
        topology_count=4,
        diversity_count=8,
    )
    assert len(pool) == 40
    assert {p.topology_id for p in pool} == {"T0", "T1", "T2", "T3"}
    assert len({p.geometry_id for p in pool}) == len(pool)
