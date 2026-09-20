from __future__ import annotations

import numpy as np

from ecsp_nsga2.nsga2 import (
    Individual,
    environmental_selection,
    fast_non_dominated_sort,
    select_knee_by_utopia_distance,
)


def ind(name: str, values: list[float], violation: float = 0.0) -> Individual:
    return Individual(
        geometry_id=name,
        genome={},
        objectives=np.asarray(values, dtype=float),
        constraint_violation=violation,
        topology_id=name.split("_")[0],
    )


def test_non_dominated_sort_and_constraint_domination() -> None:
    a = ind("T0_A", [1.0, 1.0, 1.0, 1.0])
    b = ind("T1_B", [2.0, 2.0, 2.0, 2.0])
    c = ind("T2_C", [0.5, 5.0, 0.5, 5.0])
    infeasible = ind("T3_I", [0.0, 0.0, 0.0, 0.0], violation=1.0)
    fronts = fast_non_dominated_sort([a, b, c, infeasible])
    assert {x.geometry_id for x in fronts[0]} == {"T0_A", "T2_C"}
    assert b.rank > a.rank
    assert infeasible.rank > a.rank


def test_elitist_environmental_selection_and_knee() -> None:
    population = [
        ind("T0_A", [0.1, 0.9, 0.3, 0.7]),
        ind("T1_B", [0.9, 0.1, 0.7, 0.3]),
        ind("T2_C", [0.45, 0.45, 0.45, 0.45]),
        ind("T3_D", [0.8, 0.8, 0.8, 0.8]),
    ]
    selected = environmental_selection(population, 3)
    assert len(selected) == 3
    assert "T3_D" not in {x.geometry_id for x in selected}
    knee = select_knee_by_utopia_distance(selected)
    assert knee.geometry_id in {x.geometry_id for x in selected}
    assert np.isfinite(knee.metrics["utopia_distance"])
