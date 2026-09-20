from __future__ import annotations

from typing import Sequence
import numpy as np

from .nsga2 import Individual


DESCRIPTOR_NAMES = (
    "n_line",
    "n_arc",
    "n_branch",
    "branch_depth",
    "n_components",
    "length_mean",
    "length_std",
    "arc_radius_mean",
    "orientation_entropy",
    "minimum_gap_mm",
    "perimeter_px",
    "spatial_dispersion",
)


def descriptor_vector(ind: Individual) -> np.ndarray:
    d = ind.metrics.get("geometry_descriptors", {})
    return np.asarray([float(d.get(k, 0.0)) for k in DESCRIPTOR_NAMES], dtype=float)


def robust_standardize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    med = np.nanmedian(matrix, axis=0)
    q25 = np.nanpercentile(matrix, 25, axis=0)
    q75 = np.nanpercentile(matrix, 75, axis=0)
    scale = q75 - q25
    std = np.nanstd(matrix, axis=0)
    scale = np.where(scale > 1e-12, scale, np.where(std > 1e-12, std, 1.0))
    return (matrix - med) / scale


def farthest_point_selection(
    candidates: Sequence[Individual],
    already_selected: Sequence[Individual],
    count: int,
) -> list[Individual]:
    """Greedy max-min (farthest-point) geometry selection.

    A candidate is scored by its minimum standardised descriptor distance to
    the current selected set. The farthest candidate is added at each step.
    """

    if count <= 0 or not candidates:
        return []
    all_items = list(already_selected) + list(candidates)
    x = robust_standardize(np.vstack([descriptor_vector(i) for i in all_items]))
    n_seed = len(already_selected)
    remaining = list(range(n_seed, len(all_items)))
    chosen_idx: list[int] = []
    reference = list(range(n_seed))

    if not reference:
        # Start from the point farthest from the descriptor centroid.
        centroid = np.mean(x[remaining], axis=0)
        first = max(remaining, key=lambda i: (np.linalg.norm(x[i] - centroid), -i))
        chosen_idx.append(first)
        remaining.remove(first)
        reference.append(first)

    while remaining and len(chosen_idx) < count:
        def min_distance(i: int) -> float:
            return min(float(np.linalg.norm(x[i] - x[j])) for j in reference)

        best = max(remaining, key=lambda i: (min_distance(i), -i))
        chosen_idx.append(best)
        remaining.remove(best)
        reference.append(best)

    return [all_items[i] for i in chosen_idx]


def quota_preserving_mating_pool(
    population: Sequence[Individual],
    performance_count: int,
    topology_count: int,
    diversity_count: int,
) -> list[Individual]:
    """Build the 70 + 20 + 10 mating pool requested for this project.

    All individuals have already received Electrical+Solid evaluations. This
    pool only controls reproduction; it is not a destructive evaluation filter.
    Environmental selection remains textbook elitist NSGA-II over the full
    parent+offspring population.
    """

    ordered = sorted(
        population,
        key=lambda x: (x.rank, -x.crowding_distance, x.geometry_id),
    )
    selected: list[Individual] = ordered[:performance_count]
    selected_ids = {x.geometry_id for x in selected}

    # One best representative for each topology not yet represented.
    by_topology: dict[str, list[Individual]] = {}
    for ind in ordered:
        by_topology.setdefault(ind.topology_id, []).append(ind)
    topo_picks: list[Individual] = []
    for topo in sorted(by_topology):
        pick = by_topology[topo][0]
        if pick.geometry_id not in selected_ids:
            topo_picks.append(pick)
            selected_ids.add(pick.geometry_id)
    selected.extend(topo_picks[:topology_count])

    remaining = [x for x in population if x.geometry_id not in selected_ids]
    diverse = farthest_point_selection(remaining, selected, diversity_count)
    selected.extend(diverse)
    selected_ids.update(x.geometry_id for x in diverse)

    # If quotas overlap, top up by rank/crowding so the pool size is stable.
    target = min(len(population), performance_count + topology_count + diversity_count)
    for ind in ordered:
        if len(selected) >= target:
            break
        if ind.geometry_id not in selected_ids:
            selected.append(ind)
            selected_ids.add(ind.geometry_id)
    return selected
