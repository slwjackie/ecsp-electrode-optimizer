"""Deterministic four-objective Pareto selection, with no weighted score."""
from __future__ import annotations

from typing import Mapping, Sequence
import numpy as np

OBJECTIVES = (
    "ignition_delay_s",
    "area_undecomposed_fraction_at_evaluation_time",
    "minimum_ignition_voltage_V",
    "current_congestion",
)


def rank_designs(rows: Sequence[Mapping]) -> list[dict]:
    """Return rank (one based), front-local crowding and global utopia distance.

    Input IDs are sorted first, making tied crowding boundaries independent of
    input/file order. A constant objective contributes zero crowding distance.
    Failed evaluations are never converted into made-up objective values.
    """
    ordered = sorted((dict(row) for row in rows), key=lambda r: r["geometry_id"])
    if not ordered:
        return []
    if len({r["geometry_id"] for r in ordered}) != len(ordered):
        raise ValueError("Duplicate geometry_id in selection input")
    if any(r.get("status", "success") != "success" for r in ordered):
        raise ValueError("Selection requires successful real physics results")
    values = np.array([[float(r[n]) for n in OBJECTIVES] for r in ordered])
    if not np.isfinite(values).all():
        raise ValueError("Selection requires finite production objective values")
    # dominates[i,j] means i is no worse in every objective, better in at least one.
    dominates = (values[:, None, :] <= values[None, :, :]).all(axis=2)
    dominates &= (values[:, None, :] < values[None, :, :]).any(axis=2)
    incoming = dominates.sum(axis=0)
    pending = np.ones(len(ordered), dtype=bool)
    rank = 1
    span = np.ptp(values, axis=0)
    distances = np.linalg.norm((values - values.min(axis=0)) / np.where(span > 0, span, 1), axis=1)
    while pending.any():
        front = np.flatnonzero(pending & (incoming == 0))
        if not len(front):
            raise RuntimeError("Non-dominated sorting did not progress")
        crowding = np.zeros(len(front))
        if len(front) <= 2:
            crowding[:] = np.inf
        else:
            for column in range(len(OBJECTIVES)):
                order = np.argsort(values[front, column], kind="stable")
                column_values = values[front[order], column]
                width = column_values[-1] - column_values[0]
                if width <= 0:
                    continue
                crowding[order[[0, -1]]] = np.inf
                crowding[order[1:-1]] += (column_values[2:] - column_values[:-2]) / width
        for local, index in enumerate(front):
            ordered[index].update(pareto_rank=rank, crowding_distance=float(crowding[local]),
                                  normalized_utopia_distance=float(distances[index]))
        pending[front] = False
        incoming -= dominates[front].sum(axis=0)
        rank += 1
    return sorted(ordered, key=selection_key)


def selection_key(row: Mapping) -> tuple:
    return (int(row["pareto_rank"]), -float(row["crowding_distance"]),
            float(row["normalized_utopia_distance"]), str(row["geometry_id"]))


def select_topologies(rows: Sequence[Mapping], count: int) -> list[dict]:
    if count < 1:
        raise ValueError("Topology selection count must be positive")
    selected, seen = [], set()
    for row in rank_designs(rows):
        if row["topology_id"] not in seen:
            seen.add(row["topology_id"])
            selected.append(row)
            if len(selected) == count:
                return selected
    raise ValueError(f"Need {count} successfully evaluated unique topologies; found {len(selected)}")


def select_final(rows: Sequence[Mapping], count: int) -> list[dict]:
    if count < 1 or len(rows) < count:
        raise ValueError(f"Cannot select {count} designs from {len(rows)} results")
    # Ordering by front first includes each complete earlier front. The last
    # partially included front is filled by crowding, utopia distance, then ID.
    return rank_designs(rows)[:count]
