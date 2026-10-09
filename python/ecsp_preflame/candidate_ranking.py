"""Candidate records and Pareto report ordering for a fixed result set.

No reproduction, tournament selection, or evolutionary population update occurs.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from math import inf
from typing import Any, Sequence
import numpy as np

@dataclass
class CandidateEvaluation:
    """One ECSP electrode candidate.

    All objective values must be expressed as minimisation objectives.
    Constraint violations are non-negative; zero means feasible.
    """

    geometry_id: str
    genome: dict[str, Any]
    objectives: np.ndarray | None = None
    constraint_violation: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    rank: int = 10**9
    crowding_distance: float = 0.0
    topology_id: str = ""
    source_role: str = ""

    def is_evaluated(self) -> bool:
        return self.objectives is not None and np.all(np.isfinite(self.objectives))


def _feasible(ind: CandidateEvaluation, eps: float = 1e-12) -> bool:
    return float(ind.constraint_violation) <= eps


def constrained_dominates(a: CandidateEvaluation, b: CandidateEvaluation, eps: float = 1e-12) -> bool:
    """Deb's constraint-domination principle.

    1. Any feasible solution dominates any infeasible solution.
    2. Between infeasible solutions, lower total violation dominates.
    3. Between feasible solutions, ordinary Pareto dominance applies.
    """

    af = _feasible(a, eps)
    bf = _feasible(b, eps)
    if af and not bf:
        return True
    if bf and not af:
        return False
    if not af and not bf:
        return a.constraint_violation < b.constraint_violation - eps

    if a.objectives is None or b.objectives is None:
        raise ValueError("Both individuals must be evaluated before dominance comparison")

    av = np.asarray(a.objectives, dtype=float)
    bv = np.asarray(b.objectives, dtype=float)
    no_worse = np.all(av <= bv + eps)
    strictly_better = np.any(av < bv - eps)
    return bool(no_worse and strictly_better)


def fast_non_dominated_sort(population: Sequence[CandidateEvaluation]) -> list[list[CandidateEvaluation]]:
    """Return Pareto fronts in rank order.

    Complexity is O(MN^2) for M metrics and N already evaluated candidates.
    """

    n = len(population)
    if n == 0:
        return []

    dominates_list: list[list[int]] = [[] for _ in range(n)]
    domination_count = np.zeros(n, dtype=np.int64)
    first: list[int] = []

    for i, p in enumerate(population):
        for j, q in enumerate(population):
            if i == j:
                continue
            if constrained_dominates(p, q):
                dominates_list[i].append(j)
            elif constrained_dominates(q, p):
                domination_count[i] += 1
        if domination_count[i] == 0:
            p.rank = 0
            first.append(i)

    fronts_idx: list[list[int]] = [first]
    f = 0
    while f < len(fronts_idx) and fronts_idx[f]:
        nxt: list[int] = []
        for i in fronts_idx[f]:
            for j in dominates_list[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    population[j].rank = f + 1
                    nxt.append(j)
        if nxt:
            fronts_idx.append(nxt)
        f += 1

    return [[population[i] for i in front] for front in fronts_idx if front]


def assign_crowding_distance(front: Sequence[CandidateEvaluation]) -> None:
    if not front:
        return
    for ind in front:
        ind.crowding_distance = 0.0
    if len(front) <= 2:
        for ind in front:
            ind.crowding_distance = inf
        return

    if front[0].objectives is None:
        raise ValueError("Crowding distance requires evaluated individuals")
    m = len(front[0].objectives)
    for k in range(m):
        ordered = sorted(front, key=lambda x: float(x.objectives[k]))
        lo = float(ordered[0].objectives[k])
        hi = float(ordered[-1].objectives[k])
        span = hi - lo
        if span <= 1e-30:
            continue
        ordered[0].crowding_distance = inf
        ordered[-1].crowding_distance = inf
        for i in range(1, len(ordered) - 1):
            if np.isinf(ordered[i].crowding_distance):
                continue
            prev_v = float(ordered[i - 1].objectives[k])
            next_v = float(ordered[i + 1].objectives[k])
            ordered[i].crowding_distance += (next_v - prev_v) / span


def rank_and_crowd(population: Sequence[CandidateEvaluation]) -> list[list[CandidateEvaluation]]:
    fronts = fast_non_dominated_sort(population)
    for front in fronts:
        assign_crowding_distance(front)
    return fronts


def pareto_front(population: Sequence[CandidateEvaluation], feasible_only: bool = True) -> list[CandidateEvaluation]:
    data = [p for p in population if (not feasible_only or _feasible(p))]
    fronts = rank_and_crowd(data)
    return fronts[0] if fronts else []


def select_knee_by_utopia_distance(front: Sequence[CandidateEvaluation]) -> CandidateEvaluation:
    """Choose one reproducible compromise design from a Pareto front.

    Objectives are robustly min-max normalised. The individual nearest to the
    utopia point is selected. This does not introduce arbitrary physical-unit
    weights and is preferable to an undocumented weighted sum.
    """

    if not front:
        raise ValueError("Cannot select a recommended design from an empty front")
    matrix = np.asarray([p.objectives for p in front], dtype=float)
    lo = np.nanmin(matrix, axis=0)
    hi = np.nanmax(matrix, axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    norm = (matrix - lo) / span
    distances = np.linalg.norm(norm, axis=1)
    best = min(
        range(len(front)),
        key=lambda i: (float(distances[i]), front[i].geometry_id),
    )
    front[best].metrics["utopia_distance"] = float(distances[best])
    return front[best]
