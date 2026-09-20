from __future__ import annotations

import random
from typing import Any

import numpy as np

PRIMITIVES = ("LINE", "ARC", "BEZIER", "WAVE", "SPIRAL")
SYMMETRIC_COMPONENT_PAIRS = ((1, 1), (2, 2))


def _node(rng: random.Random, *, branch: bool = False, first: bool = False) -> dict[str, Any]:
    typ = rng.choices(PRIMITIVES, weights=(0.24, 0.19, 0.20, 0.22, 0.15), k=1)[0]
    turn = 0.0 if first else rng.uniform(-78.0, 78.0)
    return {
        "type": typ,
        "role": "branch" if branch else "generic",
        "base_turn_deg": float(turn),
        "children": [],
    }


def _open_component(rng: random.Random, *, max_branches: int) -> dict[str, Any]:
    main_count = rng.randint(2, 4)
    chain = [_node(rng, first=(i == 0)) for i in range(main_count)]
    for a, b in zip(chain[:-1], chain[1:]):
        a["children"] = [b]

    branch_count = rng.randint(0, min(2, max(0, int(max_branches))))
    hosts = list(chain)
    rng.shuffle(hosts)
    for host in hosts[:branch_count]:
        branch = _node(rng, branch=True)
        branch["base_turn_deg"] = rng.choice((-1.0, 1.0)) * rng.uniform(35.0, 100.0)
        if rng.random() < 0.45:
            tail = _node(rng)
            tail["base_turn_deg"] = rng.uniform(-45.0, 45.0)
            branch["children"] = [tail]
        host.setdefault("children", []).append(branch)
    return chain[0]


def make_open_grammar_template(*, seed: int, topology_index: int, pair: tuple[int, int], limits) -> dict[str, Any]:
    na, nc = pair
    if pair not in SYMMETRIC_COMPONENT_PAIRS:
        raise ValueError(f"Open grammar only accepts symmetric pairs, got {pair}")
    rng = random.Random(int(seed))

    def components(n: int) -> list[dict[str, Any]]:
        return [
            _open_component(
                random.Random(rng.randrange(1 << 62)),
                max_branches=int(getattr(limits, "maximum_branches_per_component", 2)),
            )
            for _ in range(n)
        ]

    return {
        "anode": {"components": components(na)},
        "cathode": {"components": components(nc)},
        "family": "open_grammar",
        "topology_index": int(topology_index),
        "component_count_pair": [int(na), int(nc)],
        "hidden_bus_assumption": (
            "Every disconnected same-polarity surface-contact component is connected "
            "to its terminal by an out-of-plane/backside conductor outside the 2-D model."
        ),
        "electrode_mask_semantics": "surface_contact_overlay",
        "topology_id": f"OPEN_{topology_index:06d}_{na}A{nc}C",
    }


def mask_pair_iou(a1, c1, a2, c2) -> float:
    a1 = np.asarray(a1, dtype=bool); c1 = np.asarray(c1, dtype=bool)
    a2 = np.asarray(a2, dtype=bool); c2 = np.asarray(c2, dtype=bool)
    inter = int(np.count_nonzero(a1 & a2)) + int(np.count_nonzero(c1 & c2))
    union = int(np.count_nonzero(a1 | a2)) + int(np.count_nonzero(c1 | c2))
    return float(inter / union) if union else 1.0


def graph_feature_vector(genome: dict[str, Any]) -> tuple[float, ...]:
    counts = {name: 0 for name in PRIMITIVES}
    branches = 0
    maximum_depth = 0
    nodes = 0

    def walk(node: dict[str, Any], depth: int = 0) -> None:
        nonlocal branches, maximum_depth, nodes
        nodes += 1
        typ = str(node.get("type", "LINE")).upper()
        if typ in counts:
            counts[typ] += 1
        children = node.get("children", [])
        branches += max(0, len(children) - 1)
        maximum_depth = max(maximum_depth, depth)
        for child in children:
            walk(child, depth + 1)

    for polarity in ("anode", "cathode"):
        for root in genome[polarity]["components"]:
            walk(root)

    na = len(genome["anode"]["components"])
    nc = len(genome["cathode"]["components"])
    return tuple(float(counts[k]) for k in PRIMITIVES) + (
        float(branches), float(maximum_depth), float(nodes), float(na), float(nc),
    )


def graph_distance(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    denominator = float(np.sum(np.maximum(np.abs(a), np.abs(b))))
    if denominator <= 1.0e-12:
        return 0.0
    return float(np.sum(np.abs(a - b)) / denominator)
