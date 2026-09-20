from __future__ import annotations

from pathlib import Path
import random

import numpy as np

from ecsp_nsga2.geometry import (
    GeometryLimits,
    crossover_genomes,
    instantiate_variant,
    make_topology_templates,
    mutate_genome,
    rasterize_and_validate,
    save_geometry,
)


def _pair(genome):
    return (
        len(genome["anode"]["components"]),
        len(genome["cathode"]["components"]),
    )


def test_bootstrap_templates_cover_requested_component_pairs():
    limits = GeometryLimits()
    templates = make_topology_templates(20, 20260828, limits)
    pairs = {_pair(genome) for genome in templates}
    assert {(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (1, 3)} <= pairs
    assert all(na <= 3 and nc <= 3 and na + nc <= 4 for na, nc in pairs)


def test_components_have_independent_poses_and_surface_contact_area():
    limits = GeometryLimits()
    template = next(
        genome
        for genome in make_topology_templates(20, 20260828, limits)
        if _pair(genome) == (2, 2)
    )
    genome = instantiate_variant(template, 0, 101, limits)
    for polarity in ("anode", "cathode"):
        starts = [tuple(root["start"]) for root in genome[polarity]["components"]]
        headings = [float(root["initial_heading_deg"]) for root in genome[polarity]["components"]]
        assert len(set(starts)) == len(starts)
        assert len(headings) == len(starts)
    raster = rasterize_and_validate(genome, limits)
    assert raster.descriptors["propellant_domain_area_fraction"] == 1.0
    assert abs(raster.descriptors["anode_area_fraction"] - 0.175) <= 0.012
    assert abs(raster.descriptors["cathode_area_fraction"] - 0.175) <= 0.012


def test_component_mutations_obey_caps_and_change_counts():
    limits = GeometryLimits()
    base = instantiate_variant(
        make_topology_templates(1, 17, limits)[0], 0, 19, limits
    )
    added = mutate_genome(base, random.Random(1), limits, operation="add_component")
    assert sum(_pair(added)) == 3
    split = mutate_genome(base, random.Random(2), limits, operation="split_component")
    assert sum(_pair(split)) == 3
    removed = mutate_genome(added, random.Random(3), limits, operation="remove_component")
    assert sum(_pair(removed)) == 2
    merged = mutate_genome(added, random.Random(4), limits, operation="merge_component")
    assert sum(_pair(merged)) == 2
    for genome in (added, split, removed, merged):
        na, nc = _pair(genome)
        assert 1 <= na <= 3
        assert 1 <= nc <= 3
        assert na + nc <= 4


def test_component_level_crossover_obeys_caps():
    limits = GeometryLimits()
    templates = make_topology_templates(20, 20260828, limits)
    first = instantiate_variant(
        next(g for g in templates if _pair(g) == (1, 2)), 0, 7, limits
    )
    second = instantiate_variant(
        next(g for g in templates if _pair(g) == (3, 1)), 0, 11, limits
    )
    for seed in range(20):
        child = crossover_genomes(first, second, random.Random(seed), limits)
        na, nc = _pair(child)
        assert 1 <= na <= 3 and 1 <= nc <= 3 and na + nc <= 4


def test_saved_geometry_contains_full_propellant_overlay_masks(tmp_path: Path):
    limits = GeometryLimits()
    genome = instantiate_variant(
        make_topology_templates(1, 31, limits)[0], 0, 37, limits
    )
    raster = rasterize_and_validate(genome, limits)
    genome["geometry_id"] = "surface_case"
    save_geometry(genome, raster, tmp_path)
    data = np.load(tmp_path / "surface_case.npz")
    np.testing.assert_array_equal(data["anode_contact_mask"], data["anode_mask"])
    np.testing.assert_array_equal(data["cathode_contact_mask"], data["cathode_mask"])
    assert np.all(data["propellant_domain_mask"] == 1)
