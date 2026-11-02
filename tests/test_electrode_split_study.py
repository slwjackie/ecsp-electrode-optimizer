"""Screenshot geometry and opt-in workflow checks; no physical results are mocked as data."""
from __future__ import annotations

import copy
import csv
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import run_electrode_parameter_study as study

SPLIT_ID = "SPLIT_W4_G2_S1"
PAIR_IDS = ["WD_W4", SPLIT_ID]


@pytest.fixture
def split_spec():
    return yaml.safe_load((ROOT / "config/electrode_split_study_25mm.yaml").read_text())


@pytest.fixture
def split_project(tmp_path):
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    base = {
        "geometry": {"domain_mm": 20., "grid_size": 96,
                     "target_area_fraction_per_polarity": .175},
        "evaluator": {"base_overrides": {"numerics": {"physicsDtype": "float64"}},
                      "cpp_cuda": {"cpu_workers": 6}},
        "physics": {"voltage_V": 260., "composition": {"sentinel": 1}},
        "preflame_model": {"timeStep_s": .00025, "onsetCriterion": {"sentinel": 7}},
        "minimum_ignition_voltage_search": {"enabled": True},
    }
    physics = root / "config/physics.yaml"
    physics.write_text(yaml.safe_dump(base))
    return root, physics, base


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def component_sizes(mask):
    """Independent four-neighbor flood fill; rectangles cannot touch diagonally."""
    remaining = set(zip(*np.nonzero(mask)))
    sizes = []
    while remaining:
        pending = [remaining.pop()]
        size = 0
        while pending:
            row, col = pending.pop()
            size += 1
            for point in ((row - 1, col), (row + 1, col),
                          (row, col - 1), (row, col + 1)):
                if point in remaining:
                    remaining.remove(point)
                    pending.append(point)
        sizes.append(size)
    return sorted(sizes)


@pytest.mark.parametrize("n", [100, 200, 400])
def test_screenshot_dimensions_polarities_and_area_on_refined_grids(n):
    a, c, meta = study.make_case(SPLIT_ID, n)
    h = 25. / n
    centers = (np.arange(n) + .5) * h
    x, y = centers[None, :], centers[:, None]
    left = (x > 7.5) & (x < 11.5)
    right = (x > 13.5) & (x < 17.5)
    top, bottom = y < 12., y > 13.
    # Analytic dimensions come directly from the requested four 4 x 12 mm pieces.
    assert np.array_equal(a, (left & top) | (right & bottom))
    assert np.array_equal(c, (right & top) | (left & bottom))
    assert not (a & c).any()
    assert a.dtype == c.dtype == np.bool_
    assert a.sum() * h**2 == pytest.approx(96.)
    assert c.sum() * h**2 == pytest.approx(96.)
    assert meta["total_electrode_area_mm2"] == 192.
    assert meta["central_split_gap_mm"] == 1.
    assert meta["segment_length_mm"] == 12.
    assert meta["intended_anode_components"] == meta["intended_cathode_components"] == 2
    assert len(meta["anode_rectangles_mm"]) == len(meta["cathode_rectangles_mm"]) == 2
    study.validate_item(a, c, meta, n)


def test_four_separate_equal_patches_and_four_percent_contact_loss():
    a, c, meta = study.make_case(SPLIT_ID, 200)
    assert component_sizes(a) == component_sizes(c) == [3072, 3072]
    old_a, old_c, _ = study.make_case("WD_W4", 200)
    assert (old_a | old_c).sum() * .125**2 == 200.
    assert ((old_a | old_c).sum() - (a | c).sum()) * .125**2 == 8.
    assert np.count_nonzero(a | c) / np.count_nonzero(old_a | old_c) == .96
    # Central cut exposes propellant; no metal bridge crosses either gap.
    assert not (a | c)[96:104].any()
    assert not (a | c)[:, 92:108].any()


@pytest.mark.parametrize("mutation", ["same_polarity_below", "bridge", "missing_pixel", "metadata_gap"])
def test_split_geometry_rejects_changed_polarity_contact_or_metadata(mutation):
    a, c, meta = study.make_case(SPLIT_ID, 200)
    if mutation == "same_polarity_below":
        a[104:], c[104:] = c[104:].copy(), a[104:].copy()
    elif mutation == "bridge":
        a[96:104, 60:92] = True
    elif mutation == "missing_pixel":
        a[0, 60] = False
    else:
        meta["central_split_gap_mm"] = 2.
    with pytest.raises(ValueError):
        study.validate_item(a, c, meta, 200)


def test_geometry_batch_uses_central_gap_and_keeps_propellant_under_contacts():
    pytest.importorskip("torch")
    item = (*study.make_case(SPLIT_ID, 200), Path("unused"))
    geom = study.build_geometry([item], 200, "cpu", .001, SimpleNamespace)
    assert geom.propellant.all() and not geom.fixed.any()
    assert geom.domain_size_m == .025 and geom.anode.shape == (1, 200, 200)
    # Although the horizontal separation is 2 mm, a 1.5 mm minimum is violated
    # by the independently specified vertical 1 mm separation.
    with pytest.raises(ValueError):
        study.build_geometry([item], 200, "cpu", .0015, SimpleNamespace)


def test_extra_metadata_cannot_override_unsplit_minimum_gap():
    pytest.importorskip("torch")
    a, c, meta = study.make_case("WD_W4", 200)
    # Legacy metadata allows extension keys, but an unvalidated optional key
    # must never turn a real 2 mm clearance into an advertised 3 mm clearance.
    meta["minimum_edge_gap_mm"] = 3.
    with pytest.raises(ValueError):
        study.build_geometry([(a, c, meta, Path("unused"))], 200, "cpu",
                             .0025, SimpleNamespace)


@pytest.mark.parametrize("mutation", ["unknown_schema", "original_schema", "missing_case", "extra_case", "changed_horizontal_gap"])
def test_split_profile_cannot_silently_change_the_registered_experiment(split_spec, mutation):
    spec = copy.deepcopy(split_spec)
    if mutation == "unknown_schema":
        spec["schema"] = "invented_profile"
    elif mutation == "original_schema":
        spec["schema"] = "full_height_strip_study_v1"
    elif mutation == "missing_case":
        spec["cases"].pop()
    elif mutation == "extra_case":
        original = yaml.safe_load((ROOT / "config/electrode_parameter_study_25mm.yaml").read_text())
        spec["cases"].append(original["cases"][0])
    else:
        next(row for row in spec["cases"] if row["id"] == SPLIT_ID)["gap_mm"] = 3.
    with pytest.raises(ValueError):
        study.validate_spec(spec)


def test_split_is_opt_in_and_existing_eight_case_profile_stays_intact(split_spec):
    assert [row["id"] for row in study.validate_spec(split_spec)["cases"]] == PAIR_IDS
    original = yaml.safe_load((ROOT / "config/electrode_parameter_study_25mm.yaml").read_text())
    assert len(study.validate_spec(original)["cases"]) == 8
    assert len(study.CASES) == 8 and SPLIT_ID not in study.CASES


def test_runtime_component_contract_changes_only_for_split(split_project):
    _, _, base = split_project
    before = copy.deepcopy(base)
    old = study.runtime_config(base, 200, "cpu", 2)
    new = study.runtime_config(base, 200, "cpu", 2, split_study=True)
    assert base == before
    assert old["geometry"]["maximum_components_per_polarity"] == 1
    assert old["geometry"]["maximum_total_components"] == 2
    assert new["geometry"]["maximum_components_per_polarity"] == 2
    assert new["geometry"]["maximum_total_components"] == 4
    assert new["geometry"]["minimum_gap_mm"] == 1.
    for key in set(base) - {"geometry", "evaluator"}:
        assert new[key] == base[key]


def test_generate_two_cases_ten_balanced_experiments_and_audit(tmp_path, split_spec):
    out = tmp_path / "split"
    study.generate(split_spec, out, experiment_seed=42)
    _, items = study.audit(out)
    assert [item[2]["geometry_id"] for item in items] == PAIR_IDS
    schedule = read_csv(out / "experiment_schedule.csv")
    template = read_csv(out / "experiment_measurements_template.csv")
    assert len(schedule) == len(template) == 10
    assert {(row["case_id"], row["replicate"]) for row in schedule} == {
        (row["case_id"], row["replicate"]) for row in template}
    for block in range(1, 6):
        assert {row["case_id"] for row in schedule if row["block"] == str(block)} == set(PAIR_IDS)
    assert study.read_json(out / "GENERATED.json")["cases"] == 2
    for sid in PAIR_IDS:
        with np.load(out / "library" / sid / "mask.npz", allow_pickle=False) as masks:
            assert masks["propellant"].all()
        assert (out / "library" / sid / "geometry.svg").exists()
    with pytest.raises(FileExistsError):
        study.generate(split_spec, out)
    study.summarize_experiments(out / "experiment_measurements_template.csv",
                               out / "experiment_summary.csv", case_ids=PAIR_IDS)
    assert [row["case_id"] for row in read_csv(out / "experiment_summary.csv")] == PAIR_IDS


class RecordingEvaluator:
    """Only tests orchestration and validity reporting, not ignition physics."""
    def __init__(self, cfg):
        self.config = copy.deepcopy(cfg)
        self.calls = []

    def evaluate_batch(self, items):
        self.calls.append([item[2]["geometry_id"] for item in items])
        return [dict(geometry_id=item[2]["geometry_id"], ignitionSucceeded=True,
                     ignitionDelay_s=.25, inputElectricalEnergyToIgnition_J=1.,
                     minimumIgnitionVoltageSearchValid=True,
                     minimumIgnitionVoltageRightCensored=False,
                     minimumIgnitionVoltage_V=120.,
                     peakMaximumTemperature_K=600.) for item in items]

    def _trial_valid(self, row):
        return row["geometry_id"] != SPLIT_ID, "fixture_only"


def test_evaluate_split_profile_resume_and_invalid_summary(split_project, split_spec):
    root, physics, _ = split_project
    out = root / "runs/split"
    study.generate(split_spec, out)
    created = []
    def factory(root, cfg, workdir):
        evaluator = RecordingEvaluator(cfg)
        created.append(evaluator)
        return evaluator
    rows = study.evaluate(root, out, physics, "cpu", 2, False, factory)
    assert [row["geometry_id"] for row in rows] == PAIR_IDS
    assert created[0].calls == [PAIR_IDS]
    assert created[0].config["geometry"]["maximum_components_per_polarity"] == 2
    assert created[0].config["geometry"]["maximum_total_components"] == 4
    assert rows[1]["result_state"] == "numerically_invalid"
    assert rows[1]["ignitionDelay_s"] is None
    assert rows[1]["peakMaximumTemperature_K"] is None
    raw = study.read_json(out / "cases" / SPLIT_ID / "result.json")["raw"]
    assert raw["ignitionDelay_s"] == .25  # Diagnostic raw values are preserved.
    comparison = read_csv(out / "split_comparison_summary.csv")
    assert [row["geometry_id"] for row in comparison] == PAIR_IDS
    assert comparison[1]["ignitionDelay_s"] == ""
    assert study.read_json(out / "RUN_FINISHED.json")["cases"] == 2
    study.evaluate(root, out, physics, "cpu", 2, True, factory)
    assert len(created) == 1 and created[0].calls == [PAIR_IDS]
    with pytest.raises(ValueError):
        study.evaluate(root, out, physics, "cpu", 2, False, factory)
    physics.write_text(physics.read_text() + "\n# changed input\n")
    with pytest.raises(ValueError):
        study.evaluate(root, out, physics, "cpu", 2, True, factory)
