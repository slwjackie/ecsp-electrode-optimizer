"""Tests for the additive study, without compiling or running production physics."""
from __future__ import annotations
import copy
import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace, ModuleType
import sys

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import run_electrode_parameter_study as study


@pytest.fixture
def spec():
    return yaml.safe_load((ROOT / "config/electrode_parameter_study_25mm.yaml").read_text())


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    base = {
        "geometry": {"domain_mm": 20, "grid_size": 96, "target_area_fraction_per_polarity": .175},
        "evaluator": {"base_overrides": {"numerics": {"physicsDtype": "float64"}},
                      "cpp_cuda": {"cpu_workers": 6}},
        "physics": {"voltage_V": 260, "end_time_s": 2, "composition": {"sentinel": 1}},
        "preflame_model": {"timeStep_s": .00025, "onsetCriterion": {"sentinel": 7}},
        "minimum_ignition_voltage_search": {"enabled": True},
        "condensed_ignition": {"onset_temperature_K": 523.15},
        "evaluation": {"algorithm": "untouched"},
        "baselines": {"area_matched_staggered": {"enabled": True}},
        "post_onset": {"sentinel": "untouched"},
    }
    physics = root / "config/physics.yaml"
    physics.write_text(yaml.safe_dump(base))
    return root, physics, base


@pytest.mark.parametrize("sid", list(study.CASES))
def test_exact_full_height_geometry(sid):
    a, c, m = study.make_case(sid, 200)
    study.validate_item(a, c, m, 200)
    assert a.shape == c.shape == (200, 200)
    assert a.dtype == c.dtype == np.bool_
    assert np.array_equal(a[0], a[-1]) and np.array_equal(c[0], c[-1])
    assert np.all(a == a[0]) and np.all(c == c[0])
    assert not (a & c).any()
    wa, wc, gap, _ = study.CASES[sid]
    assert a.sum() * .125**2 == wa * 25
    assert c.sum() * .125**2 == wc * 25
    ax, cx = np.where(a[0])[0], np.where(c[0])[0]
    assert (cx.min() - ax.max() - 1) * .125 == gap
    assert m["target_anode_area_fraction"] == wa / 25
    assert m["target_cathode_area_fraction"] == wc / 25


@pytest.mark.parametrize("n", [100, 150, 200, 400])
def test_grid_refinement_preserves_physical_design(n):
    for sid in study.CASES:
        a, c, m = study.make_case(sid, n)
        assert a.sum() * (25 / n)**2 == pytest.approx(m["anode_area_mm2"])
        study.validate_item(a, c, m, n)


@pytest.mark.parametrize("n", [True, 0, 99, 193, 201, 200.0])
def test_off_grid_geometry_is_not_silently_resized(n):
    with pytest.raises(ValueError):
        study.make_case("WD_W1", n)


def test_ratio_cases_mirror_and_constant_total_area():
    a, c, m = study.make_case("AR_1TO2", 200)
    aa, cc, mm = study.make_case("AR_2TO1", 200)
    assert np.array_equal(a, np.fliplr(cc))
    assert np.array_equal(c, np.fliplr(aa))
    for sid in ("AR_1TO2", "AR_1TO1", "AR_2TO1"):
        assert study.make_case(sid, 200)[2]["total_electrode_area_mm2"] == 150


def test_fixed_matrix_and_distinct_baselines(spec):
    validated = study.validate_spec(spec)
    assert len(validated["cases"]) == 8
    hashes = {study.digest([a, c]) for a, c, _ in
              (study.make_case(sid, 200) for sid in study.CASES)}
    assert len(hashes) == 8
    assert study.CASES["BASE_G2_W2"][3] == ("spacing", "width")
    assert study.CASES["AR_1TO1"][:2] == (3., 3.)
    spec["cases"][-1]["anode_width_mm"] = 3.
    with pytest.raises(ValueError):
        study.validate_spec(spec)


@pytest.mark.parametrize("mutation", ["pixel", "polarity", "length", "target", "role", "dtype"])
def test_bad_contact_never_reaches_physics(mutation):
    a, c, m = study.make_case("AR_1TO2", 200)
    if mutation == "pixel":
        a[0, np.where(a[0])[0][0]] = False
    elif mutation == "polarity":
        a, c = c, a
    elif mutation == "length":
        m["electrode_length_mm"] = 20.
    elif mutation == "target":
        m["target_anode_area_fraction"] = .16
    elif mutation == "role":
        m["source_role"] = "fixed_library_screening"
    else:
        a = a.astype(float)
    with pytest.raises(ValueError):
        study.validate_item(a, c, m, 200)


def test_generate_audit_template_and_no_overwrite(tmp_path, spec):
    out = tmp_path / "run"
    study.generate(spec, out)
    got, items = study.audit(out)
    assert got == study.validate_spec(spec) and len(items) == 8
    with (out / "experiment_measurements_template.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 40 and all(not r["ignition_delay_s"] for r in rows)
    with pytest.raises(FileExistsError):
        study.generate(spec, out)
    (out / "library/AR_1TO2/mask.npz").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        study.audit(out)


def test_runtime_does_not_modify_shared_physics(project):
    _, _, base = project
    before = copy.deepcopy(base)
    cfg = study.runtime_config(base, 200, "cpu", 8)
    assert base == before
    for key in set(base) - {"geometry", "evaluator"}:
        assert cfg[key] == base[key]
    assert cfg["evaluator"]["backend"] == "preflame_cpp_cuda"
    assert cfg["geometry"]["minimum_gap_mm"] == 1
    assert cfg["geometry"]["minimum_width_mm"] == 1
    assert cfg["evaluator"]["save_representative_fields"] is True


def test_geometry_batch_overlay_and_unequal_areas():
    torch = pytest.importorskip("torch")
    items = [(*study.make_case(sid, 200), Path("unused")) for sid in study.CASES]
    geometry = study.build_geometry(items, 200, "cpu", .001, SimpleNamespace)
    assert geometry.anode.dtype == torch.bool
    assert geometry.propellant.all() and not geometry.fixed.any()
    assert geometry.anode.shape == (8, 200, 200)
    assert geometry.domain_size_m == .025
    assert geometry.anode[5].sum() * 2 == geometry.cathode[5].sum()
    with pytest.raises(ValueError):
        study.build_geometry([items[0], items[0]], 200, "cpu", .001, SimpleNamespace)


def test_subclass_overrides_only_geometry(monkeypatch):
    pytest.importorskip("torch")
    class Native:
        def __init__(self, root, config, workdir):
            self.grid_size = config["grid_size"]
            self.domain_size_m = .025
            self.minimum_gap_m = .001
            self.device = "cpu"
        def evaluate_batch(self, items):
            return "inherited integration"
        def _trial_valid(self, row):
            return True, "inherited validity"
    for name in ("ecsp_preflame", "ecsp_preflame.cpp_cuda_evaluator", "ecsp_preflame.evaluator",
                 "ecsp_v6", "ecsp_v6.physics", "ecsp_v6.physics.geometry"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["ecsp_preflame.cpp_cuda_evaluator"].CppCudaPreflameEvaluator = Native
    sys.modules["ecsp_preflame.evaluator"].PreflameCandidateGeometryError = ValueError
    sys.modules["ecsp_v6.physics.geometry"].GeometryBatch = SimpleNamespace
    ev = study.create_study_evaluator(Path("."), {"geometry": {"grid_size": 200},
              "evaluator": {"grid_size": 200}}, Path("unused"))
    assert ev.__class__.evaluate_batch is Native.evaluate_batch
    assert ev.__class__._trial_valid is Native._trial_valid
    geometry = ev._build_geometry_batch([(*study.make_case("AR_1TO2", 200), Path("unused"))])
    assert geometry.propellant.all()


class FakeEvaluator:
    """Orchestration fixture, never described as a physical solver."""
    calls = 0
    config = {"fixture_only": True}
    def evaluate_batch(self, items):
        type(self).calls += 1
        return [dict(geometry_id=i[2]["geometry_id"], ignitionSucceeded=False,
                     ignitionDelay_s=None, minimumIgnitionVoltageSearchValid=True,
                     minimumIgnitionVoltageRightCensored=True,
                     minimumIgnitionVoltage_V=None) for i in items]
    def _trial_valid(self, row):
        return True, "fixture_only"


def test_evaluate_resume_and_cache_invalidation(project, spec):
    root, physics, _ = project
    out = root / "runs/study"
    study.generate(spec, out)
    FakeEvaluator.calls = 0
    factory = lambda *_: FakeEvaluator()
    rows = study.evaluate(root, out, physics, "cpu", 2, False, factory)
    assert len(rows) == 8 and FakeEvaluator.calls == 4
    assert all(r["result_state"] == "no_onset_within_horizon" for r in rows)
    assert all(r["ignitionDelay_s"] is None for r in rows)
    assert study.read_json(out / "sources_before.json") == study.read_json(out / "sources_after.json")
    study.evaluate(root, out, physics, "cpu", 2, True, factory)
    assert FakeEvaluator.calls == 4
    for group in ("spacing", "width", "area_ratio"):
        with (out / f"{group}_summary.csv").open() as stream:
            assert len(list(csv.DictReader(stream))) == 3
    with pytest.raises(ValueError, match="Existing/changed"):
        study.evaluate(root, out, physics, "cpu", 2, False, factory)
    physics.write_text(physics.read_text() + "\n# changed input\n")
    with pytest.raises(ValueError, match="Existing/changed"):
        study.evaluate(root, out, physics, "cpu", 2, True, factory)


def test_execution_failure_is_not_nonignition(project, spec):
    root, physics, _ = project
    out = root / "runs/study"
    study.generate(spec, out)
    def broken(*args):
        raise RuntimeError("compiler unavailable")
    with pytest.raises(RuntimeError, match="compiler unavailable"):
        study.evaluate(root, out, physics, "cpu", 8, False, broken)
    assert (out / "execution_failure.json").exists()
    assert not (out / "RUN_FINISHED.json").exists()
    assert not (out / "parameter_study_summary.csv").exists()


def test_raw_penalties_are_not_reported_as_delays():
    meta = study.make_case("WD_W1", 200)[2]
    raw = dict(ignitionSucceeded=False, ignitionDelay_s=1e12,
               minimumIgnitionVoltage_V=1e12, external_numerical_valid=False)
    rows = study.summary_rows([dict(geometry=meta, raw=raw)])
    assert rows[0]["result_state"] == "numerically_invalid"
    assert rows[0]["ignitionDelay_s"] is None
    assert rows[0]["minimumIgnitionVoltage_V"] is None
    assert raw["ignitionDelay_s"] == 1e12


def test_experimental_statistics_preserve_censoring(tmp_path):
    source, out = tmp_path / "measurements.csv", tmp_path / "summary.csv"
    rows = [dict(case_id="SP_G1", replicate=i+1, ignition_observed=status,
                 ignition_delay_s=delay, observation_window_s=2)
            for i, (status, delay) in enumerate([(True, .2), (True, .4), (False, "")])]
    study.write_csv(source, rows)
    study.summarize_experiments(source, out)
    with out.open() as stream:
        first = next(csv.DictReader(stream))
    assert int(first["n_observed"]) == 3 and int(first["n_no_ignition"]) == 1
    assert float(first["mean_delay_ignited_only_s"]) == pytest.approx(.3)
    assert float(first["sample_sd_delay_ignited_only_s"]) == pytest.approx(.2 / 2**.5)
    rows[-1]["ignition_delay_s"] = 2
    study.write_csv(source, rows)
    with pytest.raises(ValueError, match="censored"):
        study.summarize_experiments(source, out)
