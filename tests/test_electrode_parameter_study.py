"""Unit/mocked-orchestration checks; no test here claims a validated ignition result."""
from __future__ import annotations
import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import run_electrode_parameter_study as doe


@pytest.mark.parametrize("n", [100, 200, 400])
@pytest.mark.parametrize("sid", list(doe.CASES))
def test_exact_physical_rectangles(sid, n):
    study = doe.Study(grid_size=n)
    a, c, meta = doe.make_case(sid, study)
    audit = doe.validate_item(a, c, meta, study)
    wa, wc, gap, _ = doe.CASES[sid]
    assert audit["passed"] and a.dtype == np.bool_
    assert a.shape == c.shape == (n, n)
    assert a.sum() * (25 / n)**2 == wa * 25
    assert c.sum() * (25 / n)**2 == wc * 25
    assert not (a & c).any()
    assert np.array_equal(a[0], a[-1]) and np.array_equal(c[0], c[-1])
    ai, ci = np.where(a[0])[0], np.where(c[0])[0]
    assert (ci[0]-ai[-1]-1)*25/n == gap
    assert ai[0] == n-1-ci[-1]


@pytest.mark.parametrize("kwargs", [{"grid_size": 201}, {"grid_size": True},
    {"grid_size": 200.0}, {"replicates": 4}, {"batch_size": 0},
    {"reference_voltage_V": float("nan")}, {"end_time_s": 0}])
def test_invalid_settings_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        doe.Study(**kwargs)


def test_area_ratio_mirror_and_constant_total():
    study = doe.Study()
    a, c, _ = doe.make_case("AR_1TO2", study)
    ar, cr, _ = doe.make_case("AR_2TO1", study)
    assert np.array_equal(a, np.fliplr(cr))
    assert np.array_equal(c, np.fliplr(ar))
    for sid in ("AR_1TO2", "AR_1TO1", "AR_2TO1"):
        _, _, meta = doe.make_case(sid, study)
        assert meta["total_electrode_area_mm2"] == 150
    assert doe.CASES["BASE_G2_W2"][3] == ("spacing", "width")


@pytest.mark.parametrize("mutation", ["pixel", "swap", "shift", "crop", "dtype", "target", "id", "component", "reference"])
def test_strict_contract_rejects_tampering(mutation):
    study = doe.Study()
    a, c, m = doe.make_case("AR_1TO2", study)
    if mutation == "pixel": a[0, 0] = True
    elif mutation == "swap": a, c = c, a
    elif mutation == "shift": a = np.roll(a, 1, axis=1)
    elif mutation == "crop": a = a[:-1]
    elif mutation == "dtype": a = a.astype(float)
    elif mutation == "target": m["target_anode_area_fraction"] = .16
    elif mutation == "id": m["geometry_id"] = "E001"
    elif mutation == "component": m["intended_anode_components"] = 2
    else: m["baseline_type"] = "area_matched_staggered"
    with pytest.raises(ValueError): doe.validate_item(a, c, m, study)


def test_generation_and_audit(tmp_path):
    study = doe.Study()
    out = tmp_path / "out"
    manifest = doe.generate(study, out)
    assert manifest["case_count"] == 8
    assert doe.audit(study, out)["all_passed"]
    with (out / "experiment_template.csv").open() as f:
        template = list(csv.DictReader(f))
    assert len(template) == 40
    assert len({(r["case_id"], r["replicate"]) for r in template}) == 40
    assert all(not r["ignition_delay_s"] for r in template)
    with (out / "experiment_schedule.csv").open() as f:
        schedule = list(csv.DictReader(f))
    for block in range(1, 6):
        assert {r["case_id"] for r in schedule if r["block"] == str(block)} == set(doe.CASES)
    with pytest.raises(ValueError): doe.generate(study, out)
    with pytest.raises(ValueError): doe.audit(doe.Study(grid_size=400), out)
    (out / "library" / "SP_G1" / "metadata.json").write_text("{}")
    with pytest.raises(ValueError): doe.audit(study, out)


def minimal_base():
    return dict(geometry={"domain_mm":20, "target_area_fraction_per_polarity":.175},
        physics={"voltage_V":260., "end_time_s":2., "metric_evaluation_time_s":2.},
        bc_global={"endTime_s":2., "evaluationTime_s":2., "timeStep_s":.00025},
        evaluator={"voltage_V":260., "end_time_s":2., "native":{}, "base_overrides":{}},
        minimum_ignition_voltage_search={"enabled":True, "upper_bound_V":260.},
        optimization={"original":True}, baselines={"original":True},
        post_onset={"original":True})


def test_runtime_preserves_base_and_all_physics_blocks():
    base = minimal_base()
    old = copy.deepcopy(base)
    cfg = doe.runtime_config(base, doe.Study(), "cpu")
    assert base == old
    for k in set(base) - {"geometry", "evaluator"}: assert cfg[k] == base[k]
    assert cfg["geometry"]["domain_mm"] == 25
    assert cfg["evaluator"]["grid_size"] == 200
    assert cfg["evaluator"]["native"]["cpu_workers"] == 0
    base["physics"]["voltage_V"] = 200
    with pytest.raises(ValueError): doe.runtime_config(base, doe.Study(), "cpu")


def test_actual_upstream_revision_guard_fails_closed(tmp_path):
    for relative in doe.UPSTREAM_BLOBS:
        p = tmp_path / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# not the reviewed revision\n")
    with pytest.raises(ValueError, match="Unsupported upstream"):
        doe.verify_upstream(tmp_path)


def test_mixin_returns_real_masks_and_full_propellant(monkeypatch, tmp_path):
    # This is explicitly a TENSOR-BUILDER TEST DOUBLE, not the production PDE.
    torch = pytest.importorskip("torch")
    calls = []
    class Direct:
        def _build_geometry_batch(self, items):
            calls.append(items)
            a = torch.as_tensor(np.stack([i[0] for i in items]))
            c = torch.as_tensor(np.stack([i[1] for i in items]))
            return types.SimpleNamespace(anode=a, cathode=c, fixed=a|c,
                propellant=~(a|c), geometry_ids=[i[2]["geometry_id"] for i in items])
    fake = types.ModuleType("ecsp_nsga2.evaluator")
    fake.DirectCondensedV772NoFEvaluator = Direct
    monkeypatch.setitem(sys.modules, "ecsp_nsga2.evaluator", fake)
    probe = doe.FullHeightStripGeometryMixin()
    probe.grid_size, probe.domain_size_m = 200, .025
    items = [(*doe.make_case(sid, doe.Study()), tmp_path/sid) for sid in doe.CASES]
    result = probe._build_geometry_batch(items)
    assert len(calls) == 1
    assert result.propellant.all() and not result.fixed.any()
    assert np.array_equal(result.anode[5].numpy(), items[5][0])
    assert result.anode[5].sum() != result.cathode[5].sum()


class FakeEvaluator:
    """Mock for checkpoint/status tests; results have no physical meaning."""
    def __init__(self, cfg, calls):
        self.grid_size = cfg["evaluator"]["grid_size"]
        self.domain_size_m, self.voltage = .025, 260.
        self.device = cfg["evaluator"]["device"]
        self.config = {"test_double":True}
        self.calls = calls
    def evaluate_batch(self, items):
        self.calls.extend(i[2]["geometry_id"] for i in items)
        return [dict(geometry_id=i[2]["geometry_id"], ignitionSucceeded=False,
            ignitionDelay_s=float("nan"), converged=True,
            allElectricalLinearSolvesConverged=True, allNonlinearRobinSolvesConverged=True,
            minimumIgnitionVoltageSearchValid=True, minimumIgnitionVoltageRightCensored=True,
            minimumIgnitionVoltage_V=None) for i in items]
    def _trial_valid(self, row): return True, "valid"


def setup_run(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    p = root / "config" / "base.yaml"
    p.write_text(yaml.safe_dump(minimal_base()))
    calls = []
    monkeypatch.setattr(doe, "create_study_evaluator", lambda root, cfg, folder: FakeEvaluator(cfg, calls))
    return root, p, root / "runs" / "study", calls


def test_mocked_run_and_resume_no_duplicate_simulations(tmp_path, monkeypatch):
    root, config, out, calls = setup_run(tmp_path, monkeypatch)
    study = doe.Study()
    doe.evaluate(study, root, config, out, device="cpu", only=["SP_G1", "AR_1TO2"])
    assert len(calls) == 2
    doe.evaluate(study, root, config, out, device="cpu", resume=True)
    assert len(calls) == 8 and set(calls) == set(doe.CASES)
    doe.evaluate(study, root, config, out, device="cpu", resume=True)
    assert len(calls) == 8
    assert doe.read_json(out / "source_preservation.json")["unchanged"]
    for group in ("spacing", "width", "area_ratio"):
        with (out / f"{group}_summary.csv").open() as f:
            assert len(list(csv.DictReader(f))) == 3
    with pytest.raises(ValueError): doe.evaluate(study, root, config, out, device="cpu")
    base = yaml.safe_load(config.read_text())
    base["bc_global"]["timeStep_s"] = .0001
    config.write_text(yaml.safe_dump(base))
    with pytest.raises(ValueError, match="changed inputs"):
        doe.evaluate(study, root, config, out, device="cpu", resume=True)


def test_infrastructure_failure_is_not_nonignition(tmp_path, monkeypatch):
    root, config, out, calls = setup_run(tmp_path, monkeypatch)
    def broken(self, items): raise RuntimeError("compiler unavailable")
    monkeypatch.setattr(FakeEvaluator, "evaluate_batch", broken)
    with pytest.raises(RuntimeError):
        doe.evaluate(doe.Study(), root, config, out, device="cpu", only=["SP_G1"])
    row = doe.read_json(out / "cases" / "SP_G1" / "result.json")
    assert row["status"] == "execution_failed" and not row["numerical_valid"]
    assert "ignitionSucceeded" not in row["raw"]


def test_existing_sources_are_monitored(tmp_path, monkeypatch):
    root, config, out, calls = setup_run(tmp_path, monkeypatch)
    original = FakeEvaluator.evaluate_batch
    def tamper(self, items):
        config.write_text(config.read_text() + "\n# unexpected concurrent edit\n")
        return original(self, items)
    monkeypatch.setattr(FakeEvaluator, "evaluate_batch", tamper)
    with pytest.raises(RuntimeError, match="sources changed"):
        doe.evaluate(doe.Study(), root, config, out, device="cpu", only=["SP_G1"])


def test_experiment_statistics_do_not_impute_censored_values(tmp_path):
    source, target = tmp_path/"experiment.csv", tmp_path/"summary.csv"
    source.write_text("case_id,replicate,valid,ignited,ignition_delay_s,observation_window_s\n"
        "SP_G1,1,true,true,0.2,2\nSP_G1,2,true,true,0.4,2\n"
        "SP_G1,3,true,false,,2\nSP_G1,4,false,,,2\nSP_G1,5,,,,2\n")
    row = doe.summarize_experiment(source, target)[0]
    assert row["n_valid"] == 3 and row["n_ignited"] == 2
    assert row["n_excluded"] == row["n_right_censored"] == 1
    assert row["mean_delay_among_ignited_s"] == pytest.approx(.3)
    assert row["sample_sd_among_ignited_s"] == pytest.approx(np.sqrt(.02))
    source.write_text(source.read_text().replace("3,true,false,,2", "3,true,false,2,2"))
    with pytest.raises(ValueError): doe.summarize_experiment(source, target)


def test_original_production_geometry_builder_when_checkout_available(tmp_path):
    # Runs without compiling/integrating the physics when the real repository is present.
    # Skipped in the standalone add-on test environment; not counted as a physics test.
    if not (ROOT / "python/ecsp_nsga2/evaluator.py").exists():
        pytest.skip("Full upstream checkout is not mounted in the standalone environment")
    torch = pytest.importorskip("torch")
    from ecsp_nsga2.evaluator import BCGlobalPreflameEvaluator, BCCandidateGeometryError
    class Probe(doe.FullHeightStripGeometryMixin, BCGlobalPreflameEvaluator): pass
    probe = object.__new__(Probe)
    probe.grid_size, probe.domain_size_m, probe.minimum_gap_m = 200, .025, .001
    probe.device = torch.device("cpu")
    study = doe.Study()
    items = [(*doe.make_case(sid, study), tmp_path/sid) for sid in doe.CASES]
    geometry = probe._build_geometry_batch(items)
    assert geometry.propellant.all() and not geometry.fixed.any()
    assert all(np.array_equal(geometry.anode[i].numpy(), item[0]) for i, item in enumerate(items))
    # The original equal-area policy still rejects an asymmetric case: no global patch.
    legacy = object.__new__(BCGlobalPreflameEvaluator)
    for name in ("grid_size", "domain_size_m", "minimum_gap_m", "device"):
        setattr(legacy, name, getattr(probe, name))
    legacy.target_area_fraction_per_polarity = .12
    legacy.area_tolerance_fraction = .01
    legacy.maximum_components_per_polarity, legacy.maximum_total_components = 1, 2
    legacy.minimum_width_mm = .3
    with pytest.raises(BCCandidateGeometryError): legacy._build_geometry_batch([items[5]])


def test_addon_paths_do_not_invalidate_legacy_paired_source_fingerprint():
    # These prefixes are taken from paired_workflow.protected_hashes at BASE_COMMIT.
    old_prefixes = ("cpp", "python/ecsp_native", "python/ecsp_nsga2", "python/ecsp_v6",
                   "python/ecsp_cuda", "python/ecsp_cpp", "python/ecsp_reactive", "config")
    additions = ("python/run_electrode_parameter_study.py",
                 "studies/electrode_parameter_study_25mm.yaml",
                 "tests/test_electrode_parameter_study.py",
                 "docs/ELECTRODE_PARAMETER_STUDY_25MM_KR.md")
    assert all(not any(Path(p).is_relative_to(prefix) for prefix in old_prefixes) for p in additions)


def test_loaded_study_does_not_confuse_length_with_thickness(tmp_path):
    valid = ROOT / "studies/electrode_parameter_study_25mm.yaml"
    assert doe.Study.load(valid).grid_size == 200
    data = yaml.safe_load(valid.read_text())
    data["study"]["electrode_length_mm"] = 20
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="electrode_length_mm"):
        doe.Study.load(invalid)
