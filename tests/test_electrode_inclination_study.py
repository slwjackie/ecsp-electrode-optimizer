"""Inclination pipeline contracts; the fake evaluator tests wiring, not physics."""
from pathlib import Path
from types import SimpleNamespace
import copy
import csv
import sys

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import run_electrode_parameter_study as study


@pytest.fixture
def spec():
    return yaml.safe_load((ROOT / "config/electrode_inclination_study_25mm.yaml").read_text())


def test_zero_angle_reproduces_every_legacy_unsplit_mask():
    for sid, (wa, wc, gap, _) in study.CASES.items():
        row = dict(id="TEST_" + sid, anode_width_mm=wa, cathode_width_mm=wc,
                   gap_mm=gap, inclination_deg=0, groups=["inclination"])
        a, c, meta = study.make_inclined_case(row, 200)
        old_a, old_c, old_meta = study.make_case(sid, 200)
        assert np.array_equal(a, old_a) and np.array_equal(c, old_c)
        assert meta["actual_anode_area_mm2"] == old_meta["actual_anode_area_mm2"]


def test_generate_audit_svg_and_full_propellant(tmp_path, spec):
    out = tmp_path / "study"
    study.generate(spec, out)
    validated, items = study.audit(out)
    assert validated == study.validate_spec(spec)
    assert len(items) == len(spec["cases"])
    for a, c, m, _ in items:
        folder = out / "library" / m["geometry_id"]
        assert (folder / "geometry.svg").read_text().count("<polygon") == 2
        with np.load(folder / "mask.npz") as data:
            assert data["propellant"].all()
        assert m["actual_anode_area_mm2"] == pytest.approx(100)
        assert m["actual_cathode_area_mm2"] == pytest.approx(100)
    with (out / "experiment_schedule.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(items) * 5


def test_audit_binds_metadata_to_declared_angle_even_if_checksums_match(tmp_path, spec):
    out = tmp_path / "study"
    study.generate(spec, out)
    manifest = study.read_json(out / "geometry_manifest.json")
    row = copy.deepcopy(spec["cases"][-1])
    row["inclination_deg"] -= 1
    a, c, meta = study.make_inclined_case(row, spec["grid_size"])
    folder = out / "library" / row["id"]
    np.savez_compressed(folder / "mask.npz", anode=a, cathode=c,
                        propellant=np.ones_like(a), domain_mm=25.)
    meta["mask_file_sha256"] = study.file_hash(folder / "mask.npz")
    study.write_json(folder / "metadata.json", meta)
    manifest[-1] = meta
    study.write_json(out / "geometry_manifest.json", manifest)
    with pytest.raises(ValueError, match="declared study case"):
        study.audit(out)


def test_geometry_batch_validates_end_gap_and_masks(spec):
    pytest.importorskip("torch")
    items = [(*study.make_inclined_case(row, 200), Path("unused")) for row in spec["cases"]]
    geom = study.build_geometry(items, 200, "cpu", .001, SimpleNamespace)
    assert geom.propellant.all() and not geom.fixed.any()
    assert geom.anode.shape == (len(items), 200, 200)
    # A central gap of 17 mm cannot satisfy a 10 mm minimum at the bent ends.
    with pytest.raises(ValueError, match="requested minimum"):
        study.build_geometry(items, 200, "cpu", .010, SimpleNamespace)
    a, c, meta, path = items[-1]
    a = a.copy()
    a[0, np.flatnonzero(a[0])[0]] = False
    with pytest.raises(ValueError, match="geometry contract"):
        study.build_geometry([(a, c, meta, path)], 200, "cpu", .001, SimpleNamespace)


def test_evaluate_resume_summary_and_changed_angle(tmp_path, spec):
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    physics = root / "config/physics.yaml"
    physics.write_text(yaml.safe_dump({"geometry": {"domain_mm": 25, "grid_size": 200},
        "evaluator": {"base_overrides": {"numerics": {"physicsDtype": "float64"}}}}))
    out = root / "runs/study"
    study.generate(spec, out)

    class FakeEvaluator:
        calls = 0
        config = {"fixture_only": True}

        def evaluate_batch(self, items):
            self.calls += 1
            return [dict(geometry_id=m["geometry_id"], ignitionSucceeded=False)
                    for _, _, m, _ in items]

        def _trial_valid(self, row):
            return True, "fixture_only"

    ev = FakeEvaluator()
    rows = study.evaluate(root, out, physics, "cpu", 2, False, lambda *_: ev)
    assert ev.calls == 2
    assert [r["inclination_deg"] for r in rows] == [r["inclination_deg"] for r in spec["cases"]]
    assert all(r["ignitionDelay_s"] is None for r in rows)
    assert (out / "inclination_summary.csv").exists()
    assert not (out / "spacing_summary.csv").exists()
    study.evaluate(root, out, physics, "cpu", 2, True, lambda *_: ev)
    assert ev.calls == 2
    spec["cases"][-1]["inclination_deg"] -= 1
    study.write_json(out / "study.json", spec)
    with pytest.raises(ValueError, match="declared study case"):
        study.evaluate(root, out, physics, "cpu", 2, True, lambda *_: ev)


def test_experimental_summary_accepts_only_validated_profile_ids(tmp_path, spec):
    source, destination = tmp_path / "measurements.csv", tmp_path / "summary.csv"
    sid = spec["cases"][-1]["id"]
    study.write_csv(source, [dict(case_id=sid, replicate=1, ignition_observed=True,
        ignition_delay_s=.2, observation_window_s=2)])
    study.summarize_experiments(source, destination, study_spec=spec)
    with destination.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(spec["cases"])
    assert rows[-1]["mean_delay_ignited_only_s"] == "0.2"
    with pytest.raises(ValueError, match="case selection"):
        study.summarize_experiments(source, destination, case_ids=[sid])


def test_legacy_ids_cannot_be_repurposed(spec):
    spec["cases"][0]["id"] = "WD_W4"
    with pytest.raises(ValueError, match="legacy"):
        study.validate_spec(spec)
