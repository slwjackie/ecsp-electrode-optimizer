"""Rectangular study I/O and orchestration; no mocked result validates physics."""
from __future__ import annotations

import copy
import csv
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import run_electrode_parameter_study as study

CONFIG = "config/electrode_inclination_study_12x25mm.yaml"


@pytest.fixture
def spec():
    return yaml.safe_load((ROOT / CONFIG).read_text())


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_generated_rectangle_has_physical_extent_and_no_padding(tmp_path, spec):
    out = tmp_path / "rectangle"
    study.generate(spec, out)
    validated, items = study.audit(out)
    assert validated == study.validate_spec(spec)
    assert len(items) == 6
    for a, c, meta, _ in items:
        folder = out / "library" / meta["geometry_id"]
        assert a.shape == c.shape == (200, 96)
        assert meta["domain_width_mm"] == 12
        assert meta["domain_height_mm"] == 25
        assert meta["propellant_domain_area_mm2"] == 300
        assert meta["actual_anode_area_mm2"] == pytest.approx(25 * meta["anode_width_mm"])
        assert meta["actual_cathode_area_mm2"] == meta["actual_anode_area_mm2"]
        with np.load(folder / "mask.npz") as z:
            assert z["propellant"].shape == (200, 96)
            assert z["propellant"].all()
            assert float(z["domain_width_mm"]) == 12
            assert float(z["domain_height_mm"]) == 25
            assert np.array_equal(z["anode"], a)
        drawing = (folder / "geometry.svg").read_text()
        assert 'viewBox="0 0 12 25"' in drawing
        assert 'width="12mm" height="25mm"' in drawing
        assert drawing.count("<polygon") == 2
    schedule = read_csv(out / "experiment_schedule.csv")
    assert len(schedule) == 30
    for block in range(1, 6):
        assert {r["case_id"] for r in schedule if int(r["block"]) == block} == {
            row["id"] for row in spec["cases"]}


@pytest.mark.parametrize("width,expected", [(None, 6), (2, 4), (4, 2)])
def test_cli_selects_one_or_both_width_groups(tmp_path, width, expected):
    out = tmp_path / "rectangle"
    command = [sys.executable, str(ROOT / "python/run_electrode_parameter_study.py"),
               "generate", "--study-config", CONFIG, "--out", str(out)]
    if width is not None:
        command.extend(["--electrode-width-mm", str(width)])
    result = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    generated, items = study.audit(out)
    assert len(items) == expected
    if width is not None:
        assert {r["anode_width_mm"] for r in generated["cases"]} == {width}
        assert [r["inclination_deg"] for r in generated["cases"]] == (
            [0, 5, 10, 15] if width == 2 else [0, 5])


def test_cli_width_selection_is_not_silently_applied_to_square_study(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "python/run_electrode_parameter_study.py"),
        "generate", "--study-config", "config/electrode_inclination_study_25mm.yaml",
        "--electrode-width-mm", "2", "--out", str(tmp_path / "wrong")],
        capture_output=True, text=True, cwd=ROOT)
    assert result.returncode != 0
    assert "rectangular study config" in result.stderr
    assert not (tmp_path / "wrong").exists()


def test_audit_rejects_dimensions_even_with_updated_checksum(tmp_path, spec):
    out = tmp_path / "rectangle"
    study.generate(spec, out)
    manifest = study.read_json(out / "geometry_manifest.json")
    folder = out / "library" / manifest[0]["geometry_id"]
    with np.load(folder / "mask.npz") as z:
        arrays = {key: z[key].copy() for key in z.files}
    arrays["domain_width_mm"] = 25.
    np.savez_compressed(folder / "mask.npz", **arrays)
    manifest[0]["mask_file_sha256"] = study.file_hash(folder / "mask.npz")
    study.write_json(folder / "metadata.json", manifest[0])
    study.write_json(out / "geometry_manifest.json", manifest)
    with pytest.raises(ValueError, match="domain dimensions"):
        study.audit(out)


def test_geometry_batch_preserves_rectangle_and_rejects_mixed_domains(spec):
    torch = pytest.importorskip("torch")
    from ecsp_v6.physics.geometry import GeometryBatch
    items = [(*study.make_rectangular_case(row, 200), Path("unused")) for row in spec["cases"]]
    geometry = study.build_geometry(items, 200, "cpu", .001, GeometryBatch)
    assert geometry.anode.shape == geometry.cathode.shape == (6, 200, 96)
    assert geometry.anode.dtype == torch.bool
    assert geometry.propellant.all() and not geometry.fixed.any()
    assert geometry.domain_width_m == .012
    assert geometry.domain_height_m == geometry.domain_size_m == .025
    assert geometry.minimum_gap_m == .001
    for index, item in enumerate(items):
        assert np.array_equal(geometry.anode[index].numpy(), item[0])
    with pytest.raises(ValueError, match="requested minimum"):
        study.build_geometry(items, 200, "cpu", .002, GeometryBatch)
    square = (*study.make_case("WD_W4", 200), Path("unused"))
    with pytest.raises(ValueError, match="one domain shape"):
        study.build_geometry([items[0], square], 200, "cpu", .001, GeometryBatch)


def test_runtime_requests_true_dimensions_without_mutating_physics():
    base = {"geometry": {"domain_mm": 20, "grid_size": 96},
            "physics": {"voltage_V": 260, "composition": {"sentinel": 1}},
            "evaluator": {"base_overrides": {"numerics": {"physicsDtype": "float64"}}}}
    before = copy.deepcopy(base)
    cfg = study.runtime_config(base, 200, "cpu", 6, rectangular_study=True)
    assert base == before
    assert cfg["geometry"]["domain_width_mm"] == 12
    assert cfg["geometry"]["domain_height_mm"] == 25
    assert cfg["geometry"]["domain_mm"] == 25
    assert cfg["geometry"]["grid_size"] == 200
    assert cfg["physics"] == base["physics"]
    assert cfg["evaluator"]["save_representative_fields"] is True


def test_evaluate_resume_and_width_summaries_use_rectangular_study(tmp_path, spec):
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    physics = root / "config/physics.yaml"
    physics.write_text(yaml.safe_dump({"geometry": {"domain_mm": 25, "grid_size": 200},
        "evaluator": {"base_overrides": {"numerics": {"physicsDtype": "float64"}}}}))
    out = root / "runs/rectangle"
    study.generate(spec, out)
    created = []

    class RecordingEvaluator:
        def __init__(self, cfg):
            self.config = copy.deepcopy(cfg)
            self.calls = []

        def evaluate_batch(self, items):
            self.calls.append([item[2]["geometry_id"] for item in items])
            assert all(item[0].shape == (200, 96) for item in items)
            return [dict(geometry_id=m["geometry_id"], ignitionSucceeded=False,
                         peakMaximumTemperature_K=300.) for _, _, m, _ in items]

        def _trial_valid(self, row):
            return True, "fixture_only"

    def factory(root, cfg, workdir):
        assert cfg["geometry"]["domain_width_mm"] == 12
        assert cfg["geometry"]["domain_height_mm"] == 25
        created.append(RecordingEvaluator(cfg))
        return created[-1]

    rows = study.evaluate(root, out, physics, "cpu", 2, False, factory)
    assert len(rows) == 6 and len(created) == 1 and len(created[0].calls) == 3
    assert all(row["grid_rows"] == 200 and row["grid_columns"] == 96 for row in rows)
    assert all(row["domain_width_mm"] == 12 and row["domain_height_mm"] == 25 for row in rows)
    assert all(row["propellant_domain_area_mm2"] == 300 for row in rows)
    assert all(row["ignitionDelay_s"] is None for row in rows)
    for width, count, area in [(2, 4, 50), (4, 2, 100)]:
        subset = read_csv(out / f"width_{width}mm_summary.csv")
        assert len(subset) == count
        assert all(float(row["anode_width_mm"]) == width for row in subset)
        assert all(float(row["actual_anode_area_mm2"]) == area for row in subset)
        assert all(float(row["actual_cathode_area_mm2"]) == area for row in subset)
    assert len(read_csv(out / "inclination_summary.csv")) == 6
    assert not (out / "spacing_summary.csv").exists()
    study.evaluate(root, out, physics, "cpu", 2, True, factory)
    assert len(created) == 1 and len(created[0].calls) == 3
    with pytest.raises(ValueError, match="Existing/changed"):
        study.evaluate(root, out, physics, "cpu", 3, True, factory)


def test_rectangular_experiment_summary_uses_selected_case_ids(tmp_path, spec):
    spec["cases"] = [row for row in spec["cases"] if row["anode_width_mm"] == 4]
    out = tmp_path / "width4"
    study.generate(spec, out)
    study.summarize_experiments(out / "experiment_measurements_template.csv",
                               out / "experiment_summary.csv", study_spec=spec)
    rows = read_csv(out / "experiment_summary.csv")
    assert [row["case_id"] for row in rows] == ["W4_ANGLE_00", "W4_ANGLE_05"]
    assert all(row["n_pending"] == "5" and row["n_valid"] == "0" for row in rows)
