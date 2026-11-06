"""Fixed-end central-gap study I/O; mocked rows do not validate the PDE."""
from __future__ import annotations

import copy
import csv
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import run_electrode_parameter_study as study
from electrode_central_gap_geometry import make_central_gap_case

CONFIG = "config/electrode_central_gap_study_12x25mm.yaml"


@pytest.fixture
def spec():
    return yaml.safe_load((ROOT / CONFIG).read_text())


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_generated_fixed_end_study_keeps_entire_propellant_and_svg_coordinates(tmp_path, spec):
    out = tmp_path / "central_gap"
    study.generate(spec, out)
    canonical, items = study.audit(out)
    assert canonical == study.validate_spec(spec)
    assert [m["central_gap_mm"] for _, _, m, _ in items] == [2, 4, 6, 8]
    for a, c, metadata, _ in items:
        assert a.shape == c.shape == (200, 96)
        assert metadata["domain_width_mm"] == 12
        assert metadata["domain_height_mm"] == 25
        assert metadata["anode_width_mm"] == metadata["cathode_width_mm"] == 2
        assert metadata["end_gap_mm"] == 2
        assert metadata["actual_anode_area_mm2"] == 50
        assert metadata["actual_cathode_area_mm2"] == 50
        assert metadata["propellant_domain_area_mm2"] == 300
        folder = out / "library" / metadata["geometry_id"]
        with np.load(folder / "mask.npz") as archive:
            propellant = archive["propellant"]
            assert propellant.shape == (200, 96)
            assert propellant.all()
            # Both outer side regions and material under the contacts remain
            # in the calculation domain, rather than just the central channel.
            assert not a[0, 0] and not c[0, 0]
            assert not a[0, -1] and not c[0, -1]
            assert propellant[0, 0] and propellant[0, -1]
            assert propellant[a | c].all()
            assert float(archive["domain_width_mm"]) == 12
            assert float(archive["domain_height_mm"]) == 25
        svg = ET.fromstring((folder / "geometry.svg").read_text())
        assert svg.attrib["viewBox"] == "0 0 12 25"
        polygons = svg.findall("{http://www.w3.org/2000/svg}polygon")
        assert len(polygons) == 2
        for polarity, polygon in enumerate(polygons):
            vertices = np.array([[float(v) for v in pair.split(",")]
                                 for pair in polygon.attrib["points"].split()])
            for y in (0, 25):
                xs = sorted(vertices[np.isclose(vertices[:, 1], y), 0])
                assert xs == ([3., 5.] if polarity == 0 else [7., 9.])
            center_xs = sorted(vertices[np.isclose(vertices[:, 1], 12.5), 0])
            gap = metadata["central_gap_mm"]
            expected = [4 - gap / 2, 6 - gap / 2] if polarity == 0 else [6 + gap / 2, 8 + gap / 2]
            assert center_xs == expected
    schedule = read_csv(out / "experiment_schedule.csv")
    assert len(schedule) == 20
    for block in range(1, 6):
        assert {row["case_id"] for row in schedule if int(row["block"]) == block} == {
            row["id"] for row in spec["cases"]}


@pytest.mark.parametrize("tamper", ["domain", "propellant", "end_gap"])
def test_audit_rejects_tampering_even_after_checksum_updates(tmp_path, spec, tamper):
    out = tmp_path / "central_gap"
    study.generate(spec, out)
    manifest = study.read_json(out / "geometry_manifest.json")
    folder = out / "library" / manifest[0]["geometry_id"]
    if tamper == "end_gap":
        manifest[0]["end_gap_mm"] = 4.
    else:
        with np.load(folder / "mask.npz") as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        if tamper == "domain":
            arrays["domain_width_mm"] = 25.
        else:
            arrays["propellant"][0, 0] = False
        np.savez_compressed(folder / "mask.npz", **arrays)
        manifest[0]["mask_file_sha256"] = study.file_hash(folder / "mask.npz")
    study.write_json(folder / "metadata.json", manifest[0])
    study.write_json(out / "geometry_manifest.json", manifest)
    with pytest.raises(ValueError):
        study.audit(out)


def test_geometry_batch_keeps_lateral_and_under_contact_cells(spec):
    torch = pytest.importorskip("torch")
    from ecsp_v6.physics.geometry import GeometryBatch
    items = [(*make_central_gap_case(row, 200), Path("unused")) for row in spec["cases"]]
    geometry = study.build_geometry(items, 200, "cpu", .002, GeometryBatch)
    assert geometry.anode.shape == geometry.cathode.shape == (4, 200, 96)
    assert geometry.anode.dtype == torch.bool
    assert geometry.domain_width_m == .012
    assert geometry.domain_height_m == geometry.domain_size_m == .025
    assert geometry.propellant.all() and not geometry.fixed.any()
    assert geometry.propellant.sum().item() == 4 * 200 * 96
    for index, item in enumerate(items):
        np.testing.assert_array_equal(geometry.anode[index].numpy(), item[0])
        np.testing.assert_array_equal(geometry.cathode[index].numpy(), item[1])
    with pytest.raises(ValueError, match="requested minimum"):
        study.build_geometry(items, 200, "cpu", .0021, GeometryBatch)


def test_cli_generates_central_gap_config(tmp_path):
    out = tmp_path / "central_gap"
    result = subprocess.run([sys.executable, str(ROOT / "python/run_electrode_parameter_study.py"),
        "generate", "--study-config", CONFIG, "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    _, items = study.audit(out)
    assert len(items) == 4


@pytest.mark.parametrize("command,options", [
    ("generate-sweep", ["--angles", "0", "5"]),
    ("generate", ["--electrode-width-mm", "4"]),
])
def test_cli_rejects_incompatible_independent_controls(tmp_path, command, options):
    out = tmp_path / "incompatible"
    result = subprocess.run([sys.executable, str(ROOT / "python/run_electrode_parameter_study.py"),
        command, "--study-config", CONFIG, "--out", str(out), *options],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert not out.exists()


@pytest.mark.parametrize("skip_vmin", [False, True])
def test_evaluate_fixed_end_summaries_and_resume_preserve_inputs(tmp_path, spec, skip_vmin):
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    physics = root / "config/physics.yaml"
    base = {"physics": {"voltage_V": 260., "composition": {"sentinel": 1}},
        "preflame_model": {"timeStep_s": .00025, "endTime_s": 2.},
        "geometry": {"domain_mm": 25, "grid_size": 200},
        "minimum_ignition_voltage_search": {"enabled": True},
        "evaluation": {"objectives_minimise": ["ignition_delay_s", "minimum_ignition_voltage_V"]},
        "evaluator": {"base_overrides": {"numerics": {"physicsDtype": "float64"}}}}
    physics.write_text(yaml.safe_dump(base))
    out = root / "runs/central_gap"
    study.generate(spec, out)
    created = []

    class RecordingEvaluator:
        def __init__(self, cfg):
            self.config = copy.deepcopy(cfg)
            self.calls = []

        def evaluate_batch(self, items):
            self.calls.append([item[2]["geometry_id"] for item in items])
            assert all(item[0].shape == (200, 96) for item in items)
            return [dict(geometry_id=meta["geometry_id"], ignitionSucceeded=False,
                         peakMaximumTemperature_K=300.,
                         minimumIgnitionVoltageSearchStatus=(
                             "not_requested_reference_voltage_only" if skip_vmin else
                             "right_censored_no_ignition_at_upper_bound"))
                    for _, _, meta, _ in items]

        def _trial_valid(self, row):
            return True, "fixture_only"

    def factory(root, cfg, workdir):
        assert cfg["geometry"]["domain_width_mm"] == 12
        assert cfg["geometry"]["domain_height_mm"] == 25
        assert cfg["physics"] == base["physics"]
        assert cfg["preflame_model"] == base["preflame_model"]
        assert cfg["minimum_ignition_voltage_search"]["enabled"] is (not skip_vmin)
        assert ("minimum_ignition_voltage_V" in cfg["evaluation"]["objectives_minimise"]) is (not skip_vmin)
        created.append(RecordingEvaluator(cfg))
        return created[-1]

    rows = study.evaluate(root, out, physics, "cpu", 2, False, factory, skip_vmin=skip_vmin)
    assert len(rows) == 4 and len(created) == 1 and len(created[0].calls) == 2
    assert [row["central_gap_mm"] for row in rows] == [2, 4, 6, 8]
    assert all(row["end_gap_mm"] == 2 for row in rows)
    assert all(row["actual_anode_area_mm2"] == row["actual_cathode_area_mm2"] == 50 for row in rows)
    assert all(row["grid_rows"] == 200 and row["grid_columns"] == 96 for row in rows)
    assert all(row["result_state"] == "no_onset_within_horizon" for row in rows)
    assert all(row["ignitionDelay_s"] is None for row in rows)
    assert len(read_csv(out / "central_gap_summary.csv")) == 4
    assert not (out / "inclination_summary.csv").exists()
    assert not (out / "width_2mm_summary.csv").exists()
    assert not (out / "spacing_summary.csv").exists()
    resumed = study.evaluate(root, out, physics, "cpu", 2, True, factory, skip_vmin=skip_vmin)
    assert rows == resumed and len(created) == 1
    with pytest.raises(ValueError, match="Existing/changed"):
        study.evaluate(root, out, physics, "cpu", 2, True, factory, skip_vmin=not skip_vmin)


def test_central_gap_experiment_template_remains_distinct_from_solver_results(tmp_path, spec):
    out = tmp_path / "central_gap"
    study.generate(spec, out)
    study.summarize_experiments(out / "experiment_measurements_template.csv",
                               out / "experiment_summary.csv", study_spec=spec)
    rows = read_csv(out / "experiment_summary.csv")
    assert len(rows) == 4
    assert all(row["n_pending"] == "5" and row["n_valid"] == "0" for row in rows)
