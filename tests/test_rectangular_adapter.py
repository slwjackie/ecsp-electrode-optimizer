"""Physical dimensions and representative outputs for true rectangular arrays."""
from __future__ import annotations

import copy
import math
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch
import yaml

from ecsp_cpp_cuda.adapter import pack_config
from ecsp_preflame.electrochemical_thermal_decomposition import (
    _preflame_grid_spacing,
    _validate_preflame_runtime_contract,
    continue_preflame_onset_batch_to_evaluation,
    run_preflame_batch,
)
from ecsp_preflame.evaluator import ElectrochemicalThermalDecompositionEvaluator, EvaluatorError
from ecsp_preflame.field_diagnostics import _make_geometry, write_snapshot_artifacts
from ecsp_v6.physics.geometry import GeometryBatch
import run_electrode_parameter_study as study


ROOT = Path(__file__).resolve().parents[1]


def public_config():
    raw = yaml.safe_load((ROOT / "config/preflame_cpp_cuda_debug.yaml").read_text())
    cfg = study.runtime_config(raw, 200, "cpu", 2, rectangular_study=True)
    cfg["evaluator"]["cpp_cuda"]["cpu_runtime"] = "standalone"
    return cfg


@pytest.fixture
def evaluator(tmp_path):
    cfg = public_config()
    adapter = dict(cfg["evaluator"], physics_config=cfg)
    # Configuration tests do not need to build a native executable.
    return ElectrochemicalThermalDecompositionEvaluator(ROOT, adapter, tmp_path / "eval")


def geometry(rows=200, columns=96):
    a = torch.zeros((1, rows, columns), dtype=torch.bool)
    a[:, :, :max(1, columns // 6)] = True
    c = a.flip(-1)
    return GeometryBatch(["rect"], a, c, torch.zeros_like(a), torch.ones_like(a),
                         rows, .025, .001, domain_width_m=.012, domain_height_m=.025)


def test_rectangular_runtime_contract_has_physical_equal_spacing(evaluator):
    geom = geometry()
    assert evaluator.domain_width_m == .012
    assert evaluator.domain_height_m == evaluator.domain_size_m == .025
    _validate_preflame_runtime_contract(evaluator.config, evaluator.composition, geom)
    packed, _ = pack_config(evaluator.config, evaluator.composition, geom.grid_size)
    assert packed["dx"] == pytest.approx(.000125)
    assert geom.propellant.sum().item() * packed["dx"]**2 == pytest.approx(300e-6)
    with pytest.raises(ValueError, match=r"C\+\+/CUDA backend"):
        run_preflame_batch(geom, evaluator.config, evaluator.composition, 1., torch.float64)


def test_rectangular_runtime_rejects_stretched_or_unlabelled_grids(evaluator):
    with pytest.raises(ValueError, match="equal physical dx and dy"):
        _validate_preflame_runtime_contract(evaluator.config, evaluator.composition, geometry(columns=97))
    geom = geometry()
    geom.domain_width_m = None
    with pytest.raises(ValueError, match="domain_width_m"):
        _validate_preflame_runtime_contract(evaluator.config, evaluator.composition, geom)
    square = {"geometry": {"domainSize_m": .025}}
    assert _preflame_grid_spacing(square, 200, 200) == pytest.approx(.000125)
    with pytest.raises(ValueError, match="equal physical dx and dy"):
        _preflame_grid_spacing(square, 200, 96)


def test_representative_mask_and_archive_keep_rectangle(tmp_path):
    original = geometry()
    a, c = original.anode[0].numpy(), original.cathode[0].numpy()
    item = (a, c, {"geometry_id": "rect"}, tmp_path)
    got, raw_a, raw_c = _make_geometry(item, 200, .025, "cpu",
        domain_width_m=.012, domain_height_m=.025)
    assert got.anode.shape == (1, 200, 96)
    np.testing.assert_array_equal(raw_a, a)
    np.testing.assert_array_equal(raw_c, c)
    field = np.zeros_like(a, dtype=float)
    artifact, _ = write_snapshot_artifacts(tmp_path, "eval", temperature=field + 300,
        current_x=field, current_y=field, current_magnitude=field, potential=field,
        progress=field, joule_heat=field, anode=a, cathode=c,
        domain_mm=25., domain_width_mm=12., domain_height_mm=25., time_s=.01,
        initial_temperature_K=300., semantics="rectangular_test")
    with np.load(tmp_path / artifact["npz"]) as saved:
        assert saved["temperature_K"].shape == (200, 96)
        assert saved["x_mm"].shape == (96,) and saved["y_mm"].shape == (200,)
        assert saved["x_mm"][-1] == pytest.approx(11.9375)
        assert saved["y_mm"][-1] == pytest.approx(24.9375)
        assert float(saved["domain_width_mm"]) == 12.
        assert float(saved["domain_height_mm"]) == 25.
    assert artifact["grid_rows"] == 200 and artifact["grid_columns"] == 96


def test_rectangular_objective_continuation_uses_same_cell_spacing(evaluator):
    config = copy.deepcopy(evaluator.config)
    bc = config["preflameModel"]
    bc.update(timeStep_s=.001, endTime_s=.003, evaluationTime_s=.003,
              electricalUpdateInterval_s=.001)
    bc["thermal"].update(heat_capacity={"mode": "constant", "value": 1000.},
        thermal_conductivity={"mode": "constant", "value": 0.},
        ambientTemperature_K=300., minimumTemperature_K=1., maximumTemperature_K=1.e6,
        convectionCoefficient_W_per_m2K=0., emissivity=0.)
    for channel in bc["kinetics"]["channels"]:
        count = len(channel["alpha_grid"])
        channel["activation_energy_J_per_mol"] = [0.] * count
        channel["ln_Af_per_s"] = [math.log(.1)] * count
        channel["heat_release_J_per_kg"] = 0.
    shape = (2, 25, 12)
    def full(value):
        return torch.full(shape, value, dtype=torch.float64)
    composition = evaluator.composition
    fields = {"temperatureAtOnset_K": full(300.), "globalProgressAtOnset": full(0.),
        "alphaChannel1AtOnset": full(0.), "alphaChannel2AtOnset": full(0.),
        "mobileLPAtOnset_mol_per_m3": full(composition.initial_lp_mol_per_m3),
        "mobileWaterAtOnset_mol_per_m3": full(composition.initial_water_mol_per_m3),
        "generatedWaterProductAtOnset_mol_per_m3": full(0.),
        "electrochemicalLPConsumedAtOnset_mol_per_m3": full(0.),
        "pvaReactiveRepeatAtOnset_mol_per_m3": full(composition.initial_pva_repeat_mol_per_m3),
        "propellantMask": torch.ones(shape, dtype=torch.bool)}
    result = continue_preflame_onset_batch_to_evaluation(fields, config, composition,
        torch.tensor([0., .002], dtype=torch.float64), .003, torch.float64)
    torch.testing.assert_close(result["meanGlobalProgressAtEvaluationTime"],
        torch.tensor([3e-4, 1e-4], dtype=torch.float64), atol=1e-15, rtol=1e-12)
    assert result["temperatureAtEvaluation_K"].shape == shape
    assert torch.isfinite(result["remainingReactiveMassFractionAtEvaluationTime"]).all()


@pytest.mark.parametrize("rectangle_source", ["config", "masks", "fields"])
def test_rectangular_propagation_handoff_is_explicitly_unsupported(evaluator, tmp_path, rectangle_source):
    if rectangle_source != "config":
        evaluator.config["geometry"].pop("domainWidth_m")
        evaluator.config["geometry"].pop("domainHeight_m")
    mask_shape = (25, 12) if rectangle_source == "masks" else (25, 25)
    field_shape = (1, 25, 12) if rectangle_source == "fields" else (1, 25, 25)
    a = np.zeros(mask_shape, dtype=bool)
    item = (a, a.copy(), {"geometry_id": "handoff"}, tmp_path / "handoff")
    output = {"handoffFields": {"temperatureAtOnset_K": np.full(field_shape, 300.)}}
    with pytest.raises(EvaluatorError, match="propagation handoff requires a square domain"):
        evaluator._extract_handoff_batch([{}], output, [item])
    assert not item[3].exists()


def test_square_propagation_handoff_keeps_existing_contacts(evaluator, tmp_path, monkeypatch):
    evaluator.config["geometry"].pop("domainWidth_m")
    evaluator.config["geometry"].pop("domainHeight_m")
    monkeypatch.setattr(evaluator, "_trial_valid", lambda row: (True, "test"))
    a = np.zeros((25, 25), dtype=bool)
    a[:, :4] = True
    c = a[:, ::-1].copy()
    item = (a, c, {"geometry_id": "square"}, tmp_path / "handoff")
    output = {"handoffFields": {"temperatureAtOnset_K": np.full((1, 25, 25), 300.)}}
    result = evaluator._extract_handoff_batch([{"ignitionSucceeded": False}], output, [item])
    np.testing.assert_array_equal(result[0]["handoff"]["anodeContactMask"], a)
    np.testing.assert_array_equal(result[0]["handoff"]["cathodeContactMask"], c)
    assert (item[3] / "bc_handoff_fields.npz").is_file()


def test_compiled_representative_safe_run_keeps_true_rectangle(tmp_path):
    if shutil.which("c++") is None:
        pytest.skip("A C++ compiler is required for the compiled integration check")
    cfg = public_config()
    ev = study.create_study_evaluator(ROOT, cfg, tmp_path / "eval")
    ev.config["preflameModel"].update(endTime_s=.002, evaluationTime_s=.002)
    spec = yaml.safe_load((ROOT / "config/electrode_inclination_study_12x25mm.yaml").read_text())
    # Both electrode widths must reach the native and diagnostic paths intact.
    cases = [next(row for row in spec["cases"] if row["anode_width_mm"] == width
                  and row["inclination_deg"] == 5) for width in (2, 4)]
    items = [(*study.make_rectangular_case(row, 200), tmp_path / row["id"]) for row in cases]
    rows, hands = ev._safe_run(items, [1., 1.])
    assert len(rows) == 2 and hands == []
    for row, item in zip(rows, items):
        assert row["cppCudaExecution"]["compiled_time_loop"]
        assert row["cppCudaExecution"]["cpu_runtime"] == "standalone_executable"
        with np.load(item[3] / "fields_eval.npz") as saved:
            assert saved["temperature_K"].shape == (200, 96)
            assert saved["anode_mask"].sum() * .125**2 == pytest.approx(25 * item[2]["anode_width_mm"])
            assert saved["x_mm"].shape == (96,)
            assert saved["y_mm"].shape == (200,)
            assert np.isfinite(saved["current_density_magnitude_A_per_m2"]).all()
