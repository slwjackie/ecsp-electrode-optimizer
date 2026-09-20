"""Resume orchestration contracts; fixtures never run B/C or post-onset physics."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ecsp_nsga2 import workflow as workflow_module
from ecsp_nsga2.geometry import RasterizedGeometry
from ecsp_nsga2.nsga2 import Individual
from ecsp_nsga2.propagation import ConfiguredModelTemperatureRangeExceeded


ROOT = Path(__file__).resolve().parents[2]
FIELDS = (
    "temperatureAtOnset_K", "globalProgressAtOnset", "alphaChannel1AtOnset",
    "alphaChannel2AtOnset", "cationAtOnset_mol_per_m3", "anionAtOnset_mol_per_m3",
    "mobileLPAtOnset_mol_per_m3", "mobileWaterAtOnset_mol_per_m3",
    "pvaReactiveRepeatAtOnset_mol_per_m3", "generatedWaterProductAtOnset_mol_per_m3",
    "electrochemicalLPConsumedAtOnset_mol_per_m3", "potentialAtOnset_V",
    "qJAtOnset_W_per_m3", "qEchemAtOnset_W_per_m3", "propellantMask",
    "anodeContactMask", "cathodeContactMask",
)


def persist(output, metadata):
    output.mkdir(parents=True, exist_ok=True)
    fields = {key: np.ones((3, 3)) for key in FIELDS}
    fields["temperatureAtOnset_K"][:] = 2500.0
    np.savez(output / "bc_handoff_fields.npz", **fields)
    scalars = dict(metadata, field_file="bc_handoff_fields.npz", onsetSucceeded=True,
                   numericallyValidForPropagationHandoff=True, ignitionDelay_s=0.25)
    (output / "bc_handoff_metadata.json").write_text(json.dumps(scalars))
    return {"handoff": {**fields, **scalars}, "metrics": dict(scalars)}


@pytest.fixture
def reuse(monkeypatch):
    # This historical CLI intentionally replaces Workflow.run on import. Keep
    # that CLI behavior local to this fixture, so the NSGA-II suite is untouched.
    workflow = workflow_module.NSGA2ElectricalSolidWorkflow
    monkeypatch.setattr(workflow, "run", workflow.run)
    monkeypatch.setattr("sys.argv", ["resume_final_only.py"])
    spec = importlib.util.spec_from_file_location(
        "_resume_final_handoff_test", ROOT / "tools/resume_final_only.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = {"one": 0, "batch": 0}

    class Evaluator:
        def evaluate_handoff(self, anode, cathode, metadata, output):
            calls["one"] += 1
            return self.evaluate_handoff_batch([(anode, cathode, metadata, output)])[0]

        def evaluate_handoff_batch(self, items):
            calls["batch"] += 1
            return [persist(Path(output), metadata) for _, _, metadata, output in items]

    module._rh_install_patch([Evaluator])
    return module, Evaluator(), calls


def item(tmp_path, *, baseline=False, output=None):
    metadata = {"geometry_id": "staggered" if baseline else "DOE_S1_T000_V000"}
    if baseline:
        metadata["baseline_type"] = "area_matched_staggered"
    mask = np.zeros((3, 3), bool)
    return mask, mask, metadata, output or tmp_path / metadata["geometry_id"] / "bc_handoff"


@pytest.mark.parametrize("batch", [False, True])
def test_candidate_persisted_handoff_never_recomputes(reuse, tmp_path, batch):
    _, evaluator, calls = reuse
    payload = item(tmp_path)
    persist(payload[3], payload[2])
    result = (evaluator.evaluate_handoff_batch([payload])[0] if batch
              else evaluator.evaluate_handoff(*payload))
    assert calls == {"one": 0, "batch": 0}
    assert result["reusedPersistedHandoff"] is True
    assert result["metrics"]["reusedPersistedHandoff"] is True
    np.testing.assert_array_equal(result["handoff"]["temperatureAtOnset_K"], 2500.0)


@pytest.mark.parametrize("batch", [False, True])
def test_candidate_missing_handoff_fails_closed(reuse, tmp_path, batch):
    _, evaluator, calls = reuse
    payload = item(tmp_path)
    with pytest.raises(RuntimeError, match="--reuse-handoff missing"):
        if batch:
            evaluator.evaluate_handoff_batch([payload])
        else:
            evaluator.evaluate_handoff(*payload)
    assert calls == {"one": 0, "batch": 0}


@pytest.mark.parametrize("batch", [False, True])
def test_staggered_missing_handoff_uses_original_once_then_reuses(reuse, tmp_path, batch):
    _, evaluator, calls = reuse
    payload = item(tmp_path, baseline=True)
    run = (lambda: evaluator.evaluate_handoff_batch([payload])[0]) if batch else (
        lambda: evaluator.evaluate_handoff(*payload))
    result = run()
    assert calls == {"one": 1, "batch": 1}
    assert result["reusedPersistedHandoff"] is False
    assert result["metrics"]["reusedPersistedHandoff"] is False
    saved = json.loads((payload[3] / "bc_handoff_metadata.json").read_text())
    assert saved["reusedPersistedHandoff"] is False
    assert run()["reusedPersistedHandoff"] is True
    assert calls == {"one": 1, "batch": 1}


@pytest.mark.parametrize("batch", [False, True])
def test_staggered_persisted_handoff_never_recomputes(reuse, tmp_path, batch):
    _, evaluator, calls = reuse
    payload = item(tmp_path, baseline=True)
    persist(payload[3], payload[2])
    result = (evaluator.evaluate_handoff_batch([payload])[0] if batch
              else evaluator.evaluate_handoff(*payload))
    assert calls == {"one": 0, "batch": 0}
    assert result["reusedPersistedHandoff"] is True


@pytest.mark.parametrize("relative", [
    "final/area_matched_staggered_propagation/bc_handoff",
    "final/propagation/area_matched_staggered/bc_handoff",
])
def test_exact_baseline_output_path_authorizes_fresh_handoff(reuse, tmp_path, relative):
    _, evaluator, calls = reuse
    result = evaluator.evaluate_handoff(*item(tmp_path, output=tmp_path / relative))
    assert result["reusedPersistedHandoff"] is False
    assert calls == {"one": 1, "batch": 1}


def test_baseline_role_authorizes_without_geometry_name_parsing(reuse, tmp_path):
    _, evaluator, calls = reuse
    payload = item(tmp_path)
    payload[2]["source_role"] = "post_optimization_area_matched_staggered_reference"
    assert evaluator.evaluate_handoff(*payload)["reusedPersistedHandoff"] is False
    assert calls == {"one": 1, "batch": 1}


def test_baseline_like_candidate_name_is_not_authorization(reuse, tmp_path):
    _, evaluator, calls = reuse
    payload = item(tmp_path, output=tmp_path / "not_area_matched_staggered_propagation" / "bc_handoff")
    payload[2]["geometry_id"] = "area_matched_staggered"
    with pytest.raises(RuntimeError, match="--reuse-handoff missing"):
        evaluator.evaluate_handoff(*payload)
    assert calls == {"one": 0, "batch": 0}


@pytest.mark.parametrize("corruption", ["missing_fields", "missing_metadata", "wrong_geometry", "invalid_onset"])
def test_incomplete_or_invalid_baseline_is_not_permission_to_rerun(reuse, tmp_path, corruption):
    _, evaluator, calls = reuse
    payload = item(tmp_path, baseline=True)
    persist(payload[3], payload[2])
    meta_file = payload[3] / "bc_handoff_metadata.json"
    if corruption == "missing_fields":
        (payload[3] / "bc_handoff_fields.npz").unlink()
    elif corruption == "missing_metadata":
        meta_file.unlink()
    else:
        data = json.loads(meta_file.read_text())
        data["geometry_id" if corruption == "wrong_geometry" else "onsetSucceeded"] = (
            "wrong" if corruption == "wrong_geometry" else False)
        meta_file.write_text(json.dumps(data))
    with pytest.raises(RuntimeError):
        evaluator.evaluate_handoff(*payload)
    assert calls == {"one": 0, "batch": 0}


def test_patch_snapshots_inheritance_and_hybrid_dispatch_before_mutation(reuse, tmp_path):
    module, _, _ = reuse
    calls = {"one": 0, "native_batch": 0, "hybrid_batch": 0}

    class Base:
        def evaluate_handoff(self, *args):
            calls["one"] += 1
            return self.evaluate_handoff_batch([args])[0]

        def evaluate_handoff_batch(self, items):
            raise AssertionError("native override should dispatch")

    class Native(Base):
        def evaluate_handoff_batch(self, items):
            calls["native_batch"] += 1
            return [persist(Path(output), meta) for _, _, meta, output in items]

    class Hybrid(Base):
        def evaluate_handoff_batch(self, items):
            calls["hybrid_batch"] += 1
            return self.native.evaluate_handoff_batch(items)

    module._rh_install_patch([Base, Native, Hybrid])
    module._rh_install_patch([Base, Native, Hybrid])  # Idempotent installation.
    hybrid = Hybrid()
    hybrid.native = Native()
    result = hybrid.evaluate_handoff(*item(tmp_path, baseline=True))
    assert result["reusedPersistedHandoff"] is False
    assert calls == {"one": 1, "native_batch": 1, "hybrid_batch": 1}


@pytest.mark.parametrize("baseline", [False, True])
def test_existing_propagation_validity_rule_is_identical_for_both_roles(tmp_path, monkeypatch, baseline):
    """Both roles flow through the same production validity exception boundary."""
    from ecsp_nsga2 import post_onset
    workflow = workflow_module.NSGA2ElectricalSolidWorkflow
    handoff = {"onsetSucceeded": True, "temperatureAtOnset_K": np.full((3, 3), 2500.0)}
    bc_config = {"thermal": {"maximumTemperature_K": 2500.0}}
    post_config = {"backend": "reactive_euler", "compare_backends": False}
    propagation_config = {"duration_s": 0.01}
    calls = []

    def reject(h, p, b, output, **kwargs):
        calls.append((h, p, b, kwargs["post_onset_config"]))
        raise ConfiguredModelTemperatureRangeExceeded(2500.0, 2500.001, 4, 1.42e-10)

    monkeypatch.setattr(post_onset, "run_post_onset", reject)
    fake = SimpleNamespace(
        evaluator=SimpleNamespace(config={"bcGlobal": bc_config}),
        propagation_cfg=propagation_config, post_onset_cfg=post_config,
        _propagation_metadata=lambda *_args, **_kwargs: {},
        _resolved_propagation_config=lambda: propagation_config,
        _validate_handoff_reevaluation=lambda *_: {},
    )
    individual = Individual("staggered" if baseline else "candidate", {}, np.ones(4))
    raster = RasterizedGeometry(np.zeros((3, 3), bool), np.zeros((3, 3), bool), {}, 0.0, {})
    result = workflow._run_propagation_refinement(
        fake, individual, raster, tmp_path, baseline=baseline,
        handoff_result={"handoff": handoff})
    assert len(calls) == 1
    assert calls[0][0] is handoff
    assert calls[0][1:] == (propagation_config, bc_config, post_config)
    assert result["postOnsetModelValid"] is False
    assert result["propagationSucceeded"] is False
    assert result["postOnsetValidityReason"] == "configured_model_temperature_range_exceeded"
    assert result["configuredMaximumTemperature_K"] == 2500.0
    assert bc_config["thermal"]["maximumTemperature_K"] == 2500.0
