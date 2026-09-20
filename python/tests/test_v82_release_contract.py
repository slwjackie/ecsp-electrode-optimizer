from __future__ import annotations

from pathlib import Path
import copy
import json
import math
import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from ecsp_nsga2.bc_vmin import (
    _monotonicity_violation,
    batched_voltage_search,
    validate_voltage_search_contract,
)
from ecsp_nsga2.evaluator import (
    BCCandidateGeometryError,
    BCGlobalPreflameEvaluator,
    EvaluatorError,
    _bc_solver_converged,
    canonicalise_metrics,
)
from ecsp_nsga2.nsga2 import Individual
from ecsp_nsga2.workflow import NSGA2ElectricalSolidWorkflow
from ecsp_v6.config import load_config
from ecsp_v6.physics.bc_global import BCCandidateBatchError
from ecsp_v6.physics.composition_model import build_composition
from ecsp_v6.physics.electrochem import initial_state
from ecsp_v6.physics.geometry import load_geometry_batch
from verify_release_manifest import verify


ROOT = Path(__file__).resolve().parents[2]
BC_PROFILE_NAMES = (
    "nsga2_bc_global_preflame_propagation_a100.yaml",
    "nsga2_bc_global_preflame_propagation_debug.yaml",
    "nsga2_bc_global_native_cpu8.yaml",
    "nsga2_bc_global_native_a100_batch32.yaml",
    "nsga2_bc_global_native_a100_batch64.yaml",
    "nsga2_bc_global_native_debug.yaml",
)


def _profile(name: str) -> dict:
    return yaml.safe_load((ROOT / "config" / name).read_text(encoding="utf-8"))


def _geometry_batch() -> object:
    """Build through the production resize path, without constructing a solver."""
    evaluator = object.__new__(BCGlobalPreflameEvaluator)
    evaluator.grid_size = 17
    evaluator.domain_size_m = 0.020
    evaluator.minimum_gap_m = 0.002
    evaluator.device = torch.device("cpu")
    anode = np.zeros((17, 17), dtype=bool)
    cathode = np.zeros((17, 17), dtype=bool)
    anode[3:7, 2:5] = True
    cathode[10:14, 12:15] = True
    return evaluator._build_geometry_batch(
        [(anode, cathode, {"geometry_id": "overlay-contract"}, ROOT / "unused")]
    )


def _synthetic_bc_batch_evaluator(tmp_path: Path) -> BCGlobalPreflameEvaluator:
    evaluator = object.__new__(BCGlobalPreflameEvaluator)
    evaluator.internal_batch_size = 8
    evaluator._failed_geometry_row = lambda metadata, output, exc: {
        "geometry_id": metadata["geometry_id"],
        "physicsRejected": True,
        "physicsRejectionReason": str(exc),
    }
    return evaluator


def _synthetic_bc_items(tmp_path: Path, count: int = 4) -> list[tuple]:
    return [
        (None, None, {"geometry_id": f"candidate-{index}"}, tmp_path / str(index))
        for index in range(count)
    ]


def test_reference_bc_batch_propagates_unclassified_device_failure(
    tmp_path: Path,
) -> None:
    evaluator = _synthetic_bc_batch_evaluator(tmp_path)

    def fail_device(items):
        del items
        raise RuntimeError("simulated device loss")

    evaluator._evaluate_chunk = fail_device
    with pytest.raises(RuntimeError, match="simulated device loss"):
        evaluator.evaluate_batch(_synthetic_bc_items(tmp_path))


@pytest.mark.parametrize(
    "exception_factory",
    (
        lambda index: BCCandidateBatchError(
            "typed candidate physics failure", (index,), "test_physics"
        ),
        lambda index: BCCandidateGeometryError(
            f"typed candidate geometry failure at lane {index}"
        ),
    ),
)
def test_reference_bc_batch_isolates_only_typed_candidate_failure(
    tmp_path: Path, exception_factory,
) -> None:
    evaluator = _synthetic_bc_batch_evaluator(tmp_path)

    def isolate(items):
        bad = [
            index
            for index, item in enumerate(items)
            if item[2]["geometry_id"] == "candidate-2"
        ]
        if bad:
            raise exception_factory(bad[0])
        return [
            {"geometry_id": item[2]["geometry_id"], "physicsRejected": False}
            for item in items
        ]

    evaluator._evaluate_chunk = isolate
    rows = evaluator.evaluate_batch(_synthetic_bc_items(tmp_path))
    assert [row["geometry_id"] for row in rows] == [
        "candidate-0",
        "candidate-1",
        "candidate-2",
        "candidate-3",
    ]
    assert [row["physicsRejected"] for row in rows] == [False, False, True, False]


def test_reference_bc_batch_oom_splits_but_singleton_oom_is_fatal(
    tmp_path: Path,
) -> None:
    evaluator = _synthetic_bc_batch_evaluator(tmp_path)
    visits: list[str] = []

    def split_oom(items):
        if len(items) > 1:
            raise torch.OutOfMemoryError("synthetic allocation exhaustion")
        visits.append(items[0][2]["geometry_id"])
        return [{"geometry_id": visits[-1], "physicsRejected": False}]

    evaluator._evaluate_chunk = split_oom
    rows = evaluator.evaluate_batch(_synthetic_bc_items(tmp_path))
    assert [row["geometry_id"] for row in rows] == [
        "candidate-0",
        "candidate-1",
        "candidate-2",
        "candidate-3",
    ]

    def singleton_oom(items):
        del items
        raise torch.OutOfMemoryError("synthetic singleton exhaustion")

    evaluator._evaluate_chunk = singleton_oom
    with pytest.raises(torch.OutOfMemoryError, match="singleton exhaustion"):
        evaluator.evaluate_batch(_synthetic_bc_items(tmp_path, count=1))


def test_reference_handoff_batch_isolates_typed_candidate_failure(
    tmp_path: Path,
) -> None:
    evaluator = _synthetic_bc_batch_evaluator(tmp_path)
    evaluator.voltage = 260.0

    def run_model(items, voltage, **kwargs):
        del voltage, kwargs
        bad = [
            index
            for index, item in enumerate(items)
            if item[2]["geometry_id"] == "candidate-2"
        ]
        if bad:
            raise BCCandidateBatchError(
                "typed handoff failure", bad, "handoff_test"
            )
        return ([{"geometry_id": item[2]["geometry_id"]} for item in items], {})

    evaluator._run_model = run_model
    evaluator._extract_handoff_batch = lambda rows, output, items: [
        {
            "metrics": row,
            "handoff": {"marker": item[2]["geometry_id"]},
        }
        for row, item in zip(rows, items)
    ]
    results = evaluator.evaluate_handoff_batch(_synthetic_bc_items(tmp_path))
    assert [
        result.get("handoffPreparationFailed", False) for result in results
    ] == [False, False, True, False]
    assert results[0]["handoff"]["marker"] == "candidate-0"
    assert results[3]["handoff"]["marker"] == "candidate-3"


def test_reference_handoff_batch_propagates_infrastructure_failure(
    tmp_path: Path,
) -> None:
    evaluator = _synthetic_bc_batch_evaluator(tmp_path)
    evaluator.voltage = 260.0

    def fail(items, voltage, **kwargs):
        del items, voltage, kwargs
        raise RuntimeError("simulated handoff device loss")

    evaluator._run_model = fail
    with pytest.raises(RuntimeError, match="handoff device loss"):
        evaluator.evaluate_handoff_batch(_synthetic_bc_items(tmp_path))


def test_all_bc_profiles_use_one_surface_overlay_contract() -> None:
    for name in BC_PROFILE_NAMES:
        cfg = _profile(name)
        assert cfg["geometry"]["surface_contact_model"] == (
            "overlay_on_full_propellant_domain"
        ), name
        assert cfg["evaluator"]["base_overrides"]["interface"][
            "boundaryCouplingModel"
        ] == "surface_overlay_bv", name
        assert cfg["bc_global"]["continuedElectricalHeatingAfterOnset"] is False, name
        assert cfg["propagation_refinement"]["continued_electrical_heating"] is False, name


def test_debug_profiles_label_their_actual_evaluation_horizon() -> None:
    for name in (
        "nsga2_bc_global_native_debug.yaml",
        "nsga2_bc_global_preflame_propagation_debug.yaml",
    ):
        cfg = _profile(name)
        horizon = float(cfg["bc_global"]["evaluationTime_s"])
        assert float(cfg["physics"]["end_time_s"]) == horizon, name
        assert float(cfg["condensed_ignition"]["reference_time_s"]) == horizon, name
        assert "within_2s" not in cfg["minimum_ignition_voltage_search"][
            "interpretation"
        ], name


def test_release_metadata_scopes_legacy_hybrid_and_scripts_fail_closed() -> None:
    version = json.loads((ROOT / "VERSION.json").read_text(encoding="utf-8"))
    if version["version"].startswith("8.4.2-"):
        assert version["release_directory_name"] == "ECSP_v8_4_2_A100CPU8_GeometrySafe"
        assert version["patch_release_lineage"]["direct_parent_release"] == "8.4.1-A100CPU8-Reactive-Experimental"
        assert version["patch_release_lineage"]["direct_parent_archive_sha256"] == "9d6faba40f66252e5930f9030096e2ecde51d942cb919c7a8dc3940345a09e59"
        contract = version["geometry_v8_4_2"]
        assert contract["global_maximum_width_mm"] == 5.0
        assert contract["active_bc_production_area_relative_tolerance_per_polarity"] == 0.01
        assert contract["bootstrap_invalid_padding"] is False
        assert contract["bootstrap_complete_before_candidate_PDE"] is True
        assert contract["validated_feasible_bootstrap_count"] == 1000
        assert contract["physical_coefficients_changed"] is False
        assert contract["CUDA_hardware_validation_in_build_environment"] is False
    elif version["version"].startswith("8.4.1-"):
        assert version["release_directory_name"] == "ECSP_v8_4_1_A100CPU8"
        assert version["patch_release_lineage"]["direct_parent_release"] == "8.4.0-BCReactive-Condensed-Experimental"
        assert version["patch_release_lineage"]["direct_parent_archive_sha256"] == "924e94b39e5d6cc57ae6b781454053268daf900c8af36d17dee88ed08cc72446"
        assert version["reactive_v8_4_1"]["cuda_hardware_tests_executed_in_build_environment"] is False
        assert version["reactive_v8_4_1"]["dtype"] == "float64"
        assert version["reactive_v8_4_1"]["physical_coefficients_changed"] is False
        assert version["reactive_v8_4_1"]["thermal_pressure_coupling"] is False
    elif version["version"].startswith("8.4.0-"):
        assert version["release_directory_name"] == "ECSP_v8_4_0_BCReactive_Condensed"
        assert version["patch_release_lineage"]["direct_parent_release"] == "8.3.0-PaperReactiveEuler-Experimental"
        assert version["patch_release_lineage"]["direct_parent_archive_sha256"] == (
            "5df08f448453e43d2ee68a0c1bd499b320f00252cc3818299e0f4bc185cf256d"
        )
        assert version["bc_reactive_v8_4"]["default_without_post_onset"] == "condensed_propagation"
        assert version["bc_reactive_v8_4"]["thermal_pressure_coupling"] is False
        assert version["bc_reactive_v8_4"]["new_backend_cuda"] == "NOT_IMPLEMENTED_CPU_FP64_ONLY"
        assert version["paper_reactive_v8_3"]["final_verdict"] == "NOT_FULLY_VERIFIED"
    elif version["version"].startswith("8.3.0-"):
        assert version["release_directory_name"] == (
            "ECSP_v8_3_0_PaperReactiveEuler_Experimental"
        )
        assert version["patch_release_lineage"]["direct_parent_release"] == (
            "8.2.1-P0P1Fixed-BCNative-Hybrid-A100-CPU8"
        )
        assert version["paper_reactive_v8_3"]["final_verdict"] == (
            "NOT_FULLY_VERIFIED"
        )
    else:
        assert version["version"] == "8.2.1-P0P1Fixed-BCNative-Hybrid-A100-CPU8"
        assert version["release_directory_name"] == (
            "ECSP_v8_2_1_P0P1Fixed_BCNative_Hybrid_A100_CPU8"
        )
    assert version["patch_release_lineage"]["baseline_archive_sha256"] == (
        "799cfd001a8e28143e16e33f32424a9545584f2c95691370943c804f210832a6"
    )
    assert "not_executed" in version["a100_validation_status"]
    assert "CUDA_batch" in version["maintenance_corrections_v8_2_1"][
        "mask_storage"
    ]
    assert "hybrid_scheduler" not in version
    assert "hybrid_state_transfer_policy" not in version
    assert "legacy_v7_9_5_non_bc_hybrid_scheduler" in version
    assert "legacy_v7_9_5_non_bc_hybrid_state_transfer_policy" in version

    expected_changes = json.loads(
        (ROOT / "docs/V821_EXPECTED_CHANGES.json").read_text(encoding="utf-8")
    )
    cuda_boundary_reason = expected_changes["allowed_modified"][
        "python/ecsp_cuda/solver.py"
    ]
    assert "uint8 value comparison" in cuda_boundary_reason
    assert "0xff" in cuda_boundary_reason

    validation_script = (ROOT / "tools/validate_v8_2_1_cpu.sh").read_text(
        encoding="utf-8"
    )
    assert "--require-declared-changes" in validation_script
    assert "Missing authoritative PACKAGE_SHA256_MANIFEST_V8_2_1.txt" in validation_script

    preservation_script = (ROOT / "python/audit_v820_preservation.py").read_text(
        encoding="utf-8"
    )
    assert 'default=root / "runs" / "v8_2_1_cpu_validation" / "preservation"' in (
        preservation_script
    )

    build_script = (ROOT / "tools/build_release.sh").read_text(encoding="utf-8")
    assert 'GENERIC_SHA_MANIFEST="$ROOT/PACKAGE_SHA256_MANIFEST.txt"' in build_script
    assert 'ECSP_V820_BASELINE_ZIP must name the immutable regular-file' in build_script
    assert 'bash "$ROOT/tools/validate_v8_2_1_cpu.sh" "$VALIDATION_OUT"' in build_script
    assert 'VALIDATION_COMPLETE.json' in build_script
    assert 'expected_evidence = {' in build_script
    assert '"authoritative_manifest_sha256"' in build_script
    assert 'if validated != current_hashes:' in build_script
    assert 'os.link(source_zip, target_zip)' in build_script
    assert 'unzip -q "$TEMP_ZIP" -d "$VERIFY_DIR"' in build_script
    assert '--package-root "$VERIFY_DIR/$NAME"' in build_script
    assert '"RELEASE_BUILD_COMPLETE.json"' in build_script

    a100_preflight = (ROOT / "tools/run_bc_native_a100_preflight.sh").read_text(
        encoding="utf-8"
    )
    assert "bc_global_native_hybrid" in a100_preflight
    assert "hybrid_schedule.jsonl" in a100_preflight
    assert '{"cpu", "cuda"}.issubset(devices)' in a100_preflight
    assert "A100_PREFLIGHT_COMPLETE.json" in a100_preflight

    for name, expected_kind in (
        ("run_bc_native_a100_cpu8.sh", "corrected_v8_2_native_hybrid"),
        ("run_bc_global_a100.sh", "corrected_v8_2_python_bc_global"),
    ):
        launcher = (ROOT / "tools" / name).read_text(encoding="utf-8")
        assert "ECSP_UNSAFE_BYPASS_REQUIRED_A100_PREFLIGHT" in launcher
        assert "production workdir must not already exist" in launcher
        assert "A100_PRODUCTION_GATE.json" not in launcher  # written by shared helper
        assert "a100_device_gate.py" in launcher
        assert expected_kind in launcher

    legacy_preflight = (ROOT / "tools" / "run_bc_global_a100_preflight.sh").read_text(
        encoding="utf-8"
    )
    assert "rm -rf" not in legacy_preflight
    assert "ECSP_ALLOW_NON_A100_NONCERTIFYING_PREFLIGHT" in legacy_preflight

    device_gate = (ROOT / "python" / "a100_device_gate.py").read_text(
        encoding="utf-8"
    )
    assert '"NVIDIA" in name.upper() and "A100" in name.upper()' in device_gate
    assert "capability == [8, 0]" in device_gate
    assert "nvidia_driver_version" in device_gate

    cuda_check = (ROOT / "python" / "check_bc_native_cuda.py").read_text(
        encoding="utf-8"
    )
    assert "if not report['extension_has_cuda']" in cuda_check


def test_a100_gate_verifies_preflight_and_records_run_metadata(tmp_path: Path) -> None:
    helper = ROOT / "python" / "a100_device_gate.py"
    preflight = tmp_path / "A100_PREFLIGHT_COMPLETE.json"
    (tmp_path / "a100_device_report.json").write_text(
        json.dumps(
            {
                "schema": "ecsp.a100-production-gate/v1",
                "report_type": "selected_cuda_device_identity",
                "status": "passed_verified_nvidia_a100_sm80",
                "a100_device_identity_verified": True,
                "non_certifying_override_used": False,
                "checks": {
                    "device_name_is_nvidia_a100": True,
                    "compute_capability_is_8_0": True,
                    "driver_version_recorded": True,
                },
            }
        ),
        encoding="utf-8",
    )
    preflight.write_text(
        json.dumps(
            {
                "schema": "ecsp.a100-production-gate/v1",
                "report_type": "completed_functional_preflight",
                "preflight_kind": "corrected_v8_2_native_hybrid",
                "created_at_utc": "2026-09-06T00:00:00+00:00",
                "status": "passed",
                "functional_preflight_passed": True,
                "a100_device_identity_verified": True,
                "non_certifying_override_used": False,
                "device_report": "a100_device_report.json",
                "cuda_extension_compiled_and_loaded": True,
                "cuda_extension_has_cuda_entrypoints": True,
                "cuda_test_counts": {
                    "tests": 2,
                    "failures": 0,
                    "errors": 0,
                    "skipped": 0,
                },
                "cuda_test_ids": [
                    "python.tests.test_bc_native::test_native_cuda32_64_batch_size_invariance",
                    "python.tests.test_bc_native::test_native_cuda_cpu_full_field_and_mixed_voltage_parity",
                ],
                "cpu_and_cuda_hybrid_observed": True,
                "performance_or_peak_vram_certified": False,
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(helper),
            "verify-preflight",
            "--report",
            str(preflight),
            "--expected-kind",
            "corrected_v8_2_native_hybrid",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    workdir = tmp_path / "production"
    workdir.mkdir()
    (workdir / "RUN_COMPLETE.json").write_text(
        json.dumps({"elapsed_s": 1.0}), encoding="utf-8"
    )
    subprocess.run(
        [
            sys.executable,
            str(helper),
            "record",
            "--workdir",
            str(workdir),
            "--launcher",
            "test-launcher",
            "--launcher-exit-status",
            "0",
            "--preflight-report",
            str(preflight),
            "--expected-kind",
            "corrected_v8_2_native_hybrid",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    gate = json.loads(
        (workdir / "A100_PRODUCTION_GATE.json").read_text(encoding="utf-8")
    )
    complete = json.loads((workdir / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
    assert gate["required_preflight_executed"] is True
    assert gate["a100_device_identity_verified"] is True
    assert gate["production_run_completed_successfully"] is True
    assert complete["a100_production_gate"] == gate

    incomplete = json.loads(preflight.read_text(encoding="utf-8"))
    incomplete["cuda_test_counts"]["skipped"] = 1
    preflight.write_text(json.dumps(incomplete), encoding="utf-8")
    rejected = subprocess.run(
        [
            sys.executable,
            str(helper),
            "verify-preflight",
            "--report",
            str(preflight),
            "--expected-kind",
            "corrected_v8_2_native_hybrid",
        ],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "tests are incomplete" in rejected.stderr

    incomplete["cuda_test_counts"]["skipped"] = 0
    incomplete["cuda_test_ids"][0] = "python.tests.test_bc_native::unrelated_cuda_test"
    preflight.write_text(json.dumps(incomplete), encoding="utf-8")
    wrong_tests = subprocess.run(
        [
            sys.executable,
            str(helper),
            "verify-preflight",
            "--report",
            str(preflight),
            "--expected-kind",
            "corrected_v8_2_native_hybrid",
        ],
        capture_output=True,
        text=True,
    )
    assert wrong_tests.returncode != 0
    assert "exact required tests" in wrong_tests.stderr


def test_a100_launchers_and_direct_native_launcher_reject_existing_paths(
    tmp_path: Path,
) -> None:
    for script_name in ("run_bc_native_a100_cpu8.sh", "run_bc_global_a100.sh"):
        existing = tmp_path / script_name
        existing.mkdir()
        completed = subprocess.run(
            ["bash", str(ROOT / "tools" / script_name), str(existing), "4", "1"],
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 2
        assert "production workdir must not already exist" in completed.stderr
        assert not Path(f"{existing}.a100_preflight").exists()

    existing_file = tmp_path / "not-a-directory"
    existing_file.write_text("occupied\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "python" / "run_bc_native.py"),
            "--mode",
            "debug",
            "--workdir",
            str(existing_file),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "Use a new empty workdir" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_debug_and_audit_entrypoints_never_delete_existing_runtime_paths(
    tmp_path: Path,
) -> None:
    debug_work = tmp_path / "existing-debug"
    audit_work = tmp_path / "existing-audit-runtime"
    validator_work = tmp_path / "existing-validator-runtime"
    for path in (debug_work, audit_work, validator_work):
        path.mkdir()
        (path / "sentinel.txt").write_text("preserve\n", encoding="utf-8")

    commands = (
        ["bash", str(ROOT / "tools/run_bc_global_debug.sh"), str(debug_work)],
        [
            "bash",
            str(ROOT / "tools/validate_bc_global_pipeline.sh"),
            str(tmp_path / "generated-audit"),
            str(audit_work),
        ],
        [
            sys.executable,
            str(ROOT / "python/validate_bc_global_pipeline.py"),
            "--output-dir",
            str(tmp_path / "direct-generated-audit"),
            "--runtime-workdir",
            str(validator_work),
        ],
    )
    for command, guarded_path in zip(
        commands, (debug_work, audit_work, validator_work)
    ):
        completed = subprocess.run(command, capture_output=True, text=True)
        assert completed.returncode == 2
        assert (guarded_path / "sentinel.txt").read_text(encoding="utf-8") == (
            "preserve\n"
        )


def test_selected_device_gate_distinguishes_a100_from_noncertifying_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import a100_device_gate

    class Properties:
        name = "NVIDIA A100-SXM4-40GB"
        major = 8
        minor = 0
        total_memory = 40 * 1024**3
        multi_processor_count = 108
        uuid = "GPU-test-a100"

    properties = Properties()
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "test-torch"
    fake_torch.version = types.SimpleNamespace(cuda="12.4")
    fake_torch.backends = types.SimpleNamespace(
        cudnn=types.SimpleNamespace(version=lambda: 9010)
    )
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        init=lambda: None,
        current_device=lambda: 0,
        get_device_properties=lambda index: properties,
        device_count=lambda: 1,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        a100_device_gate,
        "_nvidia_smi_rows",
        lambda: (
            "/usr/bin/nvidia-smi",
            [
                {
                    "physical_index": "0",
                    "uuid": "GPU-test-a100",
                    "name": properties.name,
                    "memory_total_MiB": 40960,
                    "driver_version": "555.42",
                }
            ],
            None,
        ),
    )

    report, status = a100_device_gate.collect_device_report(allow_non_a100=False)
    assert status == 0
    assert report["a100_device_identity_verified"] is True
    assert report["non_certifying_override_used"] is False

    properties.name = "NVIDIA H100 80GB HBM3"
    properties.major = 9
    rejected, status = a100_device_gate.collect_device_report(allow_non_a100=False)
    assert status != 0
    assert rejected["a100_device_identity_verified"] is False

    overridden, status = a100_device_gate.collect_device_report(allow_non_a100=True)
    assert status == 0
    assert overridden["a100_device_identity_verified"] is False
    assert overridden["non_certifying_override_used"] is True
    assert "not A100 validation" in " ".join(overridden["limitations"])


def test_selected_device_gate_prefers_torch_uuid_over_numeric_smi_index() -> None:
    import a100_device_gate

    rows = [
        {
            "physical_index": "0",
            "uuid": "GPU-physical-zero",
            "name": "NVIDIA A100-SXM4-40GB",
            "memory_total_MiB": 40960,
            "driver_version": "555.42",
        },
        {
            "physical_index": "1",
            "uuid": "GPU-opened-by-torch",
            "name": "NVIDIA A100-SXM4-40GB",
            "memory_total_MiB": 40960,
            "driver_version": "555.42",
        },
    ]
    selected = a100_device_gate._select_smi_row(
        rows,
        selector="0",
        torch_name="NVIDIA A100-SXM4-40GB",
        total_memory_bytes=40 * 1024**3,
        torch_uuid="GPU-opened-by-torch",
    )
    assert selected is rows[1]


def test_onset_and_initial_temperature_have_one_effective_value() -> None:
    base = load_config(ROOT / "config" / "default_lp_pva.yaml")
    base_t0 = float(base["thermal"]["initialTemperature_K"])
    assert float(base["electrical"]["initialTemperature_K"]) == base_t0
    for name in BC_PROFILE_NAMES:
        cfg = _profile(name)
        assert float(cfg["condensed_ignition"]["onset_temperature_K"]) == float(
            cfg["bc_global"]["onsetCriterion"]["temperature_K"]
        ), name
        assert float(
            cfg["condensed_ignition"]["minimum_conversion_numerical_guard"]
        ) == float(cfg["bc_global"]["onsetCriterion"]["minimum_progress"]), name
        assert float(cfg["bc_global"]["thermal"]["initialTemperature_K"]) == base_t0, name


def test_conflicting_bc_initial_temperature_fails_closed(tmp_path: Path) -> None:
    cfg = copy.deepcopy(_profile("nsga2_bc_global_native_debug.yaml"))
    cfg["evaluator"]["base_overrides"].setdefault("thermal", {})[
        "initialTemperature_K"
    ] = 310.0
    adapter = copy.deepcopy(cfg["evaluator"])
    adapter["physics_config"] = cfg
    with pytest.raises(Exception, match="Conflicting initial temperatures"):
        BCGlobalPreflameEvaluator(
            ROOT, adapter, tmp_path / "conflicting-temperature"
        )


def test_solver_geometry_keeps_propellant_and_species_under_contacts() -> None:
    geometry = _geometry_batch()
    assert bool(torch.all(geometry.propellant))
    assert int(geometry.propellant.sum()) == geometry.grid_size**2

    config = load_config(ROOT / "config" / "default_lp_pva.yaml")
    composition = build_composition(config)
    state = initial_state(geometry, config, composition, 260.0, torch.float64)
    contacts = geometry.anode | geometry.cathode
    assert bool(torch.all(state["cation"][contacts] > 0.0))
    assert bool(torch.all(state["anion"][contacts] > 0.0))
    assert bool(torch.all(state["water"][contacts] > 0.0))
    assert bool(
        torch.all(
            state["temperature"][contacts]
            == float(config["thermal"]["initialTemperature_K"])
        )
    )


def test_legacy_direct_geometry_loader_keeps_embedded_electrode_semantics(
    tmp_path: Path,
) -> None:
    anode = np.zeros((17, 17), dtype=np.uint8)
    cathode = np.zeros((17, 17), dtype=np.uint8)
    anode[3:7, 2:5] = 1
    cathode[10:14, 12:15] = 1
    path = tmp_path / "legacy_geometry.npz"
    np.savez_compressed(path, anodeMask=anode, cathodeMask=cathode)
    rows = pd.DataFrame(
        [{"geometry_id": "legacy-direct", "npz_path": str(path)}]
    )
    geometry, _ = load_geometry_batch(
        rows,
        grid_size=17,
        domain_size_m=0.020,
        minimum_gap_m=0.002,
        reject_resize_shorts=True,
        device=torch.device("cpu"),
        manufacturability_cfg={"projectConstraintsAfterResize": False},
    )
    contacts = geometry.anode | geometry.cathode
    assert torch.equal(geometry.fixed, contacts)
    assert torch.equal(geometry.propellant, ~contacts)


def _objective_row(**extra: float | bool) -> dict:
    row: dict = {
        "condensedPhaseIgnitionDelay_s": 0.2,
        "ignitionSucceeded": True,
        "minimumIgnitionVoltageObjective_V": 100.0,
        "minimumIgnitionVoltageSearchValid": True,
        "peakCurrentCongestion": 2.0,
        "evaluationTime_s": 0.25,
        "modelStatus": "bc_global_corrected",
    }
    row.update(extra)
    return row


def test_evaluation_time_metric_is_canonical_and_at2s_is_deprecated_alias() -> None:
    result = canonicalise_metrics(
        _objective_row(
            remainingReactiveMassFractionAtEvaluationTime=0.37,
            remainingReactiveMassFractionAt2s=0.91,
        ),
        end_time_s=0.25,
        no_ignition_penalty_s=0.25,
    )
    assert result["area_undecomposed_fraction_at_evaluation_time"] == 0.37
    assert result["area_undecomposed_fraction_at_2s"] == 0.37
    assert result["objective_vector"][1] == 0.37
    assert (
        result["objective_definition"]["deprecated_aliases"][
            "area_undecomposed_fraction_at_2s"
        ]
        == "area_undecomposed_fraction_at_evaluation_time"
    )

    legacy = canonicalise_metrics(
        _objective_row(remainingReactiveMassFractionAt2s=0.42),
        end_time_s=0.25,
        no_ignition_penalty_s=0.25,
    )
    assert legacy["area_undecomposed_fraction_at_evaluation_time"] == 0.42
    assert legacy["area_undecomposed_fraction_at_2s"] == 0.42


def test_congestion_objective_prefers_evaluation_horizon_over_legacy_peak() -> None:
    result = canonicalise_metrics(
        _objective_row(
            remainingReactiveMassFractionAtEvaluationTime=0.37,
            peakCurrentCongestion=9.0,
            peakCurrentCongestionToEvaluationTime=2.5,
            current_congestion=7.0,
        ),
        end_time_s=0.5,
        no_ignition_penalty_s=0.5,
    )
    assert result["peakCurrentCongestion"] == 9.0
    assert result["peakCurrentCongestionToEvaluationTime"] == 2.5
    assert result["current_congestion"] == 2.5
    assert result["objective_vector"][3] == 2.5


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("no_ignition_penalty_s", float("nan")),
        ("no_ignition_constraint_violation", -1.0),
        ("minimum_no_ignition_violation", 1.0e-12),
        ("vmin_invalid_search_constraint_violation", 0.0),
    ),
)
def test_workflow_rejects_invalid_failure_penalty_contract_before_evaluator(
    tmp_path: Path, field: str, value: float,
) -> None:
    cfg=copy.deepcopy(_profile("nsga2_bc_global_native_debug.yaml"))
    cfg["optimization"][field]=value
    with pytest.raises(EvaluatorError,match="failure-penalty contract"):
        NSGA2ElectricalSolidWorkflow(ROOT,cfg,tmp_path/field)


def test_required_noncontinuous_ignition_failure_cannot_become_feasible() -> None:
    workflow=object.__new__(NSGA2ElectricalSolidWorkflow)
    workflow.opt={
        "require_ignition_for_feasibility":True,
        "continuous_ignition_constraint":False,
        "numerical_cap_thresholds":{
            "temperature":0.02,"species":0.02,"gas":0.02,"chemical_rate":0.02,
        },
    }
    workflow.end_time_s=1.0
    workflow.no_ignition_penalty_s=1.0
    workflow.no_ignition_constraint_violation=0.0
    workflow.minimum_no_ignition_violation=1.0e-6
    workflow.vmin_invalid_search_constraint_violation=1.0
    workflow.vmin_upper_bound_V=260.0
    workflow.ignition_onset_temperature_K=523.15
    workflow.initial_temperature_K=298.15
    candidate=Individual("missing-onset",{})
    workflow._apply_raw_metrics(
        candidate,
        _objective_row(
            condensedPhaseIgnitionDelay_s=None,
            ignitionSucceeded=False,
            remainingReactiveMassFractionAtEvaluationTime=1.0,
            peakMaximumTemperature_K=298.15,
            converged=True,
            maximumTemperatureCapFraction=0.0,
            maximumSpeciesLimiterFraction=0.0,
            maximumGasCapFraction=0.0,
            maximumChemicalRateCapFraction=0.0,
        ),
    )
    assert candidate.metrics["ignition_constraint_violation"]==1.0e-6
    assert candidate.constraint_violation>1.0e-12


def test_vmin_reports_observed_monotonicity_for_consistent_trials() -> None:
    rows = [{"ignitionSucceeded": True, "converged": True}]

    def run(indices: list[int], volts: list[float], role: str) -> list[dict]:
        del indices, role
        return [
            {"ignitionSucceeded": voltage >= 125.0, "converged": True}
            for voltage in volts
        ]

    batched_voltage_search(
        rows,
        run,
        lambda row: (bool(row["converged"]), "valid"),
        enabled=True,
        low_voltage=20.0,
        high_voltage=260.0,
        tolerance=5.0,
        max_iterations=8,
        invalid_penalty=100.0,
        censor_penalty=50.0,
        verify_final=True,
    )
    assert rows[0]["minimumIgnitionVoltageMonotonicityObserved"] is True
    assert rows[0]["minimumIgnitionVoltageSearchValid"] is True


def test_vmin_does_not_claim_unobserved_monotonicity_when_disabled() -> None:
    rows = [{"ignitionSucceeded": True, "converged": True}]
    batched_voltage_search(
        rows,
        lambda indices, volts, role: [],
        lambda row: (bool(row["converged"]), "valid"),
        enabled=False,
        low_voltage=20.0,
        high_voltage=260.0,
        tolerance=5.0,
        max_iterations=8,
        invalid_penalty=100.0,
        censor_penalty=50.0,
        verify_final=True,
    )
    assert rows[0]["minimumIgnitionVoltageMonotonicityChecked"] is False
    assert rows[0]["minimumIgnitionVoltageMonotonicityObserved"] is None
    assert rows[0]["minimumIgnitionVoltageSearchValid"] is False
    assert rows[0]["minimumIgnitionVoltage_V"] is None
    assert rows[0]["minimumIgnitionVoltageSearchStatus"] == "disabled_search_invalid"


@pytest.mark.parametrize(
    "overrides",
    (
        {"invalid_penalty": -1.0},
        {"censor_penalty": float("nan")},
        {"invalid_penalty": 1.0e308, "high_voltage": 1.0e308},
        {"low_voltage": float("nan")},
        {"tolerance": 0.0},
        {"max_iterations": 0},
        {"max_iterations": 2.5},
        {"tolerance": 1.0, "max_iterations": 1},
    ),
)
def test_vmin_contract_rejects_nonfinite_or_favourable_failure_scores(
    overrides: dict,
) -> None:
    values = {
        "low_voltage": 20.0,
        "high_voltage": 260.0,
        "tolerance": 5.0,
        "max_iterations": 8,
        "invalid_penalty": 100.0,
        "censor_penalty": 50.0,
    }
    values.update(overrides)
    with pytest.raises(ValueError):
        validate_voltage_search_contract(**values)


def test_vmin_contract_rejects_tolerance_below_fp64_bracket_spacing() -> None:
    low_voltage = 1.0e308
    high_voltage = np.nextafter(low_voltage, np.inf)
    representable_spacing = max(math.ulp(low_voltage), math.ulp(high_voltage))

    with pytest.raises(ValueError, match="below the FP64 representable spacing"):
        validate_voltage_search_contract(
            low_voltage=low_voltage,
            high_voltage=high_voltage,
            tolerance=representable_spacing * 0.5,
            max_iterations=2,
            invalid_penalty=0.0,
            censor_penalty=0.0,
        )


def test_vmin_search_keeps_extreme_finite_bracket_representable() -> None:
    high_voltage = 1.0e308
    tolerance = math.ulp(high_voltage)
    ignition_threshold = 7.5e307
    rows = [{"ignitionSucceeded": True, "converged": True}]

    def run(indices: list[int], volts: list[float], role: str) -> list[dict]:
        del indices, role
        return [
            {
                "ignitionSucceeded": voltage >= ignition_threshold,
                "converged": True,
            }
            for voltage in volts
        ]

    batched_voltage_search(
        rows,
        run,
        lambda row: (bool(row["converged"]), "valid"),
        enabled=True,
        low_voltage=0.0,
        high_voltage=high_voltage,
        tolerance=tolerance,
        max_iterations=64,
        invalid_penalty=0.0,
        censor_penalty=0.0,
    )

    result = rows[0]
    assert result["minimumIgnitionVoltageSearchValid"] is True
    assert result["minimumIgnitionVoltageBracketWidth_V"] <= tolerance
    assert result["minimumIgnitionVoltage_V"] >= ignition_threshold
    assert all(
        math.isfinite(trial["voltage_V"])
        for trial in result["minimumIgnitionVoltageTrials"]
    )


@pytest.mark.parametrize(
    "field,value,match",
    (
        ("lower_bound_V", 0.0, "must exceed the cathode voltage"),
        (
            "maximum_bisection_iterations",
            2.5,
            "maximum iterations must be an integer",
        ),
        ("right_censor_objective_penalty_V", -1.0, "penalties must be non-negative"),
    ),
)
def test_bc_evaluator_rejects_invalid_vmin_contract_at_initialisation(
    tmp_path: Path, field: str, value: float, match: str,
) -> None:
    cfg = copy.deepcopy(_profile("nsga2_bc_global_native_debug.yaml"))
    cfg["minimum_ignition_voltage_search"]["enabled"] = True
    cfg["minimum_ignition_voltage_search"][field] = value
    adapter = copy.deepcopy(cfg["evaluator"])
    adapter["physics_config"] = cfg
    with pytest.raises(Exception, match=match):
        BCGlobalPreflameEvaluator(ROOT, adapter, tmp_path / field)


def test_bc_evaluator_forbids_disabling_active_vmin_objective(tmp_path: Path) -> None:
    cfg = copy.deepcopy(_profile("nsga2_bc_global_native_debug.yaml"))
    cfg["minimum_ignition_voltage_search"]["enabled"] = False
    adapter = copy.deepcopy(cfg["evaluator"])
    adapter["physics_config"] = cfg
    with pytest.raises(Exception, match="must be true"):
        BCGlobalPreflameEvaluator(ROOT, adapter, tmp_path / "disabled-vmin")


def test_vmin_repeated_voltage_contradiction_is_invalid_not_a_threshold() -> None:
    rows = [{"ignitionSucceeded": True, "converged": True}]

    def run(indices: list[int], volts: list[float], role: str) -> list[dict]:
        del indices
        return [
            {
                "ignitionSucceeded": False
                if role == "final_upper_full_horizon_verification"
                else voltage >= 125.0,
                "converged": True,
            }
            for voltage in volts
        ]

    batched_voltage_search(
        rows,
        run,
        lambda row: (bool(row["converged"]), "valid"),
        enabled=True,
        low_voltage=20.0,
        high_voltage=260.0,
        tolerance=5.0,
        max_iterations=8,
        invalid_penalty=100.0,
        censor_penalty=50.0,
        verify_final=True,
    )
    assert rows[0]["minimumIgnitionVoltageMonotonicityObserved"] is False
    assert rows[0]["minimumIgnitionVoltageSearchValid"] is False
    assert rows[0]["minimumIgnitionVoltage_V"] is None
    assert "nonmonotonic" in rows[0]["minimumIgnitionVoltageSearchStatus"]


def test_bc_trial_rejects_nonlinear_robin_nonconvergence() -> None:
    solver = {"converged": True}
    assert _bc_solver_converged(
        {
            "allElectricalLinearSolvesConverged": True,
            "allNonlinearRobinSolvesConverged": True,
        },
        solver,
    )
    assert not _bc_solver_converged(
        {
            "allElectricalLinearSolvesConverged": True,
            "allNonlinearRobinSolvesConverged": False,
        },
        solver,
    )
    assert not _bc_solver_converged(
        {
            "allElectricalLinearSolvesConverged": False,
            "allNonlinearRobinSolvesConverged": True,
        },
        solver,
    )


def test_vmin_same_voltage_contradiction_is_order_independent() -> None:
    trials = [
        {"voltage_V": 100.0, "ignited": False, "numerically_valid": True},
        {"voltage_V": 100.0, "ignited": True, "numerically_valid": True},
    ]
    forward = _monotonicity_violation(trials)
    reverse = _monotonicity_violation(list(reversed(trials)))
    assert forward == reverse
    assert forward == {
        "lowerIgnitingVoltage_V": 100.0,
        "higherNonIgnitingVoltage_V": 100.0,
        "sameVoltageContradiction": True,
    }


def test_reference_vmin_uses_shared_monotonicity_and_full_horizon_contract(
    tmp_path: Path,
) -> None:
    evaluator = object.__new__(BCGlobalPreflameEvaluator)
    evaluator.vmin_enabled = True
    evaluator.vmin_lower_bound_V = 20.0
    evaluator.vmin_upper_bound_V = 260.0
    evaluator.vmin_tolerance_V = 5.0
    evaluator.vmin_max_iterations = 8
    evaluator.vmin_invalid_penalty_V = 100.0
    evaluator.vmin_right_censor_penalty_V = 50.0
    evaluator.vmin_verify_final = True
    evaluator.vmin_stop_successful_trials_at_ignition = True
    evaluator._trial_valid = lambda row: (bool(row["converged"]), "valid")
    calls: list[tuple[str, bool]] = []

    def fake_run_model(items, voltage, *, write_metrics, stop_on_onset):
        del write_metrics
        directory = str(items[0][3])
        calls.append((directory, stop_on_onset))
        final_verification = "final_upper_full_horizon_verification" in directory
        return ([{
            "ignitionSucceeded": (
                False if final_verification else float(voltage) >= 125.0
            ),
            "ignitionDelay_s": 0.1,
            "converged": True,
        }], None)

    evaluator._run_model = fake_run_model
    row = {"ignitionSucceeded": True, "converged": True}
    mask = np.ones((2, 2), dtype=bool)
    evaluator._attach_vmin(
        (mask, mask.copy(), {"geometry_id": "shared-vmin"}, tmp_path), row
    )

    assert row["minimumIgnitionVoltageSearchValid"] is False
    assert row["minimumIgnitionVoltageMonotonicityObserved"] is False
    assert "nonmonotonic" in row["minimumIgnitionVoltageSearchStatus"]
    assert any(
        "final_upper_full_horizon_verification" in directory and not early
        for directory, early in calls
    )


def test_reference_vmin_typed_trial_failure_preserves_reference_objectives(
    tmp_path: Path,
) -> None:
    evaluator = object.__new__(BCGlobalPreflameEvaluator)
    evaluator.vmin_enabled = True
    evaluator.vmin_lower_bound_V = 20.0
    evaluator.vmin_upper_bound_V = 260.0
    evaluator.vmin_tolerance_V = 5.0
    evaluator.vmin_max_iterations = 8
    evaluator.vmin_invalid_penalty_V = 100.0
    evaluator.vmin_right_censor_penalty_V = 50.0
    evaluator.vmin_verify_final = True
    evaluator.vmin_stop_successful_trials_at_ignition = True
    evaluator._trial_valid = lambda row: (bool(row["converged"]), "valid")

    def typed_trial_failure(items, voltage, *, write_metrics, stop_on_onset):
        del items, voltage, write_metrics, stop_on_onset
        raise BCCandidateBatchError(
            "typed trial CFL failure", (0,), "thermal_stability_cfl"
        )

    evaluator._run_model = typed_trial_failure
    evaluator._failed_geometry_row = lambda metadata, output, exc: {
        "geometry_id": metadata["geometry_id"],
        "ignitionSucceeded": False,
        "converged": False,
        "physicsRejected": True,
        "physicsRejectionReason": str(exc),
    }
    row = {
        "ignitionSucceeded": True,
        "converged": True,
        "referenceObjectiveSentinel": 12.5,
    }
    mask = np.ones((2, 2), dtype=bool)
    evaluator._attach_vmin(
        (mask, mask.copy(), {"geometry_id": "trial-failure"}, tmp_path), row
    )

    assert row["referenceObjectiveSentinel"] == 12.5
    assert row["minimumIgnitionVoltageSearchValid"] is False
    assert row["minimumIgnitionVoltageSearchStatus"] == "invalid_lower_bound_trial"
    assert row["minimumIgnitionVoltageObjective_V"] == 360.0
    assert row.get("physicsRejected", False) is False
    trial = row["minimumIgnitionVoltageTrials"][-1]
    assert trial["numerically_valid"] is False


def test_release_manifest_verifier_checks_hashes_and_complete_inventory(
    tmp_path: Path,
) -> None:
    import hashlib

    (tmp_path / "VERSION.json").write_text(
        json.dumps({"version": "8.2.0-test"}), encoding="utf-8"
    )
    payload = tmp_path / "payload.txt"
    payload.write_text("validated\n", encoding="utf-8")
    manifest = tmp_path / "PACKAGE_SHA256_MANIFEST_V8_2.txt"
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    version_digest = hashlib.sha256(
        (tmp_path / "VERSION.json").read_bytes()
    ).hexdigest()
    manifest.write_text(
        f"{digest}  payload.txt\n{version_digest}  VERSION.json\n",
        encoding="utf-8",
    )
    generic = tmp_path / "PACKAGE_SHA256_MANIFEST.txt"
    generic.write_bytes(manifest.read_bytes())
    assert verify(tmp_path, manifest)["status"] == "passed"

    generic.write_text("tampered alias\n", encoding="utf-8")
    alias_report = verify(tmp_path, manifest)
    assert alias_report["status"] == "failed"
    assert alias_report["generic_alias_matches_authoritative"] is False
    generic.write_bytes(manifest.read_bytes())

    # The generic same-content alias is excluded to avoid self-reference, but
    # historical or invented versioned manifests remain payload and may not
    # evade complete-inventory verification.
    historical = tmp_path / "PACKAGE_SHA256_MANIFEST_V8_1.txt"
    historical.write_text(
        "historical\n", encoding="utf-8"
    )
    report = verify(tmp_path, manifest)
    assert report["status"] == "failed"
    assert report["unlisted"] == ["PACKAGE_SHA256_MANIFEST_V8_1.txt"]
    historical_digest = hashlib.sha256(historical.read_bytes()).hexdigest()
    manifest.write_text(
        manifest.read_text(encoding="utf-8")
        + f"{historical_digest}  PACKAGE_SHA256_MANIFEST_V8_1.txt\n",
        encoding="utf-8",
    )
    generic.write_bytes(manifest.read_bytes())
    assert verify(tmp_path, manifest)["status"] == "passed"

    extra = tmp_path / "unlisted.txt"
    extra.write_text("not listed\n", encoding="utf-8")
    report = verify(tmp_path, manifest)
    assert report["status"] == "failed"
    assert report["unlisted"] == ["unlisted.txt"]

    extra.unlink()
    payload.write_text("mutated\n", encoding="utf-8")
    report = verify(tmp_path, manifest)
    assert report["status"] == "failed"
    assert [row["path"] for row in report["mismatched"]] == ["payload.txt"]


def test_release_manifest_verifier_separates_v821_from_historical_v820(
    tmp_path: Path,
) -> None:
    import hashlib

    version = tmp_path / "VERSION.json"
    version.write_text(json.dumps({"version": "8.2.1-test"}), encoding="utf-8")
    historical = tmp_path / "PACKAGE_SHA256_MANIFEST_V8_2.txt"
    historical.write_text("immutable v8.2.0 manifest\n", encoding="utf-8")
    payload = tmp_path / "payload.txt"
    payload.write_text("v8.2.1 payload\n", encoding="utf-8")
    authoritative = tmp_path / "PACKAGE_SHA256_MANIFEST_V8_2_1.txt"
    rows = []
    for path in (historical, payload, version):
        rows.append(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        )
    authoritative.write_text("\n".join(rows) + "\n", encoding="utf-8")
    generic = tmp_path / "PACKAGE_SHA256_MANIFEST.txt"
    generic.write_bytes(authoritative.read_bytes())

    report = verify(tmp_path, authoritative)
    assert report["status"] == "passed"
    assert report["manifest"] == "PACKAGE_SHA256_MANIFEST_V8_2_1.txt"
    assert report["generic_alias_matches_authoritative"] is True
