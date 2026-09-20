from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

from ecsp_reactive.validation import (
    FULL_VERIFICATION_VERDICT,
    PASS,
    VALIDATION_SCHEMA,
    run_validation,
)


EXPECTED_CHECKS = {
    "tait_dpdrho_equals_c2",
    "primitive_conservative_roundtrip",
    "free_stream_and_conservation",
    "reaction_heat_progress_ratio",
    "linear_potential_joule_energy",
    "periodic_thermal_integral",
    "level_set_translation",
    "planar_front_regression_thresholds",
    "weno_smooth_spatial_convergence",
    "ssprk3_temporal_convergence",
    "coupled_solver_grid_and_cfl",
    "deterministic_repeats",
    "row_decomposition_emulated_halo_parity",
}


def test_validation_schema_and_manufactured_checks_pass() -> None:
    report = run_validation(deterministic_repeats=3)
    assert report["schema"] == VALIDATION_SCHEMA
    assert set(report["checks"]) == EXPECTED_CHECKS
    assert all(row["status"] == PASS for row in report["checks"].values())
    summary = report["runnable_cpu_verification"]
    assert summary == {
        "status": PASS,
        "passed_check_count": len(EXPECTED_CHECKS),
        "failed_check_count": 0,
        "total_check_count": len(EXPECTED_CHECKS),
        "scope": "manufactured_and_discrete_reference_checks_only",
    }
    assert report["checks"]["deterministic_repeats"]["metrics"]["repeat_count"] == 3
    # Strict JSON is part of the evidence contract: NaN/Infinity is forbidden.
    json.dumps(report, allow_nan=False)


def test_validation_never_promotes_missing_capabilities_to_pass() -> None:
    report = run_validation(deterministic_repeats=2)
    assert report["verdict"] == FULL_VERIFICATION_VERDICT == "NOT_FULLY_VERIFIED"
    assert report["paper_reproduction"]["status"] == "BLOCKED_MISSING_INPUTS"
    assert report["paper_reproduction"]["fitting_attempted"] is False
    assert report["capability_validation"]["cuda_backend"]["status"] == "NOT_IMPLEMENTED"
    assert (
        report["capability_validation"]["a100_execution"]["status"]
        == "NOT_TESTED_NO_HARDWARE"
    )
    mpi = report["capability_validation"]["actual_mpi"]
    runtime_available = (
        importlib.util.find_spec("mpi4py") is not None
        and (shutil.which("mpiexec") is not None or shutil.which("mpirun") is not None)
    )
    if runtime_available:
        assert mpi["status"] == "NOT_TESTED_MULTI_RANK_NOT_RUN"
    else:
        assert mpi["status"] == "NOT_TESTED_NO_MPI_RUNTIME"
    assert mpi["status"] != PASS


def test_coupled_grid_cfl_results_do_not_masquerade_as_paper_convergence() -> None:
    report = run_validation(deterministic_repeats=2)
    row = report["checks"]["coupled_solver_grid_and_cfl"]
    assert row["status"] == PASS
    assert row["scope"]["status"] == "SYNTHETIC_MANUFACTURED_NOT_PAPER_REPRODUCTION"
    assert row["scope"]["paper_parameter_fitting"] is False
    assert row["scope"]["physical_state_clipping"] == "NONE"
    assert row["scope"]["quantity_scope"]["regression_rate"].startswith(
        "passively_advected"
    )
    assert all(
        value.startswith("NOT_VERIFIED")
        for value in row["scope"]["limitations"].values()
    )
    grid = row["grid_refinement"]
    assert [run["nx"] for run in grid["runs"]] == [20, 40, 80]
    assert min(grid["flow_temperature_observed_orders"]) >= 2.5
    assert min(grid["flow_total_energy_field_observed_orders"]) >= 2.5
    assert min(grid["species_front_speed_observed_orders"]) >= 1.0
    assert grid["pressure_order_status"].startswith("EXACT_INVARIANT")
    assert grid["energy_balance_order_status"].startswith("ROUNDOFF_BOUNDED")
    cfl = row["cfl_refinement"]
    assert [run["cfl"] for run in cfl["runs"]] == [0.4, 0.2, 0.1]
    assert cfl["reference_cfl"] == 0.025
    assert cfl["reference_status"].startswith("SMALL_CFL_SAME_SPATIAL_OPERATOR")
    assert min(cfl["temperature_observed_orders"]) >= 2.8
    assert min(cfl["total_energy_field_observed_orders"]) >= 2.8
    assert min(
        min(orders)
        for orders in cfl["regression_speed_observed_orders_by_threshold"].values()
    ) >= 2.5
    assert cfl["pressure_order_status"].startswith("EXACT_INVARIANT")
    assert row["metrics"]["positivity_face_fallback_count_all_runs"] == 0
    assert (
        row["metrics"][
            "positivity_face_fallback_max_abs_state_correction_all_runs"
        ]
        == 0.0
    )
    assert row["metrics"]["cell_state_clipping_count_all_runs"] == 0
    assert (
        row["metrics"][
            "density_pressure_temperature_floor_application_count_all_runs"
        ]
        == 0
    )
    assert row["metrics"]["reaction_rate_cap_application_count_all_runs"] == 0
    assert row["metrics"]["reinitialization_count_all_runs"] == 0

    paper = report["convergence_scope"]["paper_ecsp_case"]
    assert paper["status"] == "NOT_VERIFIED_MISSING_PAPER_INPUTS"
    assert paper["paper_reported_grid_family"] == [50, 100, 150, 200]
    assert paper["paper_cfl_values"] == "NOT_PUBLISHED"
    assert {
        paper["regression_rate"],
        paper["temperature"],
        paper["pressure"],
        paper["energy_balance"],
    } == {"NOT_VERIFIED"}
    assert report["numerical_safety"]["physical_state_clipping"] == (
        "NONE_IN_REACTIVE_SOLVER"
    )


def test_validation_cli_writes_same_strict_json_contract(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "reactive_validation.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "tools" / "validate_reactive_v8_3.py"),
            "--deterministic-repeats",
            "2",
            "--output",
            str(output),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    stdout_report = json.loads(completed.stdout)
    assert report == stdout_report
    assert report["schema"] == VALIDATION_SCHEMA
    assert report["runnable_cpu_verification"]["status"] == PASS
    assert report["verdict"] == "NOT_FULLY_VERIFIED"
