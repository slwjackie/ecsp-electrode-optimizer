#!/usr/bin/env python3
"""Fail-closed A100 identity checks and production-gate provenance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


SCHEMA = "ecsp.a100-production-gate/v1"
REQUIRED_NATIVE_CUDA_TEST_IDS = (
    "python.tests.test_bc_native::test_native_cuda32_64_batch_size_invariance",
    "python.tests.test_bc_native::test_native_cuda_cpu_full_field_and_mixed_voltage_parity",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _cuda_visible_selector(logical_index: int) -> str:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw and raw.lower() not in {"all", "void", "none"}:
        tokens = [token.strip() for token in raw.split(",") if token.strip()]
        if logical_index < len(tokens):
            return tokens[logical_index]
    return str(logical_index)


def _nvidia_smi_rows() -> tuple[str | None, list[dict[str, Any]], str | None]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None, [], "nvidia-smi was not found on PATH"
    command = [
        executable,
        "--query-gpu=index,uuid,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:  # pragma: no cover - depends on the GPU host
        return executable, [], f"nvidia-smi query failed: {type(exc).__name__}: {exc}"

    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 4)]
        if len(fields) != 5:
            return executable, [], f"Unexpected nvidia-smi CSV row: {line!r}"
        index, uuid, name, memory_mib, driver_version = fields
        try:
            parsed_memory: int | None = int(float(memory_mib))
        except ValueError:
            parsed_memory = None
        rows.append(
            {
                "physical_index": index,
                "uuid": uuid or None,
                "name": name,
                "memory_total_MiB": parsed_memory,
                "driver_version": driver_version or None,
            }
        )
    if not rows:
        return executable, [], "nvidia-smi returned no GPU rows"
    return executable, rows, None


def _select_smi_row(
    rows: list[dict[str, Any]],
    selector: str,
    torch_name: str,
    total_memory_bytes: int,
    torch_uuid: str | None,
) -> dict[str, Any] | None:
    selector_lower = selector.lower()
    # PyTorch's UUID identifies the device that it actually opened.  A numeric
    # CUDA_VISIBLE_DEVICES token is only an ordinal in the CUDA runtime's
    # enumeration, which is not guaranteed to match nvidia-smi's physical
    # index (for example when CUDA_DEVICE_ORDER or container remapping differs).
    # Prefer the runtime UUID whenever it is available.
    if torch_uuid:
        uuid_lower = torch_uuid.lower()
        exact = [
            row
            for row in rows
            if str(row.get("uuid") or "").lower() == uuid_lower
        ]
        if len(exact) == 1:
            return exact[0]
    if selector.isdigit():
        exact = [row for row in rows if str(row["physical_index"]) == selector]
        if len(exact) == 1:
            return exact[0]
    if selector_lower.startswith(("gpu-", "mig-")):
        exact = [
            row
            for row in rows
            if str(row.get("uuid") or "").lower() == selector_lower
            or str(row.get("uuid") or "").lower().startswith(selector_lower)
            or selector_lower.startswith(str(row.get("uuid") or "").lower())
        ]
        if len(exact) == 1:
            return exact[0]
    if len(rows) == 1:
        return rows[0]

    expected_mib = total_memory_bytes / (1024.0 * 1024.0)
    matching = [
        row
        for row in rows
        if str(row.get("name", "")).strip() == torch_name.strip()
        and row.get("memory_total_MiB") is not None
        and abs(float(row["memory_total_MiB"]) - expected_mib) <= 256.0
    ]
    return matching[0] if len(matching) == 1 else None


def collect_device_report(*, allow_non_a100: bool) -> tuple[dict[str, Any], int]:
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "report_type": "selected_cuda_device_identity",
        "created_at_utc": _utc_now(),
        "non_certifying_override_requested": bool(allow_non_a100),
        "status": "failed",
        "a100_device_identity_verified": False,
        "non_certifying_override_used": False,
    }
    try:
        import torch

        report["software"] = {
            "python_version": sys.version.split()[0],
            "torch_version": str(torch.__version__),
            "torch_cuda_runtime_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        }
        if torch.version.cuda is None:
            report["failure_reason"] = "Installed PyTorch has no CUDA runtime support"
            return report, 2
        if not torch.cuda.is_available():
            report["failure_reason"] = "torch.cuda.is_available() is false"
            return report, 2

        torch.cuda.init()
        logical_index = int(torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(logical_index)
        torch_uuid_raw = getattr(properties, "uuid", None)
        torch_uuid = str(torch_uuid_raw) if torch_uuid_raw is not None else None
        capability = [int(properties.major), int(properties.minor)]
        name = str(properties.name)
        total_memory = int(properties.total_memory)
        selector = _cuda_visible_selector(logical_index)
        report["selected_device"] = {
            "torch_logical_index": logical_index,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "visible_selector_for_logical_index": selector,
            "name": name,
            "torch_uuid": torch_uuid,
            "compute_capability": capability,
            "total_memory_bytes": total_memory,
            "multiprocessor_count": int(properties.multi_processor_count),
        }
        report["cuda_device_count_visible_to_torch"] = int(torch.cuda.device_count())

        smi_path, smi_rows, smi_error = _nvidia_smi_rows()
        selected_smi = _select_smi_row(
            smi_rows, selector, name, total_memory, torch_uuid
        )
        driver_version = (
            selected_smi.get("driver_version") if selected_smi is not None else None
        )
        if driver_version is None:
            versions = {
                str(row["driver_version"])
                for row in smi_rows
                if row.get("driver_version")
            }
            if len(versions) == 1:
                driver_version = versions.pop()
        report["driver"] = {
            "nvidia_smi_path": smi_path,
            "nvidia_driver_version": driver_version,
            "query_error": smi_error,
        }
        report["nvidia_smi"] = {
            "selected_gpu": selected_smi,
            "all_query_rows": smi_rows,
            "selected_uuid": (
                selected_smi.get("uuid") if selected_smi is not None else torch_uuid
            ),
            "selected_physical_index": (
                selected_smi.get("physical_index")
                if selected_smi is not None
                else None
            ),
        }

        name_match = "NVIDIA" in name.upper() and "A100" in name.upper()
        capability_match = capability == [8, 0]
        hardware_match = bool(name_match and capability_match)
        report["checks"] = {
            "cuda_enabled_torch": True,
            "cuda_device_available": True,
            "device_name_is_nvidia_a100": name_match,
            "compute_capability_is_8_0": capability_match,
            "driver_version_recorded": driver_version is not None,
        }
        if driver_version is None:
            report["failure_reason"] = (
                "NVIDIA driver version could not be recorded; nvidia-smi evidence is required"
            )
            return report, 3
        if hardware_match:
            report["status"] = "passed_verified_nvidia_a100_sm80"
            report["a100_device_identity_verified"] = True
            return report, 0
        if allow_non_a100:
            report["status"] = "passed_non_a100_non_certifying_override"
            report["non_certifying_override_used"] = True
            report["limitations"] = [
                "This run is a functional CUDA check only.",
                "It is not A100 validation or A100 certification.",
            ]
            return report, 0
        report["failure_reason"] = (
            f"Selected CUDA device must be NVIDIA A100 with compute capability 8.0; "
            f"observed name={name!r}, capability={capability}"
        )
        return report, 4
    except Exception as exc:  # pragma: no cover - depends on the GPU host
        report["failure_reason"] = f"CUDA device inspection failed: {type(exc).__name__}: {exc}"
        return report, 2


def _load_preflight(path: Path, expected_kind: str | None = None) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Cannot read preflight completion report {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Preflight completion report must be a JSON object")
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"Unexpected preflight schema: {payload.get('schema')!r}")
    if payload.get("report_type") != "completed_functional_preflight":
        raise ValueError("Report is not a completed functional preflight")
    if payload.get("status") != "passed" or payload.get("functional_preflight_passed") is not True:
        raise ValueError("Functional preflight is not recorded as passed")
    if expected_kind is not None and payload.get("preflight_kind") != expected_kind:
        raise ValueError(
            f"Expected preflight kind {expected_kind!r}, got {payload.get('preflight_kind')!r}"
        )
    verified = payload.get("a100_device_identity_verified") is True
    override = payload.get("non_certifying_override_used") is True
    if verified == override:
        raise ValueError(
            "Exactly one of verified-A100 or non-certifying override must be recorded"
        )
    device_reference = payload.get("device_report")
    if not isinstance(device_reference, str) or not device_reference:
        raise ValueError("Completed preflight does not reference its device report")
    device_path = (path.parent / device_reference).resolve()
    if device_path.parent != path.parent.resolve():
        raise ValueError("Device report must be a sibling of the preflight completion report")
    try:
        device = json.loads(device_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Cannot read linked device report {device_path}: {exc}") from exc
    if device.get("schema") != SCHEMA or device.get("report_type") != "selected_cuda_device_identity":
        raise ValueError("Linked device identity report has the wrong schema or type")
    if (device.get("a100_device_identity_verified") is True) != verified:
        raise ValueError("Preflight and device report disagree on verified-A100 status")
    if (device.get("non_certifying_override_used") is True) != override:
        raise ValueError("Preflight and device report disagree on non-certifying override status")
    if verified:
        checks = device.get("checks", {})
        required_device_checks = (
            checks.get("device_name_is_nvidia_a100") is True,
            checks.get("compute_capability_is_8_0") is True,
            checks.get("driver_version_recorded") is True,
        )
        if not all(required_device_checks):
            raise ValueError("Verified-A100 device report lacks required identity evidence")

    kind = payload.get("preflight_kind")
    if payload.get("performance_or_peak_vram_certified") is not False:
        raise ValueError("Functional preflight must not claim performance or peak-VRAM certification")
    if kind == "corrected_v8_2_native_hybrid":
        if payload.get("cuda_extension_compiled_and_loaded") is not True:
            raise ValueError("Native preflight did not compile and load the CUDA extension")
        if payload.get("cuda_extension_has_cuda_entrypoints") is not True:
            raise ValueError("Native preflight did not verify CUDA entrypoints")
        if payload.get("cpu_and_cuda_hybrid_observed") is not True:
            raise ValueError("Native preflight did not observe both CPU and CUDA scheduling")
        counts = payload.get("cuda_test_counts", {})
        test_ids = payload.get("cuda_test_ids")
        if (
            not isinstance(counts, dict)
            or int(counts.get("tests", 0)) != len(REQUIRED_NATIVE_CUDA_TEST_IDS)
            or any(int(counts.get(key, 0)) != 0 for key in ("failures", "errors", "skipped"))
        ):
            raise ValueError(f"Native CUDA parity/batch tests are incomplete: {counts!r}")
        if not isinstance(test_ids, list) or tuple(sorted(test_ids)) != REQUIRED_NATIVE_CUDA_TEST_IDS:
            raise ValueError(
                "Native CUDA preflight did not execute the exact required tests: "
                f"{test_ids!r}"
            )
    elif kind == "corrected_v8_2_python_bc_global":
        if payload.get("physics_device_reported_by_workflow") != "cuda":
            raise ValueError("Python B/C preflight did not report physicsDevice=cuda")
        if payload.get("bc_global_preflame_used") is not True:
            raise ValueError("Python B/C preflight did not exercise B/C pre-flame physics")
        if payload.get("gas_phase_cfd_used") is not False:
            raise ValueError("Python B/C preflight unexpectedly reports gas-phase CFD")
    else:
        raise ValueError(f"Unsupported preflight kind: {kind!r}")
    return payload


def _minimal_runtime_device_identity() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda_available": False}
        index = int(torch.cuda.current_device())
        props = torch.cuda.get_device_properties(index)
        return {
            "cuda_available": True,
            "torch_logical_index": index,
            "name": str(props.name),
            "compute_capability": [int(props.major), int(props.minor)],
            "total_memory_bytes": int(props.total_memory),
            "torch_version": str(torch.__version__),
            "torch_cuda_runtime_version": torch.version.cuda,
        }
    except Exception as exc:  # pragma: no cover - depends on the GPU host
        return {"inspection_error": f"{type(exc).__name__}: {exc}"}


def _record_gate(args: argparse.Namespace) -> int:
    workdir = args.workdir.resolve()
    if not workdir.is_dir():
        raise ValueError(f"Production workdir does not exist, so gate metadata cannot be recorded: {workdir}")

    if args.unsafe_bypass:
        gate: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "unsafe_required_preflight_bypassed",
            "required_preflight_executed": False,
            "a100_device_identity_verified": False,
            "non_certifying": True,
            "warning": "Required A100 functional preflight was explicitly bypassed.",
            "runtime_device_identity": _minimal_runtime_device_identity(),
        }
    else:
        if args.preflight_report is None:
            raise ValueError("--preflight-report is required unless --unsafe-bypass is set")
        preflight = _load_preflight(args.preflight_report, args.expected_kind)
        verified = preflight["a100_device_identity_verified"] is True
        gate = {
            "schema": SCHEMA,
            "status": (
                "passed_required_preflight_on_verified_a100"
                if verified
                else "passed_required_preflight_with_non_a100_non_certifying_override"
            ),
            "required_preflight_executed": True,
            "a100_device_identity_verified": verified,
            "non_certifying": not verified,
            "preflight_report": str(args.preflight_report.resolve()),
            "preflight_kind": preflight["preflight_kind"],
            "preflight_created_at_utc": preflight.get("created_at_utc"),
        }
    gate.update(
        {
            "recorded_at_utc": _utc_now(),
            "launcher": args.launcher,
            "launcher_exit_status": int(args.launcher_exit_status),
            "production_run_completed_successfully": int(args.launcher_exit_status) == 0,
            "performance_or_peak_vram_certified": False,
        }
    )
    _write_json(workdir / "A100_PRODUCTION_GATE.json", gate)

    complete_path = workdir / "RUN_COMPLETE.json"
    if complete_path.is_file():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if not isinstance(complete, dict):
            raise ValueError("RUN_COMPLETE.json must contain a JSON object")
        complete["a100_production_gate"] = gate
        _write_json(complete_path, complete)
    elif int(args.launcher_exit_status) == 0:
        raise ValueError("Successful launcher did not create RUN_COMPLETE.json")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Inspect and verify the selected CUDA device")
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--allow-non-a100", action="store_true")

    verify = subparsers.add_parser("verify-preflight", help="Fail unless a preflight completion report is valid")
    verify.add_argument("--report", type=Path, required=True)
    verify.add_argument("--expected-kind", required=True)

    record = subparsers.add_parser("record", help="Write the production gate result into run metadata")
    record.add_argument("--workdir", type=Path, required=True)
    record.add_argument("--launcher", required=True)
    record.add_argument("--launcher-exit-status", type=int, required=True)
    record.add_argument("--preflight-report", type=Path)
    record.add_argument("--expected-kind")
    record.add_argument("--unsafe-bypass", action="store_true")

    args = parser.parse_args()
    if args.command == "check":
        report, status = collect_device_report(allow_non_a100=args.allow_non_a100)
        _write_json(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return status
    if args.command == "verify-preflight":
        payload = _load_preflight(args.report, args.expected_kind)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "record":
        return _record_gate(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
