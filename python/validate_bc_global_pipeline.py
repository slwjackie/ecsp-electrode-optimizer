#!/usr/bin/env python3
from __future__ import annotations

"""Static and executable release audit for the v8 B/C ECSP pipeline."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import yaml


class Audit:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def check(self, key: str, passed: bool, requirement: str, evidence: Any) -> None:
        self.rows.append(
            {
                "id": key,
                "passed": bool(passed),
                "requirement": requirement,
                "evidence": evidence,
            }
        )

    @property
    def passed(self) -> bool:
        return bool(self.rows) and all(row["passed"] for row in self.rows)


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}")
    return data


def _write_reports(audit: Audit, output_dir: Path, runtime: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "passed" if audit.passed else "failed",
        "checks_passed": sum(int(x["passed"]) for x in audit.rows),
        "checks_total": len(audit.rows),
        "checks": audit.rows,
        "runtime": runtime,
    }
    (output_dir / "bc_global_pipeline_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# B/C global pre-flame + condensed propagation 구현 감사",
        "",
        f"- 판정: **{'통과' if audit.passed else '실패'}**",
        f"- 통과: **{payload['checks_passed']}/{payload['checks_total']}**",
        "",
        "| ID | 판정 | 요구사항 | 근거 |",
        "|---|---:|---|---|",
    ]
    for row in audit.rows:
        evidence = json.dumps(row["evidence"], ensure_ascii=False, default=str)
        evidence = evidence.replace("|", "\\|")
        lines.append(
            f"| `{row['id']}` | {'PASS' if row['passed'] else 'FAIL'} | "
            f"{row['requirement']} | `{evidence}` |"
        )
    lines.extend(["", "## 실행 정보", "", "```json", json.dumps(runtime, indent=2), "```", ""])
    (output_dir / "bc_global_pipeline_audit.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _sanitise_runtime_text(text: str, root: Path, run_dir: Path) -> str:
    """Remove host-specific absolute paths from persisted release evidence."""
    return (
        text.replace(str(run_dir), "<RUNTIME_WORKDIR>")
        .replace(str(root), "<PACKAGE_ROOT>")
        .replace(sys.executable, "<PYTHON>")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--runtime-workdir", type=Path)
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()

    root = args.package_root.resolve()
    sys.path.insert(0, str(root / "python"))
    from ecsp_v6.physics.bc_global import (
        BC_OPTIONAL_PREFLAME_DIAGNOSTIC_HISTORY_FIELDS,
        BC_REQUIRED_AUTHORIZED_PROPAGATION_HANDOFF_FIELDS,
        BC_SOLVER_ONSET_SNAPSHOT_FIELDS,
        bc_model_contract,
    )

    output_dir = (args.output_dir or (root / "docs" / "generated_bc_audit")).resolve()
    a100 = _load_yaml(root / "config/nsga2_bc_global_preflame_propagation_a100.yaml")
    debug = _load_yaml(root / "config/nsga2_bc_global_preflame_propagation_debug.yaml")
    version = json.loads((root / "VERSION.json").read_text(encoding="utf-8"))
    contract = bc_model_contract()
    audit = Audit()

    expected_used = [2, 4, 5, 6, 7, 8, 9, 11, 13, 15, 16, 17, 22, 23, 24]
    expected_excluded = [3, 18, 25, 26, 27, 28, 29, 30, 31]
    audit.check("eq.used", contract["paper_equations_used"] == expected_used,
                "B/C pre-flame 식 (22)–(24)와 필요한 보조식만 사용", contract)
    audit.check("eq.excluded", contract["paper_equations_not_used_in_preflame"] == expected_excluded,
                "Poisson/밀도합/점화후 유동/중복 한계전류 식 제외", contract)
    audit.check("eq32.diagnostic", contract["equation_32_role"].startswith("postprocessing"),
                "식 (32)는 중복 열원이 아니라 검산용", contract["equation_32_role"])
    audit.check("gas.none", contract["gas_phase_cfd_used"] is False,
                "후단 모델은 외부 기상 CFD가 아닌 응축상 propagation", contract["gas_phase_cfd_used"])

    comp = a100["physics"]["composition"]
    expected_comp = {
        "lithium_perchlorate_mass_fraction": 0.3158,
        "water_mass_fraction": 0.5842,
        "pva_mass_fraction": 0.09,
        "glycerol_mass_fraction": 0.0,
        "boric_acid_mass_fraction": 0.01,
        "tungsten_mass_fraction": 0.0,
    }
    comp_ok = all(np.isclose(float(comp[k]), v) for k, v in expected_comp.items())
    audit.check("composition.wet_recipe", comp_ok,
                "3번 논문 non-metallized wet recipe와 동일", comp)
    audit.check("composition.cured_water", np.isclose(a100["physics"]["cured_water_mass_fraction"], 0.20),
                "경화 후 수분 0.20은 literature-nominal로 명시", {
                    "value": a100["physics"]["cured_water_mass_fraction"],
                    "basis": a100["physics"]["cured_water_mass_fraction_basis"],
                })

    bc = a100["bc_global"]
    audit.check("chem.global", contract["global_reaction"].startswith("1.45 LiClO4"),
                "3번 논문 global LP/PVA 반응 사용", contract["global_reaction"])
    audit.check("chem.two_channels", len(bc["kinetics"]["channels"]) == 2,
                "두 conversion-dependent exothermic channel", bc["kinetics"]["model"])
    weights = bc["kinetics"]["mass_conversion_weights"]
    audit.check("chem.weights", len(weights) == 2
                and all(np.isfinite(value) and value >= 0.0 for value in weights)
                and abs(sum(weights) - 1.0) <= 1.0e-12,
                "global species inventory는 두 채널 가중합으로 한 번만 소비", weights)

    objectives = a100["optimization"]["objectives_minimise"]
    expected_objectives = [
        "ignition_delay_s",
        "area_undecomposed_fraction_at_evaluation_time",
        "minimum_ignition_voltage_V",
        "current_congestion",
    ]
    audit.check("objectives.four", objectives == expected_objectives,
                "Urem, t_onset, Vmin, J99/Jbar 네 목적함수", objectives)
    onset = bc["onsetCriterion"]
    audit.check("onset.and_area", all(np.isclose(onset[k], v) for k, v in {
        "temperature_K": 523.15,
        "minimum_progress": 0.01,
        "minimum_area_fraction": 0.01,
    }.items()), "T AND Xg over minimum area onset criterion", onset)

    overrides = a100["evaluator"]["base_overrides"]
    no_proxy = (
        overrides["coupled"]["usePaperMassTransferSaturation"] is False
        and overrides["interface"]["blocking"]["passivation"]["enabled"] is False
        and overrides["interface"]["blocking"]["gasCoverage"]["enabled"] is False
        and overrides["interface"]["liquidKineticsGain"] == 0.0
        and overrides["interface"]["includeActivationHeat"] is False
        and bc["augmentedElectronicConduction"]["enabled"] is False
    )
    audit.check("proxy.disabled", no_proxy,
                "질량전달 직렬 제한·passivation·gas coverage·liquid multiplier 비활성", {
                    "usePaperMassTransferSaturation": overrides["coupled"]["usePaperMassTransferSaturation"],
                    "passivation": overrides["interface"]["blocking"]["passivation"]["enabled"],
                    "gasCoverage": overrides["interface"]["blocking"]["gasCoverage"]["enabled"],
                    "liquidKineticsGain": overrides["interface"]["liquidKineticsGain"],
                    "activationHeat": overrides["interface"]["includeActivationHeat"],
                    "electronicAugmentation": bc["augmentedElectronicConduction"]["enabled"],
                })

    prop = a100["propagation_refinement"]
    audit.check("pipeline.1000", a100["optimization"]["population_size"] == 1000,
                "1,000개 형상 B/C 전수평가 기본 profile", a100["optimization"]["population_size"])
    audit.check("pipeline.diversity", 0.10 <= prop["selection"]["additional_fraction"] <= 0.20,
                "Pareto + near-Pareto/diverse 10–20%", prop["selection"])
    audit.check("pipeline.propagation", prop["enabled"] is True and "not_gas_phase_cfd" in prop["interpretation"],
                "post-onset condensed reaction-progress/level-set refinement", prop["interpretation"])
    # The implementation constants are the single source of truth.  The
    # release metadata is checked separately below so a copied hard-coded list
    # cannot drift while this audit still reports success.
    expected_solver_snapshot_fields = list(BC_SOLVER_ONSET_SNAPSHOT_FIELDS)
    audit.check(
        "pipeline.solver_snapshot",
        contract["solver_onset_snapshot_fields"] == expected_solver_snapshot_fields,
        "저수준 solver의 최초 onset 상태·species·potential·열원 snapshot",
        contract["solver_onset_snapshot_fields"],
    )
    expected_propagation_handoff_fields = list(
        BC_REQUIRED_AUTHORIZED_PROPAGATION_HANDOFF_FIELDS
    )
    audit.check(
        "pipeline.propagation_handoff",
        contract["required_authorized_propagation_handoff_fields"]
        == expected_propagation_handoff_fields
        and contract["post_onset_handoff_fields"]
        == expected_propagation_handoff_fields,
        "authorized propagation의 완전한 solver snapshot·inventory·mask·schema·수치 authorization handoff",
        contract["required_authorized_propagation_handoff_fields"],
    )
    audit.check(
        "pipeline.version_handoff",
        version["post_onset_condensed_propagation"][
            "required_propagation_handoff_fields"
        ]
        == expected_propagation_handoff_fields,
        "VERSION handoff field order exactly matches the implementation constant",
        version["post_onset_condensed_propagation"][
            "required_propagation_handoff_fields"
        ],
    )
    audit.check(
        "pipeline.optional_history",
        contract["optional_preflame_diagnostic_history_fields"]
        == list(BC_OPTIONAL_PREFLAME_DIAGNOSTIC_HISTORY_FIELDS),
        "pre-flame time histories are optional diagnostics, not propagation source fields",
        contract["optional_preflame_diagnostic_history_fields"],
    )
    audit.check("pipeline.staggered", a100["baselines"]["area_matched_staggered"]["enabled"] is True,
                "area-matched staggered는 동일 B/C+propagation 경로로 후평가", a100["baselines"]["area_matched_staggered"])
    approximate_required_finger_width = (
        a100["geometry"]["target_area_fraction_per_polarity"]
        * a100["geometry"]["domain_mm"] / 2.0
    )
    audit.check(
        "pipeline.staggered_constructible",
        a100["geometry"]["maximum_width_mm"] >= approximate_required_finger_width,
        "production width bound permits the required two-finger area-matched staggered reference",
        {
            "maximum_width_mm": a100["geometry"]["maximum_width_mm"],
            "approximate_required_width_mm": approximate_required_finger_width,
        },
    )
    audit.check("legacy.configs", all((root / rel).is_file() for rel in [
        "config/nsga2_condensed_phase_no_f_a100_cpu48_hybrid.yaml",
        "config/nsga2_condensed_phase_no_f_m2_cpp_fp64.yaml",
        "cpp/ecsp_cpp_solver.cpp",
        "cpp/ecsp_cpp_solver_hybrid.cpp",
    ]), "v7.9.5 backend/config/source 보존", "legacy files present")

    runtime: dict[str, Any] = {"executed": False, "static_only": bool(args.static_only)}
    if not args.static_only:
        cleanup = args.runtime_workdir is None
        run_dir = (
            args.runtime_workdir
            or Path(tempfile.mkdtemp(prefix="ecsp_bc_audit_"))
        ).resolve()
        if not cleanup and run_dir.exists():
            parser.error(
                "--runtime-workdir must not already exist; no caller-supplied "
                f"directory is deleted: {run_dir}"
            )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(root / "python")
        command = [
            sys.executable,
            str(root / "python/run_nsga2_electrical_solid_loop.py"),
            "--config", str(root / "config/nsga2_bc_global_preflame_propagation_debug.yaml"),
            "--workdir", str(run_dir),
            "--package-root", str(root),
            "--allow-no-feasible",
        ]
        proc = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True)
        reported_command = [
            _sanitise_runtime_text(part, root, run_dir) for part in command
        ]
        runtime = {
            "executed": True,
            "command": reported_command,
            "returncode": proc.returncode,
            "workdir": "<RUNTIME_WORKDIR>",
            "stdout_tail": _sanitise_runtime_text(
                proc.stdout[-4000:], root, run_dir
            ),
            "stderr_tail": _sanitise_runtime_text(
                proc.stderr[-4000:], root, run_dir
            ),
        }
        audit.check("runtime.exit", proc.returncode == 0,
                    "CPU FP64 debug pipeline completes", proc.returncode)
        required = [
            "final/preflame_final_pareto_designs.csv",
            "final/propagation_selection.csv",
            "final/all_propagation_refined_designs.csv",
            "final/final_pareto_designs.csv",
            "final/recommended_design.json",
            "final/area_matched_staggered.json",
            "final/area_matched_staggered_propagation.json",
            "final/recommended_vs_area_matched_staggered_propagation.json",
            "RUN_COMPLETE.json",
        ]
        missing = [name for name in required if not (run_dir / name).is_file()]
        audit.check("runtime.outputs", not missing,
                    "pre-flame, propagation, final Pareto, staggered outputs persisted", missing)
        if not missing:
            recommended = json.loads((run_dir / "final/recommended_design.json").read_text())
            complete = json.loads((run_dir / "RUN_COMPLETE.json").read_text())
            audit.check("runtime.objectives", len(recommended["all_minimisation_objectives"]) == 8,
                        "final Pareto combines 4 pre-flame + 4 propagation outputs", recommended["all_minimisation_objectives"])
            audit.check("runtime.scope", complete.get("bc_global_preflame_used") is True
                        and complete.get("post_onset_condensed_propagation_used") is True
                        and complete.get("gas_phase_cfd_used") is False,
                        "runtime scope labels match implementation", complete)
            candidate_dirs = list((run_dir / "final" / "propagation_candidates").glob("*/condensed_propagation/propagation_fields.npz"))
            audit.check("runtime.full_fields", bool(candidate_dirs),
                        "full T/Xg/level-set propagation fields written", [
                            p.relative_to(run_dir).as_posix()
                            for p in candidate_dirs[:3]
                        ])
            handoff_dirs = list(
                (run_dir / "final" / "propagation_candidates").glob(
                    "*/bc_handoff"
                )
            )
            persisted_handoff_keys: set[str] = set()
            if handoff_dirs:
                with np.load(
                    handoff_dirs[0] / "bc_handoff_fields.npz", allow_pickle=False
                ) as archive:
                    persisted_handoff_keys.update(archive.files)
                persisted_handoff_keys.update(
                    json.loads(
                        (handoff_dirs[0] / "bc_handoff_metadata.json").read_text(
                            encoding="utf-8"
                        )
                    )
                )
            missing_handoff = sorted(
                set(expected_propagation_handoff_fields)
                - persisted_handoff_keys
            )
            audit.check(
                "runtime.handoff_contract",
                bool(handoff_dirs) and not missing_handoff,
                "persisted handoff contains every required authorized propagation field",
                {"missing": missing_handoff},
            )
        if cleanup:
            # Only the directory created by tempfile.mkdtemp in this process is
            # eligible for cleanup.  A caller-supplied path is never removed.
            import shutil

            shutil.rmtree(run_dir, ignore_errors=True)

    _write_reports(audit, output_dir, runtime)
    print(json.dumps({
        "status": "passed" if audit.passed else "failed",
        "checks": f"{sum(int(x['passed']) for x in audit.rows)}/{len(audit.rows)}",
        "output_dir": str(output_dir),
    }, indent=2))
    return 0 if audit.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
