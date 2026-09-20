from __future__ import annotations

from pathlib import Path

from ecsp_v6.config import load_config


ROOT = Path(__file__).resolve().parents[2]


def test_active_solver_contains_no_f_state_or_equation() -> None:
    coupled = (ROOT / "python/ecsp_v6/physics/coupled.py").read_text(encoding="utf-8")
    electrochem = (ROOT / "python/ecsp_v6/physics/electrochem.py").read_text(encoding="utf-8")
    active = coupled + "\n" + electrochem
    forbidden = [
        'state["flameProgress"]',
        "state['flameProgress']",
        'state["F"]',
        "state['F']",
        "k_spread",
        "k_seed",
        "qFlame",
        "burnedAreaFraction",
        "establishedIgnitionDelay",
    ]
    assert not [token for token in forbidden if token in active]


def test_base_config_has_no_flame_section_and_cfd_is_disabled() -> None:
    config = load_config(ROOT / "config/default_lp_pva.yaml")
    assert "flame" not in config
    assert config["reducedCFD"]["enabled"] is False
    assert config["multiFidelity"]["enabled"] is False
    assert config["condensedPhaseMetrics"]["evaluationTime_s"] == 2.0


def test_active_package_has_no_openfoam_runtime_commands() -> None:
    paths = [
        ROOT / "python/ecsp_nsga2",
        ROOT / "python/ecsp_v6/physics",
        ROOT / "python/run_nsga2_electrical_solid_loop.py",
        ROOT / "tools",
        ROOT / "RUN_NSGA2_ONLY.sh",
    ]
    text_parts: list[str] = []
    for path in paths:
        if path.is_file():
            text_parts.append(path.read_text(encoding="utf-8"))
        else:
            for file in path.rglob("*"):
                if file.is_file() and file.suffix in {".py", ".sh", ".yaml", ".json"}:
                    text_parts.append(file.read_text(encoding="utf-8"))
    text = "\n".join(text_parts)
    forbidden_commands = ["foamRun -", "blockMesh -", "reactingFoam -", "wmake "]
    assert not [token for token in forbidden_commands if token in text]


def test_production_requires_condensed_phase_ignition_for_recommendation() -> None:
    import yaml
    config = yaml.safe_load((ROOT / "config/nsga2_condensed_phase_no_f.yaml").read_text())
    assert config["optimization"]["require_ignition_for_feasibility"] is True
    assert config["optimization"]["no_ignition_constraint_violation"] > 0.0
