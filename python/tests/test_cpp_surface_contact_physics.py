from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np

from ecsp_cpp.backend import (
    CppPhysicsRunner,
    serialise_cpp_config,
)
from ecsp_v6.config import load_config
from ecsp_v6.physics.composition_model import build_composition


def _mask(n: int, width: int):
    anode = np.zeros((n, n), dtype=bool)
    cathode = np.zeros((n, n), dtype=bool)
    anode[5:-5, 5 : 5 + width] = True
    cathode[5:-5, -(5 + width) : -5] = True
    return anode, cathode


def test_cpp_surface_contacts_do_not_remove_propellant(tmp_path: Path):
    root = Path(__file__).resolve().parents[2]
    executable = tmp_path / "ecsp_cpp_solver"
    subprocess.run(
        [
            "c++", "-std=c++17", "-O2", "-DNDEBUG",
            str(root / "cpp" / "ecsp_cpp_solver.cpp"),
            "-o", str(executable),
        ],
        check=True,
    )
    config = load_config(root / "config" / "default_lp_pva.yaml")
    composition = build_composition(config)
    runner = CppPhysicsRunner(root, executable)
    assert runner.verify() == "ecsp_cpp_solver 7.9.4"

    outputs = []
    for index, width in enumerate((2, 4)):
        anode, cathode = _mask(33, width)
        values = serialise_cpp_config(config, composition, grid_size=33, voltage=260.0)
        values["end_time"] = 0.00025
        values["eval_time"] = 0.00025
        # Force first-step condensed-phase onset so ignition diagnostics are
        # directly testable without a long thermal integration.
        values["ignition_T"] = 250.0
        row = runner.run_case(anode, cathode, values, tmp_path / f"case_{index}")
        outputs.append(row)
        assert row["surfaceContactModel"] is True
        assert row["electrodeMasksRemovePropellant"] is False
        assert row["hiddenBusConnectionAssumed"] is True
        assert row["propellantDomainAreaFraction"] == 1.0
        assert row["propellantCellCount"] == 33 * 33
        assert row["solverRevision"] == "v7_9_4_cpp_fp64_surface_contact_vmin_search_ready"
        assert row["finalElectricalConverged"] is True
        assert row["ignitionSucceeded"] is True
        assert row["appliedVoltage_V"] == 260.0
        assert row["inputElectricalEnergyToIgnition_J"] > 0.0
        assert np.isclose(
            row["inputElectricalEnergyToIgnition_J"],
            row["inputElectricalEnergyAt2s_J"],
        )

    assert outputs[0]["propellantCellCount"] == outputs[1]["propellantCellCount"]
    assert outputs[1]["totalContactAreaFraction"] > outputs[0]["totalContactAreaFraction"]
