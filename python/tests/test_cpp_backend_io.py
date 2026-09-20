from pathlib import Path
import struct

import numpy as np
import pytest

from ecsp_cpp.backend import write_cpp_config, write_mask_binary


def test_cpp_mask_binary_roundtrip_layout(tmp_path: Path):
    anode = np.zeros((4, 4), dtype=bool)
    cathode = np.zeros((4, 4), dtype=bool)
    anode[1, 1] = True
    cathode[2, 2] = True
    path = tmp_path / "mask.bin"
    write_mask_binary(anode, cathode, path)
    raw = path.read_bytes()
    assert struct.unpack("<I", raw[:4])[0] == 4
    n = 16
    a = np.frombuffer(raw[4 : 4 + n], dtype=np.uint8).reshape(4, 4)
    c = np.frombuffer(raw[4 + n : 4 + 2 * n], dtype=np.uint8).reshape(4, 4)
    np.testing.assert_array_equal(a, anode.astype(np.uint8))
    np.testing.assert_array_equal(c, cathode.astype(np.uint8))


def test_cpp_mask_binary_rejects_overlap(tmp_path: Path):
    a = np.zeros((3, 3), dtype=bool)
    c = np.zeros((3, 3), dtype=bool)
    a[1, 1] = True
    c[1, 1] = True
    with pytest.raises(ValueError, match="polarity overlap"):
        write_mask_binary(a, c, tmp_path / "bad.bin")


def test_cpp_flat_config_is_deterministic_and_boolean_safe(tmp_path: Path):
    path = tmp_path / "config.kv"
    write_cpp_config({"z": 3, "flag": True, "off": False, "x": 1.25}, path)
    assert path.read_text().splitlines() == ["z=3", "flag=true", "off=false", "x=1.25"]


def test_cpp_resolved_config_preserves_static_and_coupled_tolerances():
    from ecsp_cpp.backend import serialise_cpp_config
    from ecsp_v6.config import load_config
    from ecsp_v6.physics.composition_model import build_composition

    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "config" / "default_lp_pva.yaml")
    composition = build_composition(config)
    values = serialise_cpp_config(
        config, composition, grid_size=193, voltage=260.0
    )
    solver = config["numerics"]["potentialSolver"]
    assert values["pcg_rtol_static"] == pytest.approx(
        solver["relativeToleranceStatic"]
    )
    assert values["pcg_rtol_coupled"] == pytest.approx(
        solver["relativeToleranceCoupled"]
    )
    assert values["pcg_atol"] == pytest.approx(solver["absoluteTolerance"])
    assert values["pcg_rtol_static"] < values["pcg_rtol_coupled"]
