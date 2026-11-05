"""Real LibTorch CPU integration on a rectangle with no padded cells."""
from pathlib import Path

import pytest
import torch
import yaml

from ecsp_cpp_cuda.adapter import pack_config
from ecsp_cpp_cuda.standalone import run_standalone
from ecsp_preflame.evaluator import create_evaluator


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def native_inputs(tmp_path_factory):
    config = yaml.safe_load((ROOT / "config/preflame_cpp_cuda_debug.yaml").read_text())
    config["evaluator"]["cpp_cuda"]["cpu_runtime"] = "standalone"
    adapter = dict(config["evaluator"])
    adapter["physics_config"] = config
    evaluator = create_evaluator(ROOT, adapter, tmp_path_factory.mktemp("native_engine"), False)
    settings, table = pack_config(evaluator.config, evaluator.composition, evaluator.grid_size)
    settings.update(end=settings["dt"], evaluation=settings["dt"],
                    electrical_interval=settings["dt"], save_handoff=1, dx=.001,
                    onset_T=1000)
    # Resolve nonlinear and local contact solves tightly enough that their
    # stopping error does not dominate an axis-transposition comparison.
    settings.update(linear_rtol=1e-12, linear_atol=1e-14, local_abs=1e-10,
                    local_rel=1e-10, local_max=100, outer_phi_tol=1e-8,
                    outer_current_tol=1e-9, balance_rel=1e-8, outer_max=300)
    return settings, torch.tensor(table, dtype=torch.float64)


def test_rectangular_cpu_engine_transpose_preserves_fields_and_integrals(native_inputs):
    settings, table = native_inputs
    # 25 rows x 12 columns, each cell 1 mm square: exactly 25 mm high x 12 mm wide.
    anode = torch.zeros((2, 25, 12), dtype=torch.bool)
    cathode = torch.zeros_like(anode)
    anode[0, :, :2] = True
    cathode[0, :, -2:] = True
    # A staggered second candidate exercises fluxes along both axes.
    anode[1, 2:20, :4] = True
    cathode[1, 5:24, -4:] = True
    voltage = torch.tensor([20., 260.], dtype=torch.float64)
    direct = run_standalone(anode, cathode, voltage, table, settings, timeout_s=90)
    rotated = run_standalone(anode.transpose(1, 2).contiguous(),
                             cathode.transpose(1, 2).contiguous(),
                             voltage, table, settings, timeout_s=90)
    assert direct["state"].shape == (10, 2, 25, 12)
    assert rotated["state"].shape == (10, 2, 12, 25)
    assert torch.isfinite(direct["state"]).all()
    assert direct["linear_converged"].all()
    assert rotated["linear_converged"].all()
    assert direct["steps_completed"].tolist() == [1, 1]
    for key in ("state", "snapshot_qj", "snapshot_qe", "snapshot_T", "snapshot_X"):
        torch.testing.assert_close(direct[key], rotated[key].transpose(-1, -2),
                                   rtol=2e-6, atol=1e-5, equal_nan=True)
    # Candidate histories are physical integrals independent of axis naming.
    torch.testing.assert_close(direct["histories"], rotated["histories"],
                               rtol=2e-6, atol=1e-5, equal_nan=True)
