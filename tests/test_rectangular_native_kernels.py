"""Exercise the shared CPU/CUDA scalar kernels on actual rectangular arrays.

The small C++ harness avoids a LibTorch build and tests the very same header
used by both backends against conservation laws and manufactured fields.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A C++ compiler is required for the native kernel checks")
    folder = tmp_path_factory.mktemp("rectangular_native")
    source = folder / "kernels.cpp"
    source.write_text(r'''
#include "preflame_cell_physics.h"
using namespace preflame_cpp_cuda;
static Params params(int batch, int ny, int nx) {
    Params p;
    p.batch=batch; p.ny=ny; p.nx=nx;
    p.dx=1; p.dt=.01; p.density=1;
    p.cp0=p.cm0=p.water0=p.lp0=p.pva0=10;
    p.min_T=0; p.max_T=10000; p.max_delta=1;
    p.kin[0].size=p.kin[1].size=2;
    return p;
}
extern "C" void stencil(int batch, int ny, int nx, const double* c,
                         const double* x, double* y) {
    int64_t total=int64_t(batch)*ny*nx;
    for(int64_t i=0;i<total;i++) stencil_cell(i,total,c,x,y,ny,nx);
}
extern "C" void current(int batch, int ny, int nx, const double* state,
                         const double* transport, const bool* prop, double* out) {
    auto p=params(batch,ny,nx); p.diffusion=0;
    int64_t total=int64_t(batch)*ny*nx;
    for(int64_t i=0;i<total;i++) current_cell(i,total,state,transport,prop,out,p);
}
extern "C" void advance(int batch, int ny, int nx, const double* state,
                         const double* transport, const double* heat,
                         const bool* prop, const bool* active, double* out, double* diag) {
    auto p=params(batch,ny,nx);
    int64_t total=int64_t(batch)*ny*nx;
    const double table[6]={0,1,0,0,-1000,-1000};
    for(int64_t i=0;i<total;i++)
        advance_cell(i,total,state,prop,active,transport,heat,table,out,diag,p);
}
extern "C" void contacts(int batch, int ny, int nx, int count,
                          const int64_t* cells, const bool* anode,
                          const double* state, const double* transport,
                          const double* volts, const bool* active, double* out) {
    auto p=params(batch,ny,nx); p.full_bv=0; p.contact_normal=1;
    p.local_min=8; p.local_max=100; p.local_abs=1e-10; p.local_rel=1e-10;
    for(int j=0;j<4;j++){p.ch[j].j0=.01; p.ch[j].eq=0;}
    int64_t total=int64_t(batch)*ny*nx;
    for(int i=0;i<count;i++)
        contact_cell(i,count,total,cells,anode,state,transport,volts,active,out,p);
}
''')
    binary = folder / "kernels.so"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-shared", "-fPIC", "-ffp-contract=off",
         "-I", str(ROOT / "cpp/preflame_cpp_cuda"), str(source), "-o", str(binary)],
        check=True, capture_output=True, text=True,
    )
    library = ctypes.CDLL(str(binary))
    double = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    boolean = np.ctypeslib.ndpointer(dtype=np.bool_, flags="C_CONTIGUOUS")
    integer = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
    shape = [ctypes.c_int] * 3
    library.stencil.argtypes = shape + [double, double, double]
    library.current.argtypes = shape + [double, double, boolean, double]
    library.advance.argtypes = shape + [double, double, double, boolean, boolean, double, double]
    library.contacts.argtypes = shape + [ctypes.c_int, integer, boolean, double, double, double, boolean, double]
    for name in ("stencil", "current", "advance", "contacts"):
        getattr(library, name).restype = None
    return library


def graph_laplacian(values):
    """Independent edge flux assembly with insulated outer boundaries."""
    result = np.zeros_like(values)
    for axis in (-2, -1):
        lo = [slice(None)] * values.ndim
        hi = lo.copy()
        lo[axis] = slice(None, -1)
        hi[axis] = slice(1, None)
        flux = values[tuple(lo)] - values[tuple(hi)]
        result[tuple(lo)] += flux
        result[tuple(hi)] -= flux
    return result


@pytest.mark.parametrize("ny,nx", [(5, 13), (13, 5), (7, 7)])
def test_stencil_matches_conservative_rectangular_operator(native, ny, nx):
    # Distinct batch offsets expose accidental connections between candidates.
    rng = np.random.default_rng(42)
    values = rng.normal(size=(2, ny, nx))
    values[1] += 1000
    coeff = np.ones((5, 2, ny, nx))
    coeff[0] = 4
    coeff[0, :, 0, :] -= 1
    coeff[0, :, -1, :] -= 1
    coeff[0, :, :, 0] -= 1
    coeff[0, :, :, -1] -= 1
    out = np.empty_like(values)
    native.stencil(2, ny, nx, coeff, values, out)
    np.testing.assert_allclose(out, graph_laplacian(values), atol=8e-13, rtol=1e-12)
    np.testing.assert_allclose(out.sum(axis=(-2, -1)), 0, atol=2e-11)


@pytest.mark.parametrize("ny,nx", [(5, 13), (13, 5), (7, 7)])
def test_affine_potential_current_and_face_power(native, ny, nx):
    yy, xx = np.indices((ny, nx))
    state = np.zeros((10, 2, ny, nx))
    state[7, 0] = 2 * xx + 3 * yy
    state[7, 1] = -4 * xx + yy + 1000
    transport = np.ones((8, 2, ny, nx))
    transport[5, 0] = .5
    transport[5, 1] = 2
    prop = np.ones((2, ny, nx), dtype=bool)
    out = np.empty((3, 2, ny, nx))
    native.current(2, ny, nx, state, transport, prop, out)
    for batch, sx, sy, sigma in [(0, 2, 3, .5), (1, -4, 1, 2)]:
        np.testing.assert_allclose(out[0, batch], sigma * np.hypot(sx, sy))
        # Each internal face contributes once to the integrated Joule heat.
        expected_power = sigma * (sx**2 * ny * (nx - 1) + sy**2 * nx * (ny - 1))
        assert out[1, batch].sum() == pytest.approx(expected_power)
    np.testing.assert_array_equal(out[2], 0)


def test_rectangular_thermal_and_species_transport_conserve_and_freeze(native):
    ny, nx = 13, 5
    yy, xx = np.indices((ny, nx))
    state = np.zeros((10, 2, ny, nx))
    state[0] = 300 + xx * .3 + yy**2 * .02
    state[0, 1] += 100
    state[1] = state[2] = 7 + xx**2 * .02 + yy * .03
    state[3] = 5 + yy**2 * .01
    state[4:7] = 1  # Finished chemistry isolates conservative transport.
    transport = np.ones((8, 2, ny, nx))
    transport[:2] = .05
    transport[2] = .07
    heat = np.zeros((7, 2, ny, nx))
    prop = np.ones((2, ny, nx), dtype=bool)
    active = np.array([True, False])
    out = np.empty_like(state)
    diag = np.empty((9, 2, ny, nx))
    native.advance(2, ny, nx, state, transport, heat, prop, active, out, diag)
    for channel, diffusivity in [(0, 1), (1, .05), (2, .05), (3, .07)]:
        expected = state[channel, 0] - .01 * diffusivity * graph_laplacian(state[channel, 0])
        np.testing.assert_allclose(out[channel, 0], expected, atol=1e-13, rtol=1e-14)
        assert out[channel, 0].sum() == pytest.approx(state[channel, 0].sum())
    np.testing.assert_array_equal(out[:, 1], state[:, 1])
    np.testing.assert_array_equal(diag[:, 1], 0)
    np.testing.assert_array_equal(diag[2:7, 0], 0)


def test_rectangular_contacts_use_correct_batch_voltage_and_active_mask(native):
    ny, nx = 13, 5
    cells = np.array([0, ny * nx, 2 * ny * nx - 1], dtype=np.int64)
    anode = np.ones(3, dtype=bool)
    state = np.zeros((10, 2, ny, nx))
    state[0] = 300
    state[1:4] = 10
    transport = np.ones((8, 2, ny, nx))
    volts = np.array([.2, .4])
    active = np.array([False, True])
    out = np.empty((9, 3))
    native.contacts(2, ny, nx, 3, cells, anode, state, transport, volts, active, out)
    np.testing.assert_array_equal(out[:, 0], 0)
    assert out[0, 1] > 0
    # Compare the active lane with a truly independent candidate.
    separate = np.empty((9, 2))
    native.contacts(1, ny, nx, 2, np.array([0, ny * nx - 1], dtype=np.int64),
                    anode[1:], state[:, 1:2].copy(), transport[:, 1:2].copy(),
                    volts[1:], np.array([True]), separate)
    np.testing.assert_array_equal(out[:, 1:], separate)
