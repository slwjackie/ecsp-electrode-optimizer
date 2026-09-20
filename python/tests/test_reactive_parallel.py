from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from ecsp_reactive.parallel import (
    MPIRowDomain,
    add_outflow_row_halos,
    add_periodic_row_halos,
    all_row_partitions,
    row_partition,
)
from ecsp_reactive.provenance import ReactiveConfigurationError


@pytest.mark.parametrize("rows,size", [(10, 1), (10, 2), (10, 3), (17, 4), (32, 8)])
def test_row_decomposition_is_balanced_and_exact(rows: int, size: int) -> None:
    pieces = all_row_partitions(rows, size)
    assert pieces == [row_partition(rows, size, rank) for rank in range(size)]
    counts = [stop - start for start, stop in pieces]
    assert sum(counts) == rows
    assert max(counts) - min(counts) <= 1


def test_serial_outflow_halo_fill() -> None:
    field = np.arange(4 * 3 * 2, dtype=np.float64).reshape(4, 3, 2)
    haloed = add_outflow_row_halos(field, 3)
    np.testing.assert_array_equal(haloed[3:-3], field)
    np.testing.assert_array_equal(haloed[:3], np.broadcast_to(field[0], (3, 3, 2)))
    np.testing.assert_array_equal(haloed[-3:], np.broadcast_to(field[-1], (3, 3, 2)))


def test_serial_periodic_halo_fill_and_rank_one_domain_wrap() -> None:
    field = np.arange(6 * 2, dtype=np.float64).reshape(6, 2)
    expected = np.concatenate((field[-2:], field, field[:2]), axis=0)
    np.testing.assert_array_equal(add_periodic_row_halos(field, 2), expected)

    class RankOneCommunicator:
        @staticmethod
        def Get_rank() -> int:
            return 0

        @staticmethod
        def Get_size() -> int:
            return 1

    domain = MPIRowDomain(
        RankOneCommunicator(), global_rows=field.shape[0], halo=2, periodic=True
    )
    np.testing.assert_array_equal(domain.exchange_row_halos(field), expected)


def test_rank_count_cannot_exceed_row_count() -> None:
    with pytest.raises(ReactiveConfigurationError):
        all_row_partitions(2, 3)


@pytest.mark.skipif(importlib.util.find_spec("mpi4py") is not None, reason="mpi4py is installed")
def test_actual_mpi_fails_with_explicit_environment_status_when_unavailable() -> None:
    with pytest.raises(ReactiveConfigurationError, match="NOT_TESTED_NO_MPI_RUNTIME"):
        MPIRowDomain.world(global_rows=32)
