"""Row-wise MPI decomposition for the reactive CPU reference solver.

The paper states that MPI subdomains exchange virtual cells, but does not give
the halo width, message order, topology, or rank count.  This implementation
uses a three-cell row halo (needed by WENO5) and marks that choice as an
implementation assumption in configuration metadata.  It is optional: import
of this module never makes mpi4py a mandatory dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .provenance import ReactiveConfigurationError


def row_partition(global_rows: int, size: int, rank: int) -> tuple[int, int]:
    """Return a deterministic balanced half-open row interval."""
    if global_rows <= 0 or size <= 0 or rank < 0 or rank >= size:
        raise ReactiveConfigurationError("Invalid global_rows/size/rank partition request")
    quotient, remainder = divmod(global_rows, size)
    start = rank * quotient + min(rank, remainder)
    stop = start + quotient + int(rank < remainder)
    if stop <= start:
        raise ReactiveConfigurationError(
            "Every MPI rank must own at least one row; reduce rank count"
        )
    return start, stop


def all_row_partitions(global_rows: int, size: int) -> list[tuple[int, int]]:
    partitions = [row_partition(global_rows, size, rank) for rank in range(size)]
    if partitions[0][0] != 0 or partitions[-1][1] != global_rows:
        raise AssertionError("Row decomposition does not cover the domain")
    if any(left[1] != right[0] for left, right in zip(partitions, partitions[1:])):
        raise AssertionError("Row decomposition is not contiguous")
    return partitions


def add_outflow_row_halos(interior: np.ndarray, halo: int) -> np.ndarray:
    """Reference serial halo fill used by MPI parity tests."""
    value = np.asarray(interior)
    if value.ndim < 2 or halo < 1 or value.shape[0] < 1:
        raise ReactiveConfigurationError("Invalid field or halo width")
    shape = (value.shape[0] + 2 * halo, *value.shape[1:])
    result = np.empty(shape, dtype=value.dtype)
    result[halo:-halo] = value
    result[:halo] = value[0]
    result[-halo:] = value[-1]
    return result


def add_periodic_row_halos(interior: np.ndarray, halo: int) -> np.ndarray:
    """Fill serial row halos by wrapping the opposite side of the domain."""

    value = np.asarray(interior)
    if value.ndim < 2 or halo < 1 or value.shape[0] < halo:
        raise ReactiveConfigurationError("Invalid field or periodic halo width")
    return np.concatenate((value[-halo:], value, value[:halo]), axis=0)


@dataclass
class MPIRowDomain:
    """Actual mpi4py row-halo exchange, instantiated only when mpi4py exists."""

    communicator: Any
    global_rows: int
    halo: int = 3
    periodic: bool = False

    def __post_init__(self) -> None:
        if self.halo < 1:
            raise ReactiveConfigurationError("MPI halo width must be positive")
        self.rank = int(self.communicator.Get_rank())
        self.size = int(self.communicator.Get_size())
        self.start, self.stop = row_partition(self.global_rows, self.size, self.rank)
        if self.stop - self.start < self.halo:
            raise ReactiveConfigurationError(
                "Each MPI subdomain must own at least halo rows for WENO exchange"
            )

    @classmethod
    def world(cls, global_rows: int, halo: int = 3, periodic: bool = False) -> "MPIRowDomain":
        try:
            from mpi4py import MPI
        except ImportError as exc:
            raise ReactiveConfigurationError(
                "mpi4py is not installed; actual MPI execution is NOT_TESTED_NO_MPI_RUNTIME"
            ) from exc
        return cls(MPI.COMM_WORLD, global_rows=global_rows, halo=halo, periodic=periodic)

    @property
    def local_rows(self) -> int:
        return self.stop - self.start

    def scatter_rows(self, global_field: np.ndarray | None, root: int = 0) -> np.ndarray:
        """Scatter unequal row blocks without assuming C++ MPI datatypes."""
        if self.rank == root:
            if global_field is None or np.asarray(global_field).shape[0] != self.global_rows:
                raise ReactiveConfigurationError("Root must provide the complete global field")
            blocks = [
                np.ascontiguousarray(np.asarray(global_field)[begin:end])
                for begin, end in all_row_partitions(self.global_rows, self.size)
            ]
        else:
            blocks = None
        local = self.communicator.scatter(blocks, root=root)
        return np.ascontiguousarray(local)

    def exchange_row_halos(self, interior: np.ndarray) -> np.ndarray:
        """Exchange WENO virtual rows and apply outflow at physical boundaries."""
        value = np.ascontiguousarray(interior)
        if value.shape[0] != self.local_rows or value.shape[0] < self.halo:
            raise ReactiveConfigurationError("Local MPI field has the wrong row count")
        if self.size == 1:
            if self.periodic:
                return add_periodic_row_halos(value, self.halo)
            return add_outflow_row_halos(value, self.halo)
        result = add_outflow_row_halos(value, self.halo)
        try:
            from mpi4py import MPI
        except ImportError as exc:
            raise ReactiveConfigurationError("mpi4py disappeared after domain creation") from exc
        lower = (self.rank - 1) % self.size if self.periodic else self.rank - 1
        upper = (self.rank + 1) % self.size if self.periodic else self.rank + 1
        if not self.periodic:
            lower = MPI.PROC_NULL if lower < 0 else lower
            upper = MPI.PROC_NULL if upper >= self.size else upper
        receive_lower = np.empty_like(value[: self.halo])
        receive_upper = np.empty_like(value[: self.halo])
        self.communicator.Sendrecv(
            sendbuf=np.ascontiguousarray(value[-self.halo :]),
            dest=upper,
            sendtag=701,
            recvbuf=receive_lower,
            source=lower,
            recvtag=701,
        )
        self.communicator.Sendrecv(
            sendbuf=np.ascontiguousarray(value[: self.halo]),
            dest=lower,
            sendtag=702,
            recvbuf=receive_upper,
            source=upper,
            recvtag=702,
        )
        if lower != MPI.PROC_NULL:
            result[: self.halo] = receive_lower
        if upper != MPI.PROC_NULL:
            result[-self.halo :] = receive_upper
        return result

    def gather_rows(self, local_field: np.ndarray, root: int = 0) -> np.ndarray | None:
        value = np.ascontiguousarray(local_field)
        if value.shape[0] != self.local_rows:
            raise ReactiveConfigurationError("Local MPI field has the wrong row count")
        pieces = self.communicator.gather(value, root=root)
        if self.rank != root:
            return None
        result = np.concatenate(pieces, axis=0)
        if result.shape[0] != self.global_rows:
            raise AssertionError("MPI gather did not reconstruct all global rows")
        return result
