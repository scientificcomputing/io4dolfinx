# Copyright (C) 2026 Jørgen Schartum Dokken
#
# This file is part of io4dolfinx
#
# SPDX-License-Identifier:    MIT

"""Run the HDF5 backends with h5py built without MPI.

The serial backends (``h5py-serial`` and ``vtkhdf-serial``) write the same files as the
backends they wrap, without parallel HDF5:

- To read, every process opens the file on its own and reads its part, as many processes
  may read an HDF5 file at the same time.
- To write, the data is gathered on rank 0, which writes it with the backend it wraps on
  :data:`mpi4py.MPI.COMM_SELF`. Rank 0 then holds all the data being written.

:func:`io4dolfinx.backends.get_backend` uses them in place of ``h5py`` and ``vtkhdf`` when
h5py is built without MPI.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
from collections.abc import Callable, Generator
from typing import Any, TypeVar

from mpi4py import MPI

import numpy as np
import numpy.typing as npt

from ..comm_helpers import run_on_root
from ..structures import ArrayData, FunctionData, MeshData, MeshTagsData
from .h5py.backend import force_serial_files

__all__ = [
    "serial_files",
    "serial",
    "gather_rows",
    "gather_mesh",
    "gather_function",
    "gather_meshtags",
    "gather_array",
    "write_on_root",
]

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


@contextlib.contextmanager
def serial_files() -> Generator[None, None, None]:
    """Open HDF5 files without MPI on every rank within the context."""
    token = force_serial_files.set(True)
    try:
        yield
    finally:
        force_serial_files.reset(token)


def serial(fn: F) -> F:
    """The function, opening HDF5 files without MPI on every rank."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with serial_files():
            return fn(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def gather_rows(
    comm: MPI.Comm, start: int, rows: npt.NDArray, overlap: bool = False
) -> npt.NDArray | None:
    """The rows of all processes on rank 0, in global order.

    Args:
        comm: The communicator the rows are divided over.
        start: The global position of this process's first row.
        rows: This process's rows.
        overlap: Whether the last row of each process is the first row of the next, as
            for offsets. It is then kept once.

    Returns:
        All rows on rank 0, ``None`` on the other ranks.
    """
    parts = comm.gather((start, rows), root=0)
    if parts is None:
        return None
    ordered = [part for _, part in sorted(parts, key=lambda part: part[0])]
    if overlap:
        ordered = [part[:-1] for part in ordered[:-1]] + ordered[-1:]
    return np.concatenate(ordered)


def gather_mesh(comm: MPI.Comm, mesh: MeshData) -> MeshData | None:
    """The mesh data of all processes on rank 0, ``None`` on the other ranks."""
    geometry = gather_rows(comm, mesh.local_geometry_pos[0], mesh.local_geometry)
    topology = gather_rows(comm, mesh.local_topology_pos[0], mesh.local_topology)
    ownership_array = ownership_offset = None
    partition_range = None
    if mesh.store_partition:
        assert mesh.partition_range is not None and mesh.ownership_offset is not None
        assert mesh.ownership_array is not None and mesh.partition_global is not None
        partition_range = (0, mesh.partition_global)
        ownership_array = gather_rows(comm, mesh.partition_range[0], mesh.ownership_array)
        ownership_offset = gather_rows(
            comm, mesh.local_topology_pos[0], mesh.ownership_offset, overlap=True
        )
    if comm.rank != 0:
        return None
    assert geometry is not None and topology is not None
    return dataclasses.replace(
        mesh,
        local_geometry=geometry,
        local_geometry_pos=(0, mesh.num_nodes_global),
        local_topology=topology,
        local_topology_pos=(0, mesh.num_cells_global),
        ownership_array=ownership_array,
        ownership_offset=ownership_offset,
        partition_range=partition_range,
    )


def gather_function(comm: MPI.Comm, u: FunctionData) -> FunctionData | None:
    """The function data of all processes on rank 0, ``None`` on the other ranks."""
    cell_start = u.local_cell_range[0]
    cell_permutations = gather_rows(comm, cell_start, u.cell_permutations)
    dofmap = gather_rows(comm, u.dofmap_range[0], u.dofmap_array)
    offsets = gather_rows(comm, cell_start, u.dofmap_offsets, overlap=True)
    values = gather_rows(comm, u.dof_range[0], u.values)
    if comm.rank != 0:
        return None
    assert cell_permutations is not None and dofmap is not None
    assert offsets is not None and values is not None
    return dataclasses.replace(
        u,
        cell_permutations=cell_permutations,
        local_cell_range=(0, u.num_cells_global),
        dofmap_array=dofmap,
        dofmap_offsets=offsets,
        dofmap_range=(0, u.global_dofs_in_dofmap),
        values=values,
        dof_range=(0, u.num_dofs_global),
    )


def gather_meshtags(comm: MPI.Comm, tags: MeshTagsData) -> MeshTagsData | None:
    """The mesh tags of all processes on rank 0, ``None`` on the other ranks."""
    start = 0 if tags.local_start is None else tags.local_start
    indices = gather_rows(comm, start, tags.indices)
    values = gather_rows(comm, start, tags.values)
    if comm.rank != 0:
        return None
    assert indices is not None and values is not None
    return dataclasses.replace(tags, indices=indices, values=values, local_start=0)


def gather_array(comm: MPI.Comm, data: ArrayData) -> ArrayData | None:
    """The array data of all processes on rank 0, ``None`` on the other ranks."""
    values = gather_rows(comm, data.local_range[0], data.values)
    if comm.rank != 0:
        return None
    assert values is not None
    return dataclasses.replace(data, values=values, local_range=(0, data.global_shape[0]))


def write_on_root(comm: MPI.Comm, data: T | None, write: Callable[[T], None]):
    """Write data on rank 0, with HDF5 files opened without MPI, and raise any error it
    raises on all ranks.

    Args:
        comm: The communicator of the processes writing.
        data: The data to write on rank 0, as from the gather functions. Not used on the
            other ranks.
        write: Function writing the data.
    """

    def run():
        assert data is not None
        with serial_files():
            write(data)

    run_on_root(comm, run)
