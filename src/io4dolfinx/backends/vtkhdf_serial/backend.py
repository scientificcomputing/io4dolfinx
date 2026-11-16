"""The VTKHDF backend for h5py built without MPI.

Files are read by every process on its own and written through rank 0, see
:mod:`io4dolfinx.backends.serial`. They are the files of the VTKHDF backend.

SPDX License identifier: MIT

Copyright: Jørgen S. Dokken, Simula Research Laboratory
"""

from pathlib import Path
from typing import Any

from mpi4py import MPI

from ...structures import ArrayData, MeshData, MeshTagsData
from .. import FileMode
from ..serial import gather_array, gather_mesh, gather_meshtags, serial, write_on_root
from ..vtkhdf import backend as parallel

read_mode = parallel.read_mode
get_default_backend_args = parallel.get_default_backend_args

read_mesh_data = serial(parallel.read_mesh_data)
read_point_data = serial(parallel.read_point_data)
read_cell_data = serial(parallel.read_cell_data)
read_timestamps = serial(parallel.read_timestamps)
read_function_names = serial(parallel.read_function_names)
read_meshtags_data = serial(parallel.read_meshtags_data)
write_attributes = parallel.write_attributes
read_attributes = parallel.read_attributes
read_dofmap = parallel.read_dofmap
read_dofs = parallel.read_dofs
read_cell_perms = parallel.read_cell_perms
write_function = parallel.write_function
read_legacy_mesh = parallel.read_legacy_mesh
snapshot_checkpoint = parallel.snapshot_checkpoint
read_hdf5_array = parallel.read_hdf5_array


def write_mesh(
    filename: Path | str,
    comm: MPI.Comm,
    mesh: MeshData,
    backend_args: dict[str, Any] | None,
    mode: FileMode,
    time: float,
):
    """Write a mesh to file, see :func:`io4dolfinx.backends.vtkhdf.backend.write_mesh`."""
    data = gather_mesh(comm, mesh)
    write_on_root(
        comm,
        data,
        lambda d: parallel.write_mesh(filename, MPI.COMM_SELF, d, backend_args, mode, time),
    )


def write_meshtags(
    filename: str | Path,
    comm: MPI.Comm,
    data: MeshTagsData,
    backend_args: dict[str, Any] | None,
):
    """Write mesh tags to file, see :func:`io4dolfinx.backends.vtkhdf.backend.write_meshtags`."""
    tags = gather_meshtags(comm, data)
    write_on_root(
        comm, tags, lambda d: parallel.write_meshtags(filename, MPI.COMM_SELF, d, backend_args)
    )


def write_data(
    filename: Path | str,
    array_data: ArrayData,
    comm: MPI.Comm,
    time: str | float | None,
    mode: FileMode,
    backend_args: dict[str, Any] | None,
):
    """Write point or cell data to file, see
    :func:`io4dolfinx.backends.vtkhdf.backend.write_data`."""
    data = gather_array(comm, array_data)
    write_on_root(
        comm,
        data,
        lambda d: parallel.write_data(filename, d, MPI.COMM_SELF, time, mode, backend_args),
    )
