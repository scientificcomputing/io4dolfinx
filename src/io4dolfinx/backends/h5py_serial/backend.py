"""The h5py backend for h5py built without MPI.

Files are read by every process on its own and written through rank 0, see
:mod:`io4dolfinx.backends.serial`. They are the files of the h5py backend.

SPDX License identifier: MIT

Copyright: Jørgen S. Dokken, Simula Research Laboratory
"""

from pathlib import Path
from typing import Any

from mpi4py import MPI

import dolfinx
import numpy as np

from ...structures import FunctionData, MeshData, MeshTagsData
from .. import FileMode
from ..h5py import backend as parallel
from ..h5py import storage as h5_storage
from ..serial import (
    gather_function,
    gather_mesh,
    gather_meshtags,
    gather_rows,
    serial,
    write_on_root,
)

read_mode = parallel.read_mode
get_default_backend_args = parallel.get_default_backend_args
convert_file_mode = parallel.convert_file_mode

read_metadata = serial(parallel.read_metadata)
read_attributes = serial(parallel.read_attributes)
read_timestamps = serial(parallel.read_timestamps)
read_mesh_data = serial(parallel.read_mesh_data)
read_meshtags_data = serial(parallel.read_meshtags_data)
read_dofmap = serial(parallel.read_dofmap)
read_dofs = serial(parallel.read_dofs)
read_cell_perms = serial(parallel.read_cell_perms)
read_legacy_mesh = serial(parallel.read_legacy_mesh)
read_hdf5_array = serial(parallel.read_hdf5_array)
read_function_names = serial(parallel.read_function_names)
read_point_data = parallel.read_point_data
read_cell_data = parallel.read_cell_data
write_data = parallel.write_data


def write_attributes(
    filename: Path | str,
    comm: MPI.Comm,
    name: str,
    attributes: dict[str, np.ndarray],
    backend_args: dict[str, Any] | None = None,
):
    """Write attributes to file, see :func:`io4dolfinx.backends.h5py.backend.write_attributes`."""
    write_on_root(
        comm,
        attributes,
        lambda d: parallel.write_attributes(filename, MPI.COMM_SELF, name, d, backend_args),
    )


def write_mesh(
    filename: Path | str,
    comm: MPI.Comm,
    mesh: MeshData,
    backend_args: dict[str, Any] | None = None,
    mode: FileMode = FileMode.write,
    time: float = 0.0,
):
    """Write a mesh to file, see :func:`io4dolfinx.backends.h5py.backend.write_mesh`."""
    h5_storage.storage_options(comm, backend_args or {})
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
    backend_args: dict[str, Any] | None = None,
):
    """Write mesh tags to file, see :func:`io4dolfinx.backends.h5py.backend.write_meshtags`."""
    h5_storage.storage_options(comm, backend_args or {})
    tags = gather_meshtags(comm, data)
    write_on_root(
        comm, tags, lambda d: parallel.write_meshtags(filename, MPI.COMM_SELF, d, backend_args)
    )


def write_function(
    filename: Path,
    comm: MPI.Comm,
    u: FunctionData,
    time: float,
    mode: FileMode,
    backend_args: dict[str, Any] | None,
):
    """Write a function to file, see :func:`io4dolfinx.backends.h5py.backend.write_function`."""
    h5_storage.storage_options(comm, backend_args or {})
    data = gather_function(comm, u)
    write_on_root(
        comm,
        data,
        lambda d: parallel.write_function(filename, MPI.COMM_SELF, d, time, mode, backend_args),
    )


def snapshot_checkpoint(
    filename: Path | str,
    mode: FileMode,
    u: dolfinx.fem.Function,
    backend_args: dict[str, Any] | None,
):
    """Create or read a snapshot checkpoint, see
    :func:`io4dolfinx.backends.h5py.backend.snapshot_checkpoint`."""
    if mode != FileMode.write:
        return serial(parallel.snapshot_checkpoint)(filename, mode, u, backend_args)
    comm = u.function_space.mesh.comm
    dofmap = u.function_space.dofmap
    start, stop = np.array(dofmap.index_map.local_range) * dofmap.index_map_bs
    options = h5_storage.storage_options(comm, backend_args or {})
    values = gather_rows(comm, start, u.x.array[: stop - start])

    def write(dofs: np.ndarray):
        with parallel.h5pyfile(filename, filemode="w", comm=MPI.COMM_SELF) as h5file:
            dataset = parallel._create_dataset(
                h5file.create_group("snapshot"), "dofs", dofs.shape, dofs.dtype, options
            )
            h5_storage.write_dataset(dataset, slice(None), dofs)

    write_on_root(comm, values, write)
