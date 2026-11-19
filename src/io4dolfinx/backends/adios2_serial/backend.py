"""The ADIOS2 backend for ADIOS2 built without MPI.

Files are read by every process on its own and written through rank 0, see
:mod:`io4dolfinx.backends.serial`. They are the files of the ADIOS2 backend.

SPDX License identifier: MIT

Copyright: Jørgen S. Dokken, Simula Research Laboratory
"""

import functools
from pathlib import Path
from typing import Any

from mpi4py import MPI

import dolfinx
import numpy as np

from ...structures import FunctionData, MeshData, MeshTagsData
from .. import FileMode
from ..adios2 import backend as parallel
from ..adios2.helpers import ADIOSFile, create_adios, force_serial_adios
from ..serial import gather_function, gather_mesh, gather_meshtags, serial, write_on_root

serial_adios = functools.partial(serial, flag=force_serial_adios)
write_on_root_adios = functools.partial(write_on_root, flag=force_serial_adios)

read_mode = parallel.read_mode
get_default_backend_args = parallel.get_default_backend_args
convert_file_mode = parallel.convert_file_mode

read_metadata = serial_adios(parallel.read_metadata)
read_attributes = serial_adios(parallel.read_attributes)
read_timestamps = serial_adios(parallel.read_timestamps)
read_mesh_data = serial_adios(parallel.read_mesh_data)
read_meshtags_data = serial_adios(parallel.read_meshtags_data)
read_dofmap = serial_adios(parallel.read_dofmap)
read_dofs = serial_adios(parallel.read_dofs)
read_cell_perms = serial_adios(parallel.read_cell_perms)
read_hdf5_array = serial_adios(parallel.read_hdf5_array)
read_legacy_mesh = serial_adios(parallel.read_legacy_mesh)
read_function_names = parallel.read_function_names
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
    """Write attributes to file, see :func:`io4dolfinx.backends.adios2.backend.write_attributes`."""
    write_on_root_adios(
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
    """Write a mesh to file, see :func:`io4dolfinx.backends.adios2.backend.write_mesh`."""
    data = gather_mesh(comm, mesh)
    write_on_root_adios(
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
    """Write mesh tags to file, see :func:`io4dolfinx.backends.adios2.backend.write_meshtags`."""
    tags = gather_meshtags(comm, data)
    write_on_root_adios(
        comm, tags, lambda d: parallel.write_meshtags(filename, MPI.COMM_SELF, d, backend_args)
    )


def write_function(
    filename: Path,
    comm: MPI.Comm,
    u: FunctionData,
    time: float = 0.0,
    mode: FileMode = FileMode.append,
    backend_args: dict[str, Any] | None = None,
):
    """Write a function to file, see :func:`io4dolfinx.backends.adios2.backend.write_function`."""
    data = gather_function(comm, u)
    write_on_root_adios(
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
    :func:`io4dolfinx.backends.adios2.backend.snapshot_checkpoint`.

    Rank 0 writes the dofs of each process as a block of its own, in rank order, so that
    each process reads its block back, as from a file of the ADIOS2 backend.
    """
    if mode != FileMode.write:
        return serial_adios(parallel.snapshot_checkpoint)(filename, mode, u, backend_args)
    comm = u.function_space.mesh.comm
    dofmap = u.function_space.dofmap
    num_dofs_local = dofmap.index_map.size_local * dofmap.index_map_bs
    blocks = comm.gather(u.x.array[:num_dofs_local].copy(), root=0)
    backend_args = {} if backend_args is None else backend_args

    def write(blocks: list[np.ndarray]):
        with ADIOSFile(
            adios=create_adios(MPI.COMM_SELF),
            filename=filename,
            mode=convert_file_mode(FileMode.write),
            io_name=backend_args.get("io_name", "SnapshotCheckPoint"),
            engine=backend_args.get("engine", "BP4"),
        ) as adios_file:
            adios_file.file.BeginStep()
            dofs = adios_file.io.DefineVariable("dofs", blocks[0], count=[len(blocks[0])])
            for block in blocks:
                dofs.SetSelection([[], [len(block)]])
                adios_file.file.Put(dofs, block, parallel.adios2.Mode.Sync)
            adios_file.file.EndStep()

    write_on_root_adios(comm, blocks, write)
