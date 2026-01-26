"""
H5py interface to ADIOS4DOLFINx

SPDX License identifier: MIT

Copyright: Jørgen S. Dokken, Henrik N.T. Finsberg, Simula Research Laboratory
"""

import contextlib
from pathlib import Path
from typing import Any, Union

from mpi4py import MPI

import numpy as np
import numpy.typing as npt

from ...structures import MeshData
from ...utils import check_file_exists
from .. import FileMode


@contextlib.contextmanager
def h5pyfile(h5name, filemode="r", force_serial: bool = False, comm=None):
    """Context manager for opening an HDF5 file with h5py.

    Args:
        h5name: The name of the HDF5 file.
        filemode: The file mode.
        force_serial: Force serial access to the file.
        comm: The MPI communicator

    """
    import h5py

    if comm is None:
        comm = MPI.COMM_WORLD

    if h5py.h5.get_config().mpi and comm.size > 1 and not force_serial:
        h5file = h5py.File(h5name, filemode, driver="mpio", comm=comm)
    else:
        if comm.size > 1 and not force_serial:
            raise ValueError(
                f"h5py is not installed with MPI support, while using {comm.size} processes.",
                "If you really want to do this, turn on the `force_serial` flag.",
            )
        h5file = h5py.File(h5name, filemode)
    yield h5file
    h5file.close()


def get_default_backend_args(arguments: dict[str, Any] | None) -> dict[str, Any]:
    args = arguments or {}

    if arguments:
        # Currently no default arguments for h5py backend
        # TODO: Pehaps we would like to make this into a warning instead?
        raise RuntimeError("Unexpected backend arguments to h5py backend")

    return args


def convert_file_mode(mode: FileMode) -> str:
    match mode:
        case FileMode.append:
            return "a"
        case FileMode.read:
            return "r"
        case FileMode.write:
            return "w"
        case _:
            raise NotImplementedError(f"File mode {mode} not implemented")


def write_attributes(
    filename: Union[Path, str],
    comm: MPI.Intracomm,
    name: str,
    attributes: dict[str, np.ndarray],
    backend_args: dict[str, Any] | None = None,
):
    """Write attributes to file using H5PY.

    Args:
        filename: Path to file to write to
        comm: MPI communicator used in storage
        name: Name of the attributes
        attributes: Dictionary of attributes to write to file
        engine: ADIOS2 engine to use
    """

    with h5pyfile(filename, filemode="a", comm=comm, force_serial=False) as h5file:
        if name in h5file.keys():
            group = h5file[name]
        else:
            group = h5file.create_group(name, track_order=True)
        for key, val in attributes.items():
            group.attrs[key] = val


def read_attributes(
    filename: Union[Path, str],
    comm: MPI.Intracomm,
    name: str,
    backend_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read attributes from file using H5PY.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        name: Name of the attributes
    Returns:
        The attributes
    """
    check_file_exists(filename)
    output_attrs = {}
    with h5pyfile(filename, filemode="r", comm=comm, force_serial=False) as h5file:
        for key, val in h5file[name].attrs.items():
            output_attrs[key] = val
    return output_attrs


def read_timestamps(
    filename: Union[Path, str],
    comm: MPI.Intracomm,
    function_name: str,
    backend_args: dict[str, Any] | None = None,
) -> npt.NDArray[np.float64]:
    """
    Read time-stamps from a checkpoint file.

    Args:
        comm: MPI communicator
        filename: Path to file
        function_name: Name of the function to read time-stamps for
        backend_args: Arguments for backend, for instance file type.
        backend: What backend to use for writing.
    Returns:
        The time-stamps
    """
    check_file_exists(filename)
    raise NotImplementedError("Need to be able to save functions before implementing this")


def write_mesh(
    filename: Union[Path, str],
    comm: MPI.Intracomm,
    mesh: MeshData,
    backend_args: dict[str, Any] | None = None,
    mode: FileMode = FileMode.write,
    time: float = 0.0,
):
    """
    Write a mesh to file using H5PY

    Parameters:
        comm: MPI communicator used in storage
        mesh: Internal data structure for the mesh data to save to file
        filename: Path to file to write to.
        mode: Mode to use (write or append)
        time: Time stamp
    """
    backend_args = get_default_backend_args(backend_args)
    h5_mode = convert_file_mode(mode)

    with h5pyfile(filename, filemode=h5_mode, comm=comm, force_serial=False) as h5file:
        if "mesh" in h5file.keys() and h5_mode == "a":
            mesh_directory = h5file["mesh"]
            timestamps = mesh_directory.attrs["timestamps"]
            if np.isclose(time, timestamps).any():
                raise RuntimeError("Mesh has already been stored at time={time_stamp}.")
            else:
                mesh_directory.attrs["timestamps"] = np.append(
                    mesh_directory.attrs["timestamps"], time
                )
                idx = len(mesh_directory.attrs["timestamps"]) - 1
                write_topology = False
        else:
            mesh_directory = h5file.create_group("mesh")
            mesh_directory.attrs["timestamps"] = np.array([time], dtype=np.float64)
            idx = 0
            write_topology = True

        geometry_group = mesh_directory.create_group(f"{idx}")

        # Write geometry data
        gdim = mesh.local_geometry.shape[1]
        geometry_dataset = geometry_group.create_dataset(
            "Points", [mesh.num_nodes_global, gdim], dtype=mesh.local_geometry.dtype
        )
        geometry_dataset[mesh.local_geometry_pos[0] : mesh.local_geometry_pos[1], :] = (
            mesh.local_geometry
        )

        # Write static partitioning data
        if "PartitioningData" not in mesh_directory.keys() and mesh.store_partition:
            assert mesh.partition_range is not None
            par_dataset = mesh_directory.create_dataset("PartitioningData", [mesh.partition_global])
            par_dataset[mesh.partition_range[0] : mesh.partition_range[1]]

        if "PartitioningOffset" not in mesh_directory.keys() and mesh.store_partition:
            assert mesh.ownership_offset is not None
            par_dataset = mesh_directory.create_dataset(
                "PartitioningOffset", [mesh.num_cells_global]
            )
            par_dataset[mesh.local_topology_pos[0] : mesh.local_topology_pos[1]]

        if "PartitionProcesses" not in mesh_directory.attrs.keys() and mesh.store_partition:
            mesh_directory.attrs["PartitionProcesses"] = mesh.partition_processes

        # Write static data
        if write_topology:
            mesh_directory.attrs["CellType"] = mesh.cell_type
            mesh_directory.attrs["Degree"] = mesh.degree
            mesh_directory["LagrangeVariant"] = mesh.lagrange_variant
            num_dofs_per_cell = mesh.local_topology.shape[1]
            topology_dataset = mesh_directory.create_dataset(
                "Topology", [mesh.num_cells_global, num_dofs_per_cell]
            )
            topology_dataset[mesh.local_topology_pos[0] : mesh.local_topology_pos[1], :] = (
                mesh.local_topology
            )
