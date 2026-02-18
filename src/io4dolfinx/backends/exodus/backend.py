"""
Exodus interface to io4dolfinx

SPDX License identifier: MIT

Copyright: Jørgen S. Dokken, Henrik N.T. Finsberg, Simula Research Laboratory
"""

import contextlib
from enum import Enum
from pathlib import Path
from typing import Any

from mpi4py import MPI

import dolfinx
import h5py
import numpy as np
import numpy.typing as npt
from dolfinx.graph import adjacencylist

from ...structures import ArrayData, FunctionData, MeshData, MeshTagsData, ReadMeshData
from ...utils import check_file_exists, compute_local_range
from .. import FileMode, ReadMode
from .mesh import CellType, Mesh

# Based on: https://src.fedoraproject.org/repo/pkgs/exodusii/922137.pdf/a45d67f4a1a8762bcf66af2ec6eb35f9/922137.pdf
tetra_facet_to_vertex_map = {0: [0, 1, 3], 1: [1, 2, 3], 2: [0, 2, 3], 3: [0, 1, 2]}
# https://coreform.com/cubit_help/appendix/element_numbering.htm
# Note that triangular side-sets goes from 2 to 4 (with 0 base index)
triangle_to_vertex_map = {2: [0, 1], 3: [1, 2], 4: [2, 0]}
quad_to_vertex_map = {0: [0, 1], 1: [1, 2], 2: [2, 3], 3: [3, 0]}
hex_to_vertex_map = {
    0: [0, 1, 4, 5],
    1: [1, 2, 5, 6],
    2: [2, 3, 6, 7],
    3: [0, 3, 4, 7],
    4: [0, 1, 2, 3],
    5: [4, 5, 6, 7],
}

side_set_to_vertex_map = {
    CellType.quad: quad_to_vertex_map,
    CellType.triangle: triangle_to_vertex_map,
    CellType.tetra: tetra_facet_to_vertex_map,
    CellType.hex: hex_to_vertex_map,
}


read_mode = ReadMode.parallel


class ExodusCellType(Enum):
    TETRA = 1
    HEX = 2
    QUAD = 3
    TRIANGLE = 4
    INTERVAL = 5

    @classmethod
    def from_value(cls, value: str):
        """
        Workaround for string enum prior to Python 3.11
        """
        upper = value.upper()
        if upper == "TRI3":
            return cls.TRIANGLE
        elif upper == "QUAD":
            return cls.QUAD
        elif upper == "TETRA":
            return cls.TETRA
        elif upper in ["HEX", "HEX8"]:
            return cls.HEX
        elif upper == "BEAM2":
            return cls.INTERVAL
        else:
            raise ValueError(f"Unknown cell type: {value}")

    def __str__(self) -> str:
        if self == ExodusCellType.TETRA:
            return "tetra"
        elif self == ExodusCellType.HEX:
            return "hex"
        elif self == ExodusCellType.QUAD:
            return "quad"
        elif self == ExodusCellType.TRIANGLE:
            return "triangle"
        elif self == ExodusCellType.INTERVAL:
            return "interval"
        else:
            raise ValueError(f"Unknown cell type: {self}")


def get_default_backend_args(arguments: dict[str, Any] | None) -> dict[str, Any]:
    args = arguments or {"legacy": False}  # If meshtags is read from legacy
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
    filename: Path | str,
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
    raise NotImplementedError("The Exodus backend cannot write attributes.")


def read_attributes(
    filename: Path | str,
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
    raise NotImplementedError("The Exodus backend cannot read attributes.")


def read_timestamps(
    filename: Path | str,
    comm: MPI.Intracomm,
    function_name: str,
    backend_args: dict[str, Any] | None = None,
) -> npt.NDArray[np.float64 | str]:  # type: ignore[type-var]
    """Read time-stamps from a checkpoint file.

    Args:
        comm: MPI communicator
        filename: Path to file
        function_name: Name of the function to read time-stamps for
        backend_args: Arguments for backend, for instance file type.
        backend: What backend to use for writing.
    Returns:
        The time-stamps
    """
    raise NotImplementedError("The Exodus backend cannot read timestamps.")


def write_mesh(
    filename: Path | str,
    comm: MPI.Intracomm,
    mesh: MeshData,
    backend_args: dict[str, Any] | None = None,
    mode: FileMode = FileMode.write,
    time: float = 0.0,
):
    """Write a mesh to file using H5PY

    Args:
        comm: MPI communicator used in storage
        mesh: Internal data structure for the mesh data to save to file
        filename: Path to file to write to.
        mode: Mode to use (write or append)
        time: Time stamp
    """
    raise NotImplementedError("The Exodus backend cannot write meshes.")


def read_mesh_data(
    filename: Path | str,
    comm: MPI.Intracomm,
    time: str | float | None = 0.0,
    read_from_partition: bool = False,
    backend_args: dict[str, Any] | None = None,
) -> ReadMeshData:
    """Read mesh data from h5py based checkpoint files.

    Args:
        filename: Path to input file
        comm: The MPI communciator to distribute the mesh over
        time: Time stamp associated with mesh
        read_from_partition: Read mesh with partition from file
    Returns:
        The mesh topology, geometry, UFL domain and partition function
    """

    pass


def write_meshtags(
    filename: str | Path,
    comm: MPI.Intracomm,
    data: MeshTagsData,
    backend_args: dict[str, Any] | None = None,
):
    """Write mesh tags to file.

    Args:
        filename: Path to file to write to
        comm: MPI communicator used in storage
        data: Internal data structure for the mesh tags to save to file
        backend_args: Arguments to backend
    """
    raise NotImplementedError("The Exodus backend cannot write meshtags.")


def read_meshtags_data(
    filename: str | Path, comm: MPI.Intracomm, name: str, backend_args: dict[str, Any] | None = None
) -> MeshTagsData:
    """Read mesh tags from file.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        name: Name of the mesh tags to read
        backend_args: Arguments to backend. If "legacy_dolfin" is supplied as argument
            the HDF5 file is assumed to have been made with DOLFIN

    Returns:
        Internal data structure for the mesh tags read from file
    """
    pass


def read_dofmap(
    filename: str | Path, comm: MPI.Intracomm, name: str, backend_args: dict[str, Any] | None
) -> dolfinx.graph.AdjacencyList:
    """Read the dofmap of a function with a given name.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        name: Name of the function to read the dofmap for
        backend_args: Arguments to backend

    Returns:
        Dofmap as an {py:class}`dolfinx.graph.AdjacencyList`
    """
    raise NotImplementedError("The Exodus backend cannot read dofmap.")


def read_dofs(
    filename: str | Path,
    comm: MPI.Intracomm,
    name: str,
    time: float,
    backend_args: dict[str, Any] | None,
) -> tuple[npt.NDArray[np.float32 | np.float64 | np.complex64 | np.complex128], int]:
    """Read the dofs (values) of a function with a given name from a given timestep.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        name: Name of the function to read the dofs for
        time: Time stamp associated with the function to read
        backend_args: Arguments to backend

    Returns:
        Contiguous sequence of degrees of freedom (with respect to input data)
        and the global starting point on the process.
        Process 0 has [0, M), process 1 [M, N), process 2 [N, O) etc.
    """
    raise NotImplementedError("The Exodus backend cannot read dofs.")


def read_cell_perms(
    comm: MPI.Intracomm, filename: Path | str, backend_args: dict[str, Any] | None
) -> npt.NDArray[np.uint32]:
    """
    Read cell permutation from file with given communicator,
    Split in continuous chunks based on number of cells in the input data.

    Args:
        comm: MPI communicator used in storage
        filename: Path to file to read from
        backend_args: Arguments to backend

    Returns:
        Contiguous sequence of permutations (with respect to input data)
        Process 0 has [0, M), process 1 [M, N), process 2 [N, O) etc.
    """

    raise NotImplementedError("The Exodus backend cannot read cell perms.")


def write_function(
    filename: str | Path,
    comm: MPI.Intracomm,
    u: FunctionData,
    time: float,
    mode: FileMode,
    backend_args: dict[str, Any] | None = None,
):
    """Write a function to file.

    Args:
        comm: MPI communicator used in storage
        u: Internal data structure for the function data to save to file
        filename: Path to file to write to
        time: Time stamp associated with function
        mode: File-mode to store the function
        backend_args: Arguments to backend
    """

    raise NotImplementedError("The Exodus backend cannot write function.")


def read_legacy_mesh(
    filename: Path | str, comm: MPI.Intracomm, group: str
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.floating], str | None]:
    """Read in the mesh topology, geometry and (optionally) cell type from a
    legacy DOLFIN HDF5-file.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        group: Group in HDF5 file where mesh is stored

    Returns:
        Tuple containing:
            - Topology as a (num_cells, num_vertices_per_cell) array of global vertex indices
            - Geometry as a (num_vertices, geometric_dimension) array of vertex coordinates
            - Cell type as a string (e.g. "tetrahedron") or None if not found
    """
    raise NotImplementedError("The Exodus backend cannot read legacy mesh.")


def read_point_data(
    filename: Path | str,
    name: str,
    comm: MPI.Intracomm,
    time: float | str | None,
    backend_args: dict[str, Any] | None,
) -> tuple[np.ndarray, int]:
    """Read data from the nodes of a mesh.

    Args:
        filename: Path to file
        name: Name of point data
        comm: Communicator to launch IO on.
        time: The time stamp
        backend_args: The backend arguments

    Returns:
       Data local to process (contiguous, no mpi comm) and local start range
    """
    raise NotImplementedError("The Exodus backend cannot read point data.")


def read_cell_data(
    filename: Path | str,
    name: str,
    comm: MPI.Intracomm,
    time: str | float | None,
    backend_args: dict[str, Any] | None,
) -> tuple[npt.NDArray[np.int64], np.ndarray]:
    """Read data from the cells of a mesh.

    Args:
        filename: Path to file
        name: Name of point data
        comm: Communicator to launch IO on.
        time: The time stamp
        backend_args: The backend arguments
    Returns:
        A tuple (topology, dofs) where topology contains the
        vertex indices of the cells, dofs the degrees of
        freedom within that cell.
    """
    raise NotImplementedError("The Exodus backend does not support reading cell data.")


def write_data(
    filename: Path | str,
    array_data: ArrayData,
    comm: MPI.Intracomm,
    time: str | float | None,
    mode: FileMode,
    backend_args: dict[str, Any] | None,
):
    """Write a 2D-array to file (distributed across proceses with MPI).

    Args:
        filename: Path to file
        array_data: Data to write to file
        comm: MPI communicator to open the file with
        time: Time-stamp for data.
        mode: Append or write
        backend_args: The backend arguments
    """
    raise NotImplementedError("Exodus has not implemented this yet")
