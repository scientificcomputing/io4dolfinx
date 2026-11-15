"""
Module that uses pyvista to import grids (requires pyvista, through its helpers).
"""

from dataclasses import replace
from pathlib import Path
from typing import Any

from mpi4py import MPI

import basix
import dolfinx
import numpy as np
import numpy.typing as npt

from io4dolfinx.comm_helpers import run_on_root_and_share
from io4dolfinx.structures import ArrayData, FunctionData, MeshData, MeshTagsData, ReadMeshData
from io4dolfinx.utils import check_file_exists

from .. import FileMode, ReadMode
from . import helpers

read_mode = ReadMode.serial


def get_default_backend_args(arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Get default backend arguments given a set of input arguments.

    Args:
        arguments: Input backend arguments

    Returns:
        The arguments, with defaults for:
            - ``"gdim"``: Geometric dimension of the mesh, ``3``. VTK points have three
              coordinates; with a smaller ``gdim`` the others are dropped, and must be zero.
            - ``"dtype"``: Floating point type of the mesh coordinates, ``None`` for the type
              in the file
    """
    args: dict[str, Any] = {"gdim": 3, "dtype": None}
    args.update(arguments or {})
    return args


def _data(grid: helpers.Grid, values: dict[str, np.ndarray], name: str, kind: str) -> np.ndarray:
    """Point or cell data of a grid, with integer data cast to the type of the points."""
    if name not in values:
        raise KeyError(f"No {kind} data '{name}'. Available: {list(values)}")
    data = values[name]
    return data.astype(grid.mesh.x.dtype) if np.issubdtype(data.dtype, np.integer) else data


def read_mesh_data(
    filename: Path | str,
    comm: MPI.Comm,
    time: str | float | None = None,
    read_from_partition: bool = False,
    backend_args: dict[str, Any] | None = None,
) -> ReadMeshData:
    """Read mesh data from file.

    Unstructured grids, and structured, rectilinear and image grids, are read. All data is
    read on rank 0.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        time: Time stamp associated with the mesh to read
        read_from_partition: Whether to read partition information
        backend_args: Arguments to backend, see :func:`get_default_backend_args`

    Returns:
        Internal data structure for the mesh data read from file
    """
    backend_args = get_default_backend_args(backend_args)
    check_file_exists(filename)
    if read_from_partition:
        raise RuntimeError("Cannot read partition data with Pyvista")
    gdim = backend_args["gdim"]

    def read() -> tuple[ReadMeshData, tuple[str, int, int, np.dtype]]:
        mesh = helpers.read_grid(filename).mesh
        if not np.allclose(mesh.x[:, gdim:], 0):
            raise ValueError(
                f"The points of {filename} have non-zero coordinates beyond gdim={gdim}."
            )
        mesh = replace(mesh, x=mesh.x[:, :gdim])
        return mesh, (mesh.cell_type, mesh.degree, mesh.cells.shape[1], mesh.x.dtype)

    mesh, (cell_type, degree, num_nodes, points_dtype) = run_on_root_and_share(comm, read)
    if mesh is None:
        mesh = ReadMeshData(
            cells=np.zeros((0, num_nodes), dtype=np.int64),
            cell_type=cell_type,
            x=np.zeros((0, gdim), dtype=points_dtype),
            lvar=int(basix.LagrangeVariant.equispaced),
            degree=degree,
        )
    return replace(mesh, x=mesh.x.astype(backend_args["dtype"] or points_dtype))


def read_point_data(
    filename: Path | str,
    name: str,
    comm: MPI.Comm,
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
       The data on rank 0, of shape ``(num_points, num_components)``, and the global start
       of the data on this process, ``0``
    """

    def read() -> tuple[np.ndarray, tuple[int, np.dtype]]:
        grid = helpers.read_grid(filename)
        data = _data(grid, grid.point_data, name, "point")
        return data, (data.shape[1], data.dtype)

    data, (num_components, dtype) = run_on_root_and_share(comm, read)
    if data is None:
        data = np.zeros((0, num_components), dtype=dtype)
    return data, 0


def read_cell_data(
    filename: Path | str,
    name: str,
    comm: MPI.Comm,
    time: str | float | None,
    backend_args: dict[str, Any] | None,
) -> tuple[npt.NDArray[np.int64], np.ndarray]:
    """Read data from the cells of a mesh.

    Args:
        filename: Path to file
        name: Name of cell data
        comm: Communicator to launch IO on.
        time: The time stamp
        backend_args: The backend arguments

    Returns:
        A tuple (topology, data), where topology holds the input node indices of each cell
        and data the data of each cell, on rank 0
    """

    def read() -> tuple[tuple[npt.NDArray[np.int64], np.ndarray], tuple[int, int, np.dtype]]:
        grid = helpers.read_grid(filename)
        data = _data(grid, grid.cell_data, name, "cell")
        cells = grid.mesh.cells
        return (cells, data), (cells.shape[1], data.shape[1], data.dtype)

    local, (num_nodes, num_components, dtype) = run_on_root_and_share(comm, read)
    if local is None:
        local = (np.zeros((0, num_nodes), dtype=np.int64), np.zeros((0, num_components), dtype))
    return local


def write_attributes(
    filename: Path | str,
    comm: MPI.Comm,
    name: str,
    attributes: dict[str, np.ndarray],
    backend_args: dict[str, Any] | None,
):
    """Write attributes to file.

    Args:
        filename: Path to file to write to
        comm: MPI communicator used in storage
        name: Name of the attribute group
        attributes: Dictionary of attributes to write
        backend_args: Arguments to backend
    """
    raise NotImplementedError("The Pyvista backend cannot write attributes.")


def read_attributes(
    filename: Path | str,
    comm: MPI.Comm,
    name: str,
    backend_args: dict[str, Any] | None,
) -> dict[str, Any]:
    """Read attributes from file.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        name: Name of the attribute group
        backend_args: Arguments to backend

    Returns:
        Dictionary of attributes read from file
    """
    raise NotImplementedError("The Pyvista backend cannot read attributes.")


def read_timestamps(
    filename: Path | str,
    comm: MPI.Comm,
    function_name: str,
    backend_args: dict[str, Any] | None,
) -> npt.NDArray[np.float64 | str]:  # type: ignore[type-var]
    """Read timestamps from file.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        function_name: Name of the function to read timestamps for
        backend_args: Arguments to backend

    Returns:
        Numpy array of timestamps read from file
    """
    raise NotImplementedError("The Pyvista backend cannot read timestamps.")


def read_function_names(
    filename: Path | str, comm: MPI.Comm, backend_args: dict[str, Any] | None
) -> list[str]:
    """Read all function names from a file.

    Args:
        filename: Path to file
        comm: MPI communicator to launch IO on.
        backend_args: Arguments to backend

    Returns:
        The names of the point data, then of the cell data.
    """

    def read() -> tuple[None, list[str]]:
        grid = helpers.read_grid(filename)
        return None, list(grid.point_data) + list(grid.cell_data)

    return run_on_root_and_share(comm, read)[1]


def write_mesh(
    filename: Path | str,
    comm: MPI.Comm,
    mesh: MeshData,
    backend_args: dict[str, Any] | None,
    mode: FileMode,
    time: float,
):
    """
    Write a mesh to file.

    Args:
        comm: MPI communicator used in storage
        mesh: Internal data structure for the mesh data to save to file
        filename: Path to file to write to
        backend_args: Arguments to backend
        mode: File-mode to store the mesh
        time: Time stamp associated with the mesh
    """
    raise NotImplementedError("The Pyvista backend cannot write meshes.")


def write_meshtags(
    filename: str | Path,
    comm: MPI.Comm,
    data: MeshTagsData,
    backend_args: dict[str, Any] | None,
):
    """Write mesh tags to file.

    Args:
        filename: Path to file to write to
        comm: MPI communicator used in storage
        data: Internal data structure for the mesh tags to save to file
        backend_args: Arguments to backend
    """
    raise NotImplementedError("The Pyvista backend cannot write meshtags.")


def read_meshtags_data(
    filename: str | Path,
    comm: MPI.Comm,
    name: str,
    backend_args: dict[str, Any] | None,
) -> MeshTagsData:
    """Read mesh tags from file.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        name: Name of the mesh tags to read
        backend_args: Arguments to backend

    Returns:
        Internal data structure for the mesh tags read from file
    """
    raise NotImplementedError("The Pyvista backend cannot read meshtags.")


def read_dofmap(
    filename: str | Path,
    comm: MPI.Comm,
    name: str,
    backend_args: dict[str, Any] | None,
) -> dolfinx.graph.AdjacencyList:
    """Read the dofmap of a function with a given name.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        name: Name of the function to read the dofmap for
        backend_args: Arguments to backend

    Returns:
        Dofmap as an AdjacencyList
    """
    raise NotImplementedError("The Pyvista backend cannot make checkpoints.")


def read_dofs(
    filename: str | Path,
    comm: MPI.Comm,
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
    raise NotImplementedError("The Pyvista backend cannot make checkpoints.")


def read_cell_perms(
    comm: MPI.Comm, filename: Path | str, backend_args: dict[str, Any] | None
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
    raise NotImplementedError("The Pyvista backend cannot make checkpoints.")


def write_function(
    filename: Path,
    comm: MPI.Comm,
    u: FunctionData,
    time: float,
    mode: FileMode,
    backend_args: dict[str, Any] | None,
):
    """
    Write a function to file.

    Args:
        comm: MPI communicator used in storage
        u: Internal data structure for the function data to save to file
        filename: Path to file to write to
        time: Time stamp associated with function
        mode: File-mode to store the function
        backend_args: Arguments to backend
    """
    raise NotImplementedError("The Pyvista backend cannot make checkpoints.")


def read_legacy_mesh(
    filename: Path | str, comm: MPI.Comm, group: str
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
    raise NotImplementedError("The Pyvista backend cannot read legacy DOLFIN meshes.")


def snapshot_checkpoint(
    filename: Path | str,
    mode: FileMode,
    u: dolfinx.fem.Function,
    backend_args: dict[str, Any] | None,
):
    """Create a snapshot checkpoint of a dolfinx function.

    Args:
        filename: Path to file to read from
        mode: File-mode to store the function
        u: dolfinx function to create a snapshot checkpoint for
        backend_args: Arguments to backend
    """
    raise NotImplementedError("The Pyvista backend cannot make checkpoints.")


def read_hdf5_array(
    comm: MPI.Comm,
    filename: Path | str,
    group: str,
    backend_args: dict[str, Any] | None,
) -> tuple[np.ndarray, int]:
    """Read an array from an HDF5 file.

    Args:
        comm: MPI communicator used in storage
        filename: Path to file to read from
        group: Group in HDF5 file where array is stored
        backend_args: Arguments to backend

    Returns:
        Tuple containing:
            - Numpy array read from file
            - Global starting point on the process.
                Process 0 has [0, M), process 1 [M, N), process 2 [N, O) etc.
    """
    raise NotImplementedError("The Pyvista backend cannot read HDF5 arrays")


def write_data(
    filename: Path | str,
    array_data: ArrayData,
    comm: MPI.Comm,
    time: str | float | None,
    mode: FileMode,
    backend_args: dict[str, Any] | None,
):
    """Write a 2D-array to file (distributed across proceses with MPI).

    Args:
        filename: Path to file
        array_data: Data to write to file
        comm: MPI communicator to open the file with
        time: Time stamp
        mode: Append or write
        backend_args: The backend arguments
    """
    raise NotImplementedError("The pyvista backend does not support writing point data")
