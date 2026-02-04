"""
Module that can read the VTKHDF format using h5py.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from pathlib import Path

from mpi4py import MPI

import basix
import dolfinx

from adios4dolfinx.structures import FunctionData, MeshData, MeshTagsData, ReadMeshData
from adios4dolfinx.utils import check_file_exists, compute_local_range

from .. import FileMode, ReadMode
from ..h5py.backend import h5pyfile
from ..pyvista.backend import _arbitrary_lagrange_vtk, _cell_degree, _first_order_vtk

read_mode = ReadMode.parallel


def get_default_backend_args(arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Get default backend arguments given a set of input arguments.

    Args:
        arguments: Input backend arguments

    Returns:
        Updated backend arguments
    """
    args = arguments or {}
    return args


def find_all_unique_cell_types(comm, cell_types, num_nodes):
    """
    Given a set of cell types and number of nodes per cell, find all unique cell types
    across all ranks.

    Args:
        comm: MPI communicator
        cell_types: Local cell types
        num_nodes: Number of nodes per cell


    Returns:
        A 2D array where each row corresponds to a cell type (vtk int)
        and the number of nodes.
    """
    # Combine cell_types, num_nodes as tuple
    c_hash = np.zeros((2, len(cell_types)), dtype=np.int32)
    c_hash[0] = cell_types
    c_hash[1] = num_nodes
    indexes = np.unique(c_hash.T, axis=0, return_index=True)[1]
    local_unique_cells = c_hash.T[indexes]

    all_cell_types = np.vstack(comm.allgather(local_unique_cells))
    indexes = np.unique(all_cell_types, axis=0, return_index=True)[1]
    all_unique_cell_types = all_cell_types[indexes]
    return all_unique_cell_types


def read_mesh_data(
    filename: Path | str,
    comm: MPI.Intracomm,
    time: str | float | None,
    read_from_partition: bool,
    backend_args: dict[str, Any] | None,
) -> ReadMeshData:
    """Read mesh data from file.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        time: Time stamp associated with the mesh to read
        read_from_partition: Whether to read partition information
        backend_args: Arguments to backend

    Returns:
        Internal data structure for the mesh data read from file
    """
    backend_args = get_default_backend_args(backend_args)
    check_file_exists(filename)
    if read_from_partition:
        raise RuntimeError("Cannot read partition data with VTKHDF")

    with h5pyfile(filename, "r", comm=comm) as h5file:
        hdf = h5file["VTKHDF"]
        num_cells_global = hdf["Types"].size

        local_cell_range = compute_local_range(comm, num_cells_global)
        cell_types_local = hdf["Types"][local_cell_range[0] : local_cell_range[1]]

        num_points_global = hdf["NumberOfPoints"][0]
        local_point_range = compute_local_range(comm, num_points_global)
        points_local = hdf["Points"][local_point_range[0] : local_point_range[1]]

        # Connectivity read
        offsets = hdf["Offsets"]
        local_connectivity_offset = offsets[local_cell_range[0] : local_cell_range[1] + 1]
        topology = hdf["Connectivity"][local_connectivity_offset[0] : local_connectivity_offset[-1]]
    offset = local_connectivity_offset - local_connectivity_offset[0]

    # NOTE: Currently we limit ourselfs to a single celltype, as it makes life easier,
    # other things have to change in `MeshReadData` to support this.
    num_nodes_per_cell = offset[1:] - offset[:-1]
    unique_cells = find_all_unique_cell_types(MPI.COMM_WORLD, cell_types_local, num_nodes_per_cell)
    if unique_cells.shape[0] > 1:
        raise NotImplementedError("adios4dolfinx does not support mixed celltype grids")
    topology = topology.reshape(-1, num_nodes_per_cell[0])
    cell_type, number_of_nodes = unique_cells[0]
    gtype = backend_args.get("dtype", points_local.dtype)
    if cell_type in _first_order_vtk.keys():
        ct = _first_order_vtk[cell_type]
        degree = 1
    elif cell_type in _arbitrary_lagrange_vtk.keys():
        ct = _arbitrary_lagrange_vtk[cell_type]
        degree = _cell_degree(ct, num_nodes=number_of_nodes)
    else:
        raise ValueError(f"Unknown VTK cell type {cell_type} in {filename}")
    lvar = int(basix.LagrangeVariant.equispaced)
    return ReadMeshData(
        cells=topology, cell_type=ct, x=points_local.astype(gtype), lvar=lvar, degree=degree
    )


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
    raise NotImplementedError("This is not implemented in VTKHDF yet.")


def read_cell_data(
    filename: Path | str,
    name: str,
    comm: MPI.Intracomm,
    time: str | float | None,
    backend_args: dict[str, Any] | None,
) -> tuple[npt.NDArray[np.int64], np.ndarray]:
    backend_args = get_default_backend_args(backend_args)
    check_file_exists(filename)
    with h5pyfile(filename, "r", comm=comm) as h5file:
        hdf = h5file["VTKHDF"]
        num_cells_global = hdf["Types"].size
        cell_data = hdf["CellData"]
        cell_data_node = cell_data[name]
        cell_data_shape = cell_data_node.shape
        assert num_cells_global == cell_data_shape[0]
        local_cell_range = compute_local_range(comm, num_cells_global)
        data = cell_data_node[slice(*local_cell_range)]
    # NOTE: THis could be optimized by hand-coding some communication in
    # `read_cell_data` on the frontend side
    md = read_mesh_data(filename, comm, time=time, read_from_partition=False, backend_args=None)
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    return md.cells, data


def write_attributes(
    filename: Path | str,
    comm: MPI.Intracomm,
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
    comm: MPI.Intracomm,
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
    comm: MPI.Intracomm,
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
    filename: Path | str, comm: MPI.Intracomm, backend_args: dict[str, Any] | None
) -> list[str]:
    """Read all function names from a file.

    Args:
        filename: Path to file
        comm: MPI communicator to launch IO on.
        backend_args: Arguments to backend

    Returns:
        A list of function names.
    """
    backend_args = get_default_backend_args(backend_args)
    check_file_exists(filename)
    with h5pyfile(filename, "r", comm=comm) as h5file:
        hdf = h5file["VTKHDF"]
        function_names = set()
        if "CellData" in hdf.keys():
            for item in hdf["CellData"].keys():
                function_names.add(item)
        if "PointData" in hdf.keys():
            for item in hdf["PointData"].keys():
                function_names.add(item)
    return list(function_names)


def write_mesh(
    filename: Path | str,
    comm: MPI.Intracomm,
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
    comm: MPI.Intracomm,
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
    comm: MPI.Intracomm,
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
    comm: MPI.Intracomm,
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
    raise NotImplementedError("The Pyvista backend cannot make checkpoints.")


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
    raise NotImplementedError("The Pyvista backend cannot make checkpoints.")


def write_function(
    filename: Path,
    comm: MPI.Intracomm,
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
    comm: MPI.Intracomm,
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
