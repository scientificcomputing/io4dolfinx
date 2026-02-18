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
import netCDF4

import dolfinx
import numpy as np
import numpy.typing as npt
from dolfinx.graph import adjacencylist
import basix

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
        elif upper in ["QUAD", "QUAD4"]:
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
            return "quadrilateral"
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
    """Read mesh data from EXODUS based checkpoint files.

    Args:
        filename: Path to input file
        comm: The MPI communciator to distribute the mesh over
        time: Time stamp associated with mesh
        read_from_partition: Read mesh with partition from file
    Returns:
        The mesh topology, geometry, UFL domain and partition function
    """

    try:
        infile = netCDF4.Dataset(filename)

        # use page 171 of manual to extract data
        num_nodes = infile.dimensions["num_nodes"].size
        gdim = infile.dimensions["num_dim"].size
        num_blocks = infile.dimensions["num_el_blk"].size

        # Get coordinates of mesh
        coordinates = infile.variables.get("coord")
        if coordinates is None:
            coordinates = np.zeros((num_nodes, gdim), dtype=np.float64)
            for i, coord in enumerate(["x", "y", "z"]):
                coord_i = infile.variables.get(f"coord{coord}")
                if coord_i is not None:
                    coordinates[: coord_i.size, i] = coord_i[:]

        # Get element connectivity
        connectivity_arrays = []
        cell_types = np.empty(num_blocks, dtype=CellType)
        num_cells_per_block = np.zeros(num_blocks, dtype=np.int32)
        # Create map from topological dimension to block indices
        tdim_to_cell_index = {0: [], 1: [], 2: [], 3: []}
        for i in range(1, num_blocks + 1):
            connectivity = infile.variables.get(f"connect{i}")

            cell_type = CellType.from_value(str(ExodusCellType.from_value(connectivity.elem_type)))
            cell_types[i - 1] = cell_type
            tdim_to_cell_index[cell_type.tdim].append(i - 1)
            assert connectivity is not None, "No connectivity found"
            connectivity_arrays.append(connectivity[:] - 1)
            num_cells_per_block[i - 1] = connectivity.shape[0]
        max_dim = 0
        for i in range(4):
            tdim_to_cell_index[i] = np.asarray(tdim_to_cell_index[i], dtype=np.int32)
            if len(tdim_to_cell_index[i]) > 0:
                max_dim = i
        cell_block_indices = tdim_to_cell_index[max_dim]
        for cell in cell_types[cell_block_indices]:
            assert cell_types[cell_block_indices[0]] == cell, "Mixed cell types not supported"
        cell_type = cell_types[cell_block_indices][0]

        connectivity_array = np.vstack([connectivity_arrays[i] for i in cell_block_indices])
        cells = connectivity_array.data

        perm = dolfinx.cpp.io.perm_vtk(dolfinx.mesh.to_type(str(cell_type)), cells.shape[1])
        cells = cells[:, perm]
    finally:
        infile.close()

    return ReadMeshData(
        cells=cells,
        cell_type=str(cell_type),
        x=coordinates,
        lvar=int(basix.LagrangeVariant.equispaced),
        degree=np.int32(1),
        partition_graph=None,
    )


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
    filename: str | Path,
    comm: MPI.Intracomm,
    name: str,
    backend_args: dict[str, Any] | None = None,
) -> MeshTagsData:
    """Read mesh tags from file.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        name: Name of the mesh tags to read
        backend_args: Arguments to backend.

    Returns:
        Internal data structure for the mesh tags read from file
    """
    try:
        infile = netCDF4.Dataset(filename)

        # use page 171 of manual to extract data
        num_nodes = infile.dimensions["num_nodes"].size
        gdim = infile.dimensions["num_dim"].size
        num_blocks = infile.dimensions["num_el_blk"].size

        # Get coordinates of mesh
        coordinates = infile.variables.get("coord")
        if coordinates is None:
            coordinates = np.zeros((num_nodes, gdim), dtype=np.float64)
            for i, coord in enumerate(["x", "y", "z"]):
                coord_i = infile.variables.get(f"coord{coord}")
                if coord_i is not None:
                    coordinates[: coord_i.size, i] = coord_i[:]

        # Get element connectivity
        connectivity_arrays = []
        cell_types = np.empty(num_blocks, dtype=CellType)
        num_cells_per_block = np.zeros(num_blocks, dtype=np.int32)
        # Create map from topological dimension to block indices
        tdim_to_cell_index = {0: [], 1: [], 2: [], 3: []}
        for i in range(1, num_blocks + 1):
            connectivity = infile.variables.get(f"connect{i}")

            cell_type = CellType.from_value(str(ExodusCellType.from_value(connectivity.elem_type)))
            cell_types[i - 1] = cell_type
            tdim_to_cell_index[cell_type.tdim].append(i - 1)
            assert connectivity is not None, "No connectivity found"
            connectivity_arrays.append(connectivity[:] - 1)
            num_cells_per_block[i - 1] = connectivity.shape[0]
        max_dim = 0
        for i in range(4):
            tdim_to_cell_index[i] = np.asarray(tdim_to_cell_index[i], dtype=np.int32)
            if len(tdim_to_cell_index[i]) > 0:
                max_dim = i
        cell_block_indices = tdim_to_cell_index[max_dim]
        for cell in cell_types[cell_block_indices]:
            assert cell_types[cell_block_indices[0]] == cell, "Mixed cell types not supported"
        cell_type = cell_types[cell_block_indices][0]
        connectivity_array = np.vstack([connectivity_arrays[i] for i in cell_block_indices])

        if name == "cell":
            if "eb_prop1" in infile.variables.keys():
                block_values = infile.variables["eb_prop1"][:]

                # Extract cell block values
                cell_array = np.zeros(connectivity_array.shape[0], dtype=np.int64)
                insert_offset = np.zeros(len(cell_block_indices) + 1, dtype=np.int64)
                insert_offset[1:] = np.cumsum(num_cells_per_block[cell_block_indices])
                for i, index in enumerate(cell_block_indices):
                    cell_array[insert_offset[i] : insert_offset[i + 1]] = block_values[index]
                vals = cell_array
                indices = connectivity_array.data
                dim = cell_type.tdim
        elif name == "facet":
            # Get all facet blocks
            facet_blocks_indices = tdim_to_cell_index[max_dim - 1]
            if len(facet_blocks_indices) > 0:
                sub_geometry = np.vstack([connectivity_arrays[i] for i in facet_blocks_indices])
                facet_values = np.zeros(sub_geometry.shape[0], dtype=np.int64)
                insert_offset = np.zeros(len(facet_blocks_indices) + 1, dtype=np.int64)
                insert_offset[1:] = np.cumsum(num_cells_per_block[facet_blocks_indices])
                for i, index in enumerate(facet_blocks_indices):
                    facet_values[insert_offset[i] : insert_offset[i + 1]] = block_values[index]
            # If sidesets are used for facet markers
            elif "ss_prop1" in infile.variables.keys():
                # Extract facet values
                local_facet_index = side_set_to_vertex_map[cell_type]
                if "num_side_sets" not in infile.dimensions:
                    num_vertices_per_facet = len(local_facet_index[0])
                    sub_geometry = np.zeros((0, num_vertices_per_facet), dtype=np.int64)
                    facet_values = np.zeros(0, dtype=np.int64)
                else:
                    infile.dimensions.get("num_side_sets", 0)
                    num_facet_sets = infile.dimensions["num_side_sets"].size
                    values = infile.variables.get("ss_prop1")
                    facet_indices = []
                    facet_values = []
                    for i in range(1, num_facet_sets + 1):
                        value = values[i - 1]
                        elements = infile.variables[f"elem_ss{i}"]
                        local_facets = infile.variables[f"side_ss{i}"]
                        for element, index in zip(elements, local_facets):
                            facet_indices.append(
                                connectivity_array[element - 1, local_facet_index[index - 1]]
                            )

                            # @jorgen can you check this please, I had to change it cause
                            # each `value` object was a masked array with only one element
                            # facet_values.append(value)
                            facet_values.append(value.data.tolist())
                    sub_geometry = np.vstack(facet_indices)
                    facet_values = np.array(facet_values, dtype=np.int64)
            else:
                sub_geometry = np.zeros((0, 0), dtype=np.int64)
                facet_values = np.zeros(0, dtype=np.int64)
            vals = facet_values
            indices = sub_geometry.data
            dim = cell_type.tdim - 1

    finally:
        infile.close()
    return MeshTagsData(name=name, values=vals, indices=indices, dim=dim)


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
