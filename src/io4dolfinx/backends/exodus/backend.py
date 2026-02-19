"""
Exodus interface to io4dolfinx

SPDX License identifier: MIT

Copyright: Jørgen S. Dokken, Henrik N.T. Finsberg, Remi Delaporte-Mathurin,
           and Simula Research Laboratory
"""

from pathlib import Path
from typing import Any

from mpi4py import MPI

import basix
import dolfinx
import netCDF4
import numpy as np
import numpy.typing as npt

from ...structures import ArrayData, FunctionData, MeshData, MeshTagsData, ReadMeshData
from ...utils import check_file_exists
from .. import FileMode, ReadMode

# Based on: https://src.fedoraproject.org/repo/pkgs/exodusii/922137.pdf/a45d67f4a1a8762bcf66af2ec6eb35f9/922137.pdf
_tetra_facet_to_vertex_map = {0: [0, 1, 3], 1: [1, 2, 3], 2: [0, 2, 3], 3: [0, 1, 2]}
# https://coreform.com/cubit_help/appendix/element_numbering.htm
# Note that triangular side-sets goes from 2 to 4 (with 0 base index)
_triangle_to_vertex_map = {2: [0, 1], 3: [1, 2], 4: [2, 0]}
_quad_to_vertex_map = {0: [0, 1], 1: [1, 2], 2: [2, 3], 3: [3, 0]}
_hex_to_vertex_map = {
    0: [0, 1, 4, 5],
    1: [1, 2, 5, 6],
    2: [2, 3, 6, 7],
    3: [0, 3, 4, 7],
    4: [0, 1, 2, 3],
    5: [4, 5, 6, 7],
}

_side_set_to_vertex_map = {
    "quadrilateral": _quad_to_vertex_map,
    "triangle": _triangle_to_vertex_map,
    "tetrahedron": _tetra_facet_to_vertex_map,
    "hexahedron": _hex_to_vertex_map,
}


_exodus_to_string = {
    "TRI3": "triangle",
    "QUAD": "quadrilateral",
    "QUAD4": "quadrilateral",
    "TETRA": "tetrahedron",
    "HEX": "hexahedron",
    "HEX8": "hexahedron",
    "BEAM2": "interval",
}

read_mode = ReadMode.serial


def _get_cell_type(connectivity: netCDF4._netCDF4.Variable) -> dolfinx.mesh.CellType:
    cell_type = _exodus_to_string[connectivity.elem_type]
    return dolfinx.mesh.to_type(cell_type)


def _compute_tdim(connectivity: netCDF4._netCDF4.Variable) -> int:
    d_ct = _get_cell_type(connectivity)
    return dolfinx.mesh.cell_dim(d_ct)


def get_default_backend_args(arguments: dict[str, Any] | None) -> dict[str, Any]:
    args = arguments or {}
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
    check_file_exists(filename)
    with netCDF4.Dataset(filename, "r") as infile:
        if comm.rank == 0:
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
            all_connectivity_variables = [
                infile.variables.get(f"connect{i + 1}") for i in range(num_blocks)
            ]

            # Compute max topological dimension in mesh and find the correct
            max_tdim = _compute_tdim(max(all_connectivity_variables, key=_compute_tdim))

            # Extract only the connectivity blocks that we need
            entity_blocks = list(
                filter(lambda el: _compute_tdim(el) == max_tdim, all_connectivity_variables)
            )
            if len(entity_blocks) > 0:
                # Extract markers directly from entity-blocks
                connectivity_arrays = []
                cell_types = []
                num_entities = []
                entity_block_index = []
                for entity_block in entity_blocks:
                    connectivity_arrays.append(entity_block[:] - 1)
                    num_entities.append(entity_block.shape[0])
                    cell_types.append(_get_cell_type(entity_block))
                    entity_block_index.append(int(entity_block.name.removeprefix("connect")) - 1)
                for cell in cell_types:
                    assert cell_types[0] == cell, "Mixed cell types not supported"
                cell_type = cell_types[0]

                cells = np.vstack(connectivity_arrays)
            else:
                raise ValueError(f"No blocks found in {filename}")
            perm = dolfinx.cpp.io.perm_vtk(cell_type, cells.shape[1])
            cells = cells[:, perm]
            cell_type, gdim, xtype, num_dofs_per_cell = comm.bcast(
                (cell_type, gdim, np.dtype(coordinates.dtype).name, cells.shape[1]), root=0
            )

        else:
            cell_type, gdim, xtype, num_dofs_per_cell = comm.bcast((None, None, None, None), root=0)
            coordinates = np.zeros((0, gdim), dtype=xtype)
            cells = np.zeros((0, num_dofs_per_cell), dtype=np.int64)
    return ReadMeshData(
        cells=cells,
        cell_type=dolfinx.mesh.to_string(cell_type),
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
    if comm.rank == 0:
        with netCDF4.Dataset(filename, "r") as infile:
            # use page 171 of manual to extract data
            num_blocks = infile.dimensions["num_el_blk"].size

            # Extract all connectivity blocks
            all_connectivity_variables = [
                infile.variables.get(f"connect{i + 1}") for i in range(num_blocks)
            ]

            # Compute max topological dimension in mesh and find the correct
            max_tdim = _compute_tdim(max(all_connectivity_variables, key=_compute_tdim))
            if name == "cell":
                search_dim = max_tdim
            elif name == "facet":
                search_dim = max_tdim - 1
            else:
                raise ValueError(f"Only name 'cell' or 'facet' is supported, got '{name}'")

            # Extract only the connectivity blocks that we need
            entity_blocks = list(
                filter(lambda el: _compute_tdim(el) == search_dim, all_connectivity_variables)
            )

            if len(entity_blocks) > 0:
                # Extract markers directly from entity-blocks
                connectivity_arrays = []
                cell_types = []
                num_entities = []
                entity_block_index = []
                for entity_block in entity_blocks:
                    connectivity_arrays.append(entity_block[:] - 1)
                    num_entities.append(entity_block.shape[0])
                    cell_types.append(_get_cell_type(entity_block))
                    entity_block_index.append(int(entity_block.name.removeprefix("connect")) - 1)
                for cell in cell_types:
                    assert cell_types[0] == cell, "Mixed cell types not supported"
                cell_type = cell_types[0]

                marked_entities = np.vstack(connectivity_arrays)
                entity_values = np.zeros(marked_entities.shape[0], dtype=np.int64)
                if "eb_prop1" in infile.variables.keys():
                    block_values = infile.variables["eb_prop1"][:]

                    # First check if entities are in eb_prop1
                    insert_offset = np.zeros(len(num_entities) + 1, dtype=np.int64)
                    insert_offset[1:] = np.cumsum(num_entities)
                    for i, index in enumerate(entity_block_index):
                        entity_values[insert_offset[i] : insert_offset[i + 1]] = block_values[index]
                else:
                    marked_entities = np.zeros((0, marked_entities.shape[1]), dtype=np.int64)
                    entity_values = np.zeros(0, dtype=np.int64)
            elif name == "facet" and "ss_prop1" in infile.variables.keys():
                # If we haven't found the cell type as a block, we should be extracting facets
                # (from side-sets), then we need the parent cell
                entity_blocks = list(
                    filter(lambda el: _compute_tdim(el) == max_tdim, all_connectivity_variables)
                )
                cell_types = []
                for entity_block in entity_blocks:
                    cell_types.append(_get_cell_type(entity_block))
                for cell in cell_types:
                    assert cell_types[0] == cell, "Mixed cell types not supported"
                cell_type = cell_types[0]

                local_facet_index = _side_set_to_vertex_map[dolfinx.mesh.to_string(cell_type)]
                if "num_side_sets" not in infile.dimensions:
                    num_vertices_per_facet = len(local_facet_index[0])
                    marked_entities = np.zeros((0, num_vertices_per_facet), dtype=np.int64)
                    entity_values = np.zeros(0, dtype=np.int64)
                else:
                    # Extract facet values
                    local_facet_index = _side_set_to_vertex_map[dolfinx.mesh.to_string(cell_type)]
                    num_facet_sets = infile.dimensions["num_side_sets"].size
                    values = infile.variables.get("ss_prop1")
                    # Extract all cell blocks to get correct look-up
                    connectivity_arrays = []
                    for entity_block in entity_blocks:
                        connectivity_arrays.append(entity_block[:] - 1)
                    connectivity_array = np.vstack(connectivity_arrays)

                    # Loop through all side sets to extract the correct connectivity
                    facet_indices = []
                    facet_values = []
                    for i in range(num_facet_sets):
                        value = values[i].reshape(-1)
                        elements = infile.variables[f"elem_ss{i + 1}"]
                        local_facets = infile.variables[f"side_ss{i + 1}"]
                        for element, index in zip(elements, local_facets):
                            facet_indices.append(
                                connectivity_array[element - 1, local_facet_index[index - 1]]
                            )
                            facet_values.append(value.data.tolist())
                        marked_entities = np.vstack(facet_indices)
                        entity_values = np.array(facet_values, dtype=np.int64).flatten()
            else:
                # If we cannot find any information about the blocks we send nothing
                marked_entities = np.zeros((0, 0), dtype=np.int64)
                entity_values = np.zeros(0, dtype=np.int64)
        # Broadcast information read by this process to other processes
        (dim, _, _) = comm.bcast(
            (search_dim, marked_entities.shape[1], np.dtype(entity_values.dtype).name),
            root=0,
        )

    else:
        # Other process gets info from process that read the file
        dim, num_dofs_per_cell, vtype = comm.bcast((None, None, None), root=0)
        marked_entities = np.zeros((0, num_dofs_per_cell), dtype=np.int64)
        entity_values = np.zeros(0, dtype=vtype)
    return MeshTagsData(name=name, values=entity_values, indices=marked_entities, dim=dim)


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
    infile = netCDF4.Dataset(filename)
    timestep = 0  # FIXME - need to find the correct timestep based on time argument
    if name not in infile.variables:
        raise ValueError(
            f"Point data with name {name} not found in file.",
            "Available variables: {list(infile.variables.keys())}",
        )
    dataset = infile.variables[name][:][timestep].data
    if len(dataset.shape) == 1:
        num_components = 1
        dataset = dataset.reshape(-1, num_components)
    else:
        num_components = dataset.shape[1]

    local_start_range = 0

    return dataset, int(local_start_range)


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


def getNames(model, key):
    # name of the element variables
    name_var = []
    for vname in np.ma.getdata(model.variables[key][:]).astype("U8"):
        name_var.append("".join(vname))
    return name_var
