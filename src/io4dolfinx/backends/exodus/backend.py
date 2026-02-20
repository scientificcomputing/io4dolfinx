"""
Exodus interface to io4dolfinx.
The Exodus2 format is described in:
https://src.fedoraproject.org/repo/pkgs/exodusii/922137.pdf/a45d67f4a1a8762bcf66af2ec6eb35f9/922137.pdf
Further documentation from CUBIT on node numbering can be found at:
https://coreform.com/cubit_help/appendix/element_numbering.htm

SPDX License identifier: MIT

Copyright: Jørgen S. Dokken, Henrik N.T. Finsberg, Remi Delaporte-Mathurin,
           and Simula Research Laboratory
"""

from pathlib import Path
from typing import Any, Literal, cast

from mpi4py import MPI

import basix.ufl
import dolfinx
import netCDF4
import numpy as np
import numpy.typing as npt

from ...structures import ArrayData, FunctionData, MeshData, MeshTagsData, ReadMeshData
from ...utils import check_file_exists
from .. import FileMode, ReadMode

_interval_to_vertex_map = {0: [0, 1]}
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
    "interval": _interval_to_vertex_map,
    "quadrilateral": _quad_to_vertex_map,
    "triangle": _triangle_to_vertex_map,
    "tetrahedron": _tetra_facet_to_vertex_map,
    "hexahedron": _hex_to_vertex_map,
}


_exodus_to_string = {
    "EDGE2": ("interval", 1),
    "TRI3": ("triangle", 1),
    "QUAD": ("quadrilateral", 1),
    "QUAD4": ("quadrilateral", 1),
    "TETRA": ("tetrahedron", 1),
    "HEX": ("hexahedron", 1),
    "HEX8": ("hexahedron", 1),
    "BEAM2": ("interval", 1),
    "HEX27": ("hexahedron", 2),
}

read_mode = ReadMode.serial


def _get_cell_type(connectivity: netCDF4.Variable) -> tuple[dolfinx.mesh.CellType, int]:
    cell_type, degree = _exodus_to_string[connectivity.elem_type]
    return dolfinx.mesh.to_type(cell_type), degree


def _compute_tdim(connectivity: netCDF4.Variable) -> int:
    d_ct, _ = _get_cell_type(connectivity)
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
        backend_args: Arguments to backend
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
        backend_args: Arguments to backend
    Returns:
        The attributes
    """
    raise NotImplementedError("The Exodus backend cannot read attributes.")


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
    raise NotImplementedError("The EXODUS backend cannot make checkpoints.")


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
    raise NotImplementedError("The EXODUS backend cannot read HDF5 arrays")


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
        comm: MPI communicator
        function_name: Name of the function to read time-stamps for
        backend_args: Arguments for backend, for instance file type.

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
        filename: Path to file to write to.
        mesh: Internal data structure for the mesh data to save to file
        comm: MPI communicator used in storage
        backend_args: Arguments to backend
        mode: Mode to use (write or append)
        time: Time stamp
    """
    raise NotImplementedError("The Exodus backend cannot write meshes.")


def _read_mesh_geometry(infile: netCDF4.Dataset) -> tuple[int, npt.NDArray[np.floating]]:
    # use page 171 of manual to extract data
    num_nodes = infile.dimensions["num_nodes"].size
    gdim = infile.dimensions["num_dim"].size

    # Get coordinates of mesh
    coord_var = infile.variables.get("coord")
    if coord_var is None:
        coordinates = np.zeros((num_nodes, gdim), dtype=np.float64)
        for i, coord in enumerate(["x", "y", "z"]):
            coord_i = infile.variables.get(f"coord{coord}")
            if coord_i is not None:
                coordinates[: coord_i.size, i] = coord_i[:]
    else:
        coordinates = np.asarray(coord_var)
    return gdim, coordinates


def _get_entity_blocks(
    infile: netCDF4.Dataset, search_type: Literal["cell", "facet"]
) -> tuple[int, list[netCDF4.Variable]]:
    # use page 171 of manual to extract data
    num_blocks = infile.dimensions["num_el_blk"].size

    # Get element connectivity
    all_connectivity_variables = [infile.variables[f"connect{i + 1}"] for i in range(num_blocks)]

    # Compute max topological dimension in mesh and find the correct
    max_tdim = _compute_tdim(max(all_connectivity_variables, key=_compute_tdim))

    # Extract only the connectivity blocks that we need
    if search_type == "cell":
        search_dim = max_tdim
    elif search_type == "facet":
        search_dim = max_tdim - 1
    else:
        raise RuntimeError(f"Unknown entity type: {search_type}")
    return search_dim, list(
        filter(lambda el: _compute_tdim(el) == search_dim, all_connectivity_variables)
    )


def _extract_connectivity_data(
    entity_blocks: list[netCDF4.Variable],
) -> tuple[list[npt.NDArray[np.int64]], tuple[dolfinx.mesh.CellType, int], list[int]]:
    connectivity_arrays = []
    cell_types = []
    entity_block_index = []
    for entity_block in entity_blocks:
        connectivity_arrays.append(entity_block[:] - 1)
        cell_types.append(_get_cell_type(entity_block))
        entity_block_index.append(int(entity_block.name.removeprefix("connect")) - 1)
    for cell in cell_types:
        assert cell_types[0] == cell, "Mixed cell types not supported"
    cell_type = cell_types[0]
    return connectivity_arrays, cell_type, entity_block_index


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
        backend_args: Arguments to backend
    Returns:
        The mesh topology, geometry, UFL domain and partition function
    """
    check_file_exists(filename)
    with netCDF4.Dataset(filename, "r") as infile:
        if comm.rank == 0:
            gdim, coordinates = _read_mesh_geometry(infile)

            _tdim, entity_blocks = _get_entity_blocks(infile, "cell")
            if len(entity_blocks) > 0:
                # Extract markers directly from entity-blocks
                connectivity_arrays, (cell_type, degree), _entity_block_index = (
                    _extract_connectivity_data(entity_blocks)
                )

                cells = np.vstack(connectivity_arrays)
                if isinstance(cells, np.ma.MaskedArray):
                    cells = cells.filled()
            else:
                raise ValueError(f"No blocks found in {filename}")

            if degree == 1:
                perm = dolfinx.cpp.io.perm_vtk(cell_type, cells.shape[1])
            elif cell_type == dolfinx.mesh.CellType.hexahedron and degree == 2:
                # Ordering from Fig 4.14 of: https://sandialabs.github.io/seacas-docs/exodusII-new.pdf
                dolfinx_to_exodus = np.array(
                    [
                        0,
                        1,
                        5,
                        4,
                        2,
                        3,
                        7,
                        6,
                        8,
                        12,
                        16,
                        10,
                        9,
                        11,
                        18,
                        17,
                        13,
                        15,
                        19,
                        14,
                        26,
                        21,
                        24,
                        22,
                        23,
                        20,
                        25,
                    ]
                )
                perm = np.argsort(dolfinx_to_exodus)
            else:
                raise NotImplementedError(
                    "Reading Exodus2 mesh with {cell_type} of order {degree} is not supported."
                )

            cells = cells[:, perm]
            cell_type, gdim, xtype, degree, num_dofs_per_cell = comm.bcast(
                (cell_type, gdim, np.dtype(coordinates.dtype).name, degree, cells.shape[1]), root=0
            )

        else:
            cell_type, gdim, xtype, degree, num_dofs_per_cell = comm.bcast(
                (None, None, None, None), root=0
            )
            coordinates = np.zeros((0, gdim), dtype=xtype)
            cells = np.zeros((0, num_dofs_per_cell), dtype=np.int64)
    return ReadMeshData(
        cells=cells,
        cell_type=dolfinx.mesh.to_string(cell_type),
        x=coordinates,
        lvar=int(basix.LagrangeVariant.equispaced),
        degree=degree,
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
            # Compute max topological dimension in mesh and find the correct
            if name == "cell" or name == "facet":
                search_dim, entity_blocks = _get_entity_blocks(
                    infile, cast(Literal["cell", "facet"], name)
                )
            else:
                raise RuntimeError("Expected name='cell' or 'facet' got {name}")

            if len(entity_blocks) > 0:
                # Extract markers directly from entity-blocks
                connectivity_arrays, (cell_type, degree), entity_block_index = (
                    _extract_connectivity_data(entity_blocks)
                )
                marked_entities = np.vstack(connectivity_arrays)
                entity_values = np.zeros(marked_entities.shape[0], dtype=np.int64)
                if "eb_prop1" in infile.variables.keys():
                    block_values = infile.variables["eb_prop1"][:]

                    # First check if entities are in eb_prop1
                    insert_offset = np.zeros(len(connectivity_arrays) + 1, dtype=np.int64)
                    insert_offset[1:] = np.cumsum([c_arr.shape[0] for c_arr in connectivity_arrays])
                    for i, index in enumerate(entity_block_index):
                        entity_values[insert_offset[i] : insert_offset[i + 1]] = block_values[index]
                else:
                    num_dofs_per_cell = basix.ufl.element(
                        "Lagrange", dolfinx.mesh.to_string(cell_type), degree
                    ).dim
                    assert num_dofs_per_cell == marked_entities.shape[1]
                    marked_entities = np.zeros((0, marked_entities.shape[1]), dtype=np.int64)
                    entity_values = np.zeros(0, dtype=np.int64)
            elif name == "facet" and "ss_prop1" in infile.variables.keys():
                # If we haven't found the cell type as a block, we should be extracting facets
                # (from side-sets), then we need the parent cell
                tdim, entity_blocks = _get_entity_blocks(infile, "cell")
                cell_types = []
                for entity_block in entity_blocks:
                    cell_types.append(_get_cell_type(entity_block))
                for cell in cell_types:
                    assert cell_types[0] == cell, "Mixed cell types not supported"
                cell_type, degree = cell_types[0]
                local_facet_index = _side_set_to_vertex_map[dolfinx.mesh.to_string(cell_type)]
                if "num_side_sets" not in infile.dimensions:
                    facet_type = dolfinx.cpp.mesh.cell_entity_type(cell_type, tdim - 1, 0)
                    num_dofs_per_cell = basix.ufl.element(
                        "Lagrange", dolfinx.mesh.to_string(facet_type), degree
                    ).dim
                    marked_entities = np.zeros((0, num_dofs_per_cell), dtype=np.int64)
                    entity_values = np.zeros(0, dtype=np.int64)
                else:
                    # Extract facet values
                    local_facet_index = _side_set_to_vertex_map[dolfinx.mesh.to_string(cell_type)]
                    num_facet_sets = infile.dimensions["num_side_sets"].size
                    values = infile.variables["ss_prop1"]
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
                # If we found no blocks (for instance for facets, we search through the cells)
                tdim, entity_blocks = _get_entity_blocks(infile, "cell")
                search_dim = tdim - 1 if name == "facet" else tdim
                cell_type, degree = _get_cell_type(entity_blocks[0])
                facet_type = dolfinx.cpp.mesh.cell_entity_type(cell_type, search_dim, 0)
                num_dofs_per_cell = basix.ufl.element(
                    "Lagrange", dolfinx.mesh.to_string(facet_type), degree
                ).dim
                # If we cannot find any information about the blocks we send nothing
                marked_entities = np.zeros((0, num_dofs_per_cell), dtype=np.int64)
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
        filename: Path to file to write to
        comm: MPI communicator used in storage
        u: Internal data structure for the function data to save to file
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
    num_components = 1  # Default assumption, overriden by data read in having multiple components
    if comm.rank == 0:
        with netCDF4.Dataset(filename, "r") as infile:
            raw_names = infile.variables["name_nod_var"][:].data
            node_names = netCDF4.chartostring(raw_names)
            if name not in node_names:
                raise ValueError(
                    f"Point data with name {name} not found in file.",
                    f"Available variables: {node_names}",
                )
            index = np.flatnonzero(name == node_names)[0] + 1

            temporal_dataset = infile.variables[f"vals_nod_var{index}"]
            time_steps = infile.variables["time_whole"][:].data
            if time is None:
                time_idx = time_steps[0]
            else:
                time_indices = np.flatnonzero(np.isclose(time_steps, time))
                if len(time_indices) == 0:
                    raise ValueError(
                        f"Could not find {name}(t={time}), available time steps are {time_steps}"
                    )
                time_idx = time_indices[0]

            dataset = temporal_dataset[time_idx]
            if len(dataset.shape) == 1:
                dataset = dataset.reshape(-1, num_components)
            else:
                num_components = dataset.shape[1]
    # Broadcast num components to all other ranks
    num_components = comm.bcast(num_components, root=0)
    # Zero data on all other processes
    if comm.rank != 0:
        dataset = np.zeros((0, num_components), dtype=np.float64)
    return dataset, 0


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
    num_components = 1  # Default assumption, overriden by data read in having multiple components
    if comm.rank == 0:
        with netCDF4.Dataset(filename, "r") as infile:
            raw_names = infile.variables["name_elem_var"][:].data
            num_blocks = infile.dimensions["num_el_blk"].size

            node_names = netCDF4.chartostring(raw_names)
            if name not in node_names:
                raise ValueError(
                    f"Cell data with name {name} not found in file.",
                    f"Available variables: {node_names}",
                )
            index = np.flatnonzero(name == node_names)[0] + 1

            entity_blocks = [
                infile.variables[f"vals_elem_var{index}eb{i + 1}"] for i in range(num_blocks)
            ]
            time_steps = infile.variables["time_whole"][:].data
            if time is None:
                time_idx = time_steps[0]
            else:
                time_indices = np.flatnonzero(np.isclose(time_steps, time))
                if len(time_indices) == 0:
                    raise ValueError(
                        f"Could not find {name}(t={time}), available time steps are {time_steps}"
                    )
                time_idx = time_indices[0]

            if len(entity_blocks) > 0:
                datasets = []
                for entity_block in entity_blocks:
                    datasets.append(entity_block[time_idx])

            dataset = np.hstack(datasets)

            if len(dataset.shape) == 1:
                dataset = dataset.reshape(-1, num_components)
            else:
                num_components = dataset.shape[1]
    # Broadcast num components to all other ranks
    num_components = comm.bcast(num_components, root=0)

    # Zero data on all other processes
    if comm.rank != 0:
        dataset = np.zeros((0, num_components), dtype=np.float64)
    _time = float(time) if time is not None else None

    topology = read_mesh_data(filename, comm, _time, False, backend_args=None).cells
    return topology, dataset


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
    with netCDF4.Dataset(filename, "r") as infile:
        function_names: list[str] = []
        for key in ["name_elem_var", "name_nod_var"]:
            raw_names = infile.variables[key][:].data
            decoded_names = netCDF4.chartostring(raw_names)
            function_names.extend(decoded_names)
    return function_names


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
