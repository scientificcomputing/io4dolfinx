"""
Nastran interface to io4dolfinx.

Reads the mesh and mesh tags of Nastran bulk data files (``.bdf``, ``.nas``) through the
`Gmsh <https://gmsh.info>`_ Python API, reusing :mod:`dolfinx.io.gmsh` for the geometry,
the topology and the node ordering. All file access happens on rank 0.

- The cells are the elements of the highest topological dimension. Meshes with several
  cell types, such as hexahedra, pyramids and tetrahedra, are read as mixed meshes.
- Mesh tags are the property IDs (PID) of the elements, see :func:`read_meshtags_data`.
- Node ``i`` is the node with the ``i``-th smallest GRID ID. The GRID IDs can be read as
  point data and as vertex tags, both called ``"GRID"``, on meshes with one cell type.

SPDX License identifier: MIT
"""

from pathlib import Path
from typing import Any

from mpi4py import MPI

import basix
import dolfinx
import numpy as np
import numpy.typing as npt
from dolfinx.io.gmsh import extract_geometry

try:
    import gmsh
except ImportError:
    raise ModuleNotFoundError("The nastran-backend requires gmsh")

from ...structures import (
    ArrayData,
    FunctionData,
    MeshData,
    MeshTagsData,
    ReadMeshData,
    ReadMixedMeshData,
)
from ...utils import check_file_exists
from .. import FileMode, ReadMode
from ..gmsh.backend import _run_on_root
from . import helpers

read_mode = ReadMode.serial


def get_default_backend_args(arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Get default backend arguments given a set of input arguments.

    Args:
        arguments: Input backend arguments

    Returns:
        The arguments, with defaults for:
            - ``"dtype"``: Floating point type of the mesh coordinates, ``np.float64``
    """
    args: dict[str, Any] = {"dtype": np.float64}
    args.update(arguments or {})
    return args


def read_mesh_data(
    filename: Path | str,
    comm: MPI.Comm,
    time: str | float | None = None,
    read_from_partition: bool = False,
    backend_args: dict[str, Any] | None = None,
) -> ReadMeshData | ReadMixedMeshData:
    """Read mesh data from a Nastran file.

    The cells are the elements of the highest topological dimension. If they are of
    several types, e.g. hexahedra, pyramids and tetrahedra, the mesh is read as a mixed
    mesh. All data is read on rank 0.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        time: Time stamp associated with the mesh to read
        read_from_partition: Whether to read partition information
        backend_args: Arguments to backend, see :func:`get_default_backend_args`

    Returns:
        Internal data structure for the mesh data read from file
    """
    if read_from_partition:
        raise NotImplementedError("The nastran backend cannot read partitioning data.")
    check_file_exists(filename)
    backend_args = get_default_backend_args(backend_args)

    def read(
        preexisting_views: list[int],
    ) -> tuple[
        tuple[npt.NDArray[np.float64], list[npt.NDArray[np.int64]]],
        tuple[list[str], int, list[int]],
    ]:
        elements, _ = helpers.read_elements(filename)
        blocks = elements[max(elements)]
        degrees = {block.degree for block in blocks}
        if len(degrees) > 1:
            raise NotImplementedError(f"Cells of several degrees {degrees} in {filename}.")
        x = extract_geometry(gmsh.model)
        cells = [block.nodes for block in blocks]
        cell_types = [block.cell_type for block in blocks]
        return (x, cells), (cell_types, degrees.pop(), [c.shape[1] for c in cells])

    local, (cell_types, degree, num_nodes) = _run_on_root(comm, read)
    if local is None:
        x = np.zeros((0, 3), dtype=np.float64)
        cells = [np.zeros((0, n), dtype=np.int64) for n in num_nodes]
    else:
        x, cells = local
    x = x.astype(backend_args["dtype"], copy=False)
    lvar = int(basix.LagrangeVariant.equispaced)
    if len(cell_types) == 1:
        return ReadMeshData(cells=cells[0], cell_type=cell_types[0], x=x, lvar=lvar, degree=degree)
    return ReadMixedMeshData(cells=cells, cell_types=cell_types, x=x, lvar=lvar, degree=degree)


def read_point_data(
    filename: Path | str,
    name: str,
    comm: MPI.Comm,
    time: float | str | None,
    backend_args: dict[str, Any] | None,
) -> tuple[np.ndarray, int]:
    """Read the GRID IDs of the nodes of a mesh with one cell type.

    Note:
        The function holding the GRID IDs has the precision of the mesh. In single
        precision, IDs above :math:`2^{24}` are not exact; read them as vertex tags with
        :func:`read_meshtags_data` instead.

    Args:
        filename: Path to file
        name: Name of point data, ``"GRID"``
        comm: Communicator to launch IO on.
        time: Ignored, as a Nastran file holds no time series
        backend_args: The backend arguments

    Returns:
        Tuple containing:
            - The GRID ID of each node, of shape ``(num_nodes, 1)``, on rank 0
            - The global start of the data on this process, ``0``

    Raises:
        KeyError: If ``name`` is not ``"GRID"``.
        NotImplementedError: If the mesh has several cell types.
    """
    check_file_exists(filename)
    if name != helpers.GRID:
        raise KeyError(f"No point data '{name}' in {filename}. Available: ['{helpers.GRID}']")

    def read(preexisting_views: list[int]) -> tuple[npt.NDArray[np.int64], None]:
        elements, grid_ids = helpers.read_elements(filename)
        helpers.get_single_cell_type(elements, "Reading GRID IDs as point data", filename)
        return grid_ids, None

    grid_ids, _ = _run_on_root(comm, read)
    if grid_ids is None:
        grid_ids = np.zeros(0, dtype=np.int64)
    return grid_ids.reshape(-1, 1).astype(np.float64), 0


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
        name: Name of point data
        comm: Communicator to launch IO on.
        time: The time stamp
        backend_args: The backend arguments
    Returns:
        A tuple (topology, dofs) where topology contains the
        vertex indices of the cells, dofs the degrees of
        freedom within that cell.
    """
    raise NotImplementedError("The nastran backend does not support reading cell data.")


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
    raise NotImplementedError("The nastran backend cannot write attributes.")


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
    raise NotImplementedError("The nastran backend cannot read attributes.")


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
    raise NotImplementedError("The nastran backend cannot read timestamps.")


def read_function_names(
    filename: Path | str, comm: MPI.Comm, backend_args: dict[str, Any] | None
) -> list[str]:
    """Read all function names from a file.

    Args:
        filename: Path to file
        comm: MPI communicator to launch IO on.
        backend_args: Arguments to backend

    Returns:
        A list of function names.
    """

    return [helpers.GRID]


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
    raise NotImplementedError("The nastran backend cannot write meshes.")


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
    raise NotImplementedError("The nastran backend cannot write meshtags.")


def read_meshtags_data(
    filename: Path | str,
    comm: MPI.Comm,
    name: str,
    backend_args: dict[str, Any] | None = None,
) -> MeshTagsData:
    """Read mesh tags from a Nastran file.

    Args:
        filename: Path to the Nastran file
        comm: MPI communicator
        name: One of:

            - ``"PSOLID"``: the 3D solid elements, tagged with their property ID (PID)
            - ``"PSHELL"``: the 2D shell elements (often used for boundary conditions),
              tagged with their PID
            - ``"GRID"``: the vertices, tagged with their GRID ID. Only supported on
              meshes with one cell type.

        backend_args: Arguments to backend, unused

    Returns:
        Internal data structure for the mesh tags read from file, on rank 0

    Raises:
        NotImplementedError: If the tagged entities are of several types, or for ``"GRID"``
            on a mesh with several cell types.
    """
    check_file_exists(filename)
    if name != helpers.GRID and name not in helpers.PROPERTY_TO_DIM:
        names = [*sorted(helpers.PROPERTY_TO_DIM), helpers.GRID]
        raise ValueError(f"Unsupported meshtag name {name!r}. Expected one of: {names}.")
    dim = 0 if name == helpers.GRID else helpers.PROPERTY_TO_DIM[name]

    def read(
        preexisting_views: list[int],
    ) -> tuple[tuple[npt.NDArray[np.int64], npt.NDArray[np.int32]], int]:
        elements, grid_ids = helpers.read_elements(filename)
        if name == helpers.GRID:
            # Tag each vertex with its GRID ID, e.g. for point measures (ufl.dP). Only
            # vertex nodes are tagged: the edge, face and interior nodes of higher-order
            # cells are not mesh vertices. Each vertex is given by its node index (one
            # node per entity), which read_meshtags maps to the mesh vertex. Node i has
            # the i-th smallest GRID ID, so grid_ids[i] is its tag value, which must fit
            # in the int32 values of a MeshTags.
            cells = helpers.get_single_cell_type(
                elements, "Reading GRID IDs as vertex tags", filename
            )
            vertices = helpers.get_vertex_nodes(cells)
            if len(grid_ids) > 0 and grid_ids[-1] > np.iinfo(np.int32).max:
                raise ValueError(f"GRID IDs in {filename} do not fit in mesh tag values (int32).")
            return (vertices.reshape(-1, 1), grid_ids[vertices].astype(np.int32)), 1

        # Check that the file has elements of the dimension of the property card, all of
        # one type, as a DOLFINx MeshTags holds entities of a single type
        blocks = elements.get(dim, [])
        if len(blocks) == 0:
            raise ValueError(f"No {name} elements (dimension {dim}) found in {filename}.")
        if len(blocks) > 1:
            types = [block.cell_type for block in blocks]
            raise NotImplementedError(
                f"{name} elements of several types {types} in {filename}; mesh tags of "
                "several entity types are not supported."
            )
        return (blocks[0].nodes, blocks[0].pids), blocks[0].nodes.shape[1]

    local, num_nodes = _run_on_root(comm, read)
    if local is None:
        local = (np.zeros((0, num_nodes), dtype=np.int64), np.zeros(0, dtype=np.int32))
    return MeshTagsData(name=name, values=local[1], indices=local[0], dim=dim)


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
    raise NotImplementedError("The nastran backend cannot make checkpoints.")


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
    raise NotImplementedError("The nastran backend cannot make checkpoints.")


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
    raise NotImplementedError("The nastran backend cannot make checkpoints.")


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
    raise NotImplementedError("The nastran backend cannot make checkpoints.")


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
    raise NotImplementedError("The nastran backend cannot read legacy DOLFIN meshes.")


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
    raise NotImplementedError("The nastran backend cannot make checkpoints.")


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
    raise NotImplementedError("The nastran backend cannot read HDF5 arrays")


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
    raise NotImplementedError("The nastran backend does not support writing point data")
