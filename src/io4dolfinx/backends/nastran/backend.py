from pathlib import Path
from typing import Any

from mpi4py import MPI

import basix
import dolfinx
import gmsh
import numpy as np
import numpy.typing as npt
from dolfinx.io.gmsh import (
    _gmsh_to_cells,
    cell_perm_array,
    extract_geometry,
    extract_topology_and_markers,
)

from ...structures import ArrayData, FunctionData, MeshData, MeshTagsData, ReadMeshData
from ...utils import check_file_exists
from .. import FileMode, ReadMode

read_mode = ReadMode.serial


def get_default_backend_args(arguments: dict[str, Any] | None) -> dict[str, Any]:
    args = arguments or {}
    return args


def _promote_entities_to_physical_groups(model) -> None:
    """Promote gmsh elementary entities to physical groups.

    Gmsh reads PSOLID/PSHELL as elementary entities only.
    extract_topology_and_markers only sees physical groups,
    so we promote every entity, reusing the entity
    tag as the physical group tag (preserving Nastran PIDs).
    """
    for dim, tag in model.getEntities():
        model.addPhysicalGroup(dim=dim, tags=[tag], tag=tag)


def read_mesh_data(
    filename: Path | str,
    comm: MPI.Intracomm,
    time: str | float | None = None,
    read_from_partition: bool = False,
    backend_args: dict[str, Any] | None = None,
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

    if read_from_partition:
        raise RuntimeError("Cannot read partition data with nastran")
    check_file_exists(filename)

    if comm.rank == 0:
        gmsh.initialize()
        try:
            gmsh.open(str(filename))
            _promote_entities_to_physical_groups(gmsh.model)
            topologies, _ = extract_topology_and_markers(gmsh.model)
            if not topologies:
                raise ValueError(f"No elements found in {filename}.")
            x = extract_geometry(gmsh.model)

            elements = []
            for gmsh_type in topologies:
                _, dim, _, n_nodes, *_ = gmsh.model.mesh.getElementProperties(gmsh_type)
                elements.append((dim, n_nodes, gmsh_type))

            dims = [e[0] for e in elements]
            if len(dims) != len(set(dims)):
                raise ValueError(
                    f"Multiple element types share a topological dimension in {filename}."
                )
            _, num_nodes, gmsh_type = max(elements, key=lambda e: e[0])
            cells = topologies[gmsh_type]["topology"].astype(np.int64, copy=False)
        finally:
            gmsh.finalize()

        cell_type, degree = _gmsh_to_cells[gmsh_type]
        comm.bcast((cell_type, degree, num_nodes), root=0)
    else:
        cell_type, degree, num_nodes = comm.bcast(None, root=0)
        cells = np.empty((0, num_nodes), dtype=np.int64)
        x = np.empty((0, 3), dtype=np.float64)

    perm = cell_perm_array(dolfinx.mesh.to_type(cell_type), num_nodes)

    return ReadMeshData(
        cells=cells[:, perm],
        cell_type=cell_type,
        x=x,
        lvar=int(basix.LagrangeVariant.equispaced),
        degree=degree,
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
    raise NotImplementedError("The nastran backend cannot read point data.")


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
    raise NotImplementedError("The nastran backend does not support reading cell data.")


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
    raise NotImplementedError("The nastran backend cannot write attributes.")


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
    raise NotImplementedError("The nastran backend cannot read attributes.")


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
    raise NotImplementedError("The nastran backend cannot read timestamps.")


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

    raise NotImplementedError("The nastran backend cannot write function_names.")


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
    raise NotImplementedError("The nastran backend cannot write meshes.")


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
    raise NotImplementedError("The nastran backend cannot write meshtags.")


_PROPERTY_TO_DIM = {
    "PSOLID": 3,
    "PSHELL": 2,
}


def read_meshtags_data(
    filename: Path | str,
    comm: MPI.Intracomm,
    name: str,
    backend_args: dict[str, Any] | None = None,
) -> MeshTagsData:
    """Read mesh tags from file.

    Nastran PIDs are used as integer tag values. The ``name`` argument
    selects which property card's elements to return.

    Args:
        filename: Path to the Nastran file.
        comm: MPI communicator.
        name: Nastran property card. Supported values:
            "PSOLID" - 3D solid elements.
            "PSHELL" - 2D shell elements (often used for BCs).

        backend_args: Reserved for future use.

    Returns:
        Internal data structure for the mesh tags read from file
    """
    check_file_exists(filename)
    backend_args = get_default_backend_args(backend_args)

    if name not in _PROPERTY_TO_DIM:
        raise ValueError(
            f"Unsupported meshtag name {name!r}. Expected one of: {sorted(_PROPERTY_TO_DIM)}."
        )
    target_dim = _PROPERTY_TO_DIM[name]

    if comm.rank == 0:
        gmsh.initialize()
        try:
            gmsh.open(str(filename))
            _promote_entities_to_physical_groups(gmsh.model)
            topologies, _ = extract_topology_and_markers(gmsh.model)
            if not topologies:
                raise ValueError(f"No elements found in {filename}.")

            elements = []
            for gmsh_type in topologies:
                _, dim, _, n_nodes, *_ = gmsh.model.mesh.getElementProperties(gmsh_type)
                elements.append((dim, n_nodes, gmsh_type))

            target_elements = [e for e in elements if e[0] == target_dim]
            if not target_elements:
                raise ValueError(
                    f"No {name} elements (dimension {target_dim}) found in {filename}."
                )
            if len(target_elements) > 1:
                raise RuntimeError(
                    f"Multiple element types share dimension {target_dim} in {filename}."
                )
            _, num_nodes, gmsh_type = target_elements[0]
            mesh_entities = topologies[gmsh_type]["topology"].astype(np.int64, copy=False)
            tag_values = topologies[gmsh_type]["cell_data"].astype(np.int32, copy=False)
        finally:
            gmsh.finalize()

        cell_type, _ = _gmsh_to_cells[gmsh_type]
        comm.bcast((cell_type, num_nodes), root=0)
    else:
        cell_type, num_nodes = comm.bcast(None, root=0)
        mesh_entities = np.empty((0, num_nodes), dtype=np.int64)
        tag_values = np.empty(0, dtype=np.int32)

    perm = cell_perm_array(dolfinx.mesh.to_type(cell_type), num_nodes)

    return MeshTagsData(
        name=name,
        values=tag_values,
        indices=mesh_entities[:, perm],
        dim=target_dim,
    )


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
    raise NotImplementedError("The nastran backend cannot make checkpoints.")


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
    raise NotImplementedError("The nastran backend cannot make checkpoints.")


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
    raise NotImplementedError("The nastran backend cannot make checkpoints.")


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
    raise NotImplementedError("The nastran backend cannot make checkpoints.")


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
    raise NotImplementedError("The nastran backend cannot read HDF5 arrays")


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
        time: Time stamp
        mode: Append or write
        backend_args: The backend arguments
    """
    raise NotImplementedError("The nastran backend does not support writing point data")
