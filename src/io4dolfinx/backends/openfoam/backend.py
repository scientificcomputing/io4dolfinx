"""
OpenFOAM interface to io4dolfinx.

Reads the mesh, fields, boundary patches and zones of an OpenFOAM case through the
VTK OpenFOAM reader of `pyvista <https://pyvista.org>`_. The file is the ``.foam`` file
in the case directory (create it with ``touch case.foam``). All file access happens on
rank 0.

- The cells are the cells of the internal mesh. Hexahedra, prisms, pyramids and
  tetrahedra are supported; meshes with several of them are read as mixed meshes.
  Polyhedra are not supported, as DOLFINx has no polyhedral cells.
- Cell-centred fields (``vol*Field``) are read with :func:`read_cell_data`, and point
  fields (``point*Field``) with :func:`read_point_data`. Fields are not interpolated
  between the two.
- Boundary patches and cell, face and point zones are read from the ``polyMesh`` files
  by their labels, see :func:`read_meshtags_data`.
- The regions of a multi-region case are merged into one mesh, joined where their
  points coincide exactly, unless one region is selected with the ``region`` backend
  argument.

VTK reads points and fields in single precision. They are returned in the ``dtype``
backend argument, by default :data:`dolfinx.default_real_type`.

SPDX License identifier: MIT

Copyright: Jørgen S. Dokken and Simula Research Laboratory
"""

from pathlib import Path
from typing import Any, Callable, TypeVar

from mpi4py import MPI

import basix
import dolfinx
import numpy as np
import numpy.typing as npt

try:
    import pyvista
except ImportError:
    raise ModuleNotFoundError("The openfoam-backend requires pyvista")

from ...comm_helpers import run_on_root
from ...structures import (
    ArrayData,
    FunctionData,
    MeshData,
    MeshTagsData,
    ReadMeshData,
    ReadMixedMeshData,
)
from ...utils import check_file_exists, codim_names
from .. import FileMode, ReadMode
from . import helpers

read_mode = ReadMode.serial

Local = TypeVar("Local")
Shared = TypeVar("Shared")


def _on_root(
    comm: MPI.Comm, func: Callable[[], tuple[Local, Shared]]
) -> tuple[Local | None, Shared]:
    """Run ``func`` on rank 0, raising its errors on all ranks.

    Returns:
        The first output of ``func`` on rank 0 (``None`` elsewhere), and its second
        output on all ranks
    """
    result = run_on_root(comm, func)
    local, shared = result if result is not None else (None, None)
    return local, comm.bcast(shared, root=0)


def get_default_backend_args(arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Get default backend arguments given a set of input arguments.

    Args:
        arguments: Input backend arguments

    Returns:
        The arguments, with defaults for:
            - ``"dtype"``: Floating point type of the points and fields returned,
              :data:`dolfinx.default_real_type`. VTK reads them in single precision.
            - ``"region"``: Region of a multi-region case to read, ``""`` for the
              default region. ``None`` (default) merges all regions into one mesh.
    """
    args: dict[str, Any] = {"dtype": dolfinx.default_real_type, "region": None}
    args.update(arguments or {})
    return args


def read_mesh_data(
    filename: Path | str,
    comm: MPI.Comm,
    time: str | float | None = None,
    read_from_partition: bool = False,
    backend_args: dict[str, Any] | None = None,
) -> ReadMeshData | ReadMixedMeshData:
    """Read the internal mesh of an OpenFOAM case.

    Input node ``i`` and cell ``i`` are OpenFOAM point and cell label ``i`` for a single
    region. Several regions are merged: their cells follow each other in region order,
    and the points are joined where they coincide. A mesh with several cell types is read
    as a mixed mesh, with its cells grouped by type.

    Args:
        filename: Path to the ``.foam`` file of the case
        comm: MPI communicator to distribute the mesh over
        time: Time step of the mesh, the first one if ``None``
        read_from_partition: Read mesh with partition from file (not supported)
        backend_args: Arguments to backend, see :func:`get_default_backend_args`

    Returns:
        Internal data structure for the mesh data read from file
    """
    if read_from_partition:
        raise NotImplementedError("The openfoam backend cannot read partitioning data.")
    check_file_exists(filename)
    backend_args = get_default_backend_args(backend_args)

    def read() -> tuple[
        tuple[np.ndarray, list[npt.NDArray[np.int64]]], tuple[list[str], list[int], int]
    ]:
        case = helpers.read_case(filename, time, backend_args["region"])
        types = case.cell_types
        blocks = [case.cells[t] for t in types]
        return (case.points, blocks), (types, [b.shape[1] for b in blocks], case.points.shape[1])

    local, (cell_types, num_nodes, gdim) = _on_root(comm, read)
    if local is None:
        x = np.zeros((0, gdim))
        cells = [np.zeros((0, n), dtype=np.int64) for n in num_nodes]
    else:
        x, cells = local
    x = x.astype(backend_args["dtype"])
    lvar = int(basix.LagrangeVariant.equispaced)
    if len(cell_types) == 1:
        return ReadMeshData(cells=cells[0], cell_type=cell_types[0], x=x, lvar=lvar, degree=1)
    return ReadMixedMeshData(cells=cells, cell_types=cell_types, x=x, lvar=lvar, degree=1)


def read_point_data(
    filename: Path | str,
    name: str,
    comm: MPI.Comm,
    time: float | str | None,
    backend_args: dict[str, Any] | None,
) -> tuple[np.ndarray, int]:
    """Read an OpenFOAM point field (``point*Field``).

    Cell-centred fields are not interpolated to the points; read them with
    :func:`read_cell_data`. Only meshes of one region and one cell type are supported.

    Args:
        filename: Path to the ``.foam`` file of the case
        name: Name of the field
        comm: Communicator to launch IO on.
        time: The time step, the first one if ``None``
        backend_args: The backend arguments, see :func:`get_default_backend_args`

    Returns:
        The values at the points, of shape ``(num_points, num_components)``, on rank 0,
        and the global start of the data on this process, ``0``
    """
    check_file_exists(filename)
    backend_args = get_default_backend_args(backend_args)

    def read() -> tuple[np.ndarray, int]:
        case = helpers.read_case(filename, time, backend_args["region"], point_fields=[name])
        if len(case.regions) > 1:
            raise NotImplementedError(
                f"Point data of a case with several regions {case.regions} is not "
                "supported. Select a region with backend_args={'region': name}."
            )
        case.get_single_type_cells("Reading point data")
        values = case.point_data[name]
        return values, values.shape[1]

    values, num_components = _on_root(comm, read)
    if values is None:
        values = np.zeros((0, num_components))
    return values.astype(backend_args["dtype"]), 0


def read_cell_data(
    filename: Path | str,
    name: str,
    comm: MPI.Comm,
    time: str | float | None,
    backend_args: dict[str, Any] | None,
) -> tuple[npt.NDArray[np.int64], np.ndarray]:
    """Read a cell-centred OpenFOAM field (``vol*Field``).

    For several regions, the field must exist in every region. Only meshes of one cell
    type are supported.

    Args:
        filename: Path to the ``.foam`` file of the case
        name: Name of the field
        comm: Communicator to launch IO on.
        time: The time step, the first one if ``None``
        backend_args: The backend arguments, see :func:`get_default_backend_args`

    Returns:
        A tuple (topology, values), where topology holds the input node indices of each
        cell on rank 0, and values the value of each cell
    """
    check_file_exists(filename)
    backend_args = get_default_backend_args(backend_args)

    def read() -> tuple[tuple[npt.NDArray[np.int64], np.ndarray], tuple[int, int]]:
        case = helpers.read_case(filename, time, backend_args["region"], cell_fields=[name])
        cells = case.get_single_type_cells("Reading cell data")
        values = case.cell_data[name]
        return (cells, values), (cells.shape[1], values.shape[1])

    local, (num_nodes, num_components) = _on_root(comm, read)
    if local is None:
        local = (np.zeros((0, num_nodes), dtype=np.int64), np.zeros((0, num_components)))
    cells, values = local
    return cells, values.astype(backend_args["dtype"])


def read_timestamps(
    filename: Path | str,
    comm: MPI.Comm,
    function_name: str,
    backend_args: dict[str, Any] | None,
) -> npt.NDArray[np.float64 | str]:  # type: ignore[type-var]
    """Read the time steps at which a field exists, in every region read.

    Args:
        filename: Path to the ``.foam`` file of the case
        comm: MPI communicator used in storage
        function_name: Name of the field
        backend_args: Arguments to backend, see :func:`get_default_backend_args`

    Returns:
        The time steps, in increasing order
    """
    check_file_exists(filename)
    backend_args = get_default_backend_args(backend_args)

    def read() -> tuple[None, list[float]]:
        reader = pyvista.POpenFOAMReader(str(filename))
        time_names = reader.reader.GetTimeNames()
        names = [time_names.GetValue(i) for i in range(time_names.GetNumberOfValues())]
        regions = helpers.get_regions(reader, backend_args["region"])
        case_dir = Path(filename).parent
        times = [
            float(t)
            for t, n in zip(reader.time_values, names)
            if all(helpers.field_exists(case_dir, n, r, function_name) for r in regions)
        ]
        return None, times

    _, times = _on_root(comm, read)
    return np.array(times, dtype=np.float64)


def read_function_names(
    filename: Path | str, comm: MPI.Comm, backend_args: dict[str, Any] | None
) -> list[str]:
    """Read the names of the cell and point fields of a case.

    Args:
        filename: Path to the ``.foam`` file of the case
        comm: MPI communicator to launch IO on.
        backend_args: Arguments to backend

    Returns:
        The names of the cell fields, then of the point fields
    """
    check_file_exists(filename)

    def read() -> tuple[None, list[str]]:
        reader = pyvista.POpenFOAMReader(str(filename))
        names = list(reader.cell_array_names) + list(reader.point_array_names)
        return None, list(dict.fromkeys(names))

    return _on_root(comm, read)[1]


def _tag_sets(case: helpers.Case, dim: int) -> list[tuple[str, Any]]:
    """The named sets of entities of a dimension, in id order: cell zones for cells, the
    boundary patches and then face zones for facets, and point zones for vertices."""
    tdim = dolfinx.mesh.cell_dim(dolfinx.mesh.to_type(case.cell_types[0]))
    if dim == tdim:
        return list(case.cell_zones.items())
    if dim == tdim - 1:
        return [(p.name, p.faces) for p in case.patches] + list(case.face_zones.items())
    if dim == 0:
        return list(case.point_zones.items())
    return []


def read_meshtags_data(
    filename: Path | str,
    comm: MPI.Comm,
    name: str,
    backend_args: dict[str, Any] | None = None,
) -> MeshTagsData:
    """Read the zones, boundary patches or regions of a case as mesh tags.

    Zones and patches are read from the ``polyMesh`` files (``boundary``, ``faces``,
    ``cellZones``, ``faceZones``, ``pointZones``) by their labels. In a case with several
    regions, their names are ``"<region>/<name>"``. Ids are numbered from 1 for each
    dimension, in the order below and of the files, and :func:`read_ids` lists them.

    Args:
        filename: Path to the ``.foam`` file of the case
        comm: MPI communicator
        name: One of:

            - ``"cell"``: the cells of all cell zones.
            - ``"facet"``: the facets of all boundary patches, then of all face zones.
            - ``"peak"``: the vertices of all point zones (vertices are the peaks of a 3D
              mesh, as in :func:`dolfinx.io.gmsh.model_to_mesh`).
            - ``"regions"``: all cells, tagged with the region id.
            - The name of a single patch or zone.

            The sets combined must not overlap, as a mesh tag holds one value per entity,
            except for the two interface patches of neighbouring regions, which hold the
            same facets: those are tagged with the first.

        backend_args: Arguments to backend, see :func:`get_default_backend_args`

    Returns:
        Internal data structure for the mesh tags read from file, on rank 0

    Raises:
        KeyError: For an unknown name.
        ValueError: If the sets combined overlap.
        NotImplementedError: On a mesh with several cell types, and for facets on a
            mesh of prisms or pyramids, as mesh tags hold entities of one type.
    """
    check_file_exists(filename)
    backend_args = get_default_backend_args(backend_args)

    def read() -> tuple[tuple[npt.NDArray[np.int64], npt.NDArray[np.int32]], tuple[int, int]]:
        case = helpers.read_case(filename, None, backend_args["region"])
        cells = case.get_single_type_cells("Reading mesh tags")
        cell_type = case.cell_types[0]
        tdim = dolfinx.mesh.cell_dim(dolfinx.mesh.to_type(cell_type))
        if name == "regions":
            region_ids = np.repeat(np.arange(1, len(case.regions) + 1), np.diff(case.cell_offsets))
            return (cells, region_ids.astype(np.int32)), (tdim, cells.shape[1])

        # The dimension of the tags, and the sets of that dimension to tag, with their ids
        if name in codim_names:
            dim = tdim - codim_names[name]
            selected = [(i + 1, n, e) for i, (n, e) in enumerate(_tag_sets(case, dim))]
        else:
            matches = [
                (d, i + 1, n, e)
                for d in (tdim, 0, tdim - 1)
                for i, (n, e) in enumerate(_tag_sets(case, d))
                if n == name
            ]
            if len(matches) == 0:
                names = [n for d in (tdim, tdim - 1, 0) for n, _ in _tag_sets(case, d)]
                raise KeyError(
                    f"Unknown mesh tags '{name}' in {filename}. Expected 'cell', 'facet', "
                    f"'peak', 'regions' or one of {names}."
                )
            if len({d for d, *_ in matches}) > 1:
                raise ValueError(f"'{name}' names zones of several dimensions in {filename}.")
            dim = matches[0][0]
            selected = [(i, n, e) for _, i, n, e in matches]
        if dim == tdim - 1 and cell_type in ("prism", "pyramid"):
            raise NotImplementedError(
                f"Facet tags on {cell_type} meshes are not supported, as their facets are "
                "of two types (triangles and quadrilaterals)."
            )

        # Combine the sets, one value per entity. Interface patches between regions hold
        # the same facets, which are tagged once; any other overlap is an error.
        interfaces: set[frozenset[str]] = set()
        if dim == tdim - 1:
            interfaces = {frozenset(pair) for pair in case.get_interfaces(case.patches)}
        owner: dict[Any, str] = {}
        entities: list[npt.NDArray[np.int64]] = []
        values: list[int] = []
        for value, set_name, members in selected:
            for entity in members:
                entity = np.atleast_1d(entity)
                key = tuple(sorted(entity.tolist()))
                if (other := owner.get(key)) is not None:
                    if frozenset((other, set_name)) in interfaces:
                        continue
                    raise ValueError(
                        f"'{other}' and '{set_name}' share entities in {filename}, but mesh "
                        "tags hold one value per entity. Read them one at a time by name."
                    )
                owner[key] = set_name
                entities.append(entity)
                values.append(value)

        if dim == tdim:
            indices = cells[np.array([e[0] for e in entities], dtype=np.int64)]
        else:
            if len({len(e) for e in entities}) > 1:
                raise NotImplementedError(
                    f"The entities of '{name}' are of several types; mesh tags of several "
                    "entity types are not supported."
                )
            num_nodes = len(entities[0]) if entities else (1 if dim == 0 else dim + 1)
            indices = np.array(entities, dtype=np.int64).reshape(-1, num_nodes)
        return (indices, np.array(values, dtype=np.int32)), (dim, int(indices.shape[1]))

    local, (dim, num_nodes) = _on_root(comm, read)
    if local is None:
        local = (np.zeros((0, num_nodes), dtype=np.int64), np.zeros(0, dtype=np.int32))
    return MeshTagsData(name=name, values=local[1], indices=local[0], dim=dim)


def read_ids(
    filename: Path | str, comm: MPI.Comm, backend_args: dict[str, Any] | None = None
) -> dict[str, dict[str, int]]:
    """The ids of the zones, patches and regions of a case, as tagged by
    :func:`read_meshtags_data`.

    Args:
        filename: Path to the ``.foam`` file of the case
        comm: MPI communicator
        backend_args: Arguments to backend, see :func:`get_default_backend_args`

    Returns:
        Map from ``"cell"`` (cell zones), ``"facet"`` (boundary patches, then face zones),
        ``"peak"`` (point zones) and ``"regions"`` to a map from each name to its id
    """
    check_file_exists(filename)
    backend_args = get_default_backend_args(backend_args)

    def read() -> tuple[None, dict[str, dict[str, int]]]:
        case = helpers.read_case(filename, None, backend_args["region"])
        tdim = dolfinx.mesh.cell_dim(dolfinx.mesh.to_type(case.cell_types[0]))
        ids = {
            key: {n: i + 1 for i, (n, _) in enumerate(_tag_sets(case, tdim - codim))}
            for key, codim in codim_names.items()
            if tdim - codim in (tdim, tdim - 1, 0)
        }
        ids["regions"] = {r: i + 1 for i, r in enumerate(case.regions)}
        return None, ids

    return _on_root(comm, read)[1]


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
    raise NotImplementedError("The openfoam backend cannot write attributes.")


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
    raise NotImplementedError("The openfoam backend cannot read attributes.")


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
    raise NotImplementedError("The openfoam backend cannot write meshes.")


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
    raise NotImplementedError("The openfoam backend cannot write meshtags.")


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
    raise NotImplementedError("The openfoam backend cannot make checkpoints.")


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
    raise NotImplementedError("The openfoam backend cannot make checkpoints.")


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
    raise NotImplementedError("The openfoam backend cannot make checkpoints.")


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
    raise NotImplementedError("The openfoam backend cannot make checkpoints.")


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
    raise NotImplementedError("The openfoam backend cannot read legacy DOLFIN meshes.")


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
    raise NotImplementedError("The openfoam backend cannot make checkpoints.")


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
    raise NotImplementedError("The openfoam backend cannot read HDF5 arrays")


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
    raise NotImplementedError("The openfoam backend does not support writing point data")
