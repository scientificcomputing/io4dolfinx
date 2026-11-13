"""
Gmsh interface to io4dolfinx.

Reads and writes Gmsh ``.msh`` files (MSH 2.2 and 4.1, ASCII and binary) through the
Gmsh Python API, reusing :mod:`dolfinx.io.gmsh` for the geometry, the topology and the
node ordering. All file access happens on rank 0. Only the element types of
``dolfinx.io.gmsh`` are supported, and DOLFINx 0.11 or later is required.

In DOLFINx terms:

- As in :func:`dolfinx.io.gmsh.model_to_mesh`, the cells are the elements of the
  highest dimension in a physical group, and every cell is in exactly one physical
  group. Node ``i`` has node tag ``i + 1``. An MSH file holds a single mesh, so meshes
  are not time dependent.
- Point and cell data are post-processing views (``$NodeData``/``$ElementData``) keyed
  by node and element tags. The steps of a view are its time series. Complex data is
  split into the views ``real_{name}`` and ``imag_{name}``, as in
  :class:`dolfinx.io.XDMFFile`.
- Mesh tags are physical groups, with the physical tag as the tag value.

SPDX License identifier: MIT

Copyright: Jørgen S. Dokken and Simula Research Laboratory
"""

from pathlib import Path
from typing import Any, Callable, TypeVar

from mpi4py import MPI

import basix
import dolfinx
import fenicsx_compat
import numpy as np
import numpy.typing as npt
from dolfinx.io.gmsh import (
    _gmsh_to_cells,
    cell_perm_array,
    extract_geometry,
    extract_topology_and_markers,
)

try:
    import gmsh
except ImportError:
    raise ModuleNotFoundError("The gmsh-backend requires gmsh")

from ...structures import ArrayData, FunctionData, MeshData, MeshTagsData, ReadMeshData
from ...utils import check_file_exists, codim_names
from .. import FileMode, ReadMode

read_mode = ReadMode.serial

# Gmsh options changed by this backend, restored if Gmsh was initialized by the caller
_options = ("General.Terminal", "Mesh.Binary", "Mesh.SaveAll", "PostProcessing.SaveMesh")

Local = TypeVar("Local")
Shared = TypeVar("Shared")


def _run_on_root(
    comm: MPI.Comm, func: Callable[[list[int]], tuple[Local, Shared]]
) -> tuple[Local | None, Shared]:
    """Run a function in a Gmsh session on rank 0 and share its outcome with all ranks.

    If Gmsh is already initialized, e.g. by the caller, the function runs in a new model,
    which is removed afterwards together with the views it created, and the previous
    model and options are restored. Otherwise Gmsh is initialized and finalized around it.

    Args:
        comm: MPI communicator. Collective on all ranks.
        func: Function to run on rank 0. It takes the tags of the views that existed
            before the session, which do not belong to the file, and returns a tuple of
            data kept on rank 0 and data broadcast to all ranks.

    Returns:
        Tuple containing:
            - The first output of ``func`` on rank 0, ``None`` on other ranks
            - The second output of ``func`` on all ranks

    Raises:
        NotImplementedError: If DOLFINx is older than 0.11.
        Exception: An exception raised by ``func`` is raised on all ranks.
    """
    if fenicsx_compat.before("0.11.0.dev0"):
        raise NotImplementedError("The gmsh backend requires DOLFINx 0.11 or later.")

    result: Local | None = None
    shared: Shared | None = None
    error: Exception | None = None
    if comm.rank == 0:
        initialized = gmsh.isInitialized()
        if initialized:
            previous_model = gmsh.model.getCurrent()
            previous_options = {option: gmsh.option.getNumber(option) for option in _options}
            # Views are global in gmsh, not per model, so existing ones must be skipped and kept
            preexisting_views = list(gmsh.view.getTags())
            gmsh.model.add("io4dolfinx")
        else:
            gmsh.initialize(interruptible=False)
            preexisting_views = []
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            result, shared = func(preexisting_views)
        except Exception as e:
            error = e
        finally:
            if initialized:
                for view in gmsh.view.getTags():
                    if view not in preexisting_views:
                        gmsh.view.remove(view)
                gmsh.model.remove()
                gmsh.model.setCurrent(previous_model)
                for option, value in previous_options.items():
                    gmsh.option.setNumber(option, value)
            else:
                gmsh.finalize()
    error, shared = comm.bcast((error, shared), root=0)
    if error is not None:
        raise error
    return result, shared  # type: ignore[return-value]


def _find_view(
    name: str, preexisting_views: list[int]
) -> tuple[list[int], npt.NDArray[np.float64]] | None:
    """Find the views of the current Gmsh session holding the data with a given name.

    A gmsh view holds all time steps of one field, each step storing its time.
    Complex data is stored as two views, ``real_{name}`` and ``imag_{name}``, as
    :class:`dolfinx.io.XDMFFile` does.

    Args:
        name: Name of the data
        preexisting_views: Tags of the views that do not belong to the file, which are skipped

    Returns:
        ``None`` if there is no such data, otherwise a tuple containing:
            - The view tags, ``[view]`` for real data and ``[real, imag]`` for complex data
            - The time of each step of the views
    """
    views = {
        gmsh.view.option.getString(view, "Name"): view
        for view in gmsh.view.getTags()
        if view not in preexisting_views
    }
    real, imag = f"real_{name}", f"imag_{name}"
    if name in views:
        found = [views[name]]
    elif real in views and imag in views:
        found = [views[real], views[imag]]
    elif real in views or imag in views:
        raise RuntimeError(
            f"Found only one part of complex data '{name}', expected '{real}' and '{imag}'."
        )
    else:
        return None
    num_steps = int(gmsh.view.option.getNumber(found[0], "NbTimeStep"))
    times = [gmsh.view.getModelData(found[0], step)[3] for step in range(num_steps)]
    return found, np.array(times, dtype=np.float64)


def _tagged_entities() -> dict[
    int,
    tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int32], str, int] | None,
]:
    """Get the elements in physical groups of the current Gmsh model, by dimension.

    Uses :func:`dolfinx.io.gmsh.extract_topology_and_markers`, so as in
    :func:`dolfinx.io.gmsh.model_to_mesh` the highest dimension holds the cells, and an
    element in several physical groups appears once per group.

    Returns:
        Map from topological dimension to ``None`` if the elements of that dimension are
        of several types or of a type not supported by :mod:`dolfinx.io.gmsh`, otherwise
        to a tuple containing:
            - The element tags, sorted
            - The element nodes as zero-based node indices in DOLFINx ordering, of shape
              ``(num_elements, num_nodes_per_element)``, one row per element tag
            - The physical tag of each element
            - The DOLFINx cell type of the elements
            - The degree of the elements
    """
    topologies, _ = extract_topology_and_markers(gmsh.model)
    if len(topologies) == 0:
        raise ValueError("No physical groups in the Gmsh model.")
    types_by_dim: dict[int, list[int]] = {}
    for gmsh_type in topologies:
        if gmsh_type in _gmsh_to_cells:
            cell_type = dolfinx.mesh.to_type(_gmsh_to_cells[gmsh_type][0])
            dim = dolfinx.mesh.cell_dim(cell_type)
        else:
            dim = gmsh.model.mesh.getElementProperties(gmsh_type)[1]
        types_by_dim.setdefault(dim, []).append(gmsh_type)

    entities: dict[int, Any] = {}
    for dim, types in types_by_dim.items():
        if len(types) > 1 or types[0] not in _gmsh_to_cells:
            entities[dim] = None
            continue
        cell_name, degree = _gmsh_to_cells[types[0]]
        data = topologies[types[0]]
        tags = data["entity_tags"].astype(np.int64)
        order = np.argsort(tags, kind="stable")
        nodes = data["topology"].astype(np.int64)
        perm = cell_perm_array(dolfinx.mesh.to_type(cell_name), nodes.shape[1])
        values = data["cell_data"].astype(np.int32)
        entities[dim] = (tags[order], nodes[order][:, perm], values[order], cell_name, degree)
    return entities


def get_default_backend_args(arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Get default backend arguments given a set of input arguments.

    Args:
        arguments: Input backend arguments

    Returns:
        The arguments, with defaults for:
            - ``"gdim"``: Geometric dimension of a mesh read from file, ``3`` as in
              :func:`dolfinx.io.gmsh.read_from_msh`
            - ``"dtype"``: Floating point type of the coordinates and data read from file,
              ``np.float64``
            - ``"binary"``: Write binary instead of ASCII MSH. ``None`` keeps the encoding
              of an existing file and writes a new one as ASCII.
    """
    args: dict[str, Any] = {"gdim": 3, "dtype": np.float64, "binary": None}
    args.update(arguments or {})
    return args


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
    raise NotImplementedError("The gmsh backend cannot write attributes.")


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
    raise NotImplementedError("The gmsh backend cannot read attributes.")


def read_timestamps(
    filename: Path | str,
    comm: MPI.Comm,
    function_name: str,
    backend_args: dict[str, Any] | None,
) -> npt.NDArray[np.float64 | str]:  # type: ignore[type-var]
    """Read the times of the steps of point or cell data.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        function_name: Name of the point or cell data
        backend_args: Arguments to backend

    Returns:
        The time of each step, on all ranks
    """
    check_file_exists(filename)

    def read(preexisting_views: list[int]) -> tuple[None, npt.NDArray[np.float64]]:
        gmsh.merge(str(filename))
        if (found := _find_view(function_name, preexisting_views)) is None:
            raise KeyError(f"Could not find data '{function_name}' in {filename}.")
        return None, found[1]

    return _run_on_root(comm, read)[1]


def read_function_names(
    filename: Path | str, comm: MPI.Comm, backend_args: dict[str, Any] | None
) -> list[str]:
    """Read the names of all point and cell data in a file.

    Args:
        filename: Path to file
        comm: MPI communicator to launch IO on.
        backend_args: Arguments to backend

    Returns:
        The names of the views in the file, on all ranks. A complex pair of views
        ``real_{name}`` and ``imag_{name}`` is listed as ``name``.
    """
    check_file_exists(filename)

    def read(preexisting_views: list[int]) -> tuple[None, list[str]]:
        gmsh.merge(str(filename))
        views = [view for view in gmsh.view.getTags() if view not in preexisting_views]
        names = [gmsh.view.option.getString(view, "Name") for view in views]
        complex_names = {n[5:] for n in names if n.startswith("real_") and f"imag_{n[5:]}" in names}
        functions = [
            n[5:] if n.startswith(("real_", "imag_")) and n[5:] in complex_names else n
            for n in names
        ]
        return None, list(dict.fromkeys(functions))

    return _run_on_root(comm, read)[1]


def write_mesh(
    filename: Path | str,
    comm: MPI.Comm,
    mesh: MeshData,
    backend_args: dict[str, Any] | None = None,
    mode: FileMode = FileMode.write,
    time: float = 0.0,
):
    """Write a mesh to a Gmsh file.

    Nodes and cells get the tag one above their global index. The cells form a single
    entity in the physical group with tag 1, the default cell marker, as every cell must
    be in a physical group (see :func:`dolfinx.io.gmsh.model_to_mesh`). Cell tags written
    with :func:`write_meshtags` replace it.

    Args:
        filename: Path to file to write to, with suffix ``.msh``
        comm: MPI communicator used in storage
        mesh: Internal data structure for the mesh data to save to file
        backend_args: Arguments to backend, ``"binary"`` selects binary MSH
        mode: Only :attr:`FileMode.write`, as an MSH file holds one mesh
        time: Ignored, as an MSH file holds one mesh
    """
    if mode != FileMode.write:
        raise NotImplementedError("The gmsh backend writes a single mesh per file.")
    if Path(filename).suffix != ".msh":
        raise ValueError(f"Expected a filename with suffix .msh, got {filename}.")
    if mesh.degree > 2 and mesh.lagrange_variant != int(basix.LagrangeVariant.equispaced):
        raise NotImplementedError("Gmsh only supports equispaced Lagrange geometries.")
    cell = (mesh.cell_type, mesh.degree)
    if (gmsh_type := next((t for t, c in _gmsh_to_cells.items() if c == cell), None)) is None:
        raise NotImplementedError(
            f"Gmsh has no element for a {mesh.cell_type} of degree {mesh.degree}."
        )
    backend_args = get_default_backend_args(backend_args)

    # Rank order is global index order
    nodes = comm.gather(mesh.local_geometry, root=0)
    cells = comm.gather(mesh.local_topology, root=0)

    def write(preexisting_views: list[int]) -> tuple[None, None]:
        assert nodes is not None and cells is not None
        x = np.vstack(nodes)
        x_3d = np.zeros((x.shape[0], 3), dtype=np.float64)
        x_3d[:, : x.shape[1]] = x
        topology = np.vstack(cells)
        gmsh_topology = np.empty_like(topology)
        cell_type = dolfinx.mesh.to_type(mesh.cell_type)
        gmsh_topology[:, cell_perm_array(cell_type, topology.shape[1])] = topology

        tdim = dolfinx.mesh.cell_dim(cell_type)
        gmsh.model.addDiscreteEntity(tdim, 1)
        gmsh.model.mesh.addNodes(tdim, 1, np.arange(1, x.shape[0] + 1), x_3d.reshape(-1))
        gmsh.model.mesh.addElementsByType(
            1,
            gmsh_type,
            np.arange(1, topology.shape[0] + 1),
            gmsh_topology.reshape(-1) + 1,
        )
        gmsh.model.addPhysicalGroup(tdim, [1], 1)
        gmsh.option.setNumber("Mesh.Binary", int(bool(backend_args["binary"])))
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.write(str(filename))
        return None, None

    _run_on_root(comm, write)


def read_mesh_data(
    filename: Path | str,
    comm: MPI.Comm,
    time: str | float | None = None,
    read_from_partition: bool = False,
    backend_args: dict[str, Any] | None = None,
) -> ReadMeshData:
    """Read mesh data from a Gmsh file.

    As in :func:`dolfinx.io.gmsh.model_to_mesh`, the cells are the elements of the
    highest dimension in a physical group, and every cell must be in exactly one physical
    group. The cells must be of one type, and node tags contiguous. All data is read on
    rank 0.

    Args:
        filename: Path to file to read from
        comm: MPI communicator to distribute the mesh over
        time: Ignored, as an MSH file holds one mesh
        read_from_partition: Read mesh with partition from file (not supported)
        backend_args: Arguments to backend, see :func:`get_default_backend_args`

    Returns:
        Internal data structure for the mesh data read from file
    """
    if read_from_partition:
        raise NotImplementedError("The gmsh backend cannot read partitioning data.")
    check_file_exists(filename)
    backend_args = get_default_backend_args(backend_args)
    gdim = backend_args["gdim"]

    def read(
        preexisting_views: list[int],
    ) -> tuple[tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]], tuple[str, int, int]]:
        gmsh.merge(str(filename))
        x = extract_geometry(gmsh.model)[:, :gdim]
        tagged = _tagged_entities()
        tdim = max(tagged)
        if (cell_data := tagged[tdim]) is None:
            raise NotImplementedError(f"Mixed or unsupported cell types in {filename}.")
        tags, cells, _, cell_type, degree = cell_data
        num_cells = sum(len(t) for t in gmsh.model.mesh.getElements(dim=tdim, tag=-1)[1])
        if len(tags) != num_cells or len(np.unique(tags)) != len(tags):
            raise ValueError(
                f"All cells are expected to be in exactly one physical group, as in "
                f"dolfinx.io.gmsh.model_to_mesh; found {len(tags)} tags for {num_cells} cells."
            )
        return (x, cells), (cell_type, degree, cells.shape[1])

    local, (cell_type, degree, num_nodes_per_cell) = _run_on_root(comm, read)
    if local is None:
        x = np.zeros((0, gdim), dtype=np.float64)
        cells = np.zeros((0, num_nodes_per_cell), dtype=np.int64)
    else:
        x, cells = local
    return ReadMeshData(
        cells=cells,
        cell_type=cell_type,
        x=x.astype(backend_args["dtype"]),
        lvar=int(basix.LagrangeVariant.equispaced),
        degree=degree,
    )


def write_meshtags(
    filename: str | Path,
    comm: MPI.Comm,
    data: MeshTagsData,
    backend_args: dict[str, Any] | None = None,
):
    """Write mesh tags to a Gmsh file as physical groups.

    Each tag value becomes a physical group with the value as tag, named
    ``f"{data.name}_{value}"``. Gmsh physical tags are positive and unique per
    dimension, so tag values must be positive.

    Cell tags replace the marker of the cells they cover, moving them to the physical
    group of their value, while other cells keep their marker, e.g. the default marker 1
    of :func:`write_mesh`. Markers no cell has any more are removed. Entities of lower
    dimension are added as new elements, and only one set of mesh tags can be written
    per lower dimension.

    Args:
        filename: Path to file to write to, holding a mesh written with
            :func:`write_mesh`
        comm: MPI communicator used in storage
        data: Internal data structure for the mesh tags to save to file
        backend_args: Arguments to backend, ``"binary"`` selects binary MSH
    """
    check_file_exists(filename)
    backend_args = get_default_backend_args(backend_args)
    indices = comm.gather(data.indices, root=0)
    values = comm.gather(data.values, root=0)

    def write(preexisting_views: list[int]) -> tuple[None, None]:
        assert indices is not None and values is not None
        entities = np.vstack(indices).astype(np.int64)
        tag_values = np.hstack(values).astype(np.int32)
        if np.any(tag_values <= 0):
            raise ValueError(f"Gmsh physical tags must be positive, got {np.unique(tag_values)}.")
        if (binary := backend_args["binary"]) is None:
            # The second line of an MSH file is "version file-type data-size"
            with open(filename, "rb") as file:
                file.readline()
                binary = file.readline().split()[1] == b"1"

        gmsh.merge(str(filename))
        tagged = _tagged_entities()
        tdim = max(tagged)
        if (cell_data := tagged[tdim]) is None:
            raise NotImplementedError(f"Mixed or unsupported cell types in {filename}.")
        cell_tags, cells, _, _, degree = cell_data
        dim = data.dim
        groups = [tag for d, tag in gmsh.model.getPhysicalGroups() if d == dim]
        if dim < tdim and len(groups) > 0:
            raise NotImplementedError(
                f"{filename} already has physical groups of dimension {dim}, and only one "
                "set of mesh tags can be written per dimension below the cells."
            )

        assert data.cell_type is not None
        entity_degree = 0 if data.cell_type == "point" else degree
        cell = (data.cell_type, entity_degree)
        if (gmsh_type := next((t for t, c in _gmsh_to_cells.items() if c == cell), None)) is None:
            raise NotImplementedError(
                f"Gmsh has no element for a {data.cell_type} of degree {entity_degree}."
            )
        gmsh_entities = np.empty_like(entities)
        perm = cell_perm_array(dolfinx.mesh.to_type(data.cell_type), entities.shape[1])
        gmsh_entities[:, perm] = entities

        if dim == tdim:
            # Find the element tag of each tagged cell by matching node sets
            _, inverse = np.unique(
                np.sort(np.vstack([cells, entities]), axis=1), axis=0, return_inverse=True
            )
            inverse = inverse.reshape(-1)
            cell_position = np.empty(inverse.max() + 1, dtype=np.int64)
            cell_position[inverse[: len(cells)]] = np.arange(len(cells))
            element_tags = cell_tags[cell_position[inverse[len(cells) :]]]
            # Gmsh rejects removing elements an entity does not hold
            for d, entity in gmsh.model.getEntities(tdim):
                entity_tags = gmsh.model.mesh.getElements(d, entity)[1]
                held = np.intersect1d(np.hstack([*entity_tags, []]), element_tags)
                if len(held) > 0:
                    gmsh.model.mesh.removeElements(d, entity, held.astype(np.int64))
        else:
            first = gmsh.model.mesh.getMaxElementTag() + 1
            element_tags = np.arange(first, first + len(entities), dtype=np.int64)

        for value in np.unique(tag_values):
            selected = tag_values == value
            if value in groups:
                # A marker the cells already have, e.g. the default marker 1
                entity = gmsh.model.getEntitiesForPhysicalGroup(dim, int(value))[0]
                gmsh.model.setPhysicalName(dim, int(value), f"{data.name}_{value}")
            else:
                entity = gmsh.model.addDiscreteEntity(dim)
                gmsh.model.addPhysicalGroup(dim, [entity], int(value), f"{data.name}_{value}")
            gmsh.model.mesh.addElementsByType(
                entity, gmsh_type, element_tags[selected], gmsh_entities[selected].reshape(-1) + 1
            )
        if dim == tdim:
            # Remove the markers no cell has any more
            for tag in groups:
                group_entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
                num_cells = sum(
                    len(t) for e in group_entities for t in gmsh.model.mesh.getElements(dim, e)[1]
                )
                if num_cells == 0:
                    gmsh.model.removePhysicalGroups([(dim, tag)])

        gmsh.option.setNumber("Mesh.Binary", int(binary))
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.write(str(filename))
        # Append the views without repeating the mesh
        gmsh.option.setNumber("PostProcessing.SaveMesh", 0)
        for view in gmsh.view.getTags():
            if view not in preexisting_views:
                gmsh.view.write(view, str(filename), append=True)
        return None, None

    _run_on_root(comm, write)


def read_meshtags_data(
    filename: str | Path,
    comm: MPI.Comm,
    name: str,
    backend_args: dict[str, Any] | None = None,
) -> MeshTagsData:
    """Read mesh tags from the physical groups of a Gmsh file.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        name: ``"cell"``, ``"facet"``, ``"ridge"`` or ``"peak"`` for all physical groups of
            that codimension, as in :func:`dolfinx.io.gmsh.model_to_mesh`. Otherwise the
            physical group called ``name``, or the groups called ``f"{name}_{tag}"``, as
            written by :func:`write_meshtags`.
        backend_args: Arguments to backend

    Returns:
        Internal data structure for the mesh tags read from file, with the physical tag as
        value. All data is on rank 0.
    """
    check_file_exists(filename)

    def read(
        preexisting_views: list[int],
    ) -> tuple[tuple[npt.NDArray[np.int64], npt.NDArray[np.int32]], tuple[int, int]]:
        gmsh.merge(str(filename))
        tagged = _tagged_entities()
        tdim = max(tagged)
        groups = gmsh.model.getPhysicalGroups()
        if name in codim_names:
            dim = tdim - codim_names[name]
            group_tags = [tag for d, tag in groups if d == dim]
        else:
            matches = [
                (d, tag)
                for d, tag in groups
                if gmsh.model.getPhysicalName(d, tag) in (name, f"{name}_{tag}")
            ]
            if len(matches) == 0:
                available = sorted(gmsh.model.getPhysicalName(d, tag) for d, tag in groups)
                raise KeyError(
                    f"Could not find physical group '{name}' in {filename}. Available: {available}"
                )
            if len(dims := {d for d, _ in matches}) != 1:
                raise RuntimeError(f"Physical groups '{name}' have several dimensions {dims}.")
            dim = dims.pop()
            group_tags = [tag for _, tag in matches]
        if dim not in tagged:
            return (np.zeros((0, 0), dtype=np.int64), np.zeros(0, dtype=np.int32)), (dim, 0)
        if (entity_data := tagged[dim]) is None:
            raise NotImplementedError(f"Mixed or unsupported element types of dimension {dim}.")
        _, entities, tag_values, _, _ = entity_data
        keep = np.isin(tag_values, group_tags)
        return (entities[keep], tag_values[keep]), (dim, entities.shape[1])

    local, (dim, num_nodes_per_entity) = _run_on_root(comm, read)
    if local is None:
        local = (np.zeros((0, num_nodes_per_entity), dtype=np.int64), np.zeros(0, dtype=np.int32))
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
        Dofmap as an :class:`dolfinx.graph.AdjacencyList`
    """
    raise NotImplementedError("The gmsh backend cannot make checkpoints.")


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
    raise NotImplementedError("The gmsh backend cannot make checkpoints.")


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
    raise NotImplementedError("The gmsh backend cannot make checkpoints.")


def write_function(
    filename: Path | str,
    comm: MPI.Comm,
    u: FunctionData,
    time: float,
    mode: FileMode,
    backend_args: dict[str, Any] | None,
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
    raise NotImplementedError("The gmsh backend cannot make checkpoints.")


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
    raise NotImplementedError("The gmsh backend cannot read legacy DOLFIN meshes.")


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
    raise NotImplementedError("The gmsh backend cannot make checkpoints.")


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
    raise NotImplementedError("The gmsh backend cannot read HDF5 arrays.")


def read_point_data(
    filename: Path | str,
    name: str,
    comm: MPI.Comm,
    time: str | float | None,
    backend_args: dict[str, Any] | None,
) -> tuple[np.ndarray, int]:
    """Read point data (``$NodeData``) from a Gmsh file.

    Args:
        filename: Path to file
        name: Name of point data
        comm: Communicator to launch IO on.
        time: Time of the step to read, ``None`` for the first step
        backend_args: Arguments to backend, ``"dtype"`` sets the type of the data, or of
            its real and imaginary parts for complex data

    Returns:
        Tuple containing:
            - The data of shape ``(num_nodes, num_components)``, ordered by node tag, on
              rank 0. Nodes without data get zeros. Complex data, stored as the views
              ``real_{name}`` and ``imag_{name}``, is combined.
            - The global start of the data on this process, ``0``
    """
    check_file_exists(filename)
    backend_args = get_default_backend_args(backend_args)

    def read(
        preexisting_views: list[int],
    ) -> tuple[npt.NDArray[np.float64 | np.complex128], tuple[int, bool]]:
        gmsh.merge(str(filename))
        if (found := _find_view(name, preexisting_views)) is None:
            raise KeyError(f"Could not find point data '{name}' in {filename}.")
        views, times = found
        if time is None:
            step = 0
        else:
            steps = np.flatnonzero(np.isclose(times, float(time)))
            if len(steps) != 1:
                raise KeyError(
                    f"Expected one step of '{name}' at time {time} in {filename}, found "
                    f"{len(steps)}. Available times: {times.tolist()}"
                )
            step = int(steps[0])
        parts = []
        for view in views:
            data_type, tags, data, _, num_components = gmsh.view.getModelData(view, step)
            if data_type != "NodeData":
                raise ValueError(f"'{name}' in {filename} is {data_type}, not NodeData.")
            part = np.zeros((gmsh.model.mesh.getMaxNodeTag(), num_components), dtype=np.float64)
            part[np.asarray(tags, dtype=np.int64) - 1] = np.asarray(data).reshape(
                len(tags), num_components
            )
            parts.append(part)
        values = parts[0] if len(parts) == 1 else parts[0] + 1j * parts[1]
        return values, (num_components, len(parts) == 2)

    values, (num_components, is_complex) = _run_on_root(comm, read)
    dtype = backend_args["dtype"]
    if is_complex:
        # complex64 from float32 parts, complex128 from float64
        dtype = np.result_type(dtype, np.complex64)
    if values is None:
        values = np.zeros((0, num_components), dtype=dtype)
    return values.astype(dtype), 0


def read_cell_data(
    filename: Path | str,
    name: str,
    comm: MPI.Comm,
    time: str | float | None,
    backend_args: dict[str, Any] | None,
) -> tuple[npt.NDArray[np.int64], np.ndarray]:
    """Read cell data (``$ElementData``) from a Gmsh file.

    Args:
        filename: Path to file
        name: Name of cell data
        comm: Communicator to launch IO on.
        time: Time of the step to read, ``None`` for the first step
        backend_args: Arguments to backend, ``"dtype"`` sets the type of the data, or of
            its real and imaginary parts for complex data

    Returns:
        Tuple containing:
            - The nodes of the cells with data, as zero-based node indices in DOLFINx
              ordering, on rank 0
            - The data of these cells, of shape ``(num_cells, num_components)``, on rank 0.
              Complex data, stored as the views ``real_{name}`` and ``imag_{name}``, is
              combined.
    """
    check_file_exists(filename)
    backend_args = get_default_backend_args(backend_args)

    def read(
        preexisting_views: list[int],
    ) -> tuple[
        tuple[npt.NDArray[np.int64], npt.NDArray[np.float64 | np.complex128]],
        tuple[int, int, bool],
    ]:
        gmsh.merge(str(filename))
        if (found := _find_view(name, preexisting_views)) is None:
            raise KeyError(f"Could not find cell data '{name}' in {filename}.")
        views, times = found
        if time is None:
            step = 0
        else:
            steps = np.flatnonzero(np.isclose(times, float(time)))
            if len(steps) != 1:
                raise KeyError(
                    f"Expected one step of '{name}' at time {time} in {filename}, found "
                    f"{len(steps)}. Available times: {times.tolist()}"
                )
            step = int(steps[0])
        tagged = _tagged_entities()
        if (cell_data := tagged[max(tagged)]) is None:
            raise NotImplementedError(f"Mixed or unsupported cell types in {filename}.")
        element_tags, cells = cell_data[:2]
        parts, part_tags = [], []
        for view in views:
            data_type, tags, data, _, num_components = gmsh.view.getModelData(view, step)
            if data_type != "ElementData":
                raise ValueError(f"'{name}' in {filename} is {data_type}, not ElementData.")
            part_tags.append(np.asarray(tags, dtype=np.int64))
            parts.append(np.asarray(data, dtype=np.float64).reshape(len(tags), num_components))
        if len(parts) == 2 and not np.array_equal(part_tags[0], part_tags[1]):
            raise ValueError(f"The real and imaginary parts of '{name}' are on different cells.")
        rows = np.searchsorted(element_tags, part_tags[0])
        if np.any(rows >= len(element_tags)) or np.any(element_tags[rows] != part_tags[0]):
            raise ValueError(f"'{name}' in {filename} holds data on elements that are not cells.")
        values = parts[0] if len(parts) == 1 else parts[0] + 1j * parts[1]
        return (cells[rows], values), (num_components, cells.shape[1], len(parts) == 2)

    local, (num_components, num_nodes_per_cell, is_complex) = _run_on_root(comm, read)
    dtype = backend_args["dtype"]
    if is_complex:
        # complex64 from float32 parts, complex128 from float64
        dtype = np.result_type(dtype, np.complex64)
    if local is None:
        local = (
            np.zeros((0, num_nodes_per_cell), dtype=np.int64),
            np.zeros((0, num_components), dtype=dtype),
        )
    return local[0], local[1].astype(dtype)


def write_data(
    filename: Path | str,
    array_data: ArrayData,
    comm: MPI.Comm,
    time: str | float | None,
    mode: FileMode,
    backend_args: dict[str, Any] | None,
):
    """Write point or cell data to a Gmsh file, as a step of the view named after it.

    Point data is written as ``$NodeData`` and cell data as ``$ElementData``, keyed by
    global index + 1. Complex data is split into the views ``real_{name}`` and
    ``imag_{name}``, as :class:`dolfinx.io.XDMFFile` does. The mesh must have been
    written to the file with :func:`write_mesh`, from the same mesh.

    Args:
        filename: Path to file
        array_data: Data to write to file
        comm: MPI communicator to open the file with
        time: Time of the step, ``None`` for ``0``
        mode: :attr:`FileMode.append` adds the step, or replaces the step at the same
            time; :attr:`FileMode.write` replaces all steps of the data
        backend_args: Arguments to backend, ``"binary"`` selects binary MSH
    """
    if (data_type := {"Point": "NodeData", "Cell": "ElementData"}.get(array_data.type)) is None:
        raise ValueError(f"Unknown data type {array_data.type}, expected 'Point' or 'Cell'.")
    check_file_exists(filename)
    backend_args = get_default_backend_args(backend_args)
    step_time = 0.0 if time is None else float(time)
    values = comm.gather(array_data.values, root=0)

    def write(preexisting_views: list[int]) -> tuple[None, None]:
        assert values is not None
        if (binary := backend_args["binary"]) is None:
            # The second line of an MSH file is "version file-type data-size"
            with open(filename, "rb") as file:
                file.readline()
                binary = file.readline().split()[1] == b"1"

        gmsh.merge(str(filename))
        data = np.vstack(values)
        name = array_data.name
        if np.iscomplexobj(data):
            parts = [(f"real_{name}", data.real), (f"imag_{name}", data.imag)]
        else:
            parts = [(name, data)]
        for part_name, part in parts:
            found = _find_view(part_name, preexisting_views)
            if found is not None and len(found[0]) > 1:
                raise ValueError(f"'{part_name}' in {filename} is complex data.")
            if found is not None and mode == FileMode.write:
                gmsh.view.remove(found[0][0])
                found = None
            if found is None:
                view = gmsh.view.add(part_name)
                step = 0
            else:
                view, times = found[0][0], found[1]
                if (existing := gmsh.view.getModelData(view, 0)[0]) != data_type:
                    raise ValueError(f"'{part_name}' in {filename} is {existing}.")
                same_time = np.flatnonzero(np.isclose(times, step_time))
                step = int(same_time[0]) if len(same_time) > 0 else len(times)
            gmsh.view.addModelData(
                view,
                step,
                gmsh.model.getCurrent(),
                data_type,
                np.arange(1, part.shape[0] + 1),
                part.astype(np.float64),
                time=step_time,
                numComponents=part.shape[1],
            )

        gmsh.option.setNumber("Mesh.Binary", int(binary))
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.write(str(filename))
        # Append the views without repeating the mesh
        gmsh.option.setNumber("PostProcessing.SaveMesh", 0)
        for v in gmsh.view.getTags():
            if v not in preexisting_views:
                gmsh.view.write(v, str(filename), append=True)
        return None, None

    _run_on_root(comm, write)
