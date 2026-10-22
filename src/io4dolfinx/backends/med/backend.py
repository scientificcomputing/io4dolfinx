"""
MED interface to io4dolfinx.

MED is the HDF5 based mesh format of Salome and Code_Aster. See
https://docs.salome-platform.org/latest/dev/MEDCoupling/developer/med-file.html

Families and groups, in DOLFINx terms:

- Every entity stores one *family* id (``MAI/<type>/FAM`` for elements, negative;
  ``NOE/FAM`` for nodes, positive; ``0`` for none). Per element type, these ids are
  the values of a :class:`dolfinx.mesh.MeshTags`, with family 0 meaning untagged.
- ``FAS/<mesh>`` maps each family id to the named *groups* it belongs to, which
  ``MeshTags`` has no equivalent of.
- A family is one combination of groups, so overlapping groups (an entity in both
  ``inlet`` and ``wall``) get a family of their own. A group is the union of its
  families, read as a separate ``MeshTags`` with value 1.

Element family ids are negated when read, as integral subdomain ids in DOLFINx/FFCx
must be non-negative.

SPDX License identifier: MIT

Copyright: Jørgen S. Dokken and Simula Research Laboratory
"""

from pathlib import Path
from typing import Any, Literal

from mpi4py import MPI

import basix
import dolfinx
import h5py
import numpy as np
import numpy.typing as npt

from ...structures import ArrayData, FunctionData, MeshData, MeshTagsData, ReadMeshData
from ...utils import check_file_exists, compute_local_range
from .. import DEFAULT_MESH_NAME, FileMode, ReadMode, get_mesh_name
from ..h5py.backend import h5pyfile, read_component_major_rows

read_mode = ReadMode.parallel

# MED element type -> (DOLFINx cell type, degree)
_med_cell_types = {
    "PO1": (dolfinx.mesh.CellType.point, 1),
    "SE2": (dolfinx.mesh.CellType.interval, 1),
    "SE3": (dolfinx.mesh.CellType.interval, 2),
    "TR3": (dolfinx.mesh.CellType.triangle, 1),
    "TR6": (dolfinx.mesh.CellType.triangle, 2),
    "TR7": (dolfinx.mesh.CellType.triangle, 2),
    "QU4": (dolfinx.mesh.CellType.quadrilateral, 1),
    "QU8": (dolfinx.mesh.CellType.quadrilateral, 2),
    "QU9": (dolfinx.mesh.CellType.quadrilateral, 2),
    "TE4": (dolfinx.mesh.CellType.tetrahedron, 1),
    "T10": (dolfinx.mesh.CellType.tetrahedron, 2),
    "PY5": (dolfinx.mesh.CellType.pyramid, 1),
    "P13": (dolfinx.mesh.CellType.pyramid, 2),
    "PE6": (dolfinx.mesh.CellType.prism, 1),
    "P15": (dolfinx.mesh.CellType.prism, 2),
    "P18": (dolfinx.mesh.CellType.prism, 2),
    "HE8": (dolfinx.mesh.CellType.hexahedron, 1),
    "H20": (dolfinx.mesh.CellType.hexahedron, 2),
    "H27": (dolfinx.mesh.CellType.hexahedron, 2),
}

# Lagrange layouts: ``gmsh[k] = med[_med_to_gmsh[med_type][k]]``, from
# ``med2mshNodeIndex`` (and ``msh2medNodeIndex`` for the symmetric types) in Gmsh,
# https://github.com/live-clones/gmsh/blob/b4e84a43d02f97bc50ef37ed21e5e97096d072be/src/geo/GModelIO_MED.cpp#L113-L233
# composed with ``dolfinx.cpp.io.perm_gmsh`` to get DOLFINx order.
_med_to_gmsh = {
    "SE2": [0, 1],
    "SE3": [0, 1, 2],
    "TR3": [0, 1, 2],
    "TR6": [0, 1, 2, 3, 4, 5],
    "QU4": [0, 1, 2, 3],
    "QU9": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "TE4": [0, 2, 1, 3],
    "T10": [0, 2, 1, 3, 6, 5, 4, 7, 8, 9],
    "PY5": [0, 3, 2, 1, 4],
    "PE6": [0, 2, 1, 3, 5, 4],
    "P18": [0, 2, 1, 3, 5, 4, 8, 6, 12, 7, 14, 13, 11, 9, 10, 17, 15, 16],
    "HE8": [0, 3, 2, 1, 4, 7, 6, 5],
    "H27": [0, 3, 2, 1, 4, 7, 6, 5, 11, 8, 16, 10, 19, 9, 18, 17, 15, 12, 14, 13]
    + [20, 24, 21, 23, 22, 25, 26],
}

# Serendipity layouts (number of nodes), with the same node order in MED and VTK
_med_serendipity = {"QU8": 8, "H20": 20}


def geometry_layout(med_type: str) -> tuple[dolfinx.mesh.CellType, int, bool]:
    """Get the coordinate element layout of a MED element type.

    Args:
        med_type: MED element type, e.g. ``"T10"``

    Returns:
        Tuple containing:
            - The DOLFINx cell type
            - The degree of the coordinate element
            - True if the coordinate element is the nodal serendipity element
              (``QU8``, ``H20``), False if it is Lagrange
    """
    cell_type, degree = _med_cell_types[med_type]
    if med_type not in _med_to_gmsh and med_type not in _med_serendipity:
        raise NotImplementedError(
            f"MED element type {med_type} has no Lagrange or nodal serendipity layout."
        )
    return cell_type, degree, med_type in _med_serendipity


def get_med_to_dolfinx_permutation(med_type: str) -> npt.NDArray[np.int64]:
    """Get the permutation ``p`` with ``dolfinx[i] = med[p[i]]`` for a cell's nodes."""
    cell_type, _, serendipity = geometry_layout(med_type)
    if serendipity:
        return np.asarray(dolfinx.cpp.io.perm_vtk(cell_type, _med_serendipity[med_type]))
    med_to_gmsh = np.asarray(_med_to_gmsh[med_type])
    return med_to_gmsh[dolfinx.cpp.io.perm_gmsh(cell_type, len(med_to_gmsh))]


def get_mesh_step_and_families(
    h5file: h5py.File,
    filename: Path | str,
    backend_args: dict[str, Any] | None,
    time: str | float | None,
) -> tuple[h5py.Group, h5py.Group]:
    """Get the groups holding the nodes, connectivity and families of a mesh.

    In DOLFINx terms, ``NOE`` (*nœuds*) holds the geometry nodes, i.e. the coordinates
    in ``mesh.geometry.x``, and ``MAI`` (*mailles*) the connectivity: one block per
    element type, listing the nodes of each cell, facet, edge or point element.

    A time-dependent mesh has one group per computation step, named by its step and
    iteration numbers so that the names sort in time order. A step only stores data
    that changed, and links to the earlier step for the rest.

    Args:
        h5file: The MED file
        filename: Path to the file, used in error messages
        backend_args: Arguments to backend. ``"name"`` selects the mesh; if it is not
            given and the file holds a single mesh, that mesh is used.
        time: Time of the step to read. Ignored for a mesh with a single step, and
            ``None`` selects the first step.

    Returns:
        Tuple containing:
            - The step group, holding ``NOE`` and ``MAI``
            - The families of the mesh (``FAS/<mesh>``)
    """
    if "ENS_MAA" not in h5file.keys():
        raise ValueError(f"Could not find a set of meshes (ENS_MAA) in {filename}.")
    meshes = sorted(h5file["ENS_MAA"].keys())
    name = get_mesh_name(backend_args)
    if name not in meshes:
        if name == DEFAULT_MESH_NAME and len(meshes) == 1:
            name = meshes[0]
        else:
            raise KeyError(f"Could not find mesh '{name}' in {filename}. Available: {meshes}")
    mesh = h5file["ENS_MAA"][name]
    # Mesh type 0 is unstructured, 1 is a structured grid without connectivity
    if mesh.attrs.get("TYP", 0) != 0:
        raise NotImplementedError(f"Structured MED meshes are not supported ('{name}').")
    if "FAS" not in h5file.keys() or name not in h5file["FAS"].keys():
        raise ValueError(f"Could not find the families (FAS) of mesh '{name}' in {filename}.")

    if "NOE" in mesh.keys():
        step = mesh
    else:
        steps = [mesh[key] for key in sorted(mesh.keys())]
        if len(steps) == 1 or time is None:
            step = steps[0]
        else:
            times = np.array([float(s.attrs["PDT"]) for s in steps])
            match = np.flatnonzero(np.isclose(times, float(time)))
            if len(match) != 1:
                raise KeyError(
                    f"Expected one step of mesh '{name}' at time {time} in {filename}, found "
                    f"{len(match)}. Available times: {times.tolist()}"
                )
            step = steps[match[0]]
    if unsupported := sorted(set(step["MAI"].keys()) - _med_cell_types.keys()):
        raise NotImplementedError(f"MED element types {unsupported} are not supported.")
    return step, h5file["FAS"][name]


def read_entities(
    element_blocks: h5py.Group, dim: int, comm: MPI.Comm
) -> tuple[str, list[int], npt.NDArray[np.int64]] | None:
    """Read the entities of one topological dimension, distributed over all processes.

    Args:
        element_blocks: The group holding one connectivity block per element type
            (``MAI`` in the MED file)
        dim: Topological dimension of the entities
        comm: MPI communicator to distribute the entities over

    Returns:
        ``None`` if the mesh has no entities of dimension ``dim``, otherwise a tuple
        containing:
            - The MED element type of the entities, e.g. ``"TR3"``
            - The range ``[start, end)`` of entities read on this process
            - Their connectivity as global input node indices in DOLFINx node order,
              of shape ``(end - start, num_nodes_per_entity)``
    """
    blocks = sorted(
        key
        for key in element_blocks.keys()
        if dolfinx.mesh.cell_dim(_med_cell_types[key][0]) == dim
    )
    if len(blocks) == 0:
        return None
    if len(blocks) > 1:
        raise NotImplementedError(
            f"Mixed element types {blocks} of dimension {dim} are not supported."
        )
    med_type = blocks[0]
    block = element_blocks[med_type]

    # Read a contiguous chunk of the entities on each process
    num_entities_global = num_entities(block)
    local_range = compute_local_range(comm, num_entities_global)
    nodes = read_component_major_rows(block["NOD"], num_entities_global, local_range)

    # MED node indices are 1-based
    connectivity = nodes.astype(np.int64) - 1

    # Permute the nodes of each entity from MED to DOLFINx ordering
    perm = get_med_to_dolfinx_permutation(med_type)
    if connectivity.shape[1] != len(perm):
        raise ValueError(f"Expected {len(perm)} nodes per {med_type}, got {connectivity.shape[1]}.")
    return med_type, local_range, connectivity[:, perm]


def read_entity_families(block: h5py.Group, local_range: list[int]) -> npt.NDArray[np.int32]:
    """Read the family id of each entity of one MED element type on this process.

    Each entity has exactly one family, so this is a tag value per entity. Which
    groups a family belongs to is read with :func:`read_family_groups`.

    Args:
        block: Connectivity block of the element type (``MAI/<med_type>``)
        local_range: Range ``[start, end)`` of entities to read on this process

    Returns:
        Family ids of shape ``(end - start,)``. A block without ``FAM`` has every
        entity in family 0, i.e. in no group.
    """
    if "FAM" in block.keys():
        return block["FAM"][slice(*local_range)].astype(np.int32)
    return np.zeros(local_range[1] - local_range[0], dtype=np.int32)


def num_entities(block: h5py.Group) -> int:
    """Get the global number of entities of one MED element type.

    Args:
        block: Connectivity block of the element type (``MAI/<med_type>``)

    Returns:
        The number of entities, stored as the ``NBR`` attribute of the connectivity
    """
    return int(block["NOD"].attrs["NBR"])


def read_family_groups(
    families: h5py.Group, kind: Literal["ELEME", "NOEUD"]
) -> dict[int, list[str]]:
    """Read which named groups each family of a mesh belongs to.

    A family is one combination of groups, so one family can belong to several
    groups. The family of each entity is read with :func:`read_entity_families`.

    Args:
        families: The families of the mesh (``FAS/<mesh>``)
        kind: ``"ELEME"`` (*éléments*) for element families, ``"NOEUD"`` (*nœuds*)
            for node families

    Returns:
        Map from family id to the names of its groups. Empty if the mesh has no
        groups of this kind, in which case MED does not write ``kind``.
    """
    family_groups: dict[int, list[str]] = {}
    if kind not in families.keys():
        return family_groups
    for family in families[kind].values():
        groups = []
        if "GRO" in family.keys():
            for row in family["GRO"]["NOM"][()]:
                raw = np.asarray(row).astype(np.uint8).tobytes()
                groups.append(raw.split(b"\x00")[0].decode().rstrip())
        family_groups[int(family.attrs["NUM"])] = groups
    return family_groups


def read_groups(
    filename: Path | str, comm: MPI.Comm, backend_args: dict[str, Any] | None = None
) -> dict[str, list[int]]:
    """Read the element groups of a MED mesh and the families that make them up.

    Use this to find the group names accepted by :func:`read_meshtags_data`, or to
    decode the (negated) family ids returned for ``"cell"`` and ``"facet"``.

    Args:
        filename: Path to file
        comm: MPI communicator used in storage
        backend_args: Arguments to backend, ``"name"`` selects the mesh

    Returns:
        Map from group name to the negated family ids that belong to the group
    """
    check_file_exists(filename)
    with h5pyfile(filename, "r", comm=comm) as h5file:
        _, families = get_mesh_step_and_families(h5file, filename, backend_args, None)
        family_groups = read_family_groups(families, "ELEME")
    groups: dict[str, list[int]] = {}
    for family, names in sorted(family_groups.items(), reverse=True):
        for group in names:
            groups.setdefault(group, []).append(-family)
    return groups


def get_default_backend_args(arguments: dict[str, Any] | None) -> dict[str, Any]:
    args = arguments or {}
    return args


def write_attributes(
    filename: Path | str,
    comm: MPI.Comm,
    name: str,
    attributes: dict[str, np.ndarray],
    backend_args: dict[str, Any] | None = None,
):
    """Write attributes to file.

    Args:
        filename: Path to file to write to
        comm: MPI communicator used in storage
        name: Name of the attributes
        attributes: Dictionary of attributes to write to file
        backend_args: Arguments to backend
    """
    raise NotImplementedError("The MED backend cannot write attributes.")


def read_attributes(
    filename: Path | str,
    comm: MPI.Comm,
    name: str,
    backend_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read attributes from file.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        name: Name of the attributes
        backend_args: Arguments to backend
    Returns:
        The attributes
    """
    raise NotImplementedError("The MED backend cannot read attributes.")


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
    raise NotImplementedError("The MED backend cannot make checkpoints.")


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
    raise NotImplementedError("The MED backend cannot read HDF5 arrays.")


def read_timestamps(
    filename: Path | str,
    comm: MPI.Comm,
    function_name: str,
    backend_args: dict[str, Any] | None = None,
) -> npt.NDArray[np.float64 | str]:  # type: ignore[type-var]
    """Read time-stamps from a checkpoint file.

    Args:
        filename: Path to file
        comm: MPI communicator
        function_name: Name of the function to read time-stamps for
        backend_args: Arguments for backend, for instance file type.

    Returns:
        The time-stamps
    """
    raise NotImplementedError("The MED backend cannot read timestamps.")


def write_mesh(
    filename: Path | str,
    comm: MPI.Comm,
    mesh: MeshData,
    backend_args: dict[str, Any] | None = None,
    mode: FileMode = FileMode.write,
    time: float = 0.0,
):
    """Write a mesh to file.

    Args:
        filename: Path to file to write to.
        mesh: Internal data structure for the mesh data to save to file
        comm: MPI communicator used in storage
        backend_args: Arguments to backend
        mode: Mode to use (write or append)
        time: Time stamp
    """
    raise NotImplementedError("The MED backend cannot write meshes.")


def read_mesh_data(
    filename: Path | str,
    comm: MPI.Comm,
    time: str | float | None = 0.0,
    read_from_partition: bool = False,
    backend_args: dict[str, Any] | None = None,
) -> ReadMeshData:
    """Read mesh data from a MED file.

    Only cells of the highest topological dimension are read; these must all be of
    the same type. Lagrange layouts up to degree 2 (e.g. ``TR6``, ``T10``, ``H27``) and
    the serendipity ``QU8`` and ``H20`` are supported. Node and cell data is read in
    contiguous chunks on each process.

    Args:
        filename: Path to input file
        comm: The MPI communciator to distribute the mesh over
        time: Time of the step to read from a time-dependent mesh. Ignored for a mesh
            with a single step, and ``None`` selects the first step.
        read_from_partition: Read mesh with partition from file (not supported)
        backend_args: Arguments to backend, ``"name"`` selects the mesh
    Returns:
        The mesh topology, geometry, UFL domain and partition function
    """
    if read_from_partition:
        raise NotImplementedError("The MED backend cannot read partitioning data.")
    check_file_exists(filename)
    with h5pyfile(filename, "r", comm=comm) as h5file:
        step, _ = get_mesh_step_and_families(h5file, filename, backend_args, time)
        coo = step["NOE"]["COO"]
        num_nodes = int(coo.attrs["NBR"])
        x = read_component_major_rows(coo, num_nodes, compute_local_range(comm, num_nodes))

        element_blocks = step["MAI"]
        tdim = max(dolfinx.mesh.cell_dim(_med_cell_types[key][0]) for key in element_blocks.keys())
        entities = read_entities(element_blocks, tdim, comm)
        assert entities is not None
        med_type, _, cells = entities
        cell_type, degree, serendipity = geometry_layout(med_type)
    return ReadMeshData(
        cells=cells,
        cell_type=dolfinx.mesh.to_string(cell_type),
        x=x,
        lvar=int(basix.LagrangeVariant.equispaced),
        degree=degree,
        partition_graph=None,
        serendipity=serendipity,
    )


def write_meshtags(
    filename: str | Path,
    comm: MPI.Comm,
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
    raise NotImplementedError("The MED backend cannot write meshtags.")


def read_meshtags_data(
    filename: str | Path,
    comm: MPI.Comm,
    name: str,
    backend_args: dict[str, Any] | None = None,
) -> MeshTagsData:
    """Read mesh tags from a MED file.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        name: ``"cell"`` or ``"facet"`` for the entities of the highest topological
            dimension or one below, otherwise the name of a MED group (see
            :func:`read_groups`)
        backend_args: Arguments to backend, ``"name"`` selects the mesh and ``"time"`` the
            step of a time-dependent mesh (the first step if not given)

    Returns:
        Internal data structure for the mesh tags read from file. For ``"cell"`` and
        ``"facet"``, each entity is tagged with its negated MED family id, and entities
        without a family (id 0) are skipped. For a group, its entities are tagged with
        ``1``.
    """
    check_file_exists(filename)
    with h5pyfile(filename, "r", comm=comm) as h5file:
        time = None if backend_args is None else backend_args.get("time")
        step, families = get_mesh_step_and_families(h5file, filename, backend_args, time)
        element_blocks = step["MAI"]
        keys = sorted(element_blocks.keys())
        tdim = max(dolfinx.mesh.cell_dim(_med_cell_types[key][0]) for key in keys)
        group_families: list[int] | None = None
        if name in ("cell", "facet"):
            dim = tdim if name == "cell" else tdim - 1
        else:
            family_groups = read_family_groups(families, "ELEME")
            group_families = [f for f, groups in family_groups.items() if name in groups]
            if len(group_families) == 0:
                nodes = read_family_groups(families, "NOEUD")
                if any(name in groups for groups in nodes.values()):
                    raise NotImplementedError(f"Group '{name}' only holds nodes.")
                available = sorted({g for groups in family_groups.values() for g in groups})
                raise KeyError(
                    f"Could not find group '{name}' in {filename}. Available: {available}"
                )

            # Find the blocks holding entities of the group. FAS does not record which
            # element type a family is used on, and each process only reads its chunk
            # of every block, so the reduction is needed for all processes to agree on
            # the blocks and dimension before the collective meshtag creation.
            in_block = np.zeros(len(keys), dtype=np.int32)
            for i, key in enumerate(keys):
                r = compute_local_range(comm, num_entities(element_blocks[key]))
                in_block[i] = np.isin(
                    read_entity_families(element_blocks[key], r), group_families
                ).any()
            comm.Allreduce(MPI.IN_PLACE, in_block, op=MPI.MAX)
            blocks = [key for key, found in zip(keys, in_block) if found]
            dims = {dolfinx.mesh.cell_dim(_med_cell_types[key][0]) for key in blocks}
            if len(dims) != 1:
                raise RuntimeError(
                    f"Group '{name}' must hold entities of a single dimension, got {dims}."
                )
            dim = dims.pop()

        entities = read_entities(element_blocks, dim, comm)
        if entities is None:
            # No entities of this dimension; read_meshtags sets the number of nodes
            return MeshTagsData(
                name=name,
                values=np.zeros(0, dtype=np.int32),
                indices=np.zeros((0, 0), dtype=np.int64),
                dim=dim,
            )
        med_type, local_range, connectivity = entities
        family = read_entity_families(element_blocks[med_type], local_range)

    if group_families is None:
        keep = family != 0
        values = -family[keep]
    else:
        keep = np.isin(family, group_families)
        values = np.ones(np.count_nonzero(keep), dtype=np.int32)
    return MeshTagsData(name=name, values=values, indices=connectivity[keep], dim=dim)


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
    raise NotImplementedError("The MED backend cannot read dofmap.")


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
    raise NotImplementedError("The MED backend cannot read dofs.")


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
    raise NotImplementedError("The MED backend cannot read cell perms.")


def write_function(
    filename: str | Path,
    comm: MPI.Comm,
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
    raise NotImplementedError("The MED backend cannot write functions.")


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
    raise NotImplementedError("The MED backend cannot read legacy meshes.")


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
       Data local to process (contiguous, no mpi comm) and local start range
    """
    raise NotImplementedError("The MED backend cannot read point data.")


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
    raise NotImplementedError("The MED backend cannot read cell data.")


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
    raise NotImplementedError("The MED backend cannot read function names.")


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
        time: Time-stamp for data.
        mode: Append or write
        backend_args: The backend arguments
    """
    raise NotImplementedError("The MED backend cannot write data.")
