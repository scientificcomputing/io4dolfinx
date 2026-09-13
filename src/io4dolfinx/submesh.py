# Copyright (C) 2026 Jørgen Schartum Dokken
#
# This file is part of io4dolfinx
#
# SPDX-License-Identifier:    MIT

"""Checkpointing of submeshes.

A :func:`dolfinx.mesh.create_submesh` submesh cannot be checkpointed the way an
ordinary mesh is. Two properties of the DOLFINx construction get in the way:

1. For a submesh of co-dimension greater than zero, a cell of the submesh is an
   *entity* of the parent, and the vertices of an entity are ordered by their
   current global vertex index. Re-partitioning the parent renumbers the
   vertices, so the same submesh cell comes back with its vertices in a
   different order and the degrees of freedom inside it land in different
   positions.
2. :func:`dolfinx.mesh.create_submesh` builds its vertex map allowing owner
   changes, so a process can own a vertex -- and a dof -- that is only incident
   to cells it ghosts. Such a dof has no position in any owned cell, which is
   what :func:`io4dolfinx.utils.compute_dofmap_pos` needs.

So a function is never read onto a re-derived submesh. Instead the submesh is
stored as an ordinary, independent mesh (whose cells keep the stored node order
and whose ownership is the ordinary one), read back as such, and the data is
then transferred to a mesh re-derived from the parent when the caller needs the
entity maps that make mixed-dimensional assembly possible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mpi4py import MPI

import dolfinx
import numpy as np
import numpy.typing as npt
from dolfinx import cpp as _cpp

from .backends import FileMode, get_backend
from .checkpointing import _with_mesh_name, write_mesh
from .comm_helpers import exchange_to_owners, numpy_to_mpi
from .structures import MeshTagsData
from .utils import check_file_exists, compute_local_range, index_owner

__all__ = [
    "SubmeshCheckpoint",
    "read_submesh",
    "transfer_submesh_function",
    "write_submesh",
]

logger = logging.getLogger(__name__)


def _link_name(mesh_name: str) -> str:
    """Name of the meshtag holding the parent link of the submesh ``mesh_name``.

    The link is a tag on the *parent* mesh, so it is stored in the parent's
    namespace and must not collide with a tag a user wrote.
    """
    return f"__submesh_link_{mesh_name}"


@dataclass
class SubmeshCheckpoint:
    """A submesh re-derived from a parent mesh that was read from a checkpoint."""

    #: The submesh, derived from the parent with :func:`dolfinx.mesh.create_submesh`
    submesh: dolfinx.mesh.Mesh
    #: Map from cells of :attr:`submesh` to entities of the parent
    cell_map: Any
    #: Map from vertices of :attr:`submesh` to vertices of the parent
    vertex_map: Any
    #: Map from geometry nodes of :attr:`submesh` to nodes of the parent
    node_map: npt.NDArray[np.int32]
    #: Parent entities that make up the submesh (local indices, owned and ghost)
    parent_entities: npt.NDArray[np.int32]
    #: Index the cell had in the stored submesh, per local cell (owned and ghost)
    stored_cells: npt.NDArray[np.int64]


def write_submesh(
    filename: Path | str,
    submesh: dolfinx.mesh.Mesh,
    parent: dolfinx.mesh.Mesh,
    cell_map: Any,
    mesh_name: str,
    parent_mesh_name: str | None = None,
    parent_filename: Path | str | None = None,
    time: float = 0.0,
    mode: FileMode = FileMode.append,
    backend_args: dict[str, Any] | None = None,
    backend: str | None = None,
):
    """Write a submesh, and the link that ties it back to its parent.

    The submesh is stored as an ordinary independent mesh under ``mesh_name``, so
    :func:`io4dolfinx.read_mesh` and :func:`io4dolfinx.read_function` read it
    back with no special handling. Alongside it goes a *parent link*: for every
    stored submesh cell, the parent geometry nodes of the entity it came from.
    That is what lets :func:`read_submesh` find the same entities in a
    re-partitioned parent.

    The link tags entities of the *parent*, so it is stored in the parent's
    namespace, in ``parent_filename``. Leave that unset to keep everything in one
    checkpoint, or point it at the parent's own file to keep the two apart; the
    submesh always goes to ``filename``. Either way the parent must already have
    been written there with :func:`io4dolfinx.write_mesh`.

    Args:
        filename: Path to write the submesh to
        submesh: The submesh, as returned by :func:`dolfinx.mesh.create_submesh`
        parent: The mesh the submesh was derived from
        cell_map: Map from submesh cells to parent entities, as returned by
            :func:`dolfinx.mesh.create_submesh`
        mesh_name: Name to store the submesh under
        parent_mesh_name: Name the parent mesh is stored under
        parent_filename: File holding the parent mesh, if not ``filename``
        time: Time stamp associated with the submesh geometry
        mode: Whether to create the file or append to it
        backend_args: Arguments for the backend
        backend: Which backend to use
    """
    logger.debug(f"Writing submesh '{mesh_name}' to {filename}")
    dim = submesh.topology.dim

    # 1. The submesh itself, as a perfectly ordinary mesh.
    write_mesh(
        Path(filename),
        submesh,
        mode=mode,
        time=time,
        backend_args=backend_args,
        backend=backend,
        mesh_name=mesh_name,
    )

    # 2. The parent link. Structurally this is a meshtag on the parent: the
    #    tagged entities are the parent entities that became submesh cells, and
    #    each value is the index that cell has in the stored submesh.
    cell_imap = submesh.topology.index_map(dim)
    num_owned = cell_imap.size_local
    parent_entities = cell_map.sub_topology_to_topology(np.arange(num_owned, dtype=np.int32), False)
    values = np.arange(*cell_imap.local_range, dtype=np.int64)

    parent.topology.create_connectivity(dim, parent.topology.dim)
    parent.topology.create_connectivity(0, parent.topology.dim)
    entities_to_geometry = dolfinx.cpp.mesh.entities_to_geometry(
        parent._cpp_object, dim, parent_entities, False
    )
    indices = (
        parent.geometry.index_map()
        .local_to_global(entities_to_geometry.reshape(-1))
        .reshape(entities_to_geometry.shape)
    )

    tag_data = MeshTagsData(
        values=values,
        num_entities_global=cell_imap.size_global,
        num_dofs_per_entity=entities_to_geometry.shape[1],
        indices=indices,
        name=_link_name(mesh_name),
        local_start=cell_imap.local_range[0],
        dim=dim,
        cell_type=submesh.topology.cell_name(),
    )
    backend_cls = get_backend(backend)
    parent_args = backend_cls.get_default_backend_args(
        _with_mesh_name(backend_args, parent_mesh_name)
    )
    link_file = filename if parent_filename is None else parent_filename
    backend_cls.write_meshtags(link_file, parent.comm, tag_data, backend_args=parent_args)


def read_submesh(
    filename: Path | str,
    parent: dolfinx.mesh.Mesh,
    mesh_name: str,
    parent_mesh_name: str | None = None,
    backend_args: dict[str, Any] | None = None,
    backend: str | None = None,
) -> SubmeshCheckpoint:
    """Re-derive a stored submesh from a parent mesh that has been read back.

    The result is a genuine :func:`dolfinx.mesh.create_submesh` submesh, so its
    entity maps are accepted by :func:`dolfinx.fem.form` for mixed-dimensional
    assembly against ``parent``.

    It does **not** carry the checkpointed data. Read that onto the standalone
    submesh with :func:`io4dolfinx.read_mesh` and :func:`io4dolfinx.read_function`,
    then move it across with :func:`transfer_submesh_function`.

    Args:
        filename: File holding the parent mesh and the parent link, i.e. whatever
            was passed as ``parent_filename`` to :func:`write_submesh`
        parent: The parent mesh, as read back from the checkpoint
        mesh_name: Name the submesh was stored under
        parent_mesh_name: Name the parent mesh is stored under
        backend_args: Arguments for the backend
        backend: Which backend to use

    Returns:
        The re-derived submesh and the maps relating it to ``parent``.
    """
    logger.debug(f"Reading submesh '{mesh_name}' from {filename}")
    check_file_exists(filename)
    backend_cls = get_backend(backend)
    parent_args = backend_cls.get_default_backend_args(
        _with_mesh_name(backend_args, parent_mesh_name)
    )
    # Stored cell indices are global, so they must survive as int64.
    parent_args["values_dtype"] = np.int64
    data = backend_cls.read_meshtags_data(filename, parent.comm, _link_name(mesh_name), parent_args)
    dim = int(data.dim)

    local_entities, local_values = dolfinx.io.distribute_entity_data(
        parent, dim, data.indices, data.values.astype(np.int64)
    )
    parent.topology.create_connectivity(dim, 0)
    parent.topology.create_connectivity(dim, parent.topology.dim)
    adj = dolfinx.graph.adjacencylist(local_entities)
    tags = dolfinx.mesh.meshtags_from_entities(
        parent, dim, adj, np.asarray(local_values, dtype=np.int64)
    )

    # `distribute_entity_data` only reports entities to processes that hold them;
    # a process ghosting an entity may not be among them. Scattering through an
    # index map gives every process the owner's value for its ghosts, so the
    # submesh is built from a consistent set of entities everywhere.
    entity_map = parent.topology.index_map(dim)
    marker = dolfinx.la.vector(entity_map, dtype=np.int64)
    marker.array[:] = -1
    marker.array[tags.indices] = tags.values
    marker.scatter_forward()
    entities = np.flatnonzero(marker.array >= 0).astype(np.int32)

    submesh, cell_map, vertex_map, node_map = dolfinx.mesh.create_submesh(parent, dim, entities)
    sub_imap = submesh.topology.index_map(submesh.topology.dim)
    num_cells = sub_imap.size_local + sub_imap.num_ghosts
    parent_entities = cell_map.sub_topology_to_topology(np.arange(num_cells, dtype=np.int32), False)
    stored_cells = marker.array[parent_entities].astype(np.int64)
    return SubmeshCheckpoint(
        submesh=submesh,
        cell_map=cell_map,
        vertex_map=vertex_map,
        node_map=node_map,
        parent_entities=parent_entities,
        stored_cells=stored_cells,
    )


def _point_ownership_data(
    comm: MPI.Intracomm,
    src_owner: npt.NDArray[np.int32],
    points: npt.NDArray[np.floating],
    source_cells: npt.NDArray[np.int32],
) -> Any:
    """Build a :class:`dolfinx.geometry.PointOwnershipData` from a known routing.

    :func:`dolfinx.fem.create_interpolation_data` would derive the same object by
    searching a bounding-box tree for each point, with a padding tolerance and an
    extrapolation fallback. Here the owning rank and cell of every point are
    already known exactly, so the routing is done directly and no geometric
    search -- and no tolerance -- is involved.

    The layout required of the result is dictated by ``fem::interpolate`` and
    ``impl::scatter_values``:

    * ``src_owner`` is per interpolation point of the receiving function, in the
      order :func:`dolfinx.cpp.fem.interpolation_coords` produced them.
    * ``dest_owners`` must be in **contiguous runs by rank**; ``scatter_values``
      reduces it with a non-sorting ``unique``, so interleaved ranks would
      silently build the wrong neighbourhood. :func:`exchange_to_owners` groups
      by ascending source rank, which satisfies this.
    * ``dest_points`` is flat row-major ``(n_recv, 3)`` and ``dest_cells`` indexes
      cells of the *sending* function's mesh on the receiving process.

    Args:
        comm: The MPI communicator both meshes live on
        src_owner: Rank holding the source cell of each point
        points: The points, shape ``(num_points, 3)``
        source_cells: Source-mesh local cell of each point, on ``src_owner``

    Returns:
        Interpolation data for :meth:`dolfinx.fem.Function.interpolate_nonmatching`.
    """
    (recv_points, recv_cells), sources, _, recv_counts, _ = exchange_to_owners(
        comm, src_owner, [points, source_cells]
    )

    # One entry per received point, naming the rank it came from. `sources` is
    # ascending and `exchange_to_owners` receives in that order, so the runs are
    # contiguous by construction -- which `scatter_values` requires.
    dest_owners = np.repeat(sources, recv_counts).astype(np.int32)

    dtype = np.dtype(points.dtype)
    cls = getattr(_cpp.geometry, f"PointOwnershipData_float{8 * dtype.itemsize}")
    return dolfinx.geometry.PointOwnershipData(
        cls(
            np.ascontiguousarray(src_owner, dtype=np.int32),
            dest_owners,
            np.ascontiguousarray(recv_points, dtype=dtype).reshape(-1),
            np.ascontiguousarray(recv_cells, dtype=np.int32),
        )
    )


def transfer_submesh_function(
    u_source: dolfinx.fem.Function,
    u_dest: dolfinx.fem.Function,
    stored_cells: npt.NDArray[np.int64],
):
    """Move data from a standalone submesh onto one re-derived from a parent.

    ``u_source`` lives on the submesh as it was stored -- read with
    :func:`io4dolfinx.read_mesh` and :func:`io4dolfinx.read_function` -- and
    ``u_dest`` on the submesh :func:`read_submesh` derived from the parent. The
    two meshes cover the same cells but number their vertices differently and are
    partitioned independently, so the transfer cannot be a copy.

    Every cell of ``u_dest``'s mesh has a known counterpart in ``u_source``'s, so
    each interpolation point is routed straight to the process and cell that can
    evaluate it, with no geometric point location. This is an evaluation followed
    by a scatter and a local interpolation: it assembles no form and needs no
    entity map between the two meshes, which would have no meaning as neither is
    derived from the other.

    Args:
        u_source: Function on the submesh as stored
        u_dest: Function to fill, on the submesh re-derived from the parent
        stored_cells: For each cell of ``u_dest``'s mesh (owned and ghost), the
            index that cell had in the stored submesh. From
            :attr:`SubmeshCheckpoint.stored_cells`.

    Raises:
        NotImplementedError: For H(div)/H(curl) spaces on a manifold submesh,
            where DOLFINx's interpolation would return wrong values without
            reporting an error.
    """
    V_dest = u_dest.function_space
    dest_mesh = V_dest.mesh
    source_mesh = u_source.function_space.mesh
    comm = dest_mesh.comm
    assert isinstance(comm, MPI.Intracomm)

    tdim = dest_mesh.topology.dim
    if tdim < dest_mesh.geometry.dim and V_dest.element.needs_dof_transformations:
        raise NotImplementedError(
            "Transferring an H(div)/H(curl) function onto a submesh with"
            f" tdim ({tdim}) < gdim ({dest_mesh.geometry.dim}) is not supported."
            " DOLFINx cannot reconcile the reference and physical value sizes of"
            " these families on a manifold, and the interpolation this transfer"
            " relies on does not report that: it returns silently with values that"
            " are wrong (measured: the L2 norm of an N1curl field on a facet"
            " submesh fell from 9.096 to 4.904). Writing and reading the standalone"
            " submesh is exact for these spaces -- use `read_mesh` and"
            " `read_function` with the submesh's `mesh_name` and stop there. Only"
            " re-deriving from the parent is unavailable."
        )

    sub_imap = dest_mesh.topology.index_map(tdim)
    num_cells = sub_imap.size_local + sub_imap.num_ghosts
    if len(stored_cells) != num_cells:
        raise ValueError(
            f"Expected one stored cell index per cell of the destination mesh"
            f" ({num_cells}), got {len(stored_cells)}."
        )
    cells = np.arange(num_cells, dtype=np.int32)

    # Interpolation points of the destination, grouped per cell.
    points = _cpp.fem.interpolation_coords(
        V_dest.element._cpp_object,  # type: ignore[arg-type]
        dest_mesh.geometry._cpp_object,  # type: ignore[arg-type]
        cells,
    )
    points = np.ascontiguousarray(points.T)  # (num_points, 3)
    points_per_cell = 0 if num_cells == 0 else points.shape[0] // num_cells

    # Which process holds each stored cell in the source mesh, and where.
    source_tdim = source_mesh.topology.dim
    source_imap = source_mesh.topology.index_map(source_tdim)
    owner_rank, owner_cell = _lookup_stored_cells(
        comm,
        np.asarray(source_mesh.topology.original_cell_index[: source_imap.size_local]),
        np.asarray(stored_cells, dtype=np.int64),
        source_imap.size_global,
    )
    if num_cells > 0 and (owner_rank < 0).any():
        missing = np.unique(stored_cells[owner_rank < 0])
        raise RuntimeError(
            f"{len(missing)} cells of the destination submesh have no counterpart in"
            f" the stored submesh (for instance {missing[:5]}). The two must describe"
            " the same submesh of the same parent."
        )

    interpolation_data = _point_ownership_data(
        comm,
        np.repeat(owner_rank, points_per_cell),
        points,
        np.repeat(owner_cell, points_per_cell),
    )
    u_dest.interpolate_nonmatching(u_source, cells, interpolation_data=interpolation_data)
    u_dest.x.scatter_forward()


def _lookup_stored_cells(
    comm: MPI.Intracomm,
    owned_stored: npt.NDArray[np.int64],
    queried_stored: npt.NDArray[np.int64],
    num_cells_global: int,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """Find which process holds each stored cell, and its local index there.

    The two meshes are partitioned independently, so neither side knows where the
    other put a given cell. Both sides talk to a third: the process that
    :func:`index_owner` assigns the cell in the equal-split layout acts as a
    directory. Owners publish into it, queriers read out of it, each in one
    neighbourhood exchange, so no process ever holds a global table.

    Args:
        comm: The MPI communicator
        owned_stored: Stored index of each cell this process owns in the source
            mesh; position in the array is the local cell index
        queried_stored: Stored indices this process wants to locate
        num_cells_global: Number of cells in the stored mesh

    Returns:
        ``(rank, local_cell)`` aligned with ``queried_stored``; ``-1`` where the
        stored cell was never published.
    """
    directory_range = compute_local_range(comm, num_cells_global)
    directory_size = int(directory_range[1] - directory_range[0])
    directory_rank = np.full(directory_size, -1, dtype=np.int32)
    directory_cell = np.full(directory_size, -1, dtype=np.int32)

    # Publish: each owner tells the directory where it keeps the cell.
    owned_stored = np.asarray(owned_stored, dtype=np.int64)
    local_cells = np.arange(len(owned_stored), dtype=np.int32)
    publish_to = (
        index_owner(comm, owned_stored, num_cells_global)
        if len(owned_stored)
        else np.empty(0, dtype=np.int32)
    )
    (pub_keys, pub_cells), pub_sources, _, pub_counts, _ = exchange_to_owners(
        comm, publish_to, [owned_stored, local_cells]
    )
    slots = (pub_keys - directory_range[0]).astype(np.int64)
    directory_rank[slots] = np.repeat(pub_sources, pub_counts)
    directory_cell[slots] = pub_cells

    # Query: ask the directory, and unpack the reply into the order asked.
    queried_stored = np.asarray(queried_stored, dtype=np.int64)
    query_to = (
        index_owner(comm, queried_stored, num_cells_global)
        if len(queried_stored)
        else np.empty(0, dtype=np.int32)
    )
    (inc_keys,), q_sources, q_send_counts, q_recv_counts, q_insert = exchange_to_owners(
        comm, query_to, [queried_stored]
    )
    inc_slots = (inc_keys - directory_range[0]).astype(np.int64)
    reply_rank = directory_rank[inc_slots]
    reply_cell = directory_cell[inc_slots]

    # Send the answers back along the reverse of the query graph.
    q_destinations = np.unique(query_to)
    reverse = comm.Create_dist_graph_adjacent(
        q_destinations.tolist(), q_sources.tolist(), reorder=False
    )
    packed_rank = np.zeros(len(queried_stored), dtype=np.int32)
    packed_cell = np.zeros(len(queried_stored), dtype=np.int32)
    reverse.Neighbor_alltoallv(
        [reply_rank, q_recv_counts, numpy_to_mpi[np.int32]],
        [packed_rank, q_send_counts, numpy_to_mpi[np.int32]],
    )
    reverse.Neighbor_alltoallv(
        [reply_cell, q_recv_counts, numpy_to_mpi[np.int32]],
        [packed_cell, q_send_counts, numpy_to_mpi[np.int32]],
    )
    reverse.Free()

    # `q_insert[i]` is where local entry `i` was packed into the outgoing buffer,
    # and the reply comes back in that same packed order, so read the answers
    # back out by gathering -- scattering here would apply the inverse
    # permutation and silently mix up which cell each answer belongs to.
    return packed_rank[q_insert], packed_cell[q_insert]
