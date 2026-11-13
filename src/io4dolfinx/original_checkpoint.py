# Copyright (C) 2024 Jørgen Schartum Dokken
#
# This file is part of io4dolfinx
#
# SPDX-License-Identifier:    MIT

from __future__ import annotations

import logging
import typing

from mpi4py import MPI

import dolfinx
import fenicsx_compat
import numpy as np

from .comm_helpers import send_to_input_order
from .structures import FunctionData, MeshData
from .utils import input_cell_index, unroll_dofmap

__all__ = ["create_original_mesh_data", "create_function_data_on_original_mesh"]
logger = logging.getLogger(__name__)


def create_original_mesh_data(mesh: dolfinx.mesh.Mesh) -> MeshData:
    """
    Store data locally on output process
    """
    assert isinstance(mesh.comm, MPI.Intracomm)

    # 1. Send the topology of each owned cell, in original node indices, to the process
    # owning its original cell index
    num_owned_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    num_cells_global = mesh.topology.index_map(mesh.topology.dim).size_global
    original_cell_index = input_cell_index(mesh)[:num_owned_cells]
    original_node_index = mesh.geometry.input_global_indices
    local_geometry_dofmap = fenicsx_compat.dofmap(mesh)[:num_owned_cells, :]
    global_geometry_dofmap = original_node_index[local_geometry_dofmap]
    (sorted_recv_dofmap,), local_cell_range = send_to_input_order(
        mesh.comm, original_cell_index, [global_geometry_dofmap], num_cells_global
    )

    # 2. Send the coordinates of each owned node to the process owning its original index
    gdim = mesh.geometry.dim
    num_owned_nodes = mesh.geometry.index_map().size_local
    num_nodes_global = mesh.geometry.index_map().size_global
    (geometry,), local_node_range = send_to_input_order(
        mesh.comm,
        original_node_index[:num_owned_nodes],
        [mesh.geometry.x[:num_owned_nodes, :gdim]],
        num_nodes_global,
    )
    cmap = fenicsx_compat.cmap(mesh)

    # NOTE: Could in theory store partitioning information, but would not work nicely
    # as one would need to read this data rather than the xdmffile.
    # NOTE: Local geometry type hint skip is only required on DOLFINX<0.10 where
    # proper `dolfinx.mesh.Geometry` wrapper doesn't exist
    return MeshData(
        local_geometry=geometry,  # type: ignore[arg-type]
        local_geometry_pos=local_node_range,
        num_nodes_global=num_nodes_global,
        local_topology=sorted_recv_dofmap,
        local_topology_pos=local_cell_range,
        num_cells_global=num_cells_global,
        cell_type=mesh.topology.cell_name(),
        degree=cmap.degree,
        lagrange_variant=cmap.variant,
        store_partition=False,
        partition_processes=None,
        ownership_array=None,
        ownership_offset=None,
        partition_range=None,
        partition_global=None,
    )


def create_function_data_on_original_mesh(
    u: dolfinx.fem.Function, name: typing.Optional[str] = None
) -> FunctionData:
    """
    Create data object to save with ADIOS2
    """
    mesh = u.function_space.mesh
    assert isinstance(mesh.comm, MPI.Intracomm)
    num_owned_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    num_cells_global = mesh.topology.index_map(mesh.topology.dim).size_global
    original_cell_index = input_cell_index(mesh)[:num_owned_cells]

    # Send the permutation info and the dofs, as global dof indices, of each owned cell
    # to the process owning its original cell index. The dof values themselves stay in
    # the DOLFINx ordering.
    dofmap = u.function_space.dofmap
    index_map_bs = dofmap.index_map_bs
    unrolled_dofmap = unroll_dofmap(dofmap.list[:num_owned_cells, :], dofmap.bs)
    dmap_loc = (unrolled_dofmap // index_map_bs).reshape(-1)
    dmap_rem = (unrolled_dofmap % index_map_bs).reshape(-1)
    imap_global = dofmap.index_map.local_to_global(dmap_loc)
    num_dofs_per_cell = unrolled_dofmap.shape[1]
    dofmap_global = (imap_global * index_map_bs + dmap_rem).reshape(-1, num_dofs_per_cell)
    cell_permutations = fenicsx_compat.cell_permutation_info(mesh.topology)[:num_owned_cells]
    (cell_permutation_info, cell_dofs), local_cell_range = send_to_input_order(
        mesh.comm, original_cell_index, [cell_permutations, dofmap_global], num_cells_global
    )
    final_dofmap = cell_dofs.reshape(-1)

    # Get offsets of dofmap
    num_cells_local = local_cell_range[1] - local_cell_range[0]
    num_dofs_local_dmap = num_cells_local * num_dofs_per_cell
    dofmap_imap = fenicsx_compat.create_index_map(
        mesh.comm,
        num_dofs_local_dmap,
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int32),
    )
    local_dofmap_offsets = np.arange(num_cells_local + 1, dtype=np.int64)
    local_dofmap_offsets[:] *= num_dofs_per_cell
    local_dofmap_offsets[:] += dofmap_imap.local_range[0]

    num_dofs_local = dofmap.index_map.size_local * dofmap.index_map_bs
    num_dofs_global = dofmap.index_map.size_global * dofmap.index_map_bs
    dof_start, dof_end = (int(i) * dofmap.index_map_bs for i in dofmap.index_map.local_range)
    func_name = name if name is not None else u.name
    return FunctionData(
        cell_permutations=cell_permutation_info,
        local_cell_range=local_cell_range,
        num_cells_global=num_cells_global,
        dofmap_array=final_dofmap,
        dofmap_offsets=local_dofmap_offsets,
        values=u.x.array[:num_dofs_local].copy(),
        dof_range=(dof_start, dof_end),
        num_dofs_global=num_dofs_global,
        dofmap_range=dofmap_imap.local_range,
        global_dofs_in_dofmap=dofmap_imap.size_global,
        name=func_name,
    )
