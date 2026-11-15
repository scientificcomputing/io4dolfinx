"""
Helpers for the Nastran backend: reading the elements and GRID IDs of a Nastran bulk data
file through Gmsh.

SPDX License identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import dolfinx
import gmsh
import numpy as np
import numpy.typing as npt
from dolfinx.io.gmsh import _gmsh_to_cells, cell_perm_array, extract_topology_and_markers

__all__ = [
    "GRID",
    "PROPERTY_TO_DIM",
    "ElementBlock",
    "read_elements",
    "get_single_cell_type",
    "get_vertex_nodes",
]

#: Name of the GRID IDs as point data and vertex tags
GRID = "GRID"

#: Topological dimension of the elements of each property card
PROPERTY_TO_DIM = {"PSOLID": 3, "PSHELL": 2}


class ElementBlock(NamedTuple):
    """The elements of one type in a Nastran file."""

    cell_type: str  #: DOLFINx cell type
    degree: int  #: Degree of the elements
    #: Element nodes as zero-based node indices in DOLFINx ordering, of shape
    #: ``(num_elements, num_nodes_per_element)``
    nodes: npt.NDArray[np.int64]
    pids: npt.NDArray[np.int32]  #: Property ID of each element


def read_elements(
    filename: Path | str,
) -> tuple[dict[int, list[ElementBlock]], npt.NDArray[np.int64]]:
    """Read the elements of a Nastran file into the current Gmsh model, by dimension.

    Gmsh keeps the GRID IDs as node tags, which :mod:`dolfinx.io.gmsh` requires to be
    ``1, ..., num_nodes``, so the nodes are renumbered: node ``i`` is the node with the
    ``i``-th smallest GRID ID.

    Args:
        filename: Path to the Nastran file

    Returns:
        Tuple containing:
            - Map from topological dimension to the element blocks of that dimension,
              one per element type, sorted by DOLFINx cell type
            - The GRID ID of each node
    """
    gmsh.merge(str(filename))
    grid_ids = np.sort(gmsh.model.mesh.getNodes()[0]).astype(np.int64)
    gmsh.model.mesh.renumberNodes(grid_ids, np.arange(1, len(grid_ids) + 1))
    # Gmsh puts each PID in an elementary entity; dolfinx.io.gmsh only reads physical groups
    for dim, tag in gmsh.model.getEntities():
        gmsh.model.addPhysicalGroup(dim=dim, tags=[tag], tag=tag)
    topologies, _ = extract_topology_and_markers(gmsh.model)
    if not topologies:
        raise ValueError(f"No elements found in {filename}.")
    elements: dict[int, list[ElementBlock]] = {}
    for gmsh_type, data in topologies.items():
        if gmsh_type not in _gmsh_to_cells:
            name = gmsh.model.mesh.getElementProperties(gmsh_type)[0]
            raise NotImplementedError(f"Unsupported element type '{name}' in {filename}.")
        cell_type, degree = _gmsh_to_cells[gmsh_type]
        nodes = data["topology"].astype(np.int64)
        perm = cell_perm_array(dolfinx.mesh.to_type(cell_type), nodes.shape[1])
        dim = dolfinx.mesh.cell_dim(dolfinx.mesh.to_type(cell_type))
        pids = data["cell_data"].astype(np.int32)
        elements.setdefault(dim, []).append(ElementBlock(cell_type, degree, nodes[:, perm], pids))
    for blocks in elements.values():
        blocks.sort(key=lambda block: dolfinx.mesh.to_type(block.cell_type).value)
    return elements, grid_ids


def get_single_cell_type(
    elements: dict[int, list[ElementBlock]], what: str, filename: Path | str
) -> ElementBlock:
    """The cells of a mesh, which must be of one type.

    Args:
        elements: The element blocks of the file, by dimension
        what: Description of the data for the error message
        filename: Path to the Nastran file

    Returns:
        The element block of the cells

    Raises:
        NotImplementedError: If the cells are of several types (a mixed mesh).
    """
    blocks = elements[max(elements)]
    if len(blocks) > 1:
        types = [block.cell_type for block in blocks]
        raise NotImplementedError(
            f"{what} is not supported on meshes with several cell types {types}, as in {filename}."
        )
    return blocks[0]


def get_vertex_nodes(cells: ElementBlock) -> npt.NDArray[np.int64]:
    """The nodes at the vertices of the cells, sorted.

    Higher-order elements also have nodes on edges, faces and in the interior, which are
    not vertices. In DOLFINx ordering, the vertex nodes come first.
    """
    num_vertices = dolfinx.cpp.mesh.cell_num_vertices(dolfinx.mesh.to_type(cells.cell_type))
    return np.unique(cells.nodes[:, :num_vertices])
