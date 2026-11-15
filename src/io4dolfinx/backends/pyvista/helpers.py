"""
Helpers for the PyVista backend: reading a VTK grid with pyvista as arrays.

SPDX License identifier: MIT

Copyright: Jørgen S. Dokken and Simula Research Laboratory
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import basix
import dolfinx
import fenicsx_compat
import numpy as np

try:
    import pyvista
except ImportError:
    raise ModuleNotFoundError("The PyVista-backend requires pyvista")

from ...structures import ReadMeshData
from ...utils import as_rows
from ..vtk_cell_types import _arbitrary_lagrange_vtk, _cell_degree, _first_order_vtk

__all__ = ["Grid", "read_grid"]

# Cell types can be found at https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html
_quadratic_vtk = {
    21: "interval",
    22: "triangle",
    23: "quadrilateral",
    24: "tetrahedron",
    25: "hexahedron",
}

#: VTK pixels and voxels, the cells of image and rectilinear grids. They number their nodes
#: x fastest, as DOLFINx numbers the vertices of quadrilaterals and hexahedra.
_lexicographic_vtk = {8: "quadrilateral", 11: "hexahedron"}


@dataclass
class Grid:
    """A VTK grid of one cell type, as arrays."""

    #: The mesh, with the points of each cell in DOLFINx order. VTK points have three
    #: coordinates.
    mesh: ReadMeshData
    #: Point data, each of shape ``(num_points, num_components)``
    point_data: dict[str, np.ndarray]
    #: Cell data, each of shape ``(num_cells, num_components)``
    cell_data: dict[str, np.ndarray]


def read_grid(filename: Path | str) -> Grid:
    """Read a VTK grid with pyvista.

    Unstructured grids, and structured, rectilinear and image grids, are read, also as the
    single block of a multiblock data set (e.g. from a ``.pvd`` file).

    Args:
        filename: Path to the file

    Returns:
        The grid, as arrays

    Raises:
        NotImplementedError: For several blocks, several cell types, or an unsupported cell
            type.
    """
    data: pyvista.DataSet | pyvista.MultiBlock | None = pyvista.read(filename)
    if isinstance(data, pyvista.MultiBlock):
        fenicsx_compat.pyvista_allow_snake_case(pyvista)
        if data.number_of_blocks != 1:
            raise NotImplementedError(
                f"{filename} has {data.number_of_blocks} blocks; only one is supported."
            )
        data = data.get_block(0)
    if not isinstance(data, pyvista.DataSet):
        raise TypeError(f"Unknown data type {type(data)} in {filename}.")
    # Structured grids are cast to unstructured ones, keeping the order of points and cells
    if isinstance(data, pyvista.UnstructuredGrid):
        grid = data
    else:
        grid = data.cast_to_unstructured_grid()

    if len(vtk_types := np.unique(grid.celltypes)) != 1:
        raise NotImplementedError(f"{filename} has several cell types {vtk_types.tolist()}.")
    vtk_type = int(vtk_types[0])
    nodes = np.asarray(grid.cells_dict[vtk_types[0]], dtype=np.int64)
    if vtk_type in _lexicographic_vtk:
        cell_type, degree = _lexicographic_vtk[vtk_type], 1
        cells = nodes
    else:
        if vtk_type in _first_order_vtk:
            cell_type, degree = _first_order_vtk[vtk_type], 1
        elif vtk_type in _quadratic_vtk:
            cell_type, degree = _quadratic_vtk[vtk_type], 2
        elif vtk_type in _arbitrary_lagrange_vtk:
            cell_type = _arbitrary_lagrange_vtk[vtk_type]
            degree = _cell_degree(cell_type, nodes.shape[1])
        else:
            raise NotImplementedError(f"VTK cell type {vtk_type} is not supported.")
        cells = nodes[:, dolfinx.cpp.io.perm_vtk(dolfinx.mesh.to_type(cell_type), nodes.shape[1])]

    mesh = ReadMeshData(
        cells=cells,
        cell_type=cell_type,
        x=np.asarray(grid.points),
        lvar=int(basix.LagrangeVariant.equispaced),
        degree=degree,
    )
    return Grid(
        mesh=mesh,
        point_data={name: as_rows(grid.point_data[name]) for name in grid.point_data.keys()},
        cell_data={name: as_rows(grid.cell_data[name]) for name in grid.cell_data.keys()},
    )
