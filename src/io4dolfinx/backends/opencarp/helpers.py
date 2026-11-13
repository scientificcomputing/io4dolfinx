"""
Helpers for the openCARP backend: file names, markers, ``.elem`` and ``.lon`` files,
and the orientation of elements and facets.

SPDX License identifier: MIT

Copyright: Henrik Finsberg and Simula Research Laboratory
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ...ascii_helpers import read_count, read_rows, write_rows
from ...utils import match_rows, strip_suffix

__all__ = [
    "mesh_suffixes",
    "element_types",
    "cell_types",
    "element_tdim",
    "dolfinx_to_opencarp",
    "mirror_permutation",
    "element_faces",
    "face_element_types",
    "float_format",
    "direction_names",
    "mesh_file",
    "marker_file",
    "get_scale",
    "marker_names",
    "read_marker_index",
    "write_marker_index",
    "remove_marker_files",
    "marker_value",
    "read_elem_type",
    "read_elem",
    "write_elem",
    "orientation",
    "facet_normals",
    "oriented_facets",
    "sort_facets",
]

# Suffixes stripped from a filename to get the base name of the mesh files
mesh_suffixes = (".pts", ".elem", ".lon")

# openCARP element type of each supported DOLFINx cell type, and the reverse
element_types = {
    "triangle": "Tr",
    "quadrilateral": "Qd",
    "tetrahedron": "Tt",
    "hexahedron": "Hx",
}
cell_types = {element: cell for cell, element in element_types.items()}

# Topological dimension of each openCARP element type
element_tdim = {"Tr": 2, "Qd": 2, "Tt": 3, "Hx": 3}

# Node permutation ``p`` with ``opencarp[i] = dolfinx[p[i]]`` for each element type
dolfinx_to_opencarp = {
    "Tr": np.array([0, 1, 2]),
    "Qd": np.array([0, 1, 3, 2]),
    "Tt": np.array([0, 1, 2, 3]),
    "Hx": np.array([0, 1, 3, 2, 4, 6, 7, 5]),
}

# Node permutation mirroring an element in openCARP ordering, flipping its orientation
mirror_permutation = {
    "Tr": np.array([0, 2, 1]),
    "Qd": np.array([0, 3, 2, 1]),
    "Tt": np.array([0, 2, 1, 3]),
    "Hx": np.array([0, 3, 2, 1, 4, 7, 6, 5]),
}

# Local faces of 3D elements in openCARP ordering, with their nodes in cyclic order,
# and the openCARP element type of the faces
element_faces = {
    "Tt": np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]),
    "Hx": np.array(
        [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 7, 4], [1, 2, 6, 7], [2, 3, 5, 6], [3, 0, 4, 5]]
    ),
}
face_element_types = {"Tt": "Tr", "Hx": "Qd"}

# Format of floating point values, enough digits for µm coordinates of a heart
float_format = "%.10g"

# Names of the directions in ``.lon``, in column order
direction_names = ("fiber", "sheet")


def mesh_file(filename: Path | str, suffix: str) -> Path:
    """Get the path of one of the mesh files, e.g. ``<base>.elem``."""
    return Path(f"{strip_suffix(filename, mesh_suffixes)}{suffix}")


def marker_file(filename: Path | str, name: str, suffix: str) -> Path:
    """Get the path of the file of a marker, e.g. ``<base>_<name>.vtx``."""
    return Path(f"{strip_suffix(filename, mesh_suffixes)}_{name}{suffix}")


def get_scale(backend_args: dict[str, Any]) -> float:
    """Get the factor converting mesh coordinates to µm.

    It has no default, as a wrong unit gives a mesh that openCARP loads without
    complaint but simulates at the wrong size.
    """
    try:
        scale = float(backend_args["scale"])
    except KeyError as e:
        raise ValueError(
            "The openCARP backend needs backend_args['scale'], the factor converting mesh "
            "coordinates to µm (e.g. 1000.0 for a mesh in mm, 1.0 for a mesh in µm)."
        ) from e
    if not scale > 0:
        raise ValueError(f"backend_args['scale'] must be positive, got {scale}.")
    return scale


def marker_names(markers: dict[str, Any], dim: int) -> dict[int, str]:
    """Get the name of each marker value of entities of a given dimension.

    Args:
        markers: Map from name to value, or to ``(value, dim)``
        dim: Topological dimension of the entities

    Returns:
        Map from value to name. Markers with another dimension are left out.
    """
    names = {}
    for name, marker in markers.items():
        if isinstance(marker, (int, np.integer)):
            value, marker_dim = int(marker), None
        else:
            value = int(marker[0])
            marker_dim = int(marker[1]) if len(marker) > 1 else None
        if marker_dim is None or marker_dim == dim:
            names[value] = str(name)
    return names


def read_marker_index(filename: Path | str) -> dict[str, dict[str, Any]]:
    """Read the index of the markers written for a mesh, ``<base>.markers.json``.

    Returns:
        Map from marker name to its ``"value"``, ``"dim"`` and whether it has a
        ``.surf`` file (``"surf"``). Empty if there is no index.
    """
    path = mesh_file(filename, ".markers.json")
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_marker_index(filename: Path | str, index: dict[str, dict[str, Any]]):
    """Write the index of the markers of a mesh, or remove it if there are none."""
    path = mesh_file(filename, ".markers.json")
    if index:
        path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    else:
        path.unlink(missing_ok=True)


def remove_marker_files(filename: Path | str, names: list[str]):
    """Remove the ``.vtx`` and ``.surf`` files of markers."""
    for name in names:
        for suffix in (".vtx", ".surf"):
            marker_file(filename, name, suffix).unlink(missing_ok=True)


def marker_value(markers: dict[str, Any], index: dict[str, dict[str, Any]], name: str) -> int:
    """Get the value of a marker read from ``<base>_<name>.vtx`` or ``.surf``.

    Args:
        markers: Map from name to value, or to ``(value, dim)``
        index: The index of the markers of the mesh, see :func:`read_marker_index`
        name: Name of the marker in the filename

    Returns:
        The value in ``markers``, else in ``index``, else the name itself if it is an
        integer, else ``1``.
    """
    if name in markers:
        marker = markers[name]
        return int(marker) if isinstance(marker, (int, np.integer)) else int(marker[0])
    if name in index:
        return int(index[name]["value"])
    try:
        return int(name)
    except ValueError:
        return 1


def read_elem_type(path: Path) -> str:
    """Read the openCARP element type of the first element of an ``.elem`` file."""
    with open(path) as f:
        f.readline()
        line = f.readline().split()
    if len(line) == 0:
        raise ValueError(f"{path} has no elements.")
    if line[0] not in element_tdim:
        raise NotImplementedError(
            f"openCARP element type {line[0]} in {path} is not supported, "
            f"only {list(element_tdim)}."
        )
    return line[0]


def read_elem(
    path: Path, local_range: tuple[int, int] | list[int] | None = None
) -> tuple[str, npt.NDArray[np.int64], npt.NDArray[np.int32]]:
    """Read the elements of an ``.elem`` file.

    Args:
        path: The ``.elem`` file
        local_range: Range of elements to read, all if ``None``

    Returns:
        Tuple containing:
            - The openCARP element type
            - Nodes of each element in openCARP ordering
            - Region tag of each element, ``0`` if the file has none
    """
    elem_type = read_elem_type(path)
    num_nodes = len(dolfinx_to_opencarp[elem_type])
    if local_range is None:
        local_range = (0, read_count(path))
    rows = read_rows(path, local_range, num_nodes + 2)
    if other := sorted(set(rows[:, 0]) - {elem_type}):
        raise NotImplementedError(
            f"Meshes with several element types are not supported, {path} has "
            f"{[elem_type, *other]}."
        )
    cells = rows[:, 1 : 1 + num_nodes].astype(np.int64)
    if rows.shape[1] > 1 + num_nodes:
        tags = rows[:, 1 + num_nodes].astype(np.int32)
    else:
        tags = np.zeros(len(cells), dtype=np.int32)
    return elem_type, cells, tags


def write_elem(
    path: Path, elem_type: str, cells: npt.NDArray[np.int64], tags: npt.NDArray[np.int32]
):
    """Write elements with their region tags to an ``.elem`` file."""
    rows = np.column_stack([cells, tags])
    # Each line is the element type followed by the nodes and the tag
    fmt = " ".join([elem_type] + ["%d"] * rows.shape[1])
    write_rows(path, [str(len(cells))], rows, fmt)


def orientation(
    elem_type: str, x: npt.NDArray[np.floating], cells: npt.NDArray[np.int64]
) -> npt.NDArray:
    """Get the orientation of each element in openCARP ordering.

    Returns:
        The triple product of the edges at node 0 for 3D elements, the z-component of
        the cross product of the edges at node 0 for 2D elements. Positive for the
        orientation of openCARP's ``mesher``.
    """
    p = x[cells]
    # Nodes 1 and 2 of simplices, 1 and 3 of quadrilaterals and hexahedra, are the
    # neighbours of node 0 in the first face
    second = 2 if elem_type in ("Tr", "Tt") else 3
    normal = np.cross(p[:, 1] - p[:, 0], p[:, second] - p[:, 0])
    if element_tdim[elem_type] == 2:
        return normal[:, 2]
    third = 3 if elem_type == "Tt" else 4
    return np.einsum("ij,ij->i", normal, p[:, third] - p[:, 0])


def facet_normals(x: npt.NDArray[np.floating], facets: npt.NDArray[np.int64]) -> npt.NDArray:
    """Get a normal of each triangle or quadrilateral, with nodes in cyclic order."""
    p = x[facets]
    if facets.shape[1] == 3:
        return np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    return np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])


def oriented_facets(
    x: npt.NDArray[np.floating],
    elem_type: str,
    cells: npt.NDArray[np.int64],
    facets: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """Orient facets of a 3D mesh to point out of an element they are a face of.

    A facet between two elements points out of the element whose centre comes first
    in lexicographic ``(x, y, z)`` order, so the result does not depend on the
    ordering of the elements.

    Args:
        x: Node coordinates
        elem_type: openCARP element type of the mesh
        cells: Nodes of each element in openCARP ordering
        facets: Nodes of each facet in cyclic order

    Returns:
        The oriented facets, in the order of ``facets``
    """
    local_faces = element_faces[elem_type]
    faces = cells[:, local_faces].reshape(-1, local_faces.shape[1])
    face_cell_centres = np.repeat(x[cells].mean(axis=1), len(local_faces), axis=0)
    order = np.lexsort(face_cell_centres.T[::-1])
    match = match_rows(faces[order], facets)
    if np.any(match < 0):
        raise ValueError(f"{np.count_nonzero(match < 0)} tagged facets are not faces of the mesh.")
    cell_centres = face_cell_centres[order][match]

    facets = facets.copy()
    outward = x[facets].mean(axis=1) - cell_centres
    inward = np.einsum("ij,ij->i", facet_normals(x, facets), outward) < 0
    facets[inward] = facets[inward][:, mirror_permutation[face_element_types[elem_type]]]
    return facets


def sort_facets(facets: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Start each facet with its smallest node, keeping its orientation, and sort the
    facets, so the result does not depend on how the facets were distributed."""
    num_nodes = facets.shape[1]
    shift = np.argmin(facets, axis=1)[:, None]
    facets = np.take_along_axis(facets, (shift + np.arange(num_nodes)) % num_nodes, axis=1)
    return facets[np.lexsort(facets.T[::-1])]
