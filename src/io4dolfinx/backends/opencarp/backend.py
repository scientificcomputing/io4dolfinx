"""
openCARP interface to io4dolfinx.

openCARP stores a mesh as a set of ASCII files sharing a base name, with 0-based
node indices and coordinates in µm. See
https://opencarp.org/documentation/user-manual (section "Mesh files").

- ``<base>.pts``: number of nodes, then one ``x y z`` line per node. 2D meshes have
  ``z = 0``.
- ``<base>.elem``: number of elements, then one line per element with its type, its
  nodes and its tag, e.g. ``Tt n0 n1 n2 n3 tag``. Linear triangles (``Tr``),
  quadrilaterals (``Qd``), tetrahedra (``Tt``) and hexahedra (``Hx``) are supported,
  one type per mesh. The tag is the region of the element; in DOLFINx terms the
  value of a cell :class:`dolfinx.mesh.MeshTags`.
- ``<base>.lon``: number of directions (1 for fibers, 2 for fibers and sheets),
  then one ``fx fy fz [sx sy sz]`` line per element, in the order of ``.elem``.
- ``<base>_<marker>.vtx``: number of nodes, the domain (``extra`` for indices into the whole mesh)
  and one node index per line, for stimuli and boundary conditions.
- ``<base>_<marker>.surf``: number of faces, then one ``Tr n0 n1 n2`` or
  ``Qd n0 n1 n2 n3`` line per face of a 3D mesh.

Each file is written by rank 0.

Writing a mesh with markers and fibers takes the generic io4dolfinx functions, called
in this order with ``backend="opencarp"``:

.. code-block:: python

    args = {"scale": 1000.0, "markers": {"ENDO": (1, 2), "EPI": (2, 2)}}  # mesh in mm
    io4dolfinx.write_mesh_input_order("heart", mesh, backend="opencarp", backend_args=args)
    for tags in (cell_tags, facet_tags):
        io4dolfinx.write_meshtags(
            "heart", mesh, tags, backend="opencarp", backend_args=args, on_input_mesh=True
        )
    io4dolfinx.write_cell_data(
        "heart", f0, 0.0, io4dolfinx.FileMode.write, args, "opencarp", on_input_mesh=True
    )
    io4dolfinx.write_cell_data(
        "heart", s0, 0.0, io4dolfinx.FileMode.append, args, "opencarp", on_input_mesh=True
    )

- The mesh comes first, as tags and directions are written against ``<base>.elem``.
- Cell tags set the regions in ``<base>.elem``, facet tags give a ``.vtx`` (and in
  3D a ``.surf``) file per marker value.
- :func:`io4dolfinx.write_cell_data` with :attr:`FileMode.write
  <io4dolfinx.FileMode.write>` writes the fibers, and with :attr:`FileMode.append
  <io4dolfinx.FileMode.append>` adds the sheets. Fields are interpolated into DG-0,
  so fields in a quadrature space must first be generated in a P-1 or DG-0 space.

The ordering of nodes and elements must be the same in every step:

- With :func:`io4dolfinx.write_mesh_input_order` and ``on_input_mesh=True`` (as
  above), the files use the ordering of the input mesh and are the same whatever the
  number of processes. Use this for reproducible files.
- With :func:`io4dolfinx.write_mesh` and ``on_input_mesh=False`` (the default of
  :func:`io4dolfinx.write_meshtags` and :func:`io4dolfinx.write_cell_data`), the files
  use the DOLFINx ordering, which depends on the partitioning.

Mixing the two gives directions and markers on the wrong elements and nodes.

The files are read with :func:`io4dolfinx.read_mesh`, :func:`io4dolfinx.read_meshtags`
and :func:`io4dolfinx.read_cell_data`, given the path of any of the mesh files, e.g.
``"heart.elem"``.

Elements are written with the orientation used by openCARP's ``mesher``: positive
volume in 3D, counter-clockwise in the xy-plane in 2D. Hexahedra list one face
(nodes 0-3) followed by the nodes opposite nodes 0, 3, 2 and 1.

Backend arguments:

- ``"scale"`` (required for meshes): factor converting mesh coordinates to µm,
  e.g. ``1000.0`` for a mesh in mm. Coordinates are divided by it when read.
- ``"default_tag"``: region tag of cells without a cell tag (default ``0``).
- ``"markers"``: map from marker name to value, or to ``(value, dim)`` as used by
  cardiac-geometriesx. Names the ``.vtx`` and ``.surf`` files of a marker value,
  which otherwise use the value itself.
- ``"surf"``: write ``.surf`` files for facet tags of 3D meshes (default ``True``).
- ``"vtx_domain"``: domain written to ``.vtx`` files (default ``"extra"``, meaning
  indices into the whole mesh, as written; ``"intra"`` means the myocardium submesh).

SPDX License identifier: MIT

Copyright: Henrik Finsberg and Simula Research Laboratory
"""

import glob
from pathlib import Path
from typing import Any, Callable, TypeVar

from mpi4py import MPI

import basix
import dolfinx
import numpy as np
import numpy.typing as npt

from ...structures import ArrayData, FunctionData, MeshData, MeshTagsData, ReadMeshData
from ...utils import check_file_exists, compute_local_range
from .. import FileMode, ReadMode

read_mode = ReadMode.parallel

T = TypeVar("T")

# Suffixes stripped from a filename to get the base name of the mesh files
_mesh_suffixes = (".pts", ".elem", ".lon")

# openCARP element type of each supported DOLFINx cell type, and the reverse
_element_types = {
    "triangle": "Tr",
    "quadrilateral": "Qd",
    "tetrahedron": "Tt",
    "hexahedron": "Hx",
}
_cell_types = {element: cell for cell, element in _element_types.items()}

# Topological dimension of each openCARP element type
_tdim = {"Tr": 2, "Qd": 2, "Tt": 3, "Hx": 3}

# Node permutation ``p`` with ``opencarp[i] = dolfinx[p[i]]`` for each element type
_dolfinx_to_opencarp = {
    "Tr": np.array([0, 1, 2]),
    "Qd": np.array([0, 1, 3, 2]),
    "Tt": np.array([0, 1, 2, 3]),
    "Hx": np.array([0, 1, 3, 2, 4, 6, 7, 5]),
}

# Node permutation mirroring an element in openCARP ordering, flipping its orientation
_mirror = {
    "Tr": np.array([0, 2, 1]),
    "Qd": np.array([0, 3, 2, 1]),
    "Tt": np.array([0, 2, 1, 3]),
    "Hx": np.array([0, 3, 2, 1, 4, 7, 6, 5]),
}

# Local faces of 3D elements in openCARP ordering, with their nodes in cyclic order,
# and the openCARP element type of the faces
_faces = {
    "Tt": np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]),
    "Hx": np.array(
        [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 7, 4], [1, 2, 6, 7], [2, 3, 5, 6], [3, 0, 4, 5]]
    ),
}
_face_types = {"Tt": "Tr", "Hx": "Qd"}

# Format of floating point values, enough digits for µm coordinates of a heart
_float_format = "%.10g"

# Names of the directions in ``.lon``, in column order
_directions = ("fiber", "sheet")


def _basename(filename: Path | str) -> Path:
    """Get the base name shared by the files of a mesh.

    Args:
        filename: Base name, or the path of its ``.pts``, ``.elem`` or ``.lon`` file

    Returns:
        The base name
    """
    path = Path(filename)
    return path.with_suffix("") if path.suffix in _mesh_suffixes else path


def _mesh_file(filename: Path | str, suffix: str) -> Path:
    """Get the path of one of the mesh files, e.g. ``<base>.elem``."""
    return Path(f"{_basename(filename)}{suffix}")


def _marker_file(filename: Path | str, name: str, suffix: str) -> Path:
    """Get the path of the file of a marker, e.g. ``<base>_<name>.vtx``."""
    return Path(f"{_basename(filename)}_{name}{suffix}")


def _get_scale(backend_args: dict[str, Any]) -> float:
    """Get the factor converting mesh coordinates to µm.

    It has no default, as a wrong unit gives a mesh that openCARP loads without
    complaint but simulates at the wrong size.
    """
    if backend_args.get("scale") is None:
        raise ValueError(
            "The openCARP backend needs backend_args['scale'], the factor converting mesh "
            "coordinates to µm (e.g. 1000.0 for a mesh in mm, 1.0 for a mesh in µm)."
        )
    scale = float(backend_args["scale"])
    if not scale > 0:
        raise ValueError(f"backend_args['scale'] must be positive, got {scale}.")
    return scale


def _marker_names(markers: dict[str, Any], dim: int) -> dict[int, str]:
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


def _marker_value(markers: dict[str, Any], name: str) -> int:
    """Get the value of a marker read from ``<base>_<name>.vtx`` or ``.surf``.

    Args:
        markers: Map from name to value, or to ``(value, dim)``
        name: Name of the marker in the filename

    Returns:
        The value in ``markers``, else the name itself if it is an integer (as
        written without markers), else ``1``.
    """
    if name in markers:
        marker = markers[name]
        return int(marker) if isinstance(marker, (int, np.integer)) else int(marker[0])
    try:
        return int(name)
    except ValueError:
        return 1


def _on_root(comm: MPI.Comm, fn: Callable[[], T]) -> T | None:
    """Run a function on rank 0 and raise any error it raises on all ranks.

    Returns:
        The result of ``fn`` on rank 0, ``None`` on the other ranks
    """
    result, error = None, None
    if comm.rank == 0:
        try:
            result = fn()
        except Exception as e:
            error = e
    error = comm.bcast(error, root=0)
    if error is not None:
        raise error
    return result


def _on_all(comm: MPI.Comm, fn: Callable[[], T]) -> T:
    """Run a function on all ranks and raise an error raised on any rank on all ranks,
    so that no rank is left waiting in a later collective call."""
    result, error = None, None
    try:
        result = fn()
    except Exception as e:
        error = e
    errors = [e for e in comm.allgather(error) if e is not None]
    if errors:
        raise errors[0]
    return result  # type: ignore[return-value]


def _gather_rows(
    comm: MPI.Comm, local: npt.NDArray, local_range: tuple[int, int] | list[int], num_global: int
) -> npt.NDArray | None:
    """Gather rows distributed in contiguous chunks to rank 0.

    Returns:
        All ``num_global`` rows on rank 0, ``None`` on the other ranks
    """
    chunks = comm.gather((int(local_range[0]), np.asarray(local)), root=0)
    if comm.rank != 0:
        return None
    assert chunks is not None
    first = chunks[0][1]
    rows = np.empty((num_global, *first.shape[1:]), dtype=first.dtype)
    for start, chunk in chunks:
        rows[start : start + len(chunk)] = chunk
    return rows


def _gather_concatenate(comm: MPI.Comm, local: npt.NDArray) -> npt.NDArray | None:
    """Concatenate an array over all ranks on rank 0, in rank order."""
    chunks = comm.gather(local, root=0)
    return None if chunks is None else np.concatenate(chunks)


def _read_count(path: Path) -> int:
    """Read the number on the first line of a file."""
    with open(path) as f:
        return int(f.readline().split()[0])


def _read_rows(
    path: Path, local_range: tuple[int, int] | list[int], num_columns: int, header: int = 1
) -> npt.NDArray[np.str_]:
    """Read a range of rows after the header of a file, as strings.

    Args:
        path: The file
        local_range: Range ``[start, end)`` of rows to read
        num_columns: Number of columns, only used to shape an empty range
        header: Number of header lines

    Returns:
        The rows, of shape ``(end - start, num_columns)``
    """
    start, end = int(local_range[0]), int(local_range[1])
    if end == start:
        return np.zeros((0, num_columns), dtype=np.str_)
    return np.loadtxt(path, dtype=np.str_, skiprows=header + start, max_rows=end - start, ndmin=2)


def _elem_type(path: Path) -> str:
    """Read the openCARP element type of the first element of an ``.elem`` file."""
    with open(path) as f:
        f.readline()
        line = f.readline().split()
    if len(line) == 0:
        raise ValueError(f"{path} has no elements.")
    if line[0] not in _tdim:
        raise NotImplementedError(
            f"openCARP element type {line[0]} in {path} is not supported, only {list(_tdim)}."
        )
    return line[0]


def _read_elem(
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
    elem_type = _elem_type(path)
    num_nodes = len(_dolfinx_to_opencarp[elem_type])
    if local_range is None:
        local_range = (0, _read_count(path))
    rows = _read_rows(path, local_range, num_nodes + 2)
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


def _write_rows(path: Path, header: list[str], rows: npt.NDArray, fmt: str):
    """Write header lines followed by one formatted line per row."""
    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        np.savetxt(f, rows, fmt=fmt)


def _element_format(elem_type: str, num_columns: int) -> str:
    """Get the format of an element line with ``num_columns`` integers."""
    return " ".join([elem_type] + ["%d"] * num_columns)


def _write_elem(
    path: Path, elem_type: str, cells: npt.NDArray[np.int64], tags: npt.NDArray[np.int32]
):
    """Write elements with their region tags to an ``.elem`` file."""
    rows = np.column_stack([cells, tags])
    _write_rows(path, [str(len(cells))], rows, _element_format(elem_type, rows.shape[1]))


def _match_rows(reference: npt.NDArray[np.int64], query: npt.NDArray[np.int64]) -> npt.NDArray:
    """Find the rows of ``reference`` holding the same nodes as each row of ``query``.

    Returns:
        Index into ``reference`` of each query row, ``-1`` if it has no match. If
        several rows match, the last one is used.
    """
    if len(query) == 0:
        return np.zeros(0, dtype=np.int64)
    rows = np.vstack([np.sort(reference, axis=1), np.sort(query, axis=1)])
    _, inverse = np.unique(rows, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    lookup = np.full(inverse.max() + 1, -1, dtype=np.int64)
    lookup[inverse[: len(reference)]] = np.arange(len(reference))
    return lookup[inverse[len(reference) :]]


def _orientation(
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
    if _tdim[elem_type] == 2:
        return normal[:, 2]
    third = 3 if elem_type == "Tt" else 4
    return np.einsum("ij,ij->i", normal, p[:, third] - p[:, 0])


def _facet_normals(x: npt.NDArray[np.floating], facets: npt.NDArray[np.int64]) -> npt.NDArray:
    """Get a normal of each triangle or quadrilateral, with nodes in cyclic order."""
    p = x[facets]
    if facets.shape[1] == 3:
        return np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    return np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])


def _oriented_facets(
    x: npt.NDArray[np.floating],
    elem_type: str,
    cells: npt.NDArray[np.int64],
    facets: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """Orient facets of a 3D mesh and sort them in a canonical order.

    Each facet gets the normal pointing out of an element it is a face of, starts with
    its smallest node and the facets are sorted, so the result does not depend on how
    the facets were distributed.

    Args:
        x: Node coordinates
        elem_type: openCARP element type of the mesh
        cells: Nodes of each element in openCARP ordering
        facets: Nodes of each facet in cyclic order

    Returns:
        The oriented facets
    """
    local_faces = _faces[elem_type]
    faces = cells[:, local_faces].reshape(-1, local_faces.shape[1])
    match = _match_rows(faces, facets)
    if np.any(match < 0):
        raise ValueError(f"{np.count_nonzero(match < 0)} tagged facets are not faces of the mesh.")
    cell_centres = x[cells[match // len(local_faces)]].mean(axis=1)

    facets = facets.copy()
    outward = x[facets].mean(axis=1) - cell_centres
    inward = np.einsum("ij,ij->i", _facet_normals(x, facets), outward) < 0
    facets[inward] = facets[inward][:, _mirror[_face_types[elem_type]]]

    # Rotating the nodes keeps the orientation
    num_nodes = facets.shape[1]
    shift = np.argmin(facets, axis=1)[:, None]
    facets = np.take_along_axis(facets, (shift + np.arange(num_nodes)) % num_nodes, axis=1)
    return facets[np.lexsort(facets.T[::-1])]


def get_default_backend_args(arguments: dict[str, Any] | None) -> dict[str, Any]:
    args = dict(arguments or {})
    args.setdefault("default_tag", 0)
    args.setdefault("markers", {})
    args.setdefault("surf", True)
    args.setdefault("vtx_domain", "extra")
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
    raise NotImplementedError("The openCARP backend cannot write attributes.")


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
    raise NotImplementedError("The openCARP backend cannot read attributes.")


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
    raise NotImplementedError("The openCARP backend cannot make checkpoints.")


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
    raise NotImplementedError("The openCARP backend cannot read HDF5 arrays.")


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
    raise NotImplementedError("The openCARP backend cannot read timestamps.")


def write_mesh(
    filename: Path | str,
    comm: MPI.Comm,
    mesh: MeshData,
    backend_args: dict[str, Any] | None = None,
    mode: FileMode = FileMode.write,
    time: float = 0.0,
):
    """Write a linear mesh to ``<base>.pts`` and ``<base>.elem``.

    Supports triangle and quadrilateral meshes in 2D, and tetrahedral and hexahedral
    meshes in 3D. Every cell gets the region tag ``backend_args["default_tag"]``, until
    cell tags are written with :func:`write_meshtags`. Elements with the opposite
    orientation of openCARP's ``mesher`` are mirrored. A ``<base>.lon`` from an
    earlier mesh is removed.

    Args:
        filename: Base name of the files, or the path of one of them
        comm: MPI communicator used in storage
        mesh: Mesh data, from :func:`io4dolfinx.write_mesh_input_order` or
            :func:`io4dolfinx.write_mesh`
        backend_args: Arguments to backend, ``"scale"`` is required
        mode: Only :attr:`FileMode.write` is supported
        time: Not used
    """
    args = get_default_backend_args(backend_args)
    if mesh.cell_type not in _element_types or mesh.degree != 1:
        raise NotImplementedError(
            "The openCARP backend supports linear triangle, quadrilateral, tetrahedral and "
            f"hexahedral meshes, got a {mesh.cell_type} mesh of degree {mesh.degree}."
        )
    elem_type = _element_types[mesh.cell_type]
    gdim = mesh.local_geometry.shape[1]
    if gdim != _tdim[elem_type]:
        raise NotImplementedError(
            f"The openCARP backend needs {_tdim[elem_type]}D coordinates for a "
            f"{mesh.cell_type} mesh, got {gdim}D."
        )
    if mode != FileMode.write:
        raise NotImplementedError("The openCARP backend can only write a mesh to new files.")
    scale = _get_scale(args)

    x = _gather_rows(comm, mesh.local_geometry, mesh.local_geometry_pos, mesh.num_nodes_global)
    cells = _gather_rows(comm, mesh.local_topology, mesh.local_topology_pos, mesh.num_cells_global)

    def write():
        assert x is not None and cells is not None
        points = np.zeros((len(x), 3), dtype=x.dtype)
        points[:, :gdim] = x
        elements = cells[:, _dolfinx_to_opencarp[elem_type]]
        orientation = _orientation(elem_type, points, elements)
        if np.any(orientation == 0):
            raise ValueError(f"The mesh has {np.count_nonzero(orientation == 0)} degenerate cells.")
        flip = orientation < 0
        elements[flip] = elements[flip][:, _mirror[elem_type]]

        _write_rows(_mesh_file(filename, ".pts"), [str(len(points))], points * scale, _float_format)
        tags = np.full(len(elements), args["default_tag"], dtype=np.int32)
        _write_elem(_mesh_file(filename, ".elem"), elem_type, elements, tags)
        _mesh_file(filename, ".lon").unlink(missing_ok=True)

    _on_root(comm, write)


def read_mesh_data(
    filename: Path | str,
    comm: MPI.Comm,
    time: str | float | None = 0.0,
    read_from_partition: bool = False,
    backend_args: dict[str, Any] | None = None,
) -> ReadMeshData:
    """Read a linear mesh from ``<base>.pts`` and ``<base>.elem``.

    Node and cell data is read in contiguous chunks on each process. A mesh of 2D
    elements must lie in a plane of constant ``z``, and gets 2D coordinates.

    Args:
        filename: Base name of the files, or the path of one of them
        comm: The MPI communciator to distribute the mesh over
        time: Not used
        read_from_partition: Read mesh with partition from file (not supported)
        backend_args: Arguments to backend, ``"scale"`` is required
    Returns:
        The mesh topology, geometry, UFL domain and partition function
    """
    if read_from_partition:
        raise NotImplementedError("The openCARP backend cannot read partitioning data.")
    args = get_default_backend_args(backend_args)
    scale = _get_scale(args)
    pts = _mesh_file(filename, ".pts")
    elem = _mesh_file(filename, ".elem")
    check_file_exists(pts)
    check_file_exists(elem)
    elem_type = _elem_type(elem)

    def read():
        x = _read_rows(pts, compute_local_range(comm, _read_count(pts)), 3).astype(np.float64)
        _, cells, _ = _read_elem(elem, compute_local_range(comm, _read_count(elem)))
        return x, cells

    x, cells = _on_all(comm, read)
    if _tdim[elem_type] == 2:
        z_min = comm.allreduce(x[:, 2].min(initial=np.inf), op=MPI.MIN)
        z_max = comm.allreduce(x[:, 2].max(initial=-np.inf), op=MPI.MAX)
        if z_max > z_min:
            raise NotImplementedError(
                f"{elem} has 2D elements that do not lie in a plane of constant z. Surface "
                "meshes in 3D are not supported."
            )
        x = x[:, :2]
    return ReadMeshData(
        cells=cells[:, np.argsort(_dolfinx_to_opencarp[elem_type])],
        cell_type=_cell_types[elem_type],
        x=x / scale,
        lvar=int(basix.LagrangeVariant.equispaced),
        degree=1,
        partition_graph=None,
    )


def write_meshtags(
    filename: str | Path,
    comm: MPI.Comm,
    data: MeshTagsData,
    backend_args: dict[str, Any] | None = None,
):
    """Write mesh tags to the openCARP files of a mesh written earlier.

    - Cell tags set the region tags in ``<base>.elem``. Untagged cells keep their tag.
    - Other tags are written, per value, to ``<base>_<marker>.vtx`` with the nodes of
      the tagged entities.
    - Facet tags of a 3D mesh are also written to ``<base>_<marker>.surf``, as
      triangles or quadrilaterals with outward normals (if ``backend_args["surf"]``).

    ``<marker>`` is the name of the value in ``backend_args["markers"]``, or the
    value itself.

    Args:
        filename: Base name of the files, or the path of one of them
        comm: MPI communicator used in storage
        data: Mesh tags with nodes in the ordering the mesh was written in, see the
            ``on_input_mesh`` argument of :func:`io4dolfinx.write_meshtags`
        backend_args: Arguments to backend
    """
    args = get_default_backend_args(backend_args)
    local_indices = np.asarray(data.indices, dtype=np.int64)
    num_nodes = data.num_dofs_per_entity or local_indices.shape[-1]
    indices = _gather_concatenate(comm, local_indices.reshape(-1, num_nodes))
    values = _gather_concatenate(comm, np.asarray(data.values, dtype=np.int32))
    elem = _mesh_file(filename, ".elem")

    def write():
        assert indices is not None and values is not None
        check_file_exists(elem)
        elem_type, cells, tags = _read_elem(elem)
        if data.dim == _tdim[elem_type]:
            rows = _match_rows(cells, indices)
            if np.any(rows < 0):
                raise ValueError(f"{np.count_nonzero(rows < 0)} tagged cells are not in {elem}.")
            tags[rows] = values
            _write_elem(elem, elem_type, cells, tags)
            return

        write_surf = args["surf"] and _tdim[elem_type] == 3 and data.dim == 2
        if write_surf:
            pts = _mesh_file(filename, ".pts")
            x = _read_rows(pts, (0, _read_count(pts)), 3).astype(np.float64)
            face_type = _face_types[elem_type]
            # DOLFINx orders the nodes of a quadrilateral facet as a tensor product
            facets = indices[:, _dolfinx_to_opencarp[face_type]]
        names = _marker_names(args["markers"], data.dim)
        for value in np.unique(values):
            marked = values == value
            name = names.get(int(value), str(value))
            nodes = np.unique(indices[marked])
            _write_rows(
                _marker_file(filename, name, ".vtx"),
                [str(len(nodes)), args["vtx_domain"]],
                nodes,
                "%d",
            )
            if write_surf:
                faces = _oriented_facets(x, elem_type, cells, facets[marked])
                _write_rows(
                    _marker_file(filename, name, ".surf"),
                    [str(len(faces))],
                    faces,
                    _element_format(face_type, faces.shape[1]),
                )

    _on_root(comm, write)


def read_meshtags_data(
    filename: str | Path,
    comm: MPI.Comm,
    name: str,
    backend_args: dict[str, Any] | None = None,
) -> MeshTagsData:
    """Read mesh tags from openCARP files.

    Args:
        filename: Base name of the files, or the path of one of them
        comm: MPI communicator used in storage
        name: ``"cell"`` for the region tags in ``<base>.elem``, ``"facet"`` for the
            faces of all ``<base>_*.surf`` files, otherwise the name of a marker:
            the faces of ``<base>_<name>.surf`` or, if there is none, the nodes of
            ``<base>_<name>.vtx``.
        backend_args: Arguments to backend. The value of a marker is taken from
            ``"markers"``, else from its name if that is an integer, else it is ``1``.

    Returns:
        Internal data structure for the mesh tags read from file
    """
    args = get_default_backend_args(backend_args)
    if name == "cell":
        elem = _mesh_file(filename, ".elem")
        check_file_exists(elem)
        elem_type, cells, tags = _on_all(
            comm, lambda: _read_elem(elem, compute_local_range(comm, _read_count(elem)))
        )
        cells = cells[:, np.argsort(_dolfinx_to_opencarp[elem_type])]
        return MeshTagsData(name=name, values=tags, indices=cells, dim=_tdim[elem_type])

    base = _basename(filename)
    if name == "facet":
        files = sorted(Path(p) for p in glob.glob(f"{glob.escape(str(base))}_*.surf"))
        if len(files) == 0:
            raise FileNotFoundError(f"Could not find any {base}_*.surf files.")
        dim = 2
    elif (surf := _marker_file(filename, name, ".surf")).exists():
        files, dim = [surf], 2
    elif (vtx := _marker_file(filename, name, ".vtx")).exists():
        files, dim = [vtx], 0
    else:
        available = sorted(
            p.name[len(base.name) + 1 :].rsplit(".", 1)[0]
            for p in base.parent.glob(f"{glob.escape(base.name)}_*")
            if p.suffix in (".surf", ".vtx")
        )
        raise KeyError(f"Could not find marker '{name}' for {base}. Available: {available}")

    # Marker files are small, so rank 0 reads them all. read_meshtags reshapes the
    # empty indices of the other ranks.
    def read():
        all_indices, all_values = [], []
        for path in files:
            marker = path.name[len(base.name) + 1 : -len(path.suffix)]
            if dim == 2:
                rows = _read_rows(path, (0, _read_count(path)), 4)
                face_types = set(rows[:, 0])
                if len(face_types) > 1 or not face_types <= {"Tr", "Qd"}:
                    raise NotImplementedError(f"Unsupported face types {face_types} in {path}.")
                face_type = face_types.pop() if face_types else "Tr"
                perm = np.argsort(_dolfinx_to_opencarp[face_type])
                entities = rows[:, 1:].astype(np.int64)[:, perm]
            else:
                rows = _read_rows(path, (0, _read_count(path)), 1, header=2)
                entities = rows.astype(np.int64).reshape(-1, 1)
            all_indices.append(entities)
            all_values.append(np.full(len(entities), _marker_value(args["markers"], marker)))
        return np.vstack(all_indices), np.concatenate(all_values).astype(np.int32)

    indices, values = np.zeros((0, 1), dtype=np.int64), np.zeros(0, dtype=np.int32)
    result = _on_root(comm, read)
    if result is not None:
        indices, values = result
    return MeshTagsData(name=name, values=values, indices=indices, dim=dim)


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
    raise NotImplementedError("The openCARP backend cannot read dofmap.")


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
    raise NotImplementedError("The openCARP backend cannot read dofs.")


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
    raise NotImplementedError("The openCARP backend cannot read cell perms.")


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
    raise NotImplementedError(
        "The openCARP backend cannot write functions. Use io4dolfinx.write_cell_data for fibers."
    )


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
    raise NotImplementedError("The openCARP backend cannot read legacy meshes.")


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
    raise NotImplementedError("The openCARP backend cannot read point data.")


def read_cell_data(
    filename: Path | str,
    name: str,
    comm: MPI.Comm,
    time: str | float | None,
    backend_args: dict[str, Any] | None,
) -> tuple[npt.NDArray[np.int64], np.ndarray]:
    """Read the fiber or sheet direction of each element from ``<base>.lon``.

    Args:
        filename: Base name of the files, or the path of one of them
        name: ``"fiber"`` or ``"sheet"``
        comm: Communicator to launch IO on.
        time: Not used
        backend_args: The backend arguments
    Returns:
        A tuple (topology, dofs) where topology contains the
        vertex indices of the cells, dofs the degrees of
        freedom within that cell. A 2D mesh gets the x- and y-components.
    """
    lon = _mesh_file(filename, ".lon")
    elem = _mesh_file(filename, ".elem")
    check_file_exists(lon)
    check_file_exists(elem)
    num_directions = _read_count(lon)
    if name not in _directions[:num_directions]:
        raise KeyError(
            f"Could not find '{name}' in {lon}. Available: {list(_directions[:num_directions])}"
        )

    def read():
        local_range = compute_local_range(comm, _read_count(elem))
        elem_type, cells, _ = _read_elem(elem, local_range)
        rows = _read_rows(lon, local_range, 3 * num_directions).astype(np.float64)
        return elem_type, cells, rows

    elem_type, cells, rows = _on_all(comm, read)
    column = 3 * _directions.index(name)
    dofs = rows[:, column : column + _tdim[elem_type]]
    cells = cells[:, np.argsort(_dolfinx_to_opencarp[elem_type])]
    return cells, np.ascontiguousarray(dofs)


def read_function_names(
    filename: Path | str, comm: MPI.Comm, backend_args: dict[str, Any] | None
) -> list[str]:
    """Read the names of the directions in ``<base>.lon``.

    Args:
        filename: Base name of the files, or the path of one of them
        comm: MPI communicator to launch IO on.
        backend_args: Arguments to backend

    Returns:
        ``["fiber"]`` or ``["fiber", "sheet"]``
    """
    lon = _mesh_file(filename, ".lon")
    check_file_exists(lon)
    return list(_directions[: _read_count(lon)])


def write_data(
    filename: Path | str,
    array_data: ArrayData,
    comm: MPI.Comm,
    time: str | float | None,
    mode: FileMode,
    backend_args: dict[str, Any] | None,
):
    """Write a direction per element to ``<base>.lon``, normalised to unit length.

    2D directions get ``z = 0``.

    Writing with :attr:`FileMode.write` gives the fiber direction. Writing again
    with :attr:`FileMode.append` adds the sheet direction.

    Args:
        filename: Base name of the files, or the path of one of them
        array_data: Cell data in the ordering the mesh was written in, see the
            ``on_input_mesh`` argument of :func:`io4dolfinx.write_cell_data`
        comm: MPI communicator to open the file with
        time: Not used
        mode: Write for the fiber, append for the sheet direction
        backend_args: The backend arguments
    """
    if array_data.type != "Cell":
        raise NotImplementedError(
            "The openCARP backend can only write cell data (fibers and sheets)."
        )
    num_cells, num_components = array_data.global_shape
    if num_components not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D vector per cell, got {num_components} components.")
    directions = _gather_rows(comm, array_data.values, array_data.local_range, num_cells)
    lon = _mesh_file(filename, ".lon")
    elem = _mesh_file(filename, ".elem")

    def write():
        assert directions is not None
        check_file_exists(elem)
        if (num_elements := _read_count(elem)) != num_cells:
            raise ValueError(f"Got data for {num_cells} cells, but {elem} has {num_elements}.")
        if num_components < (tdim := _tdim[_elem_type(elem)]):
            raise ValueError(f"Expected a {tdim}D vector per cell, got {num_components}D.")
        # openCARP directions are 3D, also for 2D meshes
        vectors = np.zeros((num_cells, 3), dtype=np.float64)
        vectors[:, :num_components] = directions
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms == 0):
            raise ValueError(f"{np.count_nonzero(norms == 0)} cells have a zero direction.")
        rows = vectors / norms[:, None]

        if mode == FileMode.append:
            check_file_exists(lon)
            if _read_count(lon) != 1:
                raise ValueError(f"{lon} already holds fibers and sheets.")
            fibers = _read_rows(lon, (0, num_cells), 3).astype(np.float64)
            rows = np.hstack([fibers, rows])
        _write_rows(lon, [str(rows.shape[1] // 3)], rows, _float_format)

    _on_root(comm, write)
