"""
openCARP interface to io4dolfinx.

openCARP stores a mesh as a set of ASCII files sharing a base name, with 0-based
node indices and coordinates in µm. See
https://opencarp.org/documentation/user-manual (section "Mesh files").

- ``<base>.pts``: number of nodes, then one ``x y z`` line per node.
- ``<base>.elem``: number of elements, then one ``Tt n0 n1 n2 n3 tag`` line per
  linear tetrahedron. The tag is the region of the element; in DOLFINx terms the
  value of a cell :class:`dolfinx.mesh.MeshTags`.
- ``<base>.lon``: number of directions (1 for fibers, 2 for fibers and sheets),
  then one ``fx fy fz [sx sy sz]`` line per element, in the order of ``.elem``.
- ``<base>_<marker>.vtx``: number of nodes, the domain (``extra`` for indices into the whole mesh)
  and one node index per line, for stimuli and boundary conditions.
- ``<base>_<marker>.surf``: number of triangles, then one ``Tr n0 n1 n2`` line
  per triangle.

Nodes and cells are written in the ordering of the input mesh, so the files do
not depend on the number of processes. Each file is written by rank 0.

Backend arguments:

- ``"scale"`` (required for meshes): factor converting mesh coordinates to µm,
  e.g. ``1000.0`` for a mesh in mm. Coordinates are divided by it when read.
- ``"default_tag"``: region tag of cells without a cell tag (default ``0``).
- ``"markers"``: map from marker name to value, or to ``(value, dim)`` as used by
  cardiac-geometriesx. Names the ``.vtx`` and ``.surf`` files of a marker value,
  which otherwise use the value itself.
- ``"surf"``: write ``.surf`` files for facet tags (default ``True``).
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

# Number of nodes of the entities written for each topological dimension
_nodes_per_entity = {0: 1, 2: 3, 3: 4}

# Local faces of a tetrahedron, face ``i`` being opposite vertex ``i``
_tet_faces = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])

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


def _read_elem(
    path: Path, local_range: tuple[int, int] | list[int] | None = None
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32]]:
    """Read the tetrahedra of an ``.elem`` file.

    Args:
        path: The ``.elem`` file
        local_range: Range of elements to read, all if ``None``

    Returns:
        Tuple containing:
            - Nodes of each element, of shape ``(num_elements, 4)``
            - Region tag of each element, ``0`` if the file has none
    """
    if local_range is None:
        local_range = (0, _read_count(path))
    rows = _read_rows(path, local_range, 6)
    if unsupported := sorted(set(rows[:, 0]) - {"Tt"}):
        raise NotImplementedError(
            f"Only linear tetrahedra (Tt) are supported, {path} has {unsupported}."
        )
    cells = rows[:, 1:5].astype(np.int64)
    if rows.shape[1] > 5:
        tags = rows[:, 5].astype(np.int32)
    else:
        tags = np.zeros(len(cells), dtype=np.int32)
    return cells, tags


def _write_rows(path: Path, header: list[str], rows: npt.NDArray, fmt: str):
    """Write header lines followed by one formatted line per row."""
    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        np.savetxt(f, rows, fmt=fmt)


def _write_elem(path: Path, cells: npt.NDArray[np.int64], tags: npt.NDArray[np.int32]):
    """Write linear tetrahedra with their region tags to an ``.elem`` file."""
    _write_rows(path, [str(len(cells))], np.column_stack([cells, tags]), "Tt %d %d %d %d %d")


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


def _signed_volumes(x: npt.NDArray[np.floating], cells: npt.NDArray[np.int64]) -> npt.NDArray:
    """Get six times the signed volume of each tetrahedron."""
    p = x[cells]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    return np.einsum("ij,ij->i", normal, p[:, 3] - p[:, 0])


def _oriented_triangles(
    x: npt.NDArray[np.floating], cells: npt.NDArray[np.int64], triangles: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """Orient facets of the mesh and sort them in a canonical order.

    Each triangle gets the normal pointing out of a tetrahedron it is a face of,
    starts with its smallest node and the triangles are sorted, so the result does
    not depend on how the facets were distributed.

    Args:
        x: Node coordinates
        cells: Nodes of each tetrahedron
        triangles: Nodes of each facet

    Returns:
        The oriented triangles
    """
    faces = cells[:, _tet_faces].reshape(-1, 3)
    match = _match_rows(faces, triangles)
    if np.any(match < 0):
        raise ValueError(f"{np.count_nonzero(match < 0)} tagged facets are not faces of the mesh.")
    opposite = cells.reshape(-1)[match]

    triangles = triangles.copy()
    p = x[triangles]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    inward = np.einsum("ij,ij->i", normal, x[opposite] - p[:, 0]) > 0
    triangles[inward, 1], triangles[inward, 2] = triangles[inward, 2], triangles[inward, 1]

    # Rotating the nodes keeps the orientation
    shift = np.argmin(triangles, axis=1)[:, None]
    triangles = np.take_along_axis(triangles, (shift + np.arange(3)) % 3, axis=1)
    return triangles[np.lexsort(triangles.T[::-1])]


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
    """Write a linear tetrahedral mesh to ``<base>.pts`` and ``<base>.elem``.

    Every cell gets the region tag ``backend_args["default_tag"]``, until cell tags
    are written with :func:`write_meshtags`. Tetrahedra with negative volume have
    two nodes swapped, so that all elements are positively oriented. A ``<base>.lon``
    from an earlier mesh is removed.

    Args:
        filename: Base name of the files, or the path of one of them
        comm: MPI communicator used in storage
        mesh: Mesh data in input ordering, see
            :func:`io4dolfinx.write_mesh_input_order`
        backend_args: Arguments to backend, ``"scale"`` is required
        mode: Only :attr:`FileMode.write` is supported
        time: Not used
    """
    args = get_default_backend_args(backend_args)
    if not mesh.on_input_mesh:
        raise ValueError(
            "The openCARP backend writes meshes in the input ordering, so that the files do "
            "not depend on the number of processes. Use io4dolfinx.write_mesh_input_order or "
            "io4dolfinx.backends.opencarp.write_opencarp."
        )
    if mesh.cell_type != "tetrahedron" or mesh.degree != 1:
        raise NotImplementedError(
            "The openCARP backend only supports linear tetrahedral meshes, got a "
            f"{mesh.cell_type} mesh of degree {mesh.degree}."
        )
    if mesh.local_geometry.shape[1] != 3:
        raise ValueError(f"Expected 3D coordinates, got {mesh.local_geometry.shape[1]}D.")
    if mode != FileMode.write:
        raise NotImplementedError("The openCARP backend can only write a mesh to new files.")
    scale = _get_scale(args)

    x = _gather_rows(comm, mesh.local_geometry, mesh.local_geometry_pos, mesh.num_nodes_global)
    cells = _gather_rows(comm, mesh.local_topology, mesh.local_topology_pos, mesh.num_cells_global)

    def write():
        assert x is not None and cells is not None
        volumes = _signed_volumes(x, cells)
        if np.any(volumes == 0):
            raise ValueError(f"The mesh has {np.count_nonzero(volumes == 0)} degenerate cells.")
        flip = volumes < 0
        cells[flip, 1], cells[flip, 2] = cells[flip, 2], cells[flip, 1]

        _write_rows(_mesh_file(filename, ".pts"), [str(len(x))], x * scale, _float_format)
        tags = np.full(len(cells), args["default_tag"], dtype=np.int32)
        _write_elem(_mesh_file(filename, ".elem"), cells, tags)
        _mesh_file(filename, ".lon").unlink(missing_ok=True)

    _on_root(comm, write)


def read_mesh_data(
    filename: Path | str,
    comm: MPI.Comm,
    time: str | float | None = 0.0,
    read_from_partition: bool = False,
    backend_args: dict[str, Any] | None = None,
) -> ReadMeshData:
    """Read a linear tetrahedral mesh from ``<base>.pts`` and ``<base>.elem``.

    Node and cell data is read in contiguous chunks on each process.

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

    x = _read_rows(pts, compute_local_range(comm, _read_count(pts)), 3).astype(np.float64)
    cells, _ = _read_elem(elem, compute_local_range(comm, _read_count(elem)))
    return ReadMeshData(
        cells=cells,
        cell_type="tetrahedron",
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
    - Facet tags are written, per value, to ``<base>_<marker>.vtx`` with the nodes of
      the tagged facets, and ``<base>_<marker>.surf`` with the facets as triangles
      with outward normals (if ``backend_args["surf"]``).
    - Vertex tags are written, per value, to ``<base>_<marker>.vtx``.

    ``<marker>`` is the name of the value in ``backend_args["markers"]``, or the
    value itself.

    Args:
        filename: Base name of the files, or the path of one of them
        comm: MPI communicator used in storage
        data: Mesh tags with nodes in input ordering, see the ``on_input_mesh``
            argument of :func:`io4dolfinx.write_meshtags`
        backend_args: Arguments to backend
    """
    args = get_default_backend_args(backend_args)
    if not data.on_input_mesh:
        raise ValueError(
            "The openCARP backend writes meshtags in the input node ordering. Use "
            "io4dolfinx.write_meshtags(..., on_input_mesh=True) or "
            "io4dolfinx.backends.opencarp.write_opencarp."
        )
    if data.dim not in _nodes_per_entity:
        raise NotImplementedError(
            f"The openCARP backend can write cell, facet and vertex tags, got dimension {data.dim}."
        )
    num_nodes = _nodes_per_entity[data.dim]
    indices = _gather_concatenate(
        comm, np.asarray(data.indices, dtype=np.int64).reshape(-1, num_nodes)
    )
    values = _gather_concatenate(comm, np.asarray(data.values, dtype=np.int32))
    elem = _mesh_file(filename, ".elem")

    def write():
        assert indices is not None and values is not None
        check_file_exists(elem)
        cells, tags = _read_elem(elem)
        if data.dim == 3:
            rows = _match_rows(cells, indices)
            if np.any(rows < 0):
                raise ValueError(f"{np.count_nonzero(rows < 0)} tagged cells are not in {elem}.")
            tags[rows] = values
            _write_elem(elem, cells, tags)
            return

        write_surf = data.dim == 2 and args["surf"]
        if write_surf:
            pts = _mesh_file(filename, ".pts")
            x = _read_rows(pts, (0, _read_count(pts)), 3).astype(np.float64)
        names = _marker_names(args["markers"], data.dim)
        for value in np.unique(values):
            entities = indices[values == value]
            name = names.get(int(value), str(value))
            nodes = np.unique(entities)
            _write_rows(
                _marker_file(filename, name, ".vtx"),
                [str(len(nodes)), args["vtx_domain"]],
                nodes,
                "%d",
            )
            if write_surf:
                triangles = _oriented_triangles(x, cells, entities)
                _write_rows(
                    _marker_file(filename, name, ".surf"),
                    [str(len(triangles))],
                    triangles,
                    "Tr %d %d %d",
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
            triangles of all ``<base>_*.surf`` files, otherwise the name of a marker:
            the triangles of ``<base>_<name>.surf`` or, if there is none, the nodes of
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
        cells, tags = _read_elem(elem, compute_local_range(comm, _read_count(elem)))
        return MeshTagsData(name=name, values=tags, indices=cells, dim=3)

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

    # Marker files are small, so rank 0 reads them all
    num_nodes = _nodes_per_entity[dim]
    indices = np.zeros((0, num_nodes), dtype=np.int64)
    values = np.zeros(0, dtype=np.int32)
    if comm.rank == 0:
        all_indices, all_values = [], []
        for path in files:
            marker = path.name[len(base.name) + 1 : -len(path.suffix)]
            if dim == 2:
                rows = _read_rows(path, (0, _read_count(path)), 4)
                entities = rows[:, 1:4].astype(np.int64)
            else:
                rows = _read_rows(path, (0, _read_count(path)), 1, header=2)
                entities = rows.astype(np.int64).reshape(-1, 1)
            all_indices.append(entities)
            all_values.append(np.full(len(entities), _marker_value(args["markers"], marker)))
        indices = np.vstack(all_indices)
        values = np.concatenate(all_values).astype(np.int32)
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
        freedom within that cell.
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
    local_range = compute_local_range(comm, _read_count(elem))
    cells, _ = _read_elem(elem, local_range)
    rows = _read_rows(lon, local_range, 3 * num_directions).astype(np.float64)
    column = 3 * _directions.index(name)
    return cells, np.ascontiguousarray(rows[:, column : column + 3])


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

    Writing with :attr:`FileMode.write` gives the fiber direction. Writing again
    with :attr:`FileMode.append` adds the sheet direction.

    Args:
        filename: Base name of the files, or the path of one of them
        array_data: Cell data in input ordering, see the ``on_input_mesh`` argument
            of :func:`io4dolfinx.write_cell_data`
        comm: MPI communicator to open the file with
        time: Not used
        mode: Write for the fiber, append for the sheet direction
        backend_args: The backend arguments
    """
    if array_data.type != "Cell":
        raise NotImplementedError(
            "The openCARP backend can only write cell data (fibers and sheets)."
        )
    if not array_data.on_input_mesh:
        raise ValueError(
            "The openCARP backend writes cell data in the input cell ordering. Use "
            "io4dolfinx.write_cell_data(..., on_input_mesh=True) or "
            "io4dolfinx.backends.opencarp.write_opencarp."
        )
    num_cells, num_components = array_data.global_shape
    if num_components != 3:
        raise ValueError(f"Expected a 3D vector per cell, got {num_components} components.")
    directions = _gather_rows(comm, array_data.values, array_data.local_range, num_cells)
    lon = _mesh_file(filename, ".lon")
    elem = _mesh_file(filename, ".elem")

    def write():
        assert directions is not None
        check_file_exists(elem)
        if (num_elements := _read_count(elem)) != num_cells:
            raise ValueError(f"Got data for {num_cells} cells, but {elem} has {num_elements}.")
        norms = np.linalg.norm(directions, axis=1)
        if np.any(norms == 0):
            raise ValueError(f"{np.count_nonzero(norms == 0)} cells have a zero direction.")
        rows = directions / norms[:, None]

        if mode == FileMode.append:
            check_file_exists(lon)
            if _read_count(lon) != 1:
                raise ValueError(f"{lon} already holds fibers and sheets.")
            fibers = _read_rows(lon, (0, num_cells), 3).astype(np.float64)
            rows = np.hstack([fibers, rows])
        _write_rows(lon, [str(rows.shape[1] // 3)], rows, _float_format)

    _on_root(comm, write)
