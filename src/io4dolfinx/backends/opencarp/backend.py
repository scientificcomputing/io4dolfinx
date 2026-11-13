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
- ``<base>.markers.json``: the name, value and dimension of each marker written by
  io4dolfinx, so that reading gives the original values, and writing the mesh again
  removes the marker files of the old mesh. openCARP does not read it.

Each file is written by rank 0.

Writing a mesh with markers and fibers takes the generic io4dolfinx functions, called
in this order with ``backend="opencarp"``:

.. code-block:: python

    args = {"scale": 1000.0, "markers": {"ENDO": (1, 2), "EPI": (2, 2)}}  # mesh in mm
    io4dolfinx.write_mesh(
        "heart", mesh, backend="opencarp", backend_args=args, on_input_mesh=True
    )
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

The ordering of nodes and elements must be the same in every step, so pass the same
``on_input_mesh`` to every function:

- With ``on_input_mesh=True`` (as above), the files use the ordering of the input
  mesh and are the same whatever the number of processes. Use this for reproducible
  files.
- With ``on_input_mesh=False`` (the default), the files use the DOLFINx ordering,
  which depends on the partitioning.

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
  which otherwise use the value itself. When reading, it takes precedence over the
  values in ``<base>.markers.json``.
- ``"surf"``: write ``.surf`` files for facet tags of 3D meshes (default ``True``).
- ``"vtx_domain"``: domain written to ``.vtx`` files (default ``"extra"``, meaning
  indices into the whole mesh, as written; ``"intra"`` means the myocardium submesh).

SPDX License identifier: MIT

Copyright: Henrik Finsberg and Simula Research Laboratory
"""

import glob
from pathlib import Path
from typing import Any

from mpi4py import MPI

import basix
import dolfinx
import numpy as np
import numpy.typing as npt

from ...ascii_helpers import check_row_count, read_count, read_rows, write_rows
from ...comm_helpers import gather_concatenate, gather_rows, run_on_all, run_on_root
from ...structures import ArrayData, FunctionData, MeshData, MeshTagsData, ReadMeshData
from ...utils import check_file_exists, compute_local_range, match_rows, strip_suffix
from .. import FileMode, ReadMode
from . import helpers

read_mode = ReadMode.parallel


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

    Supports triangle and quadrilateral meshes in 2D (also with 3D coordinates in a
    plane of constant z), and tetrahedral and hexahedral meshes in 3D. Every cell gets
    the region tag ``backend_args["default_tag"]``, until cell tags are written with
    :func:`write_meshtags`. Elements with the opposite orientation of openCARP's
    ``mesher`` are mirrored. The ``<base>.lon`` and marker files of an earlier mesh
    are removed.

    Args:
        filename: Base name of the files, or the path of one of them
        comm: MPI communicator used in storage
        mesh: Mesh data, from :func:`io4dolfinx.write_mesh`
        backend_args: Arguments to backend, ``"scale"`` is required
        mode: Only :attr:`FileMode.write` is supported
        time: Not used
    """
    args = get_default_backend_args(backend_args)
    if mesh.cell_type not in helpers.element_types or mesh.degree != 1:
        raise NotImplementedError(
            "The openCARP backend supports linear triangle, quadrilateral, tetrahedral and "
            f"hexahedral meshes, got a {mesh.cell_type} mesh of degree {mesh.degree}."
        )
    elem_type = helpers.element_types[mesh.cell_type]
    gdim = mesh.local_geometry.shape[1]
    if mode != FileMode.write:
        raise NotImplementedError("The openCARP backend can only write a mesh to new files.")
    scale = helpers.get_scale(args)

    x = gather_rows(comm, mesh.local_geometry, mesh.local_geometry_pos, mesh.num_nodes_global)
    cells = gather_rows(comm, mesh.local_topology, mesh.local_topology_pos, mesh.num_cells_global)

    def write():
        assert x is not None and cells is not None
        points = np.zeros((len(x), 3), dtype=x.dtype)
        points[:, :gdim] = x
        if helpers.element_tdim[elem_type] == 2 and np.ptp(points[:, 2]) > 0:
            raise NotImplementedError(
                f"The {mesh.cell_type} mesh does not lie in a plane of constant z. Surface "
                "meshes in 3D are not supported."
            )
        elements = cells[:, helpers.dolfinx_to_opencarp[elem_type]]
        orientation = helpers.orientation(elem_type, points, elements)
        if np.any(orientation == 0):
            raise ValueError(f"The mesh has {np.count_nonzero(orientation == 0)} degenerate cells.")
        flip = orientation < 0
        elements[flip] = elements[flip][:, helpers.mirror_permutation[elem_type]]

        write_rows(
            helpers.mesh_file(filename, ".pts"),
            [str(len(points))],
            points * scale,
            helpers.float_format,
        )
        tags = np.full(len(elements), args["default_tag"], dtype=np.int32)
        helpers.write_elem(helpers.mesh_file(filename, ".elem"), elem_type, elements, tags)
        helpers.mesh_file(filename, ".lon").unlink(missing_ok=True)
        helpers.remove_marker_files(filename, list(helpers.read_marker_index(filename)))
        helpers.write_marker_index(filename, {})

    run_on_root(comm, write)


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
    scale = helpers.get_scale(args)
    pts = helpers.mesh_file(filename, ".pts")
    elem = helpers.mesh_file(filename, ".elem")
    check_file_exists(pts)
    check_file_exists(elem)
    elem_type = helpers.read_elem_type(elem)

    def read():
        x = read_rows(pts, 3, compute_local_range(comm, read_count(pts)), dtype=np.float64)
        _, cells, _ = helpers.read_elem(elem, compute_local_range(comm, read_count(elem)))
        return x, cells

    x, cells = run_on_all(comm, read)
    if helpers.element_tdim[elem_type] == 2:
        z_min = comm.allreduce(x[:, 2].min(initial=np.inf), op=MPI.MIN)
        z_max = comm.allreduce(x[:, 2].max(initial=-np.inf), op=MPI.MAX)
        if z_max > z_min:
            raise NotImplementedError(
                f"{elem} has 2D elements that do not lie in a plane of constant z. Surface "
                "meshes in 3D are not supported."
            )
        x = x[:, :2]
    return ReadMeshData(
        cells=cells[:, helpers.opencarp_to_dolfinx[elem_type]],
        cell_type=helpers.cell_types[elem_type],
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
      A facet between two elements points out of the element whose centre comes
      first in lexicographic ``(x, y, z)`` order.

    ``<marker>`` is the name of the value in ``backend_args["markers"]``, or the
    value itself. The markers are recorded in ``<base>.markers.json``, and replace
    the markers of the same dimension written earlier.

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
    indices = gather_concatenate(comm, local_indices.reshape(-1, num_nodes))
    values = gather_concatenate(comm, np.asarray(data.values, dtype=np.int32))
    elem = helpers.mesh_file(filename, ".elem")

    def write():
        assert indices is not None and values is not None
        check_file_exists(elem)
        elem_type, cells, tags = helpers.read_elem(elem)
        if data.dim == helpers.element_tdim[elem_type]:
            rows = match_rows(cells, indices)
            if np.any(rows < 0):
                raise ValueError(f"{np.count_nonzero(rows < 0)} tagged cells are not in {elem}.")
            tags[rows] = values
            helpers.write_elem(elem, elem_type, cells, tags)
            return

        names = helpers.marker_names(args["markers"], data.dim)
        marker_values = np.unique(values)
        marker_names = [names.get(int(value), str(value)) for value in marker_values]
        if len(set(marker_names)) < len(marker_names):
            raise ValueError(
                f"Marker values {marker_values.tolist()} of dimension {data.dim} get the "
                f"file names {marker_names}, which are not unique."
            )
        index = helpers.read_marker_index(filename)
        replaced = [name for name, marker in index.items() if marker["dim"] == data.dim]
        if clash := sorted(set(marker_names) & (set(index) - set(replaced))):
            index_file = helpers.mesh_file(filename, ".markers.json")
            raise ValueError(
                f"Markers {clash} of dimension {data.dim} have the same name as markers of "
                f"another dimension in {index_file}."
            )
        helpers.remove_marker_files(filename, replaced)
        index = {name: marker for name, marker in index.items() if name not in replaced}

        write_surf = args["surf"] and helpers.element_tdim[elem_type] == 3 and data.dim == 2
        if write_surf:
            pts = helpers.mesh_file(filename, ".pts")
            x = read_rows(pts, 3, dtype=np.float64)
            face_type = helpers.face_element_types[elem_type]
            # DOLFINx orders the nodes of a quadrilateral facet as a tensor product
            facets = indices[:, helpers.dolfinx_to_opencarp[face_type]]
            facets = helpers.oriented_facets(x, elem_type, cells, facets)
        for value, name in zip(marker_values, marker_names):
            marked = values == value
            nodes = np.unique(indices[marked])
            write_rows(
                helpers.marker_file(filename, name, ".vtx"),
                [str(len(nodes)), args["vtx_domain"]],
                nodes,
                "%d",
            )
            if write_surf:
                faces = helpers.sort_facets(facets[marked])
                helpers.write_elem(helpers.marker_file(filename, name, ".surf"), face_type, faces)
            index[name] = {"value": int(value), "dim": int(data.dim), "surf": bool(write_surf)}
        helpers.write_marker_index(filename, index)

    run_on_root(comm, write)


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
            faces of the ``.surf`` files of all facet markers in
            ``<base>.markers.json``, otherwise the name of a marker: the faces of
            ``<base>_<name>.surf`` or, if there is none, the nodes of
            ``<base>_<name>.vtx``.
        backend_args: Arguments to backend. The value of a marker is taken from
            ``"markers"``, else from ``<base>.markers.json``, else from its name if
            that is an integer, else it is ``1``.

    Returns:
        Internal data structure for the mesh tags read from file
    """
    args = get_default_backend_args(backend_args)
    if name == "cell":
        elem = helpers.mesh_file(filename, ".elem")
        check_file_exists(elem)
        elem_type, cells, tags = run_on_all(
            comm, lambda: helpers.read_elem(elem, compute_local_range(comm, read_count(elem)))
        )
        cells = cells[:, helpers.opencarp_to_dolfinx[elem_type]]
        return MeshTagsData(
            name=name, values=tags, indices=cells, dim=helpers.element_tdim[elem_type]
        )

    base = strip_suffix(filename, helpers.mesh_suffixes)
    index = helpers.read_marker_index(filename)
    if name == "facet":
        files = [
            helpers.marker_file(filename, marker, ".surf")
            for marker, entry in sorted(index.items())
            if entry["dim"] == 2 and entry["surf"]
        ]
        if len(files) == 0:
            index_file = helpers.mesh_file(filename, ".markers.json")
            raise FileNotFoundError(
                f"{index_file} lists no facet markers with .surf files. "
                "Read the markers of other files by name."
            )
        dim = 2
    elif (surf := helpers.marker_file(filename, name, ".surf")).exists():
        files, dim = [surf], 2
    elif (vtx := helpers.marker_file(filename, name, ".vtx")).exists():
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
                if read_count(path) == 0:
                    entities = np.zeros((0, 3), dtype=np.int64)
                else:
                    face_type, faces, _ = helpers.read_elem(path)
                    if face_type not in helpers.face_element_types.values():
                        raise NotImplementedError(f"Unsupported face type {face_type} in {path}.")
                    entities = faces[:, helpers.opencarp_to_dolfinx[face_type]]
            else:
                entities = read_rows(path, 1, header=2, dtype=np.int64)
            all_indices.append(entities)
            all_values.append(
                np.full(len(entities), helpers.marker_value(args["markers"], index, marker))
            )
        return np.vstack(all_indices), np.concatenate(all_values).astype(np.int32)

    indices, values = np.zeros((0, 1), dtype=np.int64), np.zeros(0, dtype=np.int32)
    result = run_on_root(comm, read)
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
    lon = helpers.mesh_file(filename, ".lon")
    elem = helpers.mesh_file(filename, ".elem")
    check_file_exists(lon)
    check_file_exists(elem)
    num_directions = read_count(lon)
    if name not in helpers.direction_names[:num_directions]:
        raise KeyError(
            f"Could not find '{name}' in {lon}. "
            f"Available: {list(helpers.direction_names[:num_directions])}"
        )

    run_on_root(comm, lambda: check_row_count(lon, read_count(elem)))

    def read():
        local_range = compute_local_range(comm, read_count(elem))
        elem_type, cells, _ = helpers.read_elem(elem, local_range)
        rows = read_rows(lon, 3 * num_directions, local_range, dtype=np.float64)
        return elem_type, cells, rows

    elem_type, cells, rows = run_on_all(comm, read)
    column = 3 * helpers.direction_names.index(name)
    dofs = rows[:, column : column + helpers.element_tdim[elem_type]]
    cells = cells[:, helpers.opencarp_to_dolfinx[elem_type]]
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
    lon = helpers.mesh_file(filename, ".lon")
    check_file_exists(lon)
    return list(helpers.direction_names[: read_count(lon)])


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
    if np.iscomplexobj(array_data.values):
        raise NotImplementedError("openCARP directions are real, got complex cell data.")
    num_cells, num_components = array_data.global_shape
    if num_components not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D vector per cell, got {num_components} components.")
    directions = gather_rows(comm, array_data.values, array_data.local_range, num_cells)
    lon = helpers.mesh_file(filename, ".lon")
    elem = helpers.mesh_file(filename, ".elem")

    def write():
        assert directions is not None
        check_file_exists(elem)
        if (num_elements := read_count(elem)) != num_cells:
            raise ValueError(f"Got data for {num_cells} cells, but {elem} has {num_elements}.")
        if num_components < (tdim := helpers.element_tdim[helpers.read_elem_type(elem)]):
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
            if read_count(lon) != 1:
                raise ValueError(f"{lon} already holds fibers and sheets.")
            check_row_count(lon, num_cells)
            fibers = read_rows(lon, 3, (0, num_cells), dtype=np.float64)
            rows = np.hstack([fibers, rows])
        write_rows(lon, [str(rows.shape[1] // 3)], rows, helpers.float_format)

    run_on_root(comm, write)
