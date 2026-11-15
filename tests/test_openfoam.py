"""Tests of the OpenFOAM backend, on cases written from DOLFINx meshes.

The cases are written here from scratch, so that no files from OpenFOAM, which is GPL, are
used.
"""

import functools
import gzip
import shutil
from collections.abc import Callable
from pathlib import Path

from mpi4py import MPI

import basix
import dolfinx
import fenicsx_compat
import numpy as np
import numpy.typing as npt
import pytest
import ufl
from dolfinx.mesh import CellType

import io4dolfinx

pyvista = pytest.importorskip("pyvista")
openfoam = io4dolfinx.backends.get_backend("openfoam")


def cell_faces(
    points: npt.NDArray[np.floating], nodes: npt.ArrayLike, cell_type: str
) -> list[tuple[int, ...]]:
    """The faces of a first-order cell, with their normals out of the cell.

    basix gives, for each face of the reference cell, whether its vertex order gives an
    inward normal. A cell that is a mirror image of the reference cell, which DOLFINx
    allows, has all of these reversed.

    Args:
        points: Coordinates of the points of the mesh
        nodes: The points of the cell, in DOLFINx order
        cell_type: Name of the cell type, e.g. ``"hexahedron"``

    Returns:
        The point indices of each face, going round the face so that its normal points out
        of the cell, as OpenFOAM requires
    """
    nodes = np.asarray(nodes, dtype=np.int64)
    cell = getattr(basix.CellType, cell_type)
    inward = basix.cell.facet_orientations(cell)
    element = basix.create_element(basix.ElementFamily.P, cell, 1, basix.LagrangeVariant.equispaced)
    dphi = element.tabulate(1, np.zeros((1, 3)))[1:, 0, :, 0]
    mirrored = np.linalg.det(points[nodes].T @ dphi.T) < 0
    faces = []
    for local, flip in zip(basix.topology(cell)[2], inward):
        # basix stores quadrilaterals in tensor order, with vertex 3 opposite vertex 0
        if len(local) == 4:
            local = [local[0], local[1], local[3], local[2]]
        face = nodes[local]
        faces.append(tuple(int(v) for v in (face[::-1] if bool(flip) != mirrored else face)))
    return faces


def cells_as_faces(
    mesh: dolfinx.mesh.Mesh,
) -> tuple[npt.NDArray[np.floating], list[list[tuple[int, ...]]]]:
    """The points and cells of a first-order DOLFINx mesh, as an OpenFOAM case needs them.

    Args:
        mesh: The mesh, on one process

    Returns:
        The coordinates of the points, and the faces of each cell (see :func:`cell_faces`)
    """
    points = mesh.geometry.x
    cell_type = mesh.topology.cell_type.name
    return points, [cell_faces(points, nodes, cell_type) for nodes in fenicsx_compat.dofmap(mesh)]


def num(value: float) -> str:
    return f"{float(value):.17g}"


def header(cls: str, obj: str, binary: bool = False, label_bits: int = 32) -> str:
    fmt = "binary" if binary else "ascii"
    arch = f'    arch        "LSB;label={label_bits};scalar=64";\n' if binary else ""
    return (
        f"FoamFile\n{{\n    format      {fmt};\n{arch}    class       {cls};\n"
        f"    object      {obj};\n}}\n\n"
    )


def label_list(labels: npt.ArrayLike, binary: bool, label_bits: int) -> bytes:
    labels = np.asarray(labels, dtype=np.int64)
    if binary:
        dtype = "<i8" if label_bits == 64 else "<i4"
        return f"{len(labels)}(".encode() + labels.astype(dtype).tobytes() + b")"
    return f"{len(labels)}(".encode() + " ".join(map(str, labels)).encode() + b")"


def write_case(
    case: Path,
    points: npt.NDArray[np.floating],
    cells: list[list[tuple[int, ...]]],
    patch_of: Callable[[npt.NDArray[np.floating], int], str],
    fields: dict[float, dict[str, npt.ArrayLike]] | None = None,
    point_fields: dict[float, dict[str, npt.ArrayLike]] | None = None,
    cell_zones: dict[str, npt.ArrayLike] | None = None,
    face_zones: dict[str, Callable[[npt.NDArray[np.floating]], bool]] | None = None,
    point_zones: dict[str, npt.ArrayLike] | None = None,
    patch_entries: dict[str, dict[str, str]] | None = None,
    binary: bool = False,
    label_bits: int = 32,
    compress: bool = False,
) -> dict[str, int]:
    """Write an OpenFOAM case on rank 0.

    Args:
        case: The case directory
        points: Coordinates of the points
        cells: Faces of each cell, as point labels with outward normals
        patch_of: Name of the patch of a boundary face, from its centre and cell
        fields: Map from time to field name to values per cell
        point_fields: Map from time to field name to values per point
        cell_zones: Map from cell zone name to cell labels
        face_zones: Map from face zone name to whether a face, by its centre, is in it
        point_zones: Map from point zone name to point labels
        patch_entries: Extra entries of a patch in the ``boundary`` file
        binary: Write ``faces`` as a binary ``faceCompactList``, and the zones binary
        label_bits: Size of the labels in binary files
        compress: Gzip ``faces`` and the zones

    Returns:
        The number of faces of each patch
    """
    seen: dict[tuple[int, ...], int] = {}
    faces: list[tuple[int, ...]] = []
    owner: list[int] = []
    neighbour: list[int] = []
    for c, cell in enumerate(cells):
        for face in cell:
            key = tuple(sorted(face))
            if key in seen:
                neighbour[seen[key]] = c
            else:
                seen[key] = len(faces)
                faces.append(face)
                owner.append(c)
                neighbour.append(-1)
    owner_a, neighbour_a = np.array(owner), np.array(neighbour)
    internal = np.flatnonzero(neighbour_a >= 0)
    internal = internal[np.lexsort((neighbour_a[internal], owner_a[internal]))]
    boundary = np.flatnonzero(neighbour_a < 0)
    names = [patch_of(points[list(faces[f])].mean(axis=0), owner[f]) for f in boundary]
    patches = list(dict.fromkeys(names))
    boundary = boundary[np.argsort([patches.index(n) for n in names], kind="stable")]
    order = np.concatenate([internal, boundary]).astype(int)
    counts = {name: names.count(name) for name in patches}
    if MPI.COMM_WORLD.rank == 0:
        mesh = case / "constant" / "polyMesh"
        mesh.mkdir(parents=True)
        (case / "system").mkdir()
        (case / "system" / "controlDict").write_text(
            header("dictionary", "controlDict") + "application none;\nstartTime 0;\n"
        )

        def write(path: Path, data: bytes, zipped: bool = False):
            if zipped:
                path.with_name(path.name + ".gz").write_bytes(gzip.compress(data))
            else:
                path.write_bytes(data)

        rows = "\n".join(f"({' '.join(map(num, p))})" for p in points)
        write(
            mesh / "points",
            (header("vectorField", "points") + f"{len(points)}\n(\n{rows}\n)\n").encode(),
        )
        ordered = [faces[f] for f in order]
        if binary:
            offsets = np.concatenate([[0], np.cumsum([len(f) for f in ordered])])
            labels = np.concatenate([np.array(f) for f in ordered])
            data = header("faceCompactList", "faces", True, label_bits).encode()
            data += (
                label_list(offsets, True, label_bits) + b"\n" + label_list(labels, True, label_bits)
            )
        else:
            rows = "\n".join(f"{len(f)}({' '.join(map(str, f))})" for f in ordered)
            data = (header("faceList", "faces") + f"{len(ordered)}\n(\n{rows}\n)\n").encode()
        write(mesh / "faces", data, compress)
        for name, values in (("owner", owner_a[order]), ("neighbour", neighbour_a[internal])):
            rows = "\n".join(map(str, values))
            write(
                mesh / name, (header("labelList", name) + f"{len(values)}\n(\n{rows}\n)\n").encode()
            )
        start, entries = len(internal), []
        for name in patches:
            extra = "".join(
                f"        {k} {v};\n" for k, v in (patch_entries or {}).get(name, {}).items()
            )
            if "type" not in (patch_entries or {}).get(name, {}):
                extra = "        type patch;\n" + extra
            entries.append(f"    {name}\n    {{\n{extra}        nFaces {counts[name]};\n"
                           f"        startFace {start};\n    }}")  # fmt: skip
            start += counts[name]
        write(
            mesh / "boundary",
            (
                header("polyBoundaryMesh", "boundary")
                + f"{len(patches)}\n(\n"
                + "\n".join(entries)
                + "\n)\n"
            ).encode(),
        )

        def write_zones(kind: str, zones: dict[str, npt.ArrayLike]):
            data = header("regIOobject", f"{kind}Zones", binary, label_bits).encode()
            data += f"{len(zones)}\n(\n".encode()
            for name, labels in zones.items():
                data += f"{name}\n{{\n    type {kind}Zone;\n    {kind}Labels List<label> ".encode()
                data += label_list(labels, binary, label_bits) + b";\n"
                if kind == "face":
                    # Whether each face points out of the zone, one byte each if binary
                    flips = np.arange(np.size(labels)) % 2
                    data += f"    flipMap List<bool> {len(flips)}(".encode()
                    data += (
                        flips.astype(np.uint8).tobytes()
                        if binary
                        else " ".join(map(str, flips)).encode()
                    )
                    data += b");\n"
                data += b"}\n"
            write(mesh / f"{kind}Zones", data + b")\n", compress)

        if cell_zones:
            write_zones("cell", cell_zones)
        if face_zones:
            centres_of = [points[list(f)].mean(axis=0) for f in ordered]
            write_zones(
                "face",
                {
                    n: [k for k, c in enumerate(centres_of) if inside(c)]
                    for n, inside in face_zones.items()
                },
            )
        if point_zones:
            write_zones("point", point_zones)
        for kind, data_by_time in (("vol", fields or {}), ("point", point_fields or {})):
            for time, named in data_by_time.items():
                folder = case / str(time)
                folder.mkdir(exist_ok=True)
                for name, values in named.items():
                    values = np.asarray(values, dtype=float)
                    vector = values.ndim == 2
                    cls = f"{kind}{'Vector' if vector else 'Scalar'}Field"
                    rows = "\n".join(
                        f"({' '.join(map(num, v))})" if vector else num(v) for v in values
                    )
                    bcs = "".join(
                        f"    {p}\n    {{\n        type zeroGradient;\n    }}\n" for p in patches
                    )
                    (folder / name).write_text(
                        header(cls, name)
                        + "dimensions [0 0 0 0 0 0 0];\n\n"
                        + f"internalField nonuniform List<{'vector' if vector else 'scalar'}> "
                        + f"{len(values)}\n(\n{rows}\n)\n;\n\n"
                        + f"boundaryField\n{{\n{bcs}}}\n"
                    )
        (case / "case.foam").touch()
    MPI.COMM_WORLD.barrier()
    return counts


def sides(centre: npt.NDArray[np.floating], cell: int) -> str:
    """The patch of a boundary face of the unit cube, one per side.

    Args:
        centre: The centre of the face
        cell: The cell of the face

    Returns:
        ``"xmin"``, ``"xmax"``, ``"ymin"``, ``"ymax"``, ``"zmin"`` or ``"zmax"``
    """
    for axis, name in enumerate("xyz"):
        if np.isclose(centre[axis], 0):
            return f"{name}min"
        if np.isclose(centre[axis], 1):
            return f"{name}max"
    raise ValueError(f"{centre} is not on the boundary")


def f(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """A field linear in the coordinates.

    Its average over a cell is its value at the cell centre, and its integral over the unit
    cube is 3.

    Args:
        x: Points, of shape ``(..., 3)``

    Returns:
        The field at the points
    """
    return x[..., 0] + 2 * x[..., 1] + 3 * x[..., 2]


def centres(
    points: npt.NDArray[np.floating], cells: list[list[tuple[int, ...]]]
) -> npt.NDArray[np.floating]:
    """The centre of each cell, the average of its points.

    Args:
        points: Coordinates of the points
        cells: The faces of each cell, as point indices

    Returns:
        The centres, of shape ``(num_cells, 3)``
    """
    return np.array(
        [points[sorted({v for face in cell for v in face})].mean(axis=0) for cell in cells]
    )


def integral(
    mesh: dolfinx.mesh.Mesh,
    expr: ufl.core.expr.Expr | float,
    measure: str,
    tags: dolfinx.mesh.MeshTags | None = None,
    tag: int | None = None,
    dtype: npt.DTypeLike = dolfinx.default_real_type,
) -> float:
    """The integral of an expression over a mesh, summed over all processes.

    Args:
        mesh: The mesh
        expr: The expression to integrate
        measure: Name of the measure, e.g. ``"dx"``, ``"ds"``, ``"dS"`` or ``"dP"``
        tags: Mesh tags of the measure
        tag: Value in ``tags`` to integrate over, all entities if ``None``
        dtype: Scalar type of the form

    Returns:
        The integral
    """
    m = ufl.Measure(measure, domain=mesh, subdomain_data=tags)
    form = dolfinx.fem.form(expr * (m(tag) if tag is not None else m), dtype=dtype)
    return mesh.comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)


def write_cube(path: Path, cell_type: CellType, n: int = 4) -> Path:
    """Write the unit cube from DOLFINx as an OpenFOAM case.

    The case has a patch per side, the cell zones ``"lower"`` (z < 1/2) and ``"upper"``,
    the face zone ``"middle"`` (x = 1/2), the point zone ``"corners"``, the cell fields
    ``T`` (scalar) and ``U`` (vector) at times 0 and 1, and the point field ``pT`` at time 1.

    Args:
        path: The case directory
        cell_type: The cell type of the mesh
        n: Number of cells along each side

    Returns:
        The ``.foam`` file of the case
    """
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_SELF, n, n, n, cell_type=cell_type)
    points, cells = cells_as_faces(mesh)
    c = centres(points, cells)
    U = np.column_stack([c[:, 0], -c[:, 1], 2 * c[:, 2]])
    write_case(
        path,
        points,
        cells,
        sides,
        fields={0: {"T": f(c), "U": U}, 1: {"T": 2 * f(c), "U": U}},
        point_fields={1: {"pT": 10 * points[:, 0]}},
        cell_zones={"lower": np.flatnonzero(c[:, 2] < 0.5), "upper": np.flatnonzero(c[:, 2] > 0.5)},
        face_zones={"middle": lambda centre: np.isclose(centre[0], 0.5)},
        point_zones={"corners": np.flatnonzero(np.all(np.isclose(points % 1, 0), axis=1))},
    )
    return path / "case.foam"


@pytest.mark.parametrize("cell_type", [CellType.tetrahedron, CellType.hexahedron, CellType.prism])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_read(shared_tmp_path, cell_type, dtype):
    path = write_cube(shared_tmp_path / "cube", cell_type)
    comm = MPI.COMM_WORLD
    args = {"dtype": dtype}
    mesh = io4dolfinx.read_mesh(path, comm, backend="openfoam", backend_args=args)
    assert mesh.geometry.x.dtype == dtype
    assert mesh.topology.cell_type == cell_type
    integrate = functools.partial(integral, mesh, dtype=dtype)

    # VTK reads in single precision
    tol = 1e-4 if dtype == np.float32 else 1e-6
    assert np.isclose(integrate(1, "dx"), 1, rtol=tol)

    # Cell and point fields at two times
    np.testing.assert_allclose(
        io4dolfinx.read_timestamps(path, comm, "T", backend="openfoam"), [0, 1]
    )
    np.testing.assert_allclose(
        io4dolfinx.read_timestamps(path, comm, "pT", backend="openfoam"), [1]
    )
    for t in (0, 1):
        T = io4dolfinx.read_cell_data(
            path, "T", mesh, time=t, backend="openfoam", backend_args=args
        )
        assert np.isclose(integrate(T, "dx"), 3 * (1 + t), rtol=tol)
    U = io4dolfinx.read_cell_data(path, "U", mesh, time=0, backend="openfoam", backend_args=args)
    for i, expected in enumerate((0.5, -0.5, 1)):
        assert np.isclose(integrate(U[i], "dx"), expected, rtol=tol)
    pT = io4dolfinx.read_point_data(path, "pT", mesh, time=1, backend="openfoam", backend_args=args)
    assert np.isclose(integrate(pT, "dx"), 5, rtol=tol)

    # Cell zones and regions
    ids = openfoam.read_ids(path, comm)
    cells = io4dolfinx.read_meshtags(path, mesh, "cell", backend="openfoam")
    for zone in ("lower", "upper"):
        assert np.isclose(integrate(1, "dx", cells, ids["cell"][zone]), 0.5, rtol=tol)
    regions = io4dolfinx.read_meshtags(path, mesh, "regions", backend="openfoam")
    assert np.isclose(integrate(1, "dx", regions, 1), 1, rtol=tol)

    # Point zones: the eight corners
    vertices = io4dolfinx.read_meshtags(path, mesh, "peak", backend="openfoam")
    assert np.isclose(integrate(1, "dP", vertices, ids["peak"]["corners"]), 8)

    # Surface, patches and face zones. Prism facets are of two types, which mesh tags and
    # facet integrals in DOLFINx do not support.
    if cell_type == CellType.prism:
        with pytest.raises(NotImplementedError, match="prism"):
            io4dolfinx.read_meshtags(path, mesh, "facet", backend="openfoam")
        return
    assert np.isclose(integrate(1, "ds"), 6, rtol=tol)
    facets = io4dolfinx.read_meshtags(path, mesh, "facet", backend="openfoam")
    for side in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
        assert np.isclose(integrate(1, "ds", facets, ids["facet"][side]), 1, rtol=tol)
    assert np.isclose(integrate(1, "dS", facets, ids["facet"]["middle"]), 1, rtol=tol)
    xmin = io4dolfinx.read_meshtags(path, mesh, "xmin", backend="openfoam")
    assert np.isclose(integrate(1, "ds", xmin, ids["facet"]["xmin"]), 1, rtol=tol)


def test_ids(shared_tmp_path):
    """Ids are numbered from 1 for each dimension, in the order of the files: for facets
    the patches, then the face zones."""
    path = write_cube(shared_tmp_path / "cube", CellType.hexahedron, n=2)
    ids = openfoam.read_ids(path, MPI.COMM_WORLD)
    assert list(ids) == ["cell", "facet", "peak", "regions"]
    assert ids["cell"] == {"lower": 1, "upper": 2}
    assert set(ids["facet"]) == {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax", "middle"}
    assert ids["facet"]["middle"] == 7
    assert ids["peak"] == {"corners": 1}
    assert ids["regions"] == {"": 1}


def test_errors(shared_tmp_path):
    """Cell fields are only read as cell data, and point fields as point data."""
    path = write_cube(shared_tmp_path / "cube", CellType.hexahedron, n=2)
    comm = MPI.COMM_WORLD
    with pytest.raises(ValueError, match="cell-centred"):
        openfoam.read_point_data(path, "T", comm, 1, None)
    with pytest.raises(ValueError, match="point field"):
        openfoam.read_cell_data(path, "pT", comm, 1, None)
    with pytest.raises(KeyError, match="No time step"):
        openfoam.read_cell_data(path, "T", comm, 0.5, None)
    with pytest.raises(KeyError, match="No cell field 'p'"):
        openfoam.read_cell_data(path, "p", comm, 0, None)
    with pytest.raises(KeyError, match="No point field 'pT' at time 0"):
        openfoam.read_point_data(path, "pT", comm, 0, None)
    with pytest.raises(KeyError, match="Unknown mesh tags"):
        openfoam.read_meshtags_data(path, comm, "inlet")


def test_overlapping_zones(shared_tmp_path):
    """Sets of a dimension that share entities cannot be combined into one mesh tag, but
    can be read one at a time."""
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_SELF, 2, 1, 1, cell_type=CellType.hexahedron)
    points, cells = cells_as_faces(mesh)
    write_case(
        shared_tmp_path / "c",
        points,
        cells,
        sides,
        cell_zones={"a": [0, 1], "b": [1]},
        face_zones={"inlet": lambda centre: np.isclose(centre[0], 0)},
    )
    path = shared_tmp_path / "c" / "case.foam"
    comm = MPI.COMM_WORLD
    with pytest.raises(ValueError, match="'a' and 'b' share entities"):
        openfoam.read_meshtags_data(path, comm, "cell")
    with pytest.raises(ValueError, match="'xmin' and 'inlet' share entities"):
        openfoam.read_meshtags_data(path, comm, "facet")
    for name, count in (("a", 2), ("b", 1), ("inlet", 1), ("xmin", 1)):
        tags = openfoam.read_meshtags_data(path, comm, name)
        assert comm.allreduce(len(tags.values), op=MPI.SUM) == count


@pytest.mark.parametrize("label_bits", [32, 64])
@pytest.mark.parametrize("compress", [False, True])
def test_binary_and_compressed_files(shared_tmp_path, label_bits, compress):
    """Binary (32- and 64-bit labels) and gzipped faces and zones are read as ASCII."""
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_SELF, 2, 2, 2, cell_type=CellType.hexahedron)
    points, cells = cells_as_faces(mesh)
    zones = {
        "cell_zones": {"half": [0, 2, 4, 6], "other": [1, 3, 5, 7]},
        "face_zones": {"middle": lambda centre: np.isclose(centre[0], 0.5)},
        "point_zones": {"first": [0, 2, 6, 8], "second": [4]},
    }
    write_case(shared_tmp_path / "ascii", points, cells, sides, **zones)
    write_case(
        shared_tmp_path / "binary",
        points,
        cells,
        sides,
        **zones,
        binary=True,
        label_bits=label_bits,
        compress=compress,
    )
    comm = MPI.COMM_WORLD
    for name in ("facet", "cell", "peak"):
        a = openfoam.read_meshtags_data(shared_tmp_path / "ascii" / "case.foam", comm, name)
        b = openfoam.read_meshtags_data(shared_tmp_path / "binary" / "case.foam", comm, name)
        np.testing.assert_array_equal(a.indices, b.indices)
        np.testing.assert_array_equal(a.values, b.values)
        assert comm.allreduce(len(a.values), op=MPI.SUM) > 0


def test_baffle(shared_tmp_path):
    """Two cells side by side, with separate but coincident points on the face between
    them and a patch on each side. The patches are told apart by their labels, which
    coordinates could not do."""
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_SELF, 2, 1, 1, cell_type=CellType.hexahedron)
    points, cells = cells_as_faces(mesh)
    c = centres(points, cells)
    # Give the cell on the right its own copies of the points at x = 1/2
    middle = np.flatnonzero(np.isclose(points[:, 0], 0.5))
    copies = {int(p): len(points) + i for i, p in enumerate(middle)}
    points = np.vstack([points, points[middle]])
    right = int(np.argmax(c[:, 0]))
    cells[right] = [tuple(copies.get(v, v) for v in face) for face in cells[right]]

    def patch_of(centre: npt.NDArray[np.floating], cell: int) -> str:
        if np.isclose(centre[0], 0.5):
            return "baffle_right" if cell == right else "baffle_left"
        return "outside"

    write_case(shared_tmp_path / "baffle", points, cells, patch_of)
    path = shared_tmp_path / "baffle" / "case.foam"
    mesh = io4dolfinx.read_mesh(path, MPI.COMM_WORLD, backend="openfoam")
    ids = openfoam.read_ids(path, MPI.COMM_WORLD)["facet"]
    # The x-coordinate of each cell centre: integrated over a baffle, which lies on the
    # boundary of the cell on its side only, it gives the side
    cell_x = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("DG", 0)))
    cell_x.interpolate(lambda x: x[0])
    for name, side in (("baffle_left", 0.25), ("baffle_right", 0.75)):
        tags = io4dolfinx.read_meshtags(path, mesh, name, backend="openfoam")
        assert np.isclose(integral(mesh, 1, "ds", tags, ids[name]), 1)
        assert np.isclose(integral(mesh, cell_x, "ds", tags, ids[name]), side)


def test_mixed_mesh(shared_tmp_path):
    """A hexahedron and a prism sharing a face are read as a mixed mesh, without data or
    tags."""
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_SELF, 1, 1, 1, cell_type=CellType.hexahedron)
    points, cells = cells_as_faces(mesh)
    # A prism on the face x = 1 of the cube
    face = [i for i in range(len(points)) if np.isclose(points[i, 0], 1)]
    bottom = sorted((i for i in face if np.isclose(points[i, 2], 0)), key=lambda i: points[i, 1])
    top = sorted((i for i in face if np.isclose(points[i, 2], 1)), key=lambda i: points[i, 1])
    points = np.vstack([points, [[2, 0, 0], [2, 0, 1]]])
    n = len(points)
    nodes = [bottom[0], n - 2, bottom[1], top[0], n - 1, top[1]]
    cells.append(cell_faces(points, nodes, "prism"))
    c = centres(points, cells)

    def patch_of(centre: npt.NDArray[np.floating], cell: int) -> str:
        return "walls"

    write_case(shared_tmp_path / "mixed", points, cells, patch_of, fields={0: {"T": f(c)}})
    path = shared_tmp_path / "mixed" / "case.foam"
    comm = MPI.COMM_WORLD
    # Data and tags are refused on any DOLFINx; the mesh itself needs DOLFINx 0.11
    with pytest.raises(NotImplementedError, match="several cell types"):
        openfoam.read_cell_data(path, "T", comm, 0, None)
    with pytest.raises(NotImplementedError, match="several cell types"):
        openfoam.read_meshtags_data(path, comm, "facet")
    if fenicsx_compat.before("0.11.0.dev0"):
        with pytest.raises(NotImplementedError, match="requires DOLFINx 0.11"):
            io4dolfinx.read_mesh(path, comm, backend="openfoam")
        return
    mesh = io4dolfinx.read_mesh(path, comm, backend="openfoam")
    assert mesh.topology.entity_types[3] == [CellType.hexahedron, CellType.prism]
    assert [comm.allreduce(im.size_local) for im in mesh.topology.index_maps(3)] == [1, 1]


def test_polyhedron(shared_tmp_path):
    """A hexahedron with its top face split in two is a polyhedron, which DOLFINx does not
    support."""
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
         [0.5, 0, 1], [0.5, 1, 1]],
        dtype=float,
    )  # fmt: skip
    # Points 8 and 9 split the top face 4-5-6-7 between the edges 4-5 and 7-6
    faces = [(0, 4, 7, 3), (1, 2, 6, 5), (0, 1, 5, 8, 4), (3, 7, 9, 6, 2), (0, 3, 2, 1),
             (4, 8, 9, 7), (8, 5, 6, 9)]  # fmt: skip
    write_case(shared_tmp_path / "poly", points, [faces], lambda c, cell: "walls")
    with pytest.raises(NotImplementedError, match="polyhedral"):
        io4dolfinx.read_mesh(
            shared_tmp_path / "poly" / "case.foam", MPI.COMM_WORLD, backend="openfoam"
        )


def write_regions(root: Path, shift: float = 0.0) -> Path:
    """Write a case with two regions from DOLFINx boxes.

    The region ``"solid"`` is [0, 1]x[0, 2]x[0, 1], and ``"fluid"`` the same box moved by
    ``1 + shift`` along x. Each has the cell field ``T``, and a mapped interface patch where
    they meet; only ``"fluid"`` has the cell field ``U``.

    Args:
        root: Directory to write the case in
        shift: Gap between the regions, 0 for regions that meet

    Returns:
        The ``.foam`` file of the case
    """
    case = root / "regions"
    for region, origin in (("solid", 0.0), ("fluid", 1.0 + shift)):
        box = dolfinx.mesh.create_box(
            MPI.COMM_SELF,
            [np.array([origin, 0, 0]), np.array([origin + 1, 2, 1])],
            [1, 2, 1],
            cell_type=CellType.hexahedron,
        )
        points, cells = cells_as_faces(box)
        c = centres(points, cells)
        other = "fluid" if region == "solid" else "solid"
        interface_x = 1 if region == "solid" else origin

        def patch_of(centre: npt.NDArray[np.floating], cell: int, interface_x=interface_x) -> str:
            return "interface" if np.isclose(centre[0], interface_x) else "walls"

        fields: dict[float, dict[str, npt.ArrayLike]] = {0: {"T": f(c)}}
        if region == "fluid":
            fields[0]["U"] = c
        write_case(
            root / region,
            points,
            cells,
            patch_of,
            fields=fields,
            cell_zones={"core": [0]},
            patch_entries={
                "interface": {
                    "type": "mappedWall",
                    "sampleRegion": other,
                    "samplePatch": "interface",
                }
            },
        )
    if MPI.COMM_WORLD.rank == 0:
        (case / "constant").mkdir(parents=True)
        for region in ("solid", "fluid"):
            shutil.copytree(
                root / region / "constant" / "polyMesh", case / "constant" / region / "polyMesh"
            )
            shutil.copytree(root / region / "0", case / "0" / region)
        shutil.copytree(root / "solid" / "system", case / "system")
        (case / "case.foam").touch()
    MPI.COMM_WORLD.barrier()
    return case / "case.foam"


def test_regions(shared_tmp_path):
    """Regions are merged into one conforming mesh; the facets between them are tagged
    once."""
    path = write_regions(shared_tmp_path)
    comm = MPI.COMM_WORLD
    mesh = io4dolfinx.read_mesh(path, comm, backend="openfoam")

    assert np.isclose(integral(mesh, 1, "dx"), 4)
    # Conforming: the boundary is that of the box [0, 2]x[0, 2]x[0, 1], without the interface
    assert np.isclose(integral(mesh, 1, "ds"), 2 * (4 + 2 + 2))

    ids = openfoam.read_ids(path, comm)
    assert ids["regions"] == {"fluid": 1, "solid": 2}
    regions = io4dolfinx.read_meshtags(path, mesh, "regions", backend="openfoam")
    for region in ("fluid", "solid"):
        assert np.isclose(integral(mesh, 1, "dx", regions, ids["regions"][region]), 2)

    # The interface is an interior facet set of area 2, tagged by the first interface patch
    facets = io4dolfinx.read_meshtags(path, mesh, "facet", backend="openfoam")
    assert np.isclose(integral(mesh, 1, "dS", facets, ids["facet"]["fluid/interface"]), 2)
    assert np.isclose(integral(mesh, 1, "dS", facets, ids["facet"]["solid/interface"]), 0)

    # T is linear, so its integral is exact: the integral of x + 2y + 3z over [0, 2]x[0, 2]x[0, 1]
    T = io4dolfinx.read_cell_data(path, "T", mesh, backend="openfoam")
    assert np.isclose(integral(mesh, T, "dx"), 4 + 8 + 6)
    with pytest.raises(KeyError, match="solid"):
        openfoam.read_cell_data(path, "U", comm, 0, None)

    solid = io4dolfinx.read_mesh(path, comm, backend="openfoam", backend_args={"region": "solid"})
    volume = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ufl.dx(domain=solid)))
    assert np.isclose(comm.allreduce(volume, op=MPI.SUM), 2)


def test_regions_must_match(shared_tmp_path):
    """Regions whose interfaces do not coincide are not merged into a broken mesh."""
    path = write_regions(shared_tmp_path, shift=1e-3)
    with pytest.raises(ValueError, match="do not coincide"):
        openfoam.read_meshtags_data(path, MPI.COMM_WORLD, "facet")
