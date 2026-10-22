import urllib.error
import urllib.request
from enum import Enum

from mpi4py import MPI

import dolfinx
import fenicsx_compat
import numpy as np
import pytest
import ufl

import io4dolfinx

h5py = pytest.importorskip("h5py")

from io4dolfinx.backends.med.backend import read_groups  # noqa: E402

# (DOLFINx cell type, geometry degree, serendipity) -> MED element type
_dolfinx_to_med = {
    ("interval", 1, False): "SE2",
    ("interval", 2, False): "SE3",
    ("interval", 2, True): "SE3",
    ("triangle", 1, False): "TR3",
    ("triangle", 2, False): "TR6",
    ("quadrilateral", 1, False): "QU4",
    ("quadrilateral", 2, False): "QU9",
    ("quadrilateral", 2, True): "QU8",
    ("tetrahedron", 1, False): "TE4",
    ("tetrahedron", 2, False): "T10",
    ("prism", 1, False): "PE6",
    ("prism", 2, False): "P18",
    ("hexahedron", 1, False): "HE8",
    ("hexahedron", 2, False): "H27",
    ("hexahedron", 2, True): "H20",
}

# MED node layouts, read off the Salome connectivity diagrams
# (https://docs.salome-platform.org/latest/gui/SMESH/connectivity.html) independently
# of the permutations in the backend: the DOLFINx vertices that are the MED vertices,
# and for each further node the MED vertices it is the centroid of.
_hex_edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)]
_hex_edges += [(0, 4), (1, 5), (2, 6), (3, 7)]
_hex_faces: list[tuple[int, ...]] = [
    (0, 1, 2, 3),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
]
_hex_faces += [(4, 5, 6, 7), tuple(range(8))]
_prism_edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)]
_quad_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
_med_layouts = {
    "SE2": ([0, 1], []),
    "SE3": ([0, 1], [(0, 1)]),
    "TR3": ([0, 1, 2], []),
    "TR6": ([0, 1, 2], [(0, 1), (1, 2), (2, 0)]),
    "QU4": ([0, 1, 3, 2], []),
    "QU8": ([0, 1, 3, 2], _quad_edges),
    "QU9": ([0, 1, 3, 2], _quad_edges + [(0, 1, 2, 3)]),
    "TE4": ([0, 1, 2, 3], []),
    "T10": ([0, 1, 2, 3], [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]),
    "PE6": ([0, 1, 2, 3, 4, 5], []),
    # The diagram stops at PE15; the face nodes follow the side faces in order
    "P18": ([0, 1, 2, 3, 4, 5], _prism_edges + [(0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)]),
    "HE8": ([0, 1, 3, 2, 4, 5, 7, 6], []),
    "H20": ([0, 1, 3, 2, 4, 5, 7, 6], _hex_edges),
    "H27": ([0, 1, 3, 2, 4, 5, 7, 6], _hex_edges + _hex_faces),
}

_festim_url = (
    "https://raw.githubusercontent.com/festim-dev/FESTIM-workshop/"
    "e2f171484578af33ea1c4030016a5a600768a1ec/book/content/meshes/salome/Mesh_1.med"
)


class DownloadStatus(Enum):
    success = 1
    failed = -1
    no_connection = -2


def download_file_if_not_exists(
    url, filename, comm: MPI.Intracomm = MPI.COMM_WORLD, rank: int = 0
) -> DownloadStatus:
    status = DownloadStatus.failed
    if comm.rank == rank:
        if not filename.exists():
            try:
                urllib.request.urlretrieve(url, filename)
                status = DownloadStatus.success
            except urllib.error.URLError as e:
                if str(e) == "<urlopen error [Errno -3] Temporary failure in name resolution>":
                    status = DownloadStatus.no_connection
                else:
                    status = DownloadStatus.failed
        else:
            status = DownloadStatus.success
    status = comm.bcast(status, root=rank)
    comm.Barrier()
    return status


def _fixture_mesh(cell_type: dolfinx.mesh.CellType, degree: int = 1) -> dolfinx.mesh.Mesh:
    """Serial mesh of the unit square/cube, with geometry of the given degree."""
    if dolfinx.mesh.cell_dim(cell_type) == 2:
        mesh = dolfinx.mesh.create_unit_square(MPI.COMM_SELF, 4, 4, cell_type)
    else:
        mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_SELF, 2, 2, 2, cell_type)
    if degree > 1:
        if hasattr(dolfinx.fem, "interpolate_geometry"):
            cmap = dolfinx.fem.coordinate_element(cell_type, degree, dtype=mesh.geometry.x.dtype)
            mesh = dolfinx.fem.interpolate_geometry(mesh, cmap)
        else:
            mesh = fenicsx_compat.reconstruct_mesh(mesh, degree)
    return mesh


def _to_med(nodes, x, med_type):
    """MED connectivity of entities given by their DOLFINx geometry nodes.

    Nodes are found by position, so ``x`` must be undeformed."""
    vertices, centroids = _med_layouts[med_type]
    med_vertices = nodes[:, vertices]
    columns = [med_vertices]
    rows = np.arange(nodes.shape[0])
    for centroid in centroids:
        target = x[med_vertices[:, centroid]].mean(axis=1)
        distance = np.linalg.norm(x[nodes] - target[:, None, :], axis=2)
        closest = distance.argmin(axis=1)
        assert np.allclose(distance[rows, closest], 0.0)
        columns.append(nodes[rows, closest][:, None])
    return np.hstack(columns)


def _deform(x):
    """Quadratic map, represented exactly by quadratic Lagrange and serendipity geometry."""
    y = x.copy()
    for i in range(x.shape[1]):
        y[:, i] += 0.1 * x[:, (i + 1) % x.shape[1]] ** 2
    return y


def _med_blocks(mesh, cell_family, facet_markers=(), serendipity=False, deform=None):
    """Coordinates and MED blocks ``{type: (0-based nodes, family)}`` of a serial mesh.

    ``cell_family`` maps cell midpoints to families and ``facet_markers`` is a sequence of
    ``(boundary marker, family)``. Serendipity drops the face and interior nodes of a
    quadratic mesh. ``deform`` maps the coordinates after the connectivity is built."""
    tdim = mesh.topology.dim
    x = mesh.geometry.x[:, : mesh.geometry.dim].copy()
    degree = fenicsx_compat.cmap(mesh).degree
    cell_type = dolfinx.mesh.to_string(mesh.topology.cell_type)
    dofmap = fenicsx_compat.dofmap(mesh)
    med_type = _dolfinx_to_med[(cell_type, degree, serendipity)]
    blocks = {med_type: (_to_med(dofmap, x, med_type), cell_family(x[dofmap].mean(axis=1)))}
    if len(facet_markers) > 0:
        mesh.topology.create_connectivity(tdim - 1, tdim)
        facets, families = [], []
        for marker, family in facet_markers:
            entities = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, marker)
            facets.append(entities)
            families.append(np.full(len(entities), family, dtype=np.int32))
        nodes = dolfinx.mesh.entities_to_geometry(mesh, tdim - 1, np.hstack(facets))
        facet_type = dolfinx.cpp.mesh.cell_entity_type(mesh.topology.cell_type, tdim - 1, 0)
        med_type = _dolfinx_to_med[(dolfinx.mesh.to_string(facet_type), degree, serendipity)]
        blocks[med_type] = (_to_med(nodes, x, med_type), np.hstack(families))

    # Drop nodes no entity uses
    used = np.unique(np.hstack([nodes.ravel() for nodes, _ in blocks.values()]))
    new_index = np.full(x.shape[0], -1, dtype=np.int64)
    new_index[used] = np.arange(len(used))
    blocks = {key: (new_index[nodes], family) for key, (nodes, family) in blocks.items()}
    x = x[used]
    return (x if deform is None else deform(x)), blocks


def _write_med(
    filename,
    x,
    blocks,
    families=None,
    node_families=None,
    mesh_name="mesh",
    mode="w",
    node_num=None,
    mesh_type=0,
    time=0.0,
    later_steps=(),
):
    """Write a MED file (call on a single rank).

    ``mesh_type`` is the MED mesh type, 0 for unstructured and 1 for structured. The
    first step is at ``time``. ``later_steps`` is a sequence of ``(step number, time,
    coordinates)``; as MED does, a later step stores its coordinates and links its
    connectivity and families to the previous step."""
    with h5py.File(filename, mode) as f:
        info = f.require_group("INFOS_GENERALES")
        for key, value in zip(("MAJ", "MIN", "REL"), (4, 1, 1)):
            info.attrs[key] = np.int32(value)
        mesh = f.require_group("ENS_MAA").create_group(mesh_name)
        mesh.attrs["ESP"] = np.int32(x.shape[1])
        mesh.attrs["TYP"] = np.int32(mesh_type)
        previous = None
        for number, step_time, step_x in [(-1, time, x), *later_steps]:
            # Steps are named by step and iteration number, the first being (-1, -1)
            step = mesh.create_group(f"{number:020d}{-1:020d}")
            step.attrs["NDT"] = np.int32(number)
            step.attrs["NOR"] = np.int32(-1)
            step.attrs["PDT"] = np.float64(step_time)
            noe = step.create_group("NOE")
            coo = noe.create_dataset("COO", data=step_x.flatten(order="F"))
            coo.attrs["NBR"] = np.int32(step_x.shape[0])
            noe.create_dataset("FAM", data=np.zeros(step_x.shape[0], dtype=np.int32))
            if node_num is not None:
                noe.create_dataset("NUM", data=node_num.astype(np.int32))
            mai = step.create_group("MAI")
            for med_type, (nodes, family) in blocks.items():
                block = mai.create_group(med_type)
                if previous is not None:
                    for name in ("NOD", "FAM"):
                        block[name] = h5py.SoftLink(f"{previous}/MAI/{med_type}/{name}")
                    continue
                nod = block.create_dataset(
                    "NOD", data=(nodes + 1).flatten(order="F").astype(np.int32)
                )
                nod.attrs["NBR"] = np.int32(nodes.shape[0])
                block.create_dataset("FAM", data=family.astype(np.int32))
            previous = step.name

        fas = f.require_group("FAS").create_group(mesh_name)
        fas.create_group("FAMILLE_ZERO").attrs["NUM"] = np.int32(0)
        for kind, kind_families in (("ELEME", families), ("NOEUD", node_families)):
            if kind_families is None:
                continue
            group = fas.create_group(kind)
            for family_id, names in kind_families.items():
                family = group.create_group(f"FAM_{family_id}_{'_'.join(names)}")
                family.attrs["NUM"] = np.int32(family_id)
                gro = family.create_group("GRO")
                gro.attrs["NBR"] = np.int32(len(names))
                nom = np.zeros((len(names), 80), dtype=np.int8)
                for i, name in enumerate(names):
                    encoded = name.encode()
                    nom[i, : len(encoded)] = np.frombuffer(encoded, dtype=np.int8)
                gro.create_dataset("NOM", data=nom)


def _on_root(comm, function, *args, **kwargs):
    if comm.rank == 0:
        function(*args, **kwargs)
    comm.Barrier()


def _integrate(mesh, measure, tag=None, meshtags=None):
    m = ufl.Measure(measure, domain=mesh, subdomain_data=meshtags)
    m = m if tag is None else m(tag)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * m))
    return mesh.comm.allreduce(value, op=MPI.SUM)


def _left_right(midpoints):
    return np.where(midpoints[:, 0] < 0.5, -2, -3).astype(np.int32)


_families = {-2: ["left"], -3: ["right"], -4: ["inlet"], -5: ["outlet"]}
_facet_markers = (
    (lambda x: np.isclose(x[0], 0.0), -4),
    (lambda x: np.isclose(x[0], 1.0), -5),
    (lambda x: np.isclose(x[1], 0.0), 0),
)


@pytest.fixture
def tmp_dir(tmp_path):
    return MPI.COMM_WORLD.bcast(tmp_path, root=0)


@pytest.mark.parametrize(
    "cell_type",
    [
        dolfinx.mesh.CellType.triangle,
        dolfinx.mesh.CellType.quadrilateral,
        dolfinx.mesh.CellType.tetrahedron,
        dolfinx.mesh.CellType.hexahedron,
    ],
)
def test_read_mesh_and_meshtags(tmp_dir, cell_type):
    comm = MPI.COMM_WORLD
    filename = tmp_dir / "mesh.med"
    ref = _fixture_mesh(cell_type)
    x, blocks = _med_blocks(ref, _left_right, _facet_markers)
    _on_root(comm, _write_med, filename, x, blocks, _families)

    mesh = io4dolfinx.read_mesh(filename, comm, backend="med")
    tdim = mesh.topology.dim
    assert mesh.topology.cell_type == cell_type
    assert mesh.topology.index_map(tdim).size_global == ref.topology.index_map(tdim).size_global
    assert mesh.geometry.index_map().size_global == x.shape[0]
    assert np.isclose(_integrate(mesh, "dx"), 1.0)

    ct = io4dolfinx.read_meshtags(filename, mesh, meshtag_name="cell", backend="med")
    assert np.isclose(_integrate(mesh, "dx", 2, ct), 0.5)
    assert np.isclose(_integrate(mesh, "dx", 3, ct), 0.5)

    ft = io4dolfinx.read_meshtags(filename, mesh, meshtag_name="facet", backend="med")
    assert ft.dim == tdim - 1
    assert np.isclose(_integrate(mesh, "ds", 4, ft), 1.0)
    assert np.isclose(_integrate(mesh, "ds", 5, ft), 1.0)
    num_untagged = comm.allreduce(np.count_nonzero(ft.values == 0), op=MPI.SUM)
    assert num_untagged == 0

    left = io4dolfinx.read_meshtags(filename, mesh, meshtag_name="left", backend="med")
    assert left.dim == tdim
    assert np.isclose(_integrate(mesh, "dx", 1, left), 0.5)
    outlet = io4dolfinx.read_meshtags(filename, mesh, meshtag_name="outlet", backend="med")
    assert outlet.dim == tdim - 1
    assert np.isclose(_integrate(mesh, "ds", 1, outlet), 1.0)

    groups = read_groups(filename, comm)
    assert groups == {"left": [2], "right": [3], "inlet": [4], "outlet": [5]}


def test_prism(tmp_dir):
    comm = MPI.COMM_WORLD
    filename = tmp_dir / "prism.med"
    x, blocks = _med_blocks(_fixture_mesh(dolfinx.mesh.CellType.prism), _left_right)
    _on_root(comm, _write_med, filename, x, blocks, _families)

    mesh = io4dolfinx.read_mesh(filename, comm, backend="med")
    assert mesh.topology.cell_type == dolfinx.mesh.CellType.prism
    assert np.isclose(_integrate(mesh, "dx"), 1.0)
    ct = io4dolfinx.read_meshtags(filename, mesh, meshtag_name="right", backend="med")
    assert np.isclose(_integrate(mesh, "dx", 1, ct), 0.5)


@pytest.mark.parametrize("numbering", ["reversed", "none"])
def test_node_numbering_ignored(tmp_dir, numbering):
    """``NOE/NUM`` is optional user numbering, connectivity refers to node positions."""
    comm = MPI.COMM_WORLD
    filename = tmp_dir / "numbering.med"
    x, blocks = _med_blocks(_fixture_mesh(dolfinx.mesh.CellType.quadrilateral), _left_right)
    node_num = np.arange(x.shape[0], 0, -1) if numbering == "reversed" else None
    _on_root(comm, _write_med, filename, x, blocks, _families, node_num=node_num)

    mesh = io4dolfinx.read_mesh(filename, comm, backend="med")
    assert np.isclose(_integrate(mesh, "dx"), 1.0)
    x_mesh = ufl.SpatialCoordinate(mesh)
    first_moment = dolfinx.fem.assemble_scalar(dolfinx.fem.form(x_mesh[0] * ufl.dx))
    assert np.isclose(comm.allreduce(first_moment, op=MPI.SUM), 0.5)


def test_overlapping_groups(tmp_dir):
    comm = MPI.COMM_WORLD
    filename = tmp_dir / "overlap.med"

    def family(midpoints):
        return np.select([midpoints[:, 0] < 0.25, midpoints[:, 0] < 0.75], [-2, -3], -4)

    x, blocks = _med_blocks(_fixture_mesh(dolfinx.mesh.CellType.triangle), family)
    families = {-2: ["A"], -3: ["A", "B"], -4: ["B"]}
    _on_root(comm, _write_med, filename, x, blocks, families)

    mesh = io4dolfinx.read_mesh(filename, comm, backend="med")
    for name in ("A", "B"):
        tags = io4dolfinx.read_meshtags(filename, mesh, meshtag_name=name, backend="med")
        assert np.isclose(_integrate(mesh, "dx", 1, tags), 0.75)
    assert read_groups(filename, comm) == {"A": [2, 3], "B": [3, 4]}


def test_multiple_meshes(tmp_dir):
    comm = MPI.COMM_WORLD
    filename = tmp_dir / "two_meshes.med"
    square = _med_blocks(_fixture_mesh(dolfinx.mesh.CellType.triangle), _left_right)
    cube = _med_blocks(_fixture_mesh(dolfinx.mesh.CellType.hexahedron), _left_right)

    def write():
        _write_med(filename, square[0], square[1], _families, mesh_name="square")
        _write_med(filename, cube[0], cube[1], _families, mesh_name="cube", mode="a")

    _on_root(comm, write)

    with pytest.raises(KeyError, match="Available"):
        io4dolfinx.read_mesh(filename, comm, backend="med")
    mesh = io4dolfinx.read_mesh(filename, comm, backend="med", mesh_name="cube")
    assert mesh.topology.cell_type == dolfinx.mesh.CellType.hexahedron
    ct = io4dolfinx.read_meshtags(
        filename, mesh, meshtag_name="left", backend="med", mesh_name="cube"
    )
    assert np.isclose(_integrate(mesh, "dx", 1, ct), 0.5)


def test_unsupported(tmp_dir):
    comm = MPI.COMM_WORLD

    # Mixed cell types of the highest dimension
    mixed = tmp_dir / "mixed.med"
    x = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
    blocks = {
        "TE4": (np.array([[0, 1, 3, 4]]), np.zeros(1)),
        "PY5": (np.array([[0, 1, 2, 3, 5]]), np.zeros(1)),
    }
    _on_root(comm, _write_med, mixed, x, blocks)
    with pytest.raises(NotImplementedError, match="Mixed element types"):
        io4dolfinx.read_mesh(mixed, comm, backend="med")

    # Unknown group, and a group of nodes only
    groups = tmp_dir / "groups.med"
    x, blocks = _med_blocks(_fixture_mesh(dolfinx.mesh.CellType.triangle), _left_right)
    _on_root(comm, _write_med, groups, x, blocks, _families, node_families={2: ["corner"]})
    mesh = io4dolfinx.read_mesh(groups, comm, backend="med")
    with pytest.raises(KeyError, match="inlet"):
        io4dolfinx.read_meshtags(groups, mesh, meshtag_name="wall", backend="med")
    with pytest.raises(NotImplementedError, match="only holds nodes"):
        io4dolfinx.read_meshtags(groups, mesh, meshtag_name="corner", backend="med")

    # Cell block without entries
    empty = tmp_dir / "empty.med"
    _on_root(comm, _write_med, empty, x, {"TR3": (np.zeros((0, 3), dtype=np.int64), np.zeros(0))})
    with pytest.raises(ValueError, match="not a positive multiple"):
        io4dolfinx.read_mesh(empty, comm, backend="med")

    # Element type without a DOLFINx cell type (polygon)
    polygon = tmp_dir / "polygon.med"
    blocks["POG"] = (np.array([[0, 1, 2]]), np.zeros(1))
    _on_root(comm, _write_med, polygon, x, blocks)
    with pytest.raises(NotImplementedError, match=r"\['POG'\] are not supported"):
        io4dolfinx.read_mesh(polygon, comm, backend="med")


def _moments(mesh, facet_tags=None):
    """Volume, a second moment and the length/area of facets tagged 4 and 5."""
    x = ufl.SpatialCoordinate(mesh)
    second = dolfinx.fem.assemble_scalar(dolfinx.fem.form(x[0] * x[1] * ufl.dx))
    values = [_integrate(mesh, "dx"), mesh.comm.allreduce(second, op=MPI.SUM)]
    if facet_tags is not None:
        values += [_integrate(mesh, "ds", tag, facet_tags) for tag in (4, 5)]
    return np.array(values)


@pytest.mark.parametrize(
    "cell_type, serendipity",
    [
        (dolfinx.mesh.CellType.triangle, False),
        (dolfinx.mesh.CellType.quadrilateral, False),
        (dolfinx.mesh.CellType.quadrilateral, True),
        (dolfinx.mesh.CellType.tetrahedron, False),
        (dolfinx.mesh.CellType.prism, False),
        (dolfinx.mesh.CellType.hexahedron, False),
        (dolfinx.mesh.CellType.hexahedron, True),
    ],
)
def test_higher_order(tmp_dir, cell_type, serendipity):
    """Curved quadratic geometry is read back exactly, so a wrong node order shows."""
    comm = MPI.COMM_WORLD
    filename = tmp_dir / "curved.med"
    ref = _fixture_mesh(cell_type, degree=2)
    markers = _facet_markers if cell_type != dolfinx.mesh.CellType.prism else ()
    x, blocks = _med_blocks(ref, _left_right, markers, serendipity=serendipity, deform=_deform)
    _on_root(comm, _write_med, filename, x, blocks, _families)

    # Reference: the quadratic Lagrange mesh, deformed in place. Both layouts represent
    # the quadratic deformation exactly.
    ref_tags = None
    if len(markers) > 0:
        tdim = ref.topology.dim
        facets = [dolfinx.mesh.locate_entities_boundary(ref, tdim - 1, m) for m, _ in markers[:2]]
        values = [np.full(len(f), tag, dtype=np.int32) for f, tag in zip(facets, (4, 5))]
        order = np.argsort(np.hstack(facets))
        ref_tags = dolfinx.mesh.meshtags(
            ref, tdim - 1, np.hstack(facets)[order], np.hstack(values)[order]
        )
    gdim = ref.geometry.dim
    ref.geometry.x[:, :gdim] = _deform(ref.geometry.x[:, :gdim])
    expected = _moments(ref, ref_tags)

    mesh = io4dolfinx.read_mesh(filename, comm, backend="med")
    assert mesh.topology.cell_type == cell_type
    assert fenicsx_compat.cmap(mesh).dim == blocks[next(iter(blocks))][0].shape[1]
    ft = None
    if len(markers) > 0:
        ft = io4dolfinx.read_meshtags(filename, mesh, meshtag_name="facet", backend="med")
    assert np.allclose(_moments(mesh, ft), expected)


@pytest.mark.parametrize(
    "cell_type, degree, serendipity",
    [
        (dolfinx.mesh.CellType.triangle, 1, False),
        (dolfinx.mesh.CellType.tetrahedron, 2, False),
        (dolfinx.mesh.CellType.hexahedron, 2, True),
    ],
)
def test_no_facet_block(tmp_dir, cell_type, degree, serendipity):
    """Facet tags of a file without facets are empty."""
    comm = MPI.COMM_WORLD
    filename = tmp_dir / "no_facets.med"
    ref = _fixture_mesh(cell_type, degree=degree)
    x, blocks = _med_blocks(ref, _left_right, serendipity=serendipity)
    _on_root(comm, _write_med, filename, x, blocks, _families)

    mesh = io4dolfinx.read_mesh(filename, comm, backend="med")
    ft = io4dolfinx.read_meshtags(filename, mesh, meshtag_name="facet", backend="med")
    assert ft.dim == mesh.topology.dim - 1
    assert comm.allreduce(len(ft.values), op=MPI.SUM) == 0


def test_structured_mesh(tmp_dir):
    """A structured grid (``TYP=1``) has no connectivity and is rejected."""
    comm = MPI.COMM_WORLD
    filename = tmp_dir / "structured.med"
    x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    _on_root(comm, _write_med, filename, x, {}, mesh_type=1)
    with pytest.raises(NotImplementedError, match="Structured MED meshes"):
        io4dolfinx.read_mesh(filename, comm, backend="med")
    with pytest.raises(NotImplementedError, match="Structured MED meshes"):
        read_groups(filename, comm)


@pytest.mark.parametrize("med_type", ["TR7", "P13", "P15"])
def test_unsupported_layouts(tmp_dir, med_type):
    comm = MPI.COMM_WORLD
    filename = tmp_dir / "unsupported.med"
    num_nodes = int(med_type[1:]) if med_type[0] == "P" else 7
    x = np.random.default_rng(0).random((num_nodes, 3 if med_type[0] == "P" else 2))
    blocks = {med_type: (np.arange(num_nodes).reshape(1, -1), np.zeros(1))}
    _on_root(comm, _write_med, filename, x, blocks)
    with pytest.raises(NotImplementedError, match="no Lagrange or nodal serendipity"):
        io4dolfinx.read_mesh(filename, comm, backend="med")


def test_time_dependent(tmp_dir):
    """Later steps store new coordinates and link the connectivity to the first step."""
    comm = MPI.COMM_WORLD
    filename = tmp_dir / "time_dependent.med"
    x, blocks = _med_blocks(
        _fixture_mesh(dolfinx.mesh.CellType.triangle), _left_right, _facet_markers
    )
    times = (0.0, 0.5, 1.0)
    later_steps = []
    for number, t in enumerate(times[1:], start=1):
        x_t = x.copy()
        x_t[:, 0] *= 1 + t
        later_steps.append((number, t, x_t))
    _on_root(comm, _write_med, filename, x, blocks, _families, later_steps=later_steps)

    for t in times:
        mesh = io4dolfinx.read_mesh(filename, comm, backend="med", time=t)
        args = {"time": t}
        ct = io4dolfinx.read_meshtags(filename, mesh, "cell", backend="med", backend_args=args)
        ft = io4dolfinx.read_meshtags(filename, mesh, "facet", backend="med", backend_args=args)
        left = io4dolfinx.read_meshtags(filename, mesh, "left", backend="med", backend_args=args)
        assert np.isclose(_integrate(mesh, "dx"), 1 + t)
        assert np.isclose(_integrate(mesh, "dx", 2, ct), (1 + t) / 2)
        assert np.isclose(_integrate(mesh, "dx", 1, left), (1 + t) / 2)
        assert np.isclose(_integrate(mesh, "ds", 5, ft), 1.0)

    mesh = io4dolfinx.read_mesh(filename, comm, backend="med", time=None)
    assert np.isclose(_integrate(mesh, "dx"), 1.0)
    with pytest.raises(KeyError, match=r"Available times: \[0.0, 0.5, 1.0\]"):
        io4dolfinx.read_mesh(filename, comm, backend="med", time=0.3)


def _medcoupling_mesh(mc, xscale=1.0, it=-1, order=-1, t=0.0, stamp_levels=False):
    """Triangulated unit square (3x3 nodes, 8 TRI3) with cell and edge groups.

    x is scaled by ``xscale``. ``it``/``order``/``t`` set the step. MEDCoupling writes
    connectivity and families into a later step only if the level meshes carry its time
    (``stamp_levels``); otherwise they are links to the first step."""
    xs = mc.DataArrayDouble([0.0, 0.5 * xscale, 1.0 * xscale])
    ys = mc.DataArrayDouble([0.0, 0.5, 1.0])
    cm = mc.MEDCouplingCMesh("mesh")
    cm.setCoords(xs, ys)
    m0 = cm.buildUnstructured()
    m0.simplexize(0)
    m0.setName("mesh")
    m1 = m0.computeSkin()
    m1.setName("mesh")
    if stamp_levels:
        m0.setTime(t, it, order)
        m1.setTime(t, it, order)

    mm = mc.MEDFileUMesh()
    mm.setName("mesh")
    mm.setMeshAtLevel(0, m0)
    mm.setMeshAtLevel(-1, m1)
    mm.setTime(it, order, t)

    # Classify on the unscaled geometry
    xc = m0.computeCellCenterOfMass()[:, 0]
    xc /= xscale
    left = xc.findIdsLowerThan(0.5)
    left.setName("left")
    right = xc.findIdsGreaterOrEqualTo(0.5)
    right.setName("right")
    mm.setGroupsAtLevel(0, [left, right])

    xe = m1.computeCellCenterOfMass()[:, 0]
    xe /= xscale
    inlet = xe.findIdsInRange(-1e-12, 1e-12)
    inlet.setName("inlet")
    outlet = xe.findIdsInRange(1.0 - 1e-12, 1.0 + 1e-12)
    outlet.setName("outlet")
    mm.setGroupsAtLevel(-1, [inlet, outlet])
    return mm


# MED requires the first step of a mesh to be (iteration, order) = (-1, -1)
_medcoupling_steps = [(-1, -1, 0.0), (1, -1, 0.5), (2, -1, 1.0)]


def _write_medcoupling(mc, filename, variant):
    if variant == "static":
        _medcoupling_mesh(mc).write(str(filename), 2)
    elif variant == "coords_only":
        mm = _medcoupling_mesh(mc)
        mm.write(str(filename), 2)
        for it, order, t in _medcoupling_steps[1:]:
            coords = mm.getCoords().deepCopy()
            coords[:, 0] *= 1.0 + t
            step = mc.MEDFileUMesh()
            step.setName("mesh")
            step.setCoords(coords)
            step.setTime(it, order, t)
            step.write(str(filename), 0)
    else:
        for k, (it, order, t) in enumerate(_medcoupling_steps):
            mm = _medcoupling_mesh(mc, 1.0 + t, it, order, t, stamp_levels=variant == "stamped")
            # Mode 2 creates the file, mode 0 appends the step
            mm.write(str(filename), 2 if k == 0 else 0)


@pytest.mark.parametrize("variant", ["static", "time_dependent", "stamped", "coords_only"])
def test_medcoupling_reference(tmp_dir, variant):
    """Files written by MEDCoupling, Salome's MED library."""
    mc = pytest.importorskip("medcoupling")
    comm = MPI.COMM_WORLD
    filename = tmp_dir / f"medcoupling_{variant}.med"
    _on_root(comm, _write_medcoupling, mc, filename, variant)

    # MEDCoupling numbers families -2 (left), -3 (right), -4 (boundary edges in no
    # group), -5 (inlet) and -6 (outlet)
    assert read_groups(filename, comm) == {
        "left": [2],
        "right": [3],
        "inlet": [5],
        "outlet": [6],
    }
    times = [0.0] if variant == "static" else [t for _, _, t in _medcoupling_steps]
    for t in times:
        mesh = io4dolfinx.read_mesh(filename, comm, backend="med", time=t)
        args = {"time": t}
        ct = io4dolfinx.read_meshtags(filename, mesh, "cell", backend="med", backend_args=args)
        ft = io4dolfinx.read_meshtags(filename, mesh, "facet", backend="med", backend_args=args)
        left = io4dolfinx.read_meshtags(filename, mesh, "left", backend="med", backend_args=args)
        inlet = io4dolfinx.read_meshtags(filename, mesh, "inlet", backend="med", backend_args=args)
        assert np.isclose(_integrate(mesh, "dx"), 1 + t)
        assert np.isclose(_integrate(mesh, "dx", 2, ct), (1 + t) / 2)
        assert np.isclose(_integrate(mesh, "dx", 1, left), (1 + t) / 2)
        assert np.isclose(_integrate(mesh, "ds", 6, ft), 1.0)
        assert np.isclose(_integrate(mesh, "ds", 1, inlet), 1.0)


def test_salome_file(tmp_dir):
    """Salome exported mesh from the FESTIM workshop: two 100x100 squares."""
    comm = MPI.COMM_WORLD
    filename = tmp_dir / "Mesh_1.med"
    status = download_file_if_not_exists(_festim_url, filename)
    if status == DownloadStatus.no_connection:
        pytest.skip("No internet connection")
    assert status == DownloadStatus.success

    mesh = io4dolfinx.read_mesh(filename, comm, backend="med")
    assert mesh.topology.index_map(2).size_global == 603
    assert mesh.geometry.index_map().size_global == 347
    assert np.isclose(_integrate(mesh, "dx"), 20000.0)

    groups = read_groups(filename, comm)
    assert groups == {
        "left_volume": [6],
        "right_volume": [7],
        "left_boundary": [8],
        "right_boundary": [9],
    }

    ct = io4dolfinx.read_meshtags(filename, mesh, meshtag_name="cell", backend="med")
    ft = io4dolfinx.read_meshtags(filename, mesh, meshtag_name="facet", backend="med")
    assert np.isclose(_integrate(mesh, "dx", 6, ct), 10000.0)
    assert np.isclose(_integrate(mesh, "dx", 7, ct), 10000.0)
    assert np.isclose(_integrate(mesh, "ds", 8, ft), 100.0)
    assert np.isclose(_integrate(mesh, "ds", 9, ft), 100.0)
    num_owned = mesh.topology.index_map(1).size_local
    assert comm.allreduce(np.count_nonzero(ft.indices < num_owned), op=MPI.SUM) == 30

    for name, measure, expected in [
        ("left_volume", "dx", 10000.0),
        ("right_volume", "dx", 10000.0),
        ("left_boundary", "ds", 100.0),
        ("right_boundary", "ds", 100.0),
    ]:
        tags = io4dolfinx.read_meshtags(filename, mesh, meshtag_name=name, backend="med")
        assert np.isclose(_integrate(mesh, measure, 1, tags), expected)
