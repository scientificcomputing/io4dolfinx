from mpi4py import MPI

import dolfinx
import fenicsx_compat
import numpy as np
import pytest
import ufl

import io4dolfinx

gmsh = pytest.importorskip("gmsh")
if fenicsx_compat.before("0.11.0.dev0"):
    pytest.skip("The nastran backend requires DOLFINx 0.11 or later.", allow_module_level=True)

# Two unit cubes [0, 1]^3 and [1, 2]x[0, 1]^2, sharing the face x = 1
GRIDS = {
    1: (0, 0, 0),
    2: (1, 0, 0),
    3: (1, 1, 0),
    4: (0, 1, 0),
    5: (0, 0, 1),
    6: (1, 0, 1),
    7: (1, 1, 1),
    8: (0, 1, 1),
    9: (2, 0, 0),
    10: (2, 1, 0),
    11: (2, 1, 1),
    12: (2, 0, 1),
    13: (1.5, 0.5, 0.5),  # Centre of the second cube
}


def field(value) -> str:
    """A small field (8 characters) of a fixed-format Nastran card."""
    return f"{value:<8}" if isinstance(value, str) else f"{value:>8}"


def card(name: str, *values) -> str:
    """A fixed-format Nastran card, continued after eight fields."""
    first, rest = values[:8], values[8:]
    line = field(name) + "".join(field(v) for v in first)
    if rest:
        line += field("+") + "\n" + field("+") + "".join(field(v) for v in rest)
    return line + "\n"


GRID_IDS = {n: n for n in GRIDS}


@pytest.fixture(params=["contiguous", "gapped"])
def grid_ids(request) -> dict[int, int]:
    """GRID ID of each node: ``1, ..., 13``, or decreasing with gaps, as Nastran allows."""
    return {n: n if request.param == "contiguous" else 1000 - 70 * n for n in GRIDS}


def write_bdf(path, cards: list[str], grid_ids: dict[int, int]):
    grids = [card("GRID", grid_ids[n], "", *(f"{c:.1f}" for c in x)) for n, x in GRIDS.items()]
    if MPI.COMM_WORLD.rank == 0:
        path.write_text("CEND\nBEGIN BULK\n" + "".join(grids + cards) + "ENDDATA\n")
    MPI.COMM_WORLD.barrier()


def hexahedral_cards(grid_ids: dict[int, int]) -> list[str]:
    """Two hexahedra with PIDs 1 and 2, and a shell element with PID 5 on x = 0."""
    g = grid_ids
    return [
        card("PSOLID", 1, 1),
        card("PSOLID", 2, 1),
        card("PSHELL", 5, 1, 0.1),
        card("CHEXA", 1, 1, *(g[n] for n in (1, 2, 3, 4, 5, 6, 7, 8))),
        card("CHEXA", 2, 2, *(g[n] for n in (2, 9, 10, 3, 6, 12, 11, 7))),
        card("CQUAD4", 3, 5, *(g[n] for n in (1, 4, 8, 5))),
    ]


def test_hexahedral_mesh_and_tags(shared_tmp_path, grid_ids):
    """A mesh of one cell type, with PIDs as cell tags (PSOLID) and facet tags (PSHELL)."""
    path = shared_tmp_path / "hexahedra.bdf"
    write_bdf(path, hexahedral_cards(grid_ids), grid_ids)
    mesh = io4dolfinx.read_mesh(path, MPI.COMM_WORLD, backend="nastran")
    assert mesh.topology.cell_type == dolfinx.mesh.CellType.hexahedron
    volume = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ufl.dx(domain=mesh)))
    assert np.isclose(mesh.comm.allreduce(volume, op=MPI.SUM), 2)

    cell_tags = io4dolfinx.read_meshtags(path, mesh, "PSOLID", backend="nastran")
    midpoints = dolfinx.mesh.compute_midpoints(mesh, 3, cell_tags.indices)
    np.testing.assert_array_equal(cell_tags.values, np.where(midpoints[:, 0] < 1, 1, 2))

    facet_tags = io4dolfinx.read_meshtags(path, mesh, "PSHELL", backend="nastran")
    num_tagged = mesh.comm.allreduce(len(facet_tags.find(5)), op=MPI.SUM)
    assert num_tagged == 1
    midpoints = dolfinx.mesh.compute_midpoints(mesh, 2, facet_tags.find(5))
    assert np.allclose(midpoints[:, 0], 0)


def expected_grid_ids(x: np.ndarray, grid_ids: dict[int, int]) -> np.ndarray:
    """The GRID ID of the node at each point of ``x``."""
    points = np.array(list(GRIDS.values()))
    nodes = [list(GRIDS)[np.argmin(np.linalg.norm(points - p, axis=1))] for p in x]
    return np.array([grid_ids[n] for n in nodes])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("cards", ["hexahedral", "mixed"])
def test_mesh_dtype(shared_tmp_path, dtype, cards):
    """Single and double precision meshes, with one or several cell types."""
    path = shared_tmp_path / f"{cards}_{np.dtype(dtype).name}.bdf"
    make_cards = hexahedral_cards if cards == "hexahedral" else mixed_cards
    write_bdf(path, make_cards(GRID_IDS), GRID_IDS)
    mesh = io4dolfinx.read_mesh(
        path, MPI.COMM_WORLD, backend="nastran", backend_args={"dtype": dtype}
    )
    assert mesh.geometry.x.dtype == dtype
    num_nodes = mesh.geometry.index_map().size_local
    found = np.vstack(mesh.comm.allgather(mesh.geometry.x[:num_nodes]))
    expected = np.array([GRIDS[n] for n in GRIDS if n != 13 or cards == "mixed"])
    np.testing.assert_allclose(np.unique(found, axis=0), np.unique(expected, axis=0))


def test_grid_ids(shared_tmp_path, grid_ids):
    """The GRID IDs are read as point data on the geometry nodes and as vertex tags."""
    path = shared_tmp_path / "grid_ids.bdf"
    write_bdf(path, hexahedral_cards(grid_ids), grid_ids)
    mesh = io4dolfinx.read_mesh(path, MPI.COMM_WORLD, backend="nastran")
    assert io4dolfinx.read_function_names(path, MPI.COMM_WORLD, backend="nastran") == ["GRID"]

    # The dofs of the geometry space are the geometry nodes
    u = io4dolfinx.read_point_data(path, "GRID", mesh, backend="nastran")
    x = mesh.geometry.x
    np.testing.assert_array_equal(u.x.array[: len(x)], expected_grid_ids(x, grid_ids))

    tags = io4dolfinx.read_meshtags(path, mesh, "GRID", backend="nastran")
    assert tags.dim == 0
    vertices = dolfinx.mesh.compute_midpoints(mesh, 0, tags.indices)
    np.testing.assert_array_equal(tags.values, expected_grid_ids(vertices, grid_ids))
    num_vertices = mesh.topology.index_map(0)
    owned = tags.indices < num_vertices.size_local
    assert mesh.comm.allreduce(np.count_nonzero(owned), op=MPI.SUM) == 12

    # A point measure on a GRID ID evaluates at that node
    dP = ufl.Measure("dP", domain=mesh, subdomain_data=tags)
    x = ufl.SpatialCoordinate(mesh)
    f = 1 + x[0] + 2 * x[1] + 3 * x[2]
    for n in (1, 7, 12):
        value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(f * dP(grid_ids[n])))
        px, py, pz = GRIDS[n]
        assert np.isclose(mesh.comm.allreduce(value, op=MPI.SUM), 1 + px + 2 * py + 3 * pz)


def mixed_cards(grid_ids: dict[int, int]) -> list[str]:
    """One hexahedron, and a second cube split into pyramids from its centre: one with the
    shared face as base, and five split into two tetrahedra each."""
    g = grid_ids
    cards = [
        card("CHEXA", 1, 1, *(g[n] for n in (1, 2, 3, 4, 5, 6, 7, 8))),
        card("CPYRAM", 2, 2, *(g[n] for n in (2, 3, 7, 6, 13))),
    ]
    faces = [(9, 10, 11, 12), (2, 9, 12, 6), (3, 10, 11, 7), (2, 9, 10, 3), (6, 12, 11, 7)]
    eid = 3
    for a, b, c, d in faces:
        for tet in ((a, b, c, 13), (a, c, d, 13)):
            cards.append(card("CTETRA", eid, 3, *(g[n] for n in tet)))
            eid += 1
    return cards


def test_mixed_mesh(shared_tmp_path, grid_ids):
    """Hexahedra, pyramids and tetrahedra are read as a mixed mesh."""
    path = shared_tmp_path / "mixed.bdf"
    write_bdf(path, mixed_cards(grid_ids), grid_ids)
    mesh = io4dolfinx.read_mesh(path, MPI.COMM_WORLD, backend="nastran")

    CellType = dolfinx.mesh.CellType
    assert mesh.topology.entity_types[3] == [
        CellType.hexahedron,
        CellType.pyramid,
        CellType.tetrahedron,
    ]
    num_cells = [mesh.comm.allreduce(im.size_local) for im in mesh.topology.index_maps(3)]
    assert num_cells == [1, 1, 10]

    # Facets shared by two cells are only found once if the node ordering of every cell
    # type is right: 6 quadrilaterals, and 44 triangle faces of which 17 are shared
    mesh.topology.create_entities(2)
    assert mesh.topology.entity_types[2] == [CellType.quadrilateral, CellType.triangle]
    assert [im.size_global for im in mesh.topology.index_maps(2)] == [6, 27]

    # Input node i is the node with the i-th smallest GRID ID
    x = mesh.geometry.x
    by_id = sorted(GRIDS, key=lambda n: grid_ids[n])
    expected = np.array([GRIDS[n] for n in by_id], dtype=x.dtype)
    np.testing.assert_allclose(x, expected[mesh.geometry.input_global_indices])


def test_mixed_meshtags_not_supported(shared_tmp_path):
    """Mesh tags of several entity types, here PSOLID on a mixed mesh, raise an error."""
    path = shared_tmp_path / "mixed_tags.bdf"
    write_bdf(path, mixed_cards(GRID_IDS), GRID_IDS)
    with pytest.raises(NotImplementedError, match="several types"):
        io4dolfinx.backends.get_backend("nastran").read_meshtags_data(
            path, MPI.COMM_WORLD, "PSOLID"
        )


@pytest.mark.parametrize("read", ["read_point_data", "read_meshtags_data"])
def test_grid_ids_not_supported_on_mixed_mesh(shared_tmp_path, read):
    """GRID IDs are only read on meshes with one cell type."""
    path = shared_tmp_path / "mixed_grid.bdf"
    write_bdf(path, mixed_cards(GRID_IDS), GRID_IDS)
    backend = io4dolfinx.backends.get_backend("nastran")
    with pytest.raises(NotImplementedError, match="several cell types"):
        if read == "read_point_data":
            backend.read_point_data(path, "GRID", MPI.COMM_WORLD, None, None)
        else:
            backend.read_meshtags_data(path, MPI.COMM_WORLD, "GRID")


def test_unknown_property_card(shared_tmp_path):
    path = shared_tmp_path / "unknown.bdf"
    write_bdf(path, mixed_cards(GRID_IDS), GRID_IDS)
    with pytest.raises(ValueError, match="Unsupported meshtag name"):
        io4dolfinx.backends.get_backend("nastran").read_meshtags_data(path, MPI.COMM_WORLD, "PBAR")


def test_mixed_mesh_requires_dolfinx_0_11(monkeypatch):
    """Mixed meshes are only created with DOLFINx 0.11 or later."""
    monkeypatch.setattr(fenicsx_compat, "before", lambda version: True)
    cells = [np.zeros((0, 8), dtype=np.int64), np.zeros((0, 4), dtype=np.int64)]
    with pytest.raises(NotImplementedError, match="requires DOLFINx 0.11"):
        io4dolfinx.utils.create_mixed_mesh(
            MPI.COMM_WORLD, cells, ["hexahedron", "tetrahedron"], np.zeros((0, 3)), 1, 1
        )
