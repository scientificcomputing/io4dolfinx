from pathlib import Path

from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np
import pytest
import ufl

import io4dolfinx
from io4dolfinx.backends.opencarp import write_opencarp

CellType = dolfinx.mesh.CellType

# openCARP element type, and number of cells, nodes, marked facets and marked nodes
# per marker of the meshes of create_problem
meshes = {
    CellType.tetrahedron: ("Tt", 6 * 27, 64, 2 * 9, 16),
    CellType.hexahedron: ("Hx", 27, 64, 9, 16),
    CellType.triangle: ("Tr", 2 * 9, 16, 3, 4),
    CellType.quadrilateral: ("Qd", 9, 16, 3, 4),
}
cell_types = pytest.mark.parametrize("cell_type", list(meshes), ids=lambda c: c.name)


def markers(tdim: int):
    return {"left": (1, tdim - 1), "right": (2, tdim - 1)}


def fiber_expr(x):
    return np.vstack([2 + x[1], x[0], np.zeros_like(x[0])])


def sheet_expr(x):
    return np.vstack([np.ones_like(x[0]), 3 + x[2], x[0]])


def expected_direction(expr, points, gdim):
    """Normalised direction at points, with the components a mesh of dimension gdim
    stores and a zero z-component in 2D, as written to ``.lon``."""
    values = np.zeros((3, len(points)))
    values[:gdim] = expr(points.T)[:gdim]
    return (values / np.linalg.norm(values, axis=0)).T


def create_problem(comm: MPI.Intracomm, cell_type: dolfinx.mesh.CellType = CellType.tetrahedron):
    """Unit cube or square with region 1 (x < 0.4) and 2, facet markers on x=0 and x=1,
    and linear fiber and sheet fields (exact in P-1)."""
    if dolfinx.mesh.cell_dim(cell_type) == 3:
        mesh = dolfinx.mesh.create_unit_cube(comm, 3, 3, 3, cell_type=cell_type)
    else:
        mesh = dolfinx.mesh.create_unit_square(comm, 3, 3, cell_type=cell_type)
    tdim = mesh.topology.dim
    gdim = mesh.geometry.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)

    cells = np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)
    midpoints = dolfinx.mesh.compute_midpoints(mesh, tdim, cells)
    regions = np.where(midpoints[:, 0] < 0.4, 1, 2).astype(np.int32)
    cell_tags = dolfinx.mesh.meshtags(mesh, tdim, cells, regions)

    left = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[0], 0))
    right = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[0], 1))
    facets = np.concatenate([left, right])
    values = np.concatenate([np.full_like(left, 1), np.full_like(right, 2)]).astype(np.int32)
    order = np.argsort(facets)
    facet_tags = dolfinx.mesh.meshtags(mesh, tdim - 1, facets[order], values[order])

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (gdim,)))
    fiber = dolfinx.fem.Function(V, name="f0")
    fiber.interpolate(lambda x: fiber_expr(x)[:gdim])
    sheet = dolfinx.fem.Function(V, name="s0")
    sheet.interpolate(lambda x: sheet_expr(x)[:gdim])
    return mesh, cell_tags, facet_tags, fiber, sheet


def write_problem(
    filename: Path,
    comm: MPI.Intracomm,
    cell_type: dolfinx.mesh.CellType = CellType.tetrahedron,
    **kwargs,
):
    mesh, cell_tags, facet_tags, fiber, sheet = create_problem(comm, cell_type)
    write_opencarp(
        filename,
        mesh,
        scale=1000.0,
        cell_tags=cell_tags,
        facet_tags=facet_tags,
        fiber=fiber,
        sheet=sheet,
        markers=markers(mesh.topology.dim),
        **kwargs,
    )


def read_lines(path: Path) -> list[list[str]]:
    return [line.split() for line in path.read_text().splitlines()]


def orientation(elem_type: str, p: np.ndarray) -> np.ndarray:
    """Signed measure of each element with nodes p in openCARP ordering, positive for
    the orientation of openCARP's mesher, and the factor to its volume or area."""
    e1 = p[:, 1] - p[:, 0]
    e2 = p[:, 2 if elem_type in ("Tr", "Tt") else 3] - p[:, 0]
    if elem_type in ("Tr", "Qd"):
        return np.cross(e1, e2)[:, 2], 2 if elem_type == "Tr" else 1
    e3 = p[:, 3 if elem_type == "Tt" else 4] - p[:, 0]
    return np.einsum("ij,ij->i", np.cross(e1, e2), e3), 6 if elem_type == "Tt" else 1


@cell_types
def test_write_content(shared_tmp_path, cell_type):
    filename = shared_tmp_path / "mesh"
    write_problem(filename, MPI.COMM_WORLD, cell_type)
    if MPI.COMM_WORLD.rank != 0:
        return
    elem_type, num_cells, num_nodes, num_facets, num_marked_nodes = meshes[cell_type]
    tdim = dolfinx.mesh.cell_dim(cell_type)

    pts = read_lines(filename.with_suffix(".pts"))
    assert int(pts[0][0]) == len(pts) - 1 == num_nodes
    x = np.array(pts[1:], dtype=np.float64)
    assert x.shape == (num_nodes, 3)
    # Scaled from the unit cube or square to µm, z = 0 in 2D
    assert np.allclose(x.min(axis=0)[:tdim], 0) and np.allclose(x.max(axis=0)[:tdim], 1000)
    assert np.all(x[:, tdim:] == 0)

    elem = read_lines(filename.with_suffix(".elem"))
    assert int(elem[0][0]) == len(elem) - 1 == num_cells
    assert {row[0] for row in elem[1:]} == {elem_type}
    cells = np.array([row[1:-1] for row in elem[1:]], dtype=np.int64)
    tags = np.array([row[-1] for row in elem[1:]], dtype=np.int32)
    assert cells.min() == 0 and cells.max() == num_nodes - 1

    # All elements oriented as by openCARP's mesher, filling the cube or square
    p = x[cells]
    measure, factor = orientation(elem_type, p)
    assert np.all(measure > 0)
    assert np.isclose(measure.sum() / factor, 1000.0**tdim)
    if elem_type == "Qd":
        np.testing.assert_allclose(p[:, 2] - p[:, 0], (p[:, 1] - p[:, 0]) + (p[:, 3] - p[:, 0]))
    if elem_type == "Hx":
        e1, e2, e3 = p[:, 1] - p[:, 0], p[:, 3] - p[:, 0], p[:, 4] - p[:, 0]
        np.testing.assert_allclose(p[:, 2] - p[:, 0], e1 + e2)
        np.testing.assert_allclose(p[:, 5] - p[:, 0], e2 + e3)
        np.testing.assert_allclose(p[:, 6] - p[:, 0], e1 + e2 + e3)
        np.testing.assert_allclose(p[:, 7] - p[:, 0], e1 + e3)

    # Region tags follow the midpoints
    midpoints = p.mean(axis=1)
    np.testing.assert_array_equal(tags, np.where(midpoints[:, 0] < 400, 1, 2))

    # One unit fiber and sheet per element, in element order
    lon = read_lines(filename.with_suffix(".lon"))
    assert lon[0] == ["2"]
    directions = np.array(lon[1:], dtype=np.float64)
    assert directions.shape == (num_cells, 6)
    for column, expr in ((0, fiber_expr), (3, sheet_expr)):
        expected = expected_direction(expr, midpoints / 1000, tdim)
        np.testing.assert_allclose(directions[:, column : column + 3], expected, atol=1e-9)

    for name, plane in (("left", 0.0), ("right", 1000.0)):
        vtx = read_lines(filename.parent / f"mesh_{name}.vtx")
        assert vtx[1] == ["extra"]
        nodes = np.array(vtx[2:], dtype=np.int64).reshape(-1)
        assert int(vtx[0][0]) == len(nodes) == num_marked_nodes
        assert np.allclose(x[nodes, 0], plane)

        surf_file = filename.parent / f"mesh_{name}.surf"
        if tdim == 2:
            assert not surf_file.exists()
            continue
        surf = read_lines(surf_file)
        assert int(surf[0][0]) == len(surf) - 1 == num_facets
        face_type = {"Tt": "Tr", "Hx": "Qd"}[elem_type]
        assert {row[0] for row in surf[1:]} == {face_type}
        faces = np.array([row[1:] for row in surf[1:]], dtype=np.int64)
        np.testing.assert_array_equal(np.unique(faces), np.sort(nodes))
        # Normals point out of the cube, and quadrilateral nodes are in cyclic order
        q = x[faces]
        if face_type == "Tr":
            normals = np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]) / 2
            area = (1000 / 3) ** 2 / 2
        else:
            normals = np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 1]) / 2
            area = (1000 / 3) ** 2
        np.testing.assert_allclose(normals[:, 0], (-1 if plane == 0 else 1) * area)


@cell_types
def test_files_independent_of_partitioning(shared_tmp_path, cell_type):
    parallel = shared_tmp_path / "parallel" / "mesh"
    serial = shared_tmp_path / "serial" / "mesh"
    if MPI.COMM_WORLD.rank == 0:
        parallel.parent.mkdir()
        serial.parent.mkdir()
    MPI.COMM_WORLD.Barrier()
    write_problem(parallel, MPI.COMM_WORLD, cell_type)
    if MPI.COMM_WORLD.rank == 0:
        write_problem(serial, MPI.COMM_SELF, cell_type)
        files = sorted(p.name for p in serial.parent.iterdir())
        assert files == sorted(p.name for p in parallel.parent.iterdir())
        # .pts, .elem, .lon and a .vtx (and in 3D a .surf) per marker
        assert len(files) == (7 if dolfinx.mesh.cell_dim(cell_type) == 3 else 5)
        for name in files:
            assert (serial.parent / name).read_bytes() == (parallel.parent / name).read_bytes()


@cell_types
def test_read_write(shared_tmp_path, cell_type):
    write_problem(shared_tmp_path / "mesh", MPI.COMM_WORLD, cell_type)
    filename = shared_tmp_path / "mesh.elem"
    comm = MPI.COMM_WORLD
    _, num_cells, num_nodes, num_facets, num_marked_nodes = meshes[cell_type]
    tdim = dolfinx.mesh.cell_dim(cell_type)
    backend_args = {"scale": 1000.0, "markers": markers(tdim)}

    mesh = io4dolfinx.read_mesh(filename, comm, backend="opencarp", backend_args=backend_args)
    assert mesh.topology.cell_type == cell_type
    assert mesh.geometry.dim == tdim
    assert mesh.topology.index_map(tdim).size_global == num_cells
    assert mesh.geometry.index_map().size_global == num_nodes
    volume = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ufl.dx(domain=mesh)))
    assert np.isclose(comm.allreduce(volume, op=MPI.SUM), 1.0)

    cell_tags = io4dolfinx.read_meshtags(filename, mesh, "cell", backend="opencarp")
    midpoints = dolfinx.mesh.compute_midpoints(mesh, tdim, cell_tags.indices)
    np.testing.assert_array_equal(cell_tags.values, np.where(midpoints[:, 0] < 0.4, 1, 2))
    num_owned = mesh.topology.index_map(tdim).size_local
    assert comm.allreduce(np.count_nonzero(cell_tags.indices < num_owned)) == num_cells

    # Facets are read from .surf files in 3D, marked nodes from .vtx files in 2D
    right = io4dolfinx.read_meshtags(
        filename, mesh, "right", backend="opencarp", backend_args=backend_args
    )
    dim = tdim - 1 if tdim == 3 else 0
    assert right.dim == dim
    assert np.all(right.values == 2)
    assert np.allclose(dolfinx.mesh.compute_midpoints(mesh, dim, right.indices)[:, 0], 1)
    num_owned = mesh.topology.index_map(dim).size_local
    num_marked = num_facets if tdim == 3 else num_marked_nodes
    assert comm.allreduce(np.count_nonzero(right.indices < num_owned)) == num_marked
    if tdim == 3:
        facet_tags = io4dolfinx.read_meshtags(
            filename, mesh, "facet", backend="opencarp", backend_args=backend_args
        )
        midpoints = dolfinx.mesh.compute_midpoints(mesh, tdim - 1, facet_tags.indices)
        np.testing.assert_allclose(midpoints[:, 0], np.where(facet_tags.values == 1, 0.0, 1.0))
        num_owned = mesh.topology.index_map(tdim - 1).size_local
        assert comm.allreduce(np.count_nonzero(facet_tags.indices < num_owned)) == 2 * num_facets

    assert io4dolfinx.read_function_names(filename, comm, backend="opencarp") == ["fiber", "sheet"]
    for name, expr in (("fiber", fiber_expr), ("sheet", sheet_expr)):
        u = io4dolfinx.read_cell_data(filename, name, mesh, backend="opencarp")
        assert u.ufl_shape == (tdim,)
        cells = np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)
        midpoints = dolfinx.mesh.compute_midpoints(mesh, tdim, cells)
        values = u.x.array.reshape(-1, tdim)[u.function_space.dofmap.list[cells, 0]]
        expected = expected_direction(expr, midpoints, tdim)[:, :tdim]
        np.testing.assert_allclose(values, expected, atol=1e-9)


def test_read_vertex_markers(shared_tmp_path):
    """Without ``.surf`` files a marker is read from its ``.vtx`` file, as vertex tags."""
    write_problem(shared_tmp_path / "mesh", MPI.COMM_WORLD, backend_args={"surf": False})
    filename = shared_tmp_path / "mesh.elem"
    assert not (shared_tmp_path / "mesh_left.surf").exists()

    mesh = io4dolfinx.read_mesh(
        filename, MPI.COMM_WORLD, backend="opencarp", backend_args={"scale": 1000.0}
    )
    left = io4dolfinx.read_meshtags(
        filename, mesh, "left", backend="opencarp", backend_args={"markers": markers(3)}
    )
    assert left.dim == 0
    assert np.all(left.values == 1)
    assert np.allclose(dolfinx.mesh.compute_midpoints(mesh, 0, left.indices)[:, 0], 0)
    num_owned = mesh.topology.index_map(0).size_local
    assert MPI.COMM_WORLD.allreduce(np.count_nonzero(left.indices < num_owned)) == 16


def test_fiber_only_and_unnamed_markers(shared_tmp_path):
    """Markers without names are written and read back by value."""
    filename = shared_tmp_path / "mesh"
    mesh, _, facet_tags, fiber, _ = create_problem(MPI.COMM_WORLD)
    write_opencarp(filename, mesh, scale=1.0, facet_tags=facet_tags, fiber=fiber)
    if MPI.COMM_WORLD.rank == 0:
        assert read_lines(filename.with_suffix(".lon"))[0] == ["1"]
        tags = {row[5] for row in read_lines(filename.with_suffix(".elem"))[1:]}
        assert tags == {"0"}

    filename = filename.with_suffix(".elem")
    assert io4dolfinx.read_function_names(filename, MPI.COMM_WORLD, backend="opencarp") == ["fiber"]
    in_mesh = io4dolfinx.read_mesh(
        filename, MPI.COMM_WORLD, backend="opencarp", backend_args={"scale": 1.0}
    )
    right = io4dolfinx.read_meshtags(filename, in_mesh, "2", backend="opencarp")
    assert np.all(right.values == 2)
    with pytest.raises(KeyError):
        io4dolfinx.read_cell_data(filename, "sheet", in_mesh, backend="opencarp")
    with pytest.raises(KeyError):
        io4dolfinx.read_meshtags(filename, in_mesh, "missing", backend="opencarp")


def test_errors(shared_tmp_path):
    comm = MPI.COMM_WORLD
    filename = shared_tmp_path / "mesh"
    mesh, cell_tags, _, fiber, sheet = create_problem(comm)

    with pytest.raises(ValueError, match="scale"):
        io4dolfinx.write_mesh_input_order(filename, mesh, backend="opencarp")
    with pytest.raises(ValueError, match="input ordering"):
        io4dolfinx.write_mesh(filename, mesh, backend="opencarp", backend_args={"scale": 1.0})
    with pytest.raises(ValueError, match="without a fiber"):
        write_opencarp(filename, mesh, scale=1.0, sheet=sheet)

    Q = dolfinx.fem.functionspace(
        mesh, basix.ufl.quadrature_element(mesh.basix_cell(), value_shape=(3,), degree=2)
    )
    with pytest.raises(ValueError, match="quadrature"):
        write_opencarp(filename, mesh, scale=1.0, fiber=dolfinx.fem.Function(Q))

    interval_mesh = dolfinx.mesh.create_unit_interval(comm, 4)
    with pytest.raises(NotImplementedError, match="interval"):
        write_opencarp(filename, interval_mesh, scale=1.0)

    # Errors found when writing on rank 0 are raised on all ranks
    write_opencarp(filename, mesh, scale=1.0)
    with pytest.raises(ValueError, match="input node ordering"):
        io4dolfinx.write_meshtags(filename, mesh, cell_tags, backend="opencarp")
    zero = dolfinx.fem.Function(fiber.function_space)
    with pytest.raises(ValueError, match="zero direction"):
        io4dolfinx.write_cell_data(
            filename, zero, 0.0, io4dolfinx.FileMode.write, None, "opencarp", on_input_mesh=True
        )
    with pytest.raises(ValueError, match="input cell ordering"):
        io4dolfinx.write_cell_data(
            filename, fiber, 0.0, io4dolfinx.FileMode.write, None, "opencarp"
        )


def test_2d_surface_not_supported(shared_tmp_path):
    """Triangles that do not lie in a plane of constant z (surfaces in 3D) are rejected."""
    filename = shared_tmp_path / "surface"
    if MPI.COMM_WORLD.rank == 0:
        filename.with_suffix(".pts").write_text("3\n0 0 0\n1 0 0\n0 1 1\n")
        filename.with_suffix(".elem").write_text("1\nTr 0 1 2 0\n")
    MPI.COMM_WORLD.Barrier()
    with pytest.raises(NotImplementedError, match="constant z"):
        io4dolfinx.read_mesh(
            filename.with_suffix(".elem"),
            MPI.COMM_WORLD,
            backend="opencarp",
            backend_args={"scale": 1.0},
        )
