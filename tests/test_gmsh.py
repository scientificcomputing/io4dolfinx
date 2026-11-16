from mpi4py import MPI

import basix
import dolfinx
import fenicsx_compat
import numpy as np
import pytest
import ufl
from dolfinx.fem import Function, functionspace
from dolfinx.mesh import CellType, create_unit_cube, create_unit_square, locate_entities, meshtags

import io4dolfinx

gmsh = pytest.importorskip("gmsh")
if fenicsx_compat.before("0.11.0.dev0"):
    pytest.skip("The gmsh backend requires DOLFINx 0.11 or later.", allow_module_level=True)


@pytest.fixture(autouse=True)
def gmsh_default_backend():
    """Use the gmsh backend where no backend is given, and restore the default after."""
    previous = io4dolfinx.get_default_backend()
    io4dolfinx.set_default_backend("gmsh")
    yield
    io4dolfinx.set_default_backend(previous)


def f(x, t, dtype):
    values = np.array([x[0] + np.sin(x[1]) + np.cos(x[0] * t), x[1] - t])
    return values * (1 + 2j) if np.issubdtype(dtype, np.complexfloating) else values


def create_curved_mesh(comm, cell_type, degree, dtype):
    """Create a unit square/cube mesh with a Lagrange geometry of the given degree."""
    if cell_type in (CellType.triangle, CellType.quadrilateral):
        mesh = create_unit_square(comm, 3, 2, cell_type=cell_type, dtype=dtype)
    else:
        mesh = create_unit_cube(comm, 2, 3, 2, cell_type=cell_type, dtype=dtype)
    if degree == 1:
        return mesh
    # Gmsh geometries are equispaced
    cmap = dolfinx.fem.coordinate_element(
        cell_type, degree, variant=basix.LagrangeVariant.equispaced, dtype=dtype
    )
    return dolfinx.fem.interpolate_geometry(mesh, cmap)


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize(
    "cell_type",
    [CellType.triangle, CellType.quadrilateral, CellType.tetrahedron, CellType.hexahedron],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_read_write_point_data(dtype, cell_type, degree, binary, shared_tmp_path):
    comm = MPI.COMM_WORLD
    filename = shared_tmp_path / "point_data.msh"
    real_dtype = np.finfo(dtype).dtype
    tol = 1e3 * np.finfo(dtype).eps

    mesh = create_curved_mesh(comm, cell_type, degree, real_dtype)
    backend_args = {"binary": binary, "gdim": mesh.geometry.dim, "dtype": real_dtype}
    io4dolfinx.write_mesh(filename, mesh, backend_args=backend_args)

    V = functionspace(mesh, ("Lagrange", degree, (2,)))
    u = Function(V, name="u", dtype=dtype)
    for t in [1.0, 2.0]:
        u.interpolate(lambda x: f(x, t, dtype))
        io4dolfinx.write_point_data(
            filename,
            u,
            time=t,
            mode=io4dolfinx.FileMode.append,
            backend_args=backend_args,
            backend="gmsh",
        )

    in_mesh = io4dolfinx.read_mesh(filename, comm, backend_args=backend_args)
    assert in_mesh.topology.index_map(in_mesh.topology.dim).size_global == (
        mesh.topology.index_map(mesh.topology.dim).size_global
    )
    u_in = io4dolfinx.read_point_data(
        filename, "u", in_mesh, time=2.0, backend_args=backend_args, backend="gmsh"
    )
    u_ref = Function(u_in.function_space, dtype=dtype)
    u_ref.interpolate(lambda x: f(x, 2.0, dtype))
    np.testing.assert_allclose(u_in.x.array, u_ref.x.array, atol=tol)

    timestamps = io4dolfinx.read_timestamps(filename, comm, "u")
    np.testing.assert_allclose(timestamps, [1.0, 2.0])
    assert io4dolfinx.read_function_names(filename, comm, backend="gmsh") == ["u"]


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize(
    "cell_type",
    [CellType.triangle, CellType.quadrilateral, CellType.tetrahedron, CellType.hexahedron],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_read_write_cell_data(dtype, cell_type, binary, shared_tmp_path):
    comm = MPI.COMM_WORLD
    filename = shared_tmp_path / "cell_data.msh"
    real_dtype = np.finfo(dtype).dtype
    tol = 1e3 * np.finfo(dtype).eps

    mesh = create_curved_mesh(comm, cell_type, 1, real_dtype)
    backend_args = {"binary": binary, "gdim": mesh.geometry.dim, "dtype": real_dtype}
    io4dolfinx.write_mesh(filename, mesh, backend_args=backend_args)

    V = functionspace(mesh, ("Lagrange", 1, (2,)))
    u = Function(V, name="q", dtype=dtype)
    for t in [1.0, 2.0]:
        u.interpolate(lambda x: f(x, t, dtype))
        io4dolfinx.write_cell_data(
            filename,
            u,
            time=t,
            mode=io4dolfinx.FileMode.append,
            backend_args=backend_args,
            backend="gmsh",
        )

    in_mesh = io4dolfinx.read_mesh(filename, comm, backend_args=backend_args)
    q_in = io4dolfinx.read_cell_data(
        filename, "q", in_mesh, time=2.0, backend_args=backend_args, backend="gmsh"
    )
    # The written data is the P1 interpolant evaluated at the cells
    u_ref = Function(functionspace(in_mesh, ("Lagrange", 1, (2,))), dtype=dtype)
    u_ref.interpolate(lambda x: f(x, 2.0, dtype))
    q_ref = Function(q_in.function_space, dtype=dtype)
    q_ref.interpolate(u_ref)
    np.testing.assert_allclose(q_in.x.array, q_ref.x.array, atol=tol)


@pytest.mark.parametrize("cell_type", [CellType.triangle, CellType.hexahedron])
def test_read_write_meshtags(cell_type, shared_tmp_path):
    comm = MPI.COMM_WORLD
    filename = shared_tmp_path / "meshtags.msh"

    mesh = create_curved_mesh(comm, cell_type, 1, np.float64)
    tdim = mesh.topology.dim
    backend_args = {"gdim": mesh.geometry.dim}
    io4dolfinx.write_mesh(filename, mesh, backend_args=backend_args)

    def facet_value(x):
        return np.where(np.isclose(x[0], 0.0), 3, 5)

    def cell_value(x):
        # Cells left of x=0.5 are tagged 2, the others keep the default marker 1
        return np.where(x[0] < 0.5, 2, 1)

    mesh.topology.create_connectivity(tdim - 1, tdim)
    facets = np.sort(
        locate_entities(mesh, tdim - 1, lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0))
    ).astype(np.int32)
    facet_midpoints = dolfinx.mesh.compute_midpoints(mesh, tdim - 1, facets).T
    ft = meshtags(mesh, tdim - 1, facets, facet_value(facet_midpoints).astype(np.int32))
    io4dolfinx.write_meshtags(filename, mesh, ft, "facets")

    cells = np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)
    cell_midpoints = dolfinx.mesh.compute_midpoints(mesh, tdim, cells).T
    cells = cells[cell_value(cell_midpoints) == 2]
    ct = meshtags(mesh, tdim, cells, np.full(len(cells), 2, dtype=np.int32))
    io4dolfinx.write_meshtags(filename, mesh, ct, "cells")

    def num_owned(mesh, dim, entities):
        return comm.allreduce(np.count_nonzero(entities < mesh.topology.index_map(dim).size_local))

    in_mesh = io4dolfinx.read_mesh(filename, comm, backend_args=backend_args)
    num_cells = mesh.topology.index_map(tdim).size_global
    for name, dim, value_fn, num_tagged in [
        ("facet", tdim - 1, facet_value, num_owned(mesh, tdim - 1, ft.indices)),
        ("facets", tdim - 1, facet_value, num_owned(mesh, tdim - 1, ft.indices)),
        ("cell", tdim, cell_value, num_cells),
        ("cells", tdim, cell_value, num_owned(mesh, tdim, ct.indices)),
    ]:
        mt = io4dolfinx.read_meshtags(filename, in_mesh, name)
        assert mt.dim == dim
        midpoints = dolfinx.mesh.compute_midpoints(in_mesh, dim, mt.indices).T
        np.testing.assert_array_equal(mt.values, value_fn(midpoints))
        assert num_owned(in_mesh, dim, mt.indices) == num_tagged

    # Every cell is in one physical group, so DOLFINx can read the file as well
    mesh_data = dolfinx.io.gmsh.read_from_msh(filename, comm, gdim=mesh.geometry.dim)
    assert mesh_data.cell_tags is not None and mesh_data.facet_tags is not None
    midpoints = dolfinx.mesh.compute_midpoints(mesh_data.mesh, tdim, mesh_data.cell_tags.indices).T
    np.testing.assert_array_equal(mesh_data.cell_tags.values, cell_value(midpoints))


def test_cell_tags_replace_markers(shared_tmp_path):
    comm = MPI.COMM_WORLD
    filename = shared_tmp_path / "markers.msh"

    mesh = create_unit_square(comm, 8, 2)
    tdim = mesh.topology.dim
    backend_args = {"gdim": mesh.geometry.dim}
    io4dolfinx.write_mesh(filename, mesh, backend_args=backend_args)

    cells = np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)
    x = dolfinx.mesh.compute_midpoints(mesh, tdim, cells)[:, 0]
    for selected, value in [(x < 0.5, 2), (x < 0.25, 3), (x > 0.75, 2)]:
        ct = meshtags(mesh, tdim, cells[selected], np.full(np.count_nonzero(selected), value))
        io4dolfinx.write_meshtags(filename, mesh, ct, "cells")

    in_mesh = io4dolfinx.read_mesh(filename, comm, backend_args=backend_args)
    mt = io4dolfinx.read_meshtags(filename, in_mesh, "cell")
    x = dolfinx.mesh.compute_midpoints(in_mesh, tdim, mt.indices)[:, 0]
    expected = np.select([x < 0.25, x < 0.5, x > 0.75], [3, 2, 2], default=1)
    np.testing.assert_array_equal(mt.values, expected)


@pytest.mark.parametrize(
    "cell_type", [CellType.quadrilateral, CellType.tetrahedron, CellType.hexahedron]
)
def test_written_file_readable_by_dolfinx(cell_type, shared_tmp_path):
    comm = MPI.COMM_WORLD
    filename = shared_tmp_path / "mesh.msh"

    mesh = create_curved_mesh(comm, cell_type, 2, np.float64)
    io4dolfinx.write_mesh(filename, mesh)
    mesh_data = dolfinx.io.gmsh.read_from_msh(filename, comm, gdim=mesh.geometry.dim)
    in_mesh = mesh_data.mesh
    tdim = mesh.topology.dim
    assert in_mesh.topology.index_map(tdim).size_global == (
        mesh.topology.index_map(tdim).size_global
    )
    assert in_mesh.geometry.index_map().size_global == mesh.geometry.index_map().size_global
    # write_mesh marks every cell with the default marker 1
    assert mesh_data.cell_tags is not None
    np.testing.assert_array_equal(mesh_data.cell_tags.values, 1)

    # A wrong node ordering tangles the cells and changes the volume of the unit domain
    volume = dolfinx.fem.form(1 * ufl.dx(domain=in_mesh), dtype=np.float64)
    assert np.isclose(comm.allreduce(dolfinx.fem.assemble_scalar(volume)), 1.0)


def test_user_gmsh_session_is_kept(shared_tmp_path):
    comm = MPI.COMM_WORLD
    filename = shared_tmp_path / "session.msh"

    mesh = create_curved_mesh(comm, CellType.triangle, 1, np.float64)
    io4dolfinx.write_mesh(filename, mesh)
    V = functionspace(mesh, ("Lagrange", 1))
    u = Function(V, name="u", dtype=np.float64)
    u.interpolate(lambda x: x[0])
    io4dolfinx.write_point_data(
        filename, u, time=0.0, mode=io4dolfinx.FileMode.append, backend_args=None, backend="gmsh"
    )

    gmsh.initialize(interruptible=False)
    try:
        gmsh.model.add("user")
        user_view = gmsh.view.add("user_data")
        io4dolfinx.read_mesh(filename, comm, backend_args={"gdim": 2})
        assert io4dolfinx.read_function_names(filename, comm, backend="gmsh") == ["u"]
        assert gmsh.isInitialized()
        assert gmsh.model.getCurrent() == "user"
        assert list(gmsh.view.getTags()) == [user_view]
    finally:
        gmsh.finalize()


def test_node_ordering_matches_dolfinx():
    """The node ordering computed from Gmsh's reference coordinates matches the
    hand-written orderings of dolfinx.io.gmsh."""
    from dolfinx.io.gmsh import _gmsh_to_cells, cell_perm_array

    from io4dolfinx.backends.gmsh.backend import _element, _gmsh_type

    gmsh.initialize(interruptible=False)
    try:
        for gmsh_type, (cell, degree) in _gmsh_to_cells.items():
            dim, element = _element(gmsh_type)
            assert element is not None
            assert element[:2] == (cell, degree)
            assert dim == dolfinx.mesh.cell_dim(dolfinx.mesh.to_type(cell))
            num_nodes = len(element[2])
            expected = cell_perm_array(dolfinx.mesh.to_type(cell), num_nodes)
            np.testing.assert_array_equal(element[2], expected)
            assert _gmsh_type(cell, degree) == gmsh_type
    finally:
        gmsh.finalize()


def generate(filename, kind, order, comm, h=0.5, incomplete=False):
    """Generate a curved Gmsh mesh with cell size ``h`` on rank 0, and return the exact
    area or volume of the domain.

    The domain is the unit ball for tetrahedra, the unit disk for triangles and
    quadrilaterals, and a cylinder of height 1 over it, one cell high, for hexahedra and
    prisms. ``incomplete`` makes Gmsh's serendipity elements.
    """
    if comm.rank == 0:
        gmsh.initialize(interruptible=False)
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            for option in ("Mesh.MeshSizeMin", "Mesh.MeshSizeMax"):
                gmsh.option.setNumber(option, h)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.setNumber("Mesh.SecondOrderIncomplete", int(incomplete))
            if kind == "tetrahedron":
                gmsh.model.occ.addSphere(0, 0, 0, 1)
            else:
                gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
                if kind in ("quadrilateral", "hexahedron"):
                    gmsh.option.setNumber("Mesh.RecombineAll", 1)
                if kind in ("hexahedron", "prism"):
                    gmsh.model.occ.extrude([(2, 1)], 0, 0, 1, numElements=[1], recombine=True)
            gmsh.model.occ.synchronize()
            tdim = gmsh.model.getDimension()
            gmsh.model.addPhysicalGroup(tdim, [e[1] for e in gmsh.model.getEntities(tdim)], 1)
            gmsh.model.mesh.generate(tdim)
            gmsh.model.mesh.setOrder(order)
            gmsh.write(str(filename))
        finally:
            gmsh.finalize()
    comm.barrier()
    return 4 / 3 * np.pi if kind == "tetrahedron" else np.pi


def check_geometry_against_gmsh(filename, mesh):
    """Check that DOLFINx's coordinate map equals Gmsh's on every local cell: the points and
    the Jacobians at the Gauss points of the reference cell, which a misplaced node of any
    kind changes."""
    from io4dolfinx.backends.gmsh.backend import _reference_map

    tdim, gdim = mesh.topology.dim, mesh.geometry.dim
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(filename))
        (gmsh_type,) = gmsh.model.mesh.getElementTypes(tdim)
        X_gmsh, _ = gmsh.model.mesh.getIntegrationPoints(gmsh_type, "Gauss3")
        J_gmsh, _, x_gmsh = gmsh.model.mesh.getJacobians(gmsh_type, X_gmsh)
        element_tags, _ = gmsh.model.mesh.getElementsByType(gmsh_type)
        M = _reference_map(mesh.topology.cell_name())
    finally:
        gmsh.finalize()
    # Gmsh's Gauss points are on Gmsh's reference cell, e.g. [-1, 1]^2 for a quadrilateral,
    # while Expression takes points on Basix's, [0, 1]^2. M = [A; b] holds the affine map
    # X = X_gmsh A + b between them, applied to [X_gmsh, 1]; for simplices it is the identity.
    X_gmsh = np.reshape(X_gmsh, (-1, 3))[:, :tdim]
    X = np.hstack([X_gmsh, np.ones((len(X_gmsh), 1))]) @ M
    # Input cell i is the element with the i-th smallest tag
    order = np.argsort(element_tags)
    x_gmsh = np.reshape(x_gmsh, (len(element_tags), len(X), 3))[order]
    # Row i of a Gmsh Jacobian holds the derivatives along reference coordinate i
    J_gmsh = np.reshape(J_gmsh, (len(element_tags), len(X), 3, 3))[order]

    num_cells = mesh.topology.index_map(tdim).size_local
    cells = np.arange(num_cells, dtype=np.int32)
    original = mesh.topology.original_cell_index[:num_cells]
    x = dolfinx.fem.Expression(ufl.SpatialCoordinate(mesh), X).eval(mesh, cells)
    x = np.reshape(x, (num_cells, len(X), gdim))
    np.testing.assert_allclose(x, x_gmsh[original][:, :, :gdim], atol=1e-12)
    J = dolfinx.fem.Expression(ufl.Jacobian(mesh), X).eval(mesh, cells)
    J = np.reshape(J, (num_cells, len(X), gdim, tdim))
    # Chain rule through the affine map between the reference cells
    expected = np.einsum("ij,cpkj->cpik", M[:tdim], J)
    np.testing.assert_allclose(J_gmsh[original][:, :, :tdim, :gdim], expected, atol=1e-11)


@pytest.mark.parametrize(
    "kind, orders",
    [
        ("triangle", [3, 4, 6]),
        ("quadrilateral", [3, 4, 6]),
        ("tetrahedron", [3, 4, 5]),
        ("hexahedron", [3, 4]),
        # Gmsh cannot evaluate prisms of order 3 and above
        ("prism", [2]),
    ],
)
def test_geometry_matches_gmsh(kind, orders, shared_tmp_path):
    """Higher-order meshes from Gmsh, beyond the orders of dolfinx.io.gmsh, have the
    geometry Gmsh computes."""
    comm = MPI.COMM_WORLD
    for order in orders:
        filename = shared_tmp_path / f"{kind}_{order}.msh"
        generate(filename, kind, order, comm, h=0.8)
        mesh = io4dolfinx.read_mesh(filename, comm, backend_args={"gdim": 3})
        assert mesh.topology.cell_name() == kind
        assert fenicsx_compat.cmap(mesh).degree == order
        check_geometry_against_gmsh(filename, mesh)


@pytest.mark.parametrize(
    "kind, orders, sizes",
    [
        ("triangle", [2, 3, 4, 5, 6], [0.8, 0.4]),
        ("quadrilateral", [3, 4, 6], [0.8, 0.4]),
        ("tetrahedron", [3, 4, 5], [0.4, 0.27]),
        ("hexahedron", [3, 4], [0.8, 0.4]),
        ("prism", [2, 3], [0.8, 0.4]),
    ],
)
def test_volume_convergence(kind, orders, sizes, shared_tmp_path):
    """The error in the area or volume of a curved domain converges at the rate of the
    boundary interpolation, ``k + 2`` for even and ``k + 1`` for odd degree ``k``."""
    comm = MPI.COMM_WORLD
    for order in orders:
        errors = []
        for h in sizes:
            filename = shared_tmp_path / f"{kind}_{order}_{h}.msh"
            exact = generate(filename, kind, order, comm, h=h)
            mesh = io4dolfinx.read_mesh(filename, comm, backend_args={"gdim": 3})
            volume = dolfinx.fem.form(1 * ufl.dx(domain=mesh), dtype=np.float64)
            errors.append(abs(comm.allreduce(dolfinx.fem.assemble_scalar(volume)) - exact))
        rate = np.log(errors[0] / errors[1]) / np.log(sizes[0] / sizes[1])
        assert rate > 2 * (order // 2) + 2 - 0.5, (order, errors, rate)


def generate_torus(filename, kind, order, h, comm):
    """Generate a curved Gmsh surface mesh of a torus with cell size ``h`` on rank 0, and
    return its exact area."""
    major, minor = 1.0, 0.4
    if comm.rank == 0:
        gmsh.initialize(interruptible=False)
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            for option in ("Mesh.MeshSizeMin", "Mesh.MeshSizeMax"):
                gmsh.option.setNumber(option, h)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.setNumber("Mesh.RecombineAll", int(kind == "quadrilateral"))
            gmsh.model.occ.addTorus(0, 0, 0, major, minor)
            gmsh.model.occ.synchronize()
            gmsh.model.addPhysicalGroup(2, [e[1] for e in gmsh.model.getEntities(2)], 1)
            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.setOrder(order)
            gmsh.write(str(filename))
        finally:
            gmsh.finalize()
    comm.barrier()
    return 4 * np.pi**2 * major * minor


# Measured convergence rates of the area of a torus. On curved surfaces, Gmsh's
# triangles of even degree 4 and above lose one order compared with its quadrilaterals.
_surface_rates = {
    "triangle": {2: 4, 3: 4, 4: 4, 5: 6, 6: 6},
    "quadrilateral": {2: 4, 3: 4, 4: 6, 5: 6, 6: 8},
}


@pytest.mark.parametrize("kind", ["triangle", "quadrilateral"])
def test_surface_area_convergence(kind, shared_tmp_path):
    """On a surface mesh every node lies on the curved surface, so unlike a volume, the
    area also depends on the nodes inside the cells. A misplaced node stops convergence."""
    comm = MPI.COMM_WORLD
    sizes = [0.3, 0.15]
    for order, expected in _surface_rates[kind].items():
        errors, h = [], []
        for size in sizes:
            filename = shared_tmp_path / f"torus_{kind}_{order}_{size}.msh"
            exact = generate_torus(filename, kind, order, size, comm)
            mesh = io4dolfinx.read_mesh(filename, comm, backend_args={"gdim": 3})
            area = dolfinx.fem.form(1 * ufl.dx(domain=mesh), dtype=np.float64)
            errors.append(abs(comm.allreduce(dolfinx.fem.assemble_scalar(area)) - exact))
            # Gmsh does not keep the requested size on a torus, so measure it
            h.append(np.sqrt(exact / mesh.topology.index_map(2).size_global))
        rate = np.log(errors[0] / errors[1]) / np.log(h[0] / h[1])
        assert rate > expected - 0.5, (order, errors, rate)


@pytest.mark.parametrize("kind, num_nodes", [("quadrilateral", 8), ("hexahedron", 20)])
def test_read_serendipity_gmsh_mesh(kind, num_nodes, shared_tmp_path):
    """Gmsh's 8-node quadrilateral and 20-node hexahedron are read as serendipity
    geometry, with the geometry Gmsh computes."""
    if not hasattr(dolfinx.io.utils, "_serendipity_coordinate_element"):
        pytest.skip("Serendipity geometry requires the main branch of DOLFINx")
    comm = MPI.COMM_WORLD
    filename = shared_tmp_path / f"{kind}_serendipity.msh"
    generate(filename, kind, 2, comm, incomplete=True)
    mesh = io4dolfinx.read_mesh(filename, comm, backend_args={"gdim": 3})
    assert mesh.topology.cell_name() == kind
    assert fenicsx_compat.dofmap(mesh).shape[1] == num_nodes
    check_geometry_against_gmsh(filename, mesh)


def test_reject_incomplete_element_without_coordinate_element(shared_tmp_path):
    """DOLFINx has no coordinate element for Gmsh's 12-node quadrilateral."""
    comm = MPI.COMM_WORLD
    filename = shared_tmp_path / "quad12.msh"
    generate(filename, "quadrilateral", 3, comm, incomplete=True)
    with pytest.raises(NotImplementedError, match="unsupported cell types"):
        io4dolfinx.read_mesh(filename, comm, backend_args={"gdim": 3})


@pytest.mark.parametrize(
    "cell_type, degree",
    [
        (CellType.triangle, 5),
        (CellType.quadrilateral, 4),
        (CellType.tetrahedron, 4),
        (CellType.hexahedron, 4),
    ],
)
def test_write_read_higher_order_mesh(cell_type, degree, shared_tmp_path):
    comm = MPI.COMM_WORLD
    filename = shared_tmp_path / "higher_order.msh"
    mesh = create_curved_mesh(comm, cell_type, degree, np.float64)
    # Bend the geometry, so that the integrals below depend on every node
    x = mesh.geometry.x
    x[:, 0] += 0.1 * np.sin(np.pi * x[:, 1]) * x[:, 0]
    x[:, 1] += 0.1 * x[:, 0] ** 2
    io4dolfinx.write_mesh(filename, mesh, backend_args={"gdim": mesh.geometry.dim})
    in_mesh = io4dolfinx.read_mesh(filename, comm, backend_args={"gdim": mesh.geometry.dim})
    assert fenicsx_compat.cmap(in_mesh).degree == degree

    def integrals(m):
        X = ufl.SpatialCoordinate(m)
        forms = [1 * ufl.dx(domain=m), X[0] ** 2 * X[1] * ufl.dx(domain=m)]
        return [
            comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(a, dtype=np.float64)))
            for a in forms
        ]

    np.testing.assert_allclose(integrals(in_mesh), integrals(mesh), rtol=1e-12)
