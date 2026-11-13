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
