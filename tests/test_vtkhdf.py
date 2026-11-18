from mpi4py import MPI

import h5py
import numpy as np
import pytest
import ufl
from dolfinx.fem import Function, assemble_scalar, form, functionspace
from dolfinx.io.vtkhdf import write_cell_data, write_mesh, write_point_data
from dolfinx.mesh import (
    CellType,
    compute_midpoints,
    create_unit_cube,
    create_unit_interval,
    create_unit_square,
    meshtags,
)

import io4dolfinx


def f(x, t):
    return x[0] - 2 * x[1] + x[2] * t


def g(x, t):
    return x[0], 2 * x[1], -x[2] * t


@pytest.fixture(params=["vtkhdf", "vtkhdf-serial"])
def vtkhdf_backend(request):
    """The VTKHDF backend, and the one for h5py without MPI."""
    return request.param


@pytest.mark.parametrize(
    "cell_type", [CellType.tetrahedron, CellType.hexahedron, CellType.tetrahedron]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_read_write_timedep_mesh(dtype, tmp_path, cell_type, vtkhdf_backend):
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path, root=0)
    comm.barrier()

    # Write temporal data
    mesh = create_unit_cube(comm, 5, 5, 5, dtype=dtype, cell_type=cell_type)
    ref_vol = mesh.comm.allreduce(
        assemble_scalar(form(1 * ufl.dx(domain=mesh), dtype=dtype)), op=MPI.SUM
    )
    ref_surf = mesh.comm.allreduce(
        assemble_scalar(form(1 * ufl.ds(domain=mesh), dtype=dtype)), op=MPI.SUM
    )

    # Write temporal data
    filename = tmp_path / f"timedep_mesh_{cell_type.name}_{np.dtype(dtype).name}.vtkhdf"
    io4dolfinx.write_mesh(filename, mesh, time=0.3, backend=vtkhdf_backend)
    mesh.geometry.x[:, 0] += 0.05 * mesh.geometry.x[:, 0]
    mesh.geometry.x[:, 1] *= 1.1 + np.sin(mesh.geometry.x[:, 0])

    ref_pert_vol = mesh.comm.allreduce(
        assemble_scalar(form(1 * ufl.dx(domain=mesh), dtype=dtype)), op=MPI.SUM
    )
    ref_pert_surf = mesh.comm.allreduce(
        assemble_scalar(form(1 * ufl.ds(domain=mesh), dtype=dtype)), op=MPI.SUM
    )

    io4dolfinx.write_mesh(
        filename,
        mesh,
        time=0.5,
        backend=vtkhdf_backend,
        mode=io4dolfinx.FileMode.append,
    )

    in_mesh = io4dolfinx.read_mesh(filename, comm, time=0.5, backend=vtkhdf_backend)
    pert_vol = mesh.comm.allreduce(
        assemble_scalar(form(1 * ufl.dx(domain=in_mesh), dtype=dtype)), op=MPI.SUM
    )
    pert_surf = mesh.comm.allreduce(
        assemble_scalar(form(1 * ufl.ds(domain=in_mesh), dtype=dtype)), op=MPI.SUM
    )
    assert np.isclose(pert_vol, ref_pert_vol)
    assert np.isclose(pert_surf, ref_pert_surf)

    in_mesh = io4dolfinx.read_mesh(filename, comm, time=0.3, backend=vtkhdf_backend)
    vol = mesh.comm.allreduce(
        assemble_scalar(form(1 * ufl.dx(domain=in_mesh), dtype=dtype)), op=MPI.SUM
    )
    surf = mesh.comm.allreduce(
        assemble_scalar(form(1 * ufl.ds(domain=in_mesh), dtype=dtype)), op=MPI.SUM
    )
    assert np.isclose(vol, ref_vol)
    assert np.isclose(surf, ref_surf)


@pytest.mark.parametrize("cell_type", [CellType.hexahedron, CellType.tetrahedron])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_write_point_data(dtype, tmp_path, cell_type, vtkhdf_backend):
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path, root=0)
    comm.barrier()

    # Write temporal data
    mesh = create_unit_cube(comm, 5, 5, 5, dtype=dtype, cell_type=cell_type)
    filename = tmp_path / f"point_data_{cell_type.name}_{np.dtype(dtype).name}.vtkhdf"
    write_mesh(str(filename), mesh)
    t = np.linspace(0.1, 1.2, 25)
    num_nodes_local = mesh.geometry.index_map().size_local
    for ti in t:
        point_data = f(mesh.geometry.x.T[:, :num_nodes_local], ti)
        write_point_data(str(filename), mesh, point_data, float(ti))
    comm.barrier()

    grid = io4dolfinx.read_mesh(filename=filename, comm=comm, time=None, backend=vtkhdf_backend)
    # Since we shuffle time we need to shuffle in the same way on each process
    np.random.shuffle(t)
    t = comm.bcast(t, root=0)
    for tj in t:
        u = io4dolfinx.read_point_data(
            filename=filename, name="u", mesh=grid, time=tj, backend=vtkhdf_backend
        )
        v_ref = Function(u.function_space, dtype=u.x.array.dtype)
        atol = 15 * np.finfo(u.x.array.dtype).eps
        v_ref.interpolate(lambda x: f(x, tj))
        np.testing.assert_allclose(u.x.array, v_ref.x.array, atol=atol)

    # Test blocked data as well (with shuffled input timestep)
    blocked_file = filename.with_stem(filename.stem + "_blocked")
    write_mesh(str(blocked_file), mesh)
    for tj in t:
        point_data = np.asarray(g(mesh.geometry.x.T[:, :num_nodes_local], tj)).T.flatten()
        write_point_data(str(blocked_file), mesh, point_data, float(tj))
    comm.barrier()

    np.random.shuffle(t)
    t = comm.bcast(t, root=0)
    for tk in t:
        u = io4dolfinx.read_point_data(
            filename=blocked_file, name="u", mesh=grid, time=tk, backend=vtkhdf_backend
        )
        v_ref = Function(u.function_space, dtype=u.x.array.dtype)
        atol = 15 * np.finfo(u.x.array.dtype).eps
        v_ref.interpolate(lambda x: g(x, tk))
        np.testing.assert_allclose(u.x.array, v_ref.x.array, atol=atol)


@pytest.mark.parametrize("cell_type", [CellType.hexahedron, CellType.tetrahedron])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_write_cell_data(dtype, tmp_path, cell_type, vtkhdf_backend):
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path, root=0)
    comm.barrier()

    # Write temporal data
    mesh = create_unit_cube(comm, 5, 5, 5, dtype=dtype, cell_type=cell_type)
    filename = tmp_path / f"cell_data_{cell_type.name}_{np.dtype(dtype).name}.vtkhdf"
    write_mesh(str(filename), mesh)

    t = np.linspace(0.1, 1.2, 25)
    num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
    midpoints = compute_midpoints(
        mesh, mesh.topology.dim, np.arange(num_cells_local, dtype=np.int32)
    )
    for ti in t:
        cell_data = f(midpoints.T, ti)
        write_cell_data(str(filename), mesh, cell_data, float(ti))
    comm.barrier()

    grid = io4dolfinx.read_mesh(filename=filename, comm=comm, time=None, backend=vtkhdf_backend)
    # Since we shuffle time we need to shuffle in the same way on each process
    np.random.shuffle(t)
    t = comm.bcast(t, root=0)
    for tj in t:
        u = io4dolfinx.read_cell_data(
            filename=filename, name="u", mesh=grid, time=tj, backend=vtkhdf_backend
        )
        v_ref = Function(u.function_space, dtype=u.x.array.dtype)
        atol = 15 * np.finfo(u.x.array.dtype).eps
        v_ref.interpolate(lambda x: f(x, tj))
        np.testing.assert_allclose(u.x.array, v_ref.x.array, atol=atol)

    # Test blocked data as well (with shuffled input timestep)
    blocked_file = filename.with_stem(filename.stem + "_blocked")
    write_mesh(str(blocked_file), mesh)
    for tj in t:
        cell_data = np.asarray(g(midpoints.T, tj)).T.flatten()
        write_cell_data(str(blocked_file), mesh, cell_data, float(tj))
    comm.barrier()

    np.random.shuffle(t)
    t = comm.bcast(t, root=0)
    for tk in t:
        u = io4dolfinx.read_cell_data(
            filename=blocked_file, name="u", mesh=grid, time=tk, backend=vtkhdf_backend
        )
        v_ref = Function(u.function_space, dtype=u.x.array.dtype)
        atol = 15 * np.finfo(u.x.array.dtype).eps
        v_ref.interpolate(lambda x: g(x, tk))
        np.testing.assert_allclose(u.x.array, v_ref.x.array, atol=atol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_write_meshtags(dtype, tmp_path, generate_reference_map, vtkhdf_backend):
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path, root=0)
    comm.barrier()

    filename = tmp_path / f"meshtags_{np.dtype(dtype).name}.vtkhdf"

    mesh = create_unit_cube(comm, 3, 3, 3, dtype=dtype, cell_type=CellType.hexahedron)
    io4dolfinx.write_mesh(
        filename,
        mesh,
        mode=io4dolfinx.FileMode.write,
        time=1.0,
        backend_args={"name": "hex"},
        backend=vtkhdf_backend,
    )
    dim = mesh.topology.dim
    mesh.topology.create_connectivity(dim, mesh.topology.dim)
    cmap = mesh.topology.index_map(dim)
    cells = np.arange(cmap.size_local, dtype=np.int32)
    ct = meshtags(mesh, dim, cells, cells + cmap.local_range[0])
    io4dolfinx.write_meshtags(
        filename, mesh, ct, "CellTags", backend_args={"name": "hex"}, backend=vtkhdf_backend
    )
    root = 0

    org_map = generate_reference_map(mesh, ct, comm, root)
    # Move mesh
    mesh.geometry.x[:, 0] *= 1 + 0.2 * np.sin(mesh.geometry.x[:, 1])

    io4dolfinx.write_mesh(
        filename,
        mesh,
        mode=io4dolfinx.FileMode.append,
        time=2.5,
        backend_args={"name": "hex"},
        backend=vtkhdf_backend,
    )

    # Add stationary meshtags (after time loop)
    mesh = create_unit_cube(comm, 7, 3, 5, dtype=dtype, cell_type=CellType.tetrahedron)
    io4dolfinx.write_mesh(
        filename,
        mesh,
        mode=io4dolfinx.FileMode.append,
        time=1.0,
        backend_args={"name": "tet"},
        backend=vtkhdf_backend,
    )
    dim = mesh.topology.dim
    org_maps = {}
    mesh.geometry.x[:, 0] *= 2.0 + mesh.geometry.x[:, 1]
    io4dolfinx.write_mesh(
        filename,
        mesh,
        mode=io4dolfinx.FileMode.append,
        time=2.5,
        backend_args={"name": "tet"},
        backend=vtkhdf_backend,
    )
    for dim in range(mesh.topology.dim + 1):
        mesh.topology.create_connectivity(dim, mesh.topology.dim)
        entities = np.arange(mesh.topology.index_map(dim).size_local, dtype=np.int32)
        et = meshtags(mesh, dim, entities, entities + mesh.topology.index_map(dim).local_range[0])
        io4dolfinx.write_meshtags(
            filename, mesh, et, f"{dim}tags", backend_args={"name": "tet"}, backend=vtkhdf_backend
        )
        org_maps[dim] = generate_reference_map(mesh, et, comm, root)

    tol = 15 * np.finfo(dtype).eps
    # Read in hex grid from second time step
    hex_mesh = io4dolfinx.read_mesh(
        filename, comm, time=1.0, backend_args={"name": "hex"}, backend=vtkhdf_backend
    )
    hex_tag = io4dolfinx.read_meshtags(
        filename, hex_mesh, "CellTags", backend_args={"name": "hex"}, backend=vtkhdf_backend
    )
    read_map = generate_reference_map(hex_mesh, hex_tag, comm, root)
    # On root process, check that midpoints are the same for each value in the meshtag
    if MPI.COMM_WORLD.rank == root:
        assert len(org_map) == len(read_map)
        for value, (_, midpoint) in org_map.items():
            _, read_midpoint = read_map[value]
            np.testing.assert_allclose(read_midpoint, midpoint, atol=tol)

    # Read tet grid from second time step
    tet_mesh = io4dolfinx.read_mesh(
        filename, comm, time=2.5, backend_args={"name": "tet"}, backend=vtkhdf_backend
    )
    for dim in range(mesh.topology.dim + 1):
        tet_tag = io4dolfinx.read_meshtags(
            filename, tet_mesh, f"{dim}tags", backend_args={"name": "tet"}, backend=vtkhdf_backend
        )
        read_map = generate_reference_map(tet_mesh, tet_tag, comm, root)
        if MPI.COMM_WORLD.rank == root:
            assert len(org_maps[dim]) == len(read_map)
            for value, (_, midpoint) in org_maps[dim].items():
                _, read_midpoint = read_map[value]
                np.testing.assert_allclose(read_midpoint, midpoint, atol=tol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_read_write_pointdata(dtype, tmp_path, vtkhdf_backend):
    tol = 15 * np.finfo(dtype).eps

    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path, root=0)
    comm.barrier()

    filename = tmp_path / "point_data.vtkhdf"

    mesh = create_unit_cube(comm, 3, 3, 3, dtype=dtype, cell_type=CellType.hexahedron)

    def f(x, t):
        return (x[0] + np.sin(x[1]) + np.cos(x[0] * t), x[2] + x[1] - t)

    io4dolfinx.write_mesh(
        filename,
        mesh,
        mode=io4dolfinx.FileMode.write,
        time=1.0,
        backend_args={"name": "hex"},
        backend=vtkhdf_backend,
    )

    f_name = "point_data"
    V = functionspace(mesh, ("Lagrange", 2, (2,)))
    u = Function(V, dtype=dtype, name=f_name)
    u.interpolate(lambda x: f(x, 1.0))
    io4dolfinx.write_point_data(
        filename,
        u,
        mode=io4dolfinx.FileMode.append,
        time=1.0,
        backend_args={"name": "hex"},
        backend=vtkhdf_backend,
    )

    io4dolfinx.write_mesh(
        filename,
        mesh,
        mode=io4dolfinx.FileMode.append,
        time=2.0,
        backend_args={"name": "hex"},
        backend=vtkhdf_backend,
    )
    u.interpolate(lambda x: f(x, 2.0))
    io4dolfinx.write_point_data(
        filename,
        u,
        mode=io4dolfinx.FileMode.append,
        time=2.0,
        backend_args={"name": "hex"},
        backend=vtkhdf_backend,
    )

    # Read in hex grid from second time step
    hex_mesh = io4dolfinx.read_mesh(
        filename, comm, time=2.0, backend_args={"name": "hex"}, backend=vtkhdf_backend
    )
    u_end = io4dolfinx.read_point_data(
        filename,
        mesh=hex_mesh,
        name=f_name,
        time=2.0,
        backend_args={"name": "hex"},
        backend=vtkhdf_backend,
    )
    u_ref = Function(u_end.function_space, dtype=dtype)
    u_ref.interpolate(lambda x: f(x, 2.0))
    np.testing.assert_allclose(u_end.x.array, u_ref.x.array, atol=tol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_read_write_celldata(dtype, tmp_path, vtkhdf_backend):
    tol = 15 * np.finfo(dtype).eps

    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path, root=0)
    comm.barrier()

    filename = tmp_path / "cell_data.vtkhdf"

    mesh = create_unit_cube(comm, 3, 3, 3, dtype=dtype, cell_type=CellType.tetrahedron)

    def f(x, t):
        return (x[0] + np.sin(x[1]) + np.cos(x[0] * t), x[1] + t, x[0] - t)

    t_0 = 2.2
    t_1 = 3.0

    backend_args = {"name": "Grid"}
    io4dolfinx.write_mesh(
        filename,
        mesh,
        mode=io4dolfinx.FileMode.write,
        time=t_0,
        backend_args=backend_args,
        backend=vtkhdf_backend,
    )

    f_name = "Data"
    V = functionspace(mesh, ("DG", 0, (3,)))
    u = Function(V, dtype=dtype, name=f_name)
    u.interpolate(lambda x: f(x, t_0))
    io4dolfinx.write_cell_data(
        filename,
        u,
        mode=io4dolfinx.FileMode.append,
        time=t_0,
        backend_args=backend_args,
        backend=vtkhdf_backend,
    )

    io4dolfinx.write_mesh(
        filename,
        mesh,
        mode=io4dolfinx.FileMode.append,
        time=t_1,
        backend_args=backend_args,
        backend=vtkhdf_backend,
    )
    u.interpolate(lambda x: f(x, t_1))
    io4dolfinx.write_cell_data(
        filename,
        u,
        mode=io4dolfinx.FileMode.append,
        time=t_1,
        backend_args=backend_args,
        backend=vtkhdf_backend,
    )

    for t in [t_1, t_0]:
        grid = io4dolfinx.read_mesh(
            filename, comm, time=t, backend_args=backend_args, backend=vtkhdf_backend
        )
        u_end = io4dolfinx.read_cell_data(
            filename,
            mesh=grid,
            name=f_name,
            time=t,
            backend_args=backend_args,
            backend=vtkhdf_backend,
        )
        u_ref = Function(u_end.function_space, dtype=dtype)
        u_ref.interpolate(lambda x: f(x, t))
        np.testing.assert_allclose(u_end.x.array, u_ref.x.array, atol=tol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_read_write_mix_data(dtype, tmp_path, vtkhdf_backend):
    tol = 15 * np.finfo(dtype).eps

    mesh = create_unit_square(MPI.COMM_WORLD, 5, 7, dtype=dtype)

    def f(x, t):
        return x[0] + t * x[1]

    def g(x, t):
        return x[1] - x[0] * t**2

    ts = [0.1, 0.3, 0.4, 0.5]

    V = functionspace(mesh, ("Lagrange", 1))
    u = Function(V, name="points", dtype=dtype)
    z = Function(V, name="some_other_array", dtype=dtype)
    Q = functionspace(mesh, ("DG", 0))
    q = Function(Q, name="cells", dtype=dtype)
    tmp_path = mesh.comm.bcast(tmp_path, root=0)
    filename = tmp_path / "mixed_data.vtkhdf"

    backend_args = {"name": "MyGrid"}
    for i, t in enumerate(ts):
        if np.isclose(t, 0.1):
            mode = io4dolfinx.FileMode.write
        else:
            mode = io4dolfinx.FileMode.append
        mesh.geometry.x[:] *= 1 + 0.1 * t
        io4dolfinx.write_mesh(
            filename, mesh, mode=mode, time=t, backend=vtkhdf_backend, backend_args=backend_args
        )
        u.interpolate(lambda x: f(x, t))
        q.interpolate(lambda x: g(x, t))
        io4dolfinx.write_point_data(
            filename,
            u,
            time=t,
            mode=io4dolfinx.FileMode.append,
            backend_args=backend_args,
            backend=vtkhdf_backend,
        )
        io4dolfinx.write_point_data(
            filename,
            z,
            time=t,
            mode=io4dolfinx.FileMode.append,
            backend_args=backend_args,
            backend=vtkhdf_backend,
        )

        mesh_in = io4dolfinx.read_mesh(
            filename, MPI.COMM_WORLD, time=t, backend_args=backend_args, backend=vtkhdf_backend
        )
        u_in = io4dolfinx.read_point_data(
            filename,
            name=u.name,
            mesh=mesh_in,
            time=t,
            backend_args=backend_args,
            backend=vtkhdf_backend,
        )
        u_ref = Function(u_in.function_space, dtype=dtype)
        u_ref.interpolate(lambda x: f(x, t))
        np.testing.assert_allclose(u_ref.x.array, u_in.x.array, atol=tol)
        c_step = i
        if not np.isclose(t, 0.3):
            io4dolfinx.write_cell_data(
                filename,
                q,
                time=t,
                backend_args=backend_args,
                mode=io4dolfinx.FileMode.append,
                backend=vtkhdf_backend,
            )
        else:
            # Read in mesh from previous step as geometry adapts,
            # while reading data from current step.
            c_step = i - 1
            mesh_in = io4dolfinx.read_mesh(
                filename,
                MPI.COMM_WORLD,
                time=ts[c_step],
                backend_args=backend_args,
                backend=vtkhdf_backend,
            )
        q_in = io4dolfinx.read_cell_data(
            filename,
            name=q.name,
            mesh=mesh_in,
            time=t,
            backend_args=backend_args,
            backend=vtkhdf_backend,
        )
        q_ref = Function(q_in.function_space, dtype=dtype)
        q_ref.interpolate(lambda x: g(x, ts[c_step]))
        np.testing.assert_allclose(q_ref.x.array, q_in.x.array, atol=tol)


@pytest.mark.parametrize("mesh_name", [None, "MyGrid"])
def test_read_timestamps(tmp_path, vtkhdf_backend, mesh_name):
    """Every array has the time steps of its mesh, also an array first written after
    the first step, as a VTKHDF array has data at every time step."""
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path, root=0)
    filename = tmp_path / "timestamps.vtkhdf"
    mesh = create_unit_square(comm, 4, 3)
    V = functionspace(mesh, ("Lagrange", 1))
    q = Function(functionspace(mesh, ("DG", 0)), name="q")
    u, late = Function(V, name="u"), Function(V, name="late")
    io4dolfinx.write_mesh(filename, mesh, time=0.0, backend=vtkhdf_backend, mesh_name=mesh_name)
    for function, t in [(q, 0.0), (u, 0.5), (q, 1.0), (u, 1.0), (late, 2.0), (q, 2.5)]:
        writer = io4dolfinx.write_cell_data if function is q else io4dolfinx.write_point_data
        writer(
            filename,
            function,
            time=t,
            mode=io4dolfinx.FileMode.append,
            backend_args=None,
            backend=vtkhdf_backend,
            mesh_name=mesh_name,
        )
    for name in ("q", "u", "late"):
        stamps = io4dolfinx.read_timestamps(
            filename, comm, name, backend=vtkhdf_backend, mesh_name=mesh_name
        )
        np.testing.assert_allclose(stamps, [0.0, 0.5, 1.0, 2.0, 2.5])
    with pytest.raises(RuntimeError, match="not associated with a time-stamp"):
        io4dolfinx.read_timestamps(
            filename, comm, "missing", backend=vtkhdf_backend, mesh_name=mesh_name
        )


def write_point(filename, function, time, backend, mesh_name=None):
    """Append the point data of ``function`` at ``time``."""
    io4dolfinx.write_point_data(
        filename,
        function,
        time=time,
        mode=io4dolfinx.FileMode.append,
        backend_args=None,
        backend=backend,
        mesh_name=mesh_name,
    )


def file_contents(filename):
    """Every dataset of an HDF5 file, by path."""
    contents = {}

    def store(name, item):
        if isinstance(item, h5py.Dataset):
            contents[name] = item[()]

    MPI.COMM_WORLD.barrier()
    with h5py.File(filename, "r") as h5file:
        h5file.visititems(store)
    MPI.COMM_WORLD.barrier()
    return contents


def assert_same_contents(contents, reference):
    assert contents.keys() == reference.keys()
    for path, data in reference.items():
        np.testing.assert_array_equal(contents[path], data, err_msg=path)


def test_splitting_scheme(tmp_path, vtkhdf_backend):
    """A fractional step scheme computes the pressure p at the half step t_{n-1/2}, and
    then the velocity u at t_n. Written in that order, each array holds its latest value
    at the time steps of the other. Writing u before p, at an earlier time than the last
    step, and writing an array twice at a time, raise errors and leave the file as it
    was."""
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path, root=0)
    filename = tmp_path / "splitting.vtkhdf"
    mesh = create_unit_square(comm, 4, 3)
    u = Function(functionspace(mesh, ("Lagrange", 1, (2,))), name="u")
    p = Function(functionspace(mesh, ("Lagrange", 1)), name="p")

    def velocity(t):
        return lambda x: np.vstack((x[0] + t, x[1] - 2 * t))

    def pressure(t):
        return lambda x: x[0] * x[1] + t

    def solve_and_write(function, t):
        function.interpolate(velocity(t) if function is u else pressure(t))
        write_point(filename, function, t, vtkhdf_backend)

    dt, num_steps = 0.1, 3
    io4dolfinx.write_mesh(filename, mesh, time=0.0, backend=vtkhdf_backend)
    solve_and_write(u, 0.0)
    for n in range(1, num_steps + 1):
        solve_and_write(p, (n - 0.5) * dt)  # Tentative velocity, pressure correction
        solve_and_write(u, n * dt)  # Velocity update

    stamps = io4dolfinx.read_timestamps(filename, comm, "u", backend=vtkhdf_backend)
    np.testing.assert_allclose(stamps, 0.5 * dt * np.arange(2 * num_steps + 1))
    np.testing.assert_allclose(
        io4dolfinx.read_timestamps(filename, comm, "p", backend=vtkhdf_backend), stamps
    )
    # u is written at the even steps, p at the odd steps
    mesh_in = io4dolfinx.read_mesh(filename, comm, backend=vtkhdf_backend)
    for k, t in enumerate(stamps):
        u_time = (k // 2) * dt
        p_time = max(k - 1 + k % 2, 1) * dt / 2
        for name, exact, exact_time in (("u", velocity, u_time), ("p", pressure, p_time)):
            stored = io4dolfinx.read_point_data(
                filename, name, mesh_in, time=t, backend=vtkhdf_backend
            )
            expected = Function(stored.function_space)
            expected.interpolate(exact(exact_time))
            np.testing.assert_allclose(stored.x.array, expected.x.array, atol=1e-14)

    # The velocity of the next step written before its pressure
    solve_and_write(u, (num_steps + 1) * dt)
    contents = file_contents(filename)
    with pytest.raises(ValueError, match="must be later than the last one"):
        solve_and_write(p, (num_steps + 0.5) * dt)
    # A second write of the velocity at a time, up to round-off
    with pytest.raises(ValueError, match="has already been written at time"):
        solve_and_write(u, (num_steps + 1) * dt * (1 + 4 * np.finfo(np.float64).eps))
    with pytest.raises(ValueError, match="has already been written at time"):
        io4dolfinx.write_mesh(
            filename, mesh, time=0.0, mode=io4dolfinx.FileMode.append, backend=vtkhdf_backend
        )
    assert_same_contents(file_contents(filename), contents)


def test_write_at_earlier_step(tmp_path, vtkhdf_backend):
    """An array written at an earlier, existing time step holds that value until its next
    write, and the steps before its first write hold its first value."""
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path, root=0)
    filename = tmp_path / "earlier_step.vtkhdf"
    mesh = create_unit_square(comm, 4, 3)
    V = functionspace(mesh, ("Lagrange", 1))
    u, late, q = Function(V, name="u"), Function(V, name="late"), Function(V, name="q")
    io4dolfinx.write_mesh(filename, mesh, time=0.0, backend=vtkhdf_backend)
    writes = [(q, 1.0), (q, 2.0), (u, 0.0), (u, 1.0), (late, 1.0)]
    for function, t in writes:
        function.x.array[:] = t
        write_point(filename, function, t, vtkhdf_backend)

    mesh_in = io4dolfinx.read_mesh(filename, comm, backend=vtkhdf_backend)
    for name, values in (("u", [0.0, 1.0, 1.0]), ("late", [1.0, 1.0, 1.0])):
        for t, value in zip([0.0, 1.0, 2.0], values):
            stored = io4dolfinx.read_point_data(
                filename, name, mesh_in, time=t, backend=vtkhdf_backend
            )
            np.testing.assert_array_equal(stored.x.array, value)


def test_time_tolerance(tmp_path, vtkhdf_backend):
    """Times are the same up to the precision of the stored time stamps (float64): round-off
    does not make a new time step, while close but distinct times stay apart."""
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path, root=0)
    filename = tmp_path / "tolerance.vtkhdf"
    mesh = create_unit_square(comm, 2, 2)
    V = functionspace(mesh, ("Lagrange", 1))
    u, v = Function(V, name="u"), Function(V, name="v")
    io4dolfinx.write_mesh(filename, mesh, time=0.0, backend=vtkhdf_backend)
    for function, t in ((u, 1e-9), (u, 0.1 + 0.2), (v, 0.3)):
        write_point(filename, function, t, vtkhdf_backend)
    stamps = io4dolfinx.read_timestamps(filename, comm, "v", backend=vtkhdf_backend)
    np.testing.assert_allclose(stamps, [0.0, 1e-9, 0.3], rtol=0, atol=1e-16)
    with pytest.raises(ValueError, match="has already been written at time"):
        write_point(filename, v, 0.1 + 0.2, vtkhdf_backend)


def test_only_geometry_changes(tmp_path, vtkhdf_backend):
    """The topology of a mesh is stored once, so a mesh written again under the same name
    must have the same number of points and cells. Only its geometry may change."""
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path, root=0)
    filename = tmp_path / "only_geometry.vtkhdf"
    mesh = create_unit_square(comm, 4, 3)
    io4dolfinx.write_mesh(filename, mesh, time=0.0, backend=vtkhdf_backend)
    contents = file_contents(filename)
    with pytest.raises(ValueError, match="Only the geometry of a mesh can change"):
        io4dolfinx.write_mesh(
            filename,
            create_unit_square(comm, 4, 4),
            time=1.0,
            mode=io4dolfinx.FileMode.append,
            backend=vtkhdf_backend,
        )
    assert_same_contents(file_contents(filename), contents)


def write_meshes(filename, backend):
    """A file with a square with cell tags and point data at t = 0, 1, 2, an interval
    with point data at t = 1, and a square added at t = 1."""
    comm = MPI.COMM_WORLD
    square = create_unit_square(comm, 4, 3)
    io4dolfinx.write_mesh(filename, square, time=0.0, backend=backend, mesh_name="square")
    cells = np.arange(square.topology.index_map(2).size_local, dtype=np.int32)
    tags = meshtags(square, 2, cells, cells + square.topology.index_map(2).local_range[0])
    io4dolfinx.write_meshtags(filename, square, tags, "cells", backend=backend, mesh_name="square")
    interval = create_unit_interval(comm, 5)
    io4dolfinx.write_mesh(
        filename,
        interval,
        time=0.0,
        mode=io4dolfinx.FileMode.append,
        backend=backend,
        mesh_name="interval",
    )
    u = Function(functionspace(square, ("Lagrange", 1)), name="u")
    for t in (0.0, 1.0, 2.0):
        u.x.array[:] = t
        write_point(filename, u, t, backend, mesh_name="square")
    v = Function(functionspace(interval, ("Lagrange", 1)), name="v")
    v.x.array[:] = 5.0
    write_point(filename, v, 1.0, backend, mesh_name="interval")
    io4dolfinx.write_mesh(
        filename,
        create_unit_square(comm, 2, 2),
        time=1.0,
        mode=io4dolfinx.FileMode.append,
        backend=backend,
        mesh_name="added",
    )


def test_meshes_share_time_steps(tmp_path, vtkhdf_backend):
    """VTK requires all blocks of a file, meshes and mesh tags, to have the same time
    steps: a new time step is added to every block, which holds its data from the step
    before, and a mesh added later gets the existing time steps."""
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path, root=0)
    filename = tmp_path / "shared_steps.vtkhdf"
    write_meshes(filename, vtkhdf_backend)

    # Every block, with the datasets it shares with its mesh (hard links), has one
    # entry per time step
    comm.barrier()
    with h5py.File(filename, "r") as h5file:
        # The root Steps link works around https://gitlab.kitware.com/vtk/vtk/-/issues/20246
        np.testing.assert_allclose(h5file["VTKHDF/Steps/Values"], [0.0, 1.0, 2.0])
        blocks = {name for name in h5file["VTKHDF"] if name not in ("Assembly", "Steps")}
        assert blocks == {"square", "square_cells", "interval", "added"}
        for name in blocks:
            block = h5file["VTKHDF"][name]
            steps = block["Steps"]
            assert steps.attrs["NSteps"] == 3
            np.testing.assert_allclose(steps["Values"], [0.0, 1.0, 2.0])
            per_step = [block[key] for key in ("NumberOfPoints", "NumberOfCells")]
            per_step += [block["NumberOfConnectivityIds"]]
            per_step += [
                steps[key]
                for key in (
                    "PartOffsets",
                    "NumberOfParts",
                    "PointOffsets",
                    "CellOffsets",
                    "ConnectivityIdOffsets",
                )
            ]
            for key in ("PointDataOffsets", "CellDataOffsets"):
                per_step += list(steps.get(key, {}).values())
            for dataset in per_step:
                assert dataset.shape[0] == 3, dataset.name
    comm.barrier()

    for t in (0.0, 1.0, 2.0):
        interval = io4dolfinx.read_mesh(
            filename, comm, time=t, backend=vtkhdf_backend, mesh_name="interval"
        )
        v = io4dolfinx.read_point_data(
            filename, "v", interval, time=t, backend=vtkhdf_backend, mesh_name="interval"
        )
        np.testing.assert_array_equal(v.x.array, 5.0)
        io4dolfinx.read_mesh(filename, comm, time=t, backend=vtkhdf_backend, mesh_name="added")
    square = io4dolfinx.read_mesh(
        filename, comm, time=2.0, backend=vtkhdf_backend, mesh_name="square"
    )
    tags = io4dolfinx.read_meshtags(
        filename, square, "cells", backend=vtkhdf_backend, mesh_name="square"
    )
    num_cells = square.topology.index_map(2).size_global
    assert comm.allreduce(len(tags.values), op=MPI.SUM) >= num_cells


def test_meshes_share_time_steps_in_vtk(tmp_path):
    """VTK reads the time steps and the data at each of them."""
    vtk = pytest.importorskip("vtk")
    from vtkmodules.util import numpy_support

    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path, root=0)
    filename = tmp_path / "shared_steps_vtk.vtkhdf"
    write_meshes(filename, "vtkhdf")
    if comm.rank == 0:
        # Select each step by time, as the pipeline does, and by index, each with its own
        # reader as a requested time overrides SetStep. VTK 9.7 needs the root Steps link
        # for the former, see https://gitlab.kitware.com/vtk/vtk/-/issues/20246
        for by_index in (False, True):
            reader = vtk.vtkHDFReader()
            reader.SetFileName(str(filename))
            reader.UpdateInformation()
            info = reader.GetOutputInformation(0)
            times = info.Get(vtk.vtkStreamingDemandDrivenPipeline.TIME_STEPS())
            np.testing.assert_allclose(times, [0.0, 1.0, 2.0])
            for step, t in enumerate(times):
                if by_index:
                    reader.SetStep(step)
                    reader.Update()
                else:
                    reader.UpdateTimeStep(t)
                blocks = {}
                iterator = reader.GetOutputDataObject(0).NewIterator()
                iterator.InitTraversal()
                while not iterator.IsDoneWithTraversal():
                    name = iterator.GetCurrentMetaData().Get(vtk.vtkCompositeDataSet.NAME())
                    blocks[name] = iterator.GetCurrentDataObject()
                    iterator.GoToNextItem()
                assert set(blocks) == {"square", "cells", "interval", "added"}
                u = numpy_support.vtk_to_numpy(blocks["square"].GetPointData().GetArray("u"))
                np.testing.assert_array_equal(u, t)
                v = numpy_support.vtk_to_numpy(blocks["interval"].GetPointData().GetArray("v"))
                np.testing.assert_array_equal(v, 5.0)


def test_append_to_file_without_record(tmp_path, vtkhdf_backend):
    """Files from older versions of io4dolfinx have no record of the steps at which the
    geometry and the arrays were written: a write is assumed at the first time step and
    wherever the offsets change."""
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path, root=0)
    filename = tmp_path / "without_record.vtkhdf"
    mesh = create_unit_square(comm, 2, 2)
    u = Function(functionspace(mesh, ("Lagrange", 1)), name="u")
    io4dolfinx.write_mesh(filename, mesh, time=0.0, backend=vtkhdf_backend)
    for t in (0.0, 1.0, 2.0):
        u.x.array[:] = t
        write_point(filename, u, t, vtkhdf_backend)
    comm.barrier()
    if comm.rank == 0:
        with h5py.File(filename, "a") as h5file:
            del h5file["io4dolfinx/mesh/Steps"]
            del h5file["io4dolfinx/mesh/PointData"]
    comm.barrier()

    with pytest.raises(ValueError, match="has already been written at time"):
        write_point(filename, u, 1.0, vtkhdf_backend)
    with pytest.raises(ValueError, match="has already been written at time"):
        io4dolfinx.write_mesh(
            filename, mesh, time=0.0, mode=io4dolfinx.FileMode.append, backend=vtkhdf_backend
        )
    mesh.geometry.x[:, 0] += 1.0
    io4dolfinx.write_mesh(
        filename, mesh, time=1.0, mode=io4dolfinx.FileMode.append, backend=vtkhdf_backend
    )
    u.x.array[:] = 3.0
    write_point(filename, u, 3.0, vtkhdf_backend)
    for t, (shift, value) in zip([0.0, 1.0, 2.0, 3.0], [(0, 0), (1, 1), (1, 2), (1, 3)]):
        mesh_in = io4dolfinx.read_mesh(filename, comm, time=t, backend=vtkhdf_backend)
        x_min = comm.allreduce(mesh_in.geometry.x[:, 0].min(), op=MPI.MIN)
        assert np.isclose(x_min, shift)
        u_in = io4dolfinx.read_point_data(filename, "u", mesh_in, time=t, backend=vtkhdf_backend)
        np.testing.assert_array_equal(u_in.x.array, value)


def write_vtk_temporal_triangles(filename, times):
    """Write a temporal triangle grid with VTK's own VTKHDF writer.

    The point array ``p`` and the cell array ``c`` change at every time step.
    """
    vtk = pytest.importorskip("vtk")
    from vtkmodules.util import numpy_support
    from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase

    if not hasattr(vtk, "vtkHDFWriter") or not hasattr(vtk.vtkHDFWriter, "SetWriteAllTimeSteps"):
        pytest.skip("VTK without a temporal VTKHDF writer")

    class TemporalTriangles(VTKPythonAlgorithmBase):
        def __init__(self):
            super().__init__(nInputPorts=0, nOutputPorts=1, outputType="vtkUnstructuredGrid")

        def RequestInformation(self, request, in_info, out_info):
            info = out_info.GetInformationObject(0)
            info.Set(vtk.vtkStreamingDemandDrivenPipeline.TIME_STEPS(), times, len(times))
            info.Set(vtk.vtkStreamingDemandDrivenPipeline.TIME_RANGE(), [times[0], times[-1]], 2)
            return 1

        def RequestData(self, request, in_info, out_info):
            t = out_info.GetInformationObject(0).Get(
                vtk.vtkStreamingDemandDrivenPipeline.UPDATE_TIME_STEP()
            )
            x = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
            grid = vtk.vtkUnstructuredGrid.GetData(out_info)
            points = vtk.vtkPoints()
            points.SetData(numpy_support.numpy_to_vtk(x, deep=True))
            grid.SetPoints(points)
            grid.Allocate(2)
            for cell in ([0, 1, 2], [0, 2, 3]):
                grid.InsertNextCell(vtk.VTK_TRIANGLE, 3, cell)
            for name, values, data in (
                ("p", x[:, 0] + t, grid.GetPointData()),
                ("c", np.array([t, 2 * t]), grid.GetCellData()),
            ):
                array = numpy_support.numpy_to_vtk(values, deep=True)
                array.SetName(name)
                data.AddArray(array)
            return 1

    source = TemporalTriangles()
    writer = vtk.vtkHDFWriter()
    writer.SetInputConnection(source.GetOutputPort())
    writer.SetFileName(str(filename))
    writer.SetWriteAllTimeSteps(True)
    writer.Write()


def test_read_timestamps_vtk_written(tmp_path):
    """Time stamps of a temporal file written by VTK, following the VTKHDF specification."""
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path, root=0)
    filename = tmp_path / "vtk_temporal.vtkhdf"
    times = [0.0, 0.25, 0.5, 1.0]
    if comm.rank == 0:
        write_vtk_temporal_triangles(filename, times)
    comm.barrier()
    for name in ("p", "c"):
        np.testing.assert_allclose(
            io4dolfinx.read_timestamps(filename, comm, name, backend="vtkhdf"), times
        )


def test_read_timestamps_dolfinx_written(tmp_path):
    """Time stamps of a temporal file written by DOLFINx, whose geometry is static."""
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path, root=0)
    filename = tmp_path / "dolfinx_temporal.vtkhdf"
    mesh = create_unit_square(comm, 3, 2)
    write_mesh(str(filename), mesh)
    num_nodes_local = mesh.geometry.index_map().size_local
    times = [0.1, 0.4, 0.7]
    for t in times:
        write_point_data(str(filename), mesh, np.full(num_nodes_local, t), t)
    comm.barrier()
    np.testing.assert_allclose(
        io4dolfinx.read_timestamps(filename, comm, "u", backend="vtkhdf"), times
    )
