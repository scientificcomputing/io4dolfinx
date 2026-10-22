from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import io4dolfinx


def test_xdmf_mesh(tmp_path):
    tmp_path = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    tmp_file = tmp_path / "mesh.xdmf"
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, tmp_file, "w") as xdmf:
        xdmf.write_mesh(mesh)

    MPI.COMM_WORLD.barrier()
    in_grid = io4dolfinx.read_mesh(tmp_file, MPI.COMM_WORLD, backend="xdmf")

    assert mesh.topology.dim == in_grid.topology.dim
    assert mesh.geometry.dim == in_grid.geometry.dim
    for i in range(mesh.topology.dim):
        mesh.topology.create_entities(i)
        o_map = mesh.topology.index_map(i)
        in_grid.topology.create_entities(i)
        map = in_grid.topology.index_map(i)
        assert o_map.size_global == map.size_global

    assert mesh.geometry.index_map().size_global == in_grid.geometry.index_map().size_global
    org_area = mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ufl.dx(domain=mesh))), op=MPI.SUM
    )
    area = mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ufl.dx(domain=in_grid))), op=MPI.SUM
    )
    assert np.isclose(area, org_area)


def test_xdmf_function(tmp_path):
    tmp_path = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 10)

    def f(x):
        return (x[0], x[1], -2 * x[1], 3 * x[0])

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (4,)))
    u = dolfinx.fem.Function(V, name="u")
    u.interpolate(f)

    tmp_file = tmp_path / "function.xdmf"
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, tmp_file, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u)

    MPI.COMM_WORLD.barrier()

    in_grid = io4dolfinx.read_mesh(tmp_file, MPI.COMM_WORLD, backend="xdmf")
    names = io4dolfinx.read_function_names(
        tmp_file, MPI.COMM_WORLD, backend_args={}, backend="xdmf"
    )
    assert len(names) == 1
    assert names[0] == "u"

    u = io4dolfinx.read_point_data(tmp_file, "u", in_grid, backend="xdmf")
    u_ref = dolfinx.fem.Function(u.function_space)
    u_ref.interpolate(f)
    eps = np.finfo(mesh.geometry.x.dtype).eps
    np.testing.assert_allclose(u.x.array, u_ref.x.array, atol=eps)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_xdmf_point_data_dtype_and_time(tmp_path, dtype):
    """Complex functions are stored as ``real_u``/``imag_u`` and read back combined."""
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path, root=0)
    real_dtype = np.real(np.zeros(1, dtype=dtype)).dtype
    mesh = dolfinx.mesh.create_unit_square(comm, 8, 10, dtype=real_dtype)
    # Four components, as DOLFINx pads two- to three-component vectors in XDMF
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (4,)))
    u = dolfinx.fem.Function(V, name="u", dtype=dtype)

    def f(x, t):
        values = np.vstack([x[0] + t, -2 * x[1], x[0] * x[1], 3 * t + 0 * x[0]]).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            values[0] += 1j * (x[1] - t)
            values[3] += 3j * x[0]
        return values

    tmp_file = tmp_path / "function.xdmf"
    times = [0.0, 0.5]
    with dolfinx.io.XDMFFile(comm, tmp_file, "w") as xdmf:
        xdmf.write_mesh(mesh)
        for t in times:
            u.interpolate(lambda x: f(x, t))
            xdmf.write_function(u, t)

    in_grid = io4dolfinx.read_mesh(tmp_file, comm, backend="xdmf")
    eps = np.finfo(real_dtype).eps
    for t in times:
        u_in = io4dolfinx.read_point_data(tmp_file, "u", in_grid, time=t, backend="xdmf")
        assert u_in.x.array.dtype == dtype
        u_ref = dolfinx.fem.Function(u_in.function_space, dtype=dtype)
        u_ref.interpolate(lambda x: f(x, t))
        np.testing.assert_allclose(u_in.x.array, u_ref.x.array, atol=10 * eps)

    # Without a time stamp the first time step is read.
    u_in = io4dolfinx.read_point_data(tmp_file, "u", in_grid, backend="xdmf")
    assert u_in.x.array.dtype == dtype

    with pytest.raises(RuntimeError, match="not found"):
        io4dolfinx.read_point_data(tmp_file, "u", in_grid, time=1.0, backend="xdmf")


def test_xdmf_point_data_incomplete_complex(tmp_path):
    """A missing complex part, or no matching attribute at all, is reported."""
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path, root=0)
    mesh = dolfinx.mesh.create_unit_square(comm, 4, 4)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (4,)))
    u = dolfinx.fem.Function(V, name="u", dtype=np.complex128)

    tmp_file = tmp_path / "function.xdmf"
    with dolfinx.io.XDMFFile(comm, tmp_file, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u)
    in_grid = io4dolfinx.read_mesh(tmp_file, comm, backend="xdmf")
    comm.barrier()

    xml = tmp_file.read_text()
    if comm.rank == 0:
        tmp_file.write_text(xml.replace('Name="imag_u"', 'Name="imag_v"'))
    comm.barrier()
    with pytest.raises(RuntimeError, match="found 'real_u' but not 'imag_u'"):
        io4dolfinx.read_point_data(tmp_file, "u", in_grid, backend="xdmf")
    comm.barrier()

    if comm.rank == 0:
        tmp_file.write_text(xml.replace('_u"', '_v"'))
    comm.barrier()
    with pytest.raises(RuntimeError, match="expected attribute 'u'"):
        io4dolfinx.read_point_data(tmp_file, "u", in_grid, backend="xdmf")


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("shape", [(), (4,)])
def test_xdmf_cell_data_dtype_and_time(tmp_path, dtype, shape):
    """DG-0 functions written by DOLFINx, real or split into ``real_q``/``imag_q``."""
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path, root=0)
    mesh = dolfinx.mesh.create_unit_square(comm, 6, 5)
    q = dolfinx.fem.Function(
        dolfinx.fem.functionspace(mesh, ("DG", 0, shape)), name="q", dtype=dtype
    )

    def f(x, t):
        values = np.vstack([x[0] + t, -2 * x[1], x[0] * x[1], 3 * t + 0 * x[0]]).astype(dtype)
        if np.issubdtype(dtype, np.complexfloating):
            values[0] += 1j * (x[1] - t)
            values[3] += 3j * x[0]
        return values[: max(1, int(np.prod(shape)))]

    tmp_file = tmp_path / "cells.xdmf"
    times = [0.0, 0.5]
    with dolfinx.io.XDMFFile(comm, tmp_file, "w") as xdmf:
        xdmf.write_mesh(mesh)
        for t in times:
            q.interpolate(lambda x: f(x, t))
            xdmf.write_function(q, t)

    in_grid = io4dolfinx.read_mesh(tmp_file, comm, backend="xdmf")
    eps = np.finfo(in_grid.geometry.x.dtype).eps
    for t in times:
        q_in = io4dolfinx.read_cell_data(tmp_file, "q", in_grid, time=t, backend="xdmf")
        assert q_in.x.array.dtype == dtype
        q_ref = dolfinx.fem.Function(q_in.function_space, dtype=dtype)
        q_ref.interpolate(lambda x: f(x, t))
        np.testing.assert_allclose(q_in.x.array, q_ref.x.array, atol=10 * eps)

    with pytest.raises(RuntimeError, match=r"q\(t=1.0\) not found"):
        io4dolfinx.read_cell_data(tmp_file, "q", in_grid, time=1.0, backend="xdmf")


def test_xdmf_cell_data_rejects_point_data(tmp_path):
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path, root=0)
    mesh = dolfinx.mesh.create_unit_square(comm, 2, 2)
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("Lagrange", 1)), name="u")
    tmp_file = tmp_path / "points.xdmf"
    with dolfinx.io.XDMFFile(comm, tmp_file, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u)
    in_grid = io4dolfinx.read_mesh(tmp_file, comm, backend="xdmf")
    with pytest.raises(ValueError, match="Center='Node'"):
        io4dolfinx.read_cell_data(tmp_file, "u", in_grid, backend="xdmf")
