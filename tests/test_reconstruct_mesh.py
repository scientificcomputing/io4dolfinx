from typing import cast

from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np
import numpy.typing as npt
import pytest
import ufl

from io4dolfinx import reconstruct_mesh


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize(
    "cell_type", [dolfinx.mesh.CellType.tetrahedron, dolfinx.mesh.CellType.hexahedron]
)
def test_reconstruct(cell_type: dolfinx.mesh.CellType, degree: int, dtype: npt.DTypeLike):
    def f(x):
        return (0.1 * x[0] ** degree + x[1] ** (degree - 1), x[0], 0.1 * x[2] ** degree)

    el = basix.ufl.element(
        "Lagrange", dolfinx.mesh.to_string(cell_type), degree, shape=(3,), dtype=dtype
    )
    mesh = dolfinx.mesh.create_unit_cube(
        MPI.COMM_WORLD, 10, 10, 10, dtype=dtype, cell_type=cell_type
    )
    V = dolfinx.fem.functionspace(mesh, el)
    u = dolfinx.fem.Function(V, dtype=dtype)
    u.interpolate(f)

    new_mesh = reconstruct_mesh(mesh, degree)
    V2 = dolfinx.fem.functionspace(new_mesh, el)
    v = dolfinx.fem.Function(V2, dtype=dtype)
    cell_map = mesh.topology.index_map(mesh.topology.dim)
    num_cells_local = cell_map.size_local + cell_map.num_ghosts
    cells = np.arange(num_cells_local, dtype=np.int32)
    v.interpolate(u, cells, cells)

    assert new_mesh.geometry.cmaps[0].degree == degree

    f_ex = ufl.as_vector(f(ufl.SpatialCoordinate(new_mesh)))

    one = np.array(1.0, dtype=dtype)
    local_vol = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(dolfinx.fem.Constant(new_mesh, one) * ufl.dx, dtype=dtype)
    )
    vol = new_mesh.comm.allreduce(local_vol, op=MPI.SUM)
    float_dtype = cast(type[np.floating], dtype)
    tol = 4e3 * degree * np.finfo(float_dtype).eps
    assert np.isclose(vol, 1, atol=tol, rtol=tol)

    local_surf = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(dolfinx.fem.Constant(new_mesh, one) * ufl.ds, dtype=dtype)
    )
    surf = new_mesh.comm.allreduce(local_surf, op=MPI.SUM)
    assert np.isclose(surf, 6, atol=tol, rtol=tol)

    diff = dolfinx.fem.form(ufl.inner(v - f_ex, v - f_ex) * ufl.dx, dtype=dtype)
    local_diff = dolfinx.fem.assemble_scalar(diff)
    diff_glob = new_mesh.comm.allreduce(local_diff, op=MPI.SUM)
    assert np.sqrt(np.isclose(diff_glob, 0.0, atol=tol, rtol=tol))
