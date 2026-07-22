from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

from io4dolfinx import reconstruct_mesh


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("R", [0.1, 1, 10])
def test_curve_mesh(degree, dtype, R):
    N = 8
    mesh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [[-1, -1], [1, 1]],
        [N, N],
        diagonal=dolfinx.mesh.DiagonalType.crossed,
        dtype=dtype,
    )
    org_area = dolfinx.fem.form(1 * ufl.dx(domain=mesh), dtype=dtype)

    curved_mesh = reconstruct_mesh(mesh, degree)

    def transform(x):
        u = R * x[0] * np.sqrt(1 - (x[1] ** 2 / (2)))
        v = R * x[1] * np.sqrt(1 - (x[0] ** 2 / (2)))
        return np.asarray([u, v])

    curved_mesh.geometry.x[:, : curved_mesh.geometry.dim] = transform(curved_mesh.geometry.x.T).T

    area = dolfinx.fem.form(1 * ufl.dx(domain=curved_mesh), dtype=dtype)
    circumference = dolfinx.fem.form(1 * ufl.ds(domain=curved_mesh), dtype=dtype)

    computed_area = curved_mesh.comm.allreduce(dolfinx.fem.assemble_scalar(area), op=MPI.SUM)
    computed_circumference = curved_mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(circumference), op=MPI.SUM
    )

    tol = 15 * np.finfo(dtype).eps
    assert np.isclose(computed_area, np.pi * R**2, atol=tol)
    assert np.isclose(computed_circumference, 2 * np.pi * R, atol=tol)

    linear_mesh = reconstruct_mesh(curved_mesh, 1)
    linear_area = dolfinx.fem.form(1 * ufl.dx(domain=linear_mesh), dtype=dtype)

    recovered_area = linear_mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(linear_area), op=MPI.SUM
    )

    # Curve original mesh
    mesh.geometry.x[:, : mesh.geometry.dim] = transform(mesh.geometry.x.T).T
    ref_area = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(org_area), op=MPI.SUM)
    assert np.isclose(recovered_area, ref_area, atol=tol)
