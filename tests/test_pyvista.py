from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import io4dolfinx

pyvista = pytest.importorskip("pyvista")


def test_read_mesh_and_cell_data(tmp_path):
    tmp_path = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    filename = tmp_path / "grid.vtu"
    grid = pyvista.examples.load_hexbeam()
    if MPI.COMM_WORLD.rank == 0:
        grid.save(filename)
    MPI.COMM_WORLD.barrier()

    mesh = io4dolfinx.read_mesh(filename, MPI.COMM_WORLD, backend="pyvista")

    vol = dolfinx.fem.form(1 * ufl.dx(domain=mesh))
    surf = dolfinx.fem.form(1 * ufl.ds(domain=mesh))

    vol_ref = 5 * 1 * 1
    surf_ref = 5 * 4 + 2

    vol_glob = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(vol), op=MPI.SUM)
    surf_glob = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(surf), op=MPI.SUM)
    assert np.isclose(vol_glob, vol_ref)
    assert np.isclose(surf_glob, surf_ref)

    names = io4dolfinx.read_function_names(filename, MPI.COMM_WORLD, backend="pyvista")
    for name in grid.cell_data.keys():
        assert name in names
    for name in grid.point_data.keys():
        assert name in names

    for name in names:
        if name in grid.cell_data.keys():
            cd = io4dolfinx.read_cell_data(filename, name, mesh, backend="pyvista")
            oci = mesh.topology.original_cell_index
            np.testing.assert_allclose(cd.x.array[:], grid.cell_data[name][oci])
        elif name in grid.point_data.keys():
            pd = io4dolfinx.read_point_data(filename, name, mesh, backend="pyvista")
            igi = mesh.geometry.input_global_indices
            np.testing.assert_allclose(pd.x.array[:], grid.point_data[name][igi])
        else:
            raise RuntimeError(f"Could not find {name} in grid")


def test_single_precision_data_on_double_precision_mesh(tmp_path):
    # A file with float32 points and data, read into a float64 mesh (backend_args dtype): the point and
    # cell data must come back as functions on that mesh, in its precision
    tmp_path = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    filename = tmp_path / "grid32.vtu"
    grid = pyvista.examples.load_hexbeam().triangulate()
    grid.points = grid.points.astype(np.float32)
    grid.point_data["height"] = grid.points[:, 2].astype(np.float32)
    grid.cell_data["label"] = np.arange(grid.n_cells, dtype=np.float32) % 3
    if MPI.COMM_WORLD.rank == 0:
        grid.save(filename)
    MPI.COMM_WORLD.barrier()

    mesh = io4dolfinx.read_mesh(
        filename, MPI.COMM_WORLD, backend="pyvista", backend_args={"dtype": np.float64}
    )
    assert mesh.geometry.x.dtype == np.float64

    height = io4dolfinx.read_point_data(filename, "height", mesh, backend="pyvista")
    assert height.x.array.dtype == np.float64
    x = height.function_space.tabulate_dof_coordinates()
    assert np.allclose(height.x.array, x[:, 2], atol=1e-6)

    label = io4dolfinx.read_cell_data(filename, "label", mesh, backend="pyvista")
    assert label.x.array.dtype == np.float64
    assert set(np.unique(label.x.array)) <= {0.0, 1.0, 2.0}
