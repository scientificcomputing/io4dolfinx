from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import adios4dolfinx

pyvista = pytest.importorskip("pyvista")


def test_read_mesh_and_cell_data(tmp_path):
    tmp_path = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    filename = tmp_path / "grid.vtu"
    grid = pyvista.examples.load_hexbeam()
    if MPI.COMM_WORLD.rank == 0:
        grid.save(filename)
    MPI.COMM_WORLD.barrier()

    mesh = adios4dolfinx.read_mesh(filename, MPI.COMM_WORLD, backend="pyvista")

    vol = dolfinx.fem.form(1 * ufl.dx(domain=mesh))
    surf = dolfinx.fem.form(1 * ufl.ds(domain=mesh))

    vol_ref = 5 * 1 * 1
    surf_ref = 5 * 4 + 2

    vol_glob = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(vol), op=MPI.SUM)
    surf_glob = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(surf), op=MPI.SUM)
    assert np.isclose(vol_glob, vol_ref)
    assert np.isclose(surf_glob, surf_ref)

    names = adios4dolfinx.read_function_names(filename, MPI.COMM_WORLD, backend="pyvista")
    for name in grid.cell_data.keys():
        assert name in names
    for name in grid.point_data.keys():
        assert name in names

    for name in names:
        if name in grid.cell_data.keys():
            cd = adios4dolfinx.read_cell_data(filename, name, mesh, backend="pyvista")
            oci = mesh.topology.original_cell_index
            np.testing.assert_allclose(cd.x.array[:], grid.cell_data[name][oci])
        elif name in grid.point_data.keys():
            pd = adios4dolfinx.read_point_data(filename, name, mesh, backend="pyvista")
            igi = mesh.geometry.input_global_indices
            np.testing.assert_allclose(pd.x.array[:], grid.point_data[name][igi])
        else:
            raise RuntimeError(f"Could not find {name} in grid")
