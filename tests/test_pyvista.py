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
    # A file with float32 points and data, read into a float64 mesh (backend_args dtype): the point
    # and cell data must come back as functions on that mesh, in its precision
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


def f(x):
    return x[..., 0] + 2 * x[..., 1] + 3 * x[..., 2]


def structured_grid(kind: str):
    """The box [0, 2]x[0, 1]x[0, 1], or the rectangle [0, 2]x[0, 1] for ``"image2D"``, as a
    pyvista grid of the given kind, with the point data p and cell data c of ``f``."""
    if kind == "image":
        grid = pyvista.ImageData(dimensions=(5, 3, 3), spacing=(0.5, 0.5, 0.5))
    elif kind == "image2D":
        grid = pyvista.ImageData(dimensions=(5, 3, 1), spacing=(0.5, 0.5, 1))
    elif kind == "rectilinear":
        grid = pyvista.RectilinearGrid([0, 0.3, 1, 2], [0, 0.6, 1], [0, 0.5, 1])
    elif kind == "structured":
        x, y, z = np.meshgrid([0, 0.3, 1, 2], [0, 0.6, 1], [0, 0.5, 1], indexing="ij")
        grid = pyvista.StructuredGrid(x, y, z)
    else:
        raise ValueError(kind)
    grid.point_data["p"] = f(np.asarray(grid.points))
    grid.cell_data["c"] = f(np.asarray(grid.cell_centers().points))
    return grid


@pytest.mark.parametrize(
    "kind, suffix", [("image", ".vti"), ("rectilinear", ".vtr"), ("structured", ".vts")]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_structured_grids(tmp_path, kind, suffix, dtype):
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path, root=0)
    filename = tmp_path / f"grid{suffix}"
    if comm.rank == 0:
        structured_grid(kind).save(filename)
    comm.barrier()

    args = {"dtype": dtype}
    mesh = io4dolfinx.read_mesh(filename, comm, backend="pyvista", backend_args=args)
    assert mesh.topology.cell_type == dolfinx.mesh.CellType.hexahedron
    assert mesh.geometry.x.dtype == dtype

    def integral(expr, measure):
        form = dolfinx.fem.form(expr * measure(domain=mesh), dtype=dtype)
        return comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)

    assert np.isclose(integral(1, ufl.dx), 2)
    assert np.isclose(integral(1, ufl.ds), 10)
    # f is linear: P1 interpolation and cell averages integrate it exactly
    p = io4dolfinx.read_point_data(filename, "p", mesh, backend="pyvista", backend_args=args)
    assert np.isclose(integral(p, ufl.dx), 7)
    c = io4dolfinx.read_cell_data(filename, "c", mesh, backend="pyvista", backend_args=args)
    assert np.isclose(integral(c, ufl.dx), 7)


@pytest.mark.parametrize("gdim", [2, 3])
def test_two_dimensional_image(tmp_path, gdim):
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path, root=0)
    filename = tmp_path / "image.vti"
    if comm.rank == 0:
        structured_grid("image2D").save(filename)
    comm.barrier()

    mesh = io4dolfinx.read_mesh(filename, comm, backend="pyvista", backend_args={"gdim": gdim})
    assert mesh.topology.cell_type == dolfinx.mesh.CellType.quadrilateral
    assert mesh.geometry.dim == gdim
    area = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ufl.dx(domain=mesh)))
    assert np.isclose(comm.allreduce(area, op=MPI.SUM), 2)
    p = io4dolfinx.read_point_data(filename, "p", mesh, backend="pyvista")
    integral = dolfinx.fem.assemble_scalar(dolfinx.fem.form(p * ufl.dx))
    assert np.isclose(comm.allreduce(integral, op=MPI.SUM), 4)


def test_gdim_must_drop_zero_coordinates(tmp_path):
    comm = MPI.COMM_WORLD
    tmp_path = comm.bcast(tmp_path, root=0)
    filename = tmp_path / "image.vti"
    if comm.rank == 0:
        structured_grid("image").save(filename)
    comm.barrier()
    with pytest.raises(ValueError, match="gdim=2"):
        io4dolfinx.read_mesh(filename, comm, backend="pyvista", backend_args={"gdim": 2})
