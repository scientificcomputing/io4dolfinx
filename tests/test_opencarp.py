from pathlib import Path

from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np
import pytest
import ufl

import io4dolfinx
from io4dolfinx.backends.opencarp import write_opencarp

markers = {"left": (1, 2), "right": (2, 2)}


def fiber_expr(x):
    return np.vstack([2 + x[1], x[0], np.zeros_like(x[0])])


def sheet_expr(x):
    return np.vstack([np.zeros_like(x[0]), np.zeros_like(x[0]), 3 + x[2]])


def normalised(values):
    return values / np.linalg.norm(values, axis=0)


def create_problem(comm: MPI.Intracomm):
    """Unit cube with region 1 (x < 0.4) and 2, facet markers on x=0 and x=1, and
    linear fiber and sheet fields (exact in P-1)."""
    mesh = dolfinx.mesh.create_unit_cube(comm, 3, 3, 3)
    tdim = mesh.topology.dim
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

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (3,)))
    fiber = dolfinx.fem.Function(V, name="f0")
    fiber.interpolate(fiber_expr)
    sheet = dolfinx.fem.Function(V, name="s0")
    sheet.interpolate(sheet_expr)
    return mesh, cell_tags, facet_tags, fiber, sheet


def write_problem(filename: Path, comm: MPI.Intracomm, backend_args=None):
    mesh, cell_tags, facet_tags, fiber, sheet = create_problem(comm)
    write_opencarp(
        filename,
        mesh,
        scale=1000.0,
        cell_tags=cell_tags,
        facet_tags=facet_tags,
        fiber=fiber,
        sheet=sheet,
        markers=markers,
        backend_args=backend_args,
    )


def read_lines(path: Path) -> list[list[str]]:
    return [line.split() for line in path.read_text().splitlines()]


def test_write_content(shared_tmp_path):
    filename = shared_tmp_path / "cube"
    write_problem(filename, MPI.COMM_WORLD)
    if MPI.COMM_WORLD.rank != 0:
        return

    pts = read_lines(filename.with_suffix(".pts"))
    assert int(pts[0][0]) == len(pts) - 1 == 64
    x = np.array(pts[1:], dtype=np.float64)
    # Scaled from the unit cube to µm
    assert np.allclose(x.min(axis=0), 0) and np.allclose(x.max(axis=0), 1000)

    elem = read_lines(filename.with_suffix(".elem"))
    assert int(elem[0][0]) == len(elem) - 1 == 6 * 27
    assert {row[0] for row in elem[1:]} == {"Tt"}
    cells = np.array([row[1:5] for row in elem[1:]], dtype=np.int64)
    tags = np.array([row[5] for row in elem[1:]], dtype=np.int32)
    assert cells.min() == 0 and cells.max() == 63

    # All elements positively oriented, filling the cube
    p = x[cells]
    volumes = np.einsum(
        "ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]
    )
    assert np.all(volumes > 0)
    assert np.isclose(volumes.sum() / 6, 1000.0**3)

    # Region tags follow the midpoints
    midpoints = p.mean(axis=1)
    np.testing.assert_array_equal(tags, np.where(midpoints[:, 0] < 400, 1, 2))

    # One unit fiber and sheet per element, in element order
    lon = read_lines(filename.with_suffix(".lon"))
    assert lon[0] == ["2"]
    directions = np.array(lon[1:], dtype=np.float64)
    assert directions.shape == (len(cells), 6)
    np.testing.assert_allclose(directions[:, :3], normalised(fiber_expr(midpoints.T / 1000)).T)
    np.testing.assert_allclose(directions[:, 3:], normalised(sheet_expr(midpoints.T / 1000)).T)

    for name, plane in (("left", 0.0), ("right", 1000.0)):
        vtx = read_lines(filename.parent / f"cube_{name}.vtx")
        assert vtx[1] == ["extra"]
        nodes = np.array(vtx[2:], dtype=np.int64).reshape(-1)
        assert int(vtx[0][0]) == len(nodes) == 16
        assert np.allclose(x[nodes, 0], plane)

        surf = read_lines(filename.parent / f"cube_{name}.surf")
        assert int(surf[0][0]) == len(surf) - 1 == 2 * 9
        assert {row[0] for row in surf[1:]} == {"Tr"}
        triangles = np.array([row[1:] for row in surf[1:]], dtype=np.int64)
        np.testing.assert_array_equal(np.unique(triangles), np.sort(nodes))
        # Normals point out of the cube
        q = x[triangles]
        normals = np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0])
        assert np.all(np.sign(normals[:, 0]) == (-1 if plane == 0 else 1))


def test_files_independent_of_partitioning(shared_tmp_path):
    parallel = shared_tmp_path / "parallel" / "cube"
    serial = shared_tmp_path / "serial" / "cube"
    if MPI.COMM_WORLD.rank == 0:
        parallel.parent.mkdir()
        serial.parent.mkdir()
    MPI.COMM_WORLD.Barrier()
    write_problem(parallel, MPI.COMM_WORLD)
    if MPI.COMM_WORLD.rank == 0:
        write_problem(serial, MPI.COMM_SELF)
        files = sorted(p.name for p in serial.parent.iterdir())
        assert files == sorted(p.name for p in parallel.parent.iterdir())
        assert len(files) == 7
        for name in files:
            assert (serial.parent / name).read_bytes() == (parallel.parent / name).read_bytes()


def test_read_write(shared_tmp_path):
    write_problem(shared_tmp_path / "cube", MPI.COMM_WORLD)
    filename = shared_tmp_path / "cube.elem"
    comm = MPI.COMM_WORLD
    backend_args = {"scale": 1000.0, "markers": markers}

    mesh = io4dolfinx.read_mesh(filename, comm, backend="opencarp", backend_args=backend_args)
    tdim = mesh.topology.dim
    assert mesh.topology.index_map(tdim).size_global == 6 * 27
    assert mesh.geometry.index_map().size_global == 64
    volume = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ufl.dx(domain=mesh)))
    assert np.isclose(comm.allreduce(volume, op=MPI.SUM), 1.0)

    cell_tags = io4dolfinx.read_meshtags(filename, mesh, "cell", backend="opencarp")
    midpoints = dolfinx.mesh.compute_midpoints(mesh, tdim, cell_tags.indices)
    np.testing.assert_array_equal(cell_tags.values, np.where(midpoints[:, 0] < 0.4, 1, 2))
    num_owned = mesh.topology.index_map(tdim).size_local
    assert comm.allreduce(np.count_nonzero(cell_tags.indices < num_owned)) == 6 * 27

    facet_tags = io4dolfinx.read_meshtags(
        filename, mesh, "facet", backend="opencarp", backend_args=backend_args
    )
    midpoints = dolfinx.mesh.compute_midpoints(mesh, tdim - 1, facet_tags.indices)
    np.testing.assert_allclose(midpoints[:, 0], np.where(facet_tags.values == 1, 0.0, 1.0))
    num_owned = mesh.topology.index_map(tdim - 1).size_local
    assert comm.allreduce(np.count_nonzero(facet_tags.indices < num_owned)) == 2 * 18

    right = io4dolfinx.read_meshtags(
        filename, mesh, "right", backend="opencarp", backend_args=backend_args
    )
    assert right.dim == tdim - 1
    assert np.all(right.values == 2)
    assert np.allclose(dolfinx.mesh.compute_midpoints(mesh, tdim - 1, right.indices)[:, 0], 1)

    assert io4dolfinx.read_function_names(filename, comm, backend="opencarp") == ["fiber", "sheet"]
    for name, expr in (("fiber", fiber_expr), ("sheet", sheet_expr)):
        u = io4dolfinx.read_cell_data(filename, name, mesh, backend="opencarp")
        num_cells = mesh.topology.index_map(tdim).size_local
        cells = np.arange(num_cells, dtype=np.int32)
        midpoints = dolfinx.mesh.compute_midpoints(mesh, tdim, cells)
        values = u.x.array.reshape(-1, 3)[u.function_space.dofmap.list[cells, 0]]
        np.testing.assert_allclose(values, normalised(expr(midpoints.T)).T)


def test_read_vertex_markers(shared_tmp_path):
    """Without ``.surf`` files a marker is read from its ``.vtx`` file, as vertex tags."""
    write_problem(shared_tmp_path / "cube", MPI.COMM_WORLD, backend_args={"surf": False})
    filename = shared_tmp_path / "cube.elem"
    assert not (shared_tmp_path / "cube_left.surf").exists()

    mesh = io4dolfinx.read_mesh(
        filename, MPI.COMM_WORLD, backend="opencarp", backend_args={"scale": 1000.0}
    )
    left = io4dolfinx.read_meshtags(
        filename, mesh, "left", backend="opencarp", backend_args={"markers": markers}
    )
    assert left.dim == 0
    assert np.all(left.values == 1)
    assert np.allclose(dolfinx.mesh.compute_midpoints(mesh, 0, left.indices)[:, 0], 0)
    num_owned = mesh.topology.index_map(0).size_local
    assert MPI.COMM_WORLD.allreduce(np.count_nonzero(left.indices < num_owned)) == 16


def test_fiber_only_and_unnamed_markers(shared_tmp_path):
    """Markers without names are written and read back by value."""
    filename = shared_tmp_path / "cube"
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
    filename = shared_tmp_path / "cube"
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

    hex_mesh = dolfinx.mesh.create_unit_cube(
        comm, 2, 2, 2, cell_type=dolfinx.mesh.CellType.hexahedron
    )
    with pytest.raises(NotImplementedError, match="tetrahedral"):
        write_opencarp(filename, hex_mesh, scale=1.0)

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
