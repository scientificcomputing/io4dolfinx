from pathlib import Path
from types import SimpleNamespace

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

import io4dolfinx
from io4dolfinx import checkpointing
from io4dolfinx.original_checkpoint import create_original_mesh_data


def f(x):
    return np.vstack([x[0] + 2 * x[1], 3 * x[2] - x[0], np.ones_like(x[0])])


@pytest.fixture
def captured(monkeypatch):
    """Capture the data passed to the backend, keyed by type."""
    data = {}

    def write_data(filename, array_data, **kwargs):
        data[array_data.type] = array_data

    monkeypatch.setattr(
        checkpointing, "get_backend", lambda backend=None: SimpleNamespace(write_data=write_data)
    )
    return data


@pytest.mark.parametrize(
    "cell_type", [dolfinx.mesh.CellType.tetrahedron, dolfinx.mesh.CellType.hexahedron]
)
def test_data_on_input_mesh(captured, cell_type):
    """Point and cell data written with ``on_input_mesh`` follow the nodes and cells of
    :func:`io4dolfinx.write_mesh` with ``on_input_mesh=True``."""
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 4, 3, 5, cell_type=cell_type)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (3,)))
    u = dolfinx.fem.Function(V)
    u.interpolate(f)

    for writer in (io4dolfinx.write_point_data, io4dolfinx.write_cell_data):
        writer("unused", u, 0.0, io4dolfinx.FileMode.write, None, "any", on_input_mesh=True)
    mesh_data = create_original_mesh_data(mesh)

    point_data = captured["Point"]
    assert tuple(point_data.local_range) == tuple(mesh_data.local_geometry_pos)
    np.testing.assert_allclose(point_data.values, f(mesh_data.local_geometry.T).T, atol=1e-12)

    # P-1 interpolated into DG-0 is the mean of the vertex values for these cells
    cell_data = captured["Cell"]
    assert tuple(cell_data.local_range) == tuple(mesh_data.local_topology_pos)
    x = np.vstack(MPI.COMM_WORLD.allgather(mesh_data.local_geometry))
    midpoints = x[mesh_data.local_topology].mean(axis=1)
    np.testing.assert_allclose(cell_data.values, f(midpoints.T).T, atol=1e-12)


@pytest.mark.parametrize("writer", [io4dolfinx.write_point_data, io4dolfinx.write_cell_data])
def test_submesh_has_no_input_order(captured, writer):
    """A submesh has no input cell indices, and keeps the input node indices of its parent
    mesh, which do not number its own nodes, so writing in the input ordering raises a
    ValueError on all ranks, whatever DOLFINx raises underneath."""
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    cells = dolfinx.mesh.locate_entities(mesh, 2, lambda x: x[0] <= 0.5 + 1e-12)
    submesh = dolfinx.mesh.create_submesh(mesh, 2, cells)[0]
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(submesh, ("Lagrange", 1)))
    with pytest.raises(ValueError, match="cannot be written in the input ordering"):
        writer("unused", u, 0.0, io4dolfinx.FileMode.write, None, "any", on_input_mesh=True)


@pytest.mark.parametrize("error", [RuntimeError, ValueError])
@pytest.mark.parametrize(
    "write",
    [
        lambda u: io4dolfinx.write_mesh(Path("unused"), u.function_space.mesh, on_input_mesh=True),
        lambda u: io4dolfinx.write_function("unused", u, on_input_mesh=True),
        lambda u: io4dolfinx.write_cell_data(
            "unused", u, 0.0, io4dolfinx.FileMode.write, None, "any", on_input_mesh=True
        ),
    ],
)
def test_no_input_cell_index(captured, monkeypatch, error, write):
    """A mesh without input cell indices raises a ValueError when writing in the input
    ordering, whichever error DOLFINx raises for the missing indices."""

    def missing(self):
        raise error("Mixed topology unsupported.")

    monkeypatch.setattr(dolfinx.mesh.Topology, "original_cell_index", property(missing))
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("Lagrange", 1)))
    with pytest.raises(ValueError, match="cannot be written in the input ordering"):
        write(u)
