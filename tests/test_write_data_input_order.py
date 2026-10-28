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
    :func:`io4dolfinx.write_mesh_input_order`."""
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 4, 3, 5, cell_type=cell_type)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (3,)))
    u = dolfinx.fem.Function(V)
    u.interpolate(f)

    for writer in (io4dolfinx.write_point_data, io4dolfinx.write_cell_data):
        writer("unused", u, 0.0, io4dolfinx.FileMode.write, None, "any", on_input_mesh=True)
    mesh_data = create_original_mesh_data(mesh)

    point_data = captured["Point"]
    assert point_data.on_input_mesh
    assert tuple(point_data.local_range) == tuple(mesh_data.local_geometry_pos)
    np.testing.assert_allclose(point_data.values, f(mesh_data.local_geometry.T).T, atol=1e-12)

    # P-1 interpolated into DG-0 is the mean of the vertex values for these cells
    cell_data = captured["Cell"]
    assert cell_data.on_input_mesh
    assert tuple(cell_data.local_range) == tuple(mesh_data.local_topology_pos)
    x = np.vstack(MPI.COMM_WORLD.allgather(mesh_data.local_geometry))
    midpoints = x[mesh_data.local_topology].mean(axis=1)
    np.testing.assert_allclose(cell_data.values, f(midpoints.T).T, atol=1e-12)
