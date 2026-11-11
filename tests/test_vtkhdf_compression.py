"""Lossless VTKHDF filters, append policies, and collective empty selections."""

from dataclasses import replace

from mpi4py import MPI

import h5py
import numpy as np
import pytest
from dolfinx.fem import Function, functionspace
from dolfinx.mesh import create_unit_square
from vtkmodules.vtkIOHDF import vtkHDFReader

import io4dolfinx
from io4dolfinx.backends.vtkhdf import backend
from io4dolfinx.structures import ArrayData
from io4dolfinx.writers import prepare_meshdata_for_storage


@pytest.mark.parametrize("compression", [None, "gzip"])
@pytest.mark.parametrize("shuffle", [False, True])
def test_compressed_roundtrip(shared_tmp_path, compression, shuffle):
    comm = MPI.COMM_WORLD
    filename = shared_tmp_path / "roundtrip.vtkhdf"
    mesh = create_unit_square(comm, 5, 4)
    options = {"compression": compression, "shuffle": shuffle}
    if compression:
        options["compression_opts"] = 4
    io4dolfinx.write_mesh(filename, mesh, backend="vtkhdf", backend_args=options)
    for time in (0.0, 1.0):
        # Omitting options on append must retain the existing mesh filters.
        if time:
            io4dolfinx.write_mesh(
                filename, mesh, time=time, mode=io4dolfinx.FileMode.append, backend="vtkhdf"
            )
        for association in ("point", "cell"):
            for components in (1, 3):
                family, degree = ("Lagrange", 1) if association == "point" else ("DG", 0)
                shape = () if components == 1 else (3,)
                V = functionspace(mesh, (family, degree, shape))
                u = Function(V, name=f"{association}{components}")
                u.interpolate(
                    (lambda x: x[0] + time)
                    if components == 1
                    else (lambda x: np.vstack((x[0] + time, x[1], 2 * x[0])))
                )
                writer = getattr(io4dolfinx, f"write_{association}_data")
                writer(
                    filename,
                    u,
                    time=time,
                    mode=io4dolfinx.FileMode.append,
                    backend_args=options if time == 0.0 else None,
                )
                reader = getattr(io4dolfinx, f"read_{association}_data")
                restored = io4dolfinx.read_mesh(filename, comm, time=time, backend="vtkhdf")
                result = reader(filename, mesh=restored, name=u.name, time=time, backend="vtkhdf")
                expected = Function(result.function_space)
                expected.interpolate(
                    (lambda x: x[0] + time)
                    if components == 1
                    else (lambda x: np.vstack((x[0] + time, x[1], 2 * x[0])))
                )
                np.testing.assert_allclose(result.x.array, expected.x.array, atol=1e-14)

    # A newly created field does not inherit the mesh's filters.
    plain = Function(functionspace(mesh, ("Lagrange", 1)), name="plain")
    io4dolfinx.write_point_data(
        filename, plain, time=1.0, mode=io4dolfinx.FileMode.append, backend_args=None
    )
    with h5py.File(filename, "r") as hdf:
        block = hdf["VTKHDF/mesh"]
        for path in (
            "Points",
            "Connectivity",
            "Offsets",
            "Types",
            "PointData/point1",
            "PointData/point3",
            "CellData/cell1",
            "CellData/cell3",
        ):
            dataset = block[path]
            assert dataset.compression == compression
            assert dataset.shuffle == shuffle
            assert dataset.chunks is not None
            if compression:
                assert dataset.compression_opts == 4
        for path in ("NumberOfPoints", "NumberOfCells", "Steps/Values", "PointData/plain"):
            assert block[path].compression is None
            assert not block[path].shuffle

    for time in (0.0, 1.0):
        restored = io4dolfinx.read_mesh(filename, comm, time=time, backend="vtkhdf")
        assert restored.topology.index_map(2).size_global == mesh.topology.index_map(2).size_global
        restored_points = np.vstack(
            comm.allgather(restored.geometry.x[: restored.geometry.index_map().size_local])
        )
        original_points = np.vstack(
            comm.allgather(mesh.geometry.x[: mesh.geometry.index_map().size_local])
        )
        np.testing.assert_allclose(
            np.sort(restored_points, axis=0), np.sort(original_points, axis=0)
        )
        vtk_reader = vtkHDFReader()
        vtk_reader.SetFileName(str(filename))
        vtk_reader.UpdateTimeStep(time)
        output = vtk_reader.GetOutputDataObject(0)
        while output.IsA("vtkMultiBlockDataSet"):
            output = output.GetBlock(0)
        assert output.GetNumberOfPoints() == mesh.geometry.index_map().size_global
        assert output.GetNumberOfCells() == mesh.topology.index_map(2).size_global
        for association in ("point", "cell"):
            attributes = output.GetPointData() if association == "point" else output.GetCellData()
            for components in (1, 3):
                values = attributes.GetArray(f"{association}{components}")
                assert values is not None
                assert values.GetNumberOfComponents() == components
                # Scalar values are x + time at points, or at cell centroids.
                for i in range(values.GetNumberOfTuples()):
                    if association == "point":
                        x = output.GetPoint(i)[0]
                    else:
                        cell = output.GetCell(i)
                        x = np.mean(
                            [
                                output.GetPoint(cell.GetPointId(j))[0]
                                for j in range(cell.GetNumberOfPoints())
                            ]
                        )
                    assert np.isclose(values.GetComponent(i, 0), x + time)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("association", ["Point", "Cell"])
def test_compressed_empty_ranks(shared_tmp_path, dtype, association):
    """Empty ranks participate, and filters preserve special floating-point values."""
    comm = MPI.COMM_WORLD
    filename = shared_tmp_path / "empty_ranks.vtkhdf"
    mesh = create_unit_square(MPI.COMM_SELF, 2, 2, dtype=dtype)
    data = prepare_meshdata_for_storage(mesh, False)
    if comm.rank:
        data = replace(
            data,
            local_geometry=data.local_geometry[:0],
            local_topology=data.local_topology[:0],
            local_geometry_pos=(data.num_nodes_global, data.num_nodes_global),
            local_topology_pos=(data.num_cells_global, data.num_cells_global),
        )
    options = {"compression": "gzip", "shuffle": True}
    backend.write_mesh(filename, comm, data, options, io4dolfinx.FileMode.write, 0.0)
    count = data.num_nodes_global if association == "Point" else data.num_cells_global
    local_range = data.local_geometry_pos if association == "Point" else data.local_topology_pos
    reference = np.resize(np.array([-0.0, np.nan, np.inf, -np.inf, 1.2345], dtype=dtype), count)
    values = reference[slice(*local_range)].reshape(-1, 1).copy()
    field = ArrayData("special", values, (count, 1), local_range, association)
    backend.write_data(filename, field, comm, 0.0, io4dolfinx.FileMode.append, options)
    with h5py.File(filename, "r") as hdf:
        block = hdf["VTKHDF/mesh"]
        np.testing.assert_array_equal(block["Offsets"][:], np.arange(0, 25, 3))
        stored = block[f"{association}Data/special"][:, 0]
        # Compare bytes as well: equality alone does not distinguish +0 from -0.
        assert stored.tobytes() == reference.tobytes()


@pytest.mark.parametrize(
    "options",
    [
        {"compression": "lzf"},
        {"compression": "gzip", "compression_opts": 10},
        {"compression": "gzip", "compression_opts": True},
        {"compression_opts": 4},
        {"shuffle": "yes"},
        {"chunks": (25000,)},
        {"scaleoffset": 2},
    ],
)
def test_invalid_options_do_not_create_file(shared_tmp_path, options):
    comm = MPI.COMM_WORLD
    mesh = create_unit_square(comm, 2, 2)
    filename = shared_tmp_path / "invalid.vtkhdf"
    with pytest.raises(ValueError):
        io4dolfinx.write_mesh(filename, mesh, backend="vtkhdf", backend_args=options)
    assert not filename.exists()


@pytest.mark.parametrize("target", ["mesh", "point", "cell"])
@pytest.mark.parametrize(
    "conflict",
    [
        {"compression": None},
        {"compression": "gzip", "compression_opts": 1},
        {"shuffle": False},
    ],
)
def test_conflicting_append_preserves_file(shared_tmp_path, target, conflict):
    comm = MPI.COMM_WORLD
    filename = shared_tmp_path / "conflict.vtkhdf"
    mesh = create_unit_square(comm, 2, 2)
    options = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
    io4dolfinx.write_mesh(filename, mesh, backend="vtkhdf", backend_args=options)
    u = Function(functionspace(mesh, ("Lagrange", 1)), name="u")
    if target != "mesh":
        getattr(io4dolfinx, f"write_{target}_data")(
            filename, u, time=0.0, mode=io4dolfinx.FileMode.append, backend_args=options
        )
    before = filename.read_bytes()
    with pytest.raises(ValueError, match="cannot append"):
        if target == "mesh":
            io4dolfinx.write_mesh(
                filename,
                mesh,
                time=1.0,
                mode=io4dolfinx.FileMode.append,
                backend="vtkhdf",
                backend_args=conflict,
            )
        else:
            getattr(io4dolfinx, f"write_{target}_data")(
                filename, u, time=1.0, mode=io4dolfinx.FileMode.append, backend_args=conflict
            )
    assert filename.read_bytes() == before


def test_rank_local_filter_failure(shared_tmp_path, monkeypatch):
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        monkeypatch.setattr(h5py.h5z, "filter_avail", lambda _: False)
    mesh = create_unit_square(comm, 2, 2)
    filename = shared_tmp_path / "unavailable.vtkhdf"
    with pytest.raises(ValueError, match="unavailable"):
        io4dolfinx.write_mesh(
            filename, mesh, backend="vtkhdf", backend_args={"compression": "gzip"}
        )
    assert not filename.exists()


def test_inconsistent_rank_options(shared_tmp_path):
    comm = MPI.COMM_WORLD
    if comm.size == 1:
        pytest.skip("Requires multiple MPI ranks")
    mesh = create_unit_square(comm, 2, 2)
    filename = shared_tmp_path / "inconsistent.vtkhdf"
    with pytest.raises(ValueError, match="identical"):
        io4dolfinx.write_mesh(
            filename,
            mesh,
            backend="vtkhdf",
            backend_args={"compression": "gzip", "compression_opts": comm.rank % 10},
        )
    assert not filename.exists()
