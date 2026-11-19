"""H5py lossless filters, time step policies, and collective writes."""

from dataclasses import replace
from typing import Any

from mpi4py import MPI

import dolfinx
import h5py
import numpy as np
import pytest

import io4dolfinx
from io4dolfinx.backends.h5py import backend as h5_backend
from io4dolfinx.backends.h5py_serial import backend as serial_backend
from io4dolfinx.structures import FunctionData, MeshTagsData
from io4dolfinx.writers import prepare_meshdata_for_storage


@pytest.fixture(params=["h5py", "h5py-serial"])
def compression_backend(request):
    if request.param == "h5py" and MPI.COMM_WORLD.size > 1 and not h5py.get_config().mpi:
        pytest.skip("Parallel writes require MPI-enabled h5py")
    return request.param


OPTIONS: list[dict[str, Any]] = [
    {},
    {"scaleoffset": False},
    {"shuffle": True},
    {"fletcher32": True},
]
if h5py.h5z.filter_avail(h5py.h5z.FILTER_DEFLATE):
    OPTIONS.append({"compression": "gzip", "compression_opts": 4, "shuffle": True})


@pytest.mark.parametrize("options", OPTIONS)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_roundtrip(shared_tmp_path, compression_backend, options, dtype):
    comm = MPI.COMM_WORLD
    backend = compression_backend
    filename = shared_tmp_path / "compressed.h5"
    mesh = dolfinx.mesh.create_unit_square(comm, 4, 3)
    args = options | {"chunk_bytes": 48}
    io4dolfinx.write_mesh(
        filename, mesh, backend=backend, backend_args=args, store_partition_info=True
    )
    u = dolfinx.fem.Function(
        dolfinx.fem.functionspace(mesh, ("Lagrange", 2)), name="u", dtype=dtype
    )
    for time in (0.0, 1.0):
        if time:
            io4dolfinx.write_mesh(
                filename, mesh, time=time, mode=io4dolfinx.FileMode.append, backend=backend
            )
        u.interpolate(lambda x: x[0] + 2 * x[1] + time)
        io4dolfinx.write_function(
            filename, u, time=time, backend=backend, backend_args=args if not time else None
        )
        restored_mesh = io4dolfinx.read_mesh(
            filename, comm, time=time, backend=backend, read_from_partition=True
        )
        v = dolfinx.fem.Function(
            dolfinx.fem.functionspace(restored_mesh, ("Lagrange", 2)), name="u", dtype=dtype
        )
        io4dolfinx.read_function(filename, v, time=time, backend=backend)
        expected = dolfinx.fem.Function(v.function_space, dtype=dtype)
        expected.interpolate(lambda x: x[0] + 2 * x[1] + time)
        np.testing.assert_allclose(v.x.array, expected.x.array, atol=1e-14)

    dim = mesh.topology.dim
    cell_map = mesh.topology.index_map(dim)
    cells = np.arange(cell_map.size_local, dtype=np.int32)
    tags = dolfinx.mesh.meshtags(mesh, dim, cells, cells + cell_map.local_range[0])
    io4dolfinx.write_meshtags(filename, mesh, tags, "tags", backend=backend, backend_args=args)
    restored_tags = io4dolfinx.read_meshtags(filename, restored_mesh, "tags", backend=backend)
    np.testing.assert_array_equal(
        np.unique(np.hstack(comm.allgather(restored_tags.values))),
        np.arange(cell_map.size_global),
    )
    with h5py.File(filename, "r") as file:
        for path in (
            "Topology",
            "PartitioningData",
            "PartitioningOffset",
            "CellPermutations",
            "functions/u/dofmap",
            "functions/u/dofmap_offsets",
            "tags/tags/Topology",
            "tags/tags/Values",
            "0/Points",
            "1/Points",
            "functions/u/0/array",
            "functions/u/1/array",
        ):
            dataset = file[f"mesh/{path}"]
            assert dataset.compression == options.get("compression")
            assert dataset.shuffle == options.get("shuffle", False)
            assert dataset.fletcher32 == options.get("fletcher32", False)
            integer_filter = np.issubdtype(dataset.dtype, np.integer) and options.get(
                "scaleoffset", not options.get("fletcher32", False)
            )
            assert (dataset.scaleoffset is not None) == integer_filter
            row_bytes = dataset.dtype.itemsize * int(np.prod(dataset.shape[1:]))
            assert dataset.chunks == (
                min(dataset.shape[0], max(1, 48 // row_bytes)),
                *dataset.shape[1:],
            )


@pytest.mark.parametrize("target", ["mesh", "function"])
def test_conflicting_append(shared_tmp_path, compression_backend, target):
    filename = shared_tmp_path / "conflict.h5"
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    backend = compression_backend
    io4dolfinx.write_mesh(filename, mesh, backend=backend)
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("Lagrange", 1)), name="u")
    io4dolfinx.write_function(filename, u, backend=backend)
    before = filename.read_bytes()
    with pytest.raises(ValueError, match="cannot append"):
        if target == "mesh":
            io4dolfinx.write_mesh(
                filename,
                mesh,
                time=1.0,
                mode=io4dolfinx.FileMode.append,
                backend=backend,
                backend_args={"shuffle": True},
            )
        else:
            io4dolfinx.write_function(
                filename, u, time=1.0, backend=backend, backend_args={"shuffle": True}
            )
    assert filename.read_bytes() == before


@pytest.mark.parametrize(
    "options",
    [
        {"compression": "lzf"},
        {"compression_opts": 4},
        {"shuffle": 1},
        {"chunk_bytes": 0},
        {"chunks": True},
        {"fletcher32": True, "scaleoffset": True},
    ],
)
def test_invalid_options(shared_tmp_path, compression_backend, options):
    filename = shared_tmp_path / "invalid.h5"
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    with pytest.raises(ValueError):
        io4dolfinx.write_mesh(filename, mesh, backend=compression_backend, backend_args=options)
    assert not filename.exists()


def test_inconsistent_options(shared_tmp_path, compression_backend):
    comm = MPI.COMM_WORLD
    if comm.size == 1:
        pytest.skip("Requires multiple ranks")
    mesh = dolfinx.mesh.create_unit_square(comm, 2, 2)
    filename = shared_tmp_path / "inconsistent.h5"
    with pytest.raises(ValueError, match="identical"):
        io4dolfinx.write_mesh(
            filename, mesh, backend=compression_backend, backend_args={"shuffle": comm.rank == 0}
        )
    assert not filename.exists()


@pytest.mark.parametrize("options", OPTIONS)
def test_snapshot(shared_tmp_path, compression_backend, options):
    filename = shared_tmp_path / "snapshot.h5"
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("Lagrange", 1)))
    u.interpolate(lambda x: x[0] + x[1])
    io4dolfinx.snapshot_checkpoint(
        u, filename, io4dolfinx.FileMode.write, backend=compression_backend, backend_args=options
    )
    v = dolfinx.fem.Function(u.function_space)
    io4dolfinx.snapshot_checkpoint(
        v, filename, io4dolfinx.FileMode.read, backend=compression_backend
    )
    np.testing.assert_array_equal(v.x.array, u.x.array)
    with h5py.File(filename, "r") as file:
        assert file["snapshot/dofs"].compression == options.get("compression")


@pytest.mark.parametrize("options", OPTIONS)
def test_empty_ranks(shared_tmp_path, compression_backend, options):
    comm = MPI.COMM_WORLD
    backend = h5_backend if compression_backend == "h5py" else serial_backend
    filename = shared_tmp_path / "empty.h5"
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_SELF, 1, 1)
    data = prepare_meshdata_for_storage(mesh, False)
    if comm.rank:
        data = replace(
            data,
            local_geometry=data.local_geometry[:0],
            local_topology=data.local_topology[:0],
            local_geometry_pos=(4, 4),
            local_topology_pos=(2, 2),
        )
    backend.write_mesh(filename, comm, data, options, io4dolfinx.FileMode.write, 0.0)
    # A synthetic DG0 checkpoint: only rank 0 owns cells and values.
    start = 2 if comm.rank else 0
    reference = np.array([np.nan, -np.inf])
    function = FunctionData(
        cell_permutations=np.zeros(2 - start, dtype=np.uint32),
        local_cell_range=(start, 2),
        num_cells_global=2,
        dofmap_array=np.arange(start, 2, dtype=np.int64),
        dofmap_offsets=np.arange(start, 3, dtype=np.int64),
        dofmap_range=(start, 2),
        global_dofs_in_dofmap=2,
        values=reference[start:],
        dof_range=(start, 2),
        num_dofs_global=2,
        name="u",
    )
    backend.write_function(filename, comm, function, 0.0, io4dolfinx.FileMode.append, options)
    tags = MeshTagsData(
        "tags",
        np.arange(start, 2, dtype=np.int32),
        data.local_topology,
        2,
        num_entities_global=2,
        num_dofs_per_entity=3,
        local_start=start,
        cell_type="triangle",
    )
    backend.write_meshtags(filename, comm, tags, options)
    with h5py.File(filename, "r") as file:
        np.testing.assert_array_equal(file["mesh/functions/u/dofmap_offsets"][:], [0, 1, 2])
        assert file["mesh/functions/u/0/array"][:].tobytes() == reference.tobytes()
        np.testing.assert_array_equal(file["mesh/tags/tags/Values"][:], [0, 1])


def test_legacy_layout_and_new_functions(shared_tmp_path, compression_backend):
    backend = compression_backend
    filename = shared_tmp_path / "legacy.h5"
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    io4dolfinx.write_mesh(filename, mesh, backend=backend, backend_args={"scaleoffset": False})
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("Lagrange", 1)), name="u")
    io4dolfinx.write_function(filename, u, backend=backend, backend_args={"scaleoffset": False})
    # Replace the first geometry and values with contiguous datasets, as in old files.
    MPI.COMM_WORLD.barrier()
    if MPI.COMM_WORLD.rank == 0:
        with h5py.File(filename, "a") as file:
            for path in ("mesh/0/Points", "mesh/functions/u/0/array"):
                values = file[path][:]
                del file[path]
                file.create_dataset(path, data=values)
    MPI.COMM_WORLD.barrier()
    io4dolfinx.write_mesh(
        filename, mesh, time=1.0, mode=io4dolfinx.FileMode.append, backend=backend
    )
    io4dolfinx.write_function(filename, u, time=1.0, backend=backend)
    u.name = "new"
    io4dolfinx.write_function(filename, u, backend=backend, backend_args={"shuffle": True})
    with h5py.File(filename, "r") as file:
        assert file["mesh/1/Points"].chunks is None
        assert file["mesh/functions/u/1/array"].chunks is None
        assert file["mesh/functions/new/0/array"].shuffle
        assert file["mesh/functions/new/dofmap"].scaleoffset is not None


@pytest.mark.parametrize("options", OPTIONS)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("shape", [(), (2,)])
@pytest.mark.parametrize("on_input_mesh", [False, True])
def test_function_only(shared_tmp_path, compression_backend, options, dtype, shape, on_input_mesh):
    """Read function checkpoints with an existing mesh, without storing that mesh."""
    filename = shared_tmp_path / "function_only.h5"
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 3)
    space = dolfinx.fem.functionspace(mesh, ("Lagrange", 2, shape))
    u = dolfinx.fem.Function(space, name="u", dtype=dtype)
    read_mesh = mesh
    if not on_input_mesh:
        mesh_filename = shared_tmp_path / "mesh_only.h5"
        io4dolfinx.write_mesh(mesh_filename, mesh, backend=compression_backend)
        read_mesh = io4dolfinx.read_mesh(mesh_filename, mesh.comm, backend=compression_backend)
    read_space = dolfinx.fem.functionspace(read_mesh, ("Lagrange", 2, shape))
    v = dolfinx.fem.Function(read_space, name="u", dtype=dtype)
    expected_function = dolfinx.fem.Function(read_space, dtype=dtype)
    for time in (0.0, 1.0):

        def field(x):
            values = x[0] + 2 * x[1] + time
            if np.issubdtype(dtype, np.complexfloating):
                values = values + 1j * (x[0] - x[1] + time)
            return np.vstack((values, 2 * values)) if shape else values

        u.interpolate(field)
        io4dolfinx.write_function(
            filename,
            u,
            time=time,
            mode=io4dolfinx.FileMode.write if time == 0.0 else io4dolfinx.FileMode.append,
            backend=compression_backend,
            on_input_mesh=on_input_mesh,
            backend_args=options if time == 0.0 else None,
        )
        io4dolfinx.read_function(filename, v, time=time, backend=compression_backend)
        expected_function.interpolate(field)
        np.testing.assert_allclose(v.x.array, expected_function.x.array, atol=1e-14)

    # Earlier steps remain readable after appending; mesh geometry/topology are absent.
    u = dolfinx.fem.Function(read_space, name="u", dtype=dtype)
    io4dolfinx.read_function(filename, u, time=0.0, backend=compression_backend)
    io4dolfinx.read_function(filename, v, time=1.0, backend=compression_backend)
    difference = 1 + (1j if np.issubdtype(dtype, np.complexfloating) else 0)
    expected = np.tile([difference, 2 * difference], len(u.x.array) // 2) if shape else difference
    np.testing.assert_allclose(v.x.array - u.x.array, expected, atol=1e-14)
    with h5py.File(filename, "r") as file:
        group = file["mesh"]
        assert "Topology" not in group and "0" not in group
        for step in (0, 1):
            dataset = group[f"functions/u/{step}/array"]
            assert dataset.compression == options.get("compression")
            assert dataset.shuffle == options.get("shuffle", False)
            assert dataset.fletcher32 == options.get("fletcher32", False)
            assert dataset.scaleoffset is None
