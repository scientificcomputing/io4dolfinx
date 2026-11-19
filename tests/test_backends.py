from mpi4py import MPI

import dolfinx
import h5py
import numpy as np
import pytest

import io4dolfinx
from io4dolfinx.backends import BUILTIN_BACKENDS


@pytest.fixture(autouse=True)
def reset_default_backend():
    """
    Fixture to start each test with 'adios2' as the global default backend, and to
    restore the default afterwards, preventing state leakage to other tests.
    """
    default = io4dolfinx.get_default_backend()
    io4dolfinx.set_default_backend("adios2")
    yield
    io4dolfinx.set_default_backend(default)


def test_explicit_backend_overrides_default(tmp_path):
    """
    Test that explicitly passing `backend="h5py"` overrides the global
    default backend (which is "adios2").
    """
    comm = MPI.COMM_WORLD

    # Ensure default is currently adios2
    assert io4dolfinx.backends._DEFAULT_BACKEND == "adios2"

    mesh = dolfinx.mesh.create_unit_square(comm, 5, 5)

    # We use .h5 suffix, but the backend argument is what actually dictates the writer
    fname = comm.bcast(tmp_path, root=0) / "override_test.h5"

    # Explicitly pass the h5py backend
    io4dolfinx.write_mesh(fname, mesh, backend="h5py")

    comm.Barrier()

    # Verify that h5py was actually used by attempting to open it as an HDF5 file.
    # If adios2 was used, this would raise an OSError/ValueError.
    if comm.rank == 0:
        assert fname.exists()
        with h5py.File(fname, "r") as f:
            assert "mesh" in f.keys()


def test_set_default_backend_takes_effect(tmp_path):
    """
    Test that calling `set_default_backend("h5py")` successfully changes the
    default behavior for API calls where `backend` is not explicitly provided.
    """
    comm = MPI.COMM_WORLD

    # Update the global default backend to h5py
    io4dolfinx.set_default_backend("h5py")

    mesh = dolfinx.mesh.create_unit_square(comm, 5, 5)

    fname = comm.bcast(tmp_path, root=0) / "default_update_test.h5"

    # Call the API without providing the `backend` argument
    io4dolfinx.write_mesh(fname, mesh)

    comm.Barrier()

    # Verify that h5py was implicitly used based on the new default
    if comm.rank == 0:
        assert fname.exists()
        with h5py.File(fname, "r") as f:
            assert "mesh" in f.keys()
            assert "Topology" in f["mesh"].keys()


def test_list_builtin_backends():
    """
    Test that list_builtin_backends returns a valid list containing
    a subset of the supported built-in backends based on the current environment.
    """
    # Call the function to get the list of available backends
    available_backends = io4dolfinx.backends.list_builtin_backends()

    # Verify the return type is a list
    assert isinstance(available_backends, list)

    # Depending on the test environment, at least one backend should be available
    assert len(available_backends) > 0

    # Verify that all returned backends are recognized as built-in backends
    for backend in available_backends:
        assert isinstance(backend, str)
        assert backend in BUILTIN_BACKENDS

    # We can be reasonably certain that 'h5py' or 'adios2' should be
    # present if the io4dolfinx test suite is running successfully
    assert "h5py" in available_backends or "adios2" in available_backends


class _SerialConfig:
    mpi = False


@pytest.mark.parametrize("backend", ["h5py", "vtkhdf"])
def test_serial_h5py_fallback(monkeypatch, backend):
    """With h5py built without MPI, the HDF5 backends run as their serial versions, with a
    warning to install h5py with MPI."""
    monkeypatch.setattr(h5py, "get_config", lambda: _SerialConfig())
    serial = io4dolfinx.get_backend(f"{backend}-serial")
    with pytest.warns(UserWarning, match=f"runs as {backend}-serial"):
        assert io4dolfinx.get_backend(backend) is serial


def test_serial_adios2_fallback(monkeypatch):
    """With ADIOS2 built without MPI, the adios2 backend runs as adios2-serial, with a warning
    to install ADIOS2 with MPI."""
    adios2 = pytest.importorskip("adios2")
    monkeypatch.setattr(adios2, "is_built_with_mpi", False)
    serial = io4dolfinx.get_backend("adios2-serial")
    with pytest.warns(UserWarning, match="runs as adios2-serial"):
        assert io4dolfinx.get_backend("adios2") is serial


@pytest.mark.parametrize(
    "writer, reader", [("adios2", "adios2-serial"), ("adios2-serial", "adios2")]
)
def test_serial_adios2_writes_adios2_files(tmp_path, writer, reader):
    """adios2 and adios2-serial write the same files: each reads the checkpoints, partitions
    and snapshots of the other."""
    adios2 = pytest.importorskip("adios2")
    if not adios2.is_built_with_mpi:
        pytest.skip("Needs ADIOS2 built with MPI")
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_square(comm, 4, 5)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: x[0] + 2 * x[1] ** 2)
    atol = 10 * np.finfo(u.x.array.dtype).eps

    filename = tmp_path / "checkpoint.bp"
    io4dolfinx.write_mesh(filename, mesh, store_partition_info=True, backend=writer)
    io4dolfinx.write_function(filename, u, time=0.5, backend=writer)
    for read_from_partition in (False, True):
        mesh_in = io4dolfinx.read_mesh(
            filename, comm, read_from_partition=read_from_partition, backend=reader
        )
        u_in = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh_in, ("Lagrange", 2)))
        io4dolfinx.read_function(filename, u_in, time=0.5, backend=reader)
        u_ex = dolfinx.fem.Function(u_in.function_space)
        u_ex.interpolate(lambda x: x[0] + 2 * x[1] ** 2)
        np.testing.assert_allclose(u_in.x.array, u_ex.x.array, atol=atol)

    snapshot = tmp_path / "snapshot.bp"
    io4dolfinx.snapshot_checkpoint(u, snapshot, io4dolfinx.FileMode.write, backend=writer)
    v = dolfinx.fem.Function(V)
    io4dolfinx.snapshot_checkpoint(v, snapshot, io4dolfinx.FileMode.read, backend=reader)
    # Ghosts of u are interpolated locally, those of v come from their owners
    num_owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    np.testing.assert_array_equal(v.x.array[:num_owned], u.x.array[:num_owned])


@pytest.mark.skipif(h5py.get_config().mpi, reason="Needs h5py built without MPI, as on PyPI")
def test_checkpoint_with_serial_h5py(tmp_path):
    """With h5py from PyPI, the default backend writes and reads a checkpoint on any number of
    processes, warning to install h5py with MPI."""
    io4dolfinx.set_default_backend("h5py")
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_square(comm, 4, 5)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: x[0] + 2 * x[1] ** 2)

    filename = tmp_path / "serial_h5py.h5"
    with pytest.warns(UserWarning, match="runs as h5py-serial"):
        io4dolfinx.write_mesh(filename, mesh)
        io4dolfinx.write_function(filename, u, time=0.5)
    with pytest.warns(UserWarning, match="runs as h5py-serial"):
        mesh_in = io4dolfinx.read_mesh(filename, comm)
    u_in = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh_in, ("Lagrange", 2)))
    with pytest.warns(UserWarning, match="runs as h5py-serial"):
        io4dolfinx.read_function(filename, u_in, time=0.5)

    u_ex = dolfinx.fem.Function(u_in.function_space)
    u_ex.interpolate(lambda x: x[0] + 2 * x[1] ** 2)
    np.testing.assert_allclose(u_in.x.array, u_ex.x.array, atol=1e-14)
