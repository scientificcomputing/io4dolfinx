from mpi4py import MPI

import dolfinx
import h5py
import pytest

import io4dolfinx
from io4dolfinx.backends import BUILTIN_BAKENDS


@pytest.fixture(autouse=True)
def reset_default_backend():
    """
    Fixture to ensure the global default backend is always reset to 'adios2'
    after each test, preventing state leakage to other tests.
    """
    # Setup: Ensure starting state is adios2
    io4dolfinx.set_default_backend("adios2")
    yield
    # Teardown: Reset back to adios2
    io4dolfinx.set_default_backend("adios2")


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
        assert backend in BUILTIN_BAKENDS

    # We can be reasonably certain that 'h5py' or 'adios2' should be
    # present if the io4dolfinx test suite is running successfully
    assert "h5py" in available_backends or "adios2" in available_backends
