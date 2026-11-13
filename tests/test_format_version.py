"""Checkpoint format versions: stamps, reading and appending, see io4dolfinx.format."""

from importlib.metadata import version

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

import io4dolfinx
from io4dolfinx import checkpointing, format, writers
from io4dolfinx.backends import get_backend


def filename(path, backend):
    return path / ("checkpoint.bp" if backend == "adios2" else "checkpoint.h5")


def write_checkpoint(path, backend, time=0.0, mode=io4dolfinx.FileMode.write):
    """Write a mesh, a function and mesh tags."""
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("Lagrange", 1)), name="u")
    u.interpolate(lambda x: x[0] + 2 * x[1] + time)
    io4dolfinx.write_mesh(path, mesh, backend=backend, mode=mode)
    io4dolfinx.write_function(path, u, time=time, backend=backend)
    cells = np.arange(mesh.topology.index_map(2).size_local, dtype=np.int32)
    tags = dolfinx.mesh.meshtags(mesh, 2, cells, np.full_like(cells, 3))
    io4dolfinx.write_meshtags(path, mesh, tags, meshtag_name="tags", backend=backend)
    return mesh, u


def read_checkpoint(path, backend, time=0.0):
    """Read the mesh, function and mesh tags, and check the function values."""
    mesh = io4dolfinx.read_mesh(path, MPI.COMM_WORLD, backend=backend)
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("Lagrange", 1)))
    io4dolfinx.read_function(path, u, time=time, name="u", backend=backend)
    expected = dolfinx.fem.Function(u.function_space)
    expected.interpolate(lambda x: x[0] + 2 * x[1] + time)
    np.testing.assert_allclose(u.x.array, expected.x.array, atol=1e-12)
    io4dolfinx.read_meshtags(path, mesh, "tags", backend=backend)


def metadata(path, backend, kind, name=None):
    return format.object_metadata(get_backend(backend), path, MPI.COMM_WORLD, kind, name)


OBJECTS = [("mesh", None), ("function", "u"), ("meshtags", "tags")]


def test_objects_are_stamped(shared_tmp_path, backend):
    path = filename(shared_tmp_path, backend)
    write_checkpoint(path, backend)
    for kind, name in OBJECTS:
        stamp = metadata(path, backend, kind, name)
        assert format.format_version(stamp) == format.FORMAT_VERSION, kind
        assert stamp[format.VERSION_KEY] == version("io4dolfinx")
        assert stamp[format.DOLFINX_VERSION_KEY] == dolfinx.__version__
    read_checkpoint(path, backend)


def test_unstamped_checkpoint_is_read(shared_tmp_path, backend, monkeypatch):
    """Checkpoints written before stamps were introduced are read as format 1.0."""
    for module in (checkpointing, writers):
        monkeypatch.setattr(module, "stamp", dict)
    path = filename(shared_tmp_path, backend)
    write_checkpoint(path, backend)
    monkeypatch.undo()
    for kind, name in OBJECTS:
        assert metadata(path, backend, kind, name) == {}
    read_checkpoint(path, backend)


def test_newer_minor_version_is_read(shared_tmp_path, backend, monkeypatch):
    monkeypatch.setattr(format, "FORMAT_VERSION", (format.FORMAT_VERSION[0], 99))
    path = filename(shared_tmp_path, backend)
    write_checkpoint(path, backend)
    monkeypatch.undo()
    read_checkpoint(path, backend)


def test_newer_major_version_is_not_read(shared_tmp_path, backend, monkeypatch):
    monkeypatch.setattr(format, "FORMAT_VERSION", (format.FORMAT_VERSION[0] + 1, 0))
    path = filename(shared_tmp_path, backend)
    write_checkpoint(path, backend)
    monkeypatch.undo()
    with pytest.raises(ValueError, match="Upgrade io4dolfinx"):
        io4dolfinx.read_mesh(path, MPI.COMM_WORLD, backend=backend)


def test_append_to_older_minor_version(shared_tmp_path, backend, monkeypatch):
    """Appending to an object with an older minor version keeps its stamp."""
    path = filename(shared_tmp_path, backend)
    mesh, u = write_checkpoint(path, backend)
    newer = (format.FORMAT_VERSION[0], format.FORMAT_VERSION[1] + 1)
    monkeypatch.setattr(format, "FORMAT_VERSION", newer)
    u.interpolate(lambda x: x[0] + 2 * x[1] + 1.0)
    io4dolfinx.write_function(path, u, time=1.0, backend=backend)
    monkeypatch.undo()
    stamp = metadata(path, backend, "function", "u")
    assert format.format_version(stamp) == format.FORMAT_VERSION
    read_checkpoint(path, backend, time=1.0)


def test_append_to_other_major_version(shared_tmp_path, backend, monkeypatch):
    path = filename(shared_tmp_path, backend)
    monkeypatch.setattr(format, "FORMAT_VERSION", (format.FORMAT_VERSION[0] + 1, 0))
    mesh, u = write_checkpoint(path, backend)
    monkeypatch.undo()
    with pytest.raises(ValueError, match="cannot append"):
        io4dolfinx.write_function(path, u, time=1.0, backend=backend)


def test_h5py_write_function_in_write_mode(shared_tmp_path):
    """Writing a function to a new h5py file creates its groups, and stamps it."""
    if "h5py" not in io4dolfinx.backends.list_builtin_backends():
        pytest.skip("h5py backend not available")
    pytest.importorskip("h5py")
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("Lagrange", 1)), name="u")
    path = shared_tmp_path / "function_only.h5"
    io4dolfinx.write_function(path, u, backend="h5py", mode=io4dolfinx.FileMode.write)
    stamp = metadata(path, "h5py", "function", "u")
    assert format.format_version(stamp) == format.FORMAT_VERSION
