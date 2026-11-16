"""Tests of the converter of adios4dolfinx checkpoints (format 0), on the checkpoints written
by ``tests/create_legacy_checkpoint.py`` with adios4dolfinx 0.7.1."""

import pathlib

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

import io4dolfinx
from io4dolfinx.converter import convert, main

pytest.importorskip("adios2")

# Written by tests/create_legacy_checkpoint.py, which CI runs before the tests
legacy_dir = pathlib.Path("legacy_checkpoint").absolute()
suffixes = {"adios2": ".bp", "h5py": ".h5"}


def f(x):
    return np.vstack([x[0], -x[1]])


def g(x):
    return x[0] + 2 * x[1]


def check_function(filename, mesh, element, name, expression, backend):
    V = dolfinx.fem.functionspace(mesh, element)
    u = dolfinx.fem.Function(V)
    io4dolfinx.read_function(filename, u, name=name, backend=backend)
    u_ex = dolfinx.fem.Function(V)
    u_ex.interpolate(expression)
    np.testing.assert_allclose(u.x.array, u_ex.x.array, atol=1e-14)


@pytest.mark.parametrize("command_line", [False, True])
def test_convert(shared_tmp_path, backend, command_line):
    """The converted checkpoint is read as any other, with the backend chosen from its
    suffix."""
    source = legacy_dir / "adios4dolfinx_checkpoint.bp"
    output = shared_tmp_path / f"converted{suffixes[backend]}"
    if command_line:
        main([str(source), str(output)])
    else:
        convert(source, output)

    comm = MPI.COMM_WORLD
    mesh = io4dolfinx.read_mesh(output, comm, backend=backend)
    assert mesh.topology.index_map(mesh.topology.dim).size_global == 200
    check_function(output, mesh, ("N1curl", 3), "u", f, backend)


def test_names(shared_tmp_path, backend):
    """Each function of the file gets the name at its position."""
    source = legacy_dir / "adios4dolfinx_two_functions.bp"
    output = shared_tmp_path / "two_functions.checkpoint"
    main([str(source), str(output), "--names", "u", "v", "--backend", backend])

    mesh = io4dolfinx.read_mesh(output, MPI.COMM_WORLD, backend=backend)
    check_function(output, mesh, ("N1curl", 3), "u", f, backend)
    check_function(output, mesh, ("Lagrange", 2), "v", g, backend)

    with pytest.raises(ValueError, match="holds 2 functions, so it needs 2 names"):
        convert(source, shared_tmp_path / "other.bp", names=["u"])
    with pytest.raises(ValueError, match="must differ"):
        convert(source, shared_tmp_path / "other.bp", names=["u", "u"])


def test_errors(shared_tmp_path):
    """Files that are not format 0, outputs without a known suffix or that exist, and
    backends that cannot write checkpoints are refused."""
    comm = MPI.COMM_WORLD
    current = shared_tmp_path / "current.bp"
    mesh = dolfinx.mesh.create_unit_square(comm, 2, 2)
    io4dolfinx.write_mesh(current, mesh, backend="adios2")

    with pytest.raises(ValueError, match="can be read directly"):
        convert(current, shared_tmp_path / "converted.bp")
    with pytest.raises(ValueError, match="Cannot tell the backend"):
        convert(current, shared_tmp_path / "converted.xdmf")
    with pytest.raises(ValueError, match="Cannot write checkpoints with the 'vtkhdf' backend"):
        convert(current, shared_tmp_path / "converted.bp", backend="vtkhdf")
    with pytest.raises(FileExistsError, match="exists"):
        convert(current, current)
    with pytest.raises(ValueError, match="cannot replace"):
        convert(current, current, overwrite=True)
    with pytest.raises(FileNotFoundError):
        convert(shared_tmp_path / "missing.bp", shared_tmp_path / "converted.bp")
