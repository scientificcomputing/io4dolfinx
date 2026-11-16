"""Tests of the converter, on the checkpoints written by ``tests/create_legacy_checkpoint.py``
with adios4dolfinx 0.7.1 (format 0) and the files written by ``tests/create_legacy_data.py``
with legacy DOLFIN."""

import pathlib

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest

import io4dolfinx
from io4dolfinx.converter import convert, main

pytest.importorskip("adios2")

# Written by tests/create_legacy_checkpoint.py and tests/create_legacy_data.py, which CI runs
# before the tests
legacy_dir = pathlib.Path("legacy_checkpoint").absolute()
dolfin_dir = pathlib.Path("legacy").absolute()
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


def test_convert_dolfin(shared_tmp_path, backend):
    """The mesh and a mesh function of a legacy DOLFIN file become a mesh and mesh tags, with
    the node order of the file, so DOLFIN functions are read onto the converted mesh."""
    source = dolfin_dir / "mesh.h5"
    output = shared_tmp_path / f"dolfin{suffixes[backend]}"
    main([str(source), str(output), "--meshtags", "facet_tags"])

    mesh = io4dolfinx.read_mesh(output, MPI.COMM_WORLD, backend=backend)
    tags = io4dolfinx.read_meshtags(output, mesh, "facet_tags", backend=backend)
    fdim = mesh.topology.dim - 1
    assert tags.dim == fdim
    mesh.topology.create_connectivity(fdim, fdim + 1)
    # exterior_facet_indices returns owned facets, while the tags also hold ghosts
    exterior = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    num_owned = mesh.topology.index_map(fdim).size_local
    boundary, interior = tags.find(2), tags.find(1)
    np.testing.assert_array_equal(np.sort(boundary[boundary < num_owned]), np.sort(exterior))
    np.testing.assert_array_equal(
        np.sort(interior[interior < num_owned]), np.setdiff1d(np.arange(num_owned), exterior)
    )

    W = dolfinx.fem.functionspace(mesh, ("DG", 2, (mesh.geometry.dim,)))
    w = dolfinx.fem.Function(W)
    io4dolfinx.read_function_from_legacy_h5(source, mesh.comm, w, group="w", backend=backend)
    w_ex = dolfinx.fem.Function(W)
    w_ex.interpolate(lambda x: (x[0], 3 * x[2], 7 * x[1]))
    np.testing.assert_allclose(w.x.array, w_ex.x.array, atol=1e-14)

    with pytest.raises(ValueError, match="read_function_from_legacy_h5"):
        convert(source, shared_tmp_path / "other.h5", names=["w"])
    with pytest.raises(ValueError, match="has no mesh tags"):
        convert(
            legacy_dir / "adios4dolfinx_checkpoint.bp",
            shared_tmp_path / "other.h5",
            meshtags=["facet_tags"],
        )
