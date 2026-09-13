from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

import io4dolfinx

SUFFIX = {"adios2": ".bp", "h5py": ".h5"}


def _reference(x, vector: bool, gdim: int):
    """A function that is exactly representable in every space tested below."""
    if vector:
        return np.vstack([np.sin(1.3 * x[0] + 0.2), x[1], 0.3 + 0.0 * x[2]])[:gdim]
    return np.sin(1.3 * x[0] + 0.2) + x[1] - 0.5 * x[2]


def _make_submesh(comm, codim, n=4):
    """A cube, and a submesh of co-dimension ``codim`` of it."""
    mesh = dolfinx.mesh.create_unit_cube(
        comm, n, n, n, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
    )
    dim = mesh.topology.dim - codim
    mesh.topology.create_entities(dim)
    if codim == 0:

        def locator(x):
            return x[0] <= 0.5 + 1e-12
    else:

        def locator(x):
            return np.isclose(x[0], 0.0)

    entities = dolfinx.mesh.locate_entities(mesh, dim, locator)
    submesh, cell_map, _, _ = dolfinx.mesh.create_submesh(mesh, dim, entities)
    return mesh, submesh, cell_map


def _max_error(u, reference):
    """Largest difference over dofs owned by any process.

    A process may own no dofs at all -- a small submesh spread over many ranks --
    so the local maximum is only taken when there is something to take it over.
    The reduction itself still has to run on every process.
    """
    V = u.function_space
    num_owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    local = (
        float(np.max(np.abs(u.x.array[:num_owned] - reference.x.array[:num_owned])))
        if num_owned
        else 0.0
    )
    return V.mesh.comm.allreduce(local, MPI.MAX)


@pytest.mark.parametrize("codim", [0, 1])
@pytest.mark.parametrize(
    "family,degree",
    [("Lagrange", 1), ("Lagrange", 2), ("Lagrange", 3), ("Lagrange", 4), ("DG", 0), ("DG", 4)],
)
@pytest.mark.parametrize("same_file", [True, False])
def test_submesh_roundtrip(tmp_path, backend, codim, family, degree, same_file):
    """Write a submesh and a function on it, read both back, reattach to the parent."""
    comm = MPI.COMM_WORLD
    folder = comm.bcast(tmp_path, root=0)
    parent_file = folder / f"parent{SUFFIX[backend]}"
    sub_file = parent_file if same_file else folder / f"sub{SUFFIX[backend]}"

    mesh, submesh, cell_map = _make_submesh(comm, codim)
    gdim = submesh.geometry.dim

    def f(x):
        return _reference(x, vector=False, gdim=gdim)

    V = dolfinx.fem.functionspace(submesh, (family, degree))
    u = dolfinx.fem.Function(V, name="u")
    u.interpolate(f)

    io4dolfinx.write_mesh(parent_file, mesh, backend=backend)
    io4dolfinx.write_submesh(
        sub_file,
        submesh,
        mesh,
        cell_map,
        mesh_name="wall",
        parent_filename=None if same_file else parent_file,
        mode=io4dolfinx.FileMode.append if same_file else io4dolfinx.FileMode.write,
        backend=backend,
    )
    io4dolfinx.write_function(
        sub_file, u, time=0.0, mode=io4dolfinx.FileMode.append, mesh_name="wall", backend=backend
    )
    del mesh, submesh, cell_map, u

    # Steps 1-3: the submesh is an ordinary mesh, so this is the ordinary path.
    parent = io4dolfinx.read_mesh(parent_file, comm, backend=backend)
    stored = io4dolfinx.read_mesh(sub_file, comm, mesh_name="wall", backend=backend)
    V_stored = dolfinx.fem.functionspace(stored, (family, degree))
    u_stored = dolfinx.fem.Function(V_stored, name="u")
    io4dolfinx.read_function(
        sub_file, u_stored, time=0.0, name="u", mesh_name="wall", backend=backend
    )
    reference = dolfinx.fem.Function(V_stored)
    reference.interpolate(f)
    assert _max_error(u_stored, reference) < 1e-13

    # Steps 4-5: re-derive from the parent and move the data across.
    checkpoint = io4dolfinx.read_submesh(parent_file, parent, mesh_name="wall", backend=backend)
    V_sub = dolfinx.fem.functionspace(checkpoint.submesh, (family, degree))
    u_sub = dolfinx.fem.Function(V_sub)
    io4dolfinx.transfer_submesh_function(u_stored, u_sub, checkpoint.stored_cells)

    reference_sub = dolfinx.fem.Function(V_sub)
    reference_sub.interpolate(f)
    assert _max_error(u_sub, reference_sub) < 1e-13

    # An L2 error assembled on the re-derived submesh alone: one mesh, no maps.
    error = ufl.inner(u_sub - reference_sub, u_sub - reference_sub) * ufl.dx
    l2 = np.sqrt(comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(error)), MPI.SUM))
    assert l2 < 1e-13


@pytest.mark.parametrize("family", ["N1curl", "RT"])
@pytest.mark.parametrize("degree", [1, 2])
def test_submesh_roundtrip_codim0_vector(tmp_path, backend, family, degree):
    """H(div)/H(curl) on a co-dimension 0 submesh, where dof transformations apply."""
    comm = MPI.COMM_WORLD
    folder = comm.bcast(tmp_path, root=0)
    parent_file = folder / f"parent{SUFFIX[backend]}"

    mesh, submesh, cell_map = _make_submesh(comm, codim=0)
    gdim = submesh.geometry.dim

    def f(x):
        return _reference(x, vector=True, gdim=gdim)

    V = dolfinx.fem.functionspace(submesh, (family, degree))
    u = dolfinx.fem.Function(V, name="u")
    u.interpolate(f)

    io4dolfinx.write_mesh(parent_file, mesh, backend=backend)
    io4dolfinx.write_submesh(
        parent_file, submesh, mesh, cell_map, mesh_name="wall", backend=backend
    )
    io4dolfinx.write_function(
        parent_file, u, time=0.0, mode=io4dolfinx.FileMode.append, mesh_name="wall", backend=backend
    )
    del mesh, submesh, cell_map, u

    parent = io4dolfinx.read_mesh(parent_file, comm, backend=backend)
    stored = io4dolfinx.read_mesh(parent_file, comm, mesh_name="wall", backend=backend)
    V_stored = dolfinx.fem.functionspace(stored, (family, degree))
    u_stored = dolfinx.fem.Function(V_stored, name="u")
    io4dolfinx.read_function(
        parent_file, u_stored, time=0.0, name="u", mesh_name="wall", backend=backend
    )
    reference = dolfinx.fem.Function(V_stored)
    reference.interpolate(f)
    assert _max_error(u_stored, reference) < 1e-13

    checkpoint = io4dolfinx.read_submesh(parent_file, parent, mesh_name="wall", backend=backend)
    V_sub = dolfinx.fem.functionspace(checkpoint.submesh, (family, degree))
    u_sub = dolfinx.fem.Function(V_sub)
    io4dolfinx.transfer_submesh_function(u_stored, u_sub, checkpoint.stored_cells)
    reference_sub = dolfinx.fem.Function(V_sub)
    reference_sub.interpolate(f)
    assert _max_error(u_sub, reference_sub) < 1e-13


@pytest.mark.parametrize("family", ["N1curl", "RT"])
def test_manifold_hcurl_roundtrips_but_transfer_raises(tmp_path, backend, family):
    """H(div)/H(curl) on a manifold submesh: storable, but not reattachable.

    Writing and reading the standalone submesh is exact for these spaces, and
    that is checked here so the capability is not lost by accident.

    The transfer, though, is refused. DOLFINx cannot reconcile the reference and
    physical value sizes of these families when ``tdim < gdim``, and its
    interpolation does not report the problem -- it returns values that are
    simply wrong (the L2 norm of an N1curl field measured 9.096 before and 4.904
    after). A loud refusal is the only safe behaviour.
    """
    comm = MPI.COMM_WORLD
    folder = comm.bcast(tmp_path, root=0)
    parent_file = folder / f"parent{SUFFIX[backend]}"

    mesh, submesh, cell_map = _make_submesh(comm, codim=1)
    V = dolfinx.fem.functionspace(submesh, (family, 1))
    u = dolfinx.fem.Function(V, name="u")
    # These spaces cannot be interpolated into on a manifold, so fill by global
    # dof index instead: reproducible no matter how the mesh is partitioned.
    imap = V.dofmap.index_map
    num_owned = imap.size_local * V.dofmap.index_map_bs
    global_index = imap.local_to_global(np.arange(imap.size_local, dtype=np.int32))
    u.x.array[:num_owned] = np.sin(0.37 * global_index.astype(np.float64) + 0.11)
    u.x.scatter_forward()
    norm_written = comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(u, u) * ufl.dx)), MPI.SUM
    )

    io4dolfinx.write_mesh(parent_file, mesh, backend=backend)
    io4dolfinx.write_submesh(
        parent_file, submesh, mesh, cell_map, mesh_name="wall", backend=backend
    )
    io4dolfinx.write_function(
        parent_file, u, time=0.0, mode=io4dolfinx.FileMode.append, mesh_name="wall", backend=backend
    )
    del mesh, submesh, cell_map

    parent = io4dolfinx.read_mesh(parent_file, comm, backend=backend)
    stored = io4dolfinx.read_mesh(parent_file, comm, mesh_name="wall", backend=backend)
    u_stored = dolfinx.fem.Function(dolfinx.fem.functionspace(stored, (family, 1)), name="u")
    io4dolfinx.read_function(
        parent_file, u_stored, time=0.0, name="u", mesh_name="wall", backend=backend
    )

    # Steps 1-3 are exact even here.
    norm_read = comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(u_stored, u_stored) * ufl.dx)),
        MPI.SUM,
    )
    assert np.isclose(norm_read, norm_written, rtol=0, atol=1e-12)

    checkpoint = io4dolfinx.read_submesh(parent_file, parent, mesh_name="wall", backend=backend)
    u_sub = dolfinx.fem.Function(dolfinx.fem.functionspace(checkpoint.submesh, (family, 1)))
    with pytest.raises(NotImplementedError, match="not supported"):
        io4dolfinx.transfer_submesh_function(u_stored, u_sub, checkpoint.stored_cells)


@pytest.mark.parametrize("codim", [0, 1])
def test_read_submesh_supports_mixed_dimensional_assembly(tmp_path, backend, codim):
    """The re-derived submesh carries entity maps usable against its parent.

    This is the whole reason to re-derive rather than stop at the standalone
    submesh, and it is the only cross-mesh assembly that is well founded here:
    the submesh really is derived from this parent.
    """
    comm = MPI.COMM_WORLD
    folder = comm.bcast(tmp_path, root=0)
    parent_file = folder / f"parent{SUFFIX[backend]}"

    mesh, submesh, cell_map = _make_submesh(comm, codim)
    io4dolfinx.write_mesh(parent_file, mesh, backend=backend)
    io4dolfinx.write_submesh(
        parent_file, submesh, mesh, cell_map, mesh_name="wall", backend=backend
    )
    del mesh, submesh, cell_map

    parent = io4dolfinx.read_mesh(parent_file, comm, backend=backend)
    checkpoint = io4dolfinx.read_submesh(parent_file, parent, mesh_name="wall", backend=backend)

    V_sub = dolfinx.fem.functionspace(checkpoint.submesh, ("Lagrange", 1))
    u_sub = dolfinx.fem.Function(V_sub)
    u_sub.x.array[:] = 1.0

    dim = checkpoint.submesh.topology.dim
    measure = ufl.dx if codim == 0 else ufl.ds
    form = dolfinx.fem.form(u_sub * measure(domain=parent), entity_maps=[checkpoint.cell_map])
    area = comm.allreduce(dolfinx.fem.assemble_scalar(form), MPI.SUM)
    expected = 0.5 if (codim == 0 and dim == 3) else 1.0
    assert np.isclose(area, expected)


def test_submesh_with_empty_ranks(tmp_path, backend):
    """A submesh small enough that some process owns none of it."""
    comm = MPI.COMM_WORLD
    folder = comm.bcast(tmp_path, root=0)
    parent_file = folder / f"parent{SUFFIX[backend]}"

    mesh = dolfinx.mesh.create_unit_cube(
        comm, 6, 6, 6, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
    )
    dim = mesh.topology.dim - 1
    mesh.topology.create_entities(dim)
    # A single sliver of the x=0 face, so on several ranks this is empty.
    entities = dolfinx.mesh.locate_entities(
        mesh, dim, lambda x: np.isclose(x[0], 0.0) & (x[1] <= 1.0 / 6.0 + 1e-12)
    )
    submesh, cell_map, _, _ = dolfinx.mesh.create_submesh(mesh, dim, entities)

    def f(x):
        return x[1] - 0.5 * x[2]

    V = dolfinx.fem.functionspace(submesh, ("Lagrange", 2))
    u = dolfinx.fem.Function(V, name="u")
    u.interpolate(f)

    io4dolfinx.write_mesh(parent_file, mesh, backend=backend)
    io4dolfinx.write_submesh(
        parent_file, submesh, mesh, cell_map, mesh_name="strip", backend=backend
    )
    io4dolfinx.write_function(
        parent_file,
        u,
        time=0.0,
        mode=io4dolfinx.FileMode.append,
        mesh_name="strip",
        backend=backend,
    )
    del mesh, submesh, cell_map, u

    parent = io4dolfinx.read_mesh(parent_file, comm, backend=backend)
    stored = io4dolfinx.read_mesh(parent_file, comm, mesh_name="strip", backend=backend)
    V_stored = dolfinx.fem.functionspace(stored, ("Lagrange", 2))
    u_stored = dolfinx.fem.Function(V_stored, name="u")
    io4dolfinx.read_function(
        parent_file, u_stored, time=0.0, name="u", mesh_name="strip", backend=backend
    )

    checkpoint = io4dolfinx.read_submesh(parent_file, parent, mesh_name="strip", backend=backend)
    V_sub = dolfinx.fem.functionspace(checkpoint.submesh, ("Lagrange", 2))
    u_sub = dolfinx.fem.Function(V_sub)
    io4dolfinx.transfer_submesh_function(u_stored, u_sub, checkpoint.stored_cells)

    reference = dolfinx.fem.Function(V_sub)
    reference.interpolate(f)
    assert _max_error(u_sub, reference) < 1e-13


def test_named_meshes_do_not_collide(tmp_path, backend):
    """Two meshes in one file keep their own topology, geometry and functions."""
    comm = MPI.COMM_WORLD
    folder = comm.bcast(tmp_path, root=0)
    path = folder / f"two{SUFFIX[backend]}"

    square = dolfinx.mesh.create_unit_square(comm, 4, 4)
    cube = dolfinx.mesh.create_unit_cube(comm, 3, 3, 3)
    io4dolfinx.write_mesh(path, square, backend=backend)
    io4dolfinx.write_mesh(
        path, cube, mesh_name="cube", mode=io4dolfinx.FileMode.append, backend=backend
    )

    u_square = dolfinx.fem.Function(dolfinx.fem.functionspace(square, ("Lagrange", 1)), name="u")
    u_square.interpolate(lambda x: x[0] + 2.0 * x[1])
    u_cube = dolfinx.fem.Function(dolfinx.fem.functionspace(cube, ("Lagrange", 1)), name="u")
    u_cube.interpolate(lambda x: 3.0 * x[2])
    io4dolfinx.write_function(path, u_square, time=0.0, backend=backend)
    io4dolfinx.write_function(path, u_cube, time=0.0, mesh_name="cube", backend=backend)
    del square, cube, u_square, u_cube

    read_square = io4dolfinx.read_mesh(path, comm, backend=backend)
    read_cube = io4dolfinx.read_mesh(path, comm, mesh_name="cube", backend=backend)
    assert read_square.topology.dim == 2
    assert read_cube.topology.dim == 3

    v_square = dolfinx.fem.Function(
        dolfinx.fem.functionspace(read_square, ("Lagrange", 1)), name="u"
    )
    io4dolfinx.read_function(path, v_square, time=0.0, name="u", backend=backend)
    ref_square = dolfinx.fem.Function(v_square.function_space)
    ref_square.interpolate(lambda x: x[0] + 2.0 * x[1])
    assert _max_error(v_square, ref_square) < 1e-13

    v_cube = dolfinx.fem.Function(dolfinx.fem.functionspace(read_cube, ("Lagrange", 1)), name="u")
    io4dolfinx.read_function(path, v_cube, time=0.0, name="u", mesh_name="cube", backend=backend)
    ref_cube = dolfinx.fem.Function(v_cube.function_space)
    ref_cube.interpolate(lambda x: 3.0 * x[2])
    assert _max_error(v_cube, ref_cube) < 1e-13


@pytest.mark.parametrize("kind", ["point", "cell"])
def test_submesh_point_and_cell_data(tmp_path, kind):
    """Visualisation data attached to a named submesh in a multi-block file.

    ``read_point_data`` and ``read_cell_data`` rebuild their space from the
    mesh's own coordinate element, so they work on a submesh once the mesh name
    reaches the backend.
    """
    pytest.importorskip("h5py")
    comm = MPI.COMM_WORLD
    folder = comm.bcast(tmp_path, root=0)
    path = folder / "vis.vtkhdf"

    mesh = dolfinx.mesh.create_unit_cube(
        comm, 4, 4, 4, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
    )
    fdim = mesh.topology.dim - 1
    mesh.topology.create_entities(fdim)
    facets = dolfinx.mesh.locate_entities(mesh, fdim, lambda x: np.isclose(x[0], 0.0))
    submesh = dolfinx.mesh.create_submesh(mesh, fdim, facets)[0]

    def f(x):
        return np.sin(2 * x[1]) + x[2]

    # Point data goes through the coordinate element, cell data through DG-0.
    element = ("Lagrange", 1) if kind == "point" else ("DG", 0)
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(submesh, element), name="u")
    u.interpolate(f)

    io4dolfinx.write_mesh(path, mesh, backend="vtkhdf")
    io4dolfinx.write_mesh(
        path, submesh, mesh_name="wall", mode=io4dolfinx.FileMode.append, backend="vtkhdf"
    )
    writer = io4dolfinx.write_point_data if kind == "point" else io4dolfinx.write_cell_data
    writer(
        path,
        u,
        time=0.0,
        mode=io4dolfinx.FileMode.append,
        backend_args=None,
        backend="vtkhdf",
        mesh_name="wall",
    )
    del mesh, submesh, u

    stored = io4dolfinx.read_mesh(path, comm, mesh_name="wall", backend="vtkhdf", time=0.0)
    reader = io4dolfinx.read_point_data if kind == "point" else io4dolfinx.read_cell_data
    read_back = reader(path, "u", stored, time=0.0, backend="vtkhdf", mesh_name="wall")

    reference = dolfinx.fem.Function(read_back.function_space)
    reference.interpolate(f)
    assert _max_error(read_back, reference) < 1e-13


def test_submesh_with_named_parent(tmp_path, backend):
    """Neither mesh has to use the default name."""
    comm = MPI.COMM_WORLD
    folder = comm.bcast(tmp_path, root=0)
    path = folder / f"named{SUFFIX[backend]}"

    mesh, submesh, cell_map = _make_submesh(comm, codim=1)

    def f(x):
        return np.sin(2 * x[1]) + x[2]

    u = dolfinx.fem.Function(dolfinx.fem.functionspace(submesh, ("Lagrange", 2)), name="u")
    u.interpolate(f)

    io4dolfinx.write_mesh(path, mesh, mesh_name="bulk", backend=backend)
    io4dolfinx.write_submesh(
        path,
        submesh,
        mesh,
        cell_map,
        mesh_name="wall",
        parent_mesh_name="bulk",
        backend=backend,
    )
    io4dolfinx.write_function(
        path, u, time=0.0, mode=io4dolfinx.FileMode.append, mesh_name="wall", backend=backend
    )
    del mesh, submesh, cell_map, u

    parent = io4dolfinx.read_mesh(path, comm, mesh_name="bulk", backend=backend)
    stored = io4dolfinx.read_mesh(path, comm, mesh_name="wall", backend=backend)
    u_stored = dolfinx.fem.Function(dolfinx.fem.functionspace(stored, ("Lagrange", 2)), name="u")
    io4dolfinx.read_function(path, u_stored, time=0.0, name="u", mesh_name="wall", backend=backend)

    checkpoint = io4dolfinx.read_submesh(
        path, parent, mesh_name="wall", parent_mesh_name="bulk", backend=backend
    )
    V_sub = dolfinx.fem.functionspace(checkpoint.submesh, ("Lagrange", 2))
    u_sub = dolfinx.fem.Function(V_sub)
    io4dolfinx.transfer_submesh_function(u_stored, u_sub, checkpoint.stored_cells)

    reference = dolfinx.fem.Function(V_sub)
    reference.interpolate(f)
    assert _max_error(u_sub, reference) < 1e-13
