from mpi4py import MPI

import basix
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
    io4dolfinx.transfer_submesh_function(u_stored, u_sub, checkpoint.post_code)

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
    io4dolfinx.transfer_submesh_function(u_stored, u_sub, checkpoint.post_code)
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
        io4dolfinx.transfer_submesh_function(u_stored, u_sub, checkpoint.post_code)


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


@pytest.mark.parametrize("codim", [0, 1])
@pytest.mark.parametrize("degree", [0, 1, 2])
def test_transfer_of_discontinuous_field(tmp_path, backend, codim, degree):
    """A DG field that genuinely jumps inside the submesh survives the transfer.

    The DG spaces tested elsewhere carry a smooth function, which cannot tell
    whether a value came from the right cell. Here the field is discontinuous
    across a plane of the mesh, and for degree > 0 the interpolation points sit
    on the vertices and facets that straddle it -- the places where taking the
    value from the neighbouring cell would go unnoticed by a continuous field.
    Each point is evaluated in the cell its post code names, so the side is never
    in doubt.
    """
    comm = MPI.COMM_WORLD
    folder = comm.bcast(tmp_path, root=0)
    path = folder / f"jump{SUFFIX[backend]}"

    def jumpy(x):
        # Discontinuous across x = 0.25, which is a plane of the 4x4x4 mesh.
        return np.where(x[0] < 0.25, 3.0, -1.0)

    element = ("DG", degree)
    mesh, submesh, cell_map = _make_submesh(comm, codim)
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(submesh, element), name="u")
    u.interpolate(jumpy)

    io4dolfinx.write_mesh(path, mesh, backend=backend)
    io4dolfinx.write_submesh(path, submesh, mesh, cell_map, mesh_name="wall", backend=backend)
    io4dolfinx.write_function(
        path, u, time=0.0, mode=io4dolfinx.FileMode.append, mesh_name="wall", backend=backend
    )
    del mesh, submesh, cell_map, u

    parent = io4dolfinx.read_mesh(path, comm, backend=backend)
    stored = io4dolfinx.read_mesh(path, comm, mesh_name="wall", backend=backend)
    u_stored = dolfinx.fem.Function(dolfinx.fem.functionspace(stored, element), name="u")
    io4dolfinx.read_function(path, u_stored, time=0.0, name="u", mesh_name="wall", backend=backend)

    checkpoint = io4dolfinx.read_submesh(path, parent, mesh_name="wall", backend=backend)
    V_sub = dolfinx.fem.functionspace(checkpoint.submesh, element)
    u_sub = dolfinx.fem.Function(V_sub, name="u")
    io4dolfinx.transfer_submesh_function(u_stored, u_sub, checkpoint.post_code)

    reference = dolfinx.fem.Function(V_sub)
    reference.interpolate(jumpy)
    assert _max_error(u_sub, reference) < 1e-13


def test_interface_submesh_assembly(tmp_path, backend):
    """A facet submesh on an interface, integrated against a parent field.

    Only quantities that do not depend on which side is ``"+"`` are asserted:
    the restrictions follow the facet-to-cell connectivity, so their assignment
    is not a property of the mesh and no checkpoint preserves it. It is not
    stable without IO either -- the same form on a freshly built mesh changes
    sign between one process and three. ``abs`` of the jump is invariant; a form
    that needs the *signed* jump has to pin the restrictions itself, e.g. by
    ordering the integration entities by cell marker
    (``scifem.compute_interface_data``).
    """
    comm = MPI.COMM_WORLD
    folder = comm.bcast(tmp_path, root=0)
    path = folder / f"iface{SUFFIX[backend]}"

    mesh = dolfinx.mesh.create_unit_cube(
        comm, 4, 4, 4, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
    )
    tdim = mesh.topology.dim
    mesh.topology.create_entities(tdim - 1)
    facets = dolfinx.mesh.locate_entities(mesh, tdim - 1, lambda x: np.isclose(x[0], 0.5))
    interface, cell_map, _, _ = dolfinx.mesh.create_submesh(mesh, tdim - 1, facets)

    io4dolfinx.write_mesh(path, mesh, backend=backend)
    io4dolfinx.write_submesh(
        path, interface, mesh, cell_map, mesh_name="interface", backend=backend
    )
    del mesh, interface, cell_map

    parent = io4dolfinx.read_mesh(path, comm, backend=backend)
    checkpoint = io4dolfinx.read_submesh(path, parent, mesh_name="interface", backend=backend)

    # A parent field with a jump of exactly 1 across the interface.
    k = dolfinx.fem.Function(dolfinx.fem.functionspace(parent, ("DG", 0)))
    midpoints = parent.geometry.x[parent.geometry.dofmaps[0]].mean(axis=1)
    k.x.array[: len(midpoints)] = np.where(midpoints[:, 0] < 0.5, 1.0, 2.0)
    k.x.scatter_forward()

    ptdim = parent.topology.dim
    parent.topology.create_entities(ptdim - 1)
    parent.topology.create_connectivity(ptdim - 1, ptdim)
    marked = dolfinx.mesh.locate_entities(parent, ptdim - 1, lambda x: np.isclose(x[0], 0.5))
    tags = dolfinx.mesh.meshtags(parent, ptdim - 1, marked, np.ones(len(marked), dtype=np.int32))
    dS = ufl.Measure("dS", domain=parent, subdomain_data=tags, subdomain_id=1)

    w = dolfinx.fem.Function(dolfinx.fem.functionspace(checkpoint.submesh, ("DG", 0)))
    w.x.array[:] = 1.0

    for expr, expected in [
        (w("+") * dS, 1.0),  # area of the interface
        (ufl.avg(k) * w("+") * dS, 1.5),  # restriction-symmetric
        (abs(ufl.jump(k)) * w("+") * dS, 1.0),  # restriction-invariant
    ]:
        form = dolfinx.fem.form(expr, entity_maps=[checkpoint.cell_map])
        value = comm.allreduce(dolfinx.fem.assemble_scalar(form), MPI.SUM)
        assert np.isclose(value, expected), (value, expected)


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
    io4dolfinx.transfer_submesh_function(u_stored, u_sub, checkpoint.post_code)

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


def test_multiple_submeshes_of_one_parent(tmp_path, backend):
    """Several submeshes of the same parent share one file without colliding.

    Two of them have the same co-dimension, so their post codes tag the same
    dimension of the parent, and one name is a prefix of the other -- which is
    what would break if tags were matched by prefix rather than by name.
    """
    comm = MPI.COMM_WORLD
    folder = comm.bcast(tmp_path, root=0)
    path = folder / f"many{SUFFIX[backend]}"

    def f(x):
        return np.sin(1.3 * x[0] + 0.2) + x[1] - 0.5 * x[2]

    mesh = dolfinx.mesh.create_unit_cube(
        comm, 4, 4, 4, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
    )
    tdim = mesh.topology.dim
    mesh.topology.create_entities(tdim - 1)

    # A cell submesh and two facet submeshes, on opposite faces.
    subs = {}
    for name, dim, locator in [
        ("half", tdim, lambda x: x[0] <= 0.5 + 1e-12),
        ("wall", tdim - 1, lambda x: np.isclose(x[0], 0.0)),
        ("wall2", tdim - 1, lambda x: np.isclose(x[0], 1.0)),
    ]:
        entities = dolfinx.mesh.locate_entities(mesh, dim, locator)
        submesh, cell_map, _, _ = dolfinx.mesh.create_submesh(mesh, dim, entities)
        subs[name] = (dim, submesh, cell_map)

    io4dolfinx.write_mesh(path, mesh, backend=backend)
    for name, (_, submesh, cell_map) in subs.items():
        u = dolfinx.fem.Function(dolfinx.fem.functionspace(submesh, ("Lagrange", 2)), name="u")
        u.interpolate(f)
        io4dolfinx.write_submesh(
            path,
            submesh,
            mesh,
            cell_map,
            mesh_name=name,
            mode=io4dolfinx.FileMode.append,
            backend=backend,
        )
        io4dolfinx.write_function(
            path, u, time=0.0, mode=io4dolfinx.FileMode.append, mesh_name=name, backend=backend
        )
    expected = {
        name: (dim, submesh.topology.index_map(dim).size_global)
        for name, (dim, submesh, _) in subs.items()
    }
    del mesh, subs

    parent = io4dolfinx.read_mesh(path, comm, backend=backend)
    for name, (dim, num_cells_global) in expected.items():
        stored = io4dolfinx.read_mesh(path, comm, mesh_name=name, backend=backend)
        assert stored.topology.dim == dim
        u_stored = dolfinx.fem.Function(
            dolfinx.fem.functionspace(stored, ("Lagrange", 2)), name="u"
        )
        io4dolfinx.read_function(
            path, u_stored, time=0.0, name="u", mesh_name=name, backend=backend
        )

        checkpoint = io4dolfinx.read_submesh(path, parent, mesh_name=name, backend=backend)
        assert checkpoint.submesh.topology.dim == dim
        assert checkpoint.submesh.topology.index_map(dim).size_global == num_cells_global

        # Each submesh must come back as itself: the two facet submeshes would be
        # indistinguishable by size alone.
        if name == "wall":
            assert np.all(checkpoint.submesh.geometry.x[:, 0] < 1e-12)
        elif name == "wall2":
            assert np.all(checkpoint.submesh.geometry.x[:, 0] > 1.0 - 1e-12)

        V_sub = dolfinx.fem.functionspace(checkpoint.submesh, ("Lagrange", 2))
        u_sub = dolfinx.fem.Function(V_sub, name="u")
        io4dolfinx.transfer_submesh_function(u_stored, u_sub, checkpoint.post_code)
        reference = dolfinx.fem.Function(V_sub)
        reference.interpolate(f)
        assert _max_error(u_sub, reference) < 1e-13


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
    io4dolfinx.transfer_submesh_function(u_stored, u_sub, checkpoint.post_code)

    reference = dolfinx.fem.Function(V_sub)
    reference.interpolate(f)
    assert _max_error(u_sub, reference) < 1e-13


def _curved_mesh(comm, dim, degree):
    """A disk (dim 2) or ball (dim 3) with degree-``degree`` geometry.

    Built by lifting a straight-sided box to a higher-order coordinate element
    and then pushing the nodes onto the circle/sphere, so cell edges are genuine
    curves rather than chords -- which is the point: it puts the interior
    geometry nodes somewhere a first-order mesh could not represent.
    """
    # Degree > 2 has no default node placement, so name the variant.
    variant = basix.LagrangeVariant.gll_isaac if degree > 2 else basix.LagrangeVariant.unset
    if dim == 3:
        # At least three divisions per axis, so that a cut offset from the
        # symmetry plane still has whole cells on one side of it.
        base = dolfinx.mesh.create_box(
            comm,
            [np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])],
            [3, 3, 3],
            ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
        )
    else:
        base = dolfinx.mesh.create_rectangle(
            comm,
            [np.array([-1.0, -1.0]), np.array([1.0, 1.0])],
            [4, 4],
            ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
        )
    cmap = dolfinx.fem.coordinate_element(base.topology.cell_type, degree, int(variant))
    mesh = dolfinx.fem.interpolate_geometry(base, cmap)

    # Elliptical grid mapping: sends the box onto the ball, smoothly.
    x = mesh.geometry.x
    if dim == 3:
        a, b, c = x[:, 0].copy(), x[:, 1].copy(), x[:, 2].copy()
        x[:, 0] = a * np.sqrt(1 - b**2 / 2 - c**2 / 2 + b**2 * c**2 / 3)
        x[:, 1] = b * np.sqrt(1 - c**2 / 2 - a**2 / 2 + c**2 * a**2 / 3)
        x[:, 2] = c * np.sqrt(1 - a**2 / 2 - b**2 / 2 + a**2 * b**2 / 3)
    else:
        a, b = x[:, 0].copy(), x[:, 1].copy()
        x[:, 0] = a * np.sqrt(1 - b**2 / 2)
        x[:, 1] = b * np.sqrt(1 - a**2 / 2)
    return mesh


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("degree", [2, 4])
@pytest.mark.parametrize("codim", [0, 1])
def test_point_data_on_curved_submesh(tmp_path, dim, degree, codim):
    """Point data on a submesh of a curved, higher-order mesh.

    Point data lives on the geometry nodes, so a degree-4 mesh puts most of it on
    interior and edge nodes rather than vertices. The codim-1 case is the curved
    boundary itself -- a manifold submesh whose cells are genuinely curved.
    """
    pytest.importorskip("h5py")
    comm = MPI.COMM_WORLD
    folder = comm.bcast(tmp_path, root=0)
    path = folder / "curved.vtkhdf"

    mesh = _curved_mesh(comm, dim, degree)
    tdim = mesh.topology.dim
    sub_dim = tdim - codim
    mesh.topology.create_entities(sub_dim)
    if codim == 0:
        # Offset from the symmetry plane on purpose. A cut that grazes node
        # positions makes `locate_entities` disagree between a cell's owner and
        # the ranks ghosting it on a higher-order mesh, and `create_submesh`
        # rejects the result with "Index owner change detected". Measured at
        # np=3 on a degree-2 cube: 3 mismatches at `x[0] <= 0.0`, none at -0.1,
        # with or without the curving perturbation.
        entities = dolfinx.mesh.locate_entities(mesh, sub_dim, lambda x: x[0] <= -0.1)
    else:
        mesh.topology.create_connectivity(sub_dim, tdim)
        entities = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    submesh = dolfinx.mesh.create_submesh(mesh, sub_dim, entities)[0]
    assert submesh.geometry.cmaps[0].degree == degree

    def f(x):
        return np.sin(1.7 * x[0]) + 0.5 * x[1] - 0.3 * x[tdim - 1]

    V = io4dolfinx.readers.create_geometry_function_space(submesh, 1)
    u = dolfinx.fem.Function(V, name="u")
    u.interpolate(f)
    nodes_written = submesh.geometry.index_map().size_global
    variant_written = submesh.geometry.cmaps[0].variant

    io4dolfinx.write_mesh(path, submesh, mesh_name="curved", backend="vtkhdf")
    io4dolfinx.write_point_data(
        path,
        u,
        time=0.0,
        mode=io4dolfinx.FileMode.append,
        backend_args=None,
        backend="vtkhdf",
        mesh_name="curved",
    )
    del mesh, submesh, u

    stored = io4dolfinx.read_mesh(path, comm, mesh_name="curved", backend="vtkhdf", time=0.0)
    # The higher-order geometry must survive whole: degree, node placement and
    # node count, not just the coordinates.
    assert stored.geometry.cmaps[0].degree == degree
    assert stored.geometry.cmaps[0].variant == variant_written
    assert stored.geometry.index_map().size_global == nodes_written

    read_back = io4dolfinx.read_point_data(
        path, "u", stored, time=0.0, backend="vtkhdf", mesh_name="curved"
    )
    reference = dolfinx.fem.Function(read_back.function_space)
    reference.interpolate(f)
    assert _max_error(read_back, reference) < 1e-13


@pytest.mark.parametrize("degree", [2, 4])
@pytest.mark.parametrize("store", ["adios2", "h5py", "vtkhdf"])
def test_lagrange_variant_survives_mesh_roundtrip(tmp_path, store, degree):
    """A higher-order mesh keeps its node placement, not just its node count.

    Two coordinate elements of the same degree but different Lagrange variants
    put their nodes at different reference positions, so a checkpoint that drops
    the variant restores a mesh whose cells curve differently between the same
    node coordinates -- and does so silently, since every coordinate still
    round-trips exactly.

    ``vtkhdf`` is included deliberately. VTKHDF defines its higher-order cells as
    equispaced and has no slot for a variant, so io4dolfinx records one in an
    extra attribute; this is what pins that down.
    """
    pytest.importorskip("h5py")
    comm = MPI.COMM_WORLD
    folder = comm.bcast(tmp_path, root=0)
    suffix = {"adios2": ".bp", "h5py": ".h5", "vtkhdf": ".vtkhdf"}[store]
    path = folder / f"variant{suffix}"

    base = dolfinx.mesh.create_rectangle(
        comm, [np.array([-1.0, -1.0]), np.array([1.0, 1.0])], [4, 4]
    )
    variant = basix.LagrangeVariant.gll_isaac if degree > 2 else basix.LagrangeVariant.unset
    cmap = dolfinx.fem.coordinate_element(base.topology.cell_type, degree, int(variant))
    mesh = dolfinx.fem.interpolate_geometry(base, cmap)
    written = (mesh.geometry.cmaps[0].degree, mesh.geometry.cmaps[0].variant)

    io4dolfinx.write_mesh(path, mesh, backend=store)
    read_kwargs = {"time": 0.0} if store == "vtkhdf" else {}
    read_back = io4dolfinx.read_mesh(path, comm, backend=store, **read_kwargs)
    assert (read_back.geometry.cmaps[0].degree, read_back.geometry.cmaps[0].variant) == written
