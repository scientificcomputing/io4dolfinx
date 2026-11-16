import typing
from collections import ChainMap
from collections.abc import Sequence

from mpi4py import MPI

import basix.ufl
import dolfinx
import dolfinx.io.utils
import fenicsx_compat
import ipyparallel as ipp
import numpy as np
import numpy.typing
import numpy.typing as npt
import pytest
import ufl

import io4dolfinx


def find_backends():
    backends = []
    try:
        import adios2

        if adios2.is_built_with_mpi:
            backends.append("adios2")
    except ModuleNotFoundError:
        pass
    try:
        import h5py

        if h5py.get_config().mpi:
            backends.append("h5py")
        # Opens files without MPI even if h5py has it, so runs everywhere
        backends.append("h5py-serial")
    except ModuleNotFoundError:
        pass
    return backends


@pytest.fixture(params=find_backends(), scope="function")
def backend(request):
    yield request.param


@pytest.fixture
def tmp_path(tmp_path):
    """The temporary directory of rank 0, shared by all ranks of ``MPI.COMM_WORLD``.

    Under MPI, pytest gives each rank its own directory, unless the ranks share
    ``--basetemp`` as in Debian's CI. Sharing it in every run makes tests that write
    the same file from several ranks fail here too (issue #31).
    """
    return MPI.COMM_WORLD.bcast(tmp_path, root=0)


@pytest.fixture
def shared_tmp_path(tmp_path):
    """The temporary directory of rank 0, shared by all ranks of ``MPI.COMM_WORLD``."""
    return tmp_path


@pytest.fixture(scope="module")
def cluster():
    cluster = ipp.Cluster(engines="mpi", n=2)
    rc = cluster.start_and_connect_sync()
    rc.wait_for_engines(n=2)
    yield rc
    cluster.stop_cluster_sync()


@pytest.fixture(scope="function")
def write_function(tmp_path):
    def _write_function(
        mesh,
        el,
        f,
        dtype,
        backend: typing.Literal["adios2", "h5py"],
        name="uh",
        append: bool = False,
    ) -> str:
        V = dolfinx.fem.functionspace(mesh, el)
        uh = dolfinx.fem.Function(V, dtype=dtype)
        uh.interpolate(f)
        uh.name = name
        el_hash = (
            io4dolfinx.utils.element_signature(V)
            .replace(" ", "")
            .replace(",", "")
            .replace("(", "")
            .replace(")", "")
            .replace("[", "")
            .replace("]", "")
        )
        # Consistent tmp dir across processes
        f_path = MPI.COMM_WORLD.bcast(tmp_path, root=0)
        file_hash = f"{el_hash}_{np.dtype(dtype).name}"
        if backend == "adios2":
            suffix = ".bp"
        else:
            suffix = ".h5"

        filename = (f_path / f"mesh_{file_hash}").with_suffix(suffix)
        if mesh.comm.size != 1:
            if not append:
                io4dolfinx.write_mesh(filename, mesh, backend=backend)
            io4dolfinx.write_function(filename, uh, time=0.0, backend=backend)
        else:
            if MPI.COMM_WORLD.rank == 0:
                if not append:
                    io4dolfinx.write_mesh(filename, mesh, backend=backend)
                io4dolfinx.write_function(filename, uh, time=0.0, backend=backend)

        return filename

    return _write_function


@pytest.fixture(scope="function")
def read_function():
    def _read_function(
        comm, el, f, path, dtype, backend: typing.Literal["adios2", "h5py"], name="uh"
    ):
        mesh = io4dolfinx.read_mesh(
            path,
            comm,
            ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
            backend=backend,
        )
        V = dolfinx.fem.functionspace(mesh, el)
        v = dolfinx.fem.Function(V, dtype=dtype)
        v.name = name
        io4dolfinx.read_function(path, v, backend=backend)
        v_ex = dolfinx.fem.Function(V, dtype=dtype)
        v_ex.interpolate(f)

        res = np.finfo(dtype).resolution
        np.testing.assert_allclose(v.x.array, v_ex.x.array, atol=10 * res, rtol=10 * res)

    return _read_function


@pytest.fixture(scope="function")
def get_dtype():
    def _get_dtype(in_dtype: np.dtype, is_complex: bool):
        dtype: numpy.typing.DTypeLike
        if in_dtype == np.float32:
            if is_complex:
                dtype = np.complex64
            else:
                dtype = np.float32
        elif in_dtype == np.float64:
            if is_complex:
                dtype = np.complex128
            else:
                dtype = np.float64
        else:
            raise ValueError("Unsuported dtype")
        return dtype

    return _get_dtype


@pytest.fixture(scope="function")
def write_function_time_dep(tmp_path):
    def _write_function_time_dep(
        mesh, el, f0, f1, t0, t1, dtype, backend: typing.Literal["adios2", "h5py"]
    ) -> str:
        V = dolfinx.fem.functionspace(mesh, el)
        uh = dolfinx.fem.Function(V, dtype=dtype)
        uh.interpolate(f0)
        el_hash = (
            io4dolfinx.utils.element_signature(V)
            .replace(" ", "")
            .replace(",", "")
            .replace("(", "")
            .replace(")", "")
            .replace("[", "")
            .replace("]", "")
        )
        file_hash = f"{el_hash}_{np.dtype(dtype).name}"
        # Consistent tmp dir across processes
        f_path = MPI.COMM_WORLD.bcast(tmp_path, root=0)
        if backend == "adios2":
            suffix = ".bp"
        else:
            suffix = ".h5"
        filename = (f_path / f"mesh_{file_hash}").with_suffix(suffix)
        if mesh.comm.size != 1:
            io4dolfinx.write_mesh(filename, mesh, backend=backend)
            io4dolfinx.write_function(filename, uh, time=t0, backend=backend)
            uh.interpolate(f1)
            io4dolfinx.write_function(filename, uh, time=t1, backend=backend)

        else:
            if MPI.COMM_WORLD.rank == 0:
                io4dolfinx.write_mesh(filename, mesh, backend=backend)
                io4dolfinx.write_function(filename, uh, time=t0, backend=backend)
                uh.interpolate(f1)
                io4dolfinx.write_function(filename, uh, time=t1, backend=backend)

        return filename

    return _write_function_time_dep


@pytest.fixture(scope="function")
def read_function_time_dep():
    def _read_function_time_dep(
        comm, el, f0, f1, t0, t1, path, dtype, backend: typing.Literal["adios2", "h5py"]
    ):
        mesh = io4dolfinx.read_mesh(
            path,
            comm,
            ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
            backend=backend,
        )
        V = dolfinx.fem.functionspace(mesh, el)
        v = dolfinx.fem.Function(V, dtype=dtype)

        io4dolfinx.read_function(path, v, time=t1, backend=backend)
        v_ex = dolfinx.fem.Function(V, dtype=dtype)
        v_ex.interpolate(f1)

        res = np.finfo(dtype).resolution
        assert np.allclose(v.x.array, v_ex.x.array, atol=10 * res, rtol=10 * res)

        io4dolfinx.read_function(path, v, time=t0, backend=backend)
        v_ex = dolfinx.fem.Function(V, dtype=dtype)
        v_ex.interpolate(f0)

        res = np.finfo(dtype).resolution
        assert np.allclose(v.x.array, v_ex.x.array, atol=10 * res, rtol=10 * res)

    return _read_function_time_dep


def _generate_reference_map(
    mesh: dolfinx.mesh.Mesh,
    meshtag: dolfinx.mesh.MeshTags,
    comm: MPI.Comm,
    root: int,
) -> typing.Optional[dict[str, tuple[int, npt.NDArray]]]:
    """
    Helper function to generate map from meshtag value to its corresponding index and midpoint.

    Args:
        mesh: The mesh
        meshtag: The associated meshtag
        comm: MPI communicator to gather the map from all processes with
        root (int): Rank to store data on
    Returns:
        Root rank returns the map, all other ranks return None
    """
    mesh.topology.create_connectivity(meshtag.dim, mesh.topology.dim)
    midpoints = dolfinx.mesh.compute_midpoints(mesh, meshtag.dim, meshtag.indices)
    e_map = mesh.topology.index_map(meshtag.dim)
    value_to_midpoint = {}
    for index, value in zip(meshtag.indices, meshtag.values):
        value_to_midpoint[value] = (
            int(e_map.local_range[0] + index),
            midpoints[index],
        )
    global_map = comm.gather(value_to_midpoint, root=root)
    if comm.rank == root:
        return dict(ChainMap(*global_map))  # type: ignore
    return None


@pytest.fixture
def generate_reference_map():
    return _generate_reference_map


# Helpers for tests that write files of other formats from DOLFINx meshes. Each format's node
# and side numbering stays in its test, written from the format's specification.


def _serendipity_mesh(mesh: dolfinx.mesh.Mesh) -> dolfinx.mesh.Mesh:
    """The mesh with the same topology and degree-2 serendipity geometry, from the nodes of a
    serendipity space with point evaluations, the coordinate element DOLFINx uses for these
    meshes. Skips the calling test on DOLFINx releases, which do not have it."""
    if not hasattr(dolfinx.io.utils, "_serendipity_coordinate_element"):
        pytest.skip("Serendipity geometry requires the main branch of DOLFINx")

    gdim = mesh.geometry.dim
    element = dolfinx.io.utils._serendipity_coordinate_element(mesh.topology.cell_type, np.float64)
    V = dolfinx.fem.functionspace(mesh, element)
    index_map = V.dofmap.index_map
    num_nodes = index_map.size_local + index_map.num_ghosts
    geometry = dolfinx.cpp.mesh.Geometry_float64(
        fenicsx_compat.unwrap_index_map(index_map),
        V.dofmap.list,
        dolfinx.cpp.fem.CoordinateElement_float64(element.basix_element._e),
        V.tabulate_dof_coordinates()[:, :gdim],
        index_map.local_to_global(np.arange(num_nodes, dtype=np.int32)),
    )
    return dolfinx.mesh.Mesh(
        dolfinx.cpp.mesh.Mesh_float64(mesh.comm, mesh.topology._cpp_object, geometry),
        ufl.Mesh(basix.ufl.blocked_element(element, shape=(gdim,))),
    )


def _unit_mesh(
    cell_type: str | dolfinx.mesh.CellType,
    degree: int = 1,
    serendipity: bool = False,
    n: int | Sequence[int] | None = None,
) -> dolfinx.mesh.Mesh:
    """A mesh of the unit interval, square or cube on one process.

    Args:
        cell_type: The cell type.
        degree: Degree of the geometry.
        serendipity: Use degree-2 serendipity geometry instead of Lagrange.
        n: Number of cells in each direction, or per direction. Defaults to 4 (2 for cubes),
            so that x = 1/2 is on cell boundaries.

    Returns:
        The mesh.
    """
    comm = MPI.COMM_SELF
    ct = dolfinx.mesh.to_type(cell_type) if isinstance(cell_type, str) else cell_type
    tdim = dolfinx.mesh.cell_dim(ct)
    if n is None:
        n = 2 if tdim == 3 else 4
    counts = [n] * tdim if isinstance(n, int) else list(n)
    if tdim == 1:
        mesh = dolfinx.mesh.create_unit_interval(comm, counts[0])
    elif tdim == 2:
        mesh = dolfinx.mesh.create_unit_square(comm, counts[0], counts[1], cell_type=ct)
    else:
        mesh = dolfinx.mesh.create_unit_cube(comm, counts[0], counts[1], counts[2], cell_type=ct)
    if serendipity:
        assert degree == 2, "Only degree-2 serendipity geometry exists"
        return _serendipity_mesh(mesh)
    if degree > 1:
        return fenicsx_compat.reconstruct_mesh(mesh, degree)
    return mesh


@pytest.fixture(scope="session")
def unit_mesh():
    return _unit_mesh


def _curve(mesh: dolfinx.mesh.Mesh):
    """Move the nodes of a mesh by a quadratic map, so that misplaced nodes change its
    geometry. Second-order Lagrange and serendipity geometry represent the map exactly."""
    x = mesh.geometry.x[:, : mesh.topology.dim]
    x[:] = x + 0.1 * np.roll(x, -1, axis=1) ** 2


@pytest.fixture(scope="session")
def curve():
    return _curve


def _boundary_tags(mesh: dolfinx.mesh.Mesh) -> dolfinx.mesh.MeshTags:
    """The boundary facets of a unit box mesh, tagged 1 + 2i on x_i = 0 and 2 + 2i on
    x_i = 1."""
    tdim = mesh.topology.dim
    facets, values = [], []
    for i in range(tdim):
        for j, plane in enumerate([0, 1]):
            marked = dolfinx.mesh.locate_entities_boundary(
                mesh, tdim - 1, lambda x: np.isclose(x[i], plane)
            )
            facets.append(marked)
            values.append(np.full_like(marked, 1 + 2 * i + j))
    order = np.argsort(np.hstack(facets))
    return dolfinx.mesh.meshtags(mesh, tdim - 1, np.hstack(facets)[order], np.hstack(values)[order])


@pytest.fixture(scope="session")
def boundary_tags():
    return _boundary_tags


def _node_order(
    cell_type: str, degree: int, vertices: Sequence[int], nodes: Sequence[Sequence[int]] = ()
) -> list[int]:
    """The node of a DOLFINx cell at each node of an element of another format.

    Args:
        cell_type: The cell type.
        degree: Degree of the DOLFINx geometry.
        vertices: The DOLFINx vertex at each vertex of the element.
        nodes: The other nodes of the element, each at the centre of the listed vertices of
            the element (0-based). Serendipity cells have the vertex and edge nodes of
            second-order Lagrange cells, numbered alike, so their nodes end after the edges.

    Returns:
        The DOLFINx node at each node of the element.
    """
    points = basix.ufl.element("Lagrange", cell_type, degree).basix_element.points
    order = list(vertices)
    for node in nodes:
        centre = points[[vertices[v] for v in node]].mean(axis=0)
        order.append(int(np.flatnonzero(np.isclose(points, centre).all(axis=1))[0]))
    assert sorted(order) == list(range(len(order)))
    return order


@pytest.fixture(scope="session")
def node_order():
    return _node_order


def _facet_cells(
    mesh: dolfinx.mesh.Mesh, facet_tags: dolfinx.mesh.MeshTags
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """A cell of each tagged facet, the facet's local index in it, and the facet's value, as
    formats that define facets by their cell and side need."""
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    mesh.topology.create_connectivity(tdim, tdim - 1)
    f_to_c = mesh.topology.connectivity(tdim - 1, tdim)
    c_to_f = mesh.topology.connectivity(tdim, tdim - 1)
    cells = np.array([f_to_c.links(f)[0] for f in facet_tags.indices], dtype=np.int32)
    local = np.array(
        [np.flatnonzero(c_to_f.links(c) == f)[0] for c, f in zip(cells, facet_tags.indices)],
        dtype=np.int32,
    )
    return cells, local, facet_tags.values.astype(np.int32)


@pytest.fixture(scope="session")
def facet_cells():
    return _facet_cells


def _integral(
    mesh: dolfinx.mesh.Mesh,
    expr: ufl.core.expr.Expr | float,
    measure: str,
    tags: dolfinx.mesh.MeshTags | None = None,
    tag: int | None = None,
    dtype: npt.DTypeLike = dolfinx.default_real_type,
) -> float:
    """The integral of an expression over a mesh, summed over all processes.

    Args:
        mesh: The mesh.
        expr: The expression to integrate.
        measure: Name of the measure, e.g. ``"dx"``, ``"ds"``, ``"dS"`` or ``"dP"``.
        tags: Mesh tags of the measure.
        tag: Value in ``tags`` to integrate over, all entities if ``None``.
        dtype: Scalar type of the form.

    Returns:
        The integral.
    """
    m = ufl.Measure(measure, domain=mesh, subdomain_data=tags)
    form = dolfinx.fem.form(expr * (m(tag) if tag is not None else m), dtype=dtype)
    return mesh.comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)


@pytest.fixture(scope="session")
def integral():
    return _integral


def _tag_values(tags: dolfinx.mesh.MeshTags, comm: MPI.Comm) -> list[int]:
    return sorted(set().union(*comm.allgather(set(np.unique(tags.values).tolist()))))


def _integrals(
    mesh: dolfinx.mesh.Mesh,
    cell_tags: dolfinx.mesh.MeshTags | None = None,
    facet_tags: dolfinx.mesh.MeshTags | None = None,
    fields: dict[str, dolfinx.fem.Function] | None = None,
) -> dict[str, float]:
    """Integrals of a weight over the mesh, each tagged set of cells and facets and the
    boundary, and of the weighted fields, summed over all processes."""
    x = ufl.SpatialCoordinate(mesh)
    weight = 1 + ufl.inner(x, x)
    values = {"volume": _integral(mesh, 1, "dx"), "moment": _integral(mesh, weight, "dx")}
    if cell_tags is not None:
        for value in _tag_values(cell_tags, mesh.comm):
            values[f"cells {value}"] = _integral(mesh, weight, "dx", cell_tags, value)
    if facet_tags is not None:
        values["boundary"] = _integral(mesh, weight, "ds")
        for value in _tag_values(facet_tags, mesh.comm):
            values[f"facets {value}"] = _integral(mesh, weight, "ds", facet_tags, value)
    for name, u in (fields or {}).items():
        values[name] = _integral(mesh, u * weight, "dx")
    return values


@pytest.fixture(scope="session")
def integrals():
    return _integrals
