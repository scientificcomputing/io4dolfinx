import urllib.request
from enum import Enum
from pathlib import Path

from mpi4py import MPI

import basix.ufl
import dolfinx
import fenicsx_compat
import numpy as np
import numpy.typing as npt
import pytest

import io4dolfinx

netCDF4 = pytest.importorskip("netCDF4")


class DownloadStatus(Enum):
    success = 1
    failed = -1
    no_connection = -2


def download_file_if_not_exists(
    url, filename, comm: MPI.Intracomm = MPI.COMM_WORLD, rank: int = 0
) -> DownloadStatus:
    status = DownloadStatus.failed
    if comm.rank == rank:
        if not filename.exists():
            try:
                urllib.request.urlretrieve(url, filename)
                status = DownloadStatus.success
            except urllib.error.URLError as e:
                if str(e) == "<urlopen error [Errno -3] Temporary failure in name resolution>":
                    status = DownloadStatus.no_connection
                else:
                    status = DownloadStatus.failed
        else:
            status = DownloadStatus.success
    status = comm.bcast(status, root=rank)
    comm.Barrier()
    return status


def test_read_mesh_and_cell_data(tmp_path):
    tmp_path = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    filename = tmp_path / "openmc_master_out_openmc0.e"
    url = "https://github.com/neams-th-coe/cardinal/blob/devel/test/tests/neutronics/feedback/single_level/gold/openmc_master_out_openmc0.e?raw=true"
    status = download_file_if_not_exists(url, filename)
    if status == DownloadStatus.no_connection:
        pytest.skip("No internet connection")
    mesh = io4dolfinx.read_mesh(filename, MPI.COMM_WORLD, backend="exodus")
    io4dolfinx.read_meshtags(filename, mesh, meshtag_name="cell", backend="exodus")
    io4dolfinx.read_meshtags(filename, mesh, meshtag_name="facet", backend="exodus")
    io4dolfinx.read_cell_data(
        filename, name="cell_temperature", mesh=mesh, backend="exodus", time=1.0
    )


def test_read_mesh_point_data(tmp_path):
    tmp_path = MPI.COMM_WORLD.bcast(tmp_path, root=0)

    filename = tmp_path / "openmc_master_out_openmc0.e"
    url = "https://github.com/idaholab/moose/blob/next/test/tests/kernels/2d_diffusion/gold/matdiffusion_out.e?raw=true"
    status = download_file_if_not_exists(url, filename)
    if status == DownloadStatus.no_connection:
        pytest.skip("No internet connection")

    mesh = io4dolfinx.read_mesh(filename, MPI.COMM_WORLD, backend="exodus")
    u = io4dolfinx.read_point_data(filename, name="u", mesh=mesh, backend="exodus", time=1.0)
    assert mesh.topology.index_map(mesh.topology.dim).size_global == 4
    assert mesh.geometry.index_map().size_global == 9
    assert np.isclose(mesh.comm.allreduce(np.max(u.x.array), op=MPI.MAX), 1.1140844375981802)


def test_read_second_order_mesh(tmp_path):
    tmp_path = MPI.COMM_WORLD.bcast(tmp_path, root=0)
    filename = tmp_path / "box-test_out_nek0.e"
    url = "https://github.com/neams-th-coe/cardinal/blob/devel/test/tests/deformation/simple-cube/gold/box-test_out_nek0.e?raw=true"
    status = download_file_if_not_exists(url, filename)
    if status == DownloadStatus.no_connection:
        pytest.skip("No internet connection")

    mesh = io4dolfinx.read_mesh(filename, MPI.COMM_WORLD, backend="exodus", time=5)
    assert mesh.topology.index_map(mesh.topology.dim).size_global == 64
    assert mesh.geometry.index_map().size_global == 1728

    u_x = io4dolfinx.read_point_data(filename, name="disp_x", mesh=mesh, backend="exodus", time=5.0)
    ref_min = -0.7699306351843485
    ref_max = 0.7699306351843485
    assert np.isclose(mesh.comm.allreduce(np.min(u_x.x.array), op=MPI.MIN), ref_min)
    assert np.isclose(mesh.comm.allreduce(np.max(u_x.x.array), op=MPI.MAX), ref_max)


# Exodus elements, from the Exodus II manual and the SEACAS documentation, written here
# independently of the reader's tables. The DOLFINx vertex at each Exodus vertex: Exodus goes
# round quadrilaterals, DOLFINx goes in tensor-product order.
exodus_vertices = {
    "interval": [0, 1],
    "triangle": [0, 1, 2],
    "quadrilateral": [0, 1, 3, 2],
    "tetrahedron": [0, 1, 2, 3],
    "hexahedron": [0, 1, 3, 2, 4, 5, 7, 6],
    "prism": [0, 1, 2, 3, 4, 5],
}
# The nodes after the vertices of a second-order element, each at the centre of the listed
# Exodus vertices (0-based). Edge nodes come first, and the serendipity QUAD8 and HEX20 stop
# after them. HEX27 follows Fig 4.14 of the manual: the centroid, then the faces.
_bottom: list[tuple[int, ...]] = [(0, 1), (1, 2), (2, 3), (3, 0)]
exodus_nodes: dict[str, list[tuple[int, ...]]] = {
    "interval": [(0, 1)],
    "triangle": [(0, 1), (1, 2), (2, 0)],
    "quadrilateral": [*_bottom, (0, 1, 2, 3)],
    "tetrahedron": [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)],
    "hexahedron": [
        *_bottom,
        *[(i, i + 4) for i in range(4)],
        *[(i + 4, j + 4) for i, j in _bottom],
        tuple(range(8)),
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 3, 7, 4),
        (1, 2, 6, 5),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
    ],
}
# The Exodus vertices of each side, numbered from 1
exodus_sides: dict[str, list[tuple[int, ...]]] = {
    "quadrilateral": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "tetrahedron": [(0, 1, 3), (1, 2, 3), (0, 3, 2), (0, 2, 1)],
    "hexahedron": [
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (0, 4, 7, 3),
        (0, 3, 2, 1),
        (4, 5, 6, 7),
    ],
}
# Exodus element names, by cell and number of nodes
exodus_names = {
    ("interval", 2): "BAR2",
    ("interval", 3): "BAR3",
    ("triangle", 3): "TRI3",
    ("triangle", 6): "TRI6",
    ("quadrilateral", 4): "QUAD4",
    ("quadrilateral", 8): "QUAD8",
    ("quadrilateral", 9): "QUAD9",
    ("tetrahedron", 4): "TETRA4",
    ("tetrahedron", 10): "TETRA10",
    ("hexahedron", 8): "HEX8",
    ("hexahedron", 20): "HEX20",
    ("hexahedron", 27): "HEX27",
    ("prism", 6): "WEDGE6",
}


def exodus_order(cell_type: str, degree: int, serendipity: bool, node_order) -> list[int]:
    """The node of the DOLFINx cell at each node of the Exodus element."""
    nodes = exodus_nodes[cell_type] if degree == 2 else []
    if serendipity:
        # The edge nodes, which come first
        nodes = nodes[: len(basix.topology(getattr(basix.CellType, cell_type))[1])]
    return node_order(cell_type, degree, exodus_vertices[cell_type], nodes)


def point_function(x):
    return 1 + x[0] + 2 * x[1] + 3 * x[2]


def write_exodus(
    path: Path,
    mesh: dolfinx.mesh.Mesh,
    order: list[int],
    sides: tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32]] | None,
    blocks: npt.NDArray[np.int32],
    cell_values: npt.NDArray[np.float64],
):
    """Write a DOLFINx mesh on one process to an Exodus file, with the nodes of each cell taken
    in `order`. The cells are split into element blocks 10 and 20, with side sets from
    `sides`, the point field "u" and the cell field "c", at times 0 and 1 (the fields at
    time 1 are twice those at time 0). `sides` holds a cell, local facet and value per tagged
    facet, as from the ``facet_cells`` fixture."""
    cell_type = mesh.topology.cell_type.name
    tdim = mesh.topology.dim
    gdim = tdim
    cells = fenicsx_compat.dofmap(mesh)[:, order]
    x = mesh.geometry.x[:, :gdim]
    block_ids = [10, 20]
    # Exodus numbers elements block by block
    element_order = np.argsort(blocks, kind="stable")
    element_id = np.empty_like(element_order)
    element_id[element_order] = np.arange(1, len(element_order) + 1)

    with netCDF4.Dataset(path, "w") as f:
        f.createDimension("len_name", 33)
        f.createDimension("time_step", None)
        f.createDimension("num_dim", gdim)
        f.createDimension("num_nodes", len(x))
        f.createDimension("num_elem", len(cells))
        f.createDimension("num_el_blk", len(block_ids))
        for i, c in enumerate("xyz"[:gdim]):
            f.createVariable(f"coord{c}", "f8", ("num_nodes",))[:] = x[:, i]
        f.createVariable("eb_prop1", "i4", ("num_el_blk",))[:] = block_ids
        f.createVariable("time_whole", "f8", ("time_step",))[:] = [0.0, 1.0]

        f.createDimension("num_nod_var", 1)
        names = f.createVariable("name_nod_var", "S1", ("num_nod_var", "len_name"))
        names[0] = netCDF4.stringtochar(np.array(["u"], dtype="S33"))
        u = point_function(np.vstack([x.T, np.zeros((3 - gdim, len(x)))]))
        f.createVariable("vals_nod_var1", "f8", ("time_step", "num_nodes"))[:] = [u, 2 * u]

        f.createDimension("num_elem_var", 1)
        names = f.createVariable("name_elem_var", "S1", ("num_elem_var", "len_name"))
        names[0] = netCDF4.stringtochar(np.array(["c"], dtype="S33"))
        for b, block_id in enumerate(block_ids):
            in_block = np.flatnonzero(blocks == block_id)
            f.createDimension(f"num_el_in_blk{b + 1}", len(in_block))
            f.createDimension(f"num_nod_per_el{b + 1}", cells.shape[1])
            connect = f.createVariable(
                f"connect{b + 1}", "i4", (f"num_el_in_blk{b + 1}", f"num_nod_per_el{b + 1}")
            )
            connect.elem_type = exodus_names[(cell_type, len(order))]
            connect[:] = cells[in_block] + 1
            vals = f.createVariable(
                f"vals_elem_var1eb{b + 1}", "f8", ("time_step", f"num_el_in_blk{b + 1}")
            )
            vals[:] = [cell_values[in_block], 2 * cell_values[in_block]]

        if sides is None:
            return
        cells_of, local_facets, values = sides
        facets = basix.topology(getattr(basix.CellType, cell_type))[tdim - 1]
        vertices = exodus_vertices[cell_type]
        exodus = [{vertices[v] for v in side} for side in exodus_sides[cell_type]]
        side_numbers = np.array([exodus.index(set(facets[f])) + 1 for f in local_facets])
        tag_values = np.unique(values)
        f.createDimension("num_side_sets", len(tag_values))
        f.createVariable("ss_prop1", "i4", ("num_side_sets",))[:] = tag_values
        for i, value in enumerate(tag_values):
            in_set = values == value
            f.createDimension(f"num_side_ss{i + 1}", in_set.sum())
            dims = (f"num_side_ss{i + 1}",)
            f.createVariable(f"elem_ss{i + 1}", "i4", dims)[:] = element_id[cells_of[in_set]]
            f.createVariable(f"side_ss{i + 1}", "i4", dims)[:] = side_numbers[in_set]


@pytest.mark.parametrize(
    "cell_type, degree, serendipity",
    [
        ("interval", 1, False),
        ("interval", 2, False),
        ("triangle", 1, False),
        ("triangle", 2, False),
        ("quadrilateral", 1, False),
        ("quadrilateral", 2, False),
        ("quadrilateral", 2, True),
        ("tetrahedron", 1, False),
        ("tetrahedron", 2, False),
        ("hexahedron", 1, False),
        ("hexahedron", 2, False),
        ("hexahedron", 2, True),
        ("prism", 1, False),
    ],
)
def test_read_generated(
    shared_tmp_path,
    cell_type,
    degree,
    serendipity,
    unit_mesh,
    curve,
    boundary_tags,
    node_order,
    facet_cells,
    integrals,
):
    """Round trip of a curved DOLFINx mesh through an Exodus file, with element blocks, side
    sets, and point and cell fields."""
    mesh = unit_mesh(cell_type, degree, serendipity)
    facet_tags = boundary_tags(mesh) if cell_type in exodus_sides else None
    curve(mesh)
    order = exodus_order(cell_type, degree, serendipity, node_order)
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    midpoints = dolfinx.mesh.compute_midpoints(mesh, tdim, np.arange(num_cells, dtype=np.int32))
    blocks = np.where(midpoints[:, 0] < 0.5, 10, 20).astype(np.int32)
    cell_values = 1 + midpoints[:, 0] + midpoints[:, tdim - 1] ** 2
    filename = shared_tmp_path / f"{cell_type}_{len(order)}.e"
    if MPI.COMM_WORLD.rank == 0:
        sides = facet_cells(mesh, facet_tags) if facet_tags is not None else None
        write_exodus(filename, mesh, order, sides, blocks, cell_values)
    MPI.COMM_WORLD.Barrier()

    # Reference values, from the mesh written
    cell_tags = dolfinx.mesh.meshtags(mesh, tdim, np.arange(num_cells, dtype=np.int32), blocks)
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("Lagrange", degree)))
    u.interpolate(lambda x: 2 * point_function(x))
    c = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("DG", 0)))
    c.x.array[:] = 2 * cell_values
    expected = integrals(mesh, cell_tags, facet_tags, {"u": u, "c": c})

    comm = MPI.COMM_WORLD
    mesh_in = io4dolfinx.read_mesh(filename, comm, backend="exodus")
    cell_tags_in = io4dolfinx.read_meshtags(
        filename, mesh_in, meshtag_name="cell", backend="exodus"
    )
    facet_tags_in = (
        io4dolfinx.read_meshtags(filename, mesh_in, meshtag_name="facet", backend="exodus")
        if facet_tags is not None
        else None
    )
    u_in = io4dolfinx.read_point_data(filename, name="u", mesh=mesh_in, backend="exodus", time=1.0)
    c_in = io4dolfinx.read_cell_data(filename, name="c", mesh=mesh_in, backend="exodus", time=1.0)
    found = integrals(mesh_in, cell_tags_in, facet_tags_in, {"u": u_in, "c": c_in})

    assert fenicsx_compat.cmap(mesh_in).dim == len(order)
    assert mesh_in.geometry.index_map().size_global == mesh.geometry.index_map().size_global
    assert found.keys() == expected.keys()
    for name, value in expected.items():
        assert np.isclose(found[name], value), name


def test_unsupported_element(shared_tmp_path):
    """Elements without a DOLFINx coordinate element, such as the 15-node wedge, raise."""
    filename = shared_tmp_path / "wedge15.e"
    if MPI.COMM_WORLD.rank == 0:
        with netCDF4.Dataset(filename, "w") as f:
            f.createDimension("num_dim", 3)
            f.createDimension("num_nodes", 15)
            f.createDimension("num_el_blk", 1)
            f.createDimension("num_el_in_blk1", 1)
            f.createDimension("num_nod_per_el1", 15)
            for c in "xyz":
                f.createVariable(f"coord{c}", "f8", ("num_nodes",))[:] = np.zeros(15)
            connect = f.createVariable("connect1", "i4", ("num_el_in_blk1", "num_nod_per_el1"))
            connect.elem_type = "WEDGE15"
            connect[:] = np.arange(1, 16)
    MPI.COMM_WORLD.Barrier()
    with pytest.raises(NotImplementedError, match="WEDGE15 with 15 nodes"):
        io4dolfinx.read_mesh(filename, MPI.COMM_SELF, backend="exodus")
