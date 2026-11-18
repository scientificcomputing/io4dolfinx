"""
Module that can read the VTKHDF format using h5py.
"""

from pathlib import Path
from typing import Any

from mpi4py import MPI

import basix
import dolfinx
import fenicsx_compat
import h5py
import numpy as np
import numpy.typing as npt

from io4dolfinx.structures import ArrayData, FunctionData, MeshData, MeshTagsData, ReadMeshData
from io4dolfinx.utils import check_file_exists, compute_local_range, same_time

from .. import DEFAULT_MESH_NAME, FileMode, ReadMode
from ..h5py import storage as h5_storage
from ..h5py.backend import convert_file_mode, h5pyfile
from ..vtk_cell_types import _arbitrary_lagrange_vtk, _cell_degree, _first_order_vtk

read_mode = ReadMode.parallel

_vtk_hdf_version = np.array([2, 1], dtype=np.int32)
#: Top-level group for metadata io4dolfinx needs but VTKHDF has no slot for.
#: The specification reserves everything outside ``/VTKHDF`` for exactly this:
#: "Top-level groups outside of /VTKHDF do not contain any information related to
#: the VTK data model and are outside of the scope of this specification. They can
#: be useful to store meta-information that could be read and written by custom
#: VTKHDF implementation."
_metadata_group = "/io4dolfinx"
#: Basix Lagrange variant of a mesh's coordinate element, per mesh name.
_lagrange_variant_attr = "LagrangeVariant"
#: Steps at which a mesh's geometry or a point or cell array was written, in the mesh's
#: or array's metadata group. Every step needs data in VTKHDF, so the steps before the
#: first write share the data of the first write, and the offsets alone cannot tell at
#: which step that was. The writer needs it to fill in steps written out of order.
_written_steps = "Steps"


def _array_metadata_group(mesh_name: str, extension: str, name: str) -> str:
    """The io4dolfinx metadata group of a point or cell array.

    Args:
        mesh_name: Name of the mesh the array is stored on
        extension: ``"Point"`` for point data, ``"Cell"`` for cell data
        name: Name of the array

    Returns:
        Path of the group in the file
    """
    return f"{_metadata_group}/{mesh_name}/{extension}Data/{name}"


def get_default_backend_args(arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Get default backend arguments given a set of input arguments.

    Args:
        arguments: Input backend arguments

    Returns:
        Updated backend arguments
    """
    # `arguments or {...}` would drop the default whenever the caller passed any
    # other argument, so set it on a copy instead.
    args = dict(arguments) if arguments else {}
    args.setdefault("name", DEFAULT_MESH_NAME)
    return args


def find_all_unique_cell_types(comm, cell_types, num_nodes):
    """
    Given a set of cell types and number of nodes per cell, find all unique cell types
    across all ranks.

    Args:
        comm: MPI communicator
        cell_types: Local cell types
        num_nodes: Number of nodes per cell


    Returns:
        A 2D array where each row corresponds to a cell type (vtk int)
        and the number of nodes.
    """
    # Combine cell_types, num_nodes as tuple
    c_hash = np.zeros((2, len(cell_types)), dtype=np.int32)
    c_hash[0] = cell_types
    c_hash[1] = num_nodes
    indexes = np.unique(c_hash.T, axis=0, return_index=True)[1]
    local_unique_cells = c_hash.T[indexes]

    all_cell_types = np.vstack(comm.allgather(local_unique_cells))
    indexes = np.unique(all_cell_types, axis=0, return_index=True)[1]
    all_unique_cell_types = all_cell_types[indexes]
    return all_unique_cell_types


def _decode_bytes_if_needed(value: bytes | str) -> str:
    """Decode bytes to string if necessary (for h5py compatibility)"""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _get_vtk_group(h5file, name: str) -> h5py.Group:
    """
    Navigates the VTKHDF group hierarchy to find the specific first
    UnstructuredGrid with a specific name.
    Handles both MultiBlockDataSet and direct UnstructuredGrid types.
    """
    hdf = h5file["VTKHDF"]
    file_type = _decode_bytes_if_needed(hdf.attrs["Type"])

    if file_type == "MultiBlockDataSet":
        ass = hdf["Assembly"]

        def visitor(path):
            mesh_group = path.rsplit("/", 1)
            mesh_name = mesh_group[0] if len(mesh_group) == 1 else mesh_group[1]
            if mesh_name == name:
                obj = ass.get(path)
                # Check attributes carefully
                if "Type" in obj.attrs.keys():
                    attr_type = obj.attrs["Type"]
                    if isinstance(attr_type, bytes):
                        attr_type = attr_type.decode("utf-8")
                    if attr_type == "UnstructuredGrid":
                        return path
            return None

        mesh_node = ass.visit_links(visitor)
        if mesh_node is None:
            raise RuntimeError(f"Could not find unique mesh named '{name}' in Assembly.")

        return ass[mesh_node]

    elif file_type == "UnstructuredGrid":
        return hdf
    else:
        raise RuntimeError(f"Not supported file type {file_type}")


def _get_all_temporal_unstructured_grids(h5file: h5py.File) -> list[h5py.Group]:
    """The unstructured grids (meshes and mesh tags) with time steps in a file.

    Args:
        h5file: An open MultiBlockDataSet file

    Returns:
        The groups of the grids, empty if the file has no ``VTKHDF`` group
    """
    if "VTKHDF" not in h5file:
        return []
    return [
        block
        for block in h5file["VTKHDF"].values()
        if isinstance(block, h5py.Group)
        and "Steps" in block
        and _decode_bytes_if_needed(block.attrs.get("Type", "")) == "UnstructuredGrid"
    ]


def _get_longest_time_sequence(h5file: h5py.File) -> np.ndarray:
    """The longest time sequence of the grids in a file.

    VTK requires all grids to share their time steps, which older versions of io4dolfinx
    did not ensure.

    Args:
        h5file: An open MultiBlockDataSet file

    Returns:
        The time stamps, empty if no grid has time steps
    """
    values = [block["Steps"]["Values"][:] for block in _get_all_temporal_unstructured_grids(h5file)]
    return max(values, key=len, default=np.empty(0, dtype=np.float64))


def _find_time_step_index(stamps: np.ndarray, time: float, filename: str | Path) -> int | None:
    """Index of a time among the time steps of a file.

    Args:
        stamps: Time stamps of the file, increasing
        time: The time to look for, see :func:`io4dolfinx.utils.same_time`
        filename: Name of the file, for error messages

    Returns:
        The index of the time step, or ``None`` if ``time`` is a new time

    Raises:
        ValueError: If ``time`` is new but earlier than the last time step. VTK requires
            increasing time steps.
    """
    match = np.flatnonzero(same_time(stamps, time))
    if len(match) > 1:
        raise ValueError(f"Time {time} matches several time steps in {filename}.")
    if len(match) == 1:
        return int(match[0])
    if len(stamps) > 0 and time < stamps[-1]:
        raise ValueError(
            f"Cannot add time {time} to {filename}: a new time step must be later than "
            f"the last one, {stamps[-1]}."
        )
    return None


def _extend_1d_dataset(dataset: h5py.Dataset, num_steps: int):
    """Extend a per-step dataset with entries that repeat its last entry.

    Args:
        dataset: A dataset with one entry per time step
        num_steps: Number of entries after extension. A dataset with as many entries or
            more is left unchanged.
    """
    num_old = dataset.shape[0]
    if 0 < num_old < num_steps:
        dataset.resize(num_steps, axis=0)
        dataset[num_old:] = dataset[num_old - 1]


def _set_time_steps(h5file: h5py.File, stamps: np.ndarray):
    """Give every grid in a file the same time steps.

    The per-step datasets of a grid are extended to the new steps, where the grid keeps
    its data from its last step.

    Args:
        h5file: An open MultiBlockDataSet file
        stamps: The time stamps, which extend the existing time steps of every grid
    """
    num_steps = len(stamps)
    for block in _get_all_temporal_unstructured_grids(h5file):
        steps = block["Steps"]
        per_step = [
            block[key]
            for key in ("NumberOfPoints", "NumberOfCells", "NumberOfConnectivityIds")
            if key in block
        ]
        per_step += [
            steps[key]
            for key in (
                "PartOffsets",
                "NumberOfParts",
                "PointOffsets",
                "CellOffsets",
                "ConnectivityIdOffsets",
            )
            if key in steps
        ]
        for key in ("PointDataOffsets", "CellDataOffsets", "FieldDataOffsets"):
            if key in steps:
                per_step += list(steps[key].values())
        # Datasets shared by a mesh and its tags are extended once
        for dataset in per_step:
            _extend_1d_dataset(dataset, num_steps)
        values = steps["Values"]
        if values.shape[0] != num_steps:
            values.resize(num_steps, axis=0)
        values[:] = stamps
        steps.attrs["NSteps"] = np.int64(num_steps)


def _get_time_steps_and_data_offset(
    h5file: h5py.File, record: str, offsets_path: str
) -> tuple[np.ndarray, np.ndarray]:
    """Steps at which io4dolfinx wrote the geometry or an array, and the offset of the
    data written at each.

    Files without the record (older versions of io4dolfinx) have a write at the first
    step and wherever the offset changes.

    Args:
        h5file: An open file
        record: Path of the record of the written steps, see `_create_temporal_record`
        offsets_path: Path of the per-step offsets of the geometry or the array

    Returns:
        The written steps and their offsets, both empty if the offsets do not exist
    """
    if offsets_path not in h5file:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    offsets = h5file[offsets_path][:]
    if record in h5file:
        steps = h5file[record][:]
    else:
        steps = np.flatnonzero(np.hstack([[True], offsets[1:] != offsets[:-1]]))
    return steps, offsets[steps]


def _fill_time_steps(
    num_steps: int, written_steps: np.ndarray, written_values: np.ndarray
) -> np.ndarray:
    """Values of a per-step dataset, such as data offsets, at every time step.

    A step takes the value of the latest write at or before it. The steps before the
    first write take the value of the first write, as VTKHDF requires data at every step.

    Args:
        num_steps: Number of time steps
        written_steps: Steps at which a value was written, in any order
        written_values: The value written at each of ``written_steps``

    Returns:
        The value at each of the ``num_steps`` time steps
    """
    order = np.argsort(written_steps)
    previous_write = np.searchsorted(written_steps[order], np.arange(num_steps), side="right") - 1
    return written_values[order][np.maximum(previous_write, 0)]


def _create_temporal_record(h5file: h5py.File, record: str, written_steps: np.ndarray):
    """Create, or replace, the record of the steps at which the geometry or an array has
    been written.

    Args:
        h5file: An open file
        record: Path of the record, in the io4dolfinx metadata group
        written_steps: The steps at which the geometry or the array has been written
    """
    written_steps = np.sort(written_steps).astype(np.int64)
    if record in h5file:
        h5file[record].resize(written_steps.shape)
        h5file[record][:] = written_steps
    else:
        h5file.create_dataset(record, data=written_steps, chunks=True, maxshape=(None,))


def _get_time_index(hdf: h5py.Group, time: float | str, filename: str | Path) -> int:
    """Finds the index of a specific time stamp."""
    if "Steps" not in hdf.keys():
        raise RuntimeError(f"No timestepping information found in {filename}.")
    stamps = hdf["Steps"]["Values"][:]
    pos = np.flatnonzero(same_time(stamps, float(time)))
    if len(pos) == 0:
        raise RuntimeError(f"Could not find mesh at t={time} in {filename}.")
    elif len(pos) > 1:
        raise RuntimeError(f"Multiple time steps for mesh at t={time} in {filename}")
    return pos[0]


def _read_lagrange_variant(h5file, name: str) -> int:
    """Lagrange variant of mesh ``name``, or the VTKHDF default.

    A file written by anything other than io4dolfinx -- or by io4dolfinx before
    this metadata existed -- holds VTKHDF's equispaced higher-order cells.

    Args:
        h5file: The open HDF5 file
        name: Name of the mesh

    Returns:
        The basix Lagrange variant as an integer.
    """
    equispaced = int(basix.LagrangeVariant.equispaced)
    metadata = h5file.get(_metadata_group)
    if metadata is None or name not in metadata:
        return equispaced
    return int(metadata[name].attrs.get(_lagrange_variant_attr, equispaced))


def read_mesh_data(
    filename: Path | str,
    comm: MPI.Comm,
    time: str | float | None,
    read_from_partition: bool,
    backend_args: dict[str, Any] | None,
) -> ReadMeshData:
    """Read mesh data from file.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        time: Time stamp associated with the mesh to read
        read_from_partition: Whether to read partition information
        backend_args: Arguments to backend

    Returns:
        Internal data structure for the mesh data read from file
    """
    backend_args = get_default_backend_args(backend_args)
    check_file_exists(filename)
    if read_from_partition:
        raise RuntimeError("Cannot read partition data with VTKHDF")
    with h5pyfile(filename, "r", comm=comm) as h5file:
        hdf = _get_vtk_group(h5file, backend_args["name"])
        lagrange_variant = _read_lagrange_variant(h5file, backend_args["name"])
        if time is None:
            num_cells_global = hdf["Types"].size
            local_cell_range = compute_local_range(comm, num_cells_global)
            cell_types_local = hdf["Types"][slice(*local_cell_range)]

            num_points_global = hdf["NumberOfPoints"][0]
            local_point_range = compute_local_range(comm, num_points_global)
            points_local = hdf["Points"][slice(*local_point_range)]

            # Connectivity read
            offsets = hdf["Offsets"]
            local_connectivity_offset = offsets[local_cell_range[0] : local_cell_range[1] + 1]
            topology = hdf["Connectivity"][
                local_connectivity_offset[0] : local_connectivity_offset[-1]
            ]
            offset = local_connectivity_offset - local_connectivity_offset[0]
        else:
            time_index = _get_time_index(hdf, time, filename)

            stamps = hdf["Steps"]["Values"][:]
            # Get number of points
            point_node = hdf["Points"]
            step_start = hdf["Steps"]["PointOffsets"][time_index]

            # NOTE: currently, it doesn't seem like we follow:
            # https://docs.vtk.org/en/latest/vtk_file_formats/vtkhdf_file_format/vtkhdf_specifications.html#temporal-unstructuredgrid-and-polydata
            # As only one num_points is stored irregardless of time steps added.
            if hdf["NumberOfPoints"].shape[0] != len(stamps):
                num_pts = hdf["NumberOfPoints"][0]
            else:
                num_pts = hdf["NumberOfPoints"][time_index]
            lr = compute_local_range(comm, num_pts)
            points_local = point_node[step_start + lr[0] : step_start + lr[1]]

            # Get cell-types in step
            cell_start = hdf["Steps"]["CellOffsets"][time_index]
            if hdf["NumberOfCells"].shape[0] != len(stamps):
                num_cells = hdf["NumberOfCells"][0]
            else:
                num_cells = hdf["NumberOfCells"][time_index]
            local_cell_range = compute_local_range(comm, num_cells)
            cell_types_local = hdf["Types"][
                cell_start + local_cell_range[0] : cell_start + local_cell_range[1]
            ]

            # Get connectivity in step
            connectivity_start = hdf["Steps"]["ConnectivityIdOffsets"][time_index]
            # Connectivity read
            offsets = hdf["Offsets"]
            local_connectivity_offset = offsets[
                connectivity_start + local_cell_range[0] : connectivity_start
                + local_cell_range[1]
                + 1
            ]
            topology = hdf["Connectivity"][
                local_connectivity_offset[0] : local_connectivity_offset[-1]
            ]
            offset = local_connectivity_offset - local_connectivity_offset[0]

    # NOTE: Currently we limit ourselfs to a single celltype, as it makes life easier,
    # other things have to change in `MeshReadData` to support this.
    num_nodes_per_cell = offset[1:] - offset[:-1]
    unique_cells = find_all_unique_cell_types(comm, cell_types_local, num_nodes_per_cell)
    if unique_cells.shape[0] > 1:
        raise NotImplementedError("io4dolfinx does not support mixed celltype grids")
    if unique_cells.shape[0] == 0:
        name = backend_args["name"]
        raise ValueError(f"Grid '{name}' in {filename} has no cells on any process.")
    # Take the width from the globally agreed cell type rather than from this
    # process's own cells: a mesh with fewer cells than processes leaves some of
    # them with nothing local to measure.
    cell_type, number_of_nodes = unique_cells[0]
    topology = topology.reshape(-1, number_of_nodes)
    gtype = backend_args.get("dtype", points_local.dtype)
    if cell_type in _first_order_vtk.keys():
        ct = _first_order_vtk[cell_type]
        degree = 1
    elif cell_type in _arbitrary_lagrange_vtk.keys():
        ct = _arbitrary_lagrange_vtk[cell_type]
        degree = _cell_degree(ct, num_nodes=number_of_nodes)
    else:
        raise ValueError(f"Unknown VTK cell type {cell_type} in {filename}")
    perm = dolfinx.cpp.io.perm_vtk(dolfinx.mesh.to_type(ct), number_of_nodes)
    topology = topology[:, perm]
    # Equispaced is the VTKHDF default; a file io4dolfinx wrote says which
    # variant its nodes were actually placed with.
    lvar = int(lagrange_variant)
    return ReadMeshData(
        cells=topology, cell_type=ct, x=points_local.astype(gtype), lvar=lvar, degree=degree
    )


def read_point_data(
    filename: Path | str,
    name: str,
    comm: MPI.Comm,
    time: float | str | None,
    backend_args: dict[str, Any] | None,
) -> tuple[np.ndarray, int]:
    """Read data from the nodes of a mesh.

    Args:
        filename: Path to file
        name: Name of point data
        comm: Communicator to launch IO on.
        time: The time stamp
        backend_args: The backend arguments

    Returns:
       Data local to process (contiguous, no mpi comm) and local start range
    """
    backend_args = get_default_backend_args(backend_args)
    check_file_exists(filename)
    with h5pyfile(filename, "r", comm=comm) as h5file:
        hdf = _get_vtk_group(h5file, backend_args["name"])
        point_data = hdf["PointData"]
        assert point_data is not None
        if name not in point_data.keys():
            raise ValueError(f"No point data named {name} in {filename}.")
        func_node = point_data[name]

        if time is None:
            data_shape = func_node.shape[0]
            lr = compute_local_range(comm, data_shape)
            data = func_node[slice(*lr)]
            return data, lr[0]
        else:
            time_index = _get_time_index(hdf, time, filename)
            stamps = hdf["Steps"]["Values"][:]
            step_start = hdf["Steps"]["PointDataOffsets"][name][time_index]
            # NOTE: currently, it doesn't seem like we follow:
            # https://docs.vtk.org/en/latest/vtk_file_formats/vtkhdf_file_format/vtkhdf_specifications.html#temporal-unstructuredgrid-and-polydata
            # As only one num_points is stored irregardless of time steps added.
            if hdf["NumberOfPoints"].shape[0] != len(stamps):
                num_pts = hdf["NumberOfPoints"][0]
            else:
                num_pts = hdf["NumberOfPoints"][time_index]
            lr = compute_local_range(comm, num_pts)
            return func_node[step_start + lr[0] : step_start + lr[1]], lr[0]


def read_cell_data(
    filename: Path | str,
    name: str,
    comm: MPI.Comm,
    time: str | float | None,
    backend_args: dict[str, Any] | None,
) -> tuple[npt.NDArray[np.int64], np.ndarray]:
    backend_args = get_default_backend_args(backend_args)
    check_file_exists(filename)
    with h5pyfile(filename, "r", comm=comm) as h5file:
        hdf = _get_vtk_group(h5file, backend_args["name"])

        if "CellData" not in hdf.keys():
            raise RuntimeError(f"No cell data found in {filename}.")
        cell_data = hdf["CellData"]
        assert cell_data is not None
        if name not in cell_data.keys():
            raise ValueError(f"No cell data with name {name} in {filename}")
        cell_data_node = cell_data[name]
        assert cell_data_node is not None
        if time is None:
            cell_data_shape = cell_data_node.shape
            num_cells_global = hdf["Types"].size
            assert num_cells_global == cell_data_shape[0]
            local_cell_range = compute_local_range(comm, num_cells_global)
            data = cell_data_node[slice(*local_cell_range)]
        else:
            time_index = _get_time_index(hdf, time, filename)
            stamps = hdf["Steps"]["Values"][:]
            cd_start = hdf["Steps"]["CellDataOffsets"][name][time_index]

            # NOTE: currently, it doesn't seem like we follow:
            # https://docs.vtk.org/en/latest/vtk_file_formats/vtkhdf_file_format/vtkhdf_specifications.html#temporal-unstructuredgrid-and-polydata
            # As only one num_points is stored irregardless of time steps added.
            if hdf["NumberOfCells"].shape[0] != len(stamps):
                number_of_cells = hdf["NumberOfCells"][0]
            else:
                number_of_cells = hdf["NumberOfCells"][time_index]
            lr = compute_local_range(comm, number_of_cells)
            data = cell_data_node[cd_start + lr[0] : cd_start + lr[1]]

    # NOTE: THis could be optimized by hand-coding some communication in
    # `read_cell_data` on the frontend side
    md = read_mesh_data(
        filename, comm, time=time, read_from_partition=False, backend_args=backend_args
    )
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    return md.cells, data


def write_attributes(
    filename: Path | str,
    comm: MPI.Comm,
    name: str,
    attributes: dict[str, np.ndarray],
    backend_args: dict[str, Any] | None,
):
    """Write attributes to file.

    Args:
        filename: Path to file to write to
        comm: MPI communicator used in storage
        name: Name of the attribute group
        attributes: Dictionary of attributes to write
        backend_args: Arguments to backend
    """
    raise NotImplementedError("The VTKHDF backend cannot write attributes.")


def read_attributes(
    filename: Path | str,
    comm: MPI.Comm,
    name: str,
    backend_args: dict[str, Any] | None,
) -> dict[str, Any]:
    """Read attributes from file.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        name: Name of the attribute group
        backend_args: Arguments to backend

    Returns:
        Dictionary of attributes read from file
    """
    raise NotImplementedError("The VTKHDF backend cannot read attributes.")


def read_timestamps(
    filename: Path | str,
    comm: MPI.Comm,
    function_name: str,
    backend_args: dict[str, Any] | None,
) -> npt.NDArray[np.float64 | str]:  # type: ignore[type-var]
    """Read timestamps from file.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        function_name: Name of the function to read timestamps for
        backend_args: Arguments to backend

    Returns:
        The time steps of the mesh the function is stored on. A VTKHDF array has
        data at every time step of its mesh.
    """
    backend_args = get_default_backend_args(backend_args)
    check_file_exists(filename)
    mesh_name = backend_args["name"]
    # Temporal data storage as described in
    # https://docs.vtk.org/en/latest/vtk_file_formats/vtkhdf_file_format/vtkhdf_specifications.html#temporal-data
    with h5pyfile(filename, "r", comm=comm) as h5file:
        block = _get_vtk_group(h5file, mesh_name)
        if "Steps" not in block.keys():
            raise RuntimeError(f"{filename} does not contain time-stepping information.")
        for data in ("CellData", "PointData"):
            if data in block.keys() and function_name in block[data].keys():
                return block["Steps"]["Values"][:]
        raise RuntimeError(f"Function {function_name} is not associated with a time-stamp.")


def read_function_names(
    filename: Path | str, comm: MPI.Comm, backend_args: dict[str, Any] | None
) -> list[str]:
    """Read all function names from a file.

    Args:
        filename: Path to file
        comm: MPI communicator to launch IO on.
        backend_args: Arguments to backend

    Returns:
        A list of function names.
    """
    backend_args = get_default_backend_args(backend_args)
    check_file_exists(filename)
    with h5pyfile(filename, "r", comm=comm) as h5file:
        hdf = h5file["VTKHDF"]
        function_names = set()
        if "CellData" in hdf.keys():
            for item in hdf["CellData"].keys():
                function_names.add(item)
        if "PointData" in hdf.keys():
            for item in hdf["PointData"].keys():
                function_names.add(item)
    return list(function_names)


def _create_dataset(
    root: h5py.File | h5py.Group,
    name: str,
    shape: tuple[int, ...],
    dtype: npt.DTypeLike,
    chunks: bool,
    maxshape: tuple[int | None, ...],
    mode: str,
    resize: bool = True,
    filters: dict[str, Any] | None = None,
) -> h5py.Dataset:
    """Create a dataset, or open (and resize) an existing one in append mode.

    Args:
        filters: h5py filter arguments for a new dataset, from
            :func:`io4dolfinx.backends.h5py.storage.filter_options`. Omitted for small
            bookkeeping datasets, which every rank writes independently and which must
            therefore stay unfiltered.
    """
    if name not in root.keys():
        mode = "w"
    if mode == "w":
        dataset = root.create_dataset(
            name,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            maxshape=maxshape,
            **(filters or {}),
        )
    elif mode == "a":
        dataset = root[name]
        old_shape = dataset.shape
        # Only resize for dimension
        if resize:
            if len(old_shape) == 1:
                new_shape = (old_shape[0] + shape[0],)
            else:
                new_shape = (old_shape[0] + shape[0], *old_shape[1:])
            dataset.resize(new_shape)
    else:
        raise ValueError(f"Unknown file mode '{mode}' when creating dataset {name} in {root}")
    return dataset


def _create_group(root: h5py.File | h5py.Group, name: str, mode: str) -> h5py.Group:
    if name not in root.keys():
        mode = "w"
    if mode == "w":
        # Track order has to be on to make multiblock work:
        # https://docs.vtk.org/en/latest/vtk_file_formats/vtkhdf_file_format/vtkhdf_specifications.html#partitioneddatasetcollection-and-multiblockdataset
        group = root.create_group(name, track_order=True)
    elif mode == "a":
        group = root[name]
    else:
        raise ValueError("Unknown file mode '{h5_mode}'")
    return group


def _compute_append_slice(
    dataset: h5py.Dataset, input_size: int, original_slice: tuple[int, int] | np.ndarray, mode: str
) -> slice:
    append_offset = dataset.shape[0] - input_size if mode == "a" else 0
    return slice(*(np.asarray(original_slice) + append_offset).astype(np.int64))


def write_mesh(
    filename: Path | str,
    comm: MPI.Comm,
    mesh: MeshData,
    backend_args: dict[str, Any] | None,
    mode: FileMode,
    time: float,
):
    """
    Write a mesh to file.

    Args:
        comm: MPI communicator used in storage
        mesh: Internal data structure for the mesh data to save to file
        filename: Path to file to write to
        backend_args: Arguments to backend
        mode: File-mode to store the mesh
        time: Time stamp associated with the mesh
    """
    h5_mode = convert_file_mode(mode)
    backend_args = get_default_backend_args(backend_args)
    name = backend_args["name"]
    storage_options = h5_storage.storage_options(comm, backend_args)

    with h5pyfile(filename, h5_mode, comm=comm) as h5file:
        # Check all existing bulk arrays before changing attributes or resizing anything.
        if storage_options and h5_mode == "a" and f"/VTKHDF/{name}" in h5file:
            block = h5file[f"/VTKHDF/{name}"]
            for key in ("Points", "Offsets", "Connectivity", "Types"):
                if key in block:
                    h5_storage.check_storage_options(block[key], storage_options)

        # Check the mesh and the time before writing anything. Only the geometry of a mesh
        # can change between time steps, as its topology is stored once.
        block_path = f"/VTKHDF/{name}"
        if block_path in h5file and "NumberOfCells" in h5file[block_path]:
            num_dofs_per_cell = mesh.local_topology.shape[1]
            for key, value in (
                ("NumberOfPoints", mesh.num_nodes_global),
                ("NumberOfCells", mesh.num_cells_global),
                ("NumberOfConnectivityIds", mesh.num_cells_global * num_dofs_per_cell),
            ):
                stored = int(h5file[block_path][key][0])
                if stored != value:
                    raise ValueError(
                        f"Mesh '{name}' in {filename} has {key} {stored}, not {value}. Only "
                        "the geometry of a mesh can change between time steps."
                    )
        stamps = _get_longest_time_sequence(h5file)
        step = _find_time_step_index(stamps, time, filename)
        record = f"{_metadata_group}/{name}/{_written_steps}"
        written_steps, written_offsets = _get_time_steps_and_data_offset(
            h5file, record, f"/VTKHDF/{name}/Steps/PointOffsets"
        )
        if same_time(stamps[written_steps], time).any():
            raise ValueError(
                f"Mesh '{name}' has already been written at time {time} in {filename}."
            )
        if step is None:
            stamps = np.append(stamps, time)
            step = len(stamps) - 1
        _set_time_steps(h5file, stamps)
        num_steps = len(stamps)

        hdf = _create_group(h5file, "/VTKHDF", h5_mode)
        hdf.attrs.create("Type", "MultiBlockDataSet")
        hdf.attrs["Version"] = _vtk_hdf_version

        mesh_group = _create_group(hdf, name, h5_mode)
        mesh_group.attrs.create("Type", "UnstructuredGrid")
        mesh_group.attrs["Version"] = _vtk_hdf_version

        assembly = _create_group(hdf, "Assembly", h5_mode)

        mesh_assembly = _create_group(assembly, name, h5_mode)
        if name not in mesh_assembly.keys():
            mesh_assembly[name] = h5py.SoftLink(f"/VTKHDF/{name}")

        # VTKHDF defines its higher-order cells as equispaced and has nowhere to
        # record a Lagrange variant, so it goes in our own top-level group, which
        # the specification reserves for exactly this. Without it a degree>1 mesh
        # reads back with its coordinates intact but its cells curving
        # differently between them.
        metadata = _create_group(h5file, _metadata_group, h5_mode)
        mesh_metadata = _create_group(metadata, name, h5_mode)
        mesh_metadata.attrs[_lagrange_variant_attr] = np.int32(mesh.lagrange_variant)

        # Number of points per time step, set with the point offsets below
        number_of_points = _create_dataset(
            mesh_group,
            "NumberOfPoints",
            shape=(num_steps,),
            dtype=np.int64,
            chunks=True,
            maxshape=(None,),
            mode=h5_mode,
            resize=False,
        )

        # Store nodes
        points = _create_dataset(
            mesh_group,
            "Points",
            shape=(mesh.num_nodes_global, 3),
            dtype=mesh.local_geometry.dtype,
            chunks=True,
            maxshape=(None, 3),
            mode=h5_mode,
            filters=h5_storage.filter_options(storage_options, mesh.local_geometry.dtype),
        )
        insert_slice = _compute_append_slice(
            points, mesh.num_nodes_global, mesh.local_geometry_pos, h5_mode
        )
        h5_storage.write_dataset(
            points,
            (insert_slice, slice(0, mesh.local_geometry.shape[1])),
            mesh.local_geometry,
        )

        # DOLFINx meshes only move their geometry, so the topology is stored once and a
        # later write of the mesh rewrites it in place. Every time step reads it as part
        # 0 (zero PartOffsets, CellOffsets and ConnectivityIdOffsets), the VTKHDF layout
        # for static data.
        time_indep_datasets = {}
        for key in ["NumberOfCells", "NumberOfConnectivityIds"]:
            time_indep_datasets[key] = _create_dataset(
                mesh_group,
                key,
                shape=(num_steps,),
                dtype=np.int64,
                chunks=True,
                maxshape=(None,),
                mode=h5_mode,
                resize=False,  # Static, stored once
            )

        num_dofs_per_cell = mesh.local_topology.shape[1]
        time_indep_datasets["NumberOfCells"][:] = mesh.num_cells_global
        time_indep_datasets["NumberOfConnectivityIds"][:] = (
            mesh.num_cells_global * num_dofs_per_cell
        )

        # Store topology offsets (single celltype assumption)
        offsets = _create_dataset(
            mesh_group,
            "Offsets",
            shape=(mesh.num_cells_global + 1,),
            dtype=np.int64,
            chunks=True,
            mode=h5_mode,
            maxshape=(None,),
            resize=False,  # Static, stored once
            filters=h5_storage.filter_options(storage_options, np.int64),
        )
        # Each offset has one writer. Only the last nonempty rank writes the final offset.
        first, last = mesh.local_topology_pos
        include_end = last == mesh.num_cells_global and last > first
        offset_data = np.arange(last - first + int(include_end), dtype=np.int64)
        offset_data *= num_dofs_per_cell
        offset_data += num_dofs_per_cell * mesh.local_topology_pos[0]
        insert_slice = _compute_append_slice(
            offsets,
            mesh.num_cells_global + 1,
            (first, last + int(include_end)),
            mode=h5_mode,
        )
        h5_storage.write_dataset(offsets, insert_slice, offset_data)
        del offset_data

        # Permute and store topology data
        dx_ct = dolfinx.mesh.to_type(mesh.cell_type)
        top_perm = np.argsort(dolfinx.cpp.io.perm_vtk(dx_ct, num_dofs_per_cell))
        topology_data = mesh.local_topology[:, top_perm].flatten()
        topology = _create_dataset(
            mesh_group,
            "Connectivity",
            shape=(mesh.num_cells_global * num_dofs_per_cell,),
            dtype=np.int64,
            chunks=True,
            maxshape=(None,),
            mode=h5_mode,
            resize=False,  # Static, stored once
            filters=h5_storage.filter_options(storage_options, np.int64),
        )
        insert_slice = _compute_append_slice(
            topology,
            mesh.num_cells_global * num_dofs_per_cell,
            np.array(mesh.local_topology_pos) * num_dofs_per_cell,
            mode=h5_mode,
        )
        h5_storage.write_dataset(topology, insert_slice, topology_data)
        del topology_data

        # Store celltypes
        cell_types = np.full(
            mesh.local_topology.shape[0],
            fenicsx_compat.vtk_cell_type(dx_ct, num_dofs_per_cell),
            dtype=np.uint8,
        )
        types = _create_dataset(
            mesh_group,
            "Types",
            shape=(mesh.num_cells_global,),
            dtype=np.uint8,
            maxshape=(None,),
            chunks=True,
            mode=h5_mode,
            resize=False,  # Static, stored once
            filters=h5_storage.filter_options(storage_options, np.uint8),
        )
        insert_slice = _compute_append_slice(
            types, mesh.num_cells_global, mesh.local_topology_pos, h5_mode
        )
        h5_storage.write_dataset(types, insert_slice, cell_types)
        del cell_types

        # The file's time steps, with static topology. The geometry at each step is
        # that of the latest write at or before it, see `_fill_time_steps`.
        steps = _create_group(mesh_group, "Steps", mode=h5_mode)
        steps.attrs["NSteps"] = np.int64(num_steps)
        for key, dtype in (
            ("Values", np.float64),
            ("NumberOfParts", np.int64),
            ("PartOffsets", np.int64),
            ("PointOffsets", np.int64),
            ("CellOffsets", np.int64),
            ("ConnectivityIdOffsets", np.int64),
        ):
            _create_dataset(
                steps,
                key,
                shape=(num_steps,),
                dtype=dtype,
                chunks=True,
                maxshape=(None,),
                mode=h5_mode,
                resize=False,
            )
        steps["Values"][:] = stamps
        steps["NumberOfParts"][:] = 1
        steps["PartOffsets"][:] = 0
        steps["CellOffsets"][:] = types.shape[0] - mesh.num_cells_global
        steps["ConnectivityIdOffsets"][:] = (
            topology.shape[0] - mesh.num_cells_global * num_dofs_per_cell
        )
        for key in ("CellDataOffsets", "PointDataOffsets"):
            _create_group(steps, key, mode=h5_mode)

        written_points = np.append(number_of_points[:][written_steps], mesh.num_nodes_global)
        written_offsets = np.append(written_offsets, points.shape[0] - mesh.num_nodes_global)
        written_steps = np.append(written_steps, step)
        steps["PointOffsets"][:] = _fill_time_steps(num_steps, written_steps, written_offsets)
        number_of_points[:] = _fill_time_steps(num_steps, written_steps, written_points)
        _create_temporal_record(h5file, record, written_steps)


def write_meshtags(
    filename: str | Path,
    comm: MPI.Comm,
    data: MeshTagsData,
    backend_args: dict[str, Any] | None,
):
    """Write mesh tags to file.

    Args:
        filename: Path to file to write to
        comm: MPI communicator used in storage
        data: Internal data structure for the mesh tags to save to file
        backend_args: Arguments to backend
    """
    backend_args = get_default_backend_args(backend_args)
    name = backend_args["name"]
    h5_mode = "a"
    storage_options = h5_storage.storage_options(comm, backend_args)
    with h5pyfile(filename, h5_mode, comm=comm) as h5file:
        hdf = h5file["VTKHDF"]
        file_type = _decode_bytes_if_needed(hdf.attrs["Type"])
        # Check for type of VTKHDF file
        if file_type != "MultiBlockDataSet":
            raise ValueError(f"Cannot write meshtags to {filename} with VTK type {file_type}")

        # We place the mesh-tags in the same subgroup of assembly as the mesh
        parent_mesh_group = _get_vtk_group(h5file, name)
        block = parent_mesh_group.parent

        tag_path = f"/VTKHDF/{name}_{data.name}"
        # Check all existing bulk arrays before changing anything.
        if tag_path in h5file:
            tag_group = h5file[tag_path]
            for key in (f"CellData/{data.name}", "Offsets", "Connectivity", "Types"):
                if key in tag_group:
                    h5_storage.check_storage_options(tag_group[key], storage_options)
        if data.name not in block.keys():
            block[data.name] = h5py.SoftLink(tag_path)

        mesh_group = _create_group(hdf, tag_path, mode=h5_mode)
        mesh_group.attrs.create("Type", "UnstructuredGrid")
        mesh_group.attrs["Version"] = _vtk_hdf_version

        cell_data = _create_group(mesh_group, "CellData", mode=h5_mode)
        assert data.num_entities_global is not None
        dataset = _create_dataset(
            cell_data,
            data.name,
            shape=(data.num_entities_global,),
            dtype=data.values.dtype,
            chunks=True,
            maxshape=(None,),
            mode=h5_mode,
            filters=h5_storage.filter_options(storage_options, data.values.dtype),
        )
        assert data.local_start is not None
        insert_slice = _compute_append_slice(
            dataset,
            data.num_entities_global,
            np.array([data.local_start, data.local_start + data.indices.shape[0]]),
            mode=h5_mode,
        )
        h5_storage.write_dataset(dataset, insert_slice, data.values)

        # NOTE: The following is more or less a copy from write_mesh,
        # except that we pull out the point storage and use a softlink
        num_cells = _create_dataset(
            mesh_group,
            "NumberOfCells",
            shape=(1,),
            dtype=np.int64,
            chunks=True,
            maxshape=(None,),
            mode=h5_mode,
            resize=False,  # Static, stored once
        )
        num_cells[-1] = data.num_entities_global

        # Hardlink data should also follow hardlink for numbering
        for key in ["Points", "NumberOfPoints"]:
            if key not in mesh_group.keys():
                mesh_group[key] = parent_mesh_group[key]

        # Single celltype assumption
        num_dofs_per_cell = data.num_dofs_per_entity
        number_of_connectivities = _create_dataset(
            mesh_group,
            "NumberOfConnectivityIds",
            shape=(1,),
            dtype=np.int64,
            chunks=True,
            maxshape=(None,),
            mode=h5_mode,
            resize=False,  # Static, stored once
        )
        assert data.num_entities_global is not None and num_dofs_per_cell is not None
        number_of_connectivities[-1] = data.num_entities_global * num_dofs_per_cell

        # Store topology offsets (single celltype assumption)
        offsets = _create_dataset(
            mesh_group,
            "Offsets",
            shape=(data.num_entities_global + 1,),
            dtype=np.int64,
            chunks=True,
            mode=h5_mode,
            maxshape=(None,),
            resize=False,  # Static, stored once
            filters=h5_storage.filter_options(storage_options, np.int64),
        )
        # Each offset has one writer. Only the last nonempty rank writes the final offset.
        assert data.local_start is not None
        first, last = data.local_start, data.local_start + data.indices.shape[0]
        include_end = last == data.num_entities_global and last > first
        offset_data = np.arange(first, last + int(include_end), dtype=np.int64)
        offset_data *= num_dofs_per_cell
        insert_slice = _compute_append_slice(
            offsets,
            data.num_entities_global + 1,
            (first, last + int(include_end)),
            mode=h5_mode,
        )
        h5_storage.write_dataset(offsets, insert_slice, offset_data)
        del offset_data

        # Permute and store topology data
        if data.cell_type is None:
            raise ValueError("Cannot write meshtags without cell-type information")
        dx_ct = dolfinx.mesh.to_type(data.cell_type)
        top_perm = np.argsort(dolfinx.cpp.io.perm_vtk(dx_ct, num_dofs_per_cell))
        topology_data = data.indices[:, top_perm].flatten()
        topology = _create_dataset(
            mesh_group,
            "Connectivity",
            shape=(data.num_entities_global * num_dofs_per_cell,),
            dtype=np.int64,
            chunks=True,
            maxshape=(None,),
            mode=h5_mode,
            resize=False,  # Static, stored once
            filters=h5_storage.filter_options(storage_options, np.int64),
        )
        insert_slice = _compute_append_slice(
            topology,
            data.num_entities_global * num_dofs_per_cell,
            np.array([data.local_start, data.local_start + data.indices.shape[0]])
            * num_dofs_per_cell,
            mode=h5_mode,
        )
        h5_storage.write_dataset(topology, insert_slice, topology_data)
        del topology_data

        # Store celltypes
        cell_types = np.full(
            data.indices.shape[0],
            fenicsx_compat.vtk_cell_type(dx_ct, num_dofs_per_cell),
            dtype=np.uint8,
        )
        types = _create_dataset(
            mesh_group,
            "Types",
            shape=(data.num_entities_global,),
            dtype=np.uint8,
            maxshape=(None,),
            chunks=True,
            mode=h5_mode,
            resize=False,  # Static, stored once
            filters=h5_storage.filter_options(storage_options, np.uint8),
        )
        insert_slice = _compute_append_slice(
            types,
            data.num_entities_global,
            (data.local_start, data.local_start + data.indices.shape[0]),
            h5_mode,
        )
        h5_storage.write_dataset(types, insert_slice, cell_types)
        del cell_types

        steps = _create_group(mesh_group, "Steps", mode=h5_mode)

        # Copy n-step counter
        steps.attrs.create("NSteps", parent_mesh_group["Steps"].attrs["NSteps"])

        # Hardlink data that we know is the same across meshes
        hardlink_keys = ["NumberOfParts", "PartOffsets", "Values", "PointOffsets"]
        for key in hardlink_keys:
            steps[key] = parent_mesh_group["Steps"][key]

        # Write offset data for current time-step
        all_parts = {}
        for key in ["CellOffsets", "ConnectivityIdOffsets"]:
            all_parts[key] = _create_dataset(
                steps,
                key,
                shape=(1,),
                dtype=np.int64,
                chunks=True,
                maxshape=(None,),
                mode=h5_mode,
            )

        all_parts["CellOffsets"][-1] = types.shape[0] - data.num_entities_global
        all_parts["ConnectivityIdOffsets"][-1] = offsets.shape[0] - (data.num_entities_global + 1)
        # CellData requires an offset
        cd_off = _create_group(steps, "CellDataOffsets", mode=h5_mode)
        cd_data = _create_dataset(
            cd_off,
            data.name,
            shape=parent_mesh_group["Steps"]["CellOffsets"].shape,
            dtype=np.int64,
            chunks=True,
            maxshape=(None,),
            mode=h5_mode,
        )
        cd_data[-1] = types.shape[0] - data.num_entities_global

        # Static tags: every time step of the mesh holds the same cells
        num_steps = int(steps.attrs["NSteps"])
        for dataset in (
            num_cells,
            number_of_connectivities,
            all_parts["CellOffsets"],
            all_parts["ConnectivityIdOffsets"],
            cd_data,
        ):
            _extend_1d_dataset(dataset, num_steps)


def read_meshtags_data(
    filename: str | Path,
    comm: MPI.Comm,
    name: str,
    backend_args: dict[str, Any] | None,
) -> MeshTagsData:
    """Read mesh tags from file.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        name: Name of the mesh tags to read
        backend_args: Arguments to backend

    Returns:
        Internal data structure for the mesh tags read from file
    """
    backend_args = get_default_backend_args(backend_args)
    backend_args.update({"name": name})
    # Reuse reading cell-data
    indices, values = read_cell_data(filename, name, comm, None, backend_args=backend_args)
    # Read cell-type of grid to get topological dimension
    with h5pyfile(filename, "r", comm=comm) as h5file:
        hdf = _get_vtk_group(h5file, backend_args["name"])
        num_cells_global = hdf["Types"].size
        local_cell_range = compute_local_range(comm, num_cells_global)
        cell_types_local = hdf["Types"][slice(*local_cell_range)]
    unique_cells = find_all_unique_cell_types(comm, cell_types_local, indices.shape[1])
    if unique_cells.shape[0] > 1:
        raise NotImplementedError("io4dolfinx does not support mixed celltype grids")
    vtk_cell_type = unique_cells[0][0]
    if vtk_cell_type in _first_order_vtk.keys():
        ct = _first_order_vtk[vtk_cell_type]
    elif vtk_cell_type in _arbitrary_lagrange_vtk.keys():
        ct = _arbitrary_lagrange_vtk[vtk_cell_type]
    dim = dolfinx.mesh.cell_dim(dolfinx.mesh.to_type(ct))
    return MeshTagsData(name=name, values=values.flatten(), indices=indices, dim=dim)


def read_dofmap(
    filename: str | Path,
    comm: MPI.Comm,
    name: str,
    backend_args: dict[str, Any] | None,
) -> dolfinx.graph.AdjacencyList:
    """Read the dofmap of a function with a given name.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        name: Name of the function to read the dofmap for
        backend_args: Arguments to backend

    Returns:
        Dofmap as an AdjacencyList
    """
    raise NotImplementedError("The VTKHDF backend cannot make checkpoints.")


def read_dofs(
    filename: str | Path,
    comm: MPI.Comm,
    name: str,
    time: float,
    backend_args: dict[str, Any] | None,
) -> tuple[npt.NDArray[np.float32 | np.float64 | np.complex64 | np.complex128], int]:
    """Read the dofs (values) of a function with a given name from a given timestep.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        name: Name of the function to read the dofs for
        time: Time stamp associated with the function to read
        backend_args: Arguments to backend

    Returns:
        Contiguous sequence of degrees of freedom (with respect to input data)
        and the global starting point on the process.
        Process 0 has [0, M), process 1 [M, N), process 2 [N, O) etc.
    """
    raise NotImplementedError("The VTKHDF backend cannot make checkpoints.")


def read_cell_perms(
    comm: MPI.Comm, filename: Path | str, backend_args: dict[str, Any] | None
) -> npt.NDArray[np.uint32]:
    """
    Read cell permutation from file with given communicator,
    Split in continuous chunks based on number of cells in the input data.

    Args:
        comm: MPI communicator used in storage
        filename: Path to file to read from
        backend_args: Arguments to backend

    Returns:
        Contiguous sequence of permutations (with respect to input data)
        Process 0 has [0, M), process 1 [M, N), process 2 [N, O) etc.
    """
    raise NotImplementedError("The VTKHDF backend cannot make checkpoints.")


def write_function(
    filename: Path,
    comm: MPI.Comm,
    u: FunctionData,
    time: float,
    mode: FileMode,
    backend_args: dict[str, Any] | None,
):
    """
    Write a function to file.

    Args:
        comm: MPI communicator used in storage
        u: Internal data structure for the function data to save to file
        filename: Path to file to write to
        time: Time stamp associated with function
        mode: File-mode to store the function
        backend_args: Arguments to backend
    """
    raise NotImplementedError("The VTKHDF backend cannot make checkpoints.")


def read_legacy_mesh(
    filename: Path | str, comm: MPI.Comm, group: str
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.floating], str | None]:
    """Read in the mesh topology, geometry and (optionally) cell type from a
    legacy DOLFIN HDF5-file.

    Args:
        filename: Path to file to read from
        comm: MPI communicator used in storage
        group: Group in HDF5 file where mesh is stored

    Returns:
        Tuple containing:
            - Topology as a (num_cells, num_vertices_per_cell) array of global vertex indices
            - Geometry as a (num_vertices, geometric_dimension) array of vertex coordinates
            - Cell type as a string (e.g. "tetrahedron") or None if not found
    """
    raise NotImplementedError("The VTKHDF backend cannot read legacy DOLFIN meshes.")


def snapshot_checkpoint(
    filename: Path | str,
    mode: FileMode,
    u: dolfinx.fem.Function,
    backend_args: dict[str, Any] | None,
):
    """Create a snapshot checkpoint of a dolfinx function.

    Args:
        filename: Path to file to read from
        mode: File-mode to store the function
        u: dolfinx function to create a snapshot checkpoint for
        backend_args: Arguments to backend
    """
    raise NotImplementedError("The VTKHDF backend cannot make checkpoints.")


def read_hdf5_array(
    comm: MPI.Comm,
    filename: Path | str,
    group: str,
    backend_args: dict[str, Any] | None,
) -> tuple[np.ndarray, int]:
    """Read an array from an HDF5 file.

    Args:
        comm: MPI communicator used in storage
        filename: Path to file to read from
        group: Group in HDF5 file where array is stored
        backend_args: Arguments to backend

    Returns:
        Tuple containing:
            - Numpy array read from file
            - Global starting point on the process.
                Process 0 has [0, M), process 1 [M, N), process 2 [N, O) etc.
    """
    raise NotImplementedError("The VTKHDF backend cannot read HDF5 arrays")


def write_data(
    filename: Path | str,
    array_data: ArrayData,
    comm: MPI.Comm,
    time: str | float | None,
    mode: FileMode,
    backend_args: dict[str, Any] | None,
):
    """Write function to file by interpolating into geometry nodes.


    Args:
        filename: Path to file
        array_data: Data to write to file
        time: Time stamp
        mode: Append or write
        backend_args: The backend arguments
    """
    h5_mode = convert_file_mode(mode)
    assert h5_mode == "a"
    backend_args = get_default_backend_args(backend_args)
    mesh_name = backend_args["name"]
    extension = array_data.type
    storage_options = h5_storage.storage_options(comm, backend_args)
    with h5pyfile(filename, h5_mode, comm=comm) as h5file:
        hdf = h5file["VTKHDF"]
        file_type = _decode_bytes_if_needed(hdf.attrs["Type"])
        # Check for type of VTKHDF file
        if file_type != "MultiBlockDataSet":
            raise ValueError(f"Cannot write meshtags to {filename} with VTK type {file_type}")

        # Find mesh block to add data to
        block = _get_vtk_group(h5file, mesh_name)

        data_path = f"{extension}Data/{array_data.name}"
        if storage_options and data_path in block:
            h5_storage.check_storage_options(block[data_path], storage_options)

        # Check the time before writing anything
        assert isinstance(time, float)
        stamps = _get_longest_time_sequence(h5file)
        step = _find_time_step_index(stamps, time, filename)
        record = f"{_array_metadata_group(mesh_name, extension, array_data.name)}/{_written_steps}"
        written_steps, written_offsets = _get_time_steps_and_data_offset(
            h5file, record, f"{block.name}/Steps/{extension}DataOffsets/{array_data.name}"
        )
        if same_time(stamps[written_steps], time).any():
            raise ValueError(
                f"{extension} data '{array_data.name}' has already been written at time "
                f"{time} in {filename}."
            )
        if step is None:
            stamps = np.append(stamps, time)
            step = len(stamps) - 1
        _set_time_steps(h5file, stamps)
        num_steps = len(stamps)

        data_group = _create_group(block, f"{extension}Data", mode=h5_mode)
        dataset = _create_dataset(
            data_group,
            array_data.name,
            shape=array_data.global_shape,
            dtype=array_data.values.dtype,
            chunks=True,
            maxshape=(None, array_data.values.shape[1]),
            mode=h5_mode,
            filters=h5_storage.filter_options(storage_options, array_data.values.dtype),
        )
        insert_slice = _compute_append_slice(
            dataset,
            array_data.global_shape[0],
            np.array(
                [
                    array_data.local_range[0],
                    array_data.local_range[0] + array_data.values.shape[0],
                ]
            ),
            mode=h5_mode,
        )
        h5_storage.write_dataset(
            dataset,
            insert_slice,
            array_data.values,
        )

        # The data at each step is that of the latest write at or before it, see
        # `_fill_time_steps`.
        data_offsets = _create_group(block["Steps"], f"{extension}DataOffsets", mode=h5_mode)
        offsets = _create_dataset(
            data_offsets,
            array_data.name,
            shape=(num_steps,),
            dtype=np.int64,
            chunks=True,
            maxshape=(None,),
            mode=h5_mode,
            resize=False,
        )
        written_offsets = np.append(written_offsets, dataset.shape[0] - array_data.global_shape[0])
        written_steps = np.append(written_steps, step)
        offsets[:] = _fill_time_steps(num_steps, written_steps, written_offsets)
        _create_temporal_record(h5file, record, written_steps)
