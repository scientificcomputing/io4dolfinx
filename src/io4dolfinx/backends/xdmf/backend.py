"""
Module that uses DOLFINx/H5py to import XDMF files.
"""

import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from mpi4py import MPI

import basix
import dolfinx
import numpy as np
import numpy.typing as npt

from io4dolfinx.structures import ArrayData, FunctionData, MeshData, MeshTagsData, ReadMeshData
from io4dolfinx.utils import check_file_exists, compute_local_range

from .. import FileMode, ReadMode
from ..h5py.backend import h5pyfile

read_mode = ReadMode.parallel


def extract_function_names_and_timesteps(filename: Path | str) -> dict[str, list[str]]:
    tree = ElementTree.parse(filename)
    root = tree.getroot()
    mesh_nodes = root.findall(".//Grid[@CollectionType='Temporal']")
    function_names = []
    for mesh in mesh_nodes:
        function_names.append(mesh.attrib["Name"])

    time_stamps: dict[str, list[str]] = {name: [] for name in function_names}
    for name in function_names:
        time_steps = root.findall(f".//Grid[@Name='{name}']")
        for time in time_steps:
            step = time.find(".//Time")
            if step is not None:
                val = step.attrib["Value"]
                time_stamps[name].append(val)
    for name in function_names:
        float_steps = np.argsort(np.array(list(set(time_stamps[name])), dtype=np.float64))
        time_stamps[name] = np.array(list(set(time_stamps[name])), dtype=str)[float_steps].tolist()
    return time_stamps


def get_default_backend_args(arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Get default backend arguments given a set of input arguments.

    Args:
        arguments: Input backend arguments

    Returns:
        Updated backend arguments
    """
    args = arguments or {}
    return args


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
    assert not read_from_partition
    check_file_exists(filename)
    with dolfinx.io.XDMFFile(comm, filename, "r") as file:
        cell_shape, cell_degree = file.read_cell_type()
        cells = file.read_topology_data()
        x = file.read_geometry_data()
    return ReadMeshData(
        cells=cells,
        cell_type=cell_shape.name,
        x=x,
        lvar=int(basix.LagrangeVariant.equispaced),
        degree=cell_degree,
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
    check_file_exists(filename)
    filename = Path(filename)

    root = ElementTree.parse(filename).getroot()
    backend_args = get_default_backend_args(backend_args)

    grid = _time_step_grid(root, name, time, filename)
    attributes = _function_attributes(grid, name)
    local_range_start, dataset = _read_data_item(filename, _data_item(attributes[0]), comm)
    if len(attributes) == 2:
        _, imag = _read_data_item(filename, _data_item(attributes[1]), comm)
        # complex64 from float32 parts, complex128 from float64
        dataset = dataset + 1j * imag
    return dataset, local_range_start


def _time_step_grid(
    root: ElementTree.Element, name: str, time: str | float | None, filename: Path
) -> ElementTree.Element:
    """Grid holding function ``name`` at ``time``, or at the first time step if ``None``.

    Both DOLFINx and legacy DOLFIN write one grid per time step, with a `Time`
    child, into a temporal collection named after the function. DOLFINx names
    the time-step grids ``name``, legacy DOLFIN ``{name}_{counter}``.

    Raises:
        RuntimeError: If there is no such grid.
    """
    collection = root.find(f".//Grid[@CollectionType='Temporal'][@Name='{name}']")
    if collection is not None:
        for grid in collection.findall("./Grid"):
            step_node = grid.find("./Time")
            if step_node is None:
                continue
            if time is None or np.isclose(float(step_node.attrib["Value"]), float(time)):
                return grid
    at_time = "" if time is None else f"(t={time})"
    raise RuntimeError(f"Function {name}{at_time} not found in {filename}")


def _function_attributes(grid: ElementTree.Element, name: str) -> list[ElementTree.Element]:
    """Attributes of ``grid`` holding function ``name``.

    DOLFINx writes a real function as attribute ``name``, and a complex one as
    two attributes ``real_{name}`` and ``imag_{name}``.

    Returns:
        ``[data]`` for a real function, ``[real, imag]`` for a complex one.

    Raises:
        RuntimeError: If ``grid`` holds neither form, or only one complex part.
    """
    if (node := grid.find(f"./Attribute[@Name='{name}']")) is not None:
        return [node]
    real = grid.find(f"./Attribute[@Name='real_{name}']")
    imag = grid.find(f"./Attribute[@Name='imag_{name}']")
    if real is not None and imag is not None:
        return [real, imag]
    if real is not None or imag is not None:
        present, missing = ("real", "imag") if real is not None else ("imag", "real")
        raise RuntimeError(
            f"Complex function {name} is incomplete: found '{present}_{name}'"
            f" but not '{missing}_{name}'."
        )
    found = [attribute.attrib.get("Name") for attribute in grid.findall("./Attribute")]
    raise RuntimeError(
        f"Function {name} not found: expected attribute '{name}', or 'real_{name}'"
        f" and 'imag_{name}'. Attributes present: {found}."
    )


def _data_item(node: ElementTree.Element) -> ElementTree.Element:
    """The `DataItem` directly under ``node``.

    Raises:
        RuntimeError: If ``node`` has none.
    """
    if (data_item := node.find("./DataItem")) is None:
        raise RuntimeError(f"<{node.tag} Name='{node.attrib.get('Name')}'> has no DataItem.")
    return data_item


def _read_data_item(
    filename: Path, data_item: ElementTree.Element, comm: MPI.Comm
) -> tuple[int, np.ndarray]:
    """Read this process's contiguous block of rows of an HDF5-backed `DataItem`.

    Args:
        filename: The XDMF file, which HDF5 paths are relative to
        data_item: `DataItem` whose text is ``"file.h5:/path/to/dataset"``
        comm: Communicator the rows are split over

    Returns:
        The first global row read and the rows themselves.

    Raises:
        RuntimeError: If the dataset's shape does not match the `Dimensions`.
    """
    location = data_item.text.strip() if data_item.text is not None else ""
    data_file, sep, data_loc = location.partition(":")
    if data_item.attrib.get("Format") != "HDF" or not sep:
        raise RuntimeError(f"Expected an HDF5 DataItem 'file.h5:/path', got '{location}'.")
    dimensions = [int(d) for d in data_item.attrib["Dimensions"].split()]
    with h5pyfile(filename.parent / data_file, "r", comm=comm) as h5file:
        data = h5file[data_loc]
        # Shapes may differ by a trailing unit dimension, so compare rows and size.
        if data.shape[0] != dimensions[0] or data.size != int(np.prod(dimensions)):
            raise RuntimeError(
                f"Dataset {location} has shape {data.shape}, but its DataItem"
                f" has Dimensions {dimensions}."
            )
        lr = compute_local_range(comm, data.shape[0])
        return lr[0], data[slice(*lr)]


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
    raise NotImplementedError("The XDMF backend cannot read attributes.")


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
        Numpy array of timestamps read from file
    """
    tree = ElementTree.parse(filename)
    root = tree.getroot()
    time_stamps = []
    time_steps = root.findall(f".//Grid[@Name='{function_name}']")
    for time in time_steps:
        step = time.find(".//Time")
        if step is not None:
            val = step.attrib["Value"]
            time_stamps.append(val)
    float_steps = np.argsort(np.array(list(set(time_stamps)), dtype=np.float64))
    return np.array(list(set(time_stamps)), dtype=str)[float_steps].tolist()


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
    raise NotImplementedError("The XDMF backend cannot write attributes.")


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
    raise NotImplementedError("The XDMF backend cannot write meshes.")


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
    raise NotImplementedError("The XDMF backend cannot write meshtags.")


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
    raise NotImplementedError("The XDMF backend cannot read meshtags.")


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
    raise NotImplementedError("The XDMF backend cannot make checkpoints.")


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
    raise NotImplementedError("The XDMF backend cannot make checkpoints.")


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
    raise NotImplementedError("The XDMF backend cannot make checkpoints.")


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
    raise NotImplementedError("The XDMF backend cannot make checkpoints.")


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
    raise NotImplementedError("The XDMF backend cannot read legacy DOLFIN meshes.")


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
    raise NotImplementedError("The XDMF backend cannot make checkpoints.")


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
    raise NotImplementedError("The XDMF backend cannot read HDF5 arrays")


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
    # Find all functions in xml tree
    check_file_exists(filename)
    filename = Path(filename)

    tree = ElementTree.parse(filename)
    root = tree.getroot()
    backend_args = get_default_backend_args(backend_args)
    # Functions in checkpoint format
    checkpoint_funcs = root.findall(".//Attribute[@ItemType='FiniteElementFunction']")
    names = [func.attrib["Name"] for func in checkpoint_funcs]
    # Temporal funcs
    temporal_funcs = root.findall(".//Grid[@GridType='Collection']")
    for func in temporal_funcs:
        names.append(func.attrib["Name"])
    return list(set(names))


def read_cell_data(
    filename: Path | str,
    name: str,
    comm: MPI.Comm,
    time: str | float | None,
    backend_args: dict[str, Any] | None,
) -> tuple[npt.NDArray[np.int64], np.ndarray]:
    """Read data from the cells of a mesh.

    Reads cell-centred functions written by DOLFINx (DG-0), including complex
    ones, and DG-0 checkpoints written by legacy DOLFIN.

    Args:
        filename: Path to file
        name: Name of cell data
        comm: Communicator to launch IO on.
        time: The time stamp
        backend_args: The backend arguments
    Returns:
        A tuple (topology, dofs) where topology contains the
        vertex indices of the cells, dofs the degrees of
        freedom within that cell.
    """
    check_file_exists(filename)
    filename = Path(filename)

    root = ElementTree.parse(filename).getroot()
    backend_args = get_default_backend_args(backend_args)

    grid = _time_step_grid(root, name, time, filename)
    attributes = _function_attributes(grid, name)
    values = [_cell_values(attribute) for attribute in attributes]

    _, dofs = _read_data_item(filename, values[0], comm)
    if len(values) == 2:
        _, imag = _read_data_item(filename, values[1], comm)
        dofs = dofs + 1j * imag
    _, topology = _read_data_item(filename, _grid_topology(root, grid), comm)
    # One row per cell in both, so both processes' blocks cover the same cells.
    if topology.shape[0] != dofs.shape[0]:
        raise RuntimeError(
            f"Cell data {name} has {dofs.shape[0]} rows on this process, but the"
            f" topology has {topology.shape[0]}."
        )
    return topology.astype(np.int64), dofs.reshape(dofs.shape[0], -1)


def _cell_values(attribute: ElementTree.Element) -> ElementTree.Element:
    """`DataItem` with one value row per cell, from a cell-data attribute.

    Raises:
        ValueError: If the attribute is not DG-0 data.
    """
    if attribute.attrib.get("ItemType") == "FiniteElementFunction":
        # Legacy DOLFIN checkpoint: the dof values are the `.../vector` item.
        family = attribute.attrib["ElementFamily"]
        degree = int(attribute.attrib["ElementDegree"])
        if family != "DG" or degree != 0:
            raise ValueError(
                f"Cannot read in finite element function ({family}, {degree}) as cell data."
            )
        for data_item in attribute.findall("./DataItem"):
            if (data_item.text or "").strip().endswith("/vector"):
                return data_item
        raise RuntimeError(f"Checkpoint {attribute.attrib['Name']} has no vector DataItem.")
    if (center := attribute.attrib.get("Center")) != "Cell":
        raise ValueError(
            f"Attribute {attribute.attrib['Name']} has Center='{center}'; cell data"
            " needs Center='Cell'."
        )
    return _data_item(attribute)


def _grid_topology(root: ElementTree.Element, grid: ElementTree.Element) -> ElementTree.Element:
    """Topology `DataItem` of ``grid``, following DOLFINx's ``xi:include`` of the mesh.

    Legacy DOLFIN writes the topology into each grid. DOLFINx refers to the mesh
    grid with ``xpointer(<grid path>/*[self::Topology or self::Geometry])``.

    Raises:
        RuntimeError: If the topology cannot be found.
    """
    if (topology := grid.find("./Topology")) is not None:
        return _data_item(topology)
    # Any namespace: DOLFINx declares XInclude's as https, the standard says http.
    if (include := grid.find("./{*}include")) is not None:
        pointer = include.attrib["xpointer"]
        match = re.fullmatch(r"xpointer\(/Xdmf(/.+?)(?:\[(\d+)\])?/\*\[.*\]\)", pointer)
        if match is not None:
            # ElementTree applies a position predicate to all same-tag siblings,
            # XPath to the elements left by the preceding predicates.
            path, position = match.groups()
            mesh_grids = root.findall(f".{path}")
            index = 0 if position is None else int(position) - 1
            if index < len(mesh_grids):
                if (topology := mesh_grids[index].find("./Topology")) is not None:
                    return _data_item(topology)
    raise RuntimeError(f"No topology found for grid {grid.attrib.get('Name')}.")


def write_data(
    filename: Path | str,
    point_data: ArrayData,
    comm: MPI.Comm,
    time: str | float | None,
    mode: FileMode,
    backend_args: dict[str, Any] | None,
):
    """Write a 2D-array to file (distributed across proceses with MPI).


    Args:
        filename: Path to file
        point_data: Data to write to file
        time: Time stamp
        mode: Append or write
        backend_args: The backend arguments
    """
    raise NotImplementedError("XDMF has not implemented this yet")
