"""
Helpers reading/writing data with ADIOS2
"""

from __future__ import annotations

import contextvars
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

from mpi4py import MPI

import adios2
import dolfinx.graph
import fenicsx_compat
import numpy as np
import numpy.typing as npt

from ...utils import compute_local_range, valid_function_types

adios2 = fenicsx_compat.resolve_adios_scope(adios2)


__all__ = [
    "force_serial_adios",
    "create_adios",
    "AdiosFile",
    "ADIOSFile",
    "check_variable_exists",
    "read_array",
    "read_adjacency_list",
    "read_local_rows",
    "adios_to_numpy_dtype",
]

adios_to_numpy_dtype = {
    "float": np.float32,
    "double": np.float64,
    "float complex": np.complex64,
    "double complex": np.complex128,
    "uint32_t": np.uint32,
    "int32_t": np.int32,
    "int64_t": np.int64,
    "uint64_t": np.uint64,
}


#: Create ADIOS objects without MPI on every rank, even with ADIOS2 built with MPI. Set by the
#: adios2-serial backend, see :mod:`io4dolfinx.backends.serial`.
force_serial_adios = contextvars.ContextVar("force_serial_adios", default=False)


def create_adios(comm: MPI.Comm) -> adios2.ADIOS:
    """Create an ADIOS object.

    Args:
        comm: The communicator of the processes opening files with it.

    Returns:
        An ADIOS object on ``comm``, or one without MPI that opens files on this process
        only, if ADIOS2 is built without MPI or :data:`force_serial_adios` is set.
    """
    if adios2.is_built_with_mpi and not force_serial_adios.get():
        return adios2.ADIOS(comm)
    return adios2.ADIOS()


class AdiosFile(NamedTuple):
    io: adios2.IO
    file: adios2.Engine


@contextmanager
def ADIOSFile(
    adios: adios2.ADIOS,
    filename: Path | str,
    engine: str,
    mode: adios2.Mode,
    io_name: str,
    comm: MPI.Comm | None = None,
):
    io = adios.DeclareIO(io_name)
    io.SetEngine(engine)
    # ADIOS2 sometimes struggles with existing files/folders it should overwrite
    if mode == adios2.Mode.Write:
        filename = Path(filename)
        if filename.exists() and comm is not None and comm.rank == 0:
            if filename.is_dir():
                shutil.rmtree(filename)
            else:
                filename.unlink()
        if comm is not None:
            comm.Barrier()

    file = io.Open(str(filename), mode)
    try:
        yield AdiosFile(io=io, file=file)
    finally:
        file.Close()
        adios.RemoveIO(io_name)


def read_local_rows(
    adios_file: AdiosFile,
    comm: MPI.Comm,
    name: str,
    overlap: int = 0,
    mode: adios2.Mode = adios2.Mode.Sync,
) -> tuple[npt.NDArray, tuple[int, int], int]:
    """Read this process's rows of a variable in the current step, the rows being divided
    over the processes by :func:`io4dolfinx.utils.compute_local_range`.

    Args:
        adios_file: The open file.
        comm: The communicator the rows are divided over.
        name: Name of the variable.
        overlap: Number of rows after this process's rows to read as well, e.g. 1 for
            offsets.
        mode: ``adios2.Mode.Deferred`` to fill the array at the next ``PerformGets``.

    Returns:
        The rows read, their range (without the overlap) and the global number of rows
        (without the overlap).
    """
    variable = adios_file.io.InquireVariable(name)
    shape = variable.Shape()
    num_rows = shape[0] - overlap
    start, stop = compute_local_range(comm, num_rows)
    count = [stop - start + overlap, *shape[1:]]
    variable.SetSelection([[start] + [0] * (len(shape) - 1), count])
    data = np.empty(count, dtype=adios_to_numpy_dtype[variable.Type()])
    adios_file.file.Get(variable, data, mode)
    return data, (start, stop), num_rows


def check_variable_exists(
    adios: adios2.ADIOS,
    filename: Path | str,
    variable: str,
    engine: str,
) -> bool:
    io_name = f"{variable}_reader"

    if not Path(filename).exists():
        return False

    variable_found = False
    with ADIOSFile(
        adios=adios,
        engine=engine,
        filename=filename,
        mode=adios2.Mode.Read,
        io_name=io_name,
    ) as adios_file:
        # Find step that has cell permutation
        for _ in range(adios_file.file.Steps()):
            adios_file.file.BeginStep()
            if variable in adios_file.io.AvailableVariables().keys():
                variable_found = True
                break
            adios_file.file.EndStep()

        # Not sure if this is needed, but just in case
        if variable in adios_file.io.AvailableVariables().keys():
            variable_found = True
    return variable_found


def read_adjacency_list(
    adios: adios2.ADIOS,
    comm: MPI.Comm,
    filename: Path | str,
    data_name: str,
    offsets_name: str,
    engine: str,
) -> dolfinx.graph.AdjacencyList:
    """
    Read an adjacency-list from an ADIOS file with given communicator.
    The adjancency list is split in to a flat array (data) and its corresponding offset.

    Args:
        adios: The ADIOS instance
        comm: The MPI communicator used to read the data
        filename: Path to input file
        data_name: Name of variable containing the indices of the adjacencylist
        dofmap_offsets: Name of variable containing offsets of the adjacencylist
        engine: Type of ADIOS engine to use for reading data

    Returns:
        The local part of dofmap from input dofs

    .. note::
        No MPI communication is done during this call
    """

    # Open ADIOS engine
    io_name = f"{data_name=}_reader"

    with ADIOSFile(
        adios=adios,
        engine=engine,
        filename=filename,
        mode=adios2.Mode.Read,
        io_name=io_name,
    ) as adios_file:
        # First find step with dofmap offsets, to be able to read
        # in a full row of the dofmap
        for _ in range(adios_file.file.Steps()):
            adios_file.file.BeginStep()
            if offsets_name in adios_file.io.AvailableVariables().keys():
                break
            adios_file.file.EndStep()
        if offsets_name not in adios_file.io.AvailableVariables().keys():
            raise KeyError(f"Dof offsets not found at '{offsets_name}' in {filename}")

        # The offsets are one longer than the number of nodes, so read one row more
        in_offsets, _, _ = read_local_rows(adios_file, comm, offsets_name, overlap=1)
        shape = adios_file.io.InquireVariable(offsets_name).Shape()
        in_offsets = in_offsets.squeeze()

        # Assuming dofmap is saved in stame step
        # Get the relevant part of the dofmap
        if data_name not in adios_file.io.AvailableVariables().keys():
            raise KeyError(f"Dofs not found at {data_name} in {filename}")
        cell_dofs = adios_file.io.InquireVariable(data_name)
        if len(shape) == 1:
            cell_dofs.SetSelection([[in_offsets[0]], [in_offsets[-1] - in_offsets[0]]])
            in_dofmap = np.empty(in_offsets[-1] - in_offsets[0], dtype=cell_dofs.Type().strip("_t"))
        else:
            cell_dofs.SetSelection([[in_offsets[0], 0], [in_offsets[-1] - in_offsets[0], shape[1]]])
            in_dofmap = np.empty(
                (in_offsets[-1] - in_offsets[0], shape[1]),
                dtype=cell_dofs.Type().strip("_t"),
            )
            assert shape[1] == 1

        in_dofmap = np.empty(in_offsets[-1] - in_offsets[0], dtype=cell_dofs.Type().strip("_t"))
        adios_file.file.Get(cell_dofs, in_dofmap, adios2.Mode.Sync)
        in_offsets -= in_offsets[0]
        adios_file.file.EndStep()

    # Return local dofmap
    return dolfinx.graph.adjacencylist(in_dofmap, in_offsets.astype(np.int32))


def read_array(
    adios: adios2.ADIOS,
    filename: Path | str,
    array_name: str,
    engine: str,
    comm: MPI.Comm,
    time: float = 0.0,
    time_name: str = "",
    legacy: bool = False,
) -> tuple[npt.NDArray[valid_function_types], int]:
    """
    Read an array from file, return the global starting position of the local array

    Args:
        adios: The ADIOS instance
        filename: Path to file to read array from
        array_name: Name of array in file
        engine: Name of engine to use to read file
        comm: MPI communicator used for reading the data
        time_name: Name of time variable for modern checkpoints
        legacy: If True ignore time_name and read the first available step
    Returns:
        Local part of array and its global starting position
    """

    with ADIOSFile(
        adios=adios,
        engine=engine,
        filename=filename,
        mode=adios2.Mode.Read,
        io_name="ArrayReader",
    ) as adios_file:
        # Get time-stamp from first available step
        if legacy:
            for i in range(adios_file.file.Steps()):
                adios_file.file.BeginStep()
                if array_name in adios_file.io.AvailableVariables().keys():
                    break
                adios_file.file.EndStep()
            if array_name not in adios_file.io.AvailableVariables().keys():
                raise KeyError(f"No array found at {array_name}")
        else:
            for i in range(adios_file.file.Steps()):
                adios_file.file.BeginStep()
                if time_name in adios_file.io.AvailableVariables().keys():
                    arr = adios_file.io.InquireVariable(time_name)
                    time_shape = arr.Shape()
                    arr.SetSelection([[0], [time_shape[0]]])
                    times = np.empty(time_shape[0], dtype=adios_to_numpy_dtype[arr.Type()])
                    adios_file.file.Get(arr, times, adios2.Mode.Sync)
                    if times[0] == time:
                        break
                if i == adios_file.file.Steps() - 1:
                    raise KeyError(
                        f"No data associated with {time_name}={time} found in {filename}"
                    )

                adios_file.file.EndStep()

            if time_name not in adios_file.io.AvailableVariables().keys():
                raise KeyError(f"No data associated with {time_name}={time} found in {filename}")

            if array_name not in adios_file.io.AvailableVariables().keys():
                raise KeyError(f"No array found at {time=} for {array_name}")

        arr_shape = adios_file.io.InquireVariable(array_name).Shape()
        assert len(arr_shape) == 1 or arr_shape[1] == 1
        vals, arr_range, _ = read_local_rows(adios_file, comm, array_name)
        adios_file.file.EndStep()

    return vals.reshape(-1), arr_range[0]
