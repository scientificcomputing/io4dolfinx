"""HDF5 dataset storage options for the h5py-based backends.

Currently used by the VTKHDF backend.

Validates filter settings passed through ``backend_args``, agrees on them across
ranks, and writes local slices, collectively for files opened with MPI-IO.
Filter options are passed to :meth:`h5py.Group.create_dataset` under their h5py names,
see https://docs.h5py.org/en/stable/high/dataset.html#dataset-compression. Only gzip
compression is accepted, as ParaView's bundled HDF5 can read it without filter plugins.

SPDX License identifier: MIT
"""

import numbers
from contextlib import closing
from typing import Any

from mpi4py import MPI

import h5py
import numpy as np
import numpy.typing as npt

__all__ = [
    "DEFAULT_STORAGE_OPTIONS",
    "check_storage_options",
    "filter_options",
    "storage_options",
    "write_dataset",
]


#: Recognised storage options and their defaults for new datasets. Validated options
#: are never filled in with these: an omitted option keeps the setting of an existing
#: dataset on append. Scale-offset is a built-in HDF5 filter, so it is available in
#: every HDF5 build, and it is lossless on the integer datasets it is applied to.
DEFAULT_STORAGE_OPTIONS: dict[str, Any] = {
    "compression": None,  # "gzip" or None
    "compression_opts": None,  # gzip level 0-9; h5py uses 4 if omitted
    "shuffle": False,
    "fletcher32": False,
    "scaleoffset": True,  # Integer datasets only; off by default with fletcher32
}
_BOOLEAN_OPTIONS = tuple(
    key for key, value in DEFAULT_STORAGE_OPTIONS.items() if isinstance(value, bool)
)
_FILTERS = {
    "shuffle": h5py.h5z.FILTER_SHUFFLE,
    "fletcher32": h5py.h5z.FILTER_FLETCHER32,
    "scaleoffset": h5py.h5z.FILTER_SCALEOFFSET,
}


def _resolve(options: dict[str, Any]) -> dict[str, Any]:
    """All storage settings: explicit options, completed with the defaults.

    h5py cannot combine fletcher32 and scale-offset, so an explicit
    ``fletcher32=True`` turns off scale-offset unless that is also given explicitly.
    """
    settings = DEFAULT_STORAGE_OPTIONS | options
    if "scaleoffset" not in options and settings["fletcher32"] is True:
        settings["scaleoffset"] = False
    return settings


def _get_and_validate_storage_options(arguments: dict[str, Any]) -> dict[str, Any]:
    """Extract the storage options from backend arguments and validate them locally.

    Args:
        arguments: Backend arguments. Keys other than those in
            :data:`DEFAULT_STORAGE_OPTIONS` and ``chunks`` are ignored.

    Returns:
        The storage options present in ``arguments``, with numpy integers and booleans
        converted to Python ``int`` and ``bool``. Omitted options are not added.

    Raises:
        ValueError: If an option has an invalid value, the options are incompatible,
            or this rank's HDF5 cannot write the requested filters.
    """
    options = {key: arguments[key] for key in DEFAULT_STORAGE_OPTIONS if key in arguments}
    for key in _BOOLEAN_OPTIONS:
        if key in options:
            if not isinstance(options[key], bool | np.bool_):
                raise ValueError(f"{key} must be a boolean")
            options[key] = bool(options[key])
    compression = options.get("compression")
    if compression is not None and compression != "gzip":
        raise ValueError("HDF5 compression must be 'gzip' or None")
    if "compression_opts" in options:
        level = options["compression_opts"]
        if (
            compression != "gzip"
            or not isinstance(level, numbers.Integral)
            or isinstance(level, bool | np.bool_)
            or not 0 <= int(level) <= 9
        ):
            raise ValueError("compression_opts requires gzip and an integer level from 0 to 9")
        options["compression_opts"] = int(level)
    settings = _resolve(options)
    if settings["fletcher32"] and settings["scaleoffset"]:
        # h5py forbids the combination, as scale-offset is lossy for floats.
        raise ValueError("fletcher32 cannot be combined with scaleoffset")
    if "chunks" in arguments and arguments["chunks"] is not True:
        raise ValueError("Only automatic chunking (chunks=True) is supported")
    filters = [h5py.h5z.FILTER_DEFLATE] if compression else []
    filters += [filter_id for key, filter_id in _FILTERS.items() if settings[key]]
    for filter_id in filters:
        if not h5py.h5z.filter_avail(filter_id) or not (
            h5py.h5z.get_filter_info(filter_id) & h5py.h5z.FILTER_CONFIG_ENCODE_ENABLED
        ):
            raise ValueError(f"HDF5 filter {filter_id} is unavailable for writing")
    return options


def storage_options(comm: MPI.Comm, arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate the storage options in backend arguments on all ranks.

    Collective: call on every rank before opening the file, so that an error on one
    rank is raised on all of them.

    Args:
        comm: Communicator the file is written with.
        arguments: Backend arguments, see :func:`_get_and_validate_storage_options`.

    Returns:
        The storage options present in ``arguments``, identical on all ranks.
        Omitted options are not added, so that they keep the settings of existing
        datasets on append; :func:`filter_options` applies the defaults to new ones.

    Raises:
        ValueError: On every rank, if validation fails on any rank or the options
            differ between ranks.
    """
    options: dict[str, Any] = {}
    error = None
    try:
        options = _get_and_validate_storage_options(arguments)
    except (TypeError, ValueError) as exc:
        error = str(exc)

    # One exchange handles both errors and settings; a single rank needs no exchange.
    settings = comm.allgather((error, options)) if comm.size > 1 else [(error, options)]
    for rank_error, _ in settings:
        if rank_error is not None:
            raise ValueError(rank_error)
    if any(rank_options != options for _, rank_options in settings):
        raise ValueError("HDF5 storage options must be identical on every MPI rank")
    return options


def filter_options(options: dict[str, Any], dtype: npt.DTypeLike) -> dict[str, Any]:
    """h5py filter arguments for a new bulk dataset of the given type.

    Omitted options take their defaults. Scale-offset is applied to integer datasets
    only, where h5py's automatic bit width makes it lossless; it is lossy for
    floating-point data.

    Args:
        options: Storage options returned by :func:`storage_options`.
        dtype: Data type of the dataset.

    Returns:
        Keyword arguments for :meth:`h5py.Group.create_dataset`.
    """
    settings = _resolve(options)
    filters: dict[str, Any] = {
        key: options[key] for key in ("compression", "compression_opts") if key in options
    }
    filters |= {key: True for key in ("shuffle", "fletcher32") if settings[key]}
    if settings["scaleoffset"] and np.issubdtype(dtype, np.integer):
        filters["scaleoffset"] = True
    return filters


def check_storage_options(dataset: h5py.Dataset, options: dict[str, Any]) -> None:
    """Reject explicit changes to the immutable filters of an existing dataset."""
    for key, requested in options.items():
        if key == "scaleoffset" and not np.issubdtype(dataset.dtype, np.integer):
            continue
        actual = getattr(dataset, key)
        if key == "scaleoffset":
            actual = actual is not None
        if actual != requested:
            raise ValueError(
                f"Dataset {dataset.name} already has {key}={actual!r}; "
                f"cannot append with {key}={requested!r}"
            )


def write_dataset(
    dataset: h5py.Dataset,
    selection: slice | tuple[slice, ...],
    values: np.ndarray,
) -> None:
    """Write this rank's slice of a dataset, collectively for MPI-IO files.

    In a collective write all ranks write together, so that the MPI library can
    combine their slices into fewer and larger file operations (collective
    buffering). This is much faster than independent writes, where each rank writes
    on its own, and parallel HDF5 requires it for filtered datasets.

    Note:
        For a file opened with the ``mpio`` driver, every rank of the file's
        communicator must call this function, also ranks without data.

    Args:
        dataset: Dataset to write to.
        selection: This rank's part of the dataset.
        values: Data for ``selection``; may be empty.
    """
    if dataset.file.driver != "mpio":
        dataset[selection] = values
    elif values.size > 0:
        with dataset.collective:
            dataset[selection] = values
    else:
        _join_collective_write(dataset, values)


def _join_collective_write(dataset: h5py.Dataset, values: np.ndarray) -> None:
    """Take part in a collective write without writing data.

    Every rank must call ``H5Dwrite`` in a collective write, or the other ranks wait
    forever. h5py's ``dataset[selection] = values`` makes no ``H5Dwrite`` call for an
    empty selection, so ranks without data call it directly, selecting nothing in
    both the file and the memory buffer.
    """
    with (
        closing(dataset.id.get_space()) as file_space,
        closing(h5py.h5s.create_simple(values.shape)) as memory_space,
        closing(h5py.h5p.create(h5py.h5p.DATASET_XFER)) as transfer,
    ):
        file_space.select_none()
        memory_space.select_none()
        transfer.set_dxpl_mpio(h5py.h5fd.MPIO_COLLECTIVE)
        dataset.id.write(memory_space, file_space, values, dxpl=transfer)
