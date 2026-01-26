"""
H5py interface to ADIOS4DOLFINx

SPDX License identifier: MIT

Copyright: Jørgen S. Dokken, Henrik N.T. Finsberg, Simula Research Laboratory
"""

import contextlib
from pathlib import Path
from typing import Any, Union

from mpi4py import MPI

import numpy as np

from adios4dolfinx.utils import FileMode, check_file_exists

_backend_default_args = None


@contextlib.contextmanager
def h5pyfile(h5name, filemode="r", force_serial: bool = False, comm=None):
    """Context manager for opening an HDF5 file with h5py.

    Args:
        h5name: The name of the HDF5 file.
        filemode: The file mode.
        force_serial: Force serial access to the file.
        comm: The MPI communicator

    """
    import h5py

    if comm is None:
        comm = MPI.COMM_WORLD

    if h5py.h5.get_config().mpi and comm.size > 1 and not force_serial:
        h5file = h5py.File(h5name, filemode, driver="mpio", comm=comm)
    else:
        if comm.size > 1 and not force_serial:
            raise ValueError(
                f"h5py is not installed with MPI support, while using {comm.size} processes.",
                "If you really want to do this, turn on the `force_serial` flag.",
            )
        h5file = h5py.File(h5name, filemode)
    yield h5file
    h5file.close()


class H5PYInterface:
    @staticmethod
    def get_default_backend_args(arguments: dict[str, Any] | None) -> dict[str, Any] | None:
        if arguments is None:
            return arguments
        else:
            raise RuntimeError("Unexpected backend arguments to h5py backend")

    def convert_file_mode(mode: FileMode) -> str:
        match mode:
            case FileMode.append:
                return "a"
            case FileMode.read:
                return "r"
            case FileMode.write:
                return "w"
            case _:
                raise NotImplementedError(f"File mode {mode} not implemented")

    @staticmethod
    def write_attributes(
        filename: Union[Path, str],
        comm: MPI.Intracomm,
        name: str,
        attributes: dict[str, np.ndarray],
        backend_args: dict[str, Any] | None = _backend_default_args,
    ):
        """Write attributes to file using H5PY.

        Args:
            filename: Path to file to write to
            comm: MPI communicator used in storage
            name: Name of the attributes
            attributes: Dictionary of attributes to write to file
            engine: ADIOS2 engine to use
        """

        with h5pyfile(filename, filemode="a", comm=comm, force_serial=False) as h5file:
            if name in h5file.keys():
                group = h5file[name]
            else:
                group = h5file.create_group(name, track_order=True)
            for key, val in attributes.items():
                group.attrs[key] = val

    @staticmethod
    def read_attributes(
        filename: Union[Path, str],
        comm: MPI.Intracomm,
        name: str,
        backend_args: dict[str, Any] | None = _backend_default_args,
    ):
        """Read attributes from file using H5PY.

        Args:
            filename: Path to file to read from
            comm: MPI communicator used in storage
            name: Name of the attributes
        Returns:
            The attributes
        """
        check_file_exists(filename)
        output_attrs = {}
        with h5pyfile(filename, filemode="r", comm=comm, force_serial=False) as h5file:
            for key, val in h5file[name].attrs.items():
                output_attrs[key] = val
        return output_attrs
