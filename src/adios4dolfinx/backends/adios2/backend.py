from pathlib import Path
from typing import Any, Union

from mpi4py import MPI

import adios2
import numpy as np
import numpy.typing as npt

from adios4dolfinx.utils import FileMode, check_file_exists

from .helpers import ADIOSFile, adios_to_numpy_dtype, resolve_adios_scope

adios2 = resolve_adios_scope(adios2)


class ADIOS2Interface:
    @staticmethod
    def get_default_backend_args(arguments: dict[str, Any] | None) -> dict[str, Any] | None:
        if arguments is None:
            return {"engine": "BP4"}
        else:
            assert "engine" in arguments.keys()
            return arguments

    @staticmethod
    def convert_file_mode(mode: FileMode) -> adios2.Mode:
        match mode:
            case FileMode.append:
                return adios2.Mode.Append
            case FileMode.write:
                return adios2.Mode.Write
            case FileMode.read:
                return adios2.Mode.Read
            case _:
                raise NotImplementedError(f"FileMode {mode} not implemented.")

    @staticmethod
    def write_attributes(
        filename: Union[Path, str],
        comm: MPI.Intracomm,
        name: str,
        attributes: dict[str, np.ndarray],
        backend_args: dict[str, Any] | None = None,
    ):
        """Write attributes to file using ADIOS2.

        Args:
            filename: Path to file to write to
            comm: MPI communicator used in storage
            name: Name of the attributes
            attributes: Dictionary of attributes to write to file
            engine: ADIOS2 engine to use
        """

        adios = adios2.ADIOS(comm)
        with ADIOSFile(
            adios=adios,
            filename=filename,
            mode=adios2.Mode.Append,
            io_name="AttributeWriter",
            **backend_args,
        ) as adios_file:
            adios_file.file.BeginStep()

            for k, v in attributes.items():
                adios_file.io.DefineAttribute(f"{name}_{k}", v)

            adios_file.file.PerformPuts()
            adios_file.file.EndStep()

    @staticmethod
    def read_attributes(
        filename: Union[Path, str],
        comm: MPI.Intracomm,
        name: str,
        backend_args: dict[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        """Read attributes from file using ADIOS2.

        Args:
            filename: Path to file to read from
            comm: MPI communicator used in storage
            name: Name of the attributes
            engine: ADIOS2 engine to use
        Returns:
            The attributes
        """
        check_file_exists(filename)
        adios = adios2.ADIOS(comm)
        with ADIOSFile(
            adios=adios,
            filename=filename,
            mode=adios2.Mode.Read,
            **backend_args,
            io_name="AttributesReader",
        ) as adios_file:
            adios_file.file.BeginStep()
            attributes = {}
            for k in adios_file.io.AvailableAttributes().keys():
                if k.startswith(f"{name}_"):
                    a = adios_file.io.InquireAttribute(k)
                    attributes[k[len(name) + 1 :]] = a.Data()
            adios_file.file.EndStep()
        return attributes

    @staticmethod
    def read_timestamps(
        filename: Union[Path, str],
        comm: MPI.Intracomm,
        function_name: str,
        backend_args: dict[str, Any] | None = None,
    ) -> npt.NDArray[np.float64]:
        """
        Read time-stamps from a checkpoint file.

        Args:
            comm: MPI communicator
            filename: Path to file
            function_name: Name of the function to read time-stamps for
            backend_args: Arguments for backend, for instance file type.
            backend: What backend to use for writing.
        Returns:
            The time-stamps
        """
        check_file_exists(filename)

        adios = adios2.ADIOS(comm)

        with ADIOSFile(
            adios=adios,
            filename=filename,
            mode=adios2.Mode.Read,
            **backend_args,
            io_name="TimestepReader",
        ) as adios_file:
            time_name = f"{function_name}_time"
            time_stamps = []
            for _ in range(adios_file.file.Steps()):
                adios_file.file.BeginStep()
                if time_name in adios_file.io.AvailableVariables().keys():
                    arr = adios_file.io.InquireVariable(time_name)
                    time_shape = arr.Shape()
                    arr.SetSelection([[0], [time_shape[0]]])
                    times = np.empty(
                        time_shape[0],
                        dtype=adios_to_numpy_dtype[arr.Type()],
                    )
                    adios_file.file.Get(arr, times, adios2.Mode.Sync)
                    time_stamps.append(times[0])
                adios_file.file.EndStep()

        return np.array(time_stamps)
