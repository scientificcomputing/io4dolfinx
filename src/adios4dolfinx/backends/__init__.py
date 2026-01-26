from pathlib import Path
from typing import Any, Protocol, Union

from mpi4py import MPI

import numpy as np
import numpy.typing as npt

from adios4dolfinx.utils import FileMode


class IOBackend(Protocol):
    @staticmethod
    def get_default_backend_args(arguments: dict[str, Any] | None) -> dict[str, Any] | None: ...

    @staticmethod
    def convert_file_mode(mode: FileMode) -> Any: ...

    @staticmethod
    def write_attributes(
        filename: Union[Path, str],
        comm: MPI.Intracomm,
        name: str,
        attributes: dict[str, np.ndarray],
        backend_args: dict[str, Any] | None,
    ): ...

    @staticmethod
    def read_attributes(
        filename: Union[Path, str],
        comm: MPI.Intracomm,
        name: str | None,
        backend_args: dict[str, Any] | None,
    ) -> dict[str, Any]: ...

    @staticmethod
    def read_timestamps(
        filename: Union[Path, str],
        comm: MPI.Intracomm,
        function_name: str,
        backend_args: dict[str, Any] | None,
    ) -> npt.NDArray[np.float64]: ...

    # read_function
    # read_mesh
    # read_meshtags
    # read_timestamps
    # write_function
    # write_mesh
    # write_meshtags
    # read_function_from_legacy_h5
    # read_mesh_from_legacy_h5
    # write_function_on_input_mesh
    # write_mesh_input_order
    # snapshot_checkpoint
