from enum import Enum
from pathlib import Path
from typing import Any, Protocol, Union

from mpi4py import MPI

import numpy as np

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
        filename: Union[Path, str], comm: MPI.Intracomm, name: str | None
    ) -> dict[str, Any]: ...

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
