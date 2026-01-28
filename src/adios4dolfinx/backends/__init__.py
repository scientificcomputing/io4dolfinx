from enum import Enum
from pathlib import Path
from typing import Any, Protocol
from importlib import import_module
from mpi4py import MPI

import dolfinx
import numpy as np
import numpy.typing as npt

from ..structures import FunctionData, MeshData, MeshTagsData, ReadMeshData

__all__ = ["FileMode", "IOBackend", "get_backend"]


class FileMode(Enum):
    append = 10
    write = 20
    read = 30


# See https://peps.python.org/pep-0544/#modules-as-implementations-of-protocols
class IOBackend(Protocol):
    def get_default_backend_args(self, arguments: dict[str, Any] | None) -> dict[str, Any]: ...

    def write_attributes(
        self,
        filename: Path | str,
        comm: MPI.Intracomm,
        name: str,
        attributes: dict[str, np.ndarray],
        backend_args: dict[str, Any] | None,
    ): ...

    def read_attributes(
        self,
        filename: Path | str,
        comm: MPI.Intracomm,
        name: str,
        backend_args: dict[str, Any] | None,
    ) -> dict[str, Any]: ...

    def read_timestamps(
        self,
        filename: Path | str,
        comm: MPI.Intracomm,
        function_name: str,
        backend_args: dict[str, Any] | None,
    ) -> npt.NDArray[np.float64]: ...

    def write_mesh(
        self,
        filename: Path | str,
        comm: MPI.Intracomm,
        mesh: MeshData,
        backend_args: dict[str, Any] | None,
        mode: FileMode,
        time: float,
    ):
        """
        Write a mesh to file.

        Parameters:
            comm: MPI communicator used in storage
            mesh: Internal data structure for the mesh data to save to file
            filename: Path to file to write to
            backend_args: Arguments to backend
            mode: File-mode to store the mesh
        """
        ...

    def write_meshtags(
        self,
        filename: str | Path,
        comm: MPI.Intracomm,
        data: MeshTagsData,
        backend_args: dict[str, Any] | None,
    ): ...

    def read_mesh_data(
        self,
        filename: Path | str,
        comm: MPI.Intracomm,
        time: float,
        read_from_partition: bool,
        backend_args: dict[str, Any] | None,
    ) -> ReadMeshData: ...

    def read_meshtags_data(
        self,
        filename: str | Path,
        comm: MPI.Intracomm,
        name: str,
        backend_args: dict[str, Any] | None,
    ) -> MeshTagsData: ...

    def read_dofmap(
        self,
        filename: str | Path,
        comm: MPI.Intracomm,
        name: str,
        backend_args: dict[str, Any] | None,
    ) -> dolfinx.graph.AdjacencyList:
        """Read the dofmap of a function with a given name from file"""
        ...

    def read_dofs(
        self,
        filename: str | Path,
        comm: MPI.Intracomm,
        name: str,
        time: float,
        backend_args: dict[str, Any] | None,
    ) -> tuple[npt.NDArray[np.float32 | np.float64 | np.complex64 | np.complex128], int]:
        """Read the dofs (values) of a function with a given name from a given timestep.

        Returns:
            Contiguous sequence of degrees of freedom (with respect to input data)
            and the global starting point on the process.
            Process 0 has [0, M), process 1 [M, N), process 2 [N, O) etc.
        """
        ...

    def read_cell_perms(
        self, comm: MPI.Intracomm, filename: Path | str, backend_args: dict[str, Any] | None
    ) -> npt.NDArray[np.uint32]:
        """
        Read cell permutation from file with given communicator,
        Split in continuous chunks based on number of cells in the input data.

        Returns:
            Contiguous sequence of permutations (with respect to input data)
            Process 0 has [0, M), process 1 [M, N), process 2 [N, O) etc.
        """
        ...

    def write_function(
        self,
        filename: Path,
        comm: MPI.Intracomm,
        u: FunctionData,
        time: float,
        mode: FileMode,
        backend_args: dict[str, Any] | None,
    ): ...

    # read_timestamps
    # read_function_from_legacy_h5
    # read_mesh_from_legacy_h5
    # snapshot_checkpoint


def get_backend(backend: str) -> IOBackend:
    if backend == "h5py":
        from .h5py import backend as H5PYInterface

        return H5PYInterface
    elif backend == "adios2":
        from .adios2 import backend as ADIOS2Interface

        return ADIOS2Interface
    else:
        return import_module(backend)
