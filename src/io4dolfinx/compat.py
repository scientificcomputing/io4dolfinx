"""Layer for small backward compatibility wrappers for DOLFINx"""

from mpi4py import MPI

import dolfinx
import numpy as np
import numpy.typing as npt


def create_cell_permutations(mesh: dolfinx.mesh.Mesh):
    """Create the cell permutations for the mesh."""
    getattr(
        mesh.topology,
        "create_cell_permutations",
        getattr(mesh.topology, "create_entity_permutations"),
    )()


def cmap(mesh: dolfinx.mesh.Mesh) -> dolfinx.fem.CoordinateElement:
    """Get the basix Cmap for the mesh."""
    if hasattr(mesh.geometry, "cmaps"):
        return mesh.geometry.cmaps[0]
    if callable(mesh.geometry.cmap):
        return mesh.geometry.cmap()
    else:
        return mesh.geometry.cmap


def dofmap(mesh: dolfinx.mesh.Mesh) -> npt.NDArray[np.int32]:
    """Get the dofmap for the geometry."""
    if hasattr(mesh.geometry, "dofmaps"):
        return mesh.geometry.dofmaps[0]
    if callable(mesh.geometry.dofmap):
        return mesh.geometry.dofmap()
    else:
        return mesh.geometry.dofmap


def index_map(comm: MPI.Comm, local_size: int) -> dolfinx.common.IndexMap:
    """Create a ghost-free index map distributing ``local_size`` indices per process."""
    if hasattr(dolfinx.common, "index_map"):
        # DOLFINx > 0.11 wraps IndexMap in Python and constructs it via a factory
        return dolfinx.common.index_map(comm, local_size)
    return dolfinx.common.IndexMap(comm, local_size)  # type: ignore[call-arg, arg-type]


def cpp_index_map(
    imap: "dolfinx.common.IndexMap | dolfinx.cpp.common.IndexMap",
) -> dolfinx.cpp.common.IndexMap:
    """Unwrap an index map for handing to a C++ constructor.

    DOLFINx > 0.11 returns a Python ``IndexMap`` wrapper from accessors such as
    ``Geometry.index_map()`` and ``DofMap.index_map``, while the C++ constructors
    still expect the underlying ``dolfinx.cpp.common.IndexMap``.
    """
    if isinstance(imap, dolfinx.cpp.common.IndexMap):
        return imap
    return imap._cpp_object
