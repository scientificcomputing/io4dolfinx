# # Checkpoint format
#
# The {ref}`H5PY<h5pybackend>` and ADIOS2 backends store checkpoints in a layout of their own,
# rather than in a standard format such as XDMF or VTX. This lets them store functions in any
# finite element space, not only Lagrange spaces, and gives N-to-M checkpointing: a checkpoint
# written on N processes can be read on any number M of processes.
# This page describes the layout and the guarantees that come with it.
#
# ## Compatibility
#
# When a {py:class}`dolfinx.mesh.Mesh`, {py:class}`dolfinx.fem.Function` or
# {py:class}`dolfinx.mesh.MeshTags` is first written to a checkpoint, it is stamped with the
# checkpoint format version of the installed io4dolfinx,
# {py:data}`io4dolfinx.format.FORMAT_VERSION`. The stamp is never changed, so a file that has
# been appended to by several versions of io4dolfinx records the format of each object.
#
# - The **major** version changes when the layout changes in a way that older versions of
#   io4dolfinx cannot read. A checkpoint with a newer major version is not read: io4dolfinx
#   raises an error asking you to upgrade io4dolfinx.
# - The **minor** version changes when optional data is added, which older versions of
#   io4dolfinx ignore. A checkpoint with a newer minor version is read.
# - Appending to an object, e.g. adding a time step to a function, requires the object to have
#   the same major version as the installed io4dolfinx. Its minor version may be older.
# - Objects without a stamp were written before checkpoints were stamped, and are read as
#   format 1.0.
#
# ## Inspecting a checkpoint
#
# With each backend, we write a mesh at two time steps (the mesh moves between them), a
# function at two time steps, and a set of cell tags.

# +
from pathlib import Path

from mpi4py import MPI

import adios2
import dolfinx
import h5py
import numpy as np

import io4dolfinx
from io4dolfinx.backends import get_backend
from io4dolfinx.format import object_metadata

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("Lagrange", 1)), name="u")
cells = np.arange(mesh.topology.index_map(mesh.topology.dim).size_local, dtype=np.int32)
regions = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, cells, np.ones_like(cells))

files = {"h5py": Path("checkpoint_format.h5"), "adios2": Path("checkpoint_format.bp")}
for backend, filename in files.items():
    io4dolfinx.write_mesh(filename, mesh, backend=backend, time=0.0)
    mesh.geometry.x[:, 0] *= 2  # A moving mesh
    io4dolfinx.write_mesh(
        filename, mesh, backend=backend, time=1.0, mode=io4dolfinx.FileMode.append
    )
    mesh.geometry.x[:, 0] /= 2
    for t in (0.0, 1.0):
        u.interpolate(lambda x: x[0] + t)
        io4dolfinx.write_function(filename, u, time=t, backend=backend)
    io4dolfinx.write_meshtags(filename, mesh, regions, meshtag_name="regions", backend=backend)
# -

# {py:func}`io4dolfinx.format.object_metadata` gives the stamp of an object, which both
# backends store as attributes: the format version (`io4dolfinx_format`), and the versions of
# io4dolfinx (`io4dolfinx_version`) and DOLFINx (`dolfinx_version`) that wrote it.

for backend, filename in files.items():
    for kind, name in [("mesh", None), ("function", "u"), ("meshtags", "regions")]:
        stamp = object_metadata(get_backend(backend), filename, MPI.COMM_WORLD, kind, name)
        print(f"{backend:7} {kind:9}", stamp)

# ## Format 1.0
#
# ### H5PY
#
# Each mesh is a group named after the mesh (`mesh` unless a `mesh_name` is given), which
# holds its functions and mesh tags. In the listing below, groups end with `/`, datasets show
# their shape and type, and the attributes of a group are listed in brackets.


# +
def show(name, item):
    depth = name.count("/")
    label = name.split("/")[-1]
    if isinstance(item, h5py.Dataset):
        label += f" {item.shape} {item.dtype}"
    else:
        label += "/"
    if len(item.attrs) > 0:
        label += f"  [{', '.join(sorted(item.attrs))}]"
    print("  " * depth + label)


with h5py.File(files["h5py"], "r") as h5file:
    h5file.visititems(show)
# -

# - `<mesh>/`: the attributes `CellType`, `Degree` and `LagrangeVariant` describe the
#   coordinate element, and `timestamps` holds the time of each geometry.
#   - `<i>/Points`: the node coordinates at time step `i`, with one row per node.
#   - `Topology`: the nodes of each cell, as row indices into `Points`. It is the same at
#     every time step.
#   - `CellPermutations`: the DOLFINx permutation information of each cell, needed to read
#     the degrees of freedom of functions.
#   - `PartitioningData` and `PartitioningOffset`, with the attribute `PartitionProcesses`:
#     the processes that hold each cell, including as a ghost. They are written with
#     `store_partition_info=True`, to read the mesh with the same partitioning.
#   - `functions/<name>/`: a function, with its `timestamps`.
#     - `dofmap`: the global degree of freedom indices of each cell, one cell after another,
#       with the start of each cell in `dofmap_offsets`.
#     - `<i>/array`: the degree of freedom values at time step `i`.
#   - `tags/<name>/`: a set of mesh tags, with the dimension `dim` and `CellType` of the
#     tagged entities.
#     - `Topology`: the nodes of each tagged entity, as row indices into `Points`.
#     - `Values`: the value of each tagged entity.
#
# ### ADIOS2
#
# ADIOS2 stores a flat list of variables and attributes. A variable holds an array at each
# step, so time-dependent data are stored as one variable with a step per time stamp, and the
# times in a separate variable. The names that belong to the default mesh have no prefix; for a
# mesh with another `mesh_name`, they start with `<mesh_name>/`.

with adios2.FileReader(str(files["adios2"])) as adios_file:
    print("Variables:")
    for name, info in sorted(adios_file.available_variables().items()):
        steps = info["AvailableStepsCount"]
        print(f"  {name}: {info['Type']}, shape ({info['Shape']}), {steps} steps")
    print("Attributes:")
    for name, info in sorted(adios_file.available_attributes().items()):
        print(f"  {name}: {info['Type']} = {info['Value']}")

# - Mesh: the same data as for H5PY, with the times of `Points` in `MeshTime`.
# - Function `<name>`: `<name>_dofmap` and `<name>_XDofmap` (the start of each cell in
#   `<name>_dofmap`), and the values in `<name>_values`, with the times in `<name>_time`.
# - Mesh tags `<name>`: `<name>_topology`, `<name>_values` and the dimension `<name>_dim`.
#
# The attributes of a function or set of mesh tags, including its stamp, start with
# `<name>_`.
#
# ## Checkpoints from adios4dolfinx
#
# Checkpoints written with adios4dolfinx 0.7 and older, before a file could hold several
# functions and time steps, use an earlier layout. Their single function is stored as
# `Dofmap`, `XDofmap` and `Values`, without time steps, and the mesh has no `MeshTime`. Read
# them with the ADIOS2 backend and `"legacy": True` in the backend arguments:
#
# ```python
# backend_args = {"engine": "BP4", "legacy": True}
# mesh = io4dolfinx.read_mesh(filename, comm, backend="adios2", backend_args=backend_args)
# io4dolfinx.read_function(filename, u, backend="adios2", backend_args=backend_args)
# ```
