# # Checkpoint format
#
# The {ref}`H5PY<h5pybackend>` and ADIOS2 backends store checkpoints in a layout of their own,
# rather than in a standard format such as XDMF or VTX. This lets them store functions in any
# finite element space, not only Lagrange spaces, and gives N-to-M checkpointing: a checkpoint
# written on N processes can be read on any number M of processes.
# This page describes the layout, and the guarantees that come with it.
#
# ## Compatibility
#
# Each {py:class}`dolfinx.mesh.Mesh`, {py:class}`dolfinx.fem.Function` and
# {py:class}`dolfinx.mesh.MeshTags` in a checkpoint is stamped with the version of the
# checkpoint format it was written in, {py:data}`io4dolfinx.format.FORMAT_VERSION`, when it is
# first written. The stamp is never changed, so a file that has been appended to by several
# versions of io4dolfinx records the format of each object.
#
# - The **major** version changes when the layout changes in a way that older versions of
#   io4dolfinx cannot read. A checkpoint with a newer major version is not read: io4dolfinx
#   raises an error asking you to upgrade it.
# - The **minor** version changes when something optional is added. A checkpoint with a newer
#   minor version is read.
# - Appending to an object, e.g. adding a time step to a function, requires the same major
#   version.
# - Objects without a stamp were written before checkpoints were stamped, and are read as
#   format 1.0.
#
# ## Inspecting a checkpoint
#
# We write a mesh at two time steps, a function at two time steps and a set of cell tags with
# both backends.

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

# {py:func}`io4dolfinx.format.object_metadata` gives the stamp of an object: the format
# version, and the versions of io4dolfinx and DOLFINx that wrote it.

for backend, filename in files.items():
    for kind, name in [("mesh", None), ("function", "u"), ("meshtags", "regions")]:
        stamp = object_metadata(get_backend(backend), filename, MPI.COMM_WORLD, kind, name)
        print(f"{backend:7} {kind:9}", stamp)

# ## Format 1.0
#
# ### H5PY
#
# Each mesh is a group, named after the mesh (`mesh` unless a `mesh_name` is given), holding
# its functions and mesh tags. Below, every group ends with `/`, datasets show their shape and
# type, and the attributes of a group are listed in brackets.


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
#   coordinate element, `timestamps` the time of each geometry, and the stamp the format.
#   - `<i>/Points`: the node coordinates at time step `i`, with one row per node.
#   - `Topology`: the nodes of each cell, as rows into `Points`. It is the same at every time
#     step.
#   - `CellPermutations`: the permutation of each cell, used to read function data.
#   - `PartitioningData` and `PartitioningOffset`, with the attribute `PartitionProcesses`:
#     the process of each cell, written with `store_partition_info=True` to read the mesh
#     with the same partitioning.
#   - `functions/<name>/`: a function, with its `timestamps` and stamp.
#     - `dofmap` and `dofmap_offsets`: the degrees of freedom of each cell, as an adjacency
#       list into the arrays.
#     - `<i>/array`: the degree of freedom values at time step `i`.
#   - `tags/<name>/`: a set of mesh tags, with the dimension `dim` and `CellType` of the
#     tagged entities, and the stamp.
#     - `Topology`: the nodes of each tagged entity, as rows into `Points`.
#     - `Values`: the value of each tagged entity.
#
# ### ADIOS2
#
# ADIOS2 stores flat lists of variables and attributes. A variable holds one value per
# step, so time-dependent data are a variable with a step per time stamp, together with a
# variable holding the times. The names of the default mesh have no prefix, while the names of
# a mesh with another `mesh_name` start with `<mesh_name>/`.

with adios2.FileReader(str(files["adios2"])) as adios_file:
    print("Variables:")
    for name, info in sorted(adios_file.available_variables().items()):
        steps = info["AvailableStepsCount"]
        print(f"  {name}: {info['Type']}, shape ({info['Shape']}), {steps} steps")
    print("Attributes:")
    for name, info in sorted(adios_file.available_attributes().items()):
        print(f"  {name}: {info['Type']} = {info['Value']}")

# - Mesh: `Points` (one step per time stamp, with the times in `MeshTime`), `Topology` and
#   `CellPermutations`, with the attributes `CellType`, `Degree`, `LagrangeVariant` and the
#   stamp. With `store_partition_info=True`, the partitioning is stored as well, in
#   `PartitioningData` and `PartitioningOffset` with the attribute `PartitionProcesses`.
# - Function `<name>`: `<name>_dofmap` and `<name>_XDofmap` (the degrees of freedom of each
#   cell, as an adjacency list), and `<name>_values` (one step per time stamp, with the times
#   in `<name>_time`), with the stamp in the attributes `<name>_io4dolfinx_format` etc.
# - Mesh tags `<name>`: `<name>_topology` (the nodes of each tagged entity) and `<name>_values`,
#   with the dimension `<name>_dim` and the stamp in the attributes `<name>_io4dolfinx_format`
#   etc.
#
# ## Checkpoints from adios4dolfinx
#
# Checkpoints written with adios4dolfinx 0.7 and older, before a file could hold several
# functions and time steps, use an earlier layout: the dofmap and values of their single
# function are stored as `Dofmap`, `XDofmap` and `Values`. Read them with the ADIOS2 backend
# and `"legacy": True` in the backend arguments:
#
# ```python
# backend_args = {"engine": "BP4", "legacy": True}
# mesh = io4dolfinx.read_mesh(filename, comm, backend="adios2", backend_args=backend_args)
# io4dolfinx.read_function(filename, u, backend="adios2", backend_args=backend_args)
# ```
