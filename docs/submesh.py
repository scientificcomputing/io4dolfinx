# # Checkpointing a submesh
#
# A submesh created with {py:func}`dolfinx.mesh.create_submesh` cannot be
# checkpointed the way an ordinary mesh is, for two reasons that both come from
# how DOLFINx builds it.
#
# First, when the submesh has co-dimension greater than zero its cells are
# *entities* of the parent, and the vertices of an entity are ordered by their
# current global vertex index. Reading the parent back re-partitions it, which
# renumbers the vertices, so the same cell comes back with its vertices in a
# different order and its degrees of freedom in different positions.
#
# Second, `create_submesh` builds its vertex map allowing ownership to move, so
# a process can own a vertex -- and a degree of freedom -- that is only incident
# to cells it ghosts. Such a dof has no position in any cell that process owns,
# which is exactly what the checkpoint reader needs to find it.
#
# So io4dolfinx never reads a function onto a re-derived submesh. It stores the
# submesh as an ordinary, independent mesh, reads it back as one, and then moves
# the data across to a submesh re-derived from the parent when you need the
# entity maps that make mixed-dimensional assembly possible.

# +
from pathlib import Path

from mpi4py import MPI

import dolfinx
import numpy as np

import io4dolfinx

mesh = dolfinx.mesh.create_unit_cube(
    MPI.COMM_WORLD, 6, 6, 6, ghost_mode=dolfinx.mesh.GhostMode.shared_facet
)
tdim = mesh.topology.dim
# -

# We take the face $x=0$ as the submesh. It is a two-dimensional mesh embedded
# in three dimensions -- a manifold -- which the checkpoint handles just as well
# as a co-dimension 0 submesh.

# +
fdim = tdim - 1
mesh.topology.create_entities(fdim)
facets = dolfinx.mesh.locate_entities(mesh, fdim, lambda x: np.isclose(x[0], 0.0))
submesh, cell_map, vertex_map, node_map = dolfinx.mesh.create_submesh(mesh, fdim, facets)

V = dolfinx.fem.functionspace(submesh, ("Lagrange", 2))
u = dolfinx.fem.Function(V, name="u")
u.interpolate(lambda x: np.sin(3 * x[1]) + x[2])
# -

# ## Writing
#
# The parent goes in first. {py:func}`io4dolfinx.write_submesh` then stores the
# submesh under a name of its own, together with a *parent link*: for each stored
# submesh cell, the parent geometry nodes of the entity it came from. That link
# is what lets the submesh be found again in a re-partitioned parent.
#
# The link tags entities of the parent, so it is stored alongside the parent.
# Here everything goes in one file; pass `parent_filename` to keep the submesh in
# a file of its own.

# +
filename = Path("submesh_checkpoint.bp")
io4dolfinx.write_mesh(filename, mesh)
io4dolfinx.write_submesh(filename, submesh, mesh, cell_map, mesh_name="wall")
io4dolfinx.write_function(filename, u, time=0.0, mode=io4dolfinx.FileMode.append, mesh_name="wall")
# -

# ## Reading
#
# Both meshes are read independently. Because the submesh was stored as an
# ordinary mesh, reading a function on it is the ordinary path -- no special
# handling, and the values come back exactly.

# +
parent = io4dolfinx.read_mesh(filename, MPI.COMM_WORLD)
stored = io4dolfinx.read_mesh(filename, MPI.COMM_WORLD, mesh_name="wall")

V_stored = dolfinx.fem.functionspace(stored, ("Lagrange", 2))
u_stored = dolfinx.fem.Function(V_stored, name="u")
io4dolfinx.read_function(filename, u_stored, time=0.0, name="u", mesh_name="wall")
# -

# If all you wanted was the data, stop here. If you need to assemble a
# mixed-dimensional form over the parent and the submesh, you need a submesh
# DOLFINx recognises as derived from *this* parent, with the entity maps to prove
# it. {py:func}`io4dolfinx.read_submesh` re-derives one.

# +
checkpoint = io4dolfinx.read_submesh(filename, parent, mesh_name="wall")
V_sub = dolfinx.fem.functionspace(checkpoint.submesh, ("Lagrange", 2))
u_sub = dolfinx.fem.Function(V_sub, name="u")
io4dolfinx.transfer_submesh_function(u_stored, u_sub, checkpoint.stored_cells)
# -

# The transfer routes each interpolation point straight to the process and cell
# that can evaluate it, using the cell correspondence the checkpoint already
# records. No geometric point location is involved, so there is no search
# tolerance to tune and nothing that can silently fail to find a point.
#
# The result agrees with the original to machine precision.

# +
reference = dolfinx.fem.Function(V_sub)
reference.interpolate(lambda x: np.sin(3 * x[1]) + x[2])
num_owned = V_sub.dofmap.index_map.size_local * V_sub.dofmap.index_map_bs
error = np.max(np.abs(u_sub.x.array[:num_owned] - reference.x.array[:num_owned]))
print(f"Max error after transfer: {parent.comm.allreduce(error, MPI.MAX):.3e}")
# -

# `checkpoint.cell_map` and `checkpoint.vertex_map` are genuine
# {py:class}`dolfinx.mesh.EntityMap` objects relating the submesh to `parent`, so
# they can be passed to {py:func}`dolfinx.fem.form`.

# +
import ufl  # noqa: E402

one = dolfinx.fem.Function(V_sub)
one.x.array[:] = 1.0
area_form = dolfinx.fem.form(one * ufl.ds(domain=parent), entity_maps=[checkpoint.cell_map])
area = parent.comm.allreduce(dolfinx.fem.assemble_scalar(area_form), MPI.SUM)
print(f"Area of the submesh assembled over the parent: {area:.3f}")
# -

# ## What is not supported
#
# H(div) and H(curl) on a submesh of co-dimension greater than zero can be
# **written and read** exactly -- steps 1-3 above -- but cannot be transferred to
# a re-derived submesh. `transfer_submesh_function` raises `NotImplementedError`
# for them.
#
# The reason is worth knowing. DOLFINx cannot reconcile the reference and
# physical value sizes of these families when `tdim < gdim`, and in this
# particular path it does not say so: `interpolate_nonmatching` returns
# successfully with values that are wrong. Measured on a facet submesh of a unit
# cube, the L2 norm of an N1curl field fell from 9.096 to 4.904. The guard exists
# so that this shows up as an error rather than as a plausible-looking result.
#
# There is no sound workaround. The stored submesh is an independent mesh,
# derived from nothing, so a projection -- or any other form assembled between it
# and the re-derived submesh -- would have no basis.
