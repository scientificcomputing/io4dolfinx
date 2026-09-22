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
# Second, {py:func}`create_submesh<dolfinx.mesh.create_submesh>` builds its vertex
# map allowing ownership to move, so a process can own a vertex and a degree of
# freedom  that is only incident to the cells it ghosts.
# Such a dof has no position in any cell that process owns,
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
# submesh under a name of its own, together with its *post codes*: each parent
# entity that became a submesh cell is tagged, by its parent geometry nodes, with
# the index that cell has in the stored submesh. Addressing the entities that way
# is what lets them be found again in a re-partitioned parent, and the values are
# what the data is later routed by.
#
# The post codes tag entities of the parent, so they are stored alongside the parent.
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
io4dolfinx.transfer_submesh_function(u_stored, u_sub, checkpoint.post_code)
# -

# The {py:class}`checkpoint<io4dolfinx.SubmeshCheckpoint>` stores the
# {py:attr}`cell_map<io4dolfinx.SubmeshCheckpoint.cell_map>` and
# {py:attr}`vertex_map<io4dolfinx.SubmeshCheckpoint.vertex_map>` and
# {py:attr}`node_map<io4dolfinx.SubmeshCheckpoint.node_map>` that are
# similar to the ones created by {py:func}`dolfinx.mesh.create_submesh`.
# It also carries the
# {py:attr}`post_code<io4dolfinx.SubmeshCheckpoint.post_code>` that the transfer
# above was given, which is worth a word of its own.
#
# ## The post office
#
# The stored submesh and the re-derived one are partitioned independently, so
# neither knows where the other put a given cell. What they agree on is a cell's
# *post code*: the index it had in the stored submesh. The stored submesh knows
# it as its ordinary `topology.original_cell_index`, being a mesh like any other.
# The re-derived one cannot -- {py:func}`dolfinx.mesh.create_submesh` does not
# give a submesh an original cell index, and for a submesh of co-dimension
# greater than zero there is none to give, since the parent's input data numbers
# cells and not facets. So the post code is what
# {py:func}`io4dolfinx.write_submesh` tags onto the parent and
# {py:func}`io4dolfinx.read_submesh` hands back.
#
# Resolving one works like posting a letter. No process holds a global table of
# who owns what. Instead the post code itself decides which process acts as its
# *post office*, by a rule every process applies identically
# ({py:func}`io4dolfinx.utils.index_owner`, an equal split of the global cell range).
# Each process publishes to the post office of every cell it owns, saying where it
# keeps that cell; each process then asks the post office of every cell it wants.
# Two neighbourhood exchanges, and the memory per process stays proportional to
# the cells it actually touches.
#
# The reply; a rank and a local cell, is what makes the transfer a routing
# rather than a search. Each interpolation point goes straight to the process and
# cell that can evaluate it, so no geometric point location is involved, there is
# no search tolerance to tune, and nothing can silently fail to find a point.

# +
reference = dolfinx.fem.Function(V_sub)
reference.interpolate(lambda x: np.sin(3 * x[1]) + x[2])
num_owned = V_sub.dofmap.index_map.size_local * V_sub.dofmap.index_map_bs
error = np.max(np.abs(u_sub.x.array[:num_owned] - reference.x.array[:num_owned]))
print(f"Max error after transfer: {parent.comm.allreduce(error, MPI.MAX):.3e}")
# -

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
# The reason is worth knowing. A finite element space on a manifold does not
# carry its value shape in *physical* space, so DOLFINx cannot reconcile the
# reference and physical value sizes of these families when `tdim < gdim`
# ([FEniCS/dolfinx#3619](https://github.com/FEniCS/dolfinx/issues/3619), open at
# the time of writing). Elsewhere that surfaces as an outright error:
# *"Interpolation: elements have different value dimensions"*, but in this
# particular path it does not: `interpolate_nonmatching` returns successfully
# with values that are wrong. The guard exists so that this
# shows up as an error rather than as a plausible-looking result, and it can be
# lifted once that issue is fixed.
#
# There is no sound workaround. The stored submesh is an independent mesh,
# derived from nothing, so a projection -- or any other form assembled between it
# and the re-derived submesh -- would have no basis.
