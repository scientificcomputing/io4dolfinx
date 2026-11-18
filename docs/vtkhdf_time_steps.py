# # Time steps in VTKHDF files
#
# The {ref}`VTKHDF backend<vtkhdfbackend>` stores time series the way VTK reads them.
# The [VTKHDF specification](https://docs.vtk.org/en/latest/vtk_file_formats/vtkhdf_file_format/vtkhdf_specifications.html#temporal-data)
# and VTK's reader impose three rules:
#
# - **Every array has data at every time step.** Each point or cell array stores one
#   offset into its data per time step, and repeating the offset of the previous step
#   reuses its data. An array cannot be absent at a step.
# - **Time steps increase.** VTK looks up a requested time with a binary search, so it
#   picks the wrong step if the times are not sorted.
# - **All blocks share the time steps.** In a file with several meshes, or a mesh and its
#   tags, every block must have the same time steps; otherwise VTK reports wrong times or
#   refuses to read the file.
#
# io4dolfinx follows these rules when it writes a mesh with {py:func}`io4dolfinx.write_mesh`,
# or data with {py:func}`io4dolfinx.write_point_data` and {py:func}`io4dolfinx.write_cell_data`:
#
# 1. A mesh or an array can be written only once at a time. A second write raises a
#    `ValueError` and leaves the file unchanged. Times are the same if they agree up to
#    the precision of the stored time stamps (`float64`: a relative tolerance of
#    $\sqrt{\epsilon}\approx 1.5\cdot 10^{-8}$), so round-off does not create a new step.
# 2. A new time must be later than the last time step of the file, otherwise a
#    `ValueError` is raised.
# 3. A new time step is added to every mesh and mesh tags in the file, and a mesh added
#    later gets the existing time steps.
# 4. At a step where an array or the mesh geometry was not written, it holds its latest
#    earlier write, and the steps before its first write hold its first write. This is
#    also what VTK shows. An array may be written at an earlier step that exists, which
#    updates the steps up to its next write.
# 5. To do so, io4dolfinx records the steps at which it wrote each array and each mesh
#    in its own group, `/io4dolfinx`, which VTK ignores. For files without this record,
#    from older versions of io4dolfinx, it assumes a write at the first step and wherever
#    the offsets of the data change.
#
# Only the geometry of a mesh can change between time steps: its topology is stored once,
# so writing a mesh with another number of points or cells under the same name raises a
# `ValueError`.
#
# {py:func}`io4dolfinx.read_timestamps` therefore returns every time step of the mesh,
# for each array.
#
# ## A splitting scheme
#
# Fractional step schemes for the incompressible Navier-Stokes equations, such as the
# one in [oasisx](https://computationalphysiology.github.io/oasisx/splitting_schemes.html),
# compute the pressure at half steps. Step $n$ solves for a tentative velocity, then for
# the pressure $p^{n-\frac{1}{2}}$ at $t_{n-\frac{1}{2}}$, and finally updates the
# velocity $\mathbf{u}^n$ at $t_n$. We emulate the scheme with fields that are constant
# in space, $\mathbf{u}^n = (t_n, 2t_n)$ and $p^{n-\frac{1}{2}} = t_{n-\frac{1}{2}}$,
# and write each field when it is computed.

# +
from pathlib import Path

from mpi4py import MPI

import dolfinx
import h5py
import numpy as np
import pandas as pd

import io4dolfinx

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (2,))), name="u")
p = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("Lagrange", 1)), name="p")
filename = Path("splitting_scheme.vtkhdf")


def solve(function, t):
    """Emulate a solve at time t and write the solution."""
    if function is u:
        function.interpolate(lambda x: np.full((2, x.shape[1]), [[t], [2 * t]]))
    else:
        function.interpolate(lambda x: np.full(x.shape[1], t))
    io4dolfinx.write_point_data(
        filename,
        function,
        time=t,
        mode=io4dolfinx.FileMode.append,
        backend_args=None,
        backend="vtkhdf",
    )


dt, num_steps = 0.1, 3
io4dolfinx.write_mesh(filename, mesh, time=0.0, backend="vtkhdf")
solve(u, 0.0)
for n in range(1, num_steps + 1):
    solve(p, (n - 0.5) * dt)  # Tentative velocity and pressure correction
    solve(u, n * dt)  # Velocity update
# -

# Both arrays have every time step, also those at which they were not written:

for name in ("u", "p"):
    print(name, io4dolfinx.read_timestamps(filename, MPI.COMM_WORLD, name, backend="vtkhdf"))

# At each step, an array holds its latest write. The pressure holds its first value,
# $p^{\frac{1}{2}}$, at $t=0$. The offsets show how the file stores this: a repeated
# offset reuses the data of the step before.

# +
mesh_in = io4dolfinx.read_mesh(filename, MPI.COMM_WORLD, backend="vtkhdf")
times = io4dolfinx.read_timestamps(filename, MPI.COMM_WORLD, "u", backend="vtkhdf")
rows = []
with h5py.File(filename, "r") as h5file:
    offsets = h5file["VTKHDF/mesh/Steps/PointDataOffsets"]
    for k, t in enumerate(times):
        u_in, p_in = (
            io4dolfinx.read_point_data(filename, name, mesh_in, time=t, backend="vtkhdf")
            for name in ("u", "p")
        )
        rows.append(
            {
                "time": t,
                "u_x": u_in.x.array[0],
                "u_y": u_in.x.array[1],
                "p": p_in.x.array[0],
                "offset of u": offsets["u"][k],
                "offset of p": offsets["p"][k],
            }
        )
pd.DataFrame(rows).round(3)
# -

# ## Write orders that are rejected
#
# Writing the velocity of a step before its pressure adds the step $t_{n}$ first. The
# pressure at $t_{n-\frac{1}{2}}$ would then need a step before the last one, which VTK
# cannot read correctly, so io4dolfinx raises an error (rule 2).

solve(u, (num_steps + 1) * dt)
try:
    solve(p, (num_steps + 0.5) * dt)
except ValueError as error:
    print(f"ValueError: {error}")

# Writing the velocity again at a time where it has been written is rejected as well
# (rule 1), also if the time differs by round-off: $t_3$ is stored as `3 * dt`, while
# `2.5 * dt + 0.5 * dt` gives a slightly different number.

print(f"{num_steps * dt!r} != {(num_steps - 0.5) * dt + 0.5 * dt!r}")
try:
    solve(u, (num_steps - 0.5) * dt + 0.5 * dt)
except ValueError as error:
    print(f"ValueError: {error}")

# Neither write changed the file:

print(io4dolfinx.read_timestamps(filename, MPI.COMM_WORLD, "p", backend="vtkhdf"))

# To store the pressure of the last step, write it before the velocity, or at its own,
# later time.
