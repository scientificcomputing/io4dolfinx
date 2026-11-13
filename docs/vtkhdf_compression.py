# # Compressing VTKHDF output
#
# The {ref}`VTKHDF backend<vtkhdfbackend>` stores its datasets with lossless HDF5
# filters, reducing file sizes. By default, the integer mesh arrays are stored with
# scale-offset; all other filters are opt-in through `backend_args`. The same
# dictionary can be passed to {py:func}`io4dolfinx.write_mesh`,
# {py:func}`io4dolfinx.write_point_data`, {py:func}`io4dolfinx.write_cell_data` and
# {py:func}`io4dolfinx.write_meshtags`. Readers need no extra arguments.
#
# The filter options follow the arguments of {py:meth}`h5py.Group.create_dataset`;
# see the h5py {ref}`filter pipeline<h5py:dataset_compression>`. The supported options are:
#
# - `compression`: `"gzip"` or `None`, with `compression_opts` the gzip level (0-9).
# - `shuffle`: reorder bytes before compression, which often improves gzip.
# - `fletcher32`: store a checksum per chunk, so that corrupted data raise an error on read.
# - `scaleoffset` (on by default): store the integer mesh arrays with the minimum
#   number of bits. Floating-point data are left unchanged, as scale-offset is lossy
#   for them. The default applies to new datasets only; `scaleoffset=False` turns it
#   off, and so does `fletcher32=True`, as h5py cannot combine the two.
#
# ## Writing compressed data
#
# We write a mesh, a point field and a cell field with gzip and shuffle, if the HDF5
# library has been built with gzip support.

# +
import logging
import shutil
from pathlib import Path
from typing import Any

from mpi4py import MPI

import dolfinx
import h5py
import ipyparallel as ipp
import numpy as np
import pandas as pd

import io4dolfinx

gzip_available = h5py.h5z.filter_avail(h5py.h5z.FILTER_DEFLATE)
if not gzip_available:
    print("HDF5 is built without gzip; only the other filters are demonstrated.")


def velocity(x):
    return np.vstack((x[0] + x[1], x[0] * x[1]))


mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 16, 16)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
u = dolfinx.fem.Function(V, name="u")
u.interpolate(velocity)
q = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("DG", 0)), name="q")
q.interpolate(lambda x: x[0] - x[1])

options: dict[str, Any] = {"shuffle": True}
if gzip_available:
    options |= {"compression": "gzip", "compression_opts": 4}
filename = Path("compressed.vtkhdf")
io4dolfinx.write_mesh(filename, mesh, backend="vtkhdf", backend_args=options)
for writer, function in ((io4dolfinx.write_point_data, u), (io4dolfinx.write_cell_data, q)):
    writer(
        filename,
        function,
        time=0.0,
        mode=io4dolfinx.FileMode.append,
        backend="vtkhdf",
        backend_args=options,
    )
# -

# Each bulk dataset is chunked and filtered independently. Scale-offset, on by default,
# is only applied to the integer arrays.

with h5py.File(filename, "r") as file:
    block = file["VTKHDF/mesh"]
    for path in ("Points", "Connectivity", "Offsets", "Types", "PointData/u", "CellData/q"):
        dataset = block[path]
        print(
            f"{path:12} {str(dataset.dtype):8} {str(dataset.shape):12} "
            f"compression={dataset.compression}, "
            f"shuffle={dataset.shuffle}, scaleoffset={dataset.scaleoffset is not None}"
        )

# The data reads back unchanged, up to round-off.

restored = io4dolfinx.read_mesh(filename, MPI.COMM_WORLD, backend="vtkhdf")
u_restored = io4dolfinx.read_point_data(filename, "u", restored, time=0.0, backend="vtkhdf")
expected = dolfinx.fem.Function(u_restored.function_space)
expected.interpolate(velocity)
np.testing.assert_allclose(u_restored.x.array, expected.x.array, atol=1e-14)

# Settings apply when a dataset is created. When appending, omitted options keep the
# existing settings, while conflicting ones raise a `ValueError` before anything is
# written. In parallel, every rank must call the writer with identical options.
#
# ## Size and speed
#
# Whether compression pays off depends on the data, the number of ranks and the file
# system. Below, each storage mode writes and reads a mesh with a vector-valued point
# field and a scalar cell field on four MPI processes, for three mesh sizes.
# Timings are the medians of three runs, after one warm-up. All modes write with
# collective MPI-IO, so the differences come from the filters alone.


# +
def benchmark(directory: Path, sizes: list[int], modes: dict, repeats: int):
    import os

    from mpi4py import MPI

    import dolfinx
    import numpy as np

    import io4dolfinx

    comm = MPI.COMM_WORLD

    def measure(filename, mesh, point, cell, options):
        comm.barrier()
        start = MPI.Wtime()
        io4dolfinx.write_mesh(filename, mesh, backend="vtkhdf", backend_args=options)
        for writer, function in (
            (io4dolfinx.write_point_data, point),
            (io4dolfinx.write_cell_data, cell),
        ):
            writer(
                filename,
                function,
                time=0.0,
                mode=io4dolfinx.FileMode.append,
                backend="vtkhdf",
                backend_args=options,
            )
        comm.barrier()
        if comm.rank == 0:
            with filename.open("rb") as stream:
                os.fsync(stream.fileno())
        comm.barrier()
        write = MPI.Wtime() - start

        start = MPI.Wtime()
        restored = io4dolfinx.read_mesh(filename, comm, backend="vtkhdf")
        io4dolfinx.read_point_data(filename, point.name, restored, 0.0, backend="vtkhdf")
        io4dolfinx.read_cell_data(filename, cell.name, restored, 0.0, backend="vtkhdf")
        read = MPI.Wtime() - start
        # The slowest rank determines the time.
        return comm.allreduce(write, op=MPI.MAX), comm.allreduce(read, op=MPI.MAX)

    results = []
    for n in sizes:
        mesh = dolfinx.mesh.create_unit_square(comm, n, n)
        V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (3,)))
        point = dolfinx.fem.Function(V, name="velocity")
        point.interpolate(lambda x: np.vstack((np.sin(x[0]), np.cos(x[1]), x[0] * x[1])))
        cell = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("DG", 0)), name="pressure")
        cell.interpolate(lambda x: x[0] + 2 * x[1])
        names = list(modes)
        # One warm-up round, then rotate the order of the modes between runs.
        for repeat in range(-1, repeats):
            shift = max(repeat, 0) % len(names)
            for name in names[shift:] + names[:shift]:
                filename = directory / f"n{n}-mode{names.index(name)}.vtkhdf"
                write, read = measure(filename, mesh, point, cell, modes[name])
                if comm.rank == 0:
                    if repeat >= 0:
                        results.append(
                            {
                                "cells": 2 * n * n,
                                "mode": name,
                                "size (MB)": filename.stat().st_size / 1e6,
                                "write (ms)": 1e3 * write,
                                "read (ms)": 1e3 * read,
                            }
                        )
                    filename.unlink()
                comm.barrier()
    return results


modes: dict[str, dict[str, Any]] = {
    "no filters": {"scaleoffset": False},
    "scaleoffset (default)": {},
    "fletcher32": {"fletcher32": True},
    "shuffle + scaleoffset": {"shuffle": True},
}
if gzip_available:
    gzip = {"compression": "gzip", "compression_opts": 4}
    modes |= {
        "gzip": gzip | {"scaleoffset": False},
        "gzip + shuffle": gzip | {"shuffle": True, "scaleoffset": False},
        "gzip + shuffle + scaleoffset": gzip | {"shuffle": True},
    }

directory = Path("compression_benchmark").absolute()
directory.mkdir(exist_ok=True)
with ipp.Cluster(engines="mpi", n=4, log_level=logging.ERROR) as cluster:
    query = cluster[:].apply_async(benchmark, directory, [16, 256, 768], modes, 3)
    query.wait()
    assert query.successful(), query.error
    rows = next(result for result in query.get() if result)
shutil.rmtree(directory)
# -

# Relative to output without filters, a ratio below one means a smaller file or a
# faster write or read.

# +
results = pd.DataFrame(rows)
results["mode"] = pd.Categorical(results["mode"], categories=list(modes))
results = results.groupby(["cells", "mode"], observed=True).median()
baseline = results.xs("no filters", level="mode")
for column, ratio in (
    ("size (MB)", "size ratio"),
    ("write (ms)", "write ratio"),
    ("read (ms)", "read ratio"),
):
    results[ratio] = results[column].div(baseline[column], level="cells")
columns = ["size (MB)", "size ratio", "write (ms)", "write ratio", "read (ms)", "read ratio"]
results[columns].round(2)
# -

# The results describe smooth fields on this machine. Noisy data compress less, and
# parallel file systems behave differently, so measure representative data before
# choosing a setting.
