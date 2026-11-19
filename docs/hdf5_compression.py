# # Compressing HDF5 output
#
# The HDF5 backends (`h5py`, `vtkhdf` and their `-serial` versions) can store datasets with
# lossless HDF5 filters, set through `backend_args` in any of their writers. Readers need
# no options. The ADIOS2 backend does not compress.
#
# ## Options
#
# - `compression`: `"gzip"` or `None` (default). Only gzip is supported, as ParaView
#   reads it without plugins.
# - `compression_opts`: gzip level, 0-9 (default 4).
# - `shuffle`: reorder bytes before compression, often improving gzip (default `False`).
# - `fletcher32`: checksum per chunk, so corrupted data raise an error when read
#   (default `False`).
# - `scaleoffset`: store integer arrays with the fewest bits needed; lossless, and not
#   applied to floating-point data (default `True`). It cannot be combined with
#   `fletcher32`, which turns it off.
# - `chunk_bytes`: chunk size in bytes (default 1 MiB, which fits ParaView's chunk cache).
#
# The names follow {py:meth}`h5py.Group.create_dataset`. Invalid options, or filters
# missing from the HDF5 library, raise an error.
#
# Filters require chunked datasets: each dataset is split into chunks of whole rows,
# filtered one by one. Options apply when a dataset is created. Appending keeps the
# existing settings, and conflicting options raise a `ValueError`. In parallel, every rank
# must pass the same options, and filtered writes need h5py built with MPI; the serial
# backends write through rank 0.
#
# ## Writing compressed data
#
# A checkpoint (h5py) and a visualisation file (VTKHDF), with shuffle, and gzip if
# available.

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

options: dict[str, Any] = {"shuffle": True}
if gzip_available:
    options |= {"compression": "gzip", "compression_opts": 4}

checkpoint = Path("compressed.h5")
io4dolfinx.write_mesh(checkpoint, mesh, backend="h5py", backend_args=options)
io4dolfinx.write_function(checkpoint, u, time=0.0, backend="h5py", backend_args=options)

visualisation = Path("compressed.vtkhdf")
io4dolfinx.write_mesh(visualisation, mesh, backend="vtkhdf", backend_args=options)
io4dolfinx.write_point_data(
    visualisation,
    u,
    time=0.0,
    mode=io4dolfinx.FileMode.append,
    backend="vtkhdf",
    backend_args=options,
)
# -

# Scale-offset, on by default, is only applied to the integer arrays.


# +
def describe(filename: Path, paths: list[str]):
    with h5py.File(filename, "r") as file:
        for path in paths:
            dataset = file[path]
            print(
                f"{path:28} {str(dataset.dtype):8} compression={dataset.compression}, "
                f"shuffle={dataset.shuffle}, scaleoffset={dataset.scaleoffset is not None}"
            )


describe(checkpoint, ["mesh/0/Points", "mesh/Topology", "mesh/functions/u/0/array"])
describe(visualisation, ["VTKHDF/mesh/Points", "VTKHDF/mesh/Connectivity"])
# -

# The data reads back unchanged.

restored = io4dolfinx.read_mesh(checkpoint, MPI.COMM_WORLD, backend="h5py")
u_restored = dolfinx.fem.Function(
    dolfinx.fem.functionspace(restored, ("Lagrange", 1, (restored.geometry.dim,))), name="u"
)
io4dolfinx.read_function(checkpoint, u_restored, time=0.0, backend="h5py")
expected = dolfinx.fem.Function(u_restored.function_space)
expected.interpolate(velocity)
np.testing.assert_allclose(u_restored.x.array, expected.x.array, atol=1e-14)

# ## Size and speed
#
# Writing and reading a mesh with a vector and a scalar function on four processes,
# median of three runs. The baseline, "no filters", is also chunked, so the ratios show
# the effect of the filters alone. Results depend on the data and the file system.


# +
def benchmark(directory: Path, sizes: list[int], modes: dict, repeats: int):
    import os

    from mpi4py import MPI

    import dolfinx
    import numpy as np

    import io4dolfinx

    comm = MPI.COMM_WORLD

    def write(backend, filename, mesh, point, cell, options):
        io4dolfinx.write_mesh(filename, mesh, backend=backend, backend_args=options)
        if backend == "h5py":
            for function in (point, cell):
                io4dolfinx.write_function(
                    filename, function, time=0.0, backend=backend, backend_args=options
                )
        else:
            for writer, function in (
                (io4dolfinx.write_point_data, point),
                (io4dolfinx.write_cell_data, cell),
            ):
                writer(
                    filename,
                    function,
                    time=0.0,
                    mode=io4dolfinx.FileMode.append,
                    backend=backend,
                    backend_args=options,
                )

    def read(backend, filename, point, cell):
        restored = io4dolfinx.read_mesh(filename, comm, backend=backend)
        if backend == "h5py":
            for function in (point, cell):
                element = function.function_space.ufl_element()
                space = dolfinx.fem.functionspace(restored, element)
                result = dolfinx.fem.Function(space, name=function.name)
                io4dolfinx.read_function(filename, result, time=0.0, backend=backend)
        else:
            io4dolfinx.read_point_data(filename, point.name, restored, 0.0, backend=backend)
            io4dolfinx.read_cell_data(filename, cell.name, restored, 0.0, backend=backend)

    def measure(backend, filename, mesh, point, cell, options):
        comm.barrier()
        start = MPI.Wtime()
        write(backend, filename, mesh, point, cell, options)
        comm.barrier()
        if comm.rank == 0:
            with filename.open("rb") as stream:
                os.fsync(stream.fileno())
        comm.barrier()
        write_time = MPI.Wtime() - start
        start = MPI.Wtime()
        read(backend, filename, point, cell)
        read_time = MPI.Wtime() - start
        # The slowest rank determines the time.
        return comm.allreduce(write_time, op=MPI.MAX), comm.allreduce(read_time, op=MPI.MAX)

    results = []
    for n in sizes:
        mesh = dolfinx.mesh.create_unit_square(comm, n, n)
        V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (3,)))
        point = dolfinx.fem.Function(V, name="velocity")
        point.interpolate(lambda x: np.vstack((np.sin(x[0]), np.cos(x[1]), x[0] * x[1])))
        cell = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("DG", 0)), name="pressure")
        cell.interpolate(lambda x: x[0] + 2 * x[1])
        names = list(modes)
        for backend, suffix in (("h5py", ".h5"), ("vtkhdf", ".vtkhdf")):
            # One warm-up round, then rotate the order of the modes between runs.
            for repeat in range(-1, repeats):
                shift = max(repeat, 0) % len(names)
                for name in names[shift:] + names[:shift]:
                    filename = directory / f"n{n}-mode{names.index(name)}{suffix}"
                    times = measure(backend, filename, mesh, point, cell, modes[name])
                    if comm.rank == 0:
                        if repeat >= 0:
                            results.append(
                                {
                                    "backend": backend,
                                    "cells": 2 * n * n,
                                    "mode": name,
                                    "size (MB)": filename.stat().st_size / 1e6,
                                    "write (ms)": 1e3 * times[0],
                                    "read (ms)": 1e3 * times[1],
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

# A ratio below one means a smaller file, or a faster write or read, than without filters.

# +
results = pd.DataFrame(rows)
results["mode"] = pd.Categorical(results["mode"], categories=list(modes))
results = results.groupby(["backend", "cells", "mode"], observed=True).median()
baseline = results.xs("no filters", level="mode").reindex(results.index.droplevel("mode"))
for column, ratio in (
    ("size (MB)", "size ratio"),
    ("write (ms)", "write ratio"),
    ("read (ms)", "read ratio"),
):
    results[ratio] = results[column].to_numpy() / baseline[column].to_numpy()
columns = ["size (MB)", "size ratio", "write (ms)", "write ratio", "read (ms)", "read ratio"]
results[columns].round(2)
# -
