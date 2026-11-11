"""Compare uncompressed VTKHDF, gzip, and gzip+shuffle under MPI.

Run from the repository root, for example:
    mpiexec -n 2 python benchmarks/vtkhdf_compression.py --output /tmp/vtkhdf-mpi

Uses smooth scalar/vector fields on triangular unit-square meshes. Timings include
open/close, mesh repartitioning, and field interpolation, and report the slowest
rank. Reads use warm filesystem caches; writes close the file without an explicit
fsync unless --sync is supplied. Results describe this workload and filesystem,
not general speed guarantees.
Each mode is warmed up before measurement; measured mode order rotates each round.
Output files are overwritten between repetitions to bound disk usage.
"""

import argparse
import json
import os
import platform
from pathlib import Path

from mpi4py import MPI

import dolfinx
import h5py
import numpy as np
from dolfinx.fem import Function, functionspace
from dolfinx.mesh import create_unit_square

import io4dolfinx

MODES = {
    "uncompressed": None,
    "gzip": {"compression": "gzip", "compression_opts": 4, "shuffle": False},
    "gzip_shuffle": {"compression": "gzip", "compression_opts": 4, "shuffle": True},
}


def measure(filename, mesh, point, cell, options, sync):
    comm = mesh.comm
    comm.barrier()
    start = MPI.Wtime()
    io4dolfinx.write_mesh(filename, mesh, backend="vtkhdf", backend_args=options)
    mesh_written = MPI.Wtime()
    io4dolfinx.write_point_data(
        filename, point, time=0.0, mode=io4dolfinx.FileMode.append, backend_args=options
    )
    point_written = MPI.Wtime()
    io4dolfinx.write_cell_data(
        filename, cell, time=0.0, mode=io4dolfinx.FileMode.append, backend_args=options
    )
    cell_written = MPI.Wtime()
    if sync:
        comm.barrier()
        if comm.rank == 0:
            with filename.open("rb") as stream:
                os.fsync(stream.fileno())
        comm.barrier()
    write_finished = MPI.Wtime()
    write_times = {
        "write_seconds": write_finished - start,
        "mesh_write_seconds": mesh_written - start,
        "point_write_seconds": point_written - mesh_written,
        "cell_write_seconds": cell_written - point_written,
        "sync_seconds": write_finished - cell_written,
    }
    # Reduce outside the timed region; retain the critical-path time across ranks.
    timings = {key: comm.allreduce(value, op=MPI.MAX) for key, value in write_times.items()}
    comm.barrier()
    start = MPI.Wtime()
    restored = io4dolfinx.read_mesh(filename, comm, backend="vtkhdf")
    mesh_read = MPI.Wtime()
    io4dolfinx.read_point_data(filename, point.name, restored, 0.0, backend="vtkhdf")
    io4dolfinx.read_cell_data(filename, cell.name, restored, 0.0, backend="vtkhdf")
    fields_read = MPI.Wtime()
    read_times = {
        "read_seconds": fields_read - start,
        "mesh_read_seconds": mesh_read - start,
        "field_read_seconds": fields_read - mesh_read,
    }
    timings.update({key: comm.allreduce(value, op=MPI.MAX) for key, value in read_times.items()})
    return timings


def benchmark(output: Path, sizes: list[int], repeats: int, warmups: int, sync: bool):
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    comm.barrier()
    results = []
    for n in sizes:
        mesh = create_unit_square(comm, n, n)
        point = Function(functionspace(mesh, ("Lagrange", 1, (3,))), name="velocity")
        point.interpolate(lambda x: np.vstack((np.sin(x[0]), np.cos(x[1]), x[0] * x[1])))
        cell = Function(functionspace(mesh, ("DG", 0)), name="pressure")
        cell.interpolate(lambda x: x[0] + 2 * x[1])
        for repeat in range(-warmups, repeats):
            modes = list(MODES)
            shift = max(repeat, 0) % len(modes)
            modes = modes[shift:] + modes[:shift]
            for mode in modes:
                filename = output / f"n{n}-{mode}.vtkhdf"
                timings = measure(filename, mesh, point, cell, MODES[mode], sync)
                if repeat >= 0 and comm.rank == 0:
                    row = dict(
                        n=n,
                        cells=2 * n * n,
                        ranks=comm.size,
                        mode=mode,
                        repeat=repeat,
                        bytes=filename.stat().st_size,
                        **timings,
                    )
                    results.append(row)
                    print(json.dumps(row), flush=True)
        if comm.rank == 0:
            # Save incrementally so completed sizes survive an interrupted run.
            report = dict(
                hostname=platform.node(),
                platform=platform.platform(),
                h5py=h5py.__version__,
                hdf5=h5py.version.hdf5_version,
                dolfinx=dolfinx.__version__,
                mpi=MPI.Get_library_version().strip(),
                repeats=repeats,
                warmups=warmups,
                fsync=sync,
                modes=MODES,
                results=results,
            )
            (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 256, 768])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--sync", action="store_true", help="Include fsync after all writes")
    args = parser.parse_args()
    if any(n <= 0 for n in args.sizes) or args.repeats <= 0 or args.warmups < 0:
        parser.error("sizes and repeats must be positive; warmups must be nonnegative")
    benchmark(args.output, args.sizes, args.repeats, args.warmups, args.sync)
