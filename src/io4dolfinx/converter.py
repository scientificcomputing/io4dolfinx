# Copyright (C) 2026 Jørgen Schartum Dokken
#
# This file is part of io4dolfinx
#
# SPDX-License-Identifier:    MIT

"""Convert checkpoints of adios4dolfinx 0.7 and older, and mesh tags of legacy DOLFIN, to
the current format.

:func:`convert` copies the stored data to a checkpoint of the current format, so no mesh or
function space is created. It reads two kinds of files:

- Checkpoints of adios4dolfinx 0.7 and older (format 0, see :mod:`io4dolfinx.format`),
  written with ADIOS2. They store no function names or time stamps: each function is in a
  step of its own under the same variable names. Each function is given a name and written
  at time 0.
- HDF5 files of legacy DOLFIN (``HDF5File``), holding a mesh and mesh functions. The mesh
  and the named mesh functions are written as a mesh and mesh tags. Functions of these files
  are numbered by DOLFIN and need a function space: read them with
  :func:`io4dolfinx.read_function_from_legacy_h5` onto the mesh of the converted file, which
  has the node order of the DOLFIN file.

The conversion is also available from the command line, under MPI if wanted::

    io4dolfinx-convert old.bp new.h5 --names u
    io4dolfinx-convert dolfin_mesh.h5 new.h5 --meshtags facet_tags

or, equivalently, ``python3 -m io4dolfinx.converter``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mpi4py import MPI

import basix
import dolfinx
import h5py

from .backends import FileMode, get_backend
from .format import stamp
from .structures import FunctionData, MeshData
from .utils import check_file_exists
from .writers import write_function, write_mesh

__all__ = ["convert", "main"]

# Backends that write checkpoints, by the suffix of their files
_backends_by_suffix = {".bp": "adios2", ".h5": "h5py"}
_checkpoint_backends = ("adios2", "adios2-serial", "h5py", "h5py-serial")


def _convert_dolfin(
    input_file: Path,
    output_file: Path,
    comm: MPI.Comm,
    meshtags: Sequence[str],
    mesh_group: str,
    backend: str,
    backend_args: dict[str, Any] | None,
):
    """Write the mesh and mesh functions of a legacy DOLFIN HDF5 file as a checkpoint."""
    from .backends.h5py import backend as h5py_backend

    assert isinstance(comm, MPI.Intracomm)
    topology, x, cell_type = h5py_backend.read_legacy_mesh(input_file, comm, mesh_group)
    if cell_type is None:
        raise ValueError(f"The mesh {mesh_group} in {input_file} has no cell type.")

    def global_range(local_size: int) -> tuple[tuple[int, int], int]:
        # exscan is collective, and gives None on rank 0
        start = comm.exscan(local_size, op=MPI.SUM)
        start = start if comm.rank > 0 else 0
        return (start, start + local_size), comm.allreduce(local_size, op=MPI.SUM)

    node_range, num_nodes = global_range(len(x))
    cell_range, num_cells = global_range(len(topology))
    mesh = MeshData(
        metadata=stamp(),
        local_geometry=x,
        local_geometry_pos=node_range,
        num_nodes_global=num_nodes,
        local_topology=topology,
        local_topology_pos=cell_range,
        num_cells_global=num_cells,
        cell_type=cell_type,
        degree=1,
        lagrange_variant=int(basix.LagrangeVariant.equispaced),
        store_partition=False,
    )
    write_mesh(output_file, comm, mesh, 0.0, FileMode.write, backend_args, backend)

    writer = get_backend(backend)
    writer_args = writer.get_default_backend_args(backend_args)
    for name in meshtags:
        tags = h5py_backend.read_meshtags_data(input_file, comm, name, {"legacy": True})
        (tags.local_start, _), tags.num_entities_global = global_range(len(tags.values))
        tags.num_dofs_per_entity = tags.indices.shape[1]
        tags.cell_type = dolfinx.mesh.to_string(
            dolfinx.cpp.mesh.cell_entity_type(dolfinx.mesh.to_type(cell_type), tags.dim, 0)
        )
        tags.metadata = stamp()
        writer.write_meshtags(output_file, comm, tags, writer_args)


def convert(
    input_file: Path | str,
    output_file: Path | str,
    comm: MPI.Comm = MPI.COMM_WORLD,
    names: Sequence[str] | None = None,
    meshtags: Sequence[str] = (),
    mesh_group: str = "/mesh",
    backend: str | None = None,
    backend_args: dict[str, Any] | None = None,
    engine: str = "BP4",
    overwrite: bool = False,
):
    """Convert a checkpoint of adios4dolfinx 0.7 or older, or the mesh and mesh functions of
    a legacy DOLFIN HDF5 file, to the current format.

    The data is copied as stored, so the result is read as any checkpoint, with
    :func:`io4dolfinx.read_mesh`, :func:`io4dolfinx.read_meshtags` and
    :func:`io4dolfinx.read_function`. A legacy DOLFIN file is recognised by the topology of
    its mesh, at `mesh_group`; any other file is read as an adios4dolfinx checkpoint.

    Args:
        input_file: The file to convert.
        output_file: The checkpoint to write.
        comm: The communicator to read and write with.
        names: The name to give each function of an adios4dolfinx checkpoint, in the order
            they were written, as the file does not store names. Defaults to ``["u"]``.
        meshtags: The mesh functions of a legacy DOLFIN file to convert to mesh tags of the
            same names.
        mesh_group: The group of the mesh in a legacy DOLFIN file.
        backend: The backend to write with, ``"adios2"`` or ``"h5py"``. Chosen from the
            suffix of `output_file` if not given: ``.bp`` for ADIOS2, ``.h5`` for h5py.
        backend_args: Arguments of the backend that writes.
        engine: The ADIOS2 engine an adios4dolfinx checkpoint was written with.
        overwrite: Replace `output_file` if it exists.

    Raises:
        FileExistsError: If `output_file` exists and `overwrite` is not set.
        ValueError: If `input_file` is neither kind of file, the number of names is not
            the number of functions in it, names are given for a legacy DOLFIN file or mesh
            tags for an adios4dolfinx checkpoint, or the backend is not given and cannot be
            chosen from the suffix of `output_file`.
    """
    input_file = Path(input_file)
    output_file = Path(output_file)
    check_file_exists(input_file)
    if backend is None:
        if output_file.suffix not in _backends_by_suffix:
            raise ValueError(
                f"Cannot tell the backend from the suffix of {output_file}: give the backend, "
                f"one of {sorted(set(_backends_by_suffix.values()))}, or use one of the "
                f"suffixes {sorted(_backends_by_suffix)}."
            )
        backend = _backends_by_suffix[output_file.suffix]
    if backend not in _checkpoint_backends:
        raise ValueError(
            f"Cannot write checkpoints with the {backend!r} backend, use one of "
            f"{list(_checkpoint_backends)}."
        )
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"{output_file} exists. Set overwrite to replace it.")
    if output_file.resolve() == input_file.resolve():
        raise ValueError("The converted checkpoint cannot replace the one it is read from.")

    dolfin_file = False
    if h5py.is_hdf5(input_file):
        with h5py.File(input_file, "r") as h5file:
            dolfin_file = f"{mesh_group}/topology" in h5file
    if dolfin_file:
        if names is not None:
            raise ValueError(
                "Functions of legacy DOLFIN files are numbered by DOLFIN and need a function "
                "space: read them with io4dolfinx.read_function_from_legacy_h5 onto the mesh "
                "of the converted file."
            )
        _convert_dolfin(input_file, output_file, comm, meshtags, mesh_group, backend, backend_args)
        return
    if len(meshtags) > 0:
        raise ValueError(f"{input_file} is an adios4dolfinx checkpoint, which has no mesh tags.")

    from .backends.adios2.backend import read_format0_checkpoint

    names = ["u"] if names is None else names
    mesh, functions = read_format0_checkpoint(input_file, comm, engine)
    if len(names) != len(functions):
        raise ValueError(
            f"{input_file} holds {len(functions)} functions, so it needs {len(functions)} "
            f"names, got {list(names)}."
        )
    if len(set(names)) != len(names):
        raise ValueError(f"The names of the functions must differ, got {list(names)}.")

    write_mesh(output_file, comm, mesh, 0.0, FileMode.write, backend_args, backend)
    for name, function in zip(names, functions):
        data = FunctionData(metadata=stamp(), name=name, **function)
        write_function(output_file, comm, data, 0.0, FileMode.append, backend_args, backend)


def main(argv: Sequence[str] | None = None):
    """Run :func:`convert` from the command line."""
    parser = argparse.ArgumentParser(
        description="Convert a checkpoint of adios4dolfinx 0.7 or older, or the mesh and mesh "
        "functions of a legacy DOLFIN HDF5 file, to the current io4dolfinx format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="The checkpoint to convert")
    parser.add_argument("output", type=Path, help="The checkpoint to write")
    parser.add_argument(
        "--names",
        nargs="+",
        default=None,
        help="The name of each function of an adios4dolfinx checkpoint, in the order they "
        "were written (default: u)",
    )
    parser.add_argument(
        "--meshtags",
        nargs="+",
        default=(),
        help="The mesh functions of a legacy DOLFIN file to convert to mesh tags",
    )
    parser.add_argument(
        "--mesh-group", default="/mesh", help="The group of the mesh in a legacy DOLFIN file"
    )
    parser.add_argument(
        "--backend",
        choices=_checkpoint_backends,
        default=None,
        help="The backend to write with. Chosen from the suffix of the output if not given: "
        ".bp for adios2, .h5 for h5py",
    )
    parser.add_argument(
        "--engine", default="BP4", help="The ADIOS2 engine the input was written with"
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace the output if it exists")
    args = parser.parse_args(argv)
    convert(
        args.input,
        args.output,
        names=args.names,
        meshtags=args.meshtags,
        mesh_group=args.mesh_group,
        backend=args.backend,
        engine=args.engine,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
