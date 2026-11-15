# Copyright (C) 2026 Jørgen Schartum Dokken
#
# This file is part of io4dolfinx
#
# SPDX-License-Identifier:    MIT

"""Convert checkpoints written by adios4dolfinx 0.7 and older to the current format.

These checkpoints (format 0, see :mod:`io4dolfinx.format`) hold a mesh and functions
written with ADIOS2, with no function names or time stamps: each function is stored in a
step of its own under the same variable names. :func:`convert` copies the stored data to a
checkpoint of the current format, so no mesh or function space is created. Each function
is given a name and written at time 0.

The conversion is also available from the command line, under MPI if wanted::

    io4dolfinx-convert old.bp new.h5 --names u

or, equivalently, ``python3 -m io4dolfinx.converter``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mpi4py import MPI

from .backends import FileMode
from .format import stamp
from .structures import FunctionData
from .utils import check_file_exists
from .writers import write_function, write_mesh

__all__ = ["convert", "main"]

# Backends that write checkpoints, by the suffix of their files
_backends_by_suffix = {".bp": "adios2", ".h5": "h5py"}


def convert(
    input_file: Path | str,
    output_file: Path | str,
    comm: MPI.Comm = MPI.COMM_WORLD,
    names: Sequence[str] = ("u",),
    backend: str | None = None,
    backend_args: dict[str, Any] | None = None,
    engine: str = "BP4",
    overwrite: bool = False,
):
    """Convert a checkpoint written by adios4dolfinx 0.7 or older to the current format.

    The mesh and functions are copied as stored, so the result is read as any checkpoint,
    with :func:`io4dolfinx.read_mesh` and :func:`io4dolfinx.read_function`.

    Args:
        input_file: The checkpoint to convert.
        output_file: The checkpoint to write.
        comm: The communicator to read and write with.
        names: The name to give each function in the file, in the order they were
            written. The file does not store names.
        backend: The backend to write with, ``"adios2"`` or ``"h5py"``. Chosen from the
            suffix of `output_file` if not given: ``.bp`` for ADIOS2, ``.h5`` for h5py.
        backend_args: Arguments of the backend that writes.
        engine: The ADIOS2 engine `input_file` was written with.
        overwrite: Replace `output_file` if it exists.

    Raises:
        FileExistsError: If `output_file` exists and `overwrite` is not set.
        ValueError: If `input_file` is not a format 0 checkpoint, the number of names is not
            the number of functions in it, or the backend is not given and cannot be
            chosen from the suffix of `output_file`.
    """
    from .backends.adios2.backend import read_format0_checkpoint

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
    if backend not in _backends_by_suffix.values():
        raise ValueError(
            f"Cannot write checkpoints with the {backend!r} backend, use one of "
            f"{sorted(set(_backends_by_suffix.values()))}."
        )
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"{output_file} exists. Set overwrite to replace it.")
    if output_file.resolve() == input_file.resolve():
        raise ValueError("The converted checkpoint cannot replace the one it is read from.")

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
        description="Convert a checkpoint written by adios4dolfinx 0.7 or older to the "
        "current io4dolfinx format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="The checkpoint to convert")
    parser.add_argument("output", type=Path, help="The checkpoint to write")
    parser.add_argument(
        "--names",
        nargs="+",
        default=["u"],
        help="The name of each function in the file, in the order they were written",
    )
    parser.add_argument(
        "--backend",
        choices=sorted(set(_backends_by_suffix.values())),
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
        backend=args.backend,
        engine=args.engine,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
