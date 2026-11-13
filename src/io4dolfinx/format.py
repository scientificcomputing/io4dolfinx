# Copyright (C) 2026 Jørgen Schartum Dokken
#
# This file is part of io4dolfinx
#
# SPDX-License-Identifier:    MIT

"""Version of the layout of io4dolfinx checkpoints.

Every mesh, function and set of mesh tags in a checkpoint written by the ``h5py`` or
``adios2`` backend is stamped with the version of the checkpoint format it was written in,
when it is first written. The stamp is never changed, so a file appended to by several
versions of io4dolfinx records the format of each object.

- The major version changes when the layout changes so that older versions of io4dolfinx can
  no longer read it. A checkpoint with a newer major version than :data:`FORMAT_VERSION`
  is not read.
- The minor version changes when something optional is added. Readers ignore what they do
  not know, and treat what is missing as its default, so a checkpoint with a newer minor
  version is read, and an object with an older minor version can be appended to.

Objects without a stamp were written before checkpoints were stamped, and are read as
format 1.0. The layout of each format is described in :doc:`checkpoint_format`.
"""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from typing import Any

from mpi4py import MPI

import dolfinx
import numpy as np

__all__ = [
    "FORMAT_VERSION",
    "stamp",
    "format_version",
    "object_metadata",
    "check_readable",
    "check_appendable",
]

#: ``(major, minor)`` version of the checkpoint format written by this version of io4dolfinx
FORMAT_VERSION: tuple[int, int] = (1, 0)

#: Names of the metadata stored with each object
FORMAT_KEY = "io4dolfinx_format"
VERSION_KEY = "io4dolfinx_version"
DOLFINX_VERSION_KEY = "dolfinx_version"
METADATA_KEYS = (FORMAT_KEY, VERSION_KEY, DOLFINX_VERSION_KEY)


def stamp() -> dict[str, Any]:
    """The metadata stored with an object when it is first written.

    Returns:
        The checkpoint format version, as an ``int32`` array ``[major, minor]``, and the
        versions of io4dolfinx and DOLFINx that wrote the object, as strings
    """
    return {
        FORMAT_KEY: np.array(FORMAT_VERSION, dtype=np.int32),
        VERSION_KEY: version("io4dolfinx"),
        DOLFINX_VERSION_KEY: dolfinx.__version__,
    }


def format_version(metadata: dict[str, Any]) -> tuple[int, int] | None:
    """The checkpoint format version in the metadata of an object.

    Returns:
        ``(major, minor)``, or ``None`` if the object has no stamp
    """
    if FORMAT_KEY not in metadata:
        return None
    major, minor = (int(v) for v in np.asarray(metadata[FORMAT_KEY]).reshape(-1)[:2])
    return major, minor


def object_metadata(
    backend: Any,
    filename: Path | str,
    comm: MPI.Comm,
    kind: str,
    name: str | None = None,
    backend_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The metadata of an object in a file, if the backend stores metadata.

    Only the checkpoint backends (``h5py`` and ``adios2``) do; for any other backend, and
    for an object that does not exist, the metadata is empty.

    Args:
        backend: The backend module
        filename: Path to file
        comm: MPI communicator
        kind: ``"mesh"``, ``"function"`` or ``"meshtags"``
        name: Name of the function or mesh tags, unused for a mesh
        backend_args: Arguments to backend
    """
    reader = getattr(backend, "read_metadata", None)
    return {} if reader is None else reader(filename, comm, kind, name, backend_args)


def _describe(metadata: dict[str, Any], what: str) -> str:
    major, minor = format_version(metadata)  # type: ignore[misc]
    written_by = metadata.get(VERSION_KEY)
    by = f" by io4dolfinx {written_by}" if written_by else ""
    return f"{what} was written{by} in checkpoint format {major}.{minor}"


def check_readable(metadata: dict[str, Any], what: str) -> tuple[int, int]:
    """Check that an object can be read by this version of io4dolfinx.

    Args:
        metadata: The metadata of the object, as read by the backend
        what: Description of the object for the error message, e.g. ``"Function 'u' in
            file.bp"``

    Returns:
        The checkpoint format version of the object, ``(1, 0)`` if it has no stamp

    Raises:
        ValueError: If the object has a newer major version than this io4dolfinx reads.
    """
    found = format_version(metadata)
    if found is None:
        return (1, 0)
    if found[0] > FORMAT_VERSION[0]:
        raise ValueError(
            f"{_describe(metadata, what)}, but this version of io4dolfinx "
            f"({version('io4dolfinx')}) reads checkpoint format {FORMAT_VERSION[0]}.x. "
            "Upgrade io4dolfinx to read it."
        )
    return found


def check_appendable(metadata: dict[str, Any], what: str):
    """Check that this version of io4dolfinx can append to an existing object.

    Appending keeps the stamp of the object, so it is allowed if the object has the same
    major version, whatever its minor version.

    Args:
        metadata: The metadata of the object, empty if it does not exist yet
        what: Description of the object for the error message

    Raises:
        ValueError: If the object has another major version.
    """
    found = format_version(metadata)
    if found is not None and found[0] != FORMAT_VERSION[0]:
        raise ValueError(
            f"{_describe(metadata, what)}, and this version of io4dolfinx "
            f"({version('io4dolfinx')}) writes checkpoint format "
            f"{FORMAT_VERSION[0]}.{FORMAT_VERSION[1]}, so it cannot append to it. "
            "Write to a new file."
        )
